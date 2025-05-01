import re
import torch
import chromadb
from rapidfuzz import fuzz
from urllib3.poolmanager import PoolManager
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import requests
import ssl
import streamlit as st
from transformers import TextStreamer, AutoModelForCausalLM, AutoTokenizer
from embedding_utils import load_embedding_model, CustomEmbeddingFunction

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------
# Load Decomposition Model
# ---------------------------------------------
DECOMPOSITION_MODEL_PATH = "/home/tttung/NguyenTrinhTest/model_stage3"
model = AutoModelForCausalLM.from_pretrained(DECOMPOSITION_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(DECOMPOSITION_MODEL_PATH)
# ---------------------------------------------
# Load Classification Model
# ---------------------------------------------
CLASSIFICATION_MODEL_PATH = "/home/tttung/NguyenTrinhTest/model_stage3"
classification_model = AutoModelForCausalLM.from_pretrained(CLASSIFICATION_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
classification_tokenizer = AutoTokenizer.from_pretrained(CLASSIFICATION_MODEL_PATH)
classification_model = classification_model.to(device)

# ---------------------------------------------
# Setup Chroma
# ---------------------------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

EMBED_MODEL_NAME = "dangvantuan/vietnamese-embedding"
embed_model, embed_tokenizer = load_embedding_model(model_name=EMBED_MODEL_NAME)
embedding_function = CustomEmbeddingFunction(embed_model, embed_tokenizer)

collection = chroma_client.get_or_create_collection(
    name="vietnamese_pregancy_data",
    embedding_function=embedding_function
)
# ---------------------------------------------
# Helper to remove near-duplicate sub-questions
# ---------------------------------------------
def clean_subquestions(questions):
    cleaned = []
    for q in questions:
        q = q.replace("<|end_of_text|>", "").strip()
        if q:
            cleaned.append(q)

    unique_questions = []
    similarity_threshold = 90
    for q in cleaned:
        if not any(fuzz.ratio(q, uq) >= similarity_threshold for uq in unique_questions):
            unique_questions.append(q)
    return unique_questions

# ---------------------------------------------
# Revised Decomposition
# ---------------------------------------------
def custom_decompose_query(query):
    """
    Use the fine-tuned model to decompose a query into smaller sub-questions.

    Args:
        query (str): The original user query.

    Returns:
        list[str]: A list of valid, cleaned, and unique sub-questions.
    """
    # Define the new prompt format
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    Decompose the following query into small, clear, independent sub-questions (maximum 3 questions). Use an internal chain-of-thought process to think through the decomposition, but do not output any of your internal reasoning. As part of your reasoning, if some sub-questions are very similar in meaning, only return one of them. Return only the final list of distinct questions without adding any extraneous text.

    ### Input:
    {}

    ### Response:
    """
    
    # Tokenize inputs for inference
    inputs = tokenizer(
        [ prompt.format(query) ],  # Insert the query into the input part of the prompt
        return_tensors="pt"
    ).to(device)

    # Generate text with the model
    outputs = model.generate(**inputs, max_new_tokens=256)
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract and filter sub-questions
    valid_questions = []
    for line in raw_output.split("\n"):
        line = line.strip()
        # Ignore lines that are part of the prompt or irrelevant
        if not line or line.startswith("###") or line.startswith("Decompose the following query") or "Below is an instruction" in line:
            continue
        # Clean up formatting (remove leading symbols like "-")
        if line.startswith("-"):
            line = line[1:].strip()
        valid_questions.append(line)

    # Clean and deduplicate questions
    return clean_subquestions(valid_questions)

# ---------------------------------------------
# Classification function (returns 'yes'/'no'/'error')
# ---------------------------------------------
def classify_query(query):
    """
    Classify whether a query is related to healthcare.

    Args:
        query (str): The input query.

    Returns:
        str: 'yes' if healthcare-related, 'no' otherwise; 'error' if not parsable.
    """
    # More detailed prompt
    prompt = """Below is an instruction for a classification task, paired with an input question. The task is to determine if the input question is related to healthcare or not. A healthcare-related question involves topics such as:
    - Medical treatments or procedures
    - Medications or pharmaceutical products (including their storage conditions, usage, or effects)
    - Diseases, symptoms, or health conditions
    - Medical diagnostics or healthcare practices

    If the question references any of these topics (even indirectly), respond with 'yes'. Otherwise, respond with 'no'.

    ### Instruction:
    Classify the following question: Is it a healthcare-related question? Respond with 'yes' or 'no' only.

    ### Input:
    {}

    ### Response:"""

    # Format the prompt with the user query
    formatted_prompt = prompt.format(query)

    # Tokenize and generate
    inputs = classification_tokenizer([formatted_prompt], return_tensors="pt").to(device)
    outputs = classification_model.generate(
        **inputs,
        max_new_tokens=2,       # Short generation to restrict output
        length_penalty=0.1,     # Slight penalty to discourage long text
        do_sample=False         # Deterministic output
    )

    # Decode raw model output
    raw_response = classification_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Locate "### Response:" (if present) and extract text after that
    response_start = raw_response.lower().find("### response:")
    if response_start != -1:
        response_part = raw_response[response_start + len("### response:"):].strip()
    else:
        response_part = raw_response.strip()

    # Remove non-letter characters (punctuation, spaces, etc.) and lowercase
    cleaned = re.sub(r"[^a-zA-Z]+", "", response_part).lower()

    if cleaned in {"yes", "no"}:
        return cleaned
    else:
        print(f"Unexpected classification result: '{raw_response}'")
        return "error"

# ---------------------------------------------
# Final process flow: decompose -> classify -> retrieve
# ---------------------------------------------
def process_query(query: str):
    sub_questions = custom_decompose_query(query)
    results = {}

    for sq in sub_questions:
        label = classify_query(sq)  
        
        if label == "yes":
            # Retrieve candidate using cosine similarity.
            q_emb = embedding_function.embed([sq])
            query_results = collection.query(query_embeddings=q_emb, n_results=1)
            metadatas = query_results.get('metadatas', [])
            distances = query_results.get('distances', [])
            
            candidate_meta = None
            if metadatas and distances:
                # Handle cases where results may be nested lists.
                candidate_meta = metadatas[0]
                if isinstance(candidate_meta, list):
                    candidate_meta = candidate_meta[0]
                candidate_distance = distances[0]
                
                # Check if the similarity (or distance) meets your threshold.
                if isinstance(candidate_distance, (float, int)) and candidate_distance > 25:
                    results[sq] = {
                        "label": "yes",
                        "retrieval": {
                            "question": candidate_meta.get("question", ""),
                            "answer": "Not founded in database",
                            "article_url": "",
                            "author": ""
                        }
                    }
                    continue
            
            if candidate_meta:
                # Validate the candidate using the Qwen 7B model.
                validated_candidate = validate_candidate(query, sq, candidate_meta)
                results[sq] = {
                    "label": "yes",
                    "retrieval": validated_candidate
                }
            else:
                results[sq] = {
                    "label": "yes",
                    "retrieval": {
                        "question": "",
                        "answer": "Not founded in database",
                        "article_url": "",
                        "author": ""
                    }
                }
        else:
            # For sub-questions classified as 'no' or 'error'
            results[sq] = {
                "label": label,
                "retrieval": {
                    "question": "",
                    "answer": "Not related with medical issue",
                    "article_url": "",
                    "author": ""
                }
            }

    return sub_questions, results

def validate_candidate(user_query: str, sub_question: str, candidate_meta: dict) -> dict:
    """
    Uses the Qwen 7B fine-tuned model to validate the candidate Q&A pair.
    Returns a dictionary containing the candidate question, answer, article_url, and author.
    If the candidate is not valid, returns 'Not founded in database' as the answer.
    """
    retrieved_question = candidate_meta.get('question', '')
    retrieved_answer = candidate_meta.get('answer', '')
    
    prompt = f"""You are a knowledgeable and helpful Vietnamese medical assistant.
Your task is to evaluate whether a candidate Q&A pair is relevant to a user's query across any medical topic.
Please follow these steps in your internal chain-of-thought process (do not output any internal reasoning):
1. Carefully compare the user's query and the sub-question with the retrieved question and answer.
2. Assess whether the meanings are semantically equivalent, keeping in mind that different phrasing or synonyms may express the same idea.
3. In Vietnamese medical contexts, be aware that certain terms (for example, those related to treatment, diagnosis, or prevention) may be used interchangeably.
4. Recognize that if the retrieved question or answer is more detailed or specific while still addressing the general topic, they can still be considered equivalent if the medical content is relevant.
5. Evaluate whether the retrieved Q&A pair is medically appropriate and truly addresses the user's intent.
After you have completed your internal reasoning, if the candidate Q&A pair meets the criteria for relevance and appropriateness, respond with a single word "Yes". Otherwise, respond with a single word "No".

User Query: {user_query}

Sub-question: {sub_question}

Retrieved Question: {retrieved_question}

Retrieved Answer: {retrieved_answer}

Now, after careful internal reasoning, provide only your final decision.
"""
    # Debug: Print the prompt
    #print("DEBUG - Candidate Validation Prompt:\n", prompt)
    
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Debug: Print the raw model output.
    #print("DEBUG - Raw candidate validation output:", result)
    
    # Use regex to extract the final decision (looking for 'yes' or 'no')
    words = re.findall(r'\b(yes|no)\b', result, re.IGNORECASE)
    final_decision = words[-1].lower() if words else ""
    
    # Debug: Print the extracted decision.
    #print("DEBUG - Extracted final decision:", final_decision)
    
    if final_decision == "yes" and retrieved_answer.strip():
        return {
            "question": candidate_meta.get("question", ""),
            "answer": candidate_meta.get("answer", "").strip(),
            "article_url": candidate_meta.get("article_url", ""),
            "author": candidate_meta.get("author", "")
        }
    else:
        return {
            "question": candidate_meta.get("question", ""),
            "answer": "Not founded in database",
            "article_url": "",
            "author": ""
        }


def summarize_content(content: str) -> str:
    """
    Generate a detailed and structured summary of the provided Vietnamese medical article content 
    retrieved from the internet using the Qwen 7B fine-tuned model.

    The summary should:
    - Be organized in a numbered (or bullet-point) format.
    - Provide detailed explanations for each key medical point or treatment method mentioned.
    - Exclude any non-medical or irrelevant content (such as website navigation or promotional text).

    Args:
        content (str): The full text content to be summarized.

    Returns:
        str: The detailed, structured summary.
    """
    # Clear GPU cache to prevent state accumulation issues
    torch.cuda.empty_cache()

    # Optionally truncate content to avoid exceeding model limits (e.g., 3000 characters)
    max_content_length = 3000
    if len(content) > max_content_length:
        content = content[:max_content_length]

    prompt = f"""You are a knowledgeable and helpful Vietnamese medical assistant.
Your task is to generate a highly detailed and structured summary of the following Vietnamese medical article content retrieved from the internet.
IMPORTANT:
- Do not provide a generic or vague overview. Instead, break down the content in a clear, numbered list (or bullet-point format).
- For every key medical point (for example, treatment methods, symptoms, diagnoses, recommendations), provide a detailed explanation that includes specifics such as how the method works, its advantages, limitations, and any critical contextual details.
- Completely ignore any non-medical or irrelevant text (such as navigation menus, promotional content, or website footer information).
- Ensure that each point is unique and appears only once.
- Additionally, if the content includes the name of a doctor or hospital that authored or published the article, include that information as part of the summary.
Please follow these internal steps (do not output any internal reasoning):
1. Thoroughly analyze the content and remove all irrelevant sections.
2. Extract every essential medical detail.
3. Identify lists or enumerated information and prepare to expand each item with detailed descriptions.
4. Organize the extracted information into a clear, coherent, and logically ordered numbered list with detailed explanations.
5. Return your response in bullet points which cover the medical key points of the text.
After completing your internal reasoning, output only the final, detailed summary in the specified format.

Content:
{content}

Summary:"""

    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            early_stopping=True
        )
    except Exception as e:
        return f"Error generating summary: {e}"
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Summary:" in summary:
        final_summary = summary.split("Summary:", 1)[1].strip()
    else:
        final_summary = summary.strip()
    return final_summary

# ---------------------------------------------
# Google Search Function using Custom Search API
# ---------------------------------------------
def google_search(query):
    # Replace these with your actual Google API key and Programmable Search Engine ID.
    GOOGLE_API_KEY = "AIzaSyCgU9DninRJYG2B-Y7b6S4pUqOIjYl5Gw8"
    GOOGLE_CSE_ID = "c009c2a8789d24821"
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "gl": "vn",       # Geographic location: Vietnam.
        "hl": "vi",       # Interface language: Vietnamese.
        "lr": "lang_vi"   # Search results language: Vietnamese.
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError if the response code was unsuccessful.
        return response.json()
    except requests.RequestException as e:
        st.error(f"An error occurred during the search: {e}")
        return None

# ---------------------------------------------
# Helper function to extract full page content (text only)
# ---------------------------------------------
# Custom SSL adapter to allow smaller DH keys and disable hostname verification.
class SSLAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        # Create an SSL context that allows smaller DH keys.
        context = ssl.create_default_context()
        context.check_hostname = False           # Disable hostname checking.
        context.verify_mode = ssl.CERT_NONE        # Disable certificate verification.
        context.set_ciphers('DEFAULT:@SECLEVEL=1')  # Allow lower security (smaller DH keys).
        self.poolmanager = PoolManager(num_pools=connections,
                                       maxsize=maxsize,
                                       block=block,
                                       ssl_context=context)

def extract_page_content(url):
    try:
        session = requests.Session()
        # Define a Retry strategy: retry up to 3 times for common status codes.
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD", "OPTIONS"]
        )
        # Mount the custom SSL adapter with retry settings.
        adapter = SSLAdapter(max_retries=retries)
        session.mount("https://", adapter)
        # Use a realistic User-Agent header to mimic a browser.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0 Safari/537.36"
            )
        }
        # Increase the timeout to 30 seconds.
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Remove script and style elements.
        for tag in soup(["script", "style"]):
            tag.decompose()
        # Extract and clean the text.
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = "\n".join(lines)
        return content
    except Exception as e:
        return f"Error retrieving content: {e}"
    
def detect_ambiguity(query: str) -> bool:
    prompt_template = (
    "Below is an instruction that describes a task, paired with an input that provides further context.\n"
    "Use an internal chain-of-thought process to analyze the query, but do not output any internal reasoning. Only provide the final answer.\n\n"
    "### Instruction:\n"
    "Determine if the following query is ambiguous. A query should be classified as not ambiguous if it is clearly phrased as a question—for example, if it ends with 'là gì?', contains a question mark, or uses explicit interrogative phrases that request a definition or explanation—or if it includes multiple specific details and aspects (such as causes, symptoms, and treatment options) that provide sufficient context. Conversely, a query is considered ambiguous if it only mentions a single topic or condition without additional details, or if it is short and lacks context.\n"
    "Respond with 'yes' if the query is ambiguous, or 'no' if it is specific enough. Do not include any additional text.\n\n"
    "### Input:\n"
    "{}\n\n"
    "### Response:\n"
)
    prompt = prompt_template.format(query)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    print(f"Raw model response:\n'{full_response}'\n")
    lines = [line.strip() for line in full_response.split("\n") if line.strip()]
    if not lines:
        return True
    final_line = re.sub(r"[^\w\s]", "", lines[-1]).strip()
    if final_line == "yes":
        return True   # Ambiguous
    elif final_line == "no":
        return False  # Not ambiguous
    else:
        print(f"Could not parse a clear answer. Final line: '{final_line}'")
        return True   # Default to ambiguous

def generate_clarifying_question(query: str) -> str:
    prompt_template = (
    "Bạn là một chuyên gia y tế chuyên về điều trị bệnh.\n"
    "Nhiệm vụ của bạn là phân tích truy vấn của người dùng và đưa ra 5 câu hỏi làm rõ độc đáo nhằm thu thập các thông tin cần thiết để trả lời chính xác.\n"
    "Hãy tự động xác định những khía cạnh quan trọng cần được làm rõ dựa trên nội dung của truy vấn. Các khía cạnh có thể bao gồm (nhưng không giới hạn):\n"
    "- Loại bệnh hoặc tình trạng,\n"
    "- Các phương pháp điều trị cụ thể,\n"
    "- Triệu chứng chủ chốt,\n"
    "- Chi tiết về chẩn đoán và nơi điều trị,\n"
    "- Chi phí điều trị hoặc các yếu tố bối cảnh khác.\n"
    "Hãy đảm bảo rằng mỗi câu hỏi là độc đáo và không lặp lại, phản ánh các khía cạnh khác nhau của truy vấn.\n\n"
    "Truy vấn của người dùng:\n"
    "{query}\n\n"
    "Các câu hỏi làm rõ:"
)
    prompt = prompt_template.format(query=query)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    text_streamer = TextStreamer(tokenizer)
    output = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1024)
    raw_output = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    filtered_lines = []
    found_section = False
    for line in raw_output.split("\n"):
        if "Các câu hỏi làm rõ:" in line:
            found_section = True
            continue
        if found_section:
            filtered_lines.append(line.strip())
    final_output = "\n".join(line for line in filtered_lines if line)
    return final_output.strip()


def combine_query(original_query: str, clarification: str, model, tokenizer, device="cuda") -> str:
    """
    Sử dụng mô hình ngôn ngữ để kết hợp truy vấn thai sản ban đầu với thông tin làm rõ,
    tạo ra một câu hỏi cụ thể và rõ ràng.
    """
    prompt_template = f"""
You are a helpful assistant. Your task is to merge two user inputs into one concise, specific question that preserves the full meaning of both inputs.
Do not output any internal reasoning; only output the final merged question.

### Instructions:
1. Read the original question and the user’s clarification carefully.
2. Combine them into a single, clear, specific question that incorporates both the topic from the original question and the aspect mentioned in the clarification.
3. Do not omit any important information from either part.
4. Output only the final merged question without extra text.

### Original Question:
{original_query}

### Clarification:
{clarification}

### Final Merged Question:
"""
    inputs = tokenizer([prompt_template], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    combined_query = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    lines = [line.strip() for line in combined_query.split("\n") if line.strip()]
    final_query = lines[-1]
    return final_query

def assess_query_complexity(query: str) -> str:
    prompt = f"""You are a knowledgeable and helpful Vietnamese medical assistant specialized in pregnancy care.
Your task is to determine whether the following pregnancy-related query is complex or simple.
A query is considered "complex" if it addresses multiple aspects (for example, symptoms, prenatal care, potential complications, or stages of pregnancy).
A query is "simple" if it focuses on one specific aspect only.
For example, if a query asks "Các triệu chứng và cách chăm sóc thai sản ở phụ nữ lần đầu làm mẹ", it should be classified as "complex".
Do not output any internal reasoning; just respond with a single word: "complex" or "simple".

Query: {query}

Final Answer:"""
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    words = re.findall(r'\b(complex|simple)\b', result, re.IGNORECASE)
    final_decision = words[-1].lower() if words else "simple"
    return final_decision
