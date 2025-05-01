import os
import sys
import sqlite3
from datetime import datetime
import streamlit as st
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# =========================
# CONFIGURATION
# =========================


# Các cấu hình lưu trữ và model
CHROMA_STORAGE_PATH = "./chroma_storage"
FAISS_INDEX_PATH = "faiss_index"
EMBED_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"

# =========================
# DATABASE (DB Manager)
# =========================

# Kết nối đến SQLite database (check_same_thread=False cho Streamlit)
conn = sqlite3.connect('conversation_history.db', check_same_thread=False)
cursor = conn.cursor()

# Tạo bảng users nếu chưa tồn tại
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
)
''')

# Tạo bảng conversation_history nếu chưa tồn tại
cursor.execute('''
CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    timestamp TEXT,
    role TEXT,
    message TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')
conn.commit()

def add_user(name):
    """Thêm người dùng mới và trả về user_id."""
    cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
    conn.commit()
    return cursor.lastrowid

def get_user_by_name(name):
    cursor.execute("SELECT id FROM users WHERE name=?", (name,))
    return cursor.fetchone()

def save_message_db(user_id, role, message):
    """Lưu thông điệp vào bảng conversation_history."""
    timestamp = datetime.now().isoformat()
    cursor.execute(
        'INSERT INTO conversation_history (user_id, timestamp, role, message) VALUES (?, ?, ?, ?)',
        (user_id, timestamp, role, message)
    )
    conn.commit()

def get_conversation_history_db(user_id):
    """Truy xuất lịch sử hội thoại của user theo thứ tự thời gian."""
    cursor.execute(
        'SELECT timestamp, role, message FROM conversation_history WHERE user_id = ? ORDER BY timestamp',
        (user_id,)
    )
    return cursor.fetchall()

# =========================
# RETRIEVAL & RE-RANKING
# =========================

import chromadb
from embedding_utils import load_embedding_model, CustomEmbeddingFunction, CustomSentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from openai import OpenAI

# Cấu hình ChromaDB và embedding
chroma_client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
embed_model, embed_tokenizer = load_embedding_model(model_name=EMBED_MODEL_NAME)
embedding_function = CustomEmbeddingFunction(embed_model, embed_tokenizer)
collection = chroma_client.get_or_create_collection(
    name="vietnamese_pregancy_data",
    embedding_function=embedding_function
)

# Cấu hình FAISS retriever với lớp embedding tùy chỉnh
embedding_chain = CustomSentenceTransformerEmbeddings(embed_model, embed_tokenizer)
faiss_vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_chain, allow_dangerous_deserialization=True)
retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Khởi tạo OpenAI client
client = OpenAI(api_key="OpenAIKey")

def rerank_results(question, retrieved_chunks):
    query_emb = embedding_function.embed([question])[0]
    import numpy as np
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
    chunk_scores = []
    for chunk in retrieved_chunks:
        chunk_emb = embedding_function.embed([chunk])[0]
        score = cosine_similarity(query_emb, chunk_emb)
        chunk_scores.append((chunk, score))
    # Sắp xếp theo cosine similarity giảm dần và lấy top 5
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    top_chunks = chunk_scores[:4]
    for i, (chunk, score) in enumerate(top_chunks):
        print(f"Document {i+1} selected with similarity score: {score:.4f}")
    selected_chunks = [chunk for chunk, score in top_chunks]
    return "\n\n".join(selected_chunks)

def answer_question_faiss(query: str) -> str:
    retrieved_docs = retriever.get_relevant_documents(query)
    print("Số lượng document được tìm thấy:", len(retrieved_docs))
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    filtered_context = rerank_results(query, retrieved_texts)
    
    custom_prompt = PromptTemplate(
        input_variables=["context_str", "question"],
        template=(
            "Bạn chỉ được sử dụng các thông tin dưới đây để trả lời câu hỏi. "
            "Không sử dụng kiến thức ngoài các thông tin được cung cấp.\n\n"
            "Thông tin:\n"
            "{context_str}\n\n"
            "Câu hỏi: {question}\n\n"
            "Hãy trả lời câu hỏi một cách ngắn gọn, đầy đủ và súc tích. "
            "Đồng thầm, trong câu trả lời, vui lòng liệt kê rõ ràng và trích dẫn các đoạn văn từ thông tin trên mà bạn đã sử dụng.\n"
            "Trả lời:"
        )
    )
    qa_input = {"context_str": filtered_context, "question": query}
    prompt_text = custom_prompt.format(**qa_input)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in answer_question_faiss: {e}", file=sys.stderr)
        return "Không tạo được câu trả lời từ FAISS."

# =========================
# AGENT (Orchestrator Agent)
# =========================

# Các hàm custom từ rag_backend (giả sử chúng được định nghĩa trong module rag_backend)
from rag_backend import custom_decompose_query, assess_query_complexity, combine_query, detect_ambiguity, generate_clarifying_question, model, tokenizer

class OrchestratorAgent:
    def __init__(self, device="cuda", user_id=None):
        self.conversation_history = []
        self.current_query = ""
        self.device = device
        self.chain_of_thought = ""
        self.iteration = 0  # Số vòng làm rõ đã thực hiện
        self.user_id = user_id

    def add_to_history(self, role, message):
        entry = {"role": role, "message": message}
        self.conversation_history.append(entry)
        print(f"[{role.capitalize()}]: {message}")
        if self.user_id:
            save_message_db(self.user_id, role, message)

    def run(self, user_query: str, clarification: str = "", force_query: bool = False) -> str:
        max_iterations = 2  # Số vòng làm rõ tối đa

        if self.current_query != user_query.strip():
            self.current_query = user_query.strip()
            self.conversation_history = []
            self.iteration = 0
            self.chain_of_thought = ""
            self.add_to_history("user", self.current_query)

        if clarification.strip():
            new_query = combine_query(
                self.current_query,
                clarification.strip(),
                model=model, tokenizer=tokenizer, device=self.device
            )
            self.current_query = new_query
            self.add_to_history("user", clarification.strip())
            self.add_to_history("system", f"Refined query: {new_query}")
            self.iteration += 1

        while detect_ambiguity(self.current_query) and not force_query:
            if self.iteration < max_iterations:
                clarifying_qs = generate_clarifying_question(self.current_query)
                self.add_to_history("system", f"Clarifying questions: {clarifying_qs}")
                if clarification.strip() == "":
                    return "Clarification needed: " + clarifying_qs
                else:
                    new_query = combine_query(
                        self.current_query, clarification.strip(),
                        model=model, tokenizer=tokenizer, device=self.device
                    )
                    self.current_query = new_query
                    self.add_to_history("user", clarification.strip())
                    self.add_to_history("system", f"Refined query: {new_query}")
                    clarification = ""
                    self.iteration += 1
            else:
                self.add_to_history("system", "Maximum clarification iterations reached. Forcing query.")
                force_query = True
                break

        complexity = assess_query_complexity(self.current_query)
        self.add_to_history("system", f"Query classified as: {complexity}")

        if complexity == "complex":
            try:
                sub_questions = custom_decompose_query(self.current_query)
            except Exception as e:
                print(f"Error during decomposition: {e}", file=sys.stderr)
                sub_questions = [self.current_query]
        else:
            sub_questions = [self.current_query]

        aggregated_response = self.aggregate_results_batch(sub_questions)

        self.chain_of_thought = self.generate_chain_of_thought(
            f"Based on the aggregated response:\n{aggregated_response}\nexplain your reasoning step-by-step in Vietnamese."
        )
        final_answer = self.generate_final_answer(aggregated_response, self.chain_of_thought)
        self.add_to_history("assistant", final_answer)

        self.iteration = 0
        self.current_query = ""
        return final_answer

    def aggregate_results_batch(self, sub_questions, threshold=25) -> str:
        response = ""
        if not sub_questions:
            return "Không tạo được câu hỏi con nào."
        try:
            embeddings = embedding_function.embed(sub_questions)
            query_results = collection.query(query_embeddings=embeddings, n_results=1)
        except Exception as e:
            print(f"DEBUG: Error performing batch vector search: {e}", file=sys.stderr)
            query_results = None

        if query_results:
            metadatas_list = query_results.get("metadatas", [])
            distances_list = query_results.get("distances", [])
            for idx, sq in enumerate(sub_questions, 1):
                candidate_meta = None
                candidate_distance = None
                if idx - 1 < len(metadatas_list):
                    candidate_meta = metadatas_list[idx - 1]
                    if isinstance(candidate_meta, list):
                        candidate_meta = candidate_meta[0]
                if idx - 1 < len(distances_list):
                    candidate_distance = distances_list[idx - 1]
                    if isinstance(candidate_distance, list):
                        candidate_distance = candidate_distance[0]
                if candidate_meta and candidate_distance is not None and candidate_distance <= threshold:
                    answer = candidate_meta.get("answer", "").strip()
                    link = candidate_meta.get("link", "").strip()
                    author = candidate_meta.get("author", "").strip()
                    if answer and answer.lower() != "not founded in database":
                        response += f"Câu trả lời cho câu hỏi con {idx} ('{sq}'): {answer} (distance: {candidate_distance})\n"
                        if link:
                            response += f"Link: {link}\n"
                        if author:
                            response += f"Author: {author}\n"
                    else:
                        response += f"Câu hỏi con {idx} ('{sq}') trả về kết quả không hợp lệ (distance: {candidate_distance}).\n"
                        faiss_answer = answer_question_faiss(sq)
                        response += f"FAISS retrieval:\n{faiss_answer}\n"
                else:
                    response += f"Câu hỏi con {idx} ('{sq}') không trả về kết quả hợp lệ từ ChromaDB (distance: {candidate_distance}).\n"
                    faiss_answer = answer_question_faiss(sq)
                    response += f"FAISS retrieval:\n{faiss_answer}\n"
        else:
            for idx, sq in enumerate(sub_questions, 1):
                faiss_answer = answer_question_faiss(sq)
                response += f"Câu hỏi con {idx} ('{sq}'):\nFAISS retrieval:\n{faiss_answer}\n"
        return response

    def generate_chain_of_thought(self, prompt: str) -> str:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Hãy giải thích quá trình suy nghĩ của bạn theo từng bước bằng tiếng Việt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DEBUG: Lỗi khi tạo chain-of-thought: {e}", file=sys.stderr)
            return "Không tạo được chuỗi suy nghĩ nội bộ."

    def generate_final_answer(self, aggregated_response: str, chain_of_thought: str) -> str:
        try:
            prompt = (
                f"Dựa trên kết quả tổng hợp sau:\n{aggregated_response}\n"
                f"và quá trình suy nghĩ nội bộ sau:\n{chain_of_thought}\n"
                "Hãy đưa ra câu trả lời cuối cùng, ngắn gọn cho truy vấn của người dùng, bằng tiếng Việt. "
                "Nếu trong kết quả có thông tin tìm kiếm trực tuyến, hãy đảm bảo bao gồm link của trang web được tìm thấy."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DEBUG: Lỗi khi tạo câu trả lời cuối cùng: {e}", file=sys.stderr)
            return "Không tạo được câu trả lời cuối cùng."

# =========================
# STREAMLIT - LOGIN & APP (UI States)
# =========================

st.set_page_config(
    page_title="Hệ thống hỏi đáp y tế tiếng Việt",
    page_icon=":hospital:",
    layout="wide"
)

# Nếu chưa có trạng thái ui_state, mặc định là "login"
if "ui_state" not in st.session_state:
    st.session_state["ui_state"] = "login"

# Chuyển giao diện dựa trên trạng thái ui_state

def show_login_ui():
    st.title("Đăng nhập")
    username = st.text_input("Nhập tên của bạn:")
    if st.button("Đăng nhập"):
        if username.strip() != "":
            user = get_user_by_name(username.strip())
            if user:
                st.session_state["user_id"] = user[0]
            else:
                st.session_state["user_id"] = add_user(username.strip())
            st.session_state["ui_state"] = "system"
        else:
            st.error("Vui lòng nhập tên của bạn.")

def show_system_ui():
    # Nút Đăng xuất
    if st.button("Đăng xuất"):
        st.session_state["ui_state"] = "login"
        st.session_state.pop("agent", None)
        st.session_state.pop("user_id", None)
        st.stop()  # Dừng phiên hiện tại để về trang đăng nhập

    st.markdown('<div class="header-title">Hệ thống hỏi đáp y tế tiếng Việt</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #999999;'>Hệ thống tập trung vào truy vấn y tế bằng cách sử dụng cả ChromaDB và FAISS.</p>", unsafe_allow_html=True)
    
    if "agent" not in st.session_state:
        st.session_state["agent"] = OrchestratorAgent(device="cuda", user_id=st.session_state["user_id"])
    
    if "final_answer" not in st.session_state:
        st.session_state["final_answer"] = ""
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""
    if "clarification_needed" not in st.session_state:
        st.session_state["clarification_needed"] = False
    if "clarifying_questions" not in st.session_state:
        st.session_state["clarifying_questions"] = ""
    if "clarification_text" not in st.session_state:
        st.session_state["clarification_text"] = ""
    
    with st.container():
        st.markdown('<div class="subheader">Enter Your Medical Question</div>', unsafe_allow_html=True)
        user_query = st.text_area(
            "Type your question here:",
            placeholder="Ví dụ: Biểu hiện của bệnh tiểu đường trong thai kỳ",
            height=120
        )
        if user_query and user_query.strip() != st.session_state.get("user_query", ""):
            st.session_state["user_query"] = user_query.strip()
            st.session_state["final_answer"] = ""
            st.session_state["clarification_needed"] = False
            st.session_state["clarification_text"] = ""
            st.session_state["clarifying_questions"] = ""
            st.session_state["agent"].conversation_history = []
    
        if st.button("Process Query"):
            if st.session_state["user_query"]:
                result = st.session_state["agent"].run(st.session_state["user_query"])
                if result.startswith("Clarification needed:"):
                    clarifying_qs = result.replace("Clarification needed:", "").strip()
                    st.session_state["clarification_needed"] = True
                    st.session_state["clarifying_questions"] = clarifying_qs
                else:
                    st.session_state["final_answer"] = result
                    st.session_state["clarification_needed"] = False
            else:
                st.error("Please enter a valid medical question.")
    
        if st.session_state["clarification_needed"]:
            st.markdown('<div class="subheader">Clarification Needed</div>', unsafe_allow_html=True)
            st.info(st.session_state["clarifying_questions"])
            choice = st.radio(
                "Your query is ambiguous. Please choose an option:",
                options=["Proceed with current query", "Provide additional clarification"],
                key="clarify_choice"
            )
            if choice == "Proceed with current query":
                if st.button("Proceed"):
                    result = st.session_state["agent"].run(st.session_state["user_query"], force_query=True)
                    st.session_state["final_answer"] = result
                    st.session_state["clarification_needed"] = False
            elif choice == "Provide additional clarification":
                clarification_input = st.text_area("Enter additional clarification:", key="clarify_input")
                if st.button("Submit Clarification"):
                    if clarification_input.strip():
                        st.session_state["clarification_text"] = clarification_input.strip()
                        result = st.session_state["agent"].run(
                            st.session_state["user_query"],
                            clarification=st.session_state["clarification_text"]
                        )
                        if result.startswith("Clarification needed:"):
                            st.session_state["clarification_needed"] = True
                            st.session_state["clarifying_questions"] = result.replace("Clarification needed:", "").strip()
                        else:
                            st.session_state["final_answer"] = result
                            st.session_state["clarification_needed"] = False
                    else:
                        st.warning("Please provide additional clarification.")
    
        if st.session_state["final_answer"]:
            st.markdown('<div class="subheader">Final Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="final-answer">{st.session_state["final_answer"]}</div>', unsafe_allow_html=True)
    
        if st.session_state["agent"].conversation_history:
            st.markdown('<div class="subheader">Conversation History</div>', unsafe_allow_html=True)
            for entry in st.session_state["agent"].conversation_history:
                st.markdown(
                    f"<div class='chat-history'><strong>{entry['role'].capitalize()}:</strong> {entry['message']}</div>",
                    unsafe_allow_html=True
                )
    
        if hasattr(st.session_state["agent"], "chain_of_thought") and st.session_state["agent"].chain_of_thought:
            with st.expander("Agent Reasoning (Chain-of-Thought)"):
                st.markdown(
                    f'<div class="chain-of-thought">{st.session_state["agent"].chain_of_thought}</div>',
                    unsafe_allow_html=True
                )
    
    if st.button("Perform Web Search"):
        from rag_backend import google_search, extract_page_content
        search_results = google_search(st.session_state["user_query"])
        if search_results and "items" in search_results:
            first_result = search_results["items"][0]
            title = first_result.get("title", "No Title")
            link = first_result.get("link", "")
            content = extract_page_content(link)
            web_search_result = (
                f"<b>Tìm kiếm trực tuyến</b><br>"
                f"Tiêu đề: {title}<br>"
                f"Link: <a href='{link}' target='_blank'>{link}</a><br>"
                f"Tóm tắt: {content}"
            )
            st.markdown(web_search_result, unsafe_allow_html=True)
        else:
            st.write("Không tìm thấy kết quả trực tuyến.")
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; font-size: 14px; color: #999999;'>"
        "<b>DISCLAIMER:</b> The information provided by this system is for informational purposes only and is NOT a substitute for professional medical advice."
        "</p>",
        unsafe_allow_html=True
    )

# Chuyển giao diện dựa trên trạng thái ui_state
if st.session_state["ui_state"] == "login":
    show_login_ui()
elif st.session_state["ui_state"] == "system":
    show_system_ui()
