import os
import sys
import sqlite3
import re
from datetime import datetime
import streamlit as st
import asyncio
import concurrent.futures # Để sử dụng executor
import time # Để profiling đơn giản (tùy chọn)
st.set_page_config(page_title="MedQA Tiếng Việt Async", layout="wide")

try:
    # Lấy loop hiện có nếu đang chạy (ví dụ: trong Streamlit > 1.17)
    loop = asyncio.get_running_loop()
except RuntimeError:
    # Tạo loop mới nếu chưa có
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Thay thế bằng cách load an toàn ở trên
OPENAI_API_KEY = "OpenAIKey"
if not OPENAI_API_KEY: # Kiểm tra lại phòng trường hợp bạn xóa key trên nhưng chưa setup env var
     st.error("Chưa cấu hình OpenAI API Key!")
     st.stop()


CHROMA_STORAGE_PATH = "./chroma_storage"
FAISS_INDEX_PATH = "faiss_index"
EMBED_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
ModelName = "/home/tttung/NguyenTrinhTest/model_stage3" # Đảm bảo đường dẫn đúng

# =========================
# QWEN 2.5 FINE-TUNED MODEL SETUP
# =========================
# (Giữ nguyên phần này)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

@st.cache_resource(show_spinner="Đang tải mô hình QWEN...")
def load_qwen_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ModelName, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            ModelName,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32, # float16 chỉ nên dùng trên CUDA
            low_cpu_mem_usage=True,
            # device_map="auto" # Có thể dùng "auto" để tự phân bổ
            device_map={"": device} # Hoặc chỉ định rõ
        )
        # model.to(device) # Có thể không cần nếu device_map đã hoạt động
        print(f"QWEN Model loaded on {model.device}")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình QWEN: {e}")
        st.stop()
        return None, None, None

qwen_tokenizer, qwen_model, device = load_qwen_model()

# =========================
# UTILITIES: DECOMPOSE, SUMMARY, CONFIDENCE
# =========================

# Các hàm này sẽ được gọi từ executor khi cần chạy từ hàm async
def decompose_assertions_qwen_sync(draft: str) -> list[str]:
    if not qwen_model or not qwen_tokenizer: return []
    prompt = (
        "Bạn là một chuyên gia y tế. Tách các luận điểm độc lập (mỗi dòng một luận điểm) "
        f"từ đoạn nháp sau, không thêm kiến thức mới:\n{draft}\nLuận điểm:"
    )
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(device)
    # streamer = TextStreamer(qwen_tokenizer) # Bỏ streamer khi chạy non-interactive
    try:
        output_ids = qwen_model.generate(
            **inputs,
            # streamer=streamer,
            max_new_tokens=500,
            temperature=0.0,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id # Thêm pad_token_id để tránh warning
        )
        raw = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Xử lý output chặt chẽ hơn
        match = re.search(r"Luận điểm:(.*)", raw, re.DOTALL)
        if match:
            content = match.group(1).strip()
            lines = content.splitlines()
            return [line.strip('-•* ').strip() for line in lines if line.strip() and len(line.strip()) > 5] # Lọc dòng trống/quá ngắn
        return []
    except Exception as e:
        print(f"Error during QWEN generation (decompose): {e}")
        return []


def summarize_assertion_qwen_sync(assertion: str, context: str) -> str:
    if not qwen_model or not qwen_tokenizer: return ""
    prompt = (
        "Bạn chỉ tóm tắt thông tin trong phần sau, không thêm kiến thức mới. "
        f"Tóm tắt bằng chứng cho luận điểm '{assertion}':\n{context}\nTóm tắt:"
    )
    inputs = qwen_tokenizer(prompt, return_tensors="pt").to(device)
    # streamer = TextStreamer(qwen_tokenizer)
    try:
        output_ids = qwen_model.generate(
            **inputs,
            # streamer=streamer,
            max_new_tokens=500, # Giảm nếu tóm tắt thường ngắn
            temperature=0.0,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id
        )
        summary = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Xử lý output chặt chẽ hơn
        match = re.search(r"Tóm tắt:(.*)", summary, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback nếu không tìm thấy "Tóm tắt:"
        fallback_summary = summary.split(prompt)[-1].strip() # Lấy phần sau prompt
        return fallback_summary
    except Exception as e:
        print(f"Error during QWEN generation (summarize): {e}")
        return ""

# Hàm đồng bộ gốc để gọi API OpenAI
def assess_confidence_sync(assertion: str, draft: str, client) -> float:
    prompt = (
        "Bạn là chuyên gia y tế.\n"
        f"Đoạn nháp:\n{draft}\n"
        f"Luận điểm: {assertion}\n"
        "Chỉ trả về một số thập phân duy nhất từ 0.0 đến 1.0 thể hiện mức độ chắc chắn rằng luận điểm đã được hỗ trợ đầy đủ bởi đoạn nháp. Không giải thích gì thêm."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", # Đảm bảo model này tồn tại và phù hợp
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        text = resp.choices[0].message.content.strip()
        # Regex chặt hơn để chỉ lấy số thập phân
        m = re.search(r"^\s*(0(\.\d+)?|1(\.0+)?)\s*$", text)
        if m:
            return float(m.group(1))
        else:
            print(f"Warning: Confidence assessment for '{assertion[:30]}...' returned non-numeric: '{text}'. Defaulting to 0.0")
            return 0.0 # Hoặc một giá trị mặc định khác / xử lý lỗi
    except Exception as e:
        print(f"Error assessing confidence for '{assertion[:50]}...': {e}")
        # Có thể thử lại hoặc trả về giá trị mặc định
        return 0.0

# Hàm async wrapper để chạy assess_confidence_sync trong executor
async def assess_confidence_async(assertion: str, draft: str, client, executor) -> float:
    loop = asyncio.get_running_loop()
    try:
        # Chạy hàm đồng bộ trong executor
        score = await loop.run_in_executor(
            executor, assess_confidence_sync, assertion, draft, client
        )
        return score
    except Exception as e:
        # Ghi log lỗi nếu cần
        print(f"Exception in assess_confidence_async wrapper: {e}")
        return 0.0 # Trả về giá trị mặc định khi có lỗi

# =========================
# DATABASE (DB Manager)
# =========================
# (Giữ nguyên phần này, nhưng hàm save_message_db sẽ được gọi từ executor)
conn = sqlite3.connect('conversation_history.db', check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
)
''') # Thêm UNIQUE cho name để tránh trùng lặp

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
    try:
        cursor.execute('INSERT INTO users (name) VALUES (?)', (name,))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # User đã tồn tại
        return get_user_by_name(name)[0] # Trả về id user hiện có
    except Exception as e:
        print(f"Database error adding user: {e}")
        return None

def get_user_by_name(name):
    try:
        cursor.execute('SELECT id FROM users WHERE name=?', (name,))
        return cursor.fetchone()
    except Exception as e:
        print(f"Database error getting user: {e}")
        return None

# Hàm DB đồng bộ
def save_message_db_sync(user_id, role, message):
    if user_id is None: return # Không lưu nếu không có user_id
    ts = datetime.now().isoformat()
    try:
        cursor.execute(
            'INSERT INTO conversation_history (user_id, timestamp, role, message) VALUES (?, ?, ?, ?)',
            (user_id, ts, role, message)
        )
        conn.commit()
    except Exception as e:
        print(f"Database error saving message: {e}")

# =========================
# CACHED INITIALIZATION
# =========================
@st.cache_resource(show_spinner="Đang tải embeddings, retriever, và client...")
def init_embedding_and_retrieval():
    try:
        from embedding_utils import load_embedding_model, CustomEmbeddingFunction, CustomSentenceTransformerEmbeddings
        from langchain_community.vectorstores import FAISS
        import chromadb
        from openai import OpenAI
        embed_model, embed_tokenizer = load_embedding_model(model_name=EMBED_MODEL_NAME)
        embedding_function = CustomEmbeddingFunction(embed_model, embed_tokenizer) # Cho Chroma
        embedding_chain = CustomSentenceTransformerEmbeddings(embed_model, embed_tokenizer) # Cho LangChain/FAISS
        # ---
        # --- Vector Stores ---
        # Chroma (có thể dùng để build index hoặc truy vấn riêng)
        chroma_client = chromadb.PersistentClient(path=CHROMA_STORAGE_PATH)
        collection = chroma_client.get_or_create_collection(
            name="vietnamese_pregnancy_data_local",
            embedding_function=embedding_function
        )

        # FAISS (retriever chính)
        # Đảm bảo file index tồn tại
        if not os.path.exists(FAISS_INDEX_PATH):
             st.error(f"Không tìm thấy FAISS index tại: {FAISS_INDEX_PATH}. Cần tạo index trước.")
             st.stop()
        faiss_vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_chain, # Phải khớp với embedding dùng để tạo index
            allow_dangerous_deserialization=True # Cảnh báo bảo mật!
        )
        retriever = faiss_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8}) # Tăng/giảm k nếu cần

        # --- OpenAI Client ---
        client = OpenAI(api_key=OPENAI_API_KEY)

        # Tạo ThreadPoolExecutor để chạy các hàm đồng bộ
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10) # Điều chỉnh max_workers

        return embedding_function, collection, retriever, client, executor

    except ImportError as e:
        st.error(f"Lỗi import: {e}. Hãy chắc chắn đã cài đặt đầy đủ thư viện.")
        st.stop()
    except Exception as e:
        st.error(f"Lỗi trong quá trình khởi tạo: {e}")
        st.stop()

# Unpack resources
embedding_fn, collection, retriever, client, executor = init_embedding_and_retrieval()

# =========================
# ORCHESTRATOR AGENT (Async Version)
# =========================
class OrchestratorAgent:
    def __init__(self, user_id=None, client=None, retriever=None, executor=None):
        self.user_id = user_id
        self.history = []
        self.sources = {'draft': [], 'evidence': {}}
        self.client = client
        self.retriever = retriever
        self.executor = executor  # Executor để chạy hàm sync

    # --- Các hàm Helper (Async Wrappers) ---
    async def _run_sync_in_executor(self, func, *args):
        """Chạy hàm đồng bộ func(*args) trong executor."""
        loop = asyncio.get_running_loop()   # <— lấy loop đang chạy CORO này
        return await loop.run_in_executor(self.executor, func, *args)

    async def _generate_draft_async(self, question, top_k=8):
        docs = await self._run_sync_in_executor(self.retriever.invoke, question)
        docs = docs[:top_k] # Lấy top_k kết quả
        self.sources['draft'] = [doc.metadata for doc in docs] # Lưu metadata
        ctx = "\n\n".join(d.page_content for d in docs)
        prompt = f"Chỉ sử dụng thông tin dưới đây để viết bản nháp cho câu hỏi: '{question}'\n\n{ctx}\nLiệt kê luận điểm chính, không thêm kiến thức ngoài."

        # Chạy API call trong executor
        resp = await self._run_sync_in_executor(
            lambda: self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Bạn chỉ được dùng thông tin đã cung cấp, không sáng tạo thêm."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500 # Điều chỉnh nếu cần draft dài hơn/ngắn hơn
            )
        )
        return resp.choices[0].message.content.strip()

    async def _decompose_assertions_async(self, draft):
        # Chạy QWEN sync trong executor
        return await self._run_sync_in_executor(decompose_assertions_qwen_sync, draft)

    async def _iterative_retrieval_async(self, assertions, per=4):
        evidence = {}
        # Có thể chạy song song việc retrieve và summarize cho từng assertion nếu muốn tối ưu hơn nữa
        # Nhưng hiện tại chạy tuần tự trong executor để đơn giản
        for a in assertions:
            # Chạy retriever.invoke trong executor
            docs = await self._run_sync_in_executor(self.retriever.invoke, a)
            docs = docs[:per]
            ctx = "\n\n".join(d.page_content for d in docs)

            # Chạy summarize_assertion_qwen_sync trong executor
            summary = await self._run_sync_in_executor(summarize_assertion_qwen_sync, a, ctx)
            if summary: # Chỉ thêm nếu có tóm tắt
                 evidence[a] = summary

        self.sources['evidence'] = evidence
        return evidence

    async def _synthesize_answer_async(self, question, draft, evidence):
        ev_text = "\n".join(f"- {k}: {v}" for k, v in evidence.items())
        if not ev_text:
            ev_text = "Không tìm thấy bằng chứng bổ sung."

        prompt = (
            f"Bạn có:\n1) Bản nháp ban đầu:\n{draft}\n\n2) Bằng chứng bổ sung cho một số luận điểm:\n{ev_text}\n\n"
            f"Dựa vào cả bản nháp và bằng chứng bổ sung (nếu có), hãy tổng hợp một câu trả lời cuối cùng, mạch lạc và đầy đủ cho câu hỏi sau: '{question}'. Trích dẫn nguồn hoặc bằng chứng khi thích hợp (ví dụ: 'Theo bằng chứng [luận điểm], ...')."
        )

        # Chạy API call trong executor
        resp = await self._run_sync_in_executor(
            lambda: self.client.chat.completions.create(
                model="gpt-4o-mini", # Hoặc model mạnh hơn nếu cần chất lượng tổng hợp cao hơn
                messages=[
                    {"role": "system", "content": "Tổng hợp câu trả lời dựa trên thông tin được cung cấp. Ưu tiên bằng chứng bổ sung nếu có mâu thuẫn với bản nháp."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1, # Có thể tăng nhẹ để linh hoạt hơn
                max_tokens=800
            )
        )
        return resp.choices[0].message.content.strip()

    # --- Luồng chính (Async) ---

    async def add_history(self, role, msg):
        """Thêm vào history và lưu DB (async)."""
        self.history.append({"role": role, "message": msg})
        if self.user_id is not None:
            # Chạy hàm lưu DB đồng bộ trong executor
            await self._run_sync_in_executor(save_message_db_sync, self.user_id, role, msg)

    # --- Adaptive RAG (Async) ---
    async def run_adaptive(self, question: str, threshold: float = 0.7, max_depth: int = 2) -> str: # Giảm ngưỡng và độ sâu mặc định
        st.write("Bắt đầu Adaptive RAG...")
        start_time = time.time()

        # 1. Generate Draft
        draft_start = time.time()
        draft = await self._generate_draft_async(question)
        st.write(f"   - Tạo draft xong: {time.time() - draft_start:.2f}s")
        if not draft:
            st.warning("Không thể tạo bản nháp ban đầu.")
            return "Xin lỗi, tôi không thể tạo bản nháp cho câu hỏi này."

        # 2. Decompose Assertions
        decompose_start = time.time()
        assertions = await self._decompose_assertions_async(draft)
        st.write(f"   - Tách luận điểm ({len(assertions)}) xong: {time.time() - decompose_start:.2f}s")
        if not assertions:
            st.warning("Không thể tách luận điểm từ bản nháp. Trả về bản nháp trực tiếp.")
            # Nếu không tách được, có thể trả về draft luôn hoặc chỉ dựa vào draft để synthesize
            final_answer = await self._synthesize_answer_async(question, draft, {})
            return final_answer

        # 3. Adaptive Loop
        evidence: dict[str, str] = {}
        current_assertions = list(assertions) # Làm việc trên bản copy
        depth = 0
        while depth < max_depth:
            loop_start = time.time()
            st.write(f"   - Bắt đầu vòng lặp thích ứng {depth + 1}/{max_depth}...")

            # 3a. Assess Confidence (Parallel)
            assess_start = time.time()
            assessment_tasks = [
                assess_confidence_async(a, draft, self.client, self.executor) for a in current_assertions
            ]
            confidences = await asyncio.gather(*assessment_tasks)
            st.write(f"      - Đánh giá độ tin cậy ({len(current_assertions)} luận điểm) xong: {time.time() - assess_start:.2f}s")

            low_conf_assertions = [
                a for i, a in enumerate(current_assertions) if confidences[i] < threshold
            ]

            if not low_conf_assertions:
                st.write(f"      - Không còn luận điểm yếu (threshold={threshold}). Kết thúc vòng lặp.")
                break

            st.write(f"      - Tìm thấy {len(low_conf_assertions)} luận điểm yếu cần tìm thêm bằng chứng.")

            # 3b. Iterative Retrieval
            retrieve_start = time.time()
            # Lấy bằng chứng cho các luận điểm yếu
            new_evidence = await self._iterative_retrieval_async(low_conf_assertions)
            st.write(f"      - Truy xuất/tóm tắt bằng chứng xong: {time.time() - retrieve_start:.2f}s")
            evidence.update(new_evidence)

            # Cập nhật assertions cho vòng lặp sau (ví dụ: chỉ chạy lại cho low_conf?)
            # Hoặc đơn giản là tăng depth và dựa vào synthesize cuối cùng
            # current_assertions = low_conf_assertions # Nếu muốn chỉ kiểm tra lại những cái đã yếu

            depth += 1
            st.write(f"   - Kết thúc vòng lặp {depth}: {time.time() - loop_start:.2f}s")


        # 4. Synthesize Final Answer
        synthesize_start = time.time()
        st.write("Tổng hợp câu trả lời cuối cùng...")
        final_answer = await self._synthesize_answer_async(question, draft, evidence)
        st.write(f"   - Tổng hợp xong: {time.time() - synthesize_start:.2f}s")
        st.write(f"Tổng thời gian Adaptive RAG: {time.time() - start_time:.2f}s")

        return final_answer

    # --- ReAct RAG (Giữ nguyên đồng bộ, cần chuyển sang async nếu muốn tối ưu) ---
    def run_react_on_assertions(self, question, draft, assertions):
        """
        ReAct loop hoàn toàn trên QWEN (Synchronous).
        Cần chuyển sang async và dùng executor nếu muốn chạy song song với UI.
        """
        st.warning("Chức năng ReAct hiện đang chạy đồng bộ và có thể chặn giao diện.")
        if not qwen_model or not qwen_tokenizer: return "Lỗi: Mô hình QWEN chưa sẵn sàng cho ReAct."

        system_prompt = (
            "Bạn là chuyên gia y tế và có thể gọi công cụ retriever khi cần.\n"
            "Dùng format:\n"
            "  Thought: <suy nghĩ>\n"
            "  Action: retrieve(\"<truy vấn>\")\n"
            "Và khi kết thúc, luôn dùng:\n"
            "  Action: Answer <nội dung trả lời cuối cùng>\n"
        )
        user_prompt = (
            f"Đoạn nháp:\n{draft}\n"
            "Các luận điểm cần evidence:\n" +
            "\n".join(f"- {a}" for a in assertions) +
            f"\nHỏi: {question}\nBắt đầu với Thought/Action format."
        )
        history_react = [system_prompt, user_prompt]
        max_react_steps = 5 # Giới hạn số bước ReAct

        for _ in range(max_react_steps):
            prompt = "\n\n".join(history_react)
            inputs = qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device) # Giới hạn độ dài prompt
            try:
                 output_ids = qwen_model.generate(
                     **inputs,
                     max_new_tokens=700,
                     temperature=0.0,
                     do_sample=False,
                     pad_token_id=qwen_tokenizer.eos_token_id
                 )
                 # Lấy phần text mới được sinh ra thôi
                 new_text = qwen_tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                 history_react.append(new_text) # Chỉ thêm phần mới vào history
                 st.text(f"ReAct Step Output:\n{new_text}") # Hiển thị bước ReAct

                 # retrieve action
                 m = re.search(r'Action:\s*retrieve\("(.+?)"\)', new_text, re.IGNORECASE)
                 if m:
                     query = m.group(1)
                     st.write(f"   - ReAct retrieving: '{query}'")
                     docs = self.retriever.invoke(query)[:4] # Chạy retriever đồng bộ
                     obs = "\n\n".join(d.page_content for d in docs)
                     history_react.append(f"Observation:\n{obs}")
                     st.write(f"   - ReAct Observation added ({len(obs)} chars).")
                     continue # Tiếp tục vòng lặp ReAct

                 # final answer
                 answer_match = re.search(r'Action:\s*Answer(.*)', new_text, re.IGNORECASE | re.DOTALL)
                 if answer_match:
                     final_answer_react = answer_match.group(1).strip()
                     st.write("   - ReAct found final answer.")
                     return final_answer_react

            except Exception as e:
                 st.error(f"Lỗi trong bước ReAct: {e}")
                 return "Đã có lỗi xảy ra trong quá trình ReAct."


        st.warning("ReAct không tìm thấy câu trả lời cuối cùng sau số bước tối đa.")
        return "Không thể hoàn thành yêu cầu bằng ReAct." # Trả về thông báo lỗi/mặc định

    # --- Main run method (Async) ---
    async def run(self, question: str, use_react: bool = False) -> str:
        """Chạy RAG chính (Async)."""
        await self.add_history("user", question)
        self.sources = {'draft': [], 'evidence': {}} # Reset sources

        answer = ""
        if use_react:
            st.write("Sử dụng luồng ReAct...")
            # Chạy các bước chuẩn bị (có thể cần async)
            draft = await self._generate_draft_async(question)
            if not draft: return "Lỗi khi tạo draft cho ReAct."
            assertions = await self._decompose_assertions_async(draft)
            if not assertions: return "Lỗi khi tách luận điểm cho ReAct."

            # Chạy ReAct (hiện đang đồng bộ) trong executor
            answer = await self._run_sync_in_executor(
                self.run_react_on_assertions, question, draft, assertions
            )
        else:
            st.write("Sử dụng luồng Adaptive RAG...")
            answer = await self.run_adaptive(question) # Đã là async

        await self.add_history("assistant", answer)
        return answer

# =========================
# STREAMLIT UI (Async Handling)
# =========================


# Khởi tạo session state
if "ui_state" not in st.session_state:
    st.session_state.ui_state = "login"
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "agent" not in st.session_state:
    # Khởi tạo agent khi cần, sau khi có user_id
    st.session_state.agent = None

# --- Login Logic ---
if st.session_state.ui_state == "login":
    st.title("Đăng nhập MedQA")
    username = st.text_input("Nhập tên của bạn:", key="login_username")
    if st.button("Đăng nhập", key="login_button") and username:
        user_record = get_user_by_name(username)
        user_id = user_record[0] if user_record else add_user(username)
        if user_id is not None:
            st.session_state.user_id = user_id
            st.session_state.ui_state = "chat"
            # Khởi tạo agent sau khi đăng nhập thành công
            st.session_state.agent = OrchestratorAgent(
                user_id=st.session_state.user_id,
                client=client,
                retriever=retriever,
                executor=executor
            )
            st.rerun() # Chạy lại script để vào giao diện chat
        else:
            st.error("Không thể lấy hoặc tạo thông tin người dùng.")

# --- Chat Logic ---
elif st.session_state.ui_state == "chat":
    if st.session_state.agent is None:
        st.error("Lỗi: Agent chưa được khởi tạo. Vui lòng đăng nhập lại.")
        st.session_state.ui_state = "login"
        st.rerun()

    # --- Sidebar ---
    with st.sidebar:
        st.write(f"User ID: {st.session_state.user_id}")
        if st.button("Đăng xuất", key="logout_button"):
            # Xóa trạng thái liên quan đến người dùng
            st.session_state.user_id = None
            st.session_state.agent = None
            st.session_state.ui_state = "login"
            st.rerun()
        st.divider()
        use_react = st.checkbox("Dùng ReAct-style retrieval", value=False, key="use_react_checkbox")
        st.divider()
        # Thêm các tùy chọn khác nếu cần (threshold, max_depth, ...)

    # --- Main Chat Area ---
    st.title("MedQA Tiếng Việt (Async RAG)")
    query = st.text_area("Nhập câu hỏi của bạn:", key="query_input", height=150)

    if st.button("Gửi", key="send_button") and query:
        start_run_time = time.time()
        with st.spinner("Đang xử lý yêu cầu..."):
            try:
                async def run_agent_async():
                    return await st.session_state.agent.run(query, use_react=use_react)
                current_loop = asyncio.get_event_loop()
                if current_loop.is_running():
                     try:
                          answer = asyncio.run(run_agent_async()) # Thử asyncio.run trước
                     except RuntimeError: # Nếu đã có loop chạy, asyncio.run sẽ lỗi
                          # Chạy trong loop hiện tại
                          answer = current_loop.run_until_complete(run_agent_async())

                else:
                    # Nếu không có loop nào chạy, tự chạy
                    answer = asyncio.run(run_agent_async())


                st.success(f"Xử lý xong! (Tổng thời gian: {time.time() - start_run_time:.2f}s)")

                st.markdown("### Câu trả lời:")
                st.markdown(answer)
                st.markdown("---")

                # Hiển thị nguồn (nếu agent đã cập nhật self.sources)
                if hasattr(st.session_state.agent, 'sources'):
                    st.subheader("Nguồn RAG ban đầu (Metadata)")
                    st.json(st.session_state.agent.sources.get('draft', "N/A"))
                    st.subheader("Bằng chứng cho từng luận điểm (Tóm tắt)")
                    st.json(st.session_state.agent.sources.get('evidence', "N/A"))

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
                # In stack trace để debug
                import traceback
                st.text(traceback.format_exc())