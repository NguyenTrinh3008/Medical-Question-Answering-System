from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from chromadb.api.types import EmbeddingFunction
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
# Define the CustomEmbeddingFunction
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model, tokenizer, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]

        # Handle empty or invalid inputs
        if not texts or any(text is None or text.strip() == "" for text in texts):
            raise ValueError("Invalid input for embedding: empty or None text provided.")

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)  
            all_embeddings.extend(embeddings.cpu().numpy().tolist())
            torch.cuda.empty_cache()

        return all_embeddings

# Define a LangChain-compatible embeddings class using Transformers
class CustomSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible embedding class using HuggingFace Transformers."""
    def __init__(self, model, tokenizer, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Use CustomEmbeddingFunction for batching
        return CustomEmbeddingFunction(self.model, self.tokenizer, self.batch_size)(texts)

    def embed_query(self, text: str) -> list[float]:
        # Embed a single query
        return CustomEmbeddingFunction(self.model, self.tokenizer, self.batch_size)([text])[0]

# Function to load the tokenizer and model
def load_embedding_model(model_name="dangvantuan/vietnamese-document-embedding"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map={"": "cpu"},         # ép toàn bộ model lên CPU
        low_cpu_mem_usage=True          # load theo chunks, tiết kiệm RAM
    )
    return model, tokenizer

# Find the best matching answer using cosine similarity
def find_best_answer(query_embedding, results_embeddings, threshold=0.8):
    similarities = cosine_similarity([query_embedding], results_embeddings)[0]
    best_index = similarities.argmax()
    if similarities[best_index] > threshold:
        return best_index
    return None

def query_sub_questions(sub_questions, collection, embedding_function, n_results=1):
    results = {}
    for sub_question in sub_questions:
        # Embed the sub-question
        query_embedding = embedding_function.embed([sub_question])
        
        # Query the collection
        query_results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        # Lấy metadata và khoảng cách
        metadatas = query_results.get('metadatas', [])
        distances = query_results.get('distances', [])

        # Kết hợp metadata và khoảng cách
        combined_results = [
            {"metadata": metadata, "distance": distance}
            for metadata, distance in zip(metadatas, distances)
        ]

        # Sắp xếp kết quả theo khoảng cách (độ tương đồng)
        sorted_results = sorted(combined_results, key=lambda x: x["distance"])

        # Lưu kết quả
        results[sub_question] = sorted_results

    return results
