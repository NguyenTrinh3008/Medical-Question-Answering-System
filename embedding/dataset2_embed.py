import chromadb
import pandas as pd
from embedding_utils import CustomEmbeddingFunction, load_embedding_model

# Load model and tokenizer
print("Loading model and tokenizer...")
model, tokenizer = load_embedding_model()
print("Model and tokenizer loaded successfully.")

# Load dataset
file_path = '/home/ltnga/NguyenTrinhTest/qa_pairs_test.csv'
print(f"Loading dataset from: {file_path}")
dataset = pd.read_csv(file_path)

# Giả sử dataset mới có 2 cột: 'question' và 'answer'
dataset = dataset.dropna(subset=['question', 'answer'])
dataset['question'] = dataset['question'].astype(str).str.strip()
dataset['answer'] = dataset['answer'].astype(str).str.strip()

print(f"Total rows after cleaning: {len(dataset)}")

# Initialize Chroma PersistentClient
print("Initializing Chroma PersistentClient...")
chroma_client = chromadb.PersistentClient(path="./chroma_storage")

# Initialize embedding function
print("Initializing embedding function...")
embedding_function = CustomEmbeddingFunction(model, tokenizer, batch_size=4)

# Create or get the collection
print("Creating or getting the collection...")
collection = chroma_client.get_or_create_collection(
    name="vietnamese_pregancy_data",
    embedding_function=embedding_function
)

# Debug: Check initial collection count
print(f"Initial documents in collection: {collection.count()}")

# Upsert in smaller batches
batch_size = 500
for i in range(0, len(dataset), batch_size):
    batch = dataset.iloc[i:i + batch_size]
    
    # Sử dụng cột 'question' cho embedding
    questions = batch['question'].tolist()
    
    # Tạo id cho mỗi document (ở đây sử dụng index của dataframe, chuyển về string)
    ids = batch.index.astype(str).tolist()
    
    # Metadata có thể bao gồm cả 'question' và 'answer'
    metadatas = batch.to_dict('records')
    
    collection.upsert(documents=questions, ids=ids, metadatas=metadatas)
    print(f"Processed batch {i // batch_size + 1}: {len(questions)} documents.")

# Debug: Final collection count
print(f"Final documents in collection: {collection.count()}")
