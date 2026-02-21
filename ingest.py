import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------------
# 1. Load Environment Variables (optional now)
# -----------------------------------
load_dotenv("abc.env")

# -----------------------------------
# 2. Paths Configuration
# -----------------------------------
BASE_PATH = "Data"
LOGS_PATH = os.path.join(BASE_PATH, "logs")
INCIDENTS_PATH = os.path.join(BASE_PATH, "incidents")
VECTOR_STORE_PATH = "vector_store"

# -----------------------------------
# 3. Load Documents
# -----------------------------------
def load_documents(folder_path, doc_type):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = filename
                doc.metadata["document_type"] = doc_type

            documents.extend(docs)

    return documents


print("📂 Loading log files...")
log_docs = load_documents(LOGS_PATH, "log")

print("📂 Loading incident files...")
incident_docs = load_documents(INCIDENTS_PATH, "incident")

all_documents = log_docs + incident_docs

if not all_documents:
    raise ValueError("No documents found in Data folder")

print(f"✅ Total documents loaded: {len(all_documents)}")

# -----------------------------------
# 4. Split into Chunks
# -----------------------------------
# -----------------------------------
# 4. Split into Chunks (Smart Split)
# -----------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Split logs only
log_chunks = text_splitter.split_documents(log_docs)

# Keep incidents whole (IMPORTANT)
incident_chunks = incident_docs

chunks = log_chunks + incident_chunks

print(f"✅ Total chunks created: {len(chunks)}")
# -----------------------------------
# 5. Create HuggingFace Embeddings (FREE)
# -----------------------------------
print("🔄 Creating HuggingFace embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# 6. Create FAISS Vector Store
# -----------------------------------
vector_store = FAISS.from_documents(chunks, embeddings)

if not os.path.exists(VECTOR_STORE_PATH):
    os.makedirs(VECTOR_STORE_PATH)

vector_store.save_local(VECTOR_STORE_PATH)

print("🎉 Vector store created successfully using HuggingFace!")
print(f"📁 Saved at: {VECTOR_STORE_PATH}")