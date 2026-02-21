import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# -----------------------------------
# 1. Load Embeddings (Same as ingest.py)
# -----------------------------------
print("🔄 Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------------
# 2. Load Vector Store
# -----------------------------------
print("📂 Loading vector store...")
vector_store = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -----------------------------------
# 3. Load FREE LLM (Flan-T5)
# -----------------------------------
print("🤖 Loading local LLM...")

model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512
)

llm = HuggingFacePipeline(pipeline=pipe)

# -----------------------------------
# 4. Create Retrieval QA Chain
# -----------------------------------
# -----------------------------------
# Custom Prompt for Cause & Fix
# -----------------------------------

prompt_template = """
You are a Spark Pipeline Failure Assistant.

Use the provided context to answer clearly.

If the context contains:
- Error
- Cause
- Fix

Then return answer strictly in this format:

Cause:
<clear explanation>

Fix:
<clear solution steps>

If information is not available, say:
"Cause and Fix not found in knowledge base."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# -----------------------------------
# 5. Ask Question
# -----------------------------------
print("\n🚀 Pipeline Issue Assistant Ready!")
print("Type 'exit' to stop.\n")

while True:
    query = input("Ask about pipeline issue: ")

    if query.lower() == "exit":
        break

    response = qa_chain(query)

    print("\n📌 Answer:")
    print(response["result"])

    print("\n📂 Sources Used:")
    for doc in response["source_documents"]:
        print(f"- {doc.metadata.get('source_file')} ({doc.metadata.get('document_type')})")

    print("\n" + "-"*60 + "\n")