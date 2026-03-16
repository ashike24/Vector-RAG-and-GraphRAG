# Code 1

!pip install langchain langchain-openai langchain-community
!pip install langchain-experimental langchain-chroma langchain-huggingface
!pip install neo4j chromadb sentence-transformers tiktoken

# Code 2

import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Config ──────────────────────────────────────────────────
FILE_PATH        = "/content/drive/MyDrive/UGP1/RAG1.txt"
PERSIST_DIR      = "/tmp/chroma_db"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 100
TOP_K            = 5
# ────────────────────────────────────────────────────────────


def clean_old_store():
    """Delete any broken or readonly leftover vector stores."""
    for path in ["/content/chroma_db", "/tmp/chroma_db"]:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"🗑️  Deleted old store: {path}")


def load_and_split(file_path: str):
    """Load a single .txt file and split into chunks."""
    print(f"📄 Loading document: {file_path}")
    loader = TextLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


def get_embedding_model():
    """Load free HuggingFace embedding model."""
    print("🤖 Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded")
    return embedding_model


def build_or_load_vectorstore(chunks, embedding_model):
    """Build ChromaDB vector store or reload if valid."""

    if os.path.exists(PERSIST_DIR):
        try:
            temp = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embedding_model,
                collection_metadata={"hnsw:space": "cosine"}
            )
            count = temp._collection.count()
            if count > 0:
                print(f"♻️  Loading existing vector store ({count} vectors)...")
                return temp
            else:
                print("⚠️  Existing store is empty. Rebuilding...")
                shutil.rmtree(PERSIST_DIR)
        except Exception as e:
            print(f"⚠️  Could not load existing store ({e}). Rebuilding...")
            shutil.rmtree(PERSIST_DIR)

    print("🔨 Building new vector store...")
    os.makedirs(PERSIST_DIR, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"✅ Vector store saved to {PERSIST_DIR}")
    print(f"   Total vectors stored: {vectorstore._collection.count()}")
    return vectorstore


def retrieve(vectorstore, query: str, top_k: int = TOP_K):
    """Retrieve top-k most relevant chunks for the query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)
    return docs


def display_results(query: str, docs):
    print("\n" + "=" * 60)
    print(f"🔍 Query : {query}")
    print(f"📦 Top-{len(docs)} chunks via Vector RAG")
    print("=" * 60)
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source : {doc.metadata.get('source', 'N/A')}")
        print(f"Content:\n{doc.page_content}")
        print("-" * 40)


def main():
    print("=== Vector RAG Pipeline (HuggingFace Embeddings) ===\n")

    # Step 0 – Clean any broken old stores
    clean_old_store()

    # Step 1 – Load & split
    chunks = load_and_split(FILE_PATH)

    # Step 2 – Load embedding model
    embedding_model = get_embedding_model()

    # Step 3 – Build / load vector store
    vectorstore = build_or_load_vectorstore(chunks, embedding_model)

    # Step 4 – Query
    query = input("\n💬 Enter your query: ").strip()
    docs  = retrieve(vectorstore, query)

    # Step 5 – Display
    display_results(query, docs)


if __name__ == "__main__":
    main()
