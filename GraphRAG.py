!pip install -q langchain langchain-core langchain-openai langchain-community
!pip install -q langchain-experimental langchain-huggingface
!pip install -q neo4j sentence-transformers tiktoken
!pip install langchain-neo4j

from neo4j import GraphDatabase

uri      = "aaa"
username = "aaa"
password = "aaa"

driver = GraphDatabase.driver(uri, auth=(username, password))
with driver.session(database="system") as session:
    result = session.run("SHOW DATABASES")
    for record in result:
        print(record["name"], "→ status:", record["currentStatus"])
driver.close()

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

os.environ["OPENAI_API_KEY"] = "aaa"
os.environ["NEO4J_URI"]      = "aaa"
os.environ["NEO4J_USERNAME"] = "aaa"
os.environ["NEO4J_PASSWORD"] = "aaa"

NEO4J_DATABASE = "aaa"

# ── Config ──────────────────────────────────────────────────
FILE_PATH     = "/content/drive/MyDrive/UGP1/RAG1.txt"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 100
TOP_K         = 5
# ────────────────────────────────────────────────────────────


# ── 1. Load & Split ─────────────────────────────────────────
def load_and_split(file_path: str):
    print(f"📄 Loading document: {file_path}")
    loader    = TextLoader(file_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ── 2. Free HuggingFace Embedding Model ─────────────────────
def get_embedding_model():
    print("🤖 Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("✅ Embedding model loaded")
    return embedding_model


# ── 3. Connect to Neo4j ─────────────────────────────────────
def connect_neo4j() -> Neo4jGraph:
    print("🔌 Connecting to Neo4j...")
    graph = Neo4jGraph(
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
        database=NEO4J_DATABASE
    )
    print("✅ Connected to Neo4j")
    return graph


# ── 4. Check if graph already populated ─────────────────────
def graph_already_populated(graph: Neo4jGraph) -> bool:
    try:
        result = graph.query("MATCH (n) RETURN count(n) AS count")
        count  = result[0]["count"] if result else 0
        print(f"   Neo4j node count: {count}")
        return count > 0
    except Exception as e:
        print(f"   ⚠️  Could not check graph population: {e}")
        return False


# ── 5. Extract entities & relationships using GPT ───────────
def build_knowledge_graph(chunks, graph: Neo4jGraph):
    """
    Uses GPT-4o-mini to extract (entity → relation → entity)
    triples from every chunk and stores them in Neo4j.
    Runs ONCE — skipped automatically on subsequent runs.
    """
    print("\n🧠 Extracting entities & relationships with GPT-4o-mini...")
    print("   (This may take a few minutes for large documents...)")

    llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    transformer = LLMGraphTransformer(llm=llm)

    batch_size    = 10
    all_graph_docs = []
    total_batches  = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"   Processing batch {i // batch_size + 1}/{total_batches}...")
        try:
            graph_docs = transformer.convert_to_graph_documents(batch)
            all_graph_docs.extend(graph_docs)
        except Exception as e:
            print(f"   ⚠️  Batch error: {e}. Skipping batch.")

    print(f"\n   ✅ Total graph documents created: {len(all_graph_docs)}")

    # Preview first 3
    for i, gd in enumerate(all_graph_docs[:3]):
        print(f"\n   --- Graph Doc {i+1} ---")
        print(f"   Nodes    : {[n.id for n in gd.nodes[:5]]}")
        print(f"   Relations: {[(r.source.id, r.type, r.target.id) for r in gd.relationships[:5]]}")

    # Store in Neo4j
    graph.add_graph_documents(
        all_graph_docs,
        baseEntityLabel=True,
        include_source=True
    )
    print("\n✅ Knowledge graph stored in Neo4j")
    return all_graph_docs


# ── 6. Build vector index on Neo4j ──────────────────────────
def build_neo4j_vector_index(chunks, embedding_model):
    print("\n🔨 Building vector index in Neo4j (HuggingFace embeddings)...")

    try:
        vectorstore = Neo4jVector.from_existing_graph(
            embedding=embedding_model,
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            database=NEO4J_DATABASE,
            index_name="chunk_vector_index",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        print("✅ Neo4j vector index ready (from existing graph)")
        return vectorstore

    except Exception as e:
        print(f"⚠️  from_existing_graph failed ({e}).")
        print("   Falling back to from_documents...")

        vectorstore = Neo4jVector.from_documents(
            documents=chunks,
            embedding=embedding_model,
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
            database=NEO4J_DATABASE,
            index_name="chunk_vector_index",
        )
        print("✅ Neo4j vector index ready (from documents fallback)")
        return vectorstore


# ── 7. Graph-aware retrieval ─────────────────────────────────
def retrieve_with_graph(vectorstore, graph: Neo4jGraph, query: str, top_k: int = TOP_K):
    """
    Step A – vector search finds seed chunks.
    Step B – Cypher traversal finds connected entities (1-hop neighbours).
    """
    print(f"\n🔍 Running vector search for seed chunks...")
    seed_docs = vectorstore.similarity_search(query, k=top_k)
    print(f"   Found {len(seed_docs)} seed chunks")

    enriched_docs = []
    seen_texts    = set()

    for doc in seed_docs:
        chunk_text = doc.page_content.strip()
        if chunk_text in seen_texts:
            continue
        seen_texts.add(chunk_text)
        enriched_docs.append(doc)

        # Graph traversal: find entities connected to this chunk
        cypher = """
        MATCH (d:Document)-[:MENTIONS]->(e)
        WHERE d.text CONTAINS $snippet
        RETURN e.id AS entity, labels(e) AS types
        LIMIT 10
        """
        snippet = chunk_text[:200]
        try:
            results = graph.query(cypher, params={"snippet": snippet})
            if results:
                entity_str = ", ".join(
                    [f"{r['entity']} ({r['types'][0] if r['types'] else 'Entity'})"
                     for r in results]
                )
                meta_doc = Document(
                    page_content=f"[Graph context]\nConnected entities: {entity_str}",
                    metadata={"source": "neo4j_graph", "type": "graph_context"}
                )
                if meta_doc.page_content not in seen_texts:
                    seen_texts.add(meta_doc.page_content)
                    enriched_docs.append(meta_doc)
        except Exception as e:
            print(f"   ⚠️  Graph traversal warning: {e}")

    return enriched_docs[:top_k]


# ── 8. Display ───────────────────────────────────────────────
def display_results(query: str, docs):
    print("\n" + "=" * 60)
    print(f"🔍 Query : {query}")
    print(f"📦 Top-{len(docs)} results via Graph RAG (Neo4j)")
    print("=" * 60)
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "N/A")
        dtype  = doc.metadata.get("type", "chunk")
        print(f"\n--- Result {i} [{dtype}] ---")
        print(f"Source : {source}")
        print(f"Content:\n{doc.page_content}")
        print("-" * 40)


# ── 9. Main ──────────────────────────────────────────────────
def main():
    print("=== Graph RAG Pipeline (Neo4j + HuggingFace Embeddings) ===\n")

    # Step 1 – Load & split
    chunks = load_and_split(FILE_PATH)

    # Step 2 – Load free embedding model
    embedding_model = get_embedding_model()

    # Step 3 – Connect to Neo4j
    graph = connect_neo4j()

    # Step 4 – Build knowledge graph only if Neo4j is empty
    print("\n🔎 Checking if knowledge graph already exists...")
    if graph_already_populated(graph):
        print("✅ Graph already populated. Skipping ingestion.")
    else:
        print("   Graph is empty. Building knowledge graph...")
        build_knowledge_graph(chunks, graph)

    # Step 5 – Build vector index on Neo4j
    vectorstore = build_neo4j_vector_index(chunks, embedding_model)

    # Step 6 – Query
    query = input("\n💬 Enter your query: ").strip()
    docs  = retrieve_with_graph(vectorstore, graph, query)

    # Step 7 – Display
    display_results(query, docs)


if __name__ == "__main__":
    main()
