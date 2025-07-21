from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch

# === Step 1: Load and chunk the markdown ===
loader = UnstructuredMarkdownLoader("LLM App References from Trinity College Dublin.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

print(f"Loaded {len(chunks)} chunks")

# === Step 2: Initialize embedding model ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Step 3: Connect to OpenSearch ===
opensearch_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False
)

INDEX_NAME = "llm-index"

# === Step 4: Initialize OpenSearch vector store ===
vectorstore = OpenSearchVectorSearch(
    index_name=INDEX_NAME,
    embedding_function=embedding_model,
    opensearch_client=opensearch_client,
    opensearch_url="http://localhost:9200"
)

# === Step 5: Index documents in batches ===
BULK_SIZE = 1000
for i in range(0, len(chunks), BULK_SIZE):
    batch = chunks[i:i + BULK_SIZE]
    vectorstore.add_documents(batch, bulk_size=BULK_SIZE)
    print(f"Indexed batch {i // BULK_SIZE + 1}")

print("🎉 All chunks indexed successfully into OpenSearch.")