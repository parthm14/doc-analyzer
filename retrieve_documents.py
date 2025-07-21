import json
from operator import itemgetter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from opensearchpy import OpenSearch

# Set query
QUERY = "What nutrients are important in early childhood?"

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# OpenSearch client
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False
)

INDEX_NAME = "llm-index"

# Dense retriever
dense_retriever = OpenSearchVectorSearch(
    index_name=INDEX_NAME,
    embedding_function=embedding_model,
    opensearch_client=client,
    opensearch_url="http://localhost:9200"
)

# BM25 retriever via native API
bm25_response = client.search(
    index=INDEX_NAME,
    body={
        "query": {
            "match": {
                "content": QUERY
            }
        },
        "size": 10
    }
)

bm25_docs = [
    Document(page_content=hit["_source"]["content"], metadata=hit["_source"].get("metadata", {}))
    for hit in bm25_response["hits"]["hits"]
]

# Dense results
dense_docs = dense_retriever.similarity_search(query=QUERY, k=10)

# Deduplicate based on content
seen = set()
unique_docs = []
for doc in bm25_docs + dense_docs:
    if doc.page_content not in seen:
        unique_docs.append(doc)
        seen.add(doc.page_content)

# Lost-in-the-middle reordering heuristic
def lost_in_middle_score(doc: Document):
    metadata = doc.metadata or {}
    pos = 0
    for key in ["line", "page", "chunk", "position"]:
        if key in metadata:
            try:
                pos = int(metadata[key])
                break
            except:
                continue
    if pos <= 2 or pos >= 98:  # edge chunks
        return 2
    elif pos <= 10 or pos >= 90:
        return 1
    else:
        return 0

# Reorder based on heuristic
reordered_docs = sorted(unique_docs, key=lost_in_middle_score, reverse=True)

# Save to JSON for reranking
with open("retrieved_docs.json", "w") as f:
    json.dump(
        [{"content": doc.page_content, "metadata": doc.metadata} for doc in reordered_docs[:10]],
        f,
        indent=2
    )

# Show top results
print(f"\n🔍 Searching for: \"{QUERY}\"...\n")
for i, doc in enumerate(reordered_docs[:10]):
    print(f"Result {i+1}:\n{'-'*10}\n{doc.page_content[:500]}\n\nMetadata: {doc.metadata}\n")