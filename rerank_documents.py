from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
import json

# Load retrieved documents
with open("retrieved_docs.json", "r") as f:
    retrieved_docs = json.load(f)

# Convert to Document objects
docs = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in retrieved_docs]

# Prepare input for CrossEncoder
query = input("Enter your query again for reranking: ")
pairs = [(query, doc.page_content) for doc in docs]

# Load CrossEncoder model
print("Reranking with CrossEncoder...")
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Compute similarity scores
scores = model.predict(pairs)

# Sort documents by score
ranked_results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

# Display results
print(f"Reranked results for: \"{query}\"\n")
for i, (doc, score) in enumerate(ranked_results[:10]):
    print(f"Result {i+1} (score: {score:.4f})\n{'-'*40}")
    print(doc.page_content[:500])
    print(f"Metadata: {doc.metadata}\n")

# ✅ Save reranked documents to JSON for next step
reranked_output = [
    {
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": float(score)
    }
    for doc, score in ranked_results
]

with open("reranked_docs.json", "w") as f:
    json.dump(reranked_output, f, indent=2)

print("✅ Saved reranked results to reranked_docs.json")