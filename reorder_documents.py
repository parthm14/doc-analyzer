import json
import os

def reorder_documents_lost_in_middle(docs):
    """
    Reorders documents to mitigate the 'lost in the middle' problem:
    puts the most relevant docs at the beginning and end.
    """
    sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
    reordered = []
    left, right = 0, len(sorted_docs) - 1
    toggle = True

    for doc in sorted_docs:
        if toggle:
            reordered.insert(left, doc)
            left += 1
        else:
            reordered.insert(right, doc)
            right -= 1
        toggle = not toggle

    return reordered

# Load reranked documents
file_path = "reranked_docs.json"
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    print("Please run `rerank_documents.py` first to generate reranked_docs.json.")
    exit()

with open(file_path, "r") as f:
    reranked_docs = json.load(f)

# Reorder documents
reordered_docs = reorder_documents_lost_in_middle(reranked_docs)

# Save reordered documents
with open("reordered_docs.json", "w") as f:
    json.dump(reordered_docs, f, indent=2)

# Preview reordered results
print("\n📚 Reordered results using 'Lost in the Middle' mitigation:\n")
for i, doc in enumerate(reordered_docs[:5], start=1):
    print(f"Result {i} (score: {doc['score']:.4f}):\n{'-'*40}\n{doc['content'][:300].strip()}\n")