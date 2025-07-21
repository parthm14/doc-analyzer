import os
import json
import google.generativeai as genai

# Set your Gemini MakerSuite API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB4jDgtAkVtRC-sReOg6MW3LnACnTx5E_c"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load user query
query = input("Enter your query again for final answer generation: ")

# Load context from reordered documents
try:
    with open("reordered_docs.json", "r") as f:
        docs = json.load(f)
except FileNotFoundError:
    print("❌ File 'reordered_docs.json' not found. Please run `reorder_documents.py` first.")
    exit()

# Create a context string from the top documents
context = "\n\n".join([doc["content"] for doc in docs[:4]])

# Construct the prompt
prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:
"""

# Call Gemini Pro
print("\n💬 Generating answer using Gemini Pro...\n")

try:
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    print("✅ Answer:\n")
    print(response.text.strip())
except Exception as e:
    print(f"❌ LLM call failed:\n{e}")