# DocAnalyzer – LLM-Powered Markdown File Chatbot

**DocAnalyzer** is a document-based Q&A chatbot built with **LangChain**, **Google Gemini Pro**, **OpenSearch**, and **Streamlit**. It lets users upload markdown files, ask questions in natural language, and get grounded, AI-generated answers with memory and chat history.

---

## Features

- **Markdown Uploading**: Upload and analyze `.md` documents in the browser.
- **Hybrid Search**: Uses OpenSearch for both BM25 (keyword-based) and dense vector search.
- **Reranking**: Uses SentenceTransformers’ CrossEncoder for relevance scoring.
- **Lost-in-the-Middle Fix**: Reorders documents to keep most relevant info visible to the LLM.
- **LLM Answering**: Generates grounded answers using Google Gemini Pro.
- **Conversational UI**: Looks and feels like ChatGPT, with session memory and query continuity.
- **Previous Chat Sidebar**: Save and revisit earlier chats with topic summaries.
- **Streamlit App**: Clean, interactive UI for querying and answering.

---

## Tech Stack

| Layer         | Tool/Service                      |
|---------------|----------------------------------|
| Embeddings    | HuggingFace SentenceTransformers |
| Vector Store  | OpenSearch (Docker)              |
| LLM           | Gemini Pro via MakerSuite        |
| Reranker      | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Chunking      | LangChain                        |
| UI            | Streamlit                        |

---

## 📁 File Structure

```bash
doc-analyzer/
│
├── app.py                  # Streamlit UI + memory
├── index_documents.py      # Markdown chunking and OpenSearch ingestion
├── retrieve_documents.py   # Hybrid retrieval (BM25 + dense)
├── rerank_documents.py     # CrossEncoder-based reranking
├── reorder_documents.py    # Lost-in-the-middle mitigation
├── generate_answer.py      # Final Gemini LLM generation
├── requirements.txt
└── README.md
