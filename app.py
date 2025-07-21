import streamlit as st
import os
import google.generativeai as genai
from sentence_transformers import CrossEncoder
import uuid

# ------------------------ CONFIG ------------------------
st.set_page_config(page_title="Doc Analyzer", layout="wide")
st.title("📄 Doc Analyzer")

# Set Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB4jDgtAkVtRC-sReOg6MW3LnACnTx5E_c"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ------------------------ STATE SETUP ------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state or st.session_state.current_chat not in st.session_state.chats:
    # Create a new chat if current chat is not valid
    new_chat_id = str(uuid.uuid4())
    st.session_state.chats[new_chat_id] = []
    st.session_state.current_chat = new_chat_id

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.markdown("### 💬 New Chat")

    if st.button("➕ Start New Chat"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = []
        st.session_state.current_chat = new_chat_id
        st.rerun()

    st.markdown("### 📚 Previous Chats")
    for chat_id, messages in st.session_state.chats.items():
        if messages:
            title = messages[0]["content"][:40] + "..."
        else:
            title = "Untitled"
        if st.button(title, key=chat_id):
            st.session_state.current_chat = chat_id
            st.rerun()

# ------------------------ MAIN CHAT ------------------------
messages = st.session_state.chats[st.session_state.current_chat]

# Show chat history
for msg in messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**AI:** {msg['content']}")

# Chat input
user_query = st.chat_input("Ask something...")

if user_query:
    # Save user message
    messages.append({"role": "user", "content": user_query})

    # Dummy context
    context = "You are a helpful assistant. Use scientific and academic sources to answer precisely."

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question: {user_query}

Answer:
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        answer = f"❌ LLM call failed: {e}"

    # Save assistant response
    messages.append({"role": "assistant", "content": answer})
    st.rerun()