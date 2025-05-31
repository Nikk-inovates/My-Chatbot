import streamlit as st
import os
import sys
import json
from datetime import datetime
import traceback

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_pdf import load_pdf_text
from src.embed_text import split_text, embed_chunks, save_faiss_index, load_faiss_index
from main import ask_question, search_chunks  # Removed setup_deepseek import
from src.chatbot import log_chat_to_history

st.set_page_config(page_title="Chat with Your PDF", page_icon="📄", layout="centered")
st.title("📄 Chat with Dataset.pdf using DeepSeek + FAISS")

# Sidebar Chat History
st.sidebar.title("📚 Chat History")
history_file = "logs/chat_history.json"

if os.path.exists(history_file):
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
        if chat_history:
            for chat in reversed(chat_history[-10:]):
                with st.sidebar.expander(f"🕒 {chat['timestamp']}"):
                    st.markdown(f"**Q:** {chat['question']}")
                    st.markdown(f"**A:** {chat['answer']}")
        else:
            st.sidebar.info("No chat history yet.")
    except Exception as e:
        st.sidebar.warning("⚠️ Failed to load chat history.")
        print(traceback.format_exc())
else:
    st.sidebar.info("No chat history yet.")

# PDF Path
pdf_path = "data/knowledge.pdf"
if not os.path.exists(pdf_path):
    st.error("❌ PDF file not found at: data/knowledge.pdf")
    st.stop()

# Load PDF
with st.spinner("📄 Loading PDF..."):
    try:
        text = load_pdf_text(pdf_path)
    except Exception as e:
        st.error(f"❌ Failed to load PDF: {e}")
        st.stop()

# Embed chunks
with st.spinner("✂️ Splitting and embedding..."):
    try:
        chunks = split_text(text)
        embedding_model, index, _, chunks = embed_chunks(chunks)
        save_faiss_index(index, chunks)
        st.success("✅ Ready for questions!")
    except Exception as e:
        st.error(f"❌ Embedding error: {e}")
        st.stop()

# Log Feedback
def log_feedback(question, answer, feedback):
    try:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "feedback": feedback
        }
        os.makedirs("logs", exist_ok=True)
        with open("logs/feedback_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        st.warning(f"⚠️ Failed to log feedback: {e}")
        print(traceback.format_exc())

# 🧠 Question Form
with st.form("question_form"):
    user_question = st.text_input("Ask a question about knowledge.pdf:", key="question_input")
    submitted_question = st.form_submit_button("🚀 Send")

if submitted_question and user_question.strip():
    with st.spinner("🤖 Thinking..."):
        try:
            index, chunks = load_faiss_index()
            model_name = "deepseek/deepseek-chat-v3-0324:free"  # Hardcoded model name
            top_chunks = search_chunks(embedding_model, index, chunks, user_question)
            response = ask_question(model_name, top_chunks, user_question)

            st.session_state["last_question"] = user_question
            st.session_state["last_answer"] = response

            st.markdown("### 💬 Answer:")
            st.write(response)

            log_chat_to_history(user_question, response)

        except Exception as e:
            st.error(f"❌ Error during QA: {e}")
            print(traceback.format_exc())

# 📝 Feedback Form (Independent)
if "last_question" in st.session_state and "last_answer" in st.session_state:
    with st.form("feedback_form", clear_on_submit=True):
        st.markdown("### 📝 Feedback")
        feedback_rating = st.radio("Was this helpful?", ["👍 Yes", "👎 No"], horizontal=True)
        feedback_comment = st.text_area("Any comments?")
        submitted_feedback = st.form_submit_button("Submit Feedback")

        if submitted_feedback:
            log_feedback(
                st.session_state["last_question"],
                st.session_state["last_answer"],
                {"rating": feedback_rating, "comment": feedback_comment}
            )
            st.success("✅ Feedback submitted. Thank you!")

# Footer
st.markdown("""
    <style>
    .footer {position: fixed; left: 0; bottom: 0; width: 100%; background-color: #f0f2f6;
    color: #333; text-align: center; padding: 10px 0; font-size: 14px; border-top: 1px solid #ddd;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; z-index: 9999;}
    </style>
    <div class="footer">
        Made with ❤️ by <strong>Nikhil Panchal</strong> 🚀✨
    </div>
""", unsafe_allow_html=True)
