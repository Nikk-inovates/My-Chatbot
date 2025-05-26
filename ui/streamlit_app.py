import streamlit as st
import os
import sys
import json
from datetime import datetime
import traceback

# Add root directory to path so we can import from /src and main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Custom modules
from src.load_pdf import load_pdf_text
from src.embed_text import split_text, embed_chunks, save_faiss_index, load_faiss_index
from main import setup_deepseek, ask_question, search_chunks

# Streamlit page config
st.set_page_config(page_title="Chat with Your PDF", page_icon="📄", layout="centered")
st.title("📄 Chat with Dataset.pdf using DeepSeek + FAISS")

# PDF path
pdf_path = "data/knowledge.pdf"

if not os.path.exists(pdf_path):
    st.error("❌ PDF file not found at: data/knowledge.pdf")
    st.stop()

# Load PDF text
with st.spinner("📄 Loading and reading knowledge.pdf..."):
    try:
        text = load_pdf_text(pdf_path)
    except Exception as e:
        st.error(f"❌ Failed to load PDF: {e}")
        st.stop()

# Embed and index
with st.spinner("✂️ Splitting and embedding text..."):
    try:
        chunks = split_text(text)
        embedding_model, index, _, chunks = embed_chunks(chunks)
        save_faiss_index(index, chunks)
        st.success("✅ knowledge.pdf is ready for questions!")
    except Exception as e:
        st.error(f"❌ Embedding error: {e}")
        st.stop()

# Feedback logger
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

# User input
user_question = st.text_input("Ask a question about knowledge.pdf:")

if user_question:
    with st.spinner("🤖 Thinking..."):
        try:
            index, chunks = load_faiss_index()
            model_name = setup_deepseek()
            top_chunks = search_chunks(embedding_model, index, chunks, user_question)
            response = ask_question(model_name, top_chunks, user_question)

            st.markdown("### 💬 Answer:")
            st.write(response)

            # Feedback form
            with st.form("feedback_form"):
                st.markdown("### 📝 Feedback")
                feedback_rating = st.radio(
                    "Was this answer helpful?",
                    ["👍 Yes", "👎 No"],
                    horizontal=True
                )
                feedback_comment = st.text_area("Additional comments (optional):")
                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    log_feedback(
                        user_question,
                        response,
                        {"rating": feedback_rating, "comment": feedback_comment}
                    )
                    st.success("✅ Feedback submitted. Thank you!")

        except Exception as e:
            st.error(f"❌ Error during QA: {e}")
            print(traceback.format_exc())
