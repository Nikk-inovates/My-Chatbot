import os
import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("❌ OPENROUTER_API_KEY not found in .env file.")

MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

DEBUG = False  # Set to True to debug prompt

def setup_deepseek():
    """Returns the DeepSeek model name."""
    return MODEL_NAME

def ask_question(model_name, context_chunks, user_question):
    """
    Uses DeepSeek API to generate a response from context + question.

    Args:
        model_name (str): The LLM name.
        context_chunks (list): Relevant text chunks.
        user_question (str): Question from the user.

    Returns:
        str: AI's answer or error.
    """
    if not context_chunks:
        return "⚠️ No context available to answer the question."
    
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful AI assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}"
    )

    if DEBUG:
        print("Prompt to DeepSeek:\n", prompt)

    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers based on given context."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=HEADERS,
            json=payload
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ DeepSeek API error {response.status_code}: {response.text}"

    except Exception as e:
        return f"❌ Unexpected error: {e}"

def search_chunks(embedding_model, index, chunks, query, top_k=3):
    """
    Uses FAISS to retrieve top-k chunks relevant to the query.

    Args:
        embedding_model: SentenceTransformer model.
        index: FAISS index object.
        chunks (list): Text chunks from PDF.
        query (str): User's question.
        top_k (int): Number of top results.

    Returns:
        list: Top relevant text chunks.
    """
    if not query.strip():
        return []

    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        D, I = index.search(query_embedding, top_k)
        top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        return top_chunks

    except faiss.FaissException as faiss_err:
        print(f"❌ FAISS error during search: {faiss_err}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error in search_chunks: {e}")
        return []
