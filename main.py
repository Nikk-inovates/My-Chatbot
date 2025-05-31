from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import shutil
import logging

from src.load_pdf import load_pdf_text
from src.embed_text import split_text, embed_chunks, save_faiss_index, load_faiss_index
from src.chatbot import ask_question, search_chunks

# Load environment variables from .env file
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Constants
PDF_DIR = "data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"

# Create necessary folders
os.makedirs(PDF_DIR, exist_ok=True)

# Load embedding model globally (once)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


@app.get("/")
def root():
    return {"message": "🚀 PDF Chatbot API is running!"}


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, extract its text, chunk it, create embeddings, and save the FAISS index.
    """
    try:
        file_path = os.path.join(PDF_DIR, file.filename)
        logger.info(f"Saving uploaded file to {file_path}")
        
        # Save uploaded PDF
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF
        text = load_pdf_text(file_path)
        chunks = split_text(text)
        model, index, embeddings, chunk_list = embed_chunks(chunks)
        save_faiss_index(index, chunk_list)

        logger.info("PDF processed and index saved.")
        return {"message": "✅ PDF uploaded and processed successfully!"}
    
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        return JSONResponse(status_code=500, content={"error": f"❌ Failed to process PDF: {str(e)}"})


@app.post("/ask-question/")
async def ask_question_api(question: str = Form(...)):
    """
    Accepts a question, retrieves top-matching chunks using FAISS, and sends them to DeepSeek for an answer.
    """
    try:
        # Load FAISS index
        index, chunk_list = load_faiss_index()

        if not index or not chunk_list:
            logger.warning("No index or chunks found. Please upload a PDF first.")
            return JSONResponse(
                status_code=400,
                content={"error": "❌ No index or chunks found. Upload a PDF first."}
            )

        # Semantic search
        top_chunks = search_chunks(embedding_model, index, chunk_list, question)

        # Ask the LLM
        response = ask_question(LLM_MODEL_NAME, top_chunks, question)
        return {"answer": response}

    except Exception as e:
        logger.error(f"Failed to answer question: {e}")
        return JSONResponse(status_code=500, content={"error": f"❌ Failed to get answer: {str(e)}"})