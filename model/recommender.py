# model/recommender.py
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class QuestionRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.questions = self.df['question'].dropna().tolist()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.embeddings = None
        self._build_index()

    def _build_index(self):
        self.embeddings = self.model.encode(self.questions).astype("float32")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def recommend(self, user_input, top_k=5):
        if not user_input.strip():
            return []

        user_vec = self.model.encode([user_input]).astype("float32")
        distances, indices = self.index.search(user_vec, top_k)
        return [self.questions[i] for i in indices[0] if i < len(self.questions)]
