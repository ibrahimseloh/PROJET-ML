"""Utilitaires pour embeddings et FAISS"""
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List

class EmbeddingService:
    """Service pour embeddings avec SentenceTransformer"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Charge modèle SentenceTransformer (singleton)"""
        if self._model is None:
            self._model = SentenceTransformer(model_name)
        return self._model
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Génère embeddings pour une liste de textes"""
        if self._model is None:
            self.load_model()
        
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return np.array(embeddings).astype('float32')
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Génère embedding pour un texte unique"""
        if self._model is None:
            self.load_model()
        
        embedding = self._model.encode([text], show_progress_bar=False)
        return np.array(embedding).astype('float32')[0]

class FAISSService:
    """Service pour construction et requête FAISS"""
    
    @staticmethod
    def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Crée index FAISS"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    
    @staticmethod
    def search(index: faiss.IndexFlatL2, query_embedding: np.ndarray, k: int = 5) -> tuple:
        """Cherche k voisins les plus proches"""
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = index.search(query_embedding, k)
        return distances[0], indices[0]
