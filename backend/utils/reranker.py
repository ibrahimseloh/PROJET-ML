"""Service de reranking avec CrossEncoder"""
from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np

class RerankerService:
    """Service pour reranking de documents"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_name: str = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'):
        """Charge modèle CrossEncoder (singleton)"""
        if self._model is None:
            self._model = CrossEncoder(model_name)
        return self._model
    
    def rerank(self, query: str, texts: List[str], top_k: int = 3) -> List[Dict]:
        """Reranking des textes par pertinence"""
        if self._model is None:
            self.load_model()
        
        pairs = [[query, text] for text in texts]
        scores = self._model.predict(pairs)
        
        # Trier par score décroissant
        ranked = sorted(
            [{'text': text, 'score': score, 'index': i} 
             for i, (text, score) in enumerate(zip(texts, scores))],
            key=lambda x: x['score'],
            reverse=True
        )
        
        return ranked[:top_k]
