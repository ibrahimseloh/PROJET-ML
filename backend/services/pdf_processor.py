"""Service pour traitement PDF"""
import pdfplumber
import numpy as np
from typing import List, Dict
import re

class PDFProcessor:
    """Service pour extraction et traitement PDF"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Dict[int, str]:
        """Extrait texte par page (1-indexed pour l'utilisateur)"""
        pages_text = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    # Utiliser 1-indexed (page 1, 2, 3, ...) au lieu de 0-indexed
                    pages_text[i + 1] = text if text else ""
        except Exception as e:
            print(f"Erreur extraction PDF: {e}")
        
        return pages_text
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Nettoie et normalise texte"""
        # Enlever espaces multiples
        text = re.sub(r'\s+', ' ', text)
        # Enlever caractères spéciaux problématiques
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        return text.strip()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Crée chunks avec overlap"""
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Overlap: garder fin du chunk précédent
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
