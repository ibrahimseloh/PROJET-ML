"""Service pour traitement PDF"""
import pdfplumber
import pytesseract
import numpy as np
from typing import List, Dict
from pdf2image import convert_from_path
from PIL import Image
import re

class PDFProcessor:
    """Service pour extraction et traitement PDF avec support OCR"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> Dict[int, str]:
        """Extrait texte par page avec OCR fallback (1-indexed pour l'utilisateur)"""
        pages_text = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Essayer extraction directe d'abord
                    text = page.extract_text()
                    
                    # Si peu de texte, utiliser OCR
                    if not text or len(text.strip()) < 50:
                        print(f"Page {i+1}: Peu de texte détecté, application OCR...")
                        text = PDFProcessor._extract_with_ocr(pdf_path, i)
                    
                    pages_text[i + 1] = text if text else ""
        except Exception as e:
            print(f"Erreur extraction PDF: {e}")
        
        return pages_text
    
    @staticmethod
    def _extract_with_ocr(pdf_path: str, page_index: int) -> str:
        """Extrait texte via OCR Tesseract"""
        try:
            # Convertir PDF en images (DPI dynamique basé sur contenu)
            images = convert_from_path(
                pdf_path,
                first_page=page_index + 1,
                last_page=page_index + 1,
                dpi=150
            )
            
            if images:
                image = images[0]
                
                # Augmenter contraste pour meilleure reconnaissance
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                
                # Appliquer OCR Tesseract
                text = pytesseract.image_to_string(image, lang='fra+eng')
                return text if text else ""
        except Exception as e:
            print(f"Erreur OCR page {page_index+1}: {e}")
        
        return ""
    
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
