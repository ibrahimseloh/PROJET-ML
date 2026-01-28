"""Service pour traitement PDF avec support OCR"""
import pdfplumber
import numpy as np
from typing import List, Dict, Optional, Tuple
import re
import os
import tempfile

# Import conditionnel pour OCR
try:
    from pdf2image import convert_from_path, convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR non disponible - installez pytesseract et pdf2image")


class PDFProcessor:
    """
    Service pour extraction et traitement PDF
    Supporte les PDFs natifs ET scannés (via OCR)
    """
    
    # Seuil minimum de caractères par page pour considérer qu'il y a du texte
    MIN_TEXT_THRESHOLD = 50
    
    # Langues OCR supportées
    OCR_LANGUAGES = "fra+eng"  # Français + Anglais
    
    @classmethod
    def extract_text_from_pdf(cls, pdf_path: str, force_ocr: bool = False) -> Dict[int, str]:
        """
        Extrait texte par page (1-indexed pour l'utilisateur)
        Utilise OCR automatiquement si le PDF est scanné
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            force_ocr: Force l'utilisation de l'OCR même si du texte est détecté
            
        Returns:
            Dict[page_number, text]: Texte extrait par page
        """
        pages_text = {}
        needs_ocr = []
        
        try:
            # Première passe : extraction native avec pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1  # 1-indexed
                    text = page.extract_text() or ""
                    
                    # Vérifier si la page a suffisamment de texte
                    if force_ocr or len(text.strip()) < cls.MIN_TEXT_THRESHOLD:
                        needs_ocr.append(page_num)
                        pages_text[page_num] = ""  # Sera rempli par OCR
                    else:
                        pages_text[page_num] = text
            
            # Deuxième passe : OCR pour les pages sans texte
            if needs_ocr and OCR_AVAILABLE:
                print(f"📷 OCR requis pour {len(needs_ocr)} page(s) sur {total_pages}")
                ocr_results = cls._apply_ocr(pdf_path, needs_ocr)
                pages_text.update(ocr_results)
            elif needs_ocr and not OCR_AVAILABLE:
                print(f"⚠️ {len(needs_ocr)} page(s) nécessitent OCR mais il n'est pas installé")
                
        except Exception as e:
            print(f"Erreur extraction PDF: {e}")
        
        return pages_text
    
    @classmethod
    def _apply_ocr(cls, pdf_path: str, pages: List[int]) -> Dict[int, str]:
        """
        Applique l'OCR sur les pages spécifiées
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            pages: Liste des numéros de pages à traiter (1-indexed)
            
        Returns:
            Dict[page_number, text]: Texte OCR par page
        """
        ocr_results = {}
        
        if not OCR_AVAILABLE:
            print("❌ OCR non disponible - pytesseract ou pdf2image non installé")
            return ocr_results
        
        try:
            print(f"🔍 Démarrage OCR pour {len(pages)} page(s)...")
            
            # Convertir les pages PDF en images
            # first_page et last_page sont 1-indexed pour pdf2image
            images = convert_from_path(
                pdf_path,
                dpi=300,  # Haute résolution pour meilleure OCR
                first_page=min(pages),
                last_page=max(pages),
                thread_count=2  # Parallélisation
            )
            
            print(f"📷 {len(images)} image(s) générée(s)")
            
            # Mapper les images aux numéros de pages
            page_range = list(range(min(pages), max(pages) + 1))
            
            for img, page_num in zip(images, page_range):
                if page_num in pages:
                    try:
                        # Prétraitement de l'image pour améliorer l'OCR
                        processed_img = cls._preprocess_image(img)
                        
                        # Appliquer OCR avec config optimisée
                        text = pytesseract.image_to_string(
                            processed_img,
                            lang=cls.OCR_LANGUAGES,
                            config='--psm 3 --oem 3'  # Mode page entière
                        )
                        
                        # Si peu de résultats, essayer sans prétraitement
                        if len(text.strip()) < 20:
                            print(f"  ⚠️ Page {page_num}: peu de texte, réessai sans prétraitement...")
                            text = pytesseract.image_to_string(
                                img,
                                lang=cls.OCR_LANGUAGES,
                                config='--psm 3 --oem 3'
                            )
                        
                        ocr_results[page_num] = text
                        print(f"  ✓ Page {page_num} OCR: {len(text)} caractères")
                        
                    except Exception as page_error:
                        print(f"  ❌ Page {page_num} erreur: {page_error}")
                        ocr_results[page_num] = ""
                    
        except Exception as e:
            import traceback
            print(f"❌ Erreur OCR globale: {e}")
            traceback.print_exc()
        
        return ocr_results
    
    @staticmethod
    def _preprocess_image(image: 'Image.Image') -> 'Image.Image':
        """
        Prétraite l'image pour améliorer la qualité de l'OCR
        Prétraitement léger pour ne pas détruire le texte
        
        Args:
            image: Image PIL
            
        Returns:
            Image prétraitée
        """
        try:
            from PIL import ImageFilter, ImageEnhance, ImageOps
            
            # Convertir en RGB si nécessaire puis en niveaux de gris
            if image.mode == 'RGBA':
                # Créer un fond blanc
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            
            img = image.convert('L')
            
            # Redimensionner si trop petit (améliore OCR)
            if img.width < 1000:
                scale = 1000 / img.width
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Améliorer le contraste de façon adaptative
            img = ImageOps.autocontrast(img, cutoff=2)
            
            # Légère amélioration de la netteté
            img = img.filter(ImageFilter.SHARPEN)
            
            # PAS de binarisation agressive - laisser Tesseract gérer
            
            return img
            
        except Exception as e:
            print(f"⚠️ Prétraitement image échoué: {e}")
            # En cas d'erreur, retourner l'image originale
            return image
    
    @classmethod
    def extract_text_from_bytes(cls, pdf_bytes: bytes, force_ocr: bool = False) -> Dict[int, str]:
        """
        Extrait le texte d'un PDF à partir de bytes
        
        Args:
            pdf_bytes: Contenu du PDF en bytes
            force_ocr: Force l'utilisation de l'OCR
            
        Returns:
            Dict[page_number, text]: Texte extrait par page
        """
        pages_text = {}
        needs_ocr = []
        
        try:
            # Sauvegarder temporairement pour pdf2image si nécessaire
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            
            try:
                # Extraction avec pdfplumber depuis bytes
                with pdfplumber.open(tmp_path) as pdf:
                    total_pages = len(pdf.pages)
                    
                    for i, page in enumerate(pdf.pages):
                        page_num = i + 1
                        text = page.extract_text() or ""
                        
                        if force_ocr or len(text.strip()) < cls.MIN_TEXT_THRESHOLD:
                            needs_ocr.append(page_num)
                            pages_text[page_num] = ""
                        else:
                            pages_text[page_num] = text
                
                # OCR si nécessaire
                if needs_ocr and OCR_AVAILABLE:
                    print(f"📷 OCR requis pour {len(needs_ocr)} page(s) sur {total_pages}")
                    ocr_results = cls._apply_ocr(tmp_path, needs_ocr)
                    pages_text.update(ocr_results)
                    
            finally:
                # Nettoyer le fichier temporaire
                os.unlink(tmp_path)
                
        except Exception as e:
            print(f"Erreur extraction PDF bytes: {e}")
        
        return pages_text
    
    @staticmethod
    def is_ocr_available() -> bool:
        """Vérifie si l'OCR est disponible"""
        return OCR_AVAILABLE
    
    @staticmethod
    def get_ocr_languages() -> List[str]:
        """Retourne la liste des langues OCR installées"""
        if not OCR_AVAILABLE:
            return []
        try:
            langs = pytesseract.get_languages()
            return [l for l in langs if l != 'osd']
        except Exception:
            return []
    
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
