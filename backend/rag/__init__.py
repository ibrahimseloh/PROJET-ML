"""RAG pour documents PDF"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tempfile

from backend.services.pdf_processor import PDFProcessor
from backend.services.gemini_service import GeminiService
from backend.utils import EmbeddingService
from backend.utils.reranker import RerankerService
from backend.utils import FAISSService
from backend.prompts import PDF_QUERY_PROMPT

@dataclass
class PDFChunk:
    """Représente un chunk de PDF"""
    text: str
    page_num: int
    chunk_id: int
    embedding: Optional[np.ndarray] = None

class PDFRagPipeline:
    """Pipeline RAG complète pour PDF"""
    
    def __init__(self, gemini_service: GeminiService):
        self.gemini_service = gemini_service
        self.embedding_service = EmbeddingService()
        self.reranker_service = RerankerService()
        
        self.chunks: List[PDFChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.pdf_path: Optional[str] = None
    
    def process_pdf(self, pdf_file) -> bool:
        """Traite un fichier PDF uploadé"""
        try:
            # Sauvegarder temporairement
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(pdf_file.getbuffer())
                self.pdf_path = tmp.name
            
            print(f"📄 Traitement du PDF: {self.pdf_path}")
            
            # Extraction parallèle
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_extract = executor.submit(PDFProcessor.extract_text_from_pdf, self.pdf_path)
                future_model = executor.submit(self.embedding_service.load_model)
                
                pages_text = future_extract.result()
                _ = future_model.result()
            
            # Vérifier si du texte a été extrait
            total_chars = sum(len(text) for text in pages_text.values())
            print(f"📝 Extraction: {len(pages_text)} pages, {total_chars} caractères au total")
            
            if total_chars == 0:
                print("⚠️ Aucun texte extrait du PDF - document peut-être protégé ou corrompu")
                return False
            
            # Créer chunks
            chunk_id = 0
            for page_num, text in pages_text.items():
                cleaned = PDFProcessor.clean_text(text)
                if not cleaned or len(cleaned.strip()) < 10:
                    print(f"  ⚠️ Page {page_num}: texte insuffisant, ignorée")
                    continue
                    
                chunks = PDFProcessor.chunk_text(cleaned)
                
                for chunk_text in chunks:
                    self.chunks.append(PDFChunk(
                        text=chunk_text,
                        page_num=page_num,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
            
            print(f"📦 {len(self.chunks)} chunks créés")
            
            # Embeddings et index FAISS
            texts = [chunk.text for chunk in self.chunks]
            self.embeddings = self.embedding_service.get_embeddings(texts)
            
            for i, chunk in enumerate(self.chunks):
                chunk.embedding = self.embeddings[i]
            
            self.faiss_index = FAISSService.build_index(self.embeddings)
            
            return True
        except Exception as e:
            print(f"Erreur processing PDF: {e}")
            return False
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Récupère chunks pertinents"""
        if not self.faiss_index:
            return []
        
        query_embedding = self.embedding_service.get_embedding(query)
        distances, indices = FAISSService.search(self.faiss_index, query_embedding, k)
        
        results = []
        for idx in indices:
            chunk = self.chunks[idx]
            results.append({
                'text': chunk.text,
                'page': chunk.page_num,
                'chunk_id': chunk.chunk_id
            })
        
        return results
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """Reranking des chunks"""
        texts = [chunk['text'] for chunk in chunks]
        ranked = self.reranker_service.rerank(query, texts, top_k)
        
        # Retourner chunks réankés avec métadatas
        return [
            {
                **chunks[item['index']],
                'rerank_score': item['score']
            }
            for item in ranked
        ]
    
    def _format_clickable_citations(self, response: str, sources: List[Dict]) -> str:
        """
        Transforme les citations numérotées [1], [2], [3] en liens cliquables HTML
        Gère TOUS les formats mal formés - très agressif
        """
        import re
        
        # 1. PASSE AGRESSIVE - Remplacer TOUT ce qui ressemble à "Page N" suivi de citations
        # Pattern: quelconque texte avec "Page" + numéro + éventuellement crochets
        
        # [Page 1], [Page 2], etc.
        response = re.sub(r'\[\s*[Pp]age\s+(\d+)\s*\]', r'[\1]', response)
        
        # [Page 1. [Page 2], etc. (si mal formaté avec points/espaces)
        response = re.sub(r'\[\s*[Pp]age\s+(\d+)[\.\,\s]*\]', r'[\1]', response)
        
        # (Page 1), (page 1), etc.
        response = re.sub(r'\(\s*[Pp]age\s+(\d+)\s*\)', r'[\1]', response)
        
        # Page 1 suivi de crochets ou tirets - "Page 1]" ou "Page 1]."
        response = re.sub(r'[Pp]age\s+(\d+)\]', r'[\1]', response)
        
        # "Page 1." ou "page 1." suivi d'un numéro en crochets si présent
        response = re.sub(r'[Pp]age\s+(\d+)[\.\,\s]+(?=[\[\(])', r'[\1] ', response)
        
        # 2. PASSE COMPLÉMENTAIRE - Chercher les patterns complexes
        # "... [Page 1]." au lieu de "[1]."
        response = re.sub(r'\s+\[\s*[Pp]age\s+(\d+)\s*\][\.\,\;]*', r' [\1]', response)
        
        # 3. Transformer les citations [1], [2], [3] en liens cliquables
        def replace_citation(match):
            citation_num = int(match.group(1))
            # Vérifier que la citation correspond à une source
            if citation_num > 0 and citation_num <= len(sources):
                source = sources[citation_num - 1]
                page_num = source.get('page', 1)
                # Créer un lien cliquable avec data attributes
                return f'<a class="pdf-citation" data-page="{page_num}" data-index="{citation_num-1}" style="cursor: pointer; color: #0668e1; text-decoration: none; font-weight: 500; border-bottom: 1px dotted #0668e1;">[\n{citation_num}]</a>'
            return match.group(0)
        
        # Remplacer toutes les citations [1], [2], [3], etc.
        response = re.sub(r'\[(\d+)\]', replace_citation, response)
        return response
    
    def query(self, question: str, k: int = 5, rerank_top_k: int = 10, use_streaming: bool = False) -> Dict:
        """Requête complète RAG
        
        NOTE: Streaming est DÉSACTIVÉ pour éviter les problèmes avec Streamlit
        """
        # Retrieve
        chunks = self.retrieve(question, k=k)
        
        if not chunks:
            return {
                'question': question,
                'response': '⚠️ Aucun contenu pertinent trouvé dans le PDF.',
                'sources': [],
                'context': ''
            }
        
        # Rerank
        ranked_chunks = self.rerank(question, chunks, top_k=rerank_top_k)
        
        # Contexte - Format simplifié pour éviter que le LLM mélange avec [Page X]
        context = "\n\n".join([
            f"[{i+1}] {chunk['text']}"
            for i, chunk in enumerate(ranked_chunks)
        ])
        
        if not context.strip():
            return {
                'question': question,
                'response': '⚠️ Contexte vide - impossible de générer réponse.',
                'sources': ranked_chunks,
                'context': context
            }
        
        # Prompt - Utiliser le prompt centralisé
        prompt = PDF_QUERY_PROMPT.format(context=context, question=question)
        
        # Générer réponse - JAMAIS de streaming
        try:
            response = self.gemini_service.generate(prompt)
            print(f"\n🔍 [DEBUG] Réponse brute du LLM (premiers 500 chars):\n{response[:500]}\n")
            # Transformer les citations pour les rendre cliquables
            response = self._format_clickable_citations(response, ranked_chunks)
            print(f"🔍 [DEBUG] Réponse après formatage (premiers 500 chars):\n{response[:500]}\n")
        except Exception as e:
            response = f"❌ Erreur: {str(e)}"
        
        return {
            'question': question,
            'response': response,
            'sources': ranked_chunks,
            'context': context
        }
        
        return {
            'question': question,
            'response': response,
            'sources': ranked_chunks,
            'context': context
        }
