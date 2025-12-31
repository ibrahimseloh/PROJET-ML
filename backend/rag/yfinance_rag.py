"""RAG pour données Yahoo Finance"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backend.services.gemini_service import GeminiService
from backend.services.yfinance_service import YFinanceService
from backend.utils import EmbeddingService
from backend.utils.reranker import RerankerService
from backend.utils import FAISSService
from backend.prompts import YFINANCE_QUERY_PROMPT

@dataclass
class MarketChunk:
    """Représente un chunk de données market"""
    text: str
    ticker: str
    chunk_id: int
    chunk_type: str  # 'daily' ou 'monthly'
    embedding: Optional[np.ndarray] = None

class YFinanceRagAssistant:
    """RAG pour données Yahoo Finance"""
    
    def __init__(self, gemini_service: GeminiService, tickers: List[str] = None):
        self.gemini_service = gemini_service
        self.embedding_service = EmbeddingService()
        self.reranker_service = RerankerService()
        
        self.tickers = tickers or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.data: Dict[str, pd.DataFrame] = {}
        self.chunks: List[MarketChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
    
    def fetch_and_process_data(self, period_months: int = 20) -> bool:
        """Récupère et traite données YFinance"""
        try:
            # Fetch parallèle
            self.data = YFinanceService.parallel_fetch_tickers(self.tickers, period_months)
            
            # Load embedding model
            self.embedding_service.load_model()
            
            # Créer chunks
            chunk_id = 0
            for ticker, ticker_data in self.data.items():
                df = ticker_data['dataframe']
                
                # Chunks quotidiens (par semaine)
                # Gérer colonnes dynamiquement
                agg_dict = {}
                if 'Open' in df.columns:
                    agg_dict['Open'] = 'first'
                if 'High' in df.columns:
                    agg_dict['High'] = 'max'
                if 'Low' in df.columns:
                    agg_dict['Low'] = 'min'
                if 'Close' in df.columns:
                    agg_dict['Close'] = 'last'
                if 'Volume' in df.columns:
                    agg_dict['Volume'] = 'sum'
                
                if not agg_dict:
                    print(f"Colonnes disponibles pour {ticker}: {df.columns.tolist()}")
                    continue
                
                weekly_chunks = df.resample('W').agg(agg_dict)
                
                for date, row in weekly_chunks.iterrows():
                    text = f"{ticker} semaine {date.strftime('%Y-%m-%d')}: "
                    text += f"Open ${row['Open']:.2f}, High ${row['High']:.2f}, "
                    text += f"Low ${row['Low']:.2f}, Close ${row['Close']:.2f}, "
                    text += f"Volume {int(row['Volume']):,}"
                    
                    self.chunks.append(MarketChunk(
                        text=text,
                        ticker=ticker,
                        chunk_id=chunk_id,
                        chunk_type='weekly'
                    ))
                    chunk_id += 1
                
                # Chunks mensuels (résumé)
                monthly = df.resample('M').agg(agg_dict)
                
                for date, row in monthly.iterrows():
                    variation = ((row['Close'] - row['Open']) / row['Open'] * 100)
                    text = f"{ticker} {date.strftime('%B %Y')}: "
                    text += f"${row['Open']:.2f}→${row['Close']:.2f} "
                    text += f"({variation:+.2f}%), Volume {int(row['Volume']):,}"
                    
                    self.chunks.append(MarketChunk(
                        text=text,
                        ticker=ticker,
                        chunk_id=chunk_id,
                        chunk_type='monthly'
                    ))
                    chunk_id += 1
            
            # Embeddings
            texts = [chunk.text for chunk in self.chunks]
            self.embeddings = self.embedding_service.get_embeddings(texts)
            
            for i, chunk in enumerate(self.chunks):
                chunk.embedding = self.embeddings[i]
            
            self.faiss_index = FAISSService.build_index(self.embeddings)
            
            return True
        except Exception as e:
            print(f"Erreur fetch/process data: {e}")
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
                'ticker': chunk.ticker,
                'chunk_type': chunk.chunk_type,
                'chunk_id': chunk.chunk_id
            })
        
        return results
    
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """Reranking"""
        texts = [chunk['text'] for chunk in chunks]
        ranked = self.reranker_service.rerank(query, texts, top_k)
        
        return [
            {
                **chunks[item['index']],
                'rerank_score': item['score']
            }
            for item in ranked
        ]
    
    def query(self, question: str, tickers: Optional[List[str]] = None, k: int = 5, rerank_top_k: int = 4, use_streaming: bool = False) -> Dict:
        """Requête sur données YFinance
        
        NOTE: Streaming est DÉSACTIVÉ pour éviter les problèmes avec Streamlit
        """
        # DEBUG
        print(f"\n🔍 [QUERY] START - Question: {question}")
        print(f"🔍 [QUERY] FAISS index exists: {self.faiss_index is not None}")
        print(f"🔍 [QUERY] Total chunks available: {len(self.chunks)}")
        
        # Vérifier que l'index existe
        if not self.faiss_index:
            print(f"❌ [QUERY] No FAISS index!")
            return {
                'question': question,
                'response': '❌ Erreur: Pas de données indexées. Chargez les données d\'abord.',
                'sources': [],
                'context': '',
                'tickers': tickers or self.tickers
            }
        
        # Filtrer chunks par tickers si spécifié
        chunks = self.retrieve(question, k=k)
        print(f"🔍 [QUERY] Retrieved {len(chunks)} chunks")
        
        if tickers:
            chunks = [c for c in chunks if c['ticker'] in tickers]
            print(f"🔍 [QUERY] After filtering by tickers: {len(chunks)}")
        
        # Si pas de chunks, retourner message d'erreur
        if not chunks:
            print(f"❌ [QUERY] No chunks found!")
            return {
                'question': question,
                'response': '⚠️ Aucune donnée pertinente trouvée pour votre question.',
                'sources': [],
                'context': '',
                'tickers': tickers or self.tickers
            }
        
        # Rerank
        ranked_chunks = self.rerank(question, chunks, top_k=rerank_top_k)
        print(f"🔍 [QUERY] Ranked {len(ranked_chunks)} chunks")
        
        # Pour YFinance: utiliser UNIQUEMENT le chunk le plus récent (le premier)
        if ranked_chunks:
            most_recent_chunk = ranked_chunks[0]  # Le top-1 du ranking
            context = f"{most_recent_chunk['text']}"
            sources = ranked_chunks  # Garder tous les chunks pour affichage des sources
        else:
            context = ""
            sources = []
        
        print(f"🔍 [QUERY] Using only most recent chunk for context")
        print(f"🔍 [QUERY] Context length: {len(context)} chars")
        
        # Si contexte vide, retourner erreur
        if not context.strip():
            print(f"❌ [QUERY] Empty context!")
            return {
                'question': question,
                'response': '⚠️ Contexte vide - impossible de générer réponse.',
                'sources': sources,
                'context': context,
                'tickers': tickers or self.tickers
            }
        
        # Prompt - Utiliser le prompt centralisé
        prompt = YFINANCE_QUERY_PROMPT.format(context=context, question=question)
        print(f"🔍 [QUERY] Prompt length: {len(prompt)} chars")
        print(f"🔍 [QUERY] Calling LLM.generate() (NO STREAMING)...", flush=True)
        
        try:
            # JAMAIS de streaming en Streamlit - utiliser generate() uniquement
            response = self.gemini_service.generate(prompt)
            print(f"🔍 [QUERY] LLM response received: {len(response)} chars", flush=True)
        except Exception as e:
            print(f"❌ [QUERY] LLM Error: {e}", flush=True)
            response = f"❌ Erreur LLM: {str(e)}"
        
        return {
            'question': question,
            'response': response,
            'sources': ranked_chunks,
            'context': context,
            'tickers': tickers or self.tickers
        }
