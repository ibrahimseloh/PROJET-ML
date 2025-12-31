"""Service pour récupération données Yahoo Finance en parallèle"""
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import pandas as pd
from datetime import datetime, timedelta

class YFinanceService:
    """Service pour fetch données Yahoo Finance"""
    
    @staticmethod
    def fetch_ticker_data(ticker: str, start_date: datetime, end_date: datetime) -> Dict:
        """Récupère données pour 1 ticker"""
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            # Handle multi-index columns from yfinance
            # yfinance returns MultiIndex like: [('Close', 'AAPL'), ('High', 'AAPL'), ...]
            if isinstance(df.columns, pd.MultiIndex):
                # Get all unique price types in level 0
                price_types = df.columns.get_level_values(0).unique().tolist()
                
                # Select columns for this ticker and flatten
                new_df = pd.DataFrame(index=df.index)
                for price_type in price_types:
                    new_df[price_type] = df[(price_type, ticker)]
                df = new_df
            
            info = yf.Ticker(ticker).info
            
            return {
                'ticker': ticker,
                'data': df,
                'info': info,
                'status': 'success'
            }
        except Exception as e:
            print(f"Erreur fetch {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'data': None,
                'info': None,
                'status': 'error',
                'error': str(e)
            }
    
    @staticmethod
    def parallel_fetch_tickers(tickers: List[str], period_months: int = 20) -> Dict[str, pd.DataFrame]:
        """Fetch tous les tickers - SÉQUENTIELLEMENT pour éviter les bugs yfinance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_months * 30)
        
        data = {}
        
        # NOTE: On utilise une approche séquentielle au lieu de ThreadPoolExecutor
        # car yfinance a des problèmes avec la concurrence
        for ticker in tickers:
            result = YFinanceService.fetch_ticker_data(ticker, start_date, end_date)
            
            if result['status'] == 'success':
                data[ticker] = {
                    'dataframe': result['data'],
                    'info': result['info']
                }
            else:
                print(f"⚠️ Erreur pour {ticker}: {result.get('error', 'Unknown error')}")
        
        return data
