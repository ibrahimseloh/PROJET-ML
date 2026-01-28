"""Agents pour visualisation interactive des données"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Optional

class InteractiveChartAgent:
    """Agent pour créer graphiques interactifs avec Plotly"""
    
    def plot_single_ticker(self, df: pd.DataFrame, ticker: str) -> go.Figure:
        """Graphique interactif pour un ticker
        
        Args:
            df: DataFrame avec colonnes Open, High, Low, Close, Volume
            ticker: Nom du ticker
        """
        if df is None or df.empty:
            return go.Figure().add_annotation(text=f"Pas de données pour {ticker}")
        
        # Vérifier les colonnes requises
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            return go.Figure().add_annotation(text=f"Colonnes manquantes pour {ticker}")
        
        # Créer figure avec subplots (Prix + Volume)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"Évolution du Prix - {ticker}", "Volume")
        )
        
        # Candlestick pour les prix
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Volume
        if 'Volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name='Volume',
                    marker=dict(color='#F18F01'),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Layout
        fig.update_layout(
            title=f"{ticker} - Données historiques",
            height=600,
            hovermode='x unified',
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def plot_multiple_tickers(self, data_dict: Dict[str, Dict], tickers: List[str]) -> go.Figure:
        """Graphique comparatif pour plusieurs tickers
        
        Args:
            data_dict: Dict avec structure {ticker: {'dataframe': df, 'info': info}}
            tickers: Liste des tickers à afficher
        """
        fig = go.Figure()
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62828']
        
        for i, ticker in enumerate(tickers):
            if ticker not in data_dict:
                continue
            
            # Handle dict with 'dataframe' key
            ticker_data = data_dict[ticker]
            if isinstance(ticker_data, dict):
                df = ticker_data.get('dataframe')
            else:
                df = ticker_data
            
            if df is None or df.empty:
                continue
            
            if 'Close' not in df.columns:
                continue
            
            # Normaliser prix (base 100)
            first_price = df['Close'].iloc[0]
            normalized = (df['Close'] / first_price) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=normalized,
                    name=ticker,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f"{ticker}<br>Date: %{{x|%Y-%m-%d}}<br>Prix normalisé: %{{y:.2f}}"
                )
            )
        
        fig.update_layout(
            title="Comparaison Multi-Tickers (Prix normalisé base 100)",
            xaxis_title="Date",
            yaxis_title="Prix (Base 100)",
            height=500,
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_all_tickers(self, data_dict: Dict[str, Dict]) -> go.Figure:
        """Graphique avec tous les tickers disponibles"""
        return self.plot_multiple_tickers(data_dict, list(data_dict.keys()))
