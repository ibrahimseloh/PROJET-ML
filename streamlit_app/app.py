"""Astrali - AI Financial Assistant RAG - ChatGPT-like UI"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import io
from typing import Dict
import tempfile
import html
import markdown as md

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import GeminiConfig
from backend.rag import PDFRagPipeline
from backend.rag.yfinance_rag import YFinanceRagAssistant
from backend.services.gemini_service import GeminiService
import re
import google.generativeai as genai
import time

# ===== API VALIDATION FUNCTION =====
def validate_api_key(api_key: str) -> Dict[str, any]:
    """
    Validate Gemini API key with comprehensive checks.
    Returns: {'valid': bool, 'message': str, 'service': GeminiService or None}
    """
    validation_result = {
        'valid': False,
        'message': '',
        'service': None,
        'error_type': None
    }
    
    # 1. Check format
    if not api_key or len(api_key.strip()) == 0:
        validation_result['message'] = "La clé API ne peut pas être vide"
        validation_result['error_type'] = 'empty'
        return validation_result
    
    api_key = api_key.strip()
    
    # 2. Check pattern (basic validation - Gemini keys start with 'AIza')
    if not re.match(r'^AIza[0-9A-Za-z\-_]{35}$', api_key):
        validation_result['message'] = "Format de clé API invalide. Une clé Gemini commence par 'AIza'"
        validation_result['error_type'] = 'format'
        return validation_result
    
    # 3. Try to initialize and test the API
    try:
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Test with a simple request
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(
            "Respond with exactly: TEST_OK",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=10,
                temperature=0.1
            ),
            request_options={"timeout": 10}
        )
        
        if response and response.text:
            # API is valid and responding
            service = GeminiService(api_key=api_key)
            validation_result['valid'] = True
            validation_result['message'] = "✅ Clé API valide et fonctionnelle"
            validation_result['service'] = service
            return validation_result
        else:
            validation_result['message'] = "La réponse de l'API est vide. Vérifiez votre clé."
            validation_result['error_type'] = 'empty_response'
            return validation_result
            
    except genai.types.StopCandidateException:
        validation_result['message'] = "La réponse de l'API a été filtrée. Vérifiez les permissions."
        validation_result['error_type'] = 'filtered'
        return validation_result
    except genai.types.APIError as e:
        error_msg = str(e)
        if '401' in error_msg or 'unauthorized' in error_msg.lower():
            validation_result['message'] = "Clé API invalide ou expirée (401 Unauthorized)"
            validation_result['error_type'] = 'unauthorized'
        elif '403' in error_msg or 'forbidden' in error_msg.lower():
            validation_result['message'] = "Accès refusé (403). Vérifiez les permissions de la clé."
            validation_result['error_type'] = 'forbidden'
        elif '429' in error_msg or 'quota' in error_msg.lower():
            validation_result['message'] = "Quota API dépassé. Attendez avant de réessayer."
            validation_result['error_type'] = 'quota'
        else:
            validation_result['message'] = f"Erreur API: {error_msg[:100]}"
            validation_result['error_type'] = 'api_error'
        return validation_result
    except Exception as e:
        error_msg = str(e)
        if 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
            validation_result['message'] = "Erreur de connexion. Vérifiez votre connexion internet."
            validation_result['error_type'] = 'connection'
        else:
            validation_result['message'] = f"Erreur: {error_msg[:80]}"
            validation_result['error_type'] = 'unknown'
        return validation_result

# ===== MARKDOWN TO HTML CONVERSION =====
def markdown_to_html(markdown_text):
    """Convert markdown text to formatted HTML with CSS classes"""
    try:
        # Convert markdown to HTML
        html_content = md.markdown(markdown_text, extensions=['tables', 'codehilite', 'fenced_code'])
        # Add markdown-content class wrapper
        return f'<div class="markdown-content">{html_content}</div>'
    except Exception as e:
        # Fallback to escaped text if conversion fails
        return f'<div class="markdown-content"><p>{html.escape(markdown_text)}</p></div>'

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Astrali - Financial AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== MODERN STYLES (SLEEK & PROFESSIONAL) =====
st.markdown("""
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    
    body {
        background: linear-gradient(135deg, #f5f7ff 0%, #efe7ff 50%, #fff5f7 100%);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        min-height: 100vh;
    }
    
    .main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1.5rem;
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fc 100%);
        border-radius: 1.2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.08);
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(135deg, #f8f9fa 0%, #eff2f7 100%);
        border: 1px solid #e9ecef;
        border-radius: 0.8rem;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.08);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.12);
        transform: translateY(-2px);
    }
    
    /* Input area */
    .input-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        border: 2px solid #e9ecef;
        margin-top: 1rem;
        transition: all 0.3s;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .input-container:focus-within {
        border-color: #667eea;
        background: linear-gradient(135deg, #ffffff 0%, #f5f7ff 100%);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    /* Chat container - improved */
    .chat-box {
        height: 480px;
        max-height: 480px;
        overflow-y: auto;
        padding: 1rem;
        background: linear-gradient(180deg, #fafbfc 0%, #f0f2f6 100%);
        border: 1px solid #e9ecef;
        border-radius: 0.8rem;
        display: flex;
        flex-direction: column;
        gap: 0.8rem;
        scroll-behavior: smooth;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.03);
    }
    
    .chat-box::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-box::-webkit-scrollbar-track {
        background: linear-gradient(180deg, #f1f3f5 0%, #e8ecf1 100%);
        border-radius: 10px;
    }
    
    .chat-box::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: background 0.3s;
        box-shadow: inset 0 0 6px rgba(0,0,0,0.1);
    }
    
    .chat-box::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8, #6b3fa0);
    }
    
    /* Message styling */
    .msg-user {
        display: flex;
        justify-content: flex-end;
        animation: slideInRight 0.3s ease;
    }
    
    .msg-assistant {
        display: flex;
        justify-content: flex-start;
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Buttons */
    .btn-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.6rem 1.2rem;
        border: none;
        border-radius: 0.6rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .btn-primary:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    .btn-primary:active {
        transform: translateY(-1px);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 3px solid #e9ecef;
        background: linear-gradient(90deg, #f5f7ff 0%, transparent 100%);
        padding-left: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-left: 4px solid #667eea;
        padding: 1.2rem;
        border-radius: 0.6rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }
    
    .chat-responses-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-responses-container::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Chat input area - fixed at bottom */
    .chat-input-area {
        margin-top: 0.75rem;
        padding-top: 0.5rem;
        border-top: 1px solid #f0f0f0;
        background: linear-gradient(180deg, transparent 0%, #f5f7ff 100%);
    }
    
    /* Markdown content formatting in chat */
    .markdown-content h1, .markdown-content h2, .markdown-content h3, 
    .markdown-content h4, .markdown-content h5, .markdown-content h6 {
        margin: 1rem 0 0.5rem 0;
        font-weight: 700;
        color: #1a1a1a;
    }
    .markdown-content h1 { font-size: 1.5rem; color: #667eea; }
    .markdown-content h2 { font-size: 1.25rem; color: #764ba2; }
    .markdown-content h3 { font-size: 1.1rem; color: #667eea; }
    
    .markdown-content ul, .markdown-content ol {
        margin: 0.5rem 0 0.5rem 1.5rem;
        padding-left: 0;
    }
    .markdown-content li {
        margin-bottom: 0.3rem;
        color: #333;
    }
    
    .markdown-content code {
        background: linear-gradient(135deg, #e8eef7 0%, #f5e8f7 100%);
        color: #764ba2;
        padding: 0.3rem 0.6rem;
        border-radius: 0.4rem;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 0.9em;
        border: 1px solid #e0d5ec;
    }
    
    .markdown-content pre {
        background: linear-gradient(135deg, #f5f5f5 0%, #eff1f5 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0.6rem;
        overflow-x: auto;
        margin: 0.8rem 0;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.04);
    }
    
    .markdown-content strong, .markdown-content b {
        font-weight: 700;
        color: #667eea;
    }
    
    .markdown-content em, .markdown-content i {
        font-style: italic;
        color: #764ba2;
    }
    
    .markdown-content table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.8rem 0;
        background: white;
        border-radius: 0.6rem;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .markdown-content th, .markdown-content td {
        border: 1px solid #e8ecf4;
        padding: 0.8rem;
        text-align: left;
    }
    .markdown-content th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    .markdown-content tr:nth-child(even) {
        background: #f8f9fc;
    }
    
    .markdown-content blockquote {
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 0.8rem 0;
        color: #555;
        font-style: italic;
        background: linear-gradient(90deg, #f5f7ff 0%, transparent 100%);
        padding: 0.8rem 1rem;
        border-radius: 0.4rem;
    }
    
    .markdown-content a {
        color: #667eea;
        text-decoration: none;
        transition: all 0.2s;
        font-weight: 600;
    }
    .markdown-content a:hover {
        color: #764ba2;
        text-decoration: underline;
    }
    
    .markdown-content p {
        margin: 0.6rem 0;
        line-height: 1.7;
        color: #444;
    }
    
    .markdown-content hr {
        border: none;
        border-top: 2px solid #667eea;
        margin: 1.2rem 0;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)

# ===== SESSION INITIALIZATION =====
if 'pdf_rag' not in st.session_state:
    st.session_state.pdf_rag = None
if 'pdf_file' not in st.session_state:
    st.session_state.pdf_file = None
if 'pdf_file_name' not in st.session_state:
    st.session_state.pdf_file_name = None
if 'pdf_chat_history' not in st.session_state:
    st.session_state.pdf_chat_history = []
if 'current_pdf_page' not in st.session_state:
    st.session_state.current_pdf_page = 1
if 'pdf_zoom_level' not in st.session_state:
    st.session_state.pdf_zoom_level = 50
if 'pdf_viewer_active' not in st.session_state:
    st.session_state.pdf_viewer_active = False
if 'pdf_viewer_visible' not in st.session_state:
    st.session_state.pdf_viewer_visible = True
if 'temp_pdf_path' not in st.session_state:
    st.session_state.temp_pdf_path = None

if 'yfinance_rag' not in st.session_state:
    st.session_state.yfinance_rag = None
if 'yfinance_data_loaded' not in st.session_state:
    st.session_state.yfinance_data_loaded = False
if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ['AAPL']
if 'gemini_service' not in st.session_state:
    st.session_state.gemini_service = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key_submitted' not in st.session_state:
    st.session_state.api_key_submitted = False
if 'show_api_help' not in st.session_state:
    st.session_state.show_api_help = False
if 'api_validation_cache' not in st.session_state:
    st.session_state.api_validation_cache = {}

# ===== LANDING PAGE (avant l'API) =====
if not st.session_state.gemini_service:
    # Page de présentation
    st.set_page_config(page_title="Astrali - Assistant Financier IA", layout="centered")
    
    # Header avec gradient
    st.markdown("""
    <style>
        .landing-header {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            margin-bottom: 30px;
        }
        .landing-header h1 {
            color: white;
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .landing-header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.2em;
            margin-top: 10px;
        }
        .feature-box {
            background: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }
        .feature-box h3 {
            margin-top: 0;
            color: #667eea;
        }
        .authors {
            background: #fff3cd;
            padding: 15px;
            border-radius: 10px;
            margin-top: 30px;
            text-align: center;
        }
        .authors h4 {
            margin-top: 0;
            color: #856404;
        }
    </style>
    
    <div class="landing-header">
        <h1>✨ Astrali</h1>
        <p>Assistant Financier Intelligent avec Analyse IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Description
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Qu'est-ce qu'Astrali?
        
        **Astrali** est un assistant financier intelligent alimenté par l'IA Google Gemini.
        Il combine la puissance de l'analyse de documents PDF avec l'accès aux données financières 
        en temps réel pour vous offrir une expérience d'analyse complète.
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Capacités Principales
        
        ✅ **Analyse PDF** - Extrayez des informations de documents financiers
        
        ✅ **Données YFinance** - Accédez aux prix d'actions en temps réel
        
        ✅ **Chat IA** - Posez des questions et obtenez des réponses intelligentes
        
        ✅ **Sources Cliquables** - Naviguez directement vers les sources
        """)
    
    st.markdown("---")
    
    # Fonctionnement
    st.markdown("### 📚 Comment ça marche?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>1️⃣ Connectez l'API</h3>
            <p>Entrez votre clé API Google Gemini pour démarrer. C'est gratuit et sécurisé.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>2️⃣ Choisissez un Mode</h3>
            <p><strong>PDF:</strong> Upload des documents financiers<br>
            <strong>YFinance:</strong> Analyse des données d'actions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>3️⃣ Posez des Questions</h3>
            <p>Interagissez avec Astrali pour obtenir des insights financiers profonds.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fonctionnalités détaillées
    st.markdown("### ✨ Fonctionnalités Détaillées")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        #### 📄 Mode PDF
        - Téléchargez des rapports financiers
        - Extraction intelligente de texte
        - Navigation interactive des pages
        - Réponses précises basées sur le contenu
        """)
    
    with col2:
        st.markdown("""
        #### 📈 Mode YFinance
        - Données d'actions en temps réel
        - Analyse de tendances
        - Comparaison de plusieurs tickers
        - Insights financiers automatisés
        """)
    
    st.markdown("---")
    
    # Section API
    st.markdown("### 🔐 Prêt à commencer?")
    st.info("""
    Pour utiliser Astrali, vous avez besoin d'une **clé API Google Gemini**:
    
    1. Rendez-vous sur [Google AI Studio](https://aistudio.google.com/apikey)
    2. Connectez-vous avec votre compte Google
    3. Cliquez sur "Create API Key"
    4. Copiez la clé et entrez-la dans la barre latérale
    5. Cliquez sur "Connecter"
    
    **Gratuit et sans limite pour commencer!**
    """)
    
    # Authors
    st.markdown("""
    <div class="authors">
        <h4>👥 Développé par</h4>
        <p><strong>Fofana Ibrahim Seloh</strong> | <strong>Aya EL KOUACH</strong> | <strong>Mehdi Chanaa</strong></p>
        <p style="font-size: 0.9em; opacity: 0.7;">Astrali v1.0 - Assistants Financiers Intelligents</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Instructions dans la sidebar
    with st.sidebar:
        st.markdown("## 🔑 Initialisation")
        st.markdown("Entrez votre clé API Gemini ci-dessous pour commencer:")
        
        # Create a container for the API input section
        api_container = st.container()
        
        with api_container:
            # API Key input
            api_key = st.text_input(
                "Clé API Gemini",
                type="password",
                key="api_key_input",
                placeholder="AIzaSy...",
                help="Votre clé API Google Gemini (commence par AIza)"
            )
            
            # Validation and connection UI
            col1, col2 = st.columns([2, 1])
            
            with col1:
                connect_button = st.button(
                    "🚀 Connecter",
                    key="connect_btn",
                    use_container_width=True,
                    type="primary"
                )
            
            with col2:
                # Info button for help
                if st.button("❓", key="help_btn", help="Afficher l'aide"):
                    st.session_state.show_api_help = True
            
            # Real-time validation and feedback
            if api_key and len(api_key) > 5:  # Only validate if something is entered
                # Show validation status
                if 'api_validation_cache' not in st.session_state:
                    st.session_state.api_validation_cache = {}
                
                # Check if we already validated this key
                cached = st.session_state.api_validation_cache.get(api_key)
                
                if cached:
                    result = cached
                else:
                    # Show validating indicator
                    with st.spinner("🔍 Vérification de la clé..."):
                        result = validate_api_key(api_key)
                    st.session_state.api_validation_cache[api_key] = result
                
                # Display validation feedback
                if result['valid']:
                    st.success("✅ Clé valide et fonctionnelle", icon="✅")
                else:
                    error_messages = {
                        'empty': "⚠️ Clé vide",
                        'format': "⚠️ Format invalide (doit commencer par 'AIza')",
                        'unauthorized': "❌ Clé invalide ou expirée",
                        'forbidden': "❌ Accès refusé - Vérifiez les permissions",
                        'quota': "⏱️ Quota dépassé - Réessayez plus tard",
                        'connection': "🌐 Erreur de connexion - Vérifiez Internet",
                        'empty_response': "⚠️ Réponse vide de l'API",
                        'filtered': "🚫 Réponse filtrée par l'API",
                        'api_error': "❌ Erreur API",
                        'unknown': "❌ Erreur inconnue"
                    }
                    
                    error_display = error_messages.get(result['error_type'], result['message'])
                    st.warning(f"{error_display}\n\n{result['message']}")
            
            # Show help if requested
            if st.session_state.get('show_api_help', False):
                st.info("""
                ### 📖 Comment obtenir une clé API Gemini?
                
                1. Allez sur **[Google AI Studio](https://aistudio.google.com/apikey)**
                2. Connectez-vous avec votre compte Google
                3. Cliquez sur **"Create API Key"**
                4. Choisissez **"Create new API key in new project"**
                5. Copiez la clé générée (elle commence par `AIza`)
                6. Collez-la ci-dessus et cliquez sur **Connecter**
                
                ✅ **C'est gratuit!** Vous avez **60 requêtes/min** par défaut.
                """)
                
                if st.button("Fermer l'aide", key="close_help"):
                    st.session_state.show_api_help = False
                    st.rerun()
            
            # Handle connection button click
            if connect_button:
                if not api_key:
                    st.error("⚠️ Veuillez entrer une clé API")
                else:
                    # Validate the key
                    with st.spinner("🔐 Validation en cours..."):
                        result = validate_api_key(api_key)
                    
                    if result['valid'] and result['service']:
                        st.session_state.gemini_service = result['service']
                        st.session_state.api_key_submitted = True
                        st.balloons()
                        st.success("✅ Bienvenue dans Astrali!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Connexion échouée\n\n**Raison:** {result['message']}")
                        st.session_state.api_key_submitted = False
        
        with st.expander("❓ Besoin d'aide?"):
            st.markdown("""
            **Comment obtenir une clé API:**
            
            1. Visitez [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
            2. Cliquez sur "Create API Key"
            3. Copiez la clé
            4. Collez-la ci-dessus
            
            **C'est gratuit!** Vous avez 60 requêtes/min par défaut.
            """)
    
    st.stop()

# ===== SIDEBAR CONFIG (si API connectée) =====
with st.sidebar:
    st.title("⚙️ Configuration Astrali")
    
    st.markdown("### 🤖 API Gemini")
    
    # Afficher l'état de connexion
    if st.session_state.gemini_service:
        st.success("✅ API Gemini Connectée")
        
        # Bouton pour se déconnecter
        if st.button("🔌 Se Déconnecter", key="disconnect_btn", use_container_width=True):
            st.session_state.gemini_service = None
            st.session_state.api_key_submitted = False
            st.rerun()
    else:
        st.warning("⚠️ Non connectée")
    
    st.markdown("---")
    
    # Mode selector - seulement si API connectée
    if st.session_state.gemini_service:
        st.markdown("<div class='section-header'>📋 Mode d'utilisation</div>", unsafe_allow_html=True)
        mode = st.radio(
            "Sélectionnez votre mode",
            ["📑 Upload PDF", "📊 YFinance"],
            key="mode_selector",
            label_visibility="collapsed"
        )
    else:
        st.info("👈 Connectez-vous via la page d'accueil pour accéder aux modes")
        mode = None

# ===== GRAPHING FUNCTIONS =====
def plot_single_ticker(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Graphique interactif pour un ticker"""
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

def plot_multiple_tickers(data_dict, tickers):
    """Graphique comparatif pour plusieurs tickers"""
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
    
    trace_count = 0
    
    for ticker in tickers:
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
        
        color = colors[trace_count % len(colors)]
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=normalized,
                name=f"{ticker}",
                line=dict(color=color, width=3),
                mode='lines+markers',
                marker=dict(size=5),
                hovertemplate=f"<b>{ticker}</b><br>Date: %{{x|%Y-%m-%d}}<br>Prix (base 100): %{{y:.2f}}<extra></extra>"
            )
        )
        trace_count += 1
    
    fig.update_layout(
        title=f"Comparaison Multi-Tickers ({', '.join(tickers)}) - Prix normalisé base 100",
        xaxis_title="Date",
        yaxis_title="Prix (Base 100)",
        height=700,
        hovermode='x unified',
        template='plotly_white',
        plot_bgcolor='rgba(240, 240, 245, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12, color='#333333'),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='#333333',
            borderwidth=2
        ),
        margin=dict(r=50, b=50),
        xaxis=dict(gridwidth=1, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    )
    
    return fig

# ===== PDF VIEWER FUNCTION =====
def render_pdf_page(pdf_file, page_num: int, zoom_level: int = 100):
    """Affiche la page PDF avec zoom via DPI dynamique"""
    try:
        from pdf2image import convert_from_bytes
        import io
        
        # DPI base réduit (72) + zoom pour économiser mémoire et permettre zoom visible
        base_dpi = 72
        effective_dpi = int(base_dpi * zoom_level / 100)
        effective_dpi = max(18, min(effective_dpi, 200))  # Limite 18-200 DPI (permet 25%)
        
        # Convertir la page avec le DPI calculé
        images = convert_from_bytes(pdf_file.getbuffer(), first_page=page_num, last_page=page_num, dpi=effective_dpi)
        
        if images:
            img = images[0]
            # Afficher dans un conteneur scrollable via HTML
            import base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Conteneur avec scroll horizontal si image trop large
            st.markdown(f'''
                <div style="max-height: 500px; overflow: auto; border-radius: 0.5rem; background: #fafafa; padding: 0.5rem; border: 1px solid #ddd;">
                    <img src="data:image/png;base64,{img_base64}" style="max-width: none; display: block; margin: auto;" />
                </div>
            ''', unsafe_allow_html=True)
        
        # Get total pages (use low DPI for speed)
        all_images = convert_from_bytes(pdf_file.getbuffer(), dpi=30)
        total = len(all_images)
        
        return total
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return 1

# ===== MODE 1: PDF UPLOAD (CHATGPT-LIKE) =====
if mode == "📑 Upload PDF":
    
    # Vérifier que l'API est connectée
    if not st.session_state.gemini_service:
        st.error("❌ Impossible d'accéder à ce mode. Veuillez connecter votre API Gemini d'abord.")
        st.stop()
    
    # Handler pour navigation depuis les sources inline (DOIT être avant tout le reste)
    if 'nav_page' in st.query_params:
        try:
            page_to_nav = int(st.query_params['nav_page'])
            st.session_state.current_pdf_page = page_to_nav
            st.session_state.pdf_viewer_visible = True
            del st.query_params['nav_page']
            st.toast(f"📄 Navigation vers page {page_to_nav}", icon="✅")
            st.rerun()
        except:
            del st.query_params['nav_page']
    
    # Layout dynamique selon visibilité du PDF viewer
    if st.session_state.pdf_rag:
        # Bouton toggle pour afficher/masquer le PDF viewer (en haut à droite)
        toggle_col1, toggle_col2 = st.columns([0.9, 0.1])
        with toggle_col2:
            if st.session_state.pdf_viewer_visible:
                if st.button("📄 ✕", key="hide_pdf", help="Masquer le PDF viewer"):
                    st.session_state.pdf_viewer_visible = False
                    st.rerun()
            else:
                if st.button("📄 ◀", key="show_pdf", help="Afficher le PDF viewer"):
                    st.session_state.pdf_viewer_visible = True
                    st.rerun()
        
        # Colonnes selon état du viewer
        if st.session_state.pdf_viewer_visible:
            left_col, right_col = st.columns([1.3, 0.7], gap="small")
        else:
            left_col = st.container()
            right_col = None
        
        # ===== LEFT COLUMN: CHAT INTERFACE =====
        with left_col:
            # Header amélioré avec gradient et animation
            st.markdown("""
            <style>
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            .astrali-header {
                background: linear-gradient(-45deg, #667eea, #764ba2, #667eea, #5a67d8);
                background-size: 400% 400%;
                animation: gradientShift 8s ease infinite;
                padding: 1.2rem 1.5rem;
                border-radius: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.35);
                display: flex;
                align-items: center;
                gap: 0.8rem;
            }
            .astrali-header h1 {
                color: white;
                font-size: 1.6rem;
                font-weight: 800;
                margin: 0;
                letter-spacing: -0.5px;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .astrali-header .subtitle {
                color: rgba(255,255,255,0.85);
                font-size: 0.85rem;
                margin: 0.3rem 0 0 0;
            }
            .astrali-logo {
                font-size: 2.2rem;
                text-shadow: 0 0 0.5rem white, 0 0 1rem rgba(255,255,255,0.8);
                color: white;
                filter: brightness(2) saturate(0.5);
            }
            </style>
            <div class="astrali-header">
                <span class="astrali-logo">🧠</span>
                <div>
                    <h1>Astrali</h1>
                    <p class="subtitle">Assistant IA Financier</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            user_question = None
            should_process = (
                st.session_state.pdf_chat_history and 
                st.session_state.pdf_chat_history[-1]["role"] == "user"
            )

            # CSS pour le conteneur de chat scrollable
            st.markdown("""
            <style>
            /* Container principal du chat */
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
                background: linear-gradient(180deg, #fafbfc 0%, #f0f2f6 100%);
                border: 1px solid #e0e5ec;
                border-radius: 1rem;
                box-shadow: inset 0 2px 8px rgba(0,0,0,0.03);
            }
            
            /* Style des messages utilisateur */
            div[data-testid="stChatMessage"][class*="user"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                border-radius: 1rem 1rem 0.3rem 1rem !important;
                padding: 0.8rem 1rem !important;
                margin: 0.5rem 0 !important;
                box-shadow: 0 3px 12px rgba(102, 126, 234, 0.25);
            }
            
            div[data-testid="stChatMessage"][class*="user"] p {
                color: white !important;
            }
            
            /* Style des messages assistant */
            div[data-testid="stChatMessage"][class*="assistant"] {
                background: white !important;
                border-radius: 1rem 1rem 1rem 0.3rem !important;
                padding: 1rem !important;
                margin: 0.5rem 0 !important;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            }
            
            /* Style des boutons sources */
            div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 0.6rem !important;
                padding: 0.4rem 0.8rem !important;
                font-weight: 600 !important;
                font-size: 0.8rem !important;
                transition: all 0.2s ease !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
            }
            
            div[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"]:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
            }
            
            /* Séparateur sources */
            div[data-testid="stChatMessage"] hr {
                border: none;
                border-top: 1px solid #e8ecf4;
                margin: 0.8rem 0;
            }
            
            /* Caption sources */
            div[data-testid="stChatMessage"] .stCaption {
                color: #667eea !important;
                font-weight: 600 !important;
                font-size: 0.85rem !important;
            }
            
            /* Scrollbar du chat container */
            div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar {
                width: 6px;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-track {
                background: #f1f3f5;
                border-radius: 10px;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb {
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 10px;
            }
            
            /* Animation d'entrée des messages */
            @keyframes messageSlideIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            div[data-testid="stChatMessage"] {
                animation: messageSlideIn 0.3s ease-out;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Conteneur de chat avec éléments natifs Streamlit
            chat_container = st.container(height=480, border=True)
            
            with chat_container:
                for msg_idx, msg in enumerate(st.session_state.pdf_chat_history):
                    if msg["role"] == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(msg['content'])
                    else:
                        with st.chat_message("assistant", avatar="🧠"):
                            st.markdown(msg['content'])
                            
                            # Sources avec boutons natifs Streamlit
                            if msg.get('sources'):
                                st.markdown("---")
                                st.caption("📚 Sources:")
                                
                                # Créer une ligne de boutons pour les sources
                                cols = st.columns(min(len(msg['sources']), 5))
                                for idx, source in enumerate(msg.get('sources', [])):
                                    page_num = source.get('page', 1)
                                    score = source.get('rerank_score', 0.5)
                                    text = source.get('text', '')[:80]
                                    
                                    if score > 10:
                                        relevance_pct = max(5, min(100, int(score / 10)))
                                    else:
                                        relevance_pct = max(5, min(100, int(score * 100)))
                                    
                                    col_idx = idx % min(len(msg['sources']), 5)
                                    with cols[col_idx]:
                                        # Bouton natif Streamlit - peut modifier session_state
                                        if st.button(f"[{idx+1}] p.{page_num}", key=f"src_{msg_idx}_{idx}", 
                                                    help=f"{text}... ({relevance_pct}%)", 
                                                    use_container_width=True):
                                            st.session_state.current_pdf_page = page_num
                                            st.session_state.pdf_viewer_visible = True
                                            st.toast(f"📄 Navigation vers page {page_num}", icon="✅")
                                            st.rerun()
                
                # Indicateur de chargement
                if should_process:
                    with st.chat_message("assistant", avatar="🧠"):
                        st.markdown("⏳ *Analyse en cours...*")

            # Chat input
            st.markdown("")
            user_question = st.chat_input("💭 Pose ta question...", key="pdf_chat_input", max_chars=500)
            
            # Process message if submitted
            if user_question:
                st.session_state.pdf_chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                st.rerun()
            
            # Process in background if needed
            if should_process:
                question = st.session_state.pdf_chat_history[-1]["content"]
                try:
                    result = st.session_state.pdf_rag.query(question)
                    response_text = result.get('response', '')
                    sources = result.get('sources', [])
                    
                    st.session_state.pdf_chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "sources": sources
                    })
                    
                    st.rerun()
                except Exception as e:
                    st.session_state.pdf_chat_history.pop()
                    st.error(f"❌ Erreur: {str(e)}")
        
        # ===== RIGHT COLUMN: PDF VIEWER (Pliable) =====
        if st.session_state.pdf_viewer_visible and right_col is not None:
            with right_col:
                # CSS pour le PDF viewer amélioré
                st.markdown("""
                <style>
                /* PDF Viewer Header */
                .pdf-viewer-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 0.8rem 1rem;
                    border-radius: 0.8rem 0.8rem 0 0;
                    margin-bottom: 0;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-weight: 700;
                    font-size: 1rem;
                    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
                }
                
                /* Zoom & Nav buttons */
                .stButton > button {
                    transition: all 0.2s ease !important;
                }
                
                /* PDF container */
                .pdf-page-container {
                    background: linear-gradient(180deg, #f8f9fc 0%, #eef1f5 100%);
                    border: 1px solid #e0e5ec;
                    border-radius: 0.6rem;
                    padding: 0.5rem;
                    box-shadow: inset 0 2px 6px rgba(0,0,0,0.04);
                }
                
                /* Page indicator */
                .page-indicator {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 2rem;
                    font-weight: 700;
                    font-size: 0.9rem;
                    text-align: center;
                    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
                }
                
                /* Zoom display */
                .zoom-display {
                    background: linear-gradient(135deg, #e8ecff 0%, #f0e8ff 100%);
                    border: 2px solid #667eea;
                    color: #667eea;
                    padding: 0.4rem 0.8rem;
                    border-radius: 0.5rem;
                    font-weight: 700;
                    font-size: 0.85rem;
                    text-align: center;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Header du PDF viewer
                st.markdown(f"""
                <div class="pdf-viewer-header">
                    📄 PDF Viewer - {st.session_state.pdf_file_name[:25] if st.session_state.pdf_file_name else 'Document'}...
                </div>
                """, unsafe_allow_html=True)
                
                # Zoom controls améliorés
                zoom_col1, zoom_col2, zoom_col3, zoom_col4 = st.columns([1, 2, 1, 1], gap="small")
                
                with zoom_col1:
                    if st.button("🔍➖", use_container_width=True, key="zoom_out", help="Zoom arrière"):
                        if st.session_state.pdf_zoom_level > 25:
                            st.session_state.pdf_zoom_level -= 25
                            st.rerun()
                
                with zoom_col2:
                    st.markdown(f"<div class='zoom-display'>🔍 {st.session_state.pdf_zoom_level}%</div>", unsafe_allow_html=True)
                
                with zoom_col3:
                    if st.button("🔍➕", use_container_width=True, key="zoom_in", help="Zoom avant"):
                        if st.session_state.pdf_zoom_level < 200:
                            st.session_state.pdf_zoom_level += 25
                            st.rerun()
                
                with zoom_col4:
                    if st.button("↺", use_container_width=True, key="zoom_reset", help="Reset 50%"):
                        if st.session_state.pdf_zoom_level != 50:
                            st.session_state.pdf_zoom_level = 50
                            st.rerun()
                
                # PDF viewer - rendu direct
                total_pages = render_pdf_page(st.session_state.pdf_file, st.session_state.current_pdf_page, st.session_state.pdf_zoom_level)
                
                # Navigation améliorée
                st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1], gap="small")
                
                with nav_col1:
                    prev_disabled = st.session_state.current_pdf_page <= 1
                    if st.button("◀ Préc.", use_container_width=True, key="prev_page", 
                                help="Page précédente", disabled=prev_disabled):
                        st.session_state.current_pdf_page = max(1, st.session_state.current_pdf_page - 1)
                        st.rerun()
                
                with nav_col2:
                    st.markdown(f"<div class='page-indicator'>📖 {st.session_state.current_pdf_page} / {total_pages}</div>", unsafe_allow_html=True)
                
                with nav_col3:
                    next_disabled = st.session_state.current_pdf_page >= total_pages
                    if st.button("Suiv. ▶", use_container_width=True, key="next_page", 
                                help="Page suivante", disabled=next_disabled):
                        st.session_state.current_pdf_page = min(total_pages, st.session_state.current_pdf_page + 1)
                        st.rerun()
                
                # Page slider amélioré
                if total_pages > 1:
                    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                    new_page = st.slider("Navigation rapide", 1, total_pages, 
                                        value=st.session_state.current_pdf_page, 
                                        key="page_nav_slider", label_visibility="collapsed")
                    if new_page != st.session_state.current_pdf_page:
                        st.session_state.current_pdf_page = new_page
                        st.rerun()
    
    else:
        # Upload screen - Enhanced & Dynamic
        
        # Premium header with animation
        st.markdown("""<style>
            @keyframes fadeInDown {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes float {
                0%, 100% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            @keyframes slideInUp {
                from { opacity: 0; transform: translateY(30px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            .premium-header {
                animation: fadeInDown 0.6s ease-out;
                margin-bottom: 2rem;
            }
            
            .header-title {
                font-size: 2.2rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin: 0;
                animation: fadeInDown 0.6s ease-out;
            }
            
            .header-subtitle {
                font-size: 1rem;
                color: #999;
                margin: 0.5rem 0 0 0;
                animation: fadeInDown 0.8s ease-out;
            }
            
            .upload-zone {
                border: 2px dashed #667eea;
                border-radius: 1.2rem;
                padding: 2.5rem 1.5rem;
                text-align: center;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.02) 0%, rgba(118, 75, 162, 0.02) 100%);
                margin: 1.5rem 0;
                transition: all 0.3s ease;
                animation: slideInUp 0.8s ease-out;
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }
            
            .upload-zone:hover {
                border-color: #764ba2;
                background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
                transform: translateY(-2px);
                box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
            }
            
            .upload-icon {
                font-size: 3.5rem;
                margin-bottom: 1rem;
                animation: float 3s ease-in-out infinite;
            }
            
            .upload-text {
                color: #333;
                font-size: 1.1rem;
                font-weight: 600;
                margin: 0;
            }
            
            .upload-subtext {
                color: #999;
                font-size: 0.9rem;
                margin: 0.5rem 0 0 0;
            }
            
            .file-info-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.2rem 1.5rem;
                border-radius: 1rem;
                margin: 1.2rem 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.95rem;
                animation: slideInUp 0.6s ease-out;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
            }
            
            .file-name {
                font-weight: 700;
                font-size: 1.05rem;
            }
            
            .file-badge {
                background: rgba(255, 255, 255, 0.2);
                padding: 0.3rem 0.8rem;
                border-radius: 0.5rem;
                font-size: 0.85rem;
                font-weight: 600;
            }
            
            .progress-section {
                margin-top: 1.5rem;
            }
            
            .progress-label {
                font-weight: 600;
                color: #667eea;
                font-size: 0.9rem;
                margin-bottom: 0.8rem;
                animation: fadeInDown 0.6s ease-out;
            }
            
            .success-card {
                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                color: white;
                padding: 1.2rem 1.5rem;
                border-radius: 1rem;
                margin-top: 1.2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 0.95rem;
                animation: slideInUp 0.6s ease-out;
                box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .stat-box {
                background: #f5f5f5;
                padding: 1rem;
                border-radius: 0.8rem;
                text-align: center;
                border-left: 3px solid #667eea;
                animation: slideInUp 0.7s ease-out;
            }
            
            .stat-value {
                font-size: 1.8rem;
                font-weight: 800;
                color: #667eea;
                margin: 0;
            }
            
            .stat-label {
                font-size: 0.85rem;
                color: #999;
                margin-top: 0.5rem;
            }
            
            @media (max-width: 768px) {
                .header-title {
                    font-size: 1.8rem;
                }
                
                .upload-zone {
                    padding: 2rem 1rem;
                }
                
                .upload-icon {
                    font-size: 2.5rem;
                }
                
                .file-info-card {
                    flex-direction: column;
                    text-align: center;
                    gap: 0.5rem;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>""", unsafe_allow_html=True)
        
        st.markdown("""<div>
            <h1 class="header-title">🧠 Astrali</h1>
            <p class="header-subtitle">Intelligence financière quantitative</p>
        </div>""", unsafe_allow_html=True)
        
        # Upload area - File uploader must be visible and clickable
        uploaded_file = st.file_uploader(
            "📤 Glissez-déposez un PDF ici ou cliquez pour sélectionner un fichier",
            type="pdf",
            key="initial_upload"
        )
        
        if not uploaded_file:
            
            # Info cards
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown("""
                <div class="stat-box">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                    <p class="stat-label">Traitement rapide</p>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">Indexation en quelques secondes</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="stat-box">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                    <p class="stat-label">Analyse précise</p>
                    <p style="margin: 0; color: #666; font-size: 0.9rem;">IA quantitative avancée</p>
                </div>
                """, unsafe_allow_html=True)
        
        if uploaded_file:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            
            # File info - enhanced
            file_name_display = uploaded_file.name[:40]
            file_size_display = f"{file_size_mb:.1f}"
            st.markdown(f"""
            <div class="file-info-card">
                <span>📄 <span class="file-name">{file_name_display}</span> • {file_size_display} MB</span>
                <span class="file-badge">✓ PDF Prêt</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if new file
            if uploaded_file.name != st.session_state.pdf_file_name:
                st.session_state.pdf_file_name = uploaded_file.name
                st.session_state.pdf_file = uploaded_file
                st.session_state.pdf_chat_history = []
                st.session_state.pdf_rag = None
                st.session_state.current_pdf_page = 1
                
                # Processing with enhanced progress
                st.markdown("<div class='progress-section'>", unsafe_allow_html=True)
                st.markdown("<p class='progress-label'>🔄 Traitement du document...</p>", unsafe_allow_html=True)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    if not st.session_state.gemini_service:
                        st.error("⚠️ Configurez votre clé API d'abord")
                    else:
                        # Step 1: Initialize
                        status_text.markdown("**Étape 1/3:** Initialisation du pipeline...")
                        progress_bar.progress(20)
                        import time
                        time.sleep(0.3)
                        pdf_rag = PDFRagPipeline(st.session_state.gemini_service)
                        
                        # Step 2: Extract & Process
                        status_text.markdown("**Étape 2/3:** Extraction et traitement du texte...")
                        progress_bar.progress(50)
                        time.sleep(0.3)
                        pdf_rag.process_pdf(uploaded_file)
                        
                        # Step 3: Indexing
                        status_text.markdown("**Étape 3/3:** Indexation et création des embeddings...")
                        progress_bar.progress(85)
                        time.sleep(0.3)
                        
                        st.session_state.pdf_rag = pdf_rag
                        progress_bar.progress(100)
                        status_text.empty()
                        
                        # Success - enhanced
                        pages = len(set(c.page_num for c in pdf_rag.chunks))
                        chunks = len(pdf_rag.chunks)
                        
                        st.markdown("""
                        <div class="success-card">
                            <span>✅ Traitement réussi!</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="stats-grid">
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3, gap="medium")
                        
                        mb_display = f"{file_size_mb:.1f}"
                        
                        with col1:
                            st.markdown(f"""
                            <div class="stat-box">
                                <p class="stat-value">{pages}</p>
                                <p class="stat-label">Pages</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="stat-box">
                                <p class="stat-value">{chunks}</p>
                                <p class="stat-label">Chunks</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="stat-box">
                                <p class="stat-value">{mb_display}</p>
                                <p class="stat-label">MB</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        time.sleep(0.5)
                        st.rerun()
                        
                except Exception as e:
                    error_msg = str(e)[:100]
                    st.error(f"❌ Erreur: {error_msg}")



# ===== MODE 2: YFINANCE =====
elif mode == "📊 YFinance":
    
    # Vérifier que l'API est connectée
    if not st.session_state.gemini_service:
        st.error("❌ Impossible d'accéder à ce mode. Veuillez connecter votre API Gemini d'abord.")
        st.stop()
    
    st.title("📊 Analyse Données Boursières")
    
    with st.sidebar:
        st.subheader("Configuration")
        
        all_tickers = st.checkbox("Tous les tickers", value=False, key="all_tickers")
        
        if all_tickers:
            selected_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        else:
            selected_tickers = st.multiselect(
                "Sélectionner tickers:",
                ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                default=['AAPL'],
                key="ticker_select"
            )
        
        period = st.slider("Période (mois):", 1, 24, 6, key="period_slider")
        
        if st.button("Charger les données", key="load_data_btn", use_container_width=True):
            if not selected_tickers:
                st.error("Sélectionnez au moins un ticker")
            elif not st.session_state.gemini_service:
                st.error("API non configurée")
            else:
                with st.spinner("Chargement données Yahoo Finance..."):
                    try:
                        yfinance_rag = YFinanceRagAssistant(
                            st.session_state.gemini_service,
                            tickers=selected_tickers
                        )
                        yfinance_rag.fetch_and_process_data(period)
                        st.session_state.yfinance_rag = yfinance_rag
                        st.session_state.yfinance_data_loaded = True
                        st.session_state.selected_tickers = selected_tickers
                        st.success(f"✅ Données chargées pour {len(selected_tickers)} tickers")
                    except Exception as e:
                        st.error(f"❌ Erreur: {str(e)}")
    
    if st.session_state.yfinance_data_loaded and st.session_state.yfinance_rag:
        st.subheader("📈 Graphique")
        
        # Create charts
        try:
            # Un seul graphique avec toutes les courbes
            if len(st.session_state.selected_tickers) == 1:
                ticker = st.session_state.selected_tickers[0]
                df = st.session_state.yfinance_rag.data[ticker]['dataframe']
                fig = plot_single_ticker(df, ticker)
            else:
                # Graphique comparatif avec plusieurs courbes
                fig = plot_multiple_tickers(
                    st.session_state.yfinance_rag.data,
                    st.session_state.selected_tickers
                )
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur graphique: {str(e)}")
        
        # Chat section
        st.subheader("💬 Questions sur les données")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Input
        user_question = st.chat_input("Votre question...")
        
        if user_question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            with st.chat_message("user"):
                st.write(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("IA réfléchit..."):
                    try:
                        result = st.session_state.yfinance_rag.query(user_question)
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
                        st.session_state.chat_history.pop()
                        result = None
                
                if result:
                    st.write(result.get('response', ''))
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get('response', '')
                    })
    else:
        st.info("Configurez les tickers dans la sidebar et cliquez sur 'Charger les données'")
