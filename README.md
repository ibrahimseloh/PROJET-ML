# ✨ **Astrali** - Assistant Financier Intelligent avec IA

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-red)](https://streamlit.io)
[![Gemini API](https://img.shields.io/badge/Gemini-2.5--flash-yellow)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Astrali** est un assistant financier intelligent alimenté par l'IA Google Gemini. Il combine l'analyse de documents PDF avec les données financières en temps réel pour fournir des insights profonds et des réponses précises.

---

## 🎯 À Propos

### Qu'est-ce qu'Astrali?

Astrali est une **plateforme d'analyse financière intelligente** qui permet aux utilisateurs de:

✅ **Analyser des documents financiers** - Téléchargez des rapports annuels, bilans, états financiers  
✅ **Accéder aux données boursières** - Consultez les prix d'actions en temps réel via YFinance  
✅ **Converser avec l'IA** - Posez des questions et recevez des réponses intelligentes avec citations  
✅ **Naviguer les sources** - Cliquez sur les sources pour accéder directement aux passages pertinents  

### Technologie

- **LLM**: Google Gemini 2.5-flash
- **Frontend**: Streamlit
- **Architecture**: RAG (Retrieval-Augmented Generation)
- **Données Boursières**: YFinance
- **Traitement PDF**: pdf2image + Extraction intelligente

---

## 🚀 Démarrage Rapide

### 1️⃣ Prérequis

```bash
# Python 3.10+ (vérifier la version)
python --version
```

### 2️⃣ Installation

```bash
# Cloner le projet
cd "/home/fofana-ibrahim-seloh/Downloads/Projet ML"

# Installer les dépendances
pip install -r requirements.txt
```

### 3️⃣ Obtenir une clé API Gemini

1. Allez sur [Google AI Studio](https://aistudio.google.com/apikey)
2. Connectez-vous avec votre compte Google
3. Cliquez sur **"Create API Key"**
4. Copiez la clé (gratuite, 60 requêtes/min par défaut)

### 4️⃣ Lancer l'application

```bash
# Linux/Mac
streamlit run streamlit_app/app.py

# Ou avec le script fourni
bash run.sh
```

L'app démarre sur **http://localhost:8501**

### 5️⃣ Utiliser Astrali

1. **Page d'accueil**: Présentation et instructions
2. **Sidebar**: Entrez votre clé API Gemini
3. **Cliquez "Connecter et Démarrer"**
4. **Choisissez un mode**: PDF ou YFinance
5. **Posez vos questions!**

---

## 📚 Fonctionnalités Détaillées

### 📄 Mode PDF - Analyse de Documents

```
✨ Flux d'utilisation:
1. Téléchargez un PDF (rapport financier, bilan, etc.)
2. Le PDF est affiché dans le viewer côte à côte
3. Posez des questions sur le contenu
4. Astrali cherche les passages pertinents
5. Cliquez sur [1], [2]... pour naviguer vers la source
```

**Capacités:**
- Upload illimité de documents
- Visualisation avec contrôles de zoom (25-200%)
- Navigation page par page
- Extraction intelligente de texte
- Réponses avec citations précises
- Chat conversationnel

**Exemple:**
```
Question: "Quels sont les revenus totaux de l'année 2024?"
Réponse: "Les revenus totaux pour 2024 s'élèvent à 1,2 milliards € [1]..."
→ Cliquez [1] pour aller à la page 12 où ce texte apparaît
```

### 📈 Mode YFinance - Données Boursières

```
✨ Flux d'utilisation:
1. Sélectionnez les tickers à analyser (AAPL, MSFT, etc.)
2. Visualisez les graphiques interactifs
3. Posez des questions analytiques
4. Astrali fournit des insights financiers
```

**Capacités:**
- Support multi-tickers
- Graphiques interactifs (prix, volume, comparaisons)
- Données historiques configurable
- Questions analytiques (volatilité, tendances, etc.)
- Réponses contextualisées avec données

**Exemple:**
```
Tickers: AAPL, MSFT, GOOGL
Question: "Quel ticker a la meilleure performance cette année?"
Réponse: "MSFT affiche la meilleure performance avec +45%..."
→ Graphique comparatif normalisé s'affiche
```

---

## 🏗️ Architecture

### Structure du Projet

```
Astrali/
├── streamlit_app/
│   └── app.py                 # Application Streamlit principale
├── backend/
│   ├── config.py              # Configuration Gemini
│   ├── prompts.py             # Prompts IA
│   ├── agents/
│   │   └── __init__.py       # Graphiques interactifs
│   ├── rag/
│   │   ├── __init__.py       # PDFRagPipeline
│   │   └── yfinance_rag.py   # YFinanceRagAssistant
│   ├── services/
│   │   ├── gemini_service.py   # Interface Gemini API
│   │   ├── pdf_processor.py    # Traitement PDF
│   │   └── yfinance_service.py # Fetch données YFinance
│   └── utils/
│       └── reranker.py        # Reranking des résultats
├── .streamlit/
│   ├── config.toml            # Config Streamlit
│   └── secrets.toml           # Clé API (git-ignored)
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

### Flux de Données RAG

```
📥 Entrée Utilisateur
    ↓
🔍 Extraction (PDF ou YFinance)
    ↓
📚 Chunking & Embedding
    ↓
🎯 Recherche Sémantique
    ↓
⚖️ Reranking (par pertinence)
    ↓
🧠 Prompt Construction
    ↓
🤖 Appel Gemini
    ↓
📤 Réponse avec citations
```

---

## 🔑 Configuration API

### Variables d'Environnement

```bash
# .streamlit/secrets.toml (créé manuellement)
GEMINI_API_KEY = "votre_clé_api_ici"
```

### Configuration Streamlit

```toml
# .streamlit/config.toml
[theme]
primaryColor="#667eea"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f0f2f6"

[client]
showErrorDetails=false
```

---

## 📦 Dépendances Principales

```
streamlit>=1.52          # Framework web
google-generativeai>=0.7 # Gemini API
pdf2image>=1.16          # Extraction PDF
yfinance>=0.2.38         # Données boursières
plotly>=5.24             # Graphiques interactifs
pandas>=2.0              # Manipulation données
numpy>=1.24              # Calculs numériques
```

---

## 🤖 Comment Fonctionne RAG (Retrieval-Augmented Generation)

### 1. **Extraction & Chunking**
```python
# Le PDF/YFinance est divisé en chunks
Chunk 1: "Les revenus de 2024 s'élèvent à 1.2B..."
Chunk 2: "La marge bénéficiaire atteint 25%..."
Chunk 3: "Les coûts opérationnels ont diminué..."
```

### 2. **Embedding & Vectorisation**
```python
# Chaque chunk est transformé en vecteur sémantique
"revenus" → [0.234, -0.567, 0.890, ...]
```

### 3. **Recherche Sémantique**
```python
# Question: "Quels sont les revenus?"
# → Trouve les chunks similaires
# → Top 5 chunks retournés
```

### 4. **Reranking**
```python
# Classe les chunks par pertinence
[Chunk 1 - Score: 0.95]  ← Très pertinent
[Chunk 3 - Score: 0.67]  ← Modérément pertinent
[Chunk 2 - Score: 0.42]  ← Moins pertinent
```

### 5. **Génération de Réponse**
```python
# Gemini utilise les chunks pertinents
Prompt: "Question: Quels sont les revenus?
Contexte: [chunks relevants]
→ Réponds précisément avec citations"

Réponse: "Les revenus s'élèvent à 1.2B [1]..."
```

---

## 💡 Cas d'Usage

### Finance d'Entreprise
- Analyser des rapports financiers
- Comparer des états financiers
- Extraire des KPIs
- Valider des hypothèses

### Analyse Boursière
- Comparer plusieurs actions
- Analyser les tendances
- Évaluer la volatilité
- Identifier des opportunités

### Recherche Financière
- Extraire des données de documents
- Synthesiser des informations
- Créer des rapports automatisés
- Analyser des trends de marché

---

## 🧪 Tests & Validation

### Tester PDF Mode
```
1. Upload un PDF test (rapports annuels fournis)
2. Question: "Résume les points clés de ce document"
3. Vérifie que les réponses incluent des citations [1], [2]
4. Clique sur les citations pour vérifier la navigation
```

### Tester YFinance Mode
```
1. Sélectionne AAPL, MSFT, GOOGL
2. Question: "Quel ticker a la meilleure performance?"
3. Vérifie le graphique comparatif
4. Vérifie que la réponse cite les données affichées
```

---

## 🔒 Sécurité

- ✅ Clé API stockée dans `.streamlit/secrets.toml` (git-ignored)
- ✅ Pas de sauvegarde de l'historique de chat
- ✅ Pas de logs contenant des données sensibles
- ✅ HTTPS recommandé en production

### `.gitignore`
```
.streamlit/secrets.toml    # Clé API
*.pyc
__pycache__/
.env
```

---

## 🐛 Dépannage

### Erreur: "Impossible de se connecter à l'API"
```bash
✓ Vérifier que la clé API est correcte
✓ Vérifier la connexion Internet
✓ Essayer de créer une nouvelle clé sur aistudio.google.com
```

### Erreur: "Pas de données YFinance"
```bash
✓ Vérifier le ticker (ex: AAPL, pas Apple)
✓ Vérifier la connexion Internet
✓ YFinance a parfois des limites de rate limiting
```

### Erreur: "PDF ne s'affiche pas"
```bash
✓ Vérifier que le fichier est un PDF valide
✓ Taille maximale recommandée: 50MB
✓ Essayer avec un autre PDF
```

---

## 📝 Logs & Débogage

```bash
# Logs Streamlit
streamlit run app.py --logger.level=debug

# Logs Python
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
```

---

## 🚀 Déploiement

### Deployer sur Streamlit Cloud

```bash
# 1. Créer un compte sur share.streamlit.io
# 2. Connecter votre repo GitHub
# 3. Déployer: Astrali/streamlit_app/app.py
# 4. Ajouter GEMINI_API_KEY dans Settings
```

### Deployer Localement (Production)

```bash
# Installer Nginx/Apache
# Configurer Streamlit en mode production
streamlit run app.py \
  --server.port=80 \
  --server.address=0.0.0.0 \
  --server.headless=true
```

---

## 👥 Auteurs

**Équipe de Développement:**
- **Fofana Ibrahim Seloh** - Développeur Principal
- **Aya EL KOUACH** - Développeuse
- **Mehdi Chanaa** - Développeur

**Version**: 1.0  
**Date**: Décembre 2025  
**Statut**: Stable ✅

---

## 📄 Licence

Ce projet est sous license MIT. Voir LICENSE pour plus de détails.

---

## 📚 Ressources

- [Documentation Streamlit](https://docs.streamlit.io)
- [Google Gemini API](https://ai.google.dev/)
- [YFinance Documentation](https://yfinance.readthedocs.io)
- [Retrieval-Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag)

---

## 🤝 Support

Pour des questions ou des bugs:
1. Vérifier la section Dépannage
2. Consulter les logs
3. Vérifier la clé API
4. Redémarrer l'application

---

**Astrali v1.0** - Transforming Financial Analysis with AI ✨
