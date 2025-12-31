#!/bin/bash
# Script de lancement de l'application Streamlit

echo "🚀 Démarrage de l'Assistant Financier RAG..."
echo ""

# Vérifier si les dépendances sont installées
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit n'est pas installé. Installation..."
    pip install -r requirements.txt
fi

# Vérifier si .env existe
if [ ! -f .env ]; then
    echo "⚠️  Fichier .env non trouvé. Création à partir du template..."
    cp .env.example .env
    echo "📝 Veuillez éditer .env et ajouter votre clé API Gemini"
fi

# Lancer l'app
echo "🌐 Lancement sur http://localhost:8501"
echo ""

streamlit run streamlit_app/app.py
