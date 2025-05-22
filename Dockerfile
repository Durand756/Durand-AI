# Dockerfile pour optimiser le déploiement
FROM python:3.10-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers de requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY main.py .

# Variables d'environnement
ENV PYTHONPATH=/app
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/transformers

# Exposition du port
EXPOSE 8000

# Commande de démarrage
CMD ["python", "main.py"]
