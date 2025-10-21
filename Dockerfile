# Utiliser une image de base Python légère
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les dépendances et les installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du projet dans le conteneur
COPY . .

# Exposer le port de l’application Flask
EXPOSE 5000

# Commande pour démarrer l’application avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
