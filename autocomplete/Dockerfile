# Verwende das offizielle Python-Image als Basis
FROM python:3.9-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Installieren Sie gcc und andere Build-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Kopiere die Anforderungen in den Container und installiere sie
COPY autocomplete/requirements.txt .
RUN pip install -r requirements.txt

# Kopiere den Rest des Codes in den Container
COPY . .

# Set pythonpath
ENV PYTHONPATH="/app:/app/config"

# Standardbefehl, der ausgeführt wird, wenn der Container gestartet wird
CMD ["uvicorn", "autocomplete.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
