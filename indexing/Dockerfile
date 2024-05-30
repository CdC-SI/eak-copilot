# Verwende das offizielle Python-Image als Basis
FROM python:3.9-slim

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere die Anforderungen in den Container und installiere sie
COPY indexing/requirements.txt .
RUN pip install -r requirements.txt

# Kopiere den Rest des Codes in den Container
COPY . .

# Set pythonpath
ENV PYTHONPATH="/app:/app/config"

# Standardbefehl, der ausgeführt wird, wenn der Container gestartet wird
CMD ["uvicorn", "indexing.app.main:app", "--host", "0.0.0.0", "--port", "8011"]
