FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (build-essential for compiling C extensions like hnswlib for chromadb)
RUN apt-get update && apt-get install -y \
    build-essential \
    bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port (HF Spaces defaults to 7860, Render uses PORT env var)
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
