FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY main.py .
COPY search_chroma.py .

# Cache the embedding model during build time so it doesn't download on startup
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='mixedbread-ai/mxbai-embed-large-v1')"

# FastMCP normally communicates over stdin/stdout.
# If running an HTTP transport, EXPOSE port here.
# EXPOSE 8000

CMD ["python", "main.py"]
