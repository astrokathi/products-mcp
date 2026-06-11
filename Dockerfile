FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY main.py .
COPY search_chroma.py .

# FastMCP normally communicates over stdin/stdout.
# If running an HTTP transport, EXPOSE port here.
# EXPOSE 8000

CMD ["python", "main.py"]
