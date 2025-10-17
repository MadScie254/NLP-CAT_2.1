
FROM python:3.11.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download GloVe embeddings
RUN mkdir -p data/embeddings && \
    wget -q http://nlp.stanford.edu/data/glove.6B.zip -O data/embeddings/glove.6B.zip && \
    unzip -q data/embeddings/glove.6B.zip -d data/embeddings/ && \
    rm data/embeddings/glove.6B.zip

# Create necessary directories
RUN mkdir -p artifacts/models artifacts/results artifacts/plots results data/raw data/processed

# Set environment variables
ENV PYTHONPATH=/app
ENV TOKENIZERS_PARALLELISM=false
ENV CUDA_VISIBLE_DEVICES=""

# Expose Streamlit port
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
