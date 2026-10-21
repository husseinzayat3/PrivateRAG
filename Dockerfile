FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Downloaded embedding models live on the storage volume, so they survive rebuilds
    HF_HOME=/app/storage/huggingface \
    DATA_DIR=/app/data \
    CHROMA_DB_DIR=/app/storage/chroma_db \
    CHAT_DB_PATH=/app/storage/chat_history.db \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# sentence-transformers pulls in PyTorch; the CPU build avoids several GB of CUDA wheels
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data /app/storage \
    && chown -R app:app /app/data /app/storage
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py"]
