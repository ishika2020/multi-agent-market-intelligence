FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agentic_market_intel/ agentic_market_intel/
COPY main.py .
COPY dashboard.py .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn agentic_market_intel.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
