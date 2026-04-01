FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY app.py .
COPY flask_api.py .
COPY bm25_data.pkl .

# Copy React dist folder
COPY heart-health-ai/dist ./heart-health-ai/dist

# Expose port
EXPOSE 7860

# Environment variables must be set by HF Spaces secrets
# Run the app
CMD ["python", "app.py"]
