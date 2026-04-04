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
COPY flask_app.py .
COPY bm25_data.pkl .

# Copy React dist files (without favicon.ico as it's not always present)
COPY index.html index*.css index*.js placeholder.svg robots.txt ./heart-health-ai/dist/

# Expose port
EXPOSE 7860

# Environment variables must be set by HF Spaces secrets
# Run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "flask_app:app"]
