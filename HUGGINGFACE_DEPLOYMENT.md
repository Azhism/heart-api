# Heart Health AI - Hugging Face Spaces Deployment Guide

## Overview
This guide will help you deploy the Heart Health AI application to Hugging Face Spaces using Gradio.

## Architecture
- **Frontend**: Gradio interface (Python-based, runs on HF Spaces)
- **Backend**: Flask API (requires separate hosting)
- **AI Pipeline**: RAG-powered assessment using Groq LLM + Pinecone

## Step 1: Prepare Your Flask Backend

The Gradio app needs a running Flask API. You have two options:

### Option A: Deploy Flask to a Cloud Service
Deploy `flask_api.py` to:
- **Render.com** (free tier available)
- **Heroku** (alternative)
- **Your own server**
- **Hugging Face Spaces** (Docker-based Space)

### Option B: Use Local Backend (Testing Only)
Keep the Flask API running locally and use `http://localhost:5000` when testing.

## Step 2: Create a Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Fill in:
   - **Space name**: `heart-health-ai` (or your choice)
   - **License**: Apache 2.0
   - **Space SDK**: Gradio
   - **Visibility**: Public

## Step 3: Upload Files to Your Space

Clone the Space repo and add these files:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/heart-health-ai
cd heart-health-ai
```

### Required Files:

#### 1. `app.py` (Main Gradio App)
Copy the content from `app_hf.py` in this repo.

#### 2. `requirements.txt`
```
gradio==4.32.1
requests==2.31.0
python-dotenv==1.0.0
```

#### 3. `.env` (Optional - for API configuration)
```
API_BASE_URL=https://your-flask-backend-url.com
```

#### 4. `README.md` (Space description)
Add a description of your Space.

### Push to Hugging Face:
```bash
git add .
git commit -m "Initial Gradio app deployment"
git push
```

## Step 4: Configure Backend URL

### If Backend is Running Locally:
For testing purposes, the app will try to connect to `http://localhost:5000`.

### If Backend is Hosted Remotely:
1. Go to your Space settings
2. Add environment variable:
   - **Key**: `API_BASE_URL`
   - **Value**: `https://your-backend-url.com`
3. Restart the Space

## Step 5: Test Your Deployment

1. Your Gradio app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/heart-health-ai`
2. Ensure your Flask backend is accessible from the Space
3. Try submitting a test assessment

## Common Issues & Solutions

### "Cannot connect to API"
- ✅ Check that Flask backend is running
- ✅ Verify the API URL in `.env` or settings
- ✅ Ensure Flask API has CORS enabled
- ✅ Check firewall/network settings

### API Timeout
- Increase the timeout in `app_hf.py` (currently 120 seconds)
- The RAG pipeline can take time to process

### Models Not Loading
- Ensure your Flask backend can access Pinecone and Groq APIs
- Verify API keys in Flask environment

## File Structure

```
your-space/
├── app.py                 # Main Gradio application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (optional)
├── README.md             # Space description
└── .gitignore            # Git ignore file
```

## Deploying the Flask Backend to HF Spaces (Advanced)

If you want both frontend and backend in HF Spaces:

1. Create a **Docker-based Space**
2. Add `Dockerfile` with Flask setup
3. Include all backend dependencies
4. Use internal communication or public endpoint

### Dockerfile Example:
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "flask_api.py"]
```

## Next Steps

1. **Deploy Flask API** to a cloud service
2. **Create HF Space** with Gradio
3. **Upload files** and configure API URL
4. **Test** the deployment
5. **Share** your Space with others!

---

**For support**: Check Hugging Face Spaces documentation at https://huggingface.co/docs/hub/spaces
