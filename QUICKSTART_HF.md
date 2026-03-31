# Quick Start: Deploy to Hugging Face Spaces

## 🚀 Fast Track Deployment (5 Minutes)

### Step 1: Create a Hugging Face Space
```
1. Go to https://huggingface.co/spaces/create
2. Name: "heart-health-ai"
3. License: "Apache 2.0"
4. Space SDK: "Gradio"
5. Visibility: "Public"
```

### Step 2: Clone the Space
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/heart-health-ai
cd heart-health-ai
```

### Step 3: Add Files
Copy these files from this repo to your Space:
- `app.py` ← Copy from `app_hf.py` in this repo
- `requirements.txt` ← Copy `requirements_hf.txt` from this repo

Or run:
```bash
# From the Space directory
curl https://raw.githubusercontent.com/YOUR_REPO/main/app_hf.py > app.py
curl https://raw.githubusercontent.com/YOUR_REPO/main/requirements_hf.txt > requirements.txt
```

### Step 4: Configure Backend
1. Go to **Space Settings** → **Variables and Secrets**
2. Add: `API_BASE_URL=https://your-flask-backend.com`
3. Save and restart Space

### Step 5: Deploy
```bash
cd your-space-directory
git add .
git commit -m "Add Gradio frontend"
git push
```

Your Space will automatically build and deploy! 🎉

---

## 📝 Environment Variables

Set these in your HF Space settings:

| Variable | Value | Notes |
|----------|-------|-------|
| `API_BASE_URL` | `https://your-api.com` | Your Flask backend URL |

---

## ✅ Verification Checklist

- [ ] Space created on Hugging Face
- [ ] `app.py` and `requirements.txt` uploaded
- [ ] Backend API URL configured
- [ ] Space build completed
- [ ] Can submit test assessment
- [ ] Results display correctly

---

## 🔗 Useful Links

- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- Gradio Docs: https://gradio.app
- My Space: https://huggingface.co/spaces/YOUR_USERNAME/heart-health-ai

---

## 💡 Tips

- Keep Flask backend running at all times
- Monitor Space logs for errors
- Update `API_BASE_URL` if backend URL changes
- Test locally first before deploying

---

## 🆘 Troubleshooting

**Q: "Cannot connect to API"**
A: Check that your Flask backend is running and the URL is correct.

**Q: "Timeout error"**
A: The RAG model takes time. Wait up to 2 minutes for first response.

**Q: "GradIO app not loading"**
A: Check Space build logs for errors. Rebuild the Space if needed.

For more help: See `HUGGINGFACE_DEPLOYMENT.md`
