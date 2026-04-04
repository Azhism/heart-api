# Dil Ki Baatein: RAG-Based Cardiovascular Health Q&A System

A production-ready **Retrieval-Augmented Generation (RAG) system** for personalized cardiovascular health risk assessment, purpose-built for Pakistani patients. Combines WHO guidelines, Pakistani Hypertension League standards, and localized nutritional data into a conversational health assistant.

**Course:** NLP with Deep Learning | Assignment 3  
**Submission Date:** 5th April 2025  
**Team:** Azhab Safwan Babar & Arham Altaf  
**Domain:** Cardiovascular & Preventive Health (Pakistan)

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/azhab/heart-attack-assessment)

---

##  Quick Links

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation--setup)
- [Usage](#usage)
- [API](#api-endpoints)
- [Performance](#performance)
- [References](#references)

---

##  Features

###  The Problem We Solve
Cardiovascular disease is the **leading cause of death in Pakistan** (~30% of annual fatalities). Dil Ki Baatein fills the gap between inaccessible clinical resources and general-purpose chatbots by combining WHO guidelines + Pakistani nutritional data into a conversational interface.

###  What It Does
- **Personalized Risk Assessment** — Analyzes lifestyle, diet, symptoms against WHO & Pakistani guidelines
- **Evidence-Grounded Responses** — 19 medical PDFs + 31 Pakistani dishes nutritional database
- **Automated Quality Metrics** — LLM-as-a-Judge evaluation: **82.4% Faithfulness**, **73.5% Relevancy**
- **Sub-second Latency** — End-to-end response in 1.0–1.2 seconds (Groq LPU-accelerated)

###  Knowledge Coverage
- **19 Medical PDFs** (524+ pages): WHO HEARTS, Pakistan Hypertension League, ACC/AHA, emergency MI protocols
- **31 Pakistani Dishes**: Nutritional profiles (fat, protein, carbs, kcal) with heart-health classifications
- **12 Query Domains**: Food, BP, heart attack signs, smoking, family history, exercise, pregnancy, diagnostics, meds, risk factors, stress, obesity

###  Production Status
- ✅ **Deployed on HuggingFace Spaces** (Docker, auto-scaling)
- ✅ **React + Flask** backend with CORS
- ✅ **BM25 index serialized** for instant startup
- ✅ **Zero exposed secrets** (API keys via environment variables)

---

##  Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                │
│          Assessment.tsx + Components + UI               │
└────────────────────────┬────────────────────────────────┘
                         │
                    fetch(/api/assess)
                         │
         ┌───────────────▼────────────────┐
         │   Flask Backend (Gunicorn)     │
         │      app.py on port 7860       │
         └───────────────┬────────────────┘
                         │
         ┌───────────────┴────────────────┐
         │                                │
    ┌────▼─────┐                   ┌─────▼─────┐
    │ Retrieval │                   │ Generation │
    │ Pipeline  │                   │ Pipeline   │
    └────┬─────┘                   └─────┬─────┘
         │                               │
    ┌────┴────────────────┐          ┌──▼──────┐
    │  Hybrid Search:      │          │ LLM     │
    │  ├─ BM25 (FTS)       │          │ (Groq)  │
    │  ├─ Semantic         │          │ ├─ Llama │
    │  │  (Pinecone)       │          │ └─ Judge │
    │  └─ Rerank           │          │   (70B)  │
    │     (CrossEncoder)   │          └──────────┘
    └────┬────────────────┘
         │
    ┌────┴──────────────────────────┐
    │    Knowledge Base (Pinecone)   │
    │  • Medical guidelines (PDFs)   │
    │  • Pakistani dishes nutrition  │
    │  • 4,413 embedded chunks       │
    └───────────────────────────────┘
```

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | React 18 + TypeScript + Vite + TailwindCSS |
| **Backend** | Flask + Gunicorn (Python 3.10) |
| **Embeddings** | BAAI/bge-base-en-v1.5 (768-dim) |
| **Vector DB** | Pinecone (serverless, cosine metric) |
| **Retrieval** | BM25 + Semantic Search + RRF fusion |
| **LLM Generation** | Groq Llama 3.1 8B (600ms latency) |
| **LLM Evaluation** | Groq Llama 3.3 70B (faithfulness judge) |
| **Deployment** | HuggingFace Spaces (Docker) |

---

## 📁 Project Structure

```
Heart Attack Project/
│
├── README.md                            # This file
├── requirements.txt                     # Python dependencies (Flask, LangChain, etc.)
├── flask_app.py                         # Production Flask backend (deployed to HF Spaces)
├── ITA_Full_Project (1).ipynb           # Complete development notebook (56 documented cells)
│                                         # Covers: PDF loading → chunking → embedding →
│                                         # retrieval pipeline → LLM generation → evaluation
│
├── bm25_data.pkl                        # Serialized BM25 index (loaded at runtime)
├── Dockerfile                           # Container config for HF Spaces deployment
├── upload_to_hf.py                      # Automated deployment script
│
├── heart-health-ai/                     # React frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Index.tsx                # Landing page
│   │   │   ├── Assessment.tsx           # Main assessment interface
│   │   │   └── NotFound.tsx             # 404 fallback
│   │   ├── components/
│   │   │   ├── AssessmentResult.tsx     # Result display component
│   │   │   ├── MetricsPanel.tsx         # Faithfulness/Relevancy score cards
│   │   │   ├── RiskResultCard.tsx       # Styled risk assessment output
│   │   │   ├── AnalyzingOverlay.tsx     # Loading state animation
│   │   │   ├── TypingIndicator.tsx      # Streaming response animation
│   │   │   ├── MedicalIcons.tsx         # SVG icons for domains
│   │   │   ├── TrustElements.tsx        # Source attribution display
│   │   │   ├── NavLink.tsx              # Navigation components
│   │   │   └── ui/                      # shadcn/ui component library (20+ components)
│   │   ├── hooks/
│   │   │   └── use-mobile.tsx, use-toast.ts
│   │   ├── lib/utils.ts                 # Utility functions
│   │   ├── App.tsx                      # Main router & layout
│   │   ├── main.tsx                     # Entry point
│   │   ├── index.css, App.css           # Tailwind + global styles
│   │   └── vite-env.d.ts                # Type definitions
│   │
│   ├── dist/                            # Production React build (auto-generated)
│   ├── vite.config.ts                   # Vite bundler config
│   ├── tailwind.config.ts               # Tailwind CSS theming
│   ├── tsconfig.json, tsconfig.app.json # TypeScript configs
│   ├── eslint.config.js, postcss.config.js
│   ├── playwright.config.ts, vitest.config.ts
│   ├── bun.lockb                        # Bun package manager lock
│   ├── package.json                     # Frontend dependencies
│   └── README.md                        # Frontend-specific documentation
│
└── PDFs/                                # Knowledge Base Source Documents
    ├── WHOHEARTS_Package.pdf
    ├── WHO_Hypertension_CVD_Prevention.pdf
    ├── Pakistan_Hypertension_League_Guidelines_2018.pdf
    ├── ACC_AHA_2019_CVD_Prevention.pdf
    ├── Emergency_Management_AMI.pdf
    ├── Pakistan_STEPS_Survey_2014.pdf
    ├── South_Asian_CVD_Risk_Paper.pdf
    ├── and 12 more clinical PDFs (524+ pages total)
    │
    └── dishes_nutrition.csv             # Pakistani nutrition data (31 dishes)
                                          # Columns: Dish, Fat%, Protein%, Carbs%, Kcal/100g
```

---

##  Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 16+ (for React)
- Pinecone API key
- Groq Cloud API key

### Local Development

```bash
# 1. Clone & activate environment
cd "Heart Attack Project"
python -m venv .venv
.\.venv\Scripts\Activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
$env:PINECONE_API_KEY="pcsk_..."
$env:GROQ_API_KEY="gsk_..."
$env:PINECONE_INDEX_NAME="heart-risk-300-chunks"

# 4. Run Flask backend
python flask_app.py
# Available at http://localhost:5000

# 5. Run React frontend (separate terminal)
cd heart-health-ai
npm install
npm run dev
# Available at http://localhost:5173
```

---

##  Usage

### Via Live Demo
Visit: [HuggingFace Spaces](https://huggingface.co/spaces/azhab/heart-attack-assessment)

### Via API (cURL)
```bash
curl -X POST http://localhost:5000/api/assess \
  -H "Content-Type: application/json" \
  -d '{"query": "I eat biryani 4 times a week. Is that bad for my heart?"}'
```

### Response Format
```json
{
  "assessment": "Biryani is high in fat and calories...",
  "faithfulness": 87.5,
  "relevancy": 92.1,
  "sources": [
    {
      "source": "pakistani_dishes_nutrition.csv",
      "page": 1,
      "text": "Biryani nutrition: 450 cal, 18g fat..."
    }
  ]
}
```

---

##  RAG Pipeline (Quick Overview)

The system implements a **4-stage pipeline**:

1. **Chunking & Embedding** — Documents split into 300-token chunks; embedded via BAAI/bge-base-en-v1.5
2. **Hybrid Retrieval** — BM25 (lexical) + Pinecone (semantic) + RRF fusion → top-6 candidates
3. **Re-ranking** — CrossEncoder filters to top-3 chunks
4. **Generation** — Groq Llama 3.1 8B generates answer with constrained prompt (no hallucination)
5. **Evaluation** — Offline LLM-as-a-Judge assesses faithfulness & relevancy

**Production Configuration:**
- Chunking: 300 tokens / 50 overlap (highest faithfulness ~84%)
- Retrieval: Semantic search only (V2) — 93.3% faithfulness, 74.4% relevancy
- Prompt: V3 (Balanced) — stabilized faithfulness 77%–100%

*Full ablation studies, evaluation results (20 queries), and technical details in project report*

---

##  API Endpoints

### POST `/api/assess`
Assess heart risk based on user query.

**Request:**
```json
{"query": "I smoke 10 cigarettes a day, am I at risk?"}
```

**Response:**
```json
{
  "assessment": "Smoking 10 cigarettes daily significantly increases...",
  "faithfulness": 85.2,
  "relevancy": 89.7,
  "sources": [...]
}
```

### GET `/api/health`
Health check endpoint.

---

##  Performance

- **Latency:** 1.0–1.2 seconds end-to-end
  - Embeddings: 80ms
  - Retrieval: 135ms
  - LLM generation: 600ms (Groq LPU)
  
- **Quality:** 20-query benchmark
  - Faithfulness: 82.4% avg (50%–100% range)
  - Relevancy: 73.5% avg (55%–82% range)

- **Scale:**
  - Knowledge base: ~4,413 chunks
  - Embedding dimension: 768 (Pinecone)
  - Models: Llama 3.1 8B (fast) + 70B (evaluation)

---

##  Troubleshooting

| Issue | Solution |
|-------|----------|
| **Blank page on HF Spaces** | Check HF Spaces logs; ensure dist/ copied |
| **Models not loading** | Verify `PINECONE_API_KEY` and `GROQ_API_KEY` in HF secrets |
| **API returns 503** | Wait 2-5 min for container rebuild; check logs |
| **CORS errors in browser** | Backend should have `CORS(app)` enabled |

---

##  Privacy & Medical Disclaimer

⚠️ **IMPORTANT:**
- **Does NOT provide medical diagnosis** — Educational purposes only
- **Should NOT replace professional medical advice** — Consult a real doctor
- **No user data stored** — Queries not logged beyond session
- **Users assume full responsibility** for health decisions

---

## 📚 References

[1] Khan, I., Yasmeen, F., Ahmad, J., Abdulla, A., ud Din, Z., Iqbal, Z., & Iqbal, M. (2018). Developing a meal-planning exchange list for commonly consumed Pakistani dishes. Pakistan Journal of Scientific and Industrial Research. University of Agriculture Peshawar.

[2] Arnett, D. K., et al. (2019). 2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular Disease: Executive Summary. Journal of the American College of Cardiology, 74(10), 1376–1414.

[3] Pakistan Hypertension League. (2018). 3rd Pakistan Hypertension Guidelines. Pakistan Hypertension League.

[4] World Health Organization. (2021). HEARTS Technical Package for Cardiovascular Disease Management in Primary Health Care. WHO.

[5] American Heart Association. (2023). What Are the Warning Signs of Heart Attack? AHA Patient Education.

[6] Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems, 33, 9459–9474.

[7] Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4), 333–389.

[8] Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. Proceedings of EMNLP 2019.

[9] Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. Proceedings of SIGIR 2009.

[10] Groq Inc. (2024). Groq LPU Inference Engine. https://groq.com

[11] Pinecone Systems Inc. (2024). Pinecone Vector Database Documentation. https://docs.pinecone.io

---

## ✍️ Authors

**Azhab Safwan Babar** | **Arham Altaf**   
NLP with Deep Learning | Assignment 3 | April 2025


