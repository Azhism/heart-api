#  Dil Ki Baatein: RAG-Based Cardiovascular Health Q&A System

A production-ready **Retrieval-Augmented Generation (RAG) system** for personalized cardiovascular health risk assessment, purpose-built for Pakistani patients. Combines WHO guidelines, Pakistani Hypertension League standards, and localized nutritional data into a conversational health assistant.

**Course:** NLP with Deep Learning | Assignment 3  
**Submission Date:** 5th April 2025  
**Team:** Azhab Safwan Babar && Arham Altaf  
**Domain:** Cardiovascular & Preventive Health (Pakistan)

** Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/azhab/heart-attack-assessment)

---

##  Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [RAG Pipeline](#rag-pipeline)
- [Ablation Study Results](#ablation-study-results)
- [Deployment](#deployment)
- [API Endpoints](#api-endpoints)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features & Problem Statement

### The Problem We Solve
Cardiovascular disease is **the leading cause of death in Pakistan** (~30% of annual fatalities). Yet:
- General-purpose chatbots lack Pakistani dietary knowledge
- Clinical resources are inaccessible to non-specialists
- No platform combines WHO guidelines + Pakistani nutritional data into a conversational interface

**Dil Ki Baatein fills this gap** by enabling any Pakistani user to ask health questions in everyday language and receive **evidence-based, context-aware responses**.

### Core Capabilities
- **Personalized Risk Assessment** Analyzes lifestyle, diet, and symptoms against WHO & Pakistani guidelines (Faithfulness: **82.4%**, Relevancy: **73.5%** on 20-query evaluation)
- **RAG-Based Grounding** Responses grounded in 19 medical PDFs + 31 Pakistani dishes CSV (zero hallucination constraints via prompt engineering)
- **Hybrid Retrieval Pipeline** BM25 (lexical) + Semantic Search (Pinecone) + RRF Fusion + CrossEncoder Re-ranking
- **Automated Quality Assessment** LLM-as-a-Judge evaluation for Faithfulness (claim verification) and Relevancy (semantic matching)
- **Sub-second Latency** End-to-end response in 1.0–1.2 seconds

### Pakistan-Specific Knowledge Base
- **19 Medical PDFs, 524+ pages:**
  - WHO HEARTS Package, Hypertension & CVD Prevention Guidelines
  - 3rd Pakistan Hypertension League (PHL) Guidelines (2018)
  - Pakistan STEPS Survey 2014 epidemiological data
  - ACC/AHA 2019 Primary Prevention Guidelines
  - Emergency Management of Acute Myocardial Infarction
  - South Asian-specific CVD risk papers
  
- **Pakistani Dishes Nutritional Database (31 dishes):**
  - Biryani, Karahi, Nihari, Chapli Kabab, Haleem, etc.
  - Fat, protein, carbs, kcal per 100g
  - Heart-health classification (low/moderate/high-fat)
  - Source: Khan et al., 2018 (University of Agriculture Peshawar)

- **12 Query Domains Supported:**
  - Pakistani Food & Diet | Blood Pressure | Heart Attack Signs | Smoking & Tobacco
  - Family History | Exercise & Lifestyle | Pregnancy & BP | Elderly Health
  - Diagnostic Tests | Medications | Risk Factors | Fats & Oils

### Production-Ready
- **Deployed on HuggingFace Spaces** — Fully accessible, Docker-containerized
- **Flask + Gunicorn** — Production-grade backend with error handling
- **React + TypeScript** — Modern, responsive frontend with real-time feedback
- **Persistent State** — BM25 index serialized; Pinecone vector DB pre-indexed

---

## Architecture

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
    │  • 350+ embedded chunks        │
    └───────────────────────────────┘
```

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| **Frontend** | React 18, TypeScript, Vite, TailwindCSS, shadcn/ui |
| **Backend** | Flask, Gunicorn, Python 3.9+ |
| **ML/RAG** | LangChain, Sentence-Transformers, CrossEncoder |
| **Vector DB** | Pinecone (768-dim embeddings) |
| **LLM Generation** | Groq Llama 3.1 8B (ultra-low latency) |
| **LLM Evaluation** | Groq Llama 3.3 70B (higher reasoning) |
| **Retrieval** | BM25 (lexical) + Semantic (Pinecone) + RRF + CrossEncoder |
| **Embeddings** | BAAI/bge-base-en-v1.5 (768-dimensional) |
| **Re-ranking** | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| **Evaluation Metrics** | LLM-as-a-Judge (Faithfulness + Relevancy) |
| **Deployment** | HuggingFace Spaces (Docker, Python 3.10) |

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

## Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 16+ (for React build)
- Git
- Pinecone account with API key
- Groq Cloud account with API key

### Local Setup (Development)

#### 1. Clone & Environment
```bash
cd "D:\Classess\ITA\Rag_Project\Heart Attack Project"
python -m venv .venv
.\.venv\Scripts\Activate  # Windows PowerShell
# or: source .venv/bin/activate  # Linux/Mac
```

#### 2. Install Python Dependencies
```bash
pip install -r requirements.txt

# Or with Pinecone + Groq (for HF Spaces):
pip install -r requirements_hf.txt
```

#### 3. Set Environment Variables
```bash
# Create .env file or set system variables:
PINECONE_API_KEY=pcsk_xxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxx
PINECONE_INDEX_NAME=heart-risk-300-chunks
```

#### 4. Run Flask Backend
```bash
python flask_api.py

# Server runs on http://localhost:5000
# APIs available at http://localhost:5000/api/assess
```

#### 5. Build & Run React Frontend (Separate Terminal)
```bash
cd heart-health-ai
npm install
npm run dev

# Frontend runs on http://localhost:5173
```

---

## Usage

### Via Live Demo
Visit: **[HuggingFace Spaces](https://huggingface.co/spaces/azhab/heart-attack-assessment)**

1. Enter your health query (e.g., "I smoke 10 cigarettes a day, am I at risk?")
2. Click **"🔍 Assess My Risk"**
3. Wait 3-5 seconds for RAG pipeline
4. View **Assessment**, **Quality Scores**, and **Sources**

### Via API (cURL)
```bash
curl -X POST http://localhost:5000/api/assess \
  -H "Content-Type: application/json" \
  -d { "query": "I eat biryani 4 times a week. Is that bad for my heart?" }
```

### API Response Format
```json
{
  "assessment": "Biryani is high in fat and calories. Eating it 4x/week may increase heart disease risk...",
  "faithfulness": 87.5,
  "relevancy": 92.1,
  "sources": [
    {
      "source": "pakistani_dishes_nutrition.csv",
      "page": 1,
      "text": "Biryani nutrition: 450 cal, 18g fat..." },
    ...
  ]
}
```

---

##  RAG Pipeline Architecture

This section details the complete 4-stage retrieval-augmented generation pipeline that powers Dil Ki Baatein.

### **Stage 1: Chunking & Embedding**

#### Chunking Strategy: RecursiveCharacterTextSplitter
- **Configuration:** chunk_size=300 tokens, overlap=50 tokens (optimized via ablation study)
- **Separator hierarchy:** Double newline (paragraphs) → single newline → sentences → space → characters
- **Rationale:** Achieves highest faithfulness (~84%) by preserving diagnostic thresholds and clinical precision; prevents critical medical information from fragmenting
- **Result:** ~4,413 chunks across 19 PDFs + CSV

#### Embedding Model
- **Model:** BAAI/bge-base-en-v1.5 (768-dimensional)
- **Justification:** Top MTEB benchmark performance; optimal for medical & technical English
- **Pre-computation:** Embeddings computed on Kaggle GPU, upserted to Pinecone at initialization

---

### **Stage 2: Hybrid Retrieval** (Finding 6 Candidate Chunks)

The system implements a three-layer retrieval strategy to maximize recall and avoid over-reliance on any single method.

#### 2a. BM25 Lexical Search
- **Algorithm:** Probabilistic ranking by term frequency (TF) and inverse document frequency (IDF)
- **Strengths:** Captures exact clinical terms (medication names, BP thresholds, disease names)
- **Example:** "biryani" → 5 exact-match chunks from nutritional CSV
- **Returns:** Top-5 candidates

#### 2b. Semantic (Vector) Search via Pinecone
- **Method:** Cosine similarity between query embedding and document embeddings
- **Strengths:** Captures conceptual relationships even without keyword overlap
- **Example:** "Can deep-fried foods harm my heart?" → matches "unsaturated fatty acids and cardiovascular risk"
- **Returns:** Top-5 candidates

#### 2c. Reciprocal Rank Fusion (RRF)
- **Formula:** `score = 1/(rank_bm25 + k) + 1/(rank_semantic + k)` where k=60
- **Effect:** Combines both rankings without score normalization; documents appearing in both lists get boosted scores
- **Output:** Merged top-6 candidates (reduced from 10 through rank-based weighting)

---

### **Stage 3: Re-ranking** (Filtering to Top 3)

#### CrossEncoder Semantic Re-ranking
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Mechanism:** Jointly encodes (query, document) pairs in a single forward pass, producing a semantic relevance score
- **Advantage:** Accounts for full token interactions, unlike bi-encoder similarity
- **Latency:** ~200 ms (pre-computed via distilled MiniLM)
- **Output:** Top-3 highest-scoring chunks passed to LLM

---

### **Stage 4: Generation with Controlled Hallucination**

#### Prompt Engineering: The Production Prompt (V3)
After iterative testing of 3 prompt versions, the final balanced prompt addresses the core challenge: **LLM hallucination**.

**Problem Identified:** LLM would inject parametric medical knowledge (e.g., "quit smoking") even when the retrieved context contained no such guidance. Faithfulness fluctuated 30%–90%.

**Solution: Prompt V3 (Production)**
```
You are a friendly heart health assistant. Answer the patient's question using the guidelines below.

GUIDELINES:
{ctx}

PATIENT: "{q}"

RULES:
- Base your answer ONLY on facts in the guidelines above
- Use warm, friendly, conversational language
- Keep it simple – like talking to a friend
- Do NOT add medical advice not found in the guidelines

FORMAT:
**Assessment:** [1-2 friendly sentences summarizing what the guidelines say]
**Recommendations:** [List recommendations only if they appear in guidelines]
**Disclaimer:** This is based on medical guidelines. Please consult a real doctor for personalized advice.
```

**Results:**
- Faithfulness: Stabilized 77%–100% (previously 30%–90%)
- Relevancy: Remained strong at 74%+
- Output: Readable, friendly, clinically grounded

#### Generation Model
- **LLM:** Groq Llama 3.1 8B (`llama-3.1-8b-instant`)
- **Rationale:** 
  - Ultra-low latency (~600 ms) via Groq's LPU inference hardware
  - Strong instruction-following for structured medical outputs
  - Free API tier suitable for student/research deployment
- **Temperature:** 0.7 (balanced creativity + consistency)
- **Max tokens:** 2,000

---

### **Stage 5: Automated Quality Evaluation**

#### Faithfulness Score (LLM-as-a-Judge)
**What:** Does the LLM answer only use facts from retrieved context? No hallucination?

**Method (2-step):**
1. **Claim Extraction:** Judge LLM (`llama-3.3-70b-versatile`) extracts atomic factual claims from generated response
2. **Verification:** Each claim verified against retrieved chunks; claims from parametric knowledge are marked unsupported

**Formula:**
```
Faithfulness = (Supported Claims) / (Total Claims) × 100%
```

**Example:**
- Query: "I smoke 5 cigarettes a day. How much does this increase my risk?"
- Generated: "Smoking 5 cigarettes daily increases coronary artery disease (CAD) risk. Even 1–4 cigarettes daily increases CAD risk."
- Extracted claims: 2
- Verified in context: 2 ✓
- **Faithfulness: 100%**

#### Relevancy Score (Embedding Similarity)
**What:** Does the answer address the spirit of the user's question?

**Method:**
1. Judge generates 3 potential questions the answer naturally addresses
2. Compute cosine similarity between each question embedding and original query embedding
3. Average the three similarity scores

**Formula:**
```
Relevancy = Mean(similarity(generated_q1, query), similarity(generated_q2, query), similarity(generated_q3, query)) × 100%
```

**Example:**
- Original query: "Is biryani bad for my heart?"
- Answer: "Biryani is high-fat. The guidelines classify high-fat dishes as moderate risk..."
- Generated questions:
  - "How does biryani affect cardiovascular health?"
  - "Is biryani safe to eat regularly?"
  - "What is the nutritional impact of eating biryani?"
- Similarity scores: 0.92, 0.85, 0.88
- **Relevancy: 88.3%**

---

---

## Ablation Study & Model Selection

To identify the optimal configuration, two independent dimensions were ablated: (1) chunking strategy and (2) retrieval pipeline.

### Ablation 1: Chunking Strategy

| Strategy | Total Chunks | Avg Faithfulness | Avg Relevancy | Notes |
|----------|-------|------------------|----------------|-------|
| **Bigger** (1000 / 200 overlap) | 1,488 | 77.2% | 74.7% | Topic bleed: clinical topics conflate within single chunks |
| **Medium** (500 / 100 overlap) | 3,792 | ~82% | ~74% | Balance between topic bleed and context |
| **Current** (300 / 50 overlap) | 4,413 | ~84% | ~73% | **SELECTED**: Highest faithfulness; preserves diagnostic precision |

**Winning Configuration:** `chunk_size=300, overlap=50`  
**Rationale:** Achieves highest faithfulness (~84%) by preventing diagnostic thresholds and clinical details from fragmenting across chunks. Preserves critical medical precision for questions requiring exact guideline references.

---

### Ablation 2: Retrieval Pipeline (4 Versions)

| Version | Configuration | Faithfulness | Relevancy | Analysis |
|---------|---------------|--------------|-----------|----------|
| **V1** | Fixed Chunking + BM25 Only | 97.1% | 70.8% | Keyword-safe, but misses conceptual questions |
| **V2** | Recursive + Semantic Only | 93.3% | **74.4%** | **PRODUCTION SELECTION** — best F+R balance |
| **V3** | Recursive + Hybrid (no rerank) | 80.5% | 89.7% | Naive fusion includes noisy questionnaire fragments |
| **V4** | Recursive + Hybrid + Reranking | 89.7% | 72.4% | Reranking recovers faithfulness but drops relevancy |

**Winning Configuration:** **V2 (Semantic-Only)**  
**Rationale:** 
- 93.3% faithfulness demonstrates high clinical grounding
- 74.4% relevancy indicates strong answer-question alignment
- Simple, maintainable pipeline without reranking latency overhead
- Avoids BM25 noise (e.g., questionnaire fragments) that poison context in V3

---

## Full Evaluation Results (20-Query Benchmark)

### Overall Performance
- **Average Faithfulness:** 82.4%
- **Average Relevancy:** 73.5%
- **Best Query Faithfulness:** 100% (heart attack warning signs, smoking risk, hypertension threshold)
- **Worst Query Faithfulness:** 50% (headaches/dizziness complex reasoning)

### Representative Query Results

| Query Domain | Query | Faithfulness | Relevancy |
|--------------|-------|--------------|-----------|
| **Pakistani Food** | "I eat mutton karahi 3x/week. Is that bad?" | 80.0% | 74.6% |
| **Pakistani Food** | "Is beef nihari safe w/ high cholesterol?" | 60.0% | 55.3% |
| **Blood Pressure** | "My BP is 135/85 average. Do I have hypertension?" | **100.0%** | 76.4% |
| **Heart Attack** | "Warning signs of MI in women?" | **100.0%** | 74.8% |
| **Smoking Risk** | "I smoke 5 cigs daily. How much risk?" | **100.0%** | 81.8% |
| **Medication Authority** | "Can nurse/pharmacist prescribe BP meds?" | 83.3% | 81.8% |
| **Exercise** | "How much exercise per week?" | 80.0% | 74.1% |
| **Complex Synthesis** | "Foods to avoid for CVD prevention?" | 80.0% | 73.5% |

### Performance Pattern Analysis
- **High Faithfulness Queries (90%+):** Direct document matches (smoking risk, hypertension thresholds, warning signs)
- **Lower Faithfulness (<70%):** Multi-document synthesis required (pregnancy + hypertension, stress effects, medication safety)
- **High Relevancy (75%+):** Food queries, lifestyle domains with specific corpus coverage
- **Lower Relevancy (<70%):** Abstract concepts (stress, sedentary risk) lacking granular guideline detail

---

## Latency & Computational Efficiency

| Pipeline Stage | Latency | Notes |
|---|---|---|
| Query embedding (bge-base) | ~80 ms | Client inference |
| BM25 retrieval (top 5) | ~15 ms | In-memory, negligible |
| Pinecone semantic search (top 5) | ~120 ms | Network round-trip |
| RRF fusion | <5 ms | Pure Python, CPU-bound |
| CrossEncoder re-ranking (optional) | ~200 ms | Not used in V2 production |
| **Groq LLM generation** | **~600 ms** | Dominant; LPU-accelerated |
| **TOTAL END-TO-END** | **1.0–1.2 s** | Acceptable for healthcare chatbot |
| Evaluation pipeline (offline) | 8–12 s | Multiple LLM-as-Judge calls, batch-only |

**Key Insight:** Groq's LPU inference dominates the latency budget (55% of total). The 1.0–1.2 second user-facing response time is clinically and UX-appropriate for asynchronous health information retrieval.

---

## Best Model Configuration (Production)

| Dimension | Selected | Justification |
|-----------|----------|---------------|
| **Vectorizer** | BAAI/bge-base-en-v1.5 | Top MTEB retrieval performance; 768-dim richness for medical text |
| **Chunking** | Recursive 300 tokens / 50 overlap | Highest faithfulness (~84%); prevents diagnostic threshold fragmentation |
| **Retrieval** | Semantic Search (V2) | 93.3% faithfulness, 74.4% relevancy; avoids BM25 noise without reranking overhead |
| **Vector DB** | Pinecone (cloud) | Serverless, auto-scaling; 768-dim cosine metric; index="heart-risk-300-chunks" |
| **Keyword Fallback** | BM25 (serialized) | Exact clinical term matching; .pkl file loaded at runtime |
| **Re-ranking** | Optional CrossEncoder | Used in V4 ablation; skipped in V2 production for latency savings |
| **Generation LLM** | Groq Llama 3.1 8B | Ultra-low latency (~600 ms); strong instruction-following; free API tier |
| **Evaluation Judge** | Groq Llama 3.3 70B | Higher reasoning for claim verification; offline evaluation only |
| **Prompt Version** | V3 (Balanced) | Stabilized faithfulness (77%–100% vs 30%–90% in V1); readable + grounded |
| **Temperature** | 0.7 | Balanced: avoids both randomness and determinism in clinical language |
| **Max Tokens** | 2,000 | Sufficient for structured "Assessment + Recommendations + Disclaimer" format |

### Design Rationale
For a **patient-facing medical chatbot**, faithfulness is the primary constraint. Incorrect recommendations risk patient harm. The V2 configuration (93.3% F, 74.4% R) meets both clinical responsibility *and* practical utility without unnecessary latency complexity.

---

## Deployment on HuggingFace Spaces

### Live URL
**https://huggingface.co/spaces/azhab/heart-attack-assessment**

### Manual Push to HF Spaces
```bash
cd "heart-health-ai/heart-attack-assessment"
git add app.py requirements.txt dist/
git commit -m "Update: [describe changes]"
git push https://[TOKEN]@huggingface.co/spaces/azhab/heart-attack-assessment.git main
```

### Auto Rebuild Pipeline
- HF Spaces detects push → rebuilds Docker container (2-5 min)
- Models load at startup (module-level `init_models()` call)
- App available at https://huggingface.co/spaces/azhab/heart-attack-assessment

### Critical Deployment Components

**1. Dockerfile**
```dockerfile
FROM python:3.10-slim
WORKDIR /app

# Copy backend
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy frontend (essential!)
COPY dist/ ./heart-health-ai/dist/
COPY app.py .
COPY bm25_data.pkl .

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
```

**2. Module-Level Initialization** (Critical for Gunicorn)
```python
# app.py - at module level, BEFORE if __name__ block
# This ensures models load when Gunicorn imports the app
init_models()  # Loads SentenceTransformer, CrossEncoder, LLMs, Pinecone, BM25
```

**3. Static Asset Serving**
```python
# Flask serves React build from dist/ folder
@app.route('/')
def serve_index():
    return send_from_directory('dist', 'index.html')
```

---

## API Endpoints

### POST `/api/assess`
**Assess heart risk based on user query**

**Request:**
```json
{
  "query": "I smoke 10 cigarettes a day, am I at risk?"
}
```

**Response:**
```json
{
  "assessment": "Smoking 10 cigarettes daily significantly increases...",
  "faithfulness": 85.2,
  "relevancy": 89.7,
  "sources": [
    {
      "source": "smoking_cardiovascular_risk.pdf",
      "page": 3,
      "text": "Cigarette smoking causes..."
    }
  ]
}
```

### GET `/api/health`
**Health check endpoint**

**Response:**
```json
{
  "status": "healthy",
  "models": "loaded"
}
```

---

##  Performance Metrics

### Speed
- **Frontend → API:** ~50ms (network)
- **Retrieval:** ~200ms (Pinecone + BM25)
- **Reranking:** ~150ms (CrossEncoder)
- **Generation:** ~2-3s (Groq LLM)
- **Total E2E:** ~3-5 seconds per query

### Accuracy
- **Faithfulness (V4):** 87.3% (claims supported)
- **Relevancy (V4):** 91.2% (answer matches query)
- **Sources Found:** 100% (all queries have relevant documents)

### Scale
- **Knowledge Base:** 350+ chunks (medical + nutrition)
- **Embedding Dim:** 768 (Pinecone)
- **Model:** LLama 3.1 8B (fast) + 70B judge (accurate)

---

## Troubleshooting

### Common Issues

#### ❌ Blank Page on HF Spaces
**Cause:** React assets not served  
**Fix:**
```bash
# Ensure dist/ folder is copied to nested repo
npm run build
cp -r dist/ heart-health-ai/heart-attack-assessment/
git add heart-health-ai/dist/
```

#### ❌ Models Not Loading
**Cause:** `init_models()` not called at module level  
**Fix:**
```python
# app.py - ADD THIS LINE at module level (before if __name__)
init_models()  # Gunicorn-compatible initialization
```

#### ❌ API Returns 503 Error
**Cause:** Models still loading or API key invalid  
**Fix:**
- Check HF Logs for "✅ Models loaded successfully!"
- Verify `PINECONE_API_KEY` and `GROQ_API_KEY` in env
- Wait 2-5 min for container rebuild

#### ❌ Pinecone Connection Error
**Cause:** Missing or invalid API key  
**Fix:**
```bash
# Verify in HF Spaces settings → Secrets
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=heart-risk-300-chunks
```

#### ❌ CORS Errors in Browser Console
**Cause:** Missing CORS middleware  
**Fix:**
```python
from flask_cors import CORS
CORS(app)  # Enable all routes
```

## Knowledge Base & Reproducibility

### Document Corpus (19 PDFs, 524+ pages)

| # | Document | Relevance |
|---|----------|-----------|
| 1-2 | WHO HEARTS Package, Hypertension & CVD Prevention | Core CVD prevention framework, BP thresholds |
| 3 | WHO Diet, Nutrition & Physical Activity Guidelines | Dietary fat, sodium, sugar thresholds |
| 4 | 3rd Pakistan Hypertension League (PHL) Guidelines 2018 | Pakistan-specific BP diagnosis and management |
| 5 | Pakistan STEPS Survey 2014 | National prevalence data for CVD risk factors |
| 6 | National Action Framework for NCDs 2021–30 | Government policy and prevention targets |
| 7-8 | Pakistan Heart Journal + Trans Fatty Acids Papers (AKU) | Dietary guidance, vanaspati ghee prevalence |
| 9 | Caring For Your Heart (AKUH) | Patient-facing risk assessment questionnaire |
| 10 | Emergency Management of Acute Myocardial Infarction | MI protocols and clinical logic |
| 11 | ACC/AHA 2019 CVD Prevention Guidelines | Primary prevention, exercise, statin thresholds |
| 12 | Heart Attack Warning Signs | Symptom awareness for laymen |
| 13 | Frequency of ACS in Pakistani Patients | Atypical presentations in Pakistani population |
| 14-15 | South Asia CVD Risk Papers | South Asian-specific risk multipliers, diaspora epidemiology |
| 16 | UK South Asians CVD Risk | Hypertension treatment decision logic |
| 17-19 | Additional Clinical References | Supplementary guidelines and protocols |

### Pakistani Dishes Nutritional Database (31 dishes)

**Source:** Khan et al., 2018 (University of Agriculture Peshawar)  
**Columns:** Dish Name, Fat %, Protein %, Carbs %, Kcal/100g  
**Usage:** Direct lookup for food-related queries; classification into low-fat (<5%), moderate (5–15%), high-fat (>15%)

**Sample Dishes:** Biryani, Karahi, Nihari, Chapli Kabab, Haleem, Samosa, Pakora, and 24 more

---

### System Configuration for Reproducibility

**Python Environment:**
- Python 3.10+
- LangChain 0.1.x (text splitters, BM25 retriever, document loaders)
- sentence-transformers 2.x (BAAI/bge-base-en-v1.5)
- pinecone-client 3.x (cloud, serverless index)
- groq 0.x (llama-3.1-8b-instant, llama-3.3-70b-versatile)
- rank_bm25 0.2.2 (BM25 implementation)
- flask 2.x + flask_cors
- pandas, fitz (PyMuPDF), scikit-learn

**Preprocessing Steps:**
1. Load all 19 PDF documents using PyPDFLoader
2. Apply RecursiveCharacterTextSplitter: `chunk_size=500, overlap=100`
3. Load CSV nutritional data; convert rows to Document objects
4. Generate embeddings in batch mode using BAAI/bge-base-en-v1.5
5. Upsert embeddings to Pinecone (768 dims, cosine metric)
6. Build BM25 index from document list; serialize to .pkl

**Pinecone Index:**
- Index name: `heart-risk-300-chunks`
- Dimension: 768
- Metric: cosine
- Cloud: AWS (Starter tier)
- Namespace: default

---

## � Limitations & Future Work

### Current Limitations

1. **Corpus Coverage Gap for Pakistani Cuisine**
   - Nutritional CSV provides dish-level data, but no PDF document offers clinical commentary on Pakistani dietary patterns
   - Future: Incorporate Pakistan-specific dietary guidelines and AKU nutrition guidelines as PDF sources

2. **Faithfulness Variance on Complex Multi-Document Queries**
   - Queries requiring synthesis across multiple documents (stress, sedentary risk, pregnancy + hypertension) show faithfulness 50–70%
   - Higher surface area for unsupported claims when synthesizing
   - Future: Multi-hop retrieval strategy or query decomposition

3. **No Urdu Language Support**
   - Current system operates in English only
   - Pakistani patients may prefer Urdu interface
   - Future: Multilingual embedding models (e.g., Qwen-7B, Aya-23) + multilingual judge LLMs

4. **Static Corpus Without Auto-Refresh**
   - Medical guidelines are updated periodically
   - No automatic mechanisms to detect and re-index new guidelines
   - Future: Periodic batch re-indexing; monitoring of guideline updates

5. **Limited Tone Personalization**
   - Prompt V3 enforces friendly tone but cannot adapt to individual patient preferences
   - Future: Persona-aware prompt templating based on user demographics

6. **Evaluation Metrics Limited to Binary Grounding**
   - Faithfulness only checks if claims are explicitly supported
   - Doesn't detect harmful omissions (e.g., failing to mention drug contraindications
)
   - Future: Negative claim checking; active learning from false negatives

### Recommended V5 Configuration (Future)

Based on ablation findings, a hypothetical V5 combining:
- Recursive 300-char chunking (proven ~84% faithfulness)
- Semantic retrieval only (V2)
- **+ CrossEncoder re-ranking** (added)

Is predicted to achieve:
- Faithfulness: >92% (recovered via filtering)
- Relevancy: >75% (maintained)
- Should be evaluated in future work before production deployment

---

## 🤝 Contributing

### To Add New Knowledge
1. Add PDF to `PDFs/` folder
2. Re-run notebook cells for embedding & Pinecone upload
3. Update `requirements.txt` if needed
4. Push changes to HF Spaces

### To Improve RAG Quality
- Adjust chunk size (currently 300 chars, 50 overlap)
- Modify reranking threshold
- Fine-tune LLM prompts in `Assessment.tsx`


---

---

## Authors & Attribution

**Team Members:**
- **Azhab Safwaan Babar**
- **Arham Altaf**

**Course:** NLP with Deep Learning   
**Assignment:** Assignment 3 — Mini-Project 1  
**Institution:** Institute of Bussines Administrattoo Karachi  
**Submission Date:** 5th April 2025  
**Domain:** Cardiovascular & Preventive Health (Pakistan)

**Key Resources & Acknowledgements:**
- **WHO HEARTS Package & Guidelines** — Core prevention framework
- **3rd Pakistan Hypertension League Guidelines** — National standards
- **Khan et al., 2018** — Pakistani nutrition data source
- **Groq Inc.** — Ultra-low latency LLM inference
- **Pinecone** — Serverless vector database
- **HuggingFace** — Deployment infrastructure
- **LangChain Community** — RAG pipeline tooling
- **sentence-transformers** — BAAI embedding models

---

## 📖 References

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

---

## Privacy & Medical Disclaimer

 **CRITICAL:** This system:
- **Does NOT provide medical diagnosis** — It retrieves and summarizes medical guidelines only
- **Is for educational purposes only** — Designed for learning about cardiovascular health
- **Should NOT replace professional medical advice** — Always consult a qualified healthcare provider
- **Requires doctor consultation for any health concerns** — Patient safety is paramount
- **Not liable for adverse outcomes** — Users assume full responsibility for medical decisions

**Data Privacy:**
- No user queries are logged or stored beyond the current session
- No personal health information (PHI) is retained
- The system does not connect to any patient medical records
- All API calls are between client and Groq/Pinecone infrastructure only

**Use Case:**
✅ Learning about WHO cardiovascular prevention guidelines  
✅ Understanding Pakistani dietary impacts on heart health  
✅ General awareness about cardiac risk factors  
✅ Educational assessment tool for coursework  
❌ NOT for clinical decision-making  
❌ NOT for patient diagnosis or treatment planning  
❌ NOT a replacement for doctor consultation

