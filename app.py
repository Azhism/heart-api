"""
Combined Flask App: Serves React Frontend + Heart Risk Assessment API
Deployment: HF Spaces (port 7860) or Local (port 5000)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from pinecone import Pinecone
import pickle
import os
import re
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURATION - USE ENVIRONMENT VARIABLES
# ============================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "heart-risk-rag-bigger-context")

# Validate required keys are set
if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("ERROR: Set PINECONE_API_KEY and GROQ_API_KEY environment variables")

app = Flask(__name__, static_folder='heart-health-ai/dist', static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['JSON_SORT_KEYS'] = False

# ============================================
# GLOBAL MODELS (CACHED)
# ============================================
embedding_model = None
reranker = None
llm = None
judge_llm = None
pinecone_index = None
bm25_retriever = None


def init_models():
    """Initialize ML models and retrievers"""
    global embedding_model, reranker, llm, judge_llm, pinecone_index, bm25_retriever

    print("Loading models...")
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=2000
    )
    judge_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=2000
    )

    # Load Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Load BM25
    pkl_path = os.path.join(os.path.dirname(__file__), "bm25_data.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    bm25_retriever = BM25Retriever.from_documents(data["docs"], k=5)

    print("✅ Models loaded successfully!")


# ============================================
# RETRIEVAL FUNCTIONS
# ============================================
def semantic_search(query, k=5):
    """Search by semantic similarity using Pinecone"""
    query_vector = embedding_model.encode(query).tolist()
    results = pinecone_index.query(
        vector=query_vector,
        top_k=k,
        include_metadata=True
    )
    return [
        Document(
            page_content=match["metadata"]["text"],
            metadata={
                "source": match["metadata"]["source"],
                "page": match["metadata"]["page"]
            }
        )
        for match in results["matches"]
    ]


def hybrid_search(query, k=5):
    """Combine BM25 + semantic search with RRF"""
    bm25_results = bm25_retriever.invoke(query)
    semantic_results = semantic_search(query, k=k)

    scores = {}
    all_docs = {}

    for rank, doc in enumerate(bm25_results):
        doc_id = doc.page_content[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 60)
        all_docs[doc_id] = doc

    for rank, doc in enumerate(semantic_results):
        doc_id = doc.page_content[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + 60)
        all_docs[doc_id] = doc

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_docs[doc_id] for doc_id, _ in sorted_docs[:k]]


def rerank_docs(query, docs, top_k=3):
    """Rerank documents using CrossEncoder"""
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]


# ============================================
# EVALUATION FUNCTIONS
# ============================================
def evaluate_faithfulness(query, context, answer):
    """Evaluate how faithful the answer is to the context"""
    
    # Extract claims from answer
    claims_prompt = f"""Extract all factual claims from this answer as a numbered list.
Each claim should be a single, verifiable statement.

Answer: {answer}

Return ONLY a numbered list of claims, nothing else."""

    claims_response = judge_llm.invoke(claims_prompt)
    claims_text = claims_response.content

    # Verify against context
    verify_prompt = f"""You are a fact checker. Given the following context from medical guidelines,
verify each claim and return a JSON with this exact format.
Use ONLY true or false for supported field.

Context:
{context[:3000]}

Claims to verify:
{claims_text}

Return ONLY the JSON. Format:
{{"claims": [{{"claim": "claim text", "supported": true/false}}]}}"""

    verify_response = judge_llm.invoke(verify_prompt)
    response_text = verify_response.content

    try:
        json_text = re.search(r'\{.*\}', response_text, re.DOTALL).group()
        json_text = json_text.replace('"supported": not verified', '"supported": false')
        json_text = json_text.replace('"supported": Not verified', '"supported": false')
        json_text = json_text.replace('"supported": "true"', '"supported": true')
        json_text = json_text.replace('"supported": "false"', '"supported": false')

        verification = json.loads(json_text)
        claims = verification['claims']

        supported = sum(1 for c in claims if isinstance(c.get('supported'), bool) and c['supported'])
        total = len(claims)
        score = (supported / total) * 100 if total > 0 else 0

        return round(score, 1)

    except:
        return 0


def evaluate_relevancy(query, answer):
    """Evaluate relevancy of answer to query"""
    answer_vector = embedding_model.encode([answer])
    query_vector = embedding_model.encode([query])
    score = cosine_similarity(query_vector, answer_vector)[0][0]
    return round(float(score) * 100, 1)


# ============================================
# EXTRACT RECOMMENDATIONS
# ============================================
def extract_recommendations(answer_text):
    """Extract recommendations from answer"""
    lines = answer_text.split('\n')
    recommendations = []

    for line in lines:
        if any(line.strip().startswith(prefix) for prefix in ['•', '-', '1.', '2.', '3.', '*']):
            text = line.strip().lstrip('•-123456.*: ').strip()
            if text:
                recommendations.append(text)

    if not recommendations:
        if '**Recommendations:**' in answer_text or 'Recommendations:' in answer_text:
            parts = re.split(r'\*\*Recommendations?\*\*|Recommendations?:', answer_text)
            if len(parts) > 1:
                recs_text = parts[1]
                items = re.split(r'\n(?=[•\-\*])', recs_text)
                for item in items:
                    cleaned = item.strip().lstrip('•-*').strip()
                    if cleaned and len(cleaned) > 10:
                        recommendations.append(cleaned)

    return recommendations[:3]


# ============================================
# API ENDPOINTS
# ============================================
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Flask API is running'})


@app.route('/api/assess', methods=['POST'])
def assess_risk():
    """Main endpoint for heart risk assessment"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400

        # Retrieve and rerank
        hybrid_results = hybrid_search(query, k=6)
        final_docs = rerank_docs(query, hybrid_results, top_k=3)

        # Build context
        context_for_llm = "\n\n---\n\n".join([
            f"Source: {doc.metadata['source']}\n{doc.page_content}"
            for doc in final_docs
        ])

        # Generate assessment
        prompt = f"""You are a friendly heart health assistant. Answer the patient's question using the guidelines below.

GUIDELINES:
{context_for_llm}

PATIENT: "{query}"

RULES:
- Base your answer ONLY on facts in the guidelines above
- Use warm, friendly, conversational language
- Keep it simple — like talking to a friend
- Do NOT add medical advice not found in the guidelines

FORMAT:

**Assessment:**
[Write 1-2 friendly sentences summarizing what the guidelines say about their situation]

**Recommendations:**
- [List recommendations only if they appear in guidelines]
- [If no recommendations, say "The guidelines don't provide specific recommendations for this"]

**Disclaimer:** This is based on medical guidelines. Please consult a real doctor for personalized advice.

ANSWER:"""

        response = llm.invoke(prompt).content

        # Evaluate
        f_score = evaluate_faithfulness(query, context_for_llm, response)
        r_score = evaluate_relevancy(query, response)

        # Extract recommendations
        recommendations = extract_recommendations(response)

        return jsonify({
            'success': True,
            'assessment': response,
            'recommendations': recommendations,
            'metrics': {
                'faithfulness': f_score,
                'relevancy': r_score
            },
            'sources': [
                {
                    'id': i + 1,
                    'source': doc.metadata['source'],
                    'page': doc.metadata['page'],
                    'content': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content
                }
                for i, doc in enumerate(final_docs)
            ]
        })

    except Exception as e:
        print(f"Error in /api/assess: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health-tips', methods=['GET'])
def get_health_tips():
    """Get general health tips"""
    tips = [
        "Regular exercise reduces heart disease risk by up to 30%",
        "A balanced diet rich in fiber helps maintain healthy cholesterol levels",
        "Limiting salt intake helps control blood pressure",
        "Managing stress through meditation or yoga improves heart health",
        "Adequate sleep (7-8 hours) is crucial for cardiovascular health",
        "Avoiding smoking is one of the most important preventive measures"
    ]
    return jsonify({'tips': tips})


# ============================================
# SERVE REACT FRONTEND
# ============================================
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets (JS, CSS, etc.) - assets are at root level"""
    # Try to find the file in the static folder (root level)
    file_path = os.path.join(app.static_folder, filename)
    if os.path.exists(file_path):
        # Determine correct MIME type
        if filename.endswith('.js'):
            return send_from_directory(app.static_folder, filename, mimetype='application/javascript; charset=utf-8')
        elif filename.endswith('.css'):
            return send_from_directory(app.static_folder, filename, mimetype='text/css; charset=utf-8')
        else:
            return send_from_directory(app.static_folder, filename)
    
    # If not found, return 404
    return jsonify({'error': 'Asset not found: ' + filename}), 404


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    """Serve React app - catch-all for SPA routing"""
    # Skip API routes - let them be handled by their handlers
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    
    # Try to serve the file directly if it exists
    if path:
        file_path = os.path.join(app.static_folder, path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Determine correct MIME type
            if path.endswith('.js'):
                return send_from_directory(app.static_folder, path, mimetype='application/javascript; charset=utf-8')
            elif path.endswith('.css'):
                return send_from_directory(app.static_folder, path, mimetype='text/css; charset=utf-8')
            else:
                return send_from_directory(app.static_folder, path)
    
    # Otherwise, serve index.html for React routing
    return send_from_directory(app.static_folder, 'index.html')


# ============================================
# ERROR HANDLERS
# ============================================
@app.errorhandler(404)
def not_found(error):
    # For API requests, return JSON error
    if request.path.startswith('/api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    # For frontend routes, serve index.html (SPA routing)
    return send_from_directory(app.static_folder, 'index.html'), 200


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================
# STARTUP
# ============================================
if __name__ == '__main__':
    print("Initializing Combined Flask + React App...")
    init_models()
    
    port = int(os.getenv('PORT', 7860))
    print(f"\n🚀 Starting Combined App on http://localhost:{port}")
    print("📱 Frontend: http://localhost:{port}")
    print("📚 API Endpoints:")
    print("   GET  /health")
    print("   POST /api/assess")
    print("   GET  /api/health-tips")
    
    app.run(debug=False, port=port, host='0.0.0.0', use_reloader=False)
