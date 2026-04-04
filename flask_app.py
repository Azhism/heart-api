from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from pinecone import Pinecone
import os, re, pickle, json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def resolve_static_folder():
    """Prefer the build output directory used by the Dockerfile."""
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, 'heart-health-ai', 'dist'),
        os.path.join(base_dir, 'dist'),
        'dist',
    ]

    for candidate in candidates:
        index_file = os.path.join(candidate, 'index.html')
        if os.path.exists(index_file):
            return candidate

    return candidates[0]


static_folder = resolve_static_folder()

app = Flask(__name__, static_folder=static_folder, static_url_path='')
CORS(app)

# Environment variables / Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "heart-risk-300-chunks")

# Global models (loaded once)
embedding_model = None
reranker = None
llm = None
judge_llm = None
index = None
bm25_retriever = None

def load_models():
    global embedding_model, reranker, llm, judge_llm, index, bm25_retriever
    
    if embedding_model is not None:
        return  # Already loaded
    
    print("Loading models...")
    
    # Embedding model
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    
    # Reranker
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # LLMs
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
    
    # Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        print(f"❌ Pinecone connection failed: {e}")
        raise
    
    # BM25
    try:
        pkl_path = os.path.join(os.path.dirname(__file__), "bm25_data.pkl")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        bm25_retriever = BM25Retriever.from_documents(data["docs"], k=5)
        print("✅ BM25 loaded")
    except Exception as e:
        print(f"❌ BM25 loading failed: {e}")
        raise
    
    print("✅ All models loaded successfully")

def semantic_search(query, k=5):
    global embedding_model, index
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return [
        Document(
            page_content=match["metadata"]["text"],
            metadata={"source": match["metadata"]["source"], "page": match["metadata"]["page"]}
        )
        for match in results["matches"]
    ]

def hybrid_search(query, k=5):
    global bm25_retriever
    
    # BM25 results
    bm25_results = bm25_retriever.invoke(query)
    
    # Semantic results
    semantic_results = semantic_search(query, k=k)
    
    # RRF combination
    scores = {}
    all_docs = {}
    
    for rank, doc in enumerate(bm25_results):
        doc_id = doc.page_content[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1/(rank + 60)
        all_docs[doc_id] = doc
    
    for rank, doc in enumerate(semantic_results):
        doc_id = doc.page_content[:100]
        scores[doc_id] = scores.get(doc_id, 0) + 1/(rank + 60)
        all_docs[doc_id] = doc
    
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_docs[doc_id] for doc_id, _ in sorted_docs[:k]]

def rerank(query, docs, top_k=3):
    global reranker
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

def evaluate_faithfulness(query, context, answer):
    global judge_llm
    
    # Extract claims from answer
    claims_prompt = f"""Extract all factual claims from this answer as a numbered list.
Each claim should be a single, verifiable statement.

Answer: {answer}

Return ONLY a numbered list of claims, nothing else."""

    claims_response = judge_llm.invoke(claims_prompt)
    claims_text = claims_response.content
    
    if not claims_text.strip():
        return 0.0
    
    # Verify claims
    verify_prompt = f"""You are a fact checker. Given the following context from medical guidelines,
verify each claim and return a JSON with this exact format.
Use ONLY true or false for supported field.

Context:
{context[:2000]}

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
        json_text = json_text.replace('"supported": Not in context', '"supported": false')
        json_text = json_text.replace('"supported": None', '"supported": false')
        json_text = json_text.replace('"supported": "true"', '"supported": true')
        json_text = json_text.replace('"supported": "false"', '"supported": false')
        
        verification = json.loads(json_text)
        claims = verification['claims']
        
        supported = sum(1 for c in claims if c.get('supported', False))
        total = len(claims)
        score = (supported / total) * 100 if total > 0 else 0
        return round(score, 1)
    except:
        return 0.0

def evaluate_relevancy(query, answer):
    global embedding_model
    answer_vector = embedding_model.encode([answer])
    query_vector = embedding_model.encode([query])
    score = cosine_similarity(query_vector, answer_vector)[0][0]
    return round(float(score) * 100, 1)

@app.route('/', methods=['GET'])
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/assets/<path:filename>', methods=['GET'])
def serve_assets(filename):
    for folder in [app.static_folder, os.path.join(os.path.dirname(__file__), 'heart-health-ai', 'dist')]:
        candidates = [
            os.path.join(folder, 'assets', filename),
            os.path.join(folder, filename),
        ]

        for file_path in candidates:
            if os.path.exists(file_path):
                if file_path.endswith('.js'):
                    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), mimetype='application/javascript; charset=utf-8')
                if file_path.endswith('.css'):
                    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), mimetype='text/css; charset=utf-8')
                return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path))

    return jsonify({'error': f'Asset not found: {filename}'}), 404

@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/assess', methods=['POST'])
def assess_risk():
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400
        
        # Retrieve documents
        hybrid_results = hybrid_search(query, k=6)
        final_docs = rerank(query, hybrid_results, top_k=3)
        
        # Build context
        context = "\n\n---\n\n".join([
            f"Source: {doc.metadata['source']}\nPage: {doc.metadata['page']}\n{doc.page_content}"
            for doc in final_docs
        ])
        
        # Generate answer
        prompt = f"""You are a friendly heart health assistant. Answer the patient's question using the guidelines below.

GUIDELINES:
{context}

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
        
        response = llm.invoke(prompt)
        answer = response.content
        
        # Evaluate
        f_score = evaluate_faithfulness(query, context, answer)
        r_score = evaluate_relevancy(query, answer)
        
        return jsonify({
            'success': True,
            'riskLevel': 'consultation_recommended',  # Always recommend consultation
            'assessment': answer,
            'recommendations': ['Consult with a healthcare professional for personalized advice'],
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
        print(f"Error in assess_risk: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Assessment failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'pinecone_index': PINECONE_INDEX_NAME})

if __name__ == '__main__':
    try:
        load_models()
        print("Starting Flask server on port 7860...")
        app.run(host='0.0.0.0', port=7860, debug=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
        exit(1)
