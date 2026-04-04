
import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from pinecone import Pinecone
import os, re, pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Environment variables / Streamlit Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "heart-risk-300-chunks")

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0.5, max_tokens=2000)
    judge_llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=0.2, max_tokens=2000)
    return embedding_model, reranker, llm, judge_llm

@st.cache_resource
def load_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

@st.cache_resource
def load_bm25():
    pkl_path = r"D:\Classess\ITA\Rag_Project\Heart Attack Project\bm25_data.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return BM25Retriever.from_documents(data["docs"], k=5)

def semantic_search(query, index, embedding_model, k=5):
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=k, include_metadata=True)
    return [
        Document(
            page_content=match["metadata"]["text"],
            metadata={"source": match["metadata"]["source"], "page": match["metadata"]["page"]}
        )
        for match in results["matches"]
    ]

def hybrid_search(query, bm25, index, embedding_model, k=5):
    bm25_results = bm25.invoke(query)
    semantic_results = semantic_search(query, index, embedding_model, k=k)
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

def rerank(query, docs, reranker, top_k=3):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

def evaluate_faithfulness(query, context, answer, judge_llm):
    """Faithfulness evaluation with context claim extraction and detailed terminal output"""

    print("\n" + "="*50)
    print(f"CONTEXT LENGTH: {len(context)} characters")
    print("="*50 + "\n")

    # Step 0: Extract claims from context (for debugging)
    context_claims_prompt = f"""Extract all factual claims from this context as a numbered list.
Each claim should be a single, verifiable statement.

Context:
{context[:3000]}

Return ONLY a numbered list of claims, nothing else."""

    context_claims_response = judge_llm.invoke(context_claims_prompt)
    context_claims_text = context_claims_response.content
    print("EXTRACTED CLAIMS (from context):")
    print(context_claims_text)
    print()

    # Step 1: Extract claims from answer
    claims_prompt = f"""Extract all factual claims from this answer as a numbered list.
Each claim should be a single, verifiable statement.

Answer: {answer}

Return ONLY a numbered list of claims, nothing else."""

    claims_response = judge_llm.invoke(claims_prompt)
    claims_text = claims_response.content
    print("EXTRACTED CLAIMS (from answer):")
    print(claims_text)
    print()

    # Step 2: Verify each claim against context
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

    # Clean and parse JSON
    import json
    import re

    try:
        # Try to extract JSON from response
        json_text = re.search(r'\{.*\}', response_text, re.DOTALL).group()

        # Clean up common issues
        json_text = json_text.replace('"supported": not verified', '"supported": false')
        json_text = json_text.replace('"supported": Not verified', '"supported": false')
        json_text = json_text.replace('"supported": Not in context', '"supported": false')
        json_text = json_text.replace('"supported": None', '"supported": false')
        json_text = json_text.replace('"supported": "true"', '"supported": true')
        json_text = json_text.replace('"supported": "false"', '"supported": false')

        verification = json.loads(json_text)
        claims = verification['claims']

        # Count supported claims
        supported = []
        unsupported = []
        for c in claims:
            is_supported = c.get('supported', False)
            if isinstance(is_supported, bool):
                if is_supported:
                    supported.append(c['claim'])
                else:
                    unsupported.append(c['claim'])
            elif isinstance(is_supported, str):
                if is_supported.lower() in ['true', 'yes', 'supported']:
                    supported.append(c['claim'])
                else:
                    unsupported.append(c['claim'])

        total = len(claims)
        score = (len(supported) / total) * 100 if total > 0 else 0

        print(f"SUPPORTED CLAIMS ({len(supported)}/{total}):")
        for claim in supported:
            print(f"  ✅ {claim}")
        print()
        print(f"UNSUPPORTED CLAIMS ({len(unsupported)}/{total}):")
        for claim in unsupported:
            print(f"  ❌ {claim}")
        print()
        print(f"FAITHFULNESS SCORE: {len(supported)}/{total} claims supported = {score:.1f}%")

        return round(score, 1)

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response (first 500 chars): {response_text[:500]}")

        # Fallback: count by regex
        try:
            true_count = len(re.findall(r'"supported":\s*true', response_text, re.IGNORECASE))
            false_count = len(re.findall(r'"supported":\s*false', response_text, re.IGNORECASE))
            total = true_count + false_count
            if total > 0:
                score = (true_count / total) * 100
                print(f"FALLBACK SCORE: {true_count}/{total} = {score:.1f}%")
                return round(score, 1)
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")

        return 0

def evaluate_relevancy(query, answer, embedding_model):
    answer_vector = embedding_model.encode([answer])
    query_vector = embedding_model.encode([query])
    score = cosine_similarity(query_vector, answer_vector)[0][0]
    return round(float(score) * 100, 1)

st.set_page_config(page_title="❤️ Pakistan Heart Risk Assessor", page_icon="❤️", layout="wide")
st.title("❤️ Pakistan Heart Risk Assessor")
st.markdown("*Powered by RAG — Answers grounded in WHO & Pakistani medical guidelines*")
st.warning("⚠️ This is not medical advice. Please consult a real doctor for diagnosis.")

embedding_model, reranker_model, llm, judge_llm = load_models()
index = load_pinecone()
bm25 = load_bm25()

query = st.text_area("Describe your habits, symptoms, or concerns:",
    placeholder="e.g. I smoke 10 cigarettes a day, eat fried food daily, never exercise. Am I at risk?",
    height=120)

if st.button("🔍 Assess My Risk", type="primary"):
    if query.strip():
        with st.spinner("Analyzing your risk based on medical guidelines..."):
            hybrid_results = hybrid_search(query, bm25, index, embedding_model)
            final_docs = rerank(query, hybrid_results, reranker_model)

            print("\n" + "="*50)
            print(f"FINAL DOCS COUNT: {len(final_docs)}")
            for i, doc in enumerate(final_docs):
                print(f"Doc {i+1}: {doc.metadata['source']} (Page {doc.metadata['page']}) - {len(doc.page_content)} chars")
                print(f"Preview: {doc.page_content[:200]}...\n")
            print("="*50 + "\n")

            # Build context WITHOUT page numbers for faithfulness evaluation
            context_for_llm = "\n\n---\n\n".join([doc.page_content for doc in final_docs])

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

            # Use the clean context (just the text content) for faithfulness
            f_score = evaluate_faithfulness(query, context_for_llm, response, judge_llm)
            r_score = evaluate_relevancy(query, response, embedding_model)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🩺 Risk Assessment")
            st.markdown(response)
        with col2:
            st.subheader("📊 Answer Quality")
            st.metric("Faithfulness", f"{f_score}%")
            st.metric("Relevancy", f"{r_score}%")
            st.subheader("📄 Retrieved Sources")
            for i, doc in enumerate(final_docs, 1):
                with st.expander(f"Source {i}: {doc.metadata['source']} (Page {doc.metadata['page']})"):
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
    else:
        st.error("Please enter a question or describe your symptoms.")
