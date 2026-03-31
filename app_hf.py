"""
Gradio app for Heart Risk Assessment - Hugging Face Spaces
Integrates with Flask backend for RAG-powered assessments
"""

import gradio as gr
import requests
import json
import os
from typing import Tuple

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")

def assess_heart_risk(user_input: str) -> Tuple[str, str, str, str]:
    """
    Call the Flask API to assess heart risk
    
    Args:
        user_input: User's description of symptoms and habits
        
    Returns:
        Tuple of (risk_level, assessment, recommendations, sources)
    """
    if not user_input.strip():
        return "Error", "Please describe your habits or symptoms", "", ""
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/assess",
            json={"query": user_input},
            timeout=120
        )
        
        if response.status_code != 200:
            return "Error", f"API Error: {response.status_code}", "", ""
        
        data = response.json()
        
        if not data.get("success"):
            return "Error", data.get("error", "Unable to process request"), "", ""
        
        # Parse response
        risk_level = data.get("riskLevel", "Unknown").upper()
        assessment = data.get("assessment", "No assessment available")
        
        # Format recommendations
        recommendations = data.get("recommendations", [])
        rec_text = "\n".join([f"• {rec}" for rec in recommendations]) if recommendations else "No specific recommendations available"
        
        # Format sources
        sources = data.get("sources", [])
        sources_text = ""
        if sources:
            for i, source in enumerate(sources, 1):
                sources_text += f"\n**Source {i}: {source.get('source')} (Page {source.get('page')})**\n"
                sources_text += f"{source.get('content', 'N/A')}\n"
        else:
            sources_text = "No sources available"
        
        # Add metrics if available
        metrics = data.get("metrics", {})
        if metrics:
            assessment += f"\n\n📊 **Assessment Quality:**\n• Faithfulness: {metrics.get('faithfulness', 'N/A')}%\n• Relevancy: {metrics.get('relevancy', 'N/A')}%"
        
        return risk_level, assessment, rec_text, sources_text
        
    except requests.exceptions.ConnectionError:
        return "Error", f"Cannot connect to API at {API_BASE_URL}. Make sure Flask backend is running.", "", ""
    except requests.exceptions.Timeout:
        return "Error", "API request timed out. Please try again.", "", ""
    except Exception as e:
        return "Error", f"An error occurred: {str(e)}", "", ""


def get_health_tips() -> str:
    """Get general health tips from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health-tips", timeout=10)
        if response.status_code == 200:
            data = response.json()
            tips = data.get("tips", [])
            return "\n".join([f"• {tip}" for tip in tips])
        return "Unable to load tips"
    except:
        return "Unable to load tips"


# Create Gradio interface
with gr.Blocks(title="❤️ Heart Health AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ❤️ Heart Health AI
    ### AI-Powered Heart Disease Risk Assessment
    
    🔒 **Your data stays private**  
    🩺 **AI-assisted but not medical advice**
    
    Tell me about your lifestyle, symptoms, and habits—like your diet, exercise routine, 
    smoking status, and any symptoms you've noticed. I'll help assess your heart disease 
    risk based on clinical guidelines.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Your Assessment")
            user_input = gr.Textbox(
                label="Describe your habits or symptoms…",
                placeholder="E.g., I eat biryani and walk daily, am I at risk of a heart attack?",
                lines=4,
                interactive=True
            )
            assess_btn = gr.Button("🔍 Assess Risk", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### Quick Tips 💡")
            tips_display = gr.Markdown()
            refresh_tips_btn = gr.Button("Refresh Tips", size="sm")
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Risk Assessment Result")
            risk_level = gr.Textbox(label="Risk Level", interactive=False)
            assessment = gr.Textbox(label="Assessment", lines=6, interactive=False)
        
        with gr.Column():
            gr.Markdown("### Recommendations")
            recommendations = gr.Textbox(label="Recommendations", lines=6, interactive=False)
    
    gr.Markdown("---")
    with gr.Row():
        sources = gr.Textbox(label="📚 Sources & Evidence", lines=8, interactive=False)
    
    gr.Markdown("""
    ---
    **⚠️ Disclaimer:** This is not medical advice. Please consult a qualified doctor 
    for diagnosis and treatment.
    
    *Powered by RAG (Retrieval-Augmented Generation) + Groq LLM*
    """)
    
    # Event handlers
    assess_btn.click(
        fn=assess_heart_risk,
        inputs=user_input,
        outputs=[risk_level, assessment, recommendations, sources]
    )
    
    refresh_tips_btn.click(
        fn=get_health_tips,
        outputs=tips_display
    )
    
    # Load tips on startup
    demo.load(get_health_tips, outputs=tips_display)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
