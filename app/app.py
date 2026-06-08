"""
ShlokaSage — Bhagavad Gita AI Guide
Gradio chatbot for HuggingFace Spaces deployment.

Deployment options:
  Option A: HF Inference API (free, rate-limited)
  Option B: Local GGUF via llama-cpp-python

Usage (local):
  pip install gradio huggingface_hub
  python app/app.py

For HF Spaces:
  Create a new Space, copy this file as app.py, add requirements.txt
"""

import gradio as gr
from huggingface_hub import InferenceClient

# ─── Configuration ───────────────────────────────────────────────
# Change these to your model repo
MODEL_REPO = "your-username/ShlokaSage-Llama3.1-8B-QLoRA"  # ← CHANGE THIS

SYSTEM_PROMPT = """You are ShlokaSage, an expert guide on the Bhagavad Gita. You have deep knowledge of all 700 verses, their Sanskrit text, transliterations, and interpretations from major scholars including Shankaracharya, Ramanuja, Swami Sivananda, Swami Ramsukhdas, Swami Chinmayananda, and A.C. Bhaktivedanta Swami Prabhupada.

When answering questions:
- Always include the relevant Sanskrit shloka and transliteration when discussing a specific verse
- Cite specific chapter and verse numbers (e.g., BG 2.47)
- Weave in insights from scholars naturally, naming them when their perspective adds unique value
- Cross-reference related verses when helpful
- For life situations, map them to the most relevant verses with practical explanations
- Be warm and accessible, like a knowledgeable teacher — not an academic paper"""

TITLE = "🕉️ ShlokaSage — Bhagavad Gita AI Guide"

DESCRIPTION = """**ShlokaSage** is a fine-tuned Llama 3.1 8B model trained on the Bhagavad Gita's 700 verses 
with translations and commentaries from 22 scholars across multiple philosophical traditions.

**Ask me anything about the Gita:**
- 📖 Verse explanations: *"What does verse 2.47 mean?"*
- 🧘 Life guidance: *"I'm stressed about my career, what does the Gita say?"*
- 🎯 Thematic queries: *"Which verses talk about anger?"*
- 📜 Story & context: *"Why did Arjuna refuse to fight?"*
- 🔬 Philosophical depth: *"How do Shankaracharya and Ramanuja differ on verse 9.34?"*
"""

EXAMPLES = [
    "What does Bhagavad Gita Chapter 2 Verse 47 mean?",
    "I keep worrying about my exam results even though I've studied hard. What does the Gita say?",
    "Which verses in the Gita talk about controlling anger?",
    "What was happening when Krishna showed his Vishwaroop?",
    "I feel lost and don't know what to do with my life. Can the Gita help?",
    "Explain the concept of the three Gunas from the Gita.",
    "What does Krishna say about death and the soul?",
]

# ─── Model Client ────────────────────────────────────────────────

client = InferenceClient(MODEL_REPO)


def respond(message: str, history: list, system_prompt: str, max_tokens: int, temperature: float):
    """Generate streaming response."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    for user_msg, assistant_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    response = ""
    try:
        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=0.9,
        ):
            token = chunk.choices[0].delta.content
            if token:
                response += token
                yield response
    except Exception as e:
        yield f"⚠️ Error generating response: {str(e)}\n\nPlease try again or check if the model is loaded."


# ─── Gradio UI ───────────────────────────────────────────────────

with gr.Blocks(
    title="ShlokaSage",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
    ),
) as demo:
    
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)
    
    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[
            gr.Textbox(
                value=SYSTEM_PROMPT,
                label="System Prompt",
                visible=False,  # hidden by default
            ),
            gr.Slider(
                minimum=128,
                maximum=1024,
                value=512,
                step=64,
                label="Max Response Length",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            ),
        ],
        examples=EXAMPLES,
        cache_examples=False,
    )
    
    gr.Markdown("""
    ---
    **About ShlokaSage:** Fine-tuned on ~2,000 QA pairs covering verse explanations, 
    life-situation mappings, thematic queries, narrative context, and multi-perspective 
    philosophical analysis. Trained using QLoRA on Llama 3.1 8B Instruct.
    
    *Built by [Your Name] as a GenAI portfolio project.*
    """)


if __name__ == "__main__":
    demo.launch()