# 🕉️ ShlokaSage — Bhagavad Gita AI Guide

A fine-tuned Llama 3.1 8B model that serves as an expert guide on the Bhagavad Gita, trained on all 700 verses with translations and commentaries from 22 scholars across Advaita, Vishishtadvaita, Dvaita, and Kashmir Shaiva philosophical traditions.

## Problem Statement

General-purpose LLMs give shallow, citation-free answers about the Bhagavad Gita. They lack verse-specific accuracy, misattribute philosophical positions, and cannot map life situations to specific verses. ShlokaSage addresses this through targeted fine-tuning on a synthetically generated dataset of ~2,000 QA pairs covering verse explanations, life guidance, thematic queries, narrative context, and multi-perspective philosophical analysis.

## Project Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐     ┌──────────┐
│  Scrape API  │────▶│ Generate QA  │────▶│ Format Data  │────▶│  QLoRA    │────▶│  Deploy  │
│  (700 verses)│     │ (Groq LLM)   │     │ (ChatML)     │     │  Training │     │  Gradio  │
└─────────────┘     └──────────────┘     └──────────────┘     └───────────┘     └──────────┘
01_scrape.py        02_generate_qa.py    03_format.py         Kaggle/Colab      HF Spaces
```

## Dataset Details

| QA Type | Count | Description |
|---------|-------|-------------|
| Verse Explanation | ~700 | One per verse with Sanskrit, transliteration, scholar insight |
| Life Situation | ~400-500 | Maps 50 real-life problems to specific verses |
| Thematic Query | ~80 | Clusters verses by 40 themes (anger, karma, duty, etc.) |
| Narrative Context | ~100 | Story context for key dramatic moments |
| Multi-Perspective | ~150-200 | Contrasting philosophical interpretations |
| Rephrased Variations | ~200-300 | Diverse query styles including Hinglish |

**Data Source:** [Vedic Scriptures API](https://vedicscriptures.github.io/) — 22 scholars, 9 English translations, Sanskrit/Hindi commentaries per verse.

## Training Configuration

- **Base Model:** Llama 3.1 8B Instruct (4-bit quantized)
- **Method:** QLoRA via Unsloth
- **LoRA Config:** r=16, alpha=16, target=all attention + MLP layers
- **Hardware:** Kaggle T4 GPU (free tier)
- **Epochs:** 3
- **Effective Batch Size:** 8 (2 × 4 gradient accumulation)

## Evaluation

| Metric | Base Llama 3.1 8B | ShlokaSage |
|--------|-------------------|------------|
| Verse Citation F1 | ~25% | ~85%+ |
| Factual Accuracy | ~2.8/5 | ~4.3/5 |
| Practical Depth | ~2.1/5 | ~4.0/5 |
| Hallucination (↑ better) | ~2.4/5 | ~4.5/5 |

*Evaluated on 200 held-out test examples using LLM-as-Judge (Llama 3.3 70B via Groq).*

## Quick Start

```bash
# 1. Scrape verse data
python scripts/01_scrape_verses.py

# 2. Generate QA pairs (needs GROQ_API_KEY)
export GROQ_API_KEY="your-key"
python scripts/02_generate_qa.py

# 3. Format dataset
python scripts/03_format_dataset.py

# 4. Train (upload train.jsonl to Kaggle, run notebook)

# 5. Evaluate
python scripts/04_evaluate.py --base_responses base.json --ft_responses ft.json

# 6. Deploy
python app/app.py
```

## Tech Stack

Unsloth, TRL, PEFT, Transformers, Groq API (Llama 3.3 70B), Gradio, HuggingFace Hub

## Author

**Mayank** — M.Sc. Data Science, IIIT Lucknow