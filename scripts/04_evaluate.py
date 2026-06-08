"""
04_evaluate.py
Evaluates base model vs fine-tuned model on the held-out test set.

Metrics:
  1. Verse Citation Accuracy (automated regex check)
  2. ROUGE-L score
  3. LLM-as-Judge via Groq (factual accuracy, depth, hallucination)

Prerequisites:
  - data/splits/test_with_metadata.json
  - Base model responses (generated separately or inline)
  - Fine-tuned model responses (generated separately or inline)
  - GROQ_API_KEY for LLM-as-Judge

Usage:
  # Generate responses from both models first (on Kaggle/Colab), save as JSON
  # Then run evaluation locally:
  python scripts/04_evaluate.py --base_responses base_responses.json --ft_responses ft_responses.json

  # Or just run LLM-as-Judge on already generated response files:
  python scripts/04_evaluate.py --base_responses base_responses.json --ft_responses ft_responses.json --skip_rouge
"""

import json
import re
import os
import argparse
import time
from pathlib import Path
from collections import defaultdict

SPLITS_DIR = Path("data/splits")
RESULTS_DIR = Path("data/evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Metric 1: Verse Citation Accuracy ──────────────────────────

def extract_verse_citations(text: str) -> set:
    """Extract verse citations like 'BG 2.47', 'Chapter 2 Verse 47', '2.47' etc."""
    patterns = [
        r'BG\s*(\d{1,2})\.(\d{1,3})',
        r'Chapter\s*(\d{1,2})\s*[,.]?\s*Verse\s*(\d{1,3})',
        r'(\d{1,2})\.(\d{1,3})',
    ]
    citations = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for ch, v in matches:
            citations.add(f"{int(ch)}.{int(v)}")
    return citations


def compute_citation_accuracy(ground_truth: str, prediction: str) -> dict:
    """Check if the model cited the correct verses."""
    gt_citations = extract_verse_citations(ground_truth)
    pred_citations = extract_verse_citations(prediction)
    
    if not gt_citations:
        return {"has_citations": False, "precision": None, "recall": None, "f1": None}
    
    correct = gt_citations & pred_citations
    precision = len(correct) / len(pred_citations) if pred_citations else 0
    recall = len(correct) / len(gt_citations) if gt_citations else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "has_citations": True,
        "gt_citations": list(gt_citations),
        "pred_citations": list(pred_citations),
        "correct": list(correct),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ─── Metric 2: ROUGE-L ──────────────────────────────────────────

def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L score (longest common subsequence based)."""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if not ref_tokens or not hyp_tokens:
        return 0.0
    
    lcs = lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ─── Metric 3: LLM-as-Judge ─────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a Bhagavad Gita Q&A model. Rate the response on 4 criteria.

QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
MODEL RESPONSE: {response}

Rate each criterion from 1 (worst) to 5 (best):

1. FACTUAL_ACCURACY: Is the verse meaning, scholar attribution, and chapter/verse citation correct?
   1=completely wrong, 3=partially correct, 5=fully accurate

2. VERSE_CITATION: Does the response cite specific verses (chapter.verse) and include Sanskrit shlokas?
   1=no citations at all, 3=some citations, 5=accurate citations with Sanskrit text

3. PRACTICAL_DEPTH: Does it go beyond textbook meaning to give practical/philosophical insight?
   1=superficial one-liner, 3=decent explanation, 5=deep insight with scholar perspectives

4. HALLUCINATION: Does the response invent fake verses, misattribute scholars, or fabricate content?
   1=heavy hallucination, 3=minor inaccuracies, 5=no hallucination detected

RESPOND ONLY WITH THIS EXACT FORMAT (no other text):
FACTUAL_ACCURACY: <score>
VERSE_CITATION: <score>
PRACTICAL_DEPTH: <score>
HALLUCINATION: <score>"""


def llm_judge(question: str, ground_truth: str, response: str, groq_client) -> dict:
    """Use Groq LLM as judge to evaluate a response."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth[:1000],
        response=response[:1000],
    )
    
    try:
        result = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        text = result.choices[0].message.content.strip()
        
        scores = {}
        for line in text.split("\n"):
            line = line.strip()
            for metric in ["FACTUAL_ACCURACY", "VERSE_CITATION", "PRACTICAL_DEPTH", "HALLUCINATION"]:
                if line.startswith(metric):
                    try:
                        score = int(line.split(":")[-1].strip())
                        scores[metric.lower()] = min(5, max(1, score))
                    except ValueError:
                        pass
        
        time.sleep(2.5)  # rate limit
        return scores if len(scores) == 4 else None
        
    except Exception as e:
        print(f"  Judge error: {e}")
        time.sleep(10)
        return None


# ─── Response Generation Helper (for Kaggle/Colab) ──────────────

GENERATE_SCRIPT = """
# ─── Run this on Kaggle/Colab AFTER training ─────────────────
# Generates responses from both base and fine-tuned models
# Save the output JSON files and download them for evaluation

import json
import torch
from unsloth import FastLanguageModel

SYSTEM_PROMPT = "You are ShlokaSage, an expert guide on the Bhagavad Gita..."  # use full prompt from training

# Load test questions
with open("test_with_metadata.json") as f:
    test_data = json.load(f)

def generate_responses(model, tokenizer, test_data, output_path):
    FastLanguageModel.for_inference(model)
    responses = []
    for i, item in enumerate(test_data):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        responses.append({"question": item["question"], "ground_truth": item["ground_truth"], "response": response, "type": item["type"]})
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(test_data)}")
    with open(output_path, "w") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(responses)} responses to {output_path}")

# Generate BASE model responses (before fine-tuning or load fresh base)
base_model, base_tokenizer = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", max_seq_length=4096, load_in_4bit=True)
generate_responses(base_model, base_tokenizer, test_data, "base_responses.json")

# Generate FINE-TUNED model responses
# (model variable should already be your trained model)
generate_responses(model, tokenizer, test_data, "ft_responses.json")
"""


# ─── Main Evaluation ────────────────────────────────────────────

def evaluate_responses(responses: list, label: str, groq_client=None) -> dict:
    """Run all metrics on a set of responses."""
    citation_scores = []
    rouge_scores = []
    judge_scores = defaultdict(list)
    
    for i, item in enumerate(responses):
        gt = item["ground_truth"]
        pred = item["response"]
        
        # Citation accuracy
        cit = compute_citation_accuracy(gt, pred)
        if cit["has_citations"]:
            citation_scores.append(cit["f1"])
        
        # ROUGE-L
        rouge = compute_rouge_l(gt, pred)
        rouge_scores.append(rouge)
        
        # LLM-as-Judge
        if groq_client:
            scores = llm_judge(item["question"], gt, pred, groq_client)
            if scores:
                for k, v in scores.items():
                    judge_scores[k].append(v)
        
        if (i + 1) % 20 == 0:
            print(f"  [{label}] Evaluated {i+1}/{len(responses)}")
    
    results = {
        "label": label,
        "total_examples": len(responses),
        "citation_accuracy_f1": sum(citation_scores) / len(citation_scores) if citation_scores else 0,
        "citation_count": len(citation_scores),
        "rouge_l": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0,
    }
    
    if judge_scores:
        for k, v in judge_scores.items():
            results[f"judge_{k}"] = sum(v) / len(v)
        results["judge_count"] = len(list(judge_scores.values())[0])
    
    return results


def print_comparison(base_results: dict, ft_results: dict):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS: Base Model vs Fine-Tuned (ShlokaSage)")
    print("=" * 65)
    
    metrics = [
        ("Verse Citation F1", "citation_accuracy_f1", "{:.1%}"),
        ("ROUGE-L", "rouge_l", "{:.3f}"),
        ("Judge: Factual Accuracy", "judge_factual_accuracy", "{:.2f}/5"),
        ("Judge: Verse Citation", "judge_verse_citation", "{:.2f}/5"),
        ("Judge: Practical Depth", "judge_practical_depth", "{:.2f}/5"),
        ("Judge: Hallucination", "judge_hallucination", "{:.2f}/5"),
    ]
    
    print(f"\n{'Metric':<30} {'Base Model':>12} {'ShlokaSage':>12} {'Δ':>8}")
    print("-" * 65)
    
    for name, key, fmt in metrics:
        base_val = base_results.get(key)
        ft_val = ft_results.get(key)
        
        if base_val is not None and ft_val is not None:
            base_str = fmt.format(base_val)
            ft_str = fmt.format(ft_val)
            delta = ft_val - base_val
            delta_str = f"+{fmt.format(delta)}" if delta > 0 else fmt.format(delta)
            print(f"{name:<30} {base_str:>12} {ft_str:>12} {delta_str:>8}")
        else:
            print(f"{name:<30} {'N/A':>12} {'N/A':>12} {'':>8}")
    
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ShlokaSage")
    parser.add_argument("--base_responses", type=str, required=True,
                        help="Path to base model responses JSON")
    parser.add_argument("--ft_responses", type=str, required=True,
                        help="Path to fine-tuned model responses JSON")
    parser.add_argument("--skip_rouge", action="store_true",
                        help="Skip ROUGE computation")
    parser.add_argument("--skip_judge", action="store_true",
                        help="Skip LLM-as-Judge (no Groq API needed)")
    args = parser.parse_args()
    
    # Load responses
    with open(args.base_responses) as f:
        base_responses = json.load(f)
    with open(args.ft_responses) as f:
        ft_responses = json.load(f)
    
    print(f"Loaded {len(base_responses)} base responses, {len(ft_responses)} fine-tuned responses")
    
    # Setup Groq client for LLM-as-Judge
    groq_client = None
    if not args.skip_judge:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            from groq import Groq
            groq_client = Groq(api_key=api_key)
            print("LLM-as-Judge enabled (Groq)")
        else:
            print("⚠️ GROQ_API_KEY not set. Skipping LLM-as-Judge.")
    
    # Evaluate both
    print("\nEvaluating base model...")
    base_results = evaluate_responses(base_responses, "Base Llama 3.1 8B", groq_client)
    
    print("\nEvaluating fine-tuned model...")
    ft_results = evaluate_responses(ft_responses, "ShlokaSage (Fine-tuned)", groq_client)
    
    # Print comparison
    print_comparison(base_results, ft_results)
    
    # Save results
    results = {
        "base_model": base_results,
        "fine_tuned": ft_results,
    }
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Per-type breakdown
    print("\n--- Per-Type Breakdown (Fine-Tuned) ---")
    by_type = defaultdict(list)
    for item in ft_responses:
        by_type[item.get("type", "unknown")].append(item)
    
    for type_name, items in sorted(by_type.items()):
        cite_scores = []
        for item in items:
            cit = compute_citation_accuracy(item["ground_truth"], item["response"])
            if cit["has_citations"]:
                cite_scores.append(cit["f1"])
        avg_cite = sum(cite_scores) / len(cite_scores) if cite_scores else 0
        print(f"  {type_name:<25} n={len(items):>3}  citation_f1={avg_cite:.1%}")


if __name__ == "__main__":
    main()