"""
03_format_dataset.py
Converts QA pairs into ChatML format for SFT training.
Creates train/test split and uploads to HuggingFace Hub.

Input:  data/processed/qa_pairs_all.json
Output: data/splits/train.jsonl, data/splits/test.jsonl

Usage: python scripts/03_format_dataset.py
       python scripts/03_format_dataset.py --push_to_hub your-username/ShlokaSage-dataset
"""

import json
import random
import argparse
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are ShlokaSage, an expert guide on the Bhagavad Gita. You have deep knowledge of all 700 verses, their Sanskrit text, transliterations, and interpretations from major scholars including Shankaracharya, Ramanuja, Swami Sivananda, Swami Ramsukhdas, Swami Chinmayananda, and A.C. Bhaktivedanta Swami Prabhupada.

When answering questions:
- Always include the relevant Sanskrit shloka and transliteration when discussing a specific verse
- Cite specific chapter and verse numbers (e.g., BG 2.47)
- Weave in insights from scholars naturally, naming them when their perspective adds unique value
- Cross-reference related verses when helpful
- For life situations, map them to the most relevant verses with practical explanations
- Be warm and accessible, like a knowledgeable teacher — not an academic paper"""


def format_to_chatml(qa_pair: dict) -> dict:
    """Convert a QA pair to ChatML conversation format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": qa_pair["question"]},
            {"role": "assistant", "content": qa_pair["answer"]},
        ]
    }


def validate_pair(pair: dict) -> bool:
    """Basic quality validation of a QA pair."""
    q = pair.get("question", "")
    a = pair.get("answer", "")
    
    if not q or not a:
        return False
    if len(q.split()) < 3:  # question too short
        return False
    if len(a.split()) < 30:  # answer too short
        return False
    if len(a.split()) > 800:  # answer too long (will eat context)
        return False
    
    return True


def create_splits(pairs: list, test_ratio: float = 0.1, seed: int = 42):
    """Create stratified train/test split."""
    random.seed(seed)
    
    # Group by type for stratified split
    by_type = {}
    for p in pairs:
        t = p.get("type", "unknown")
        by_type.setdefault(t, []).append(p)
    
    train, test = [], []
    
    for type_name, type_pairs in by_type.items():
        random.shuffle(type_pairs)
        n_test = max(1, int(len(type_pairs) * test_ratio))
        test.extend(type_pairs[:n_test])
        train.extend(type_pairs[n_test:])
    
    random.shuffle(train)
    random.shuffle(test)
    
    return train, test


def save_jsonl(data: list, path: Path):
    """Save as JSONL (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push_to_hub", type=str, default=None,
                        help="HuggingFace repo to push dataset (e.g., username/ShlokaSage-dataset)")
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    
    # Load QA pairs
    qa_path = PROCESSED_DIR / "qa_pairs_all.json"
    if not qa_path.exists():
        print(f"❌ {qa_path} not found. Run 02_generate_qa.py first!")
        return
    
    with open(qa_path, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)
    print(f"Loaded {len(all_pairs)} QA pairs")
    
    # Validate
    valid_pairs = [p for p in all_pairs if validate_pair(p)]
    removed = len(all_pairs) - len(valid_pairs)
    if removed > 0:
        print(f"Removed {removed} invalid pairs ({removed/len(all_pairs)*100:.1f}%)")
    print(f"Valid pairs: {len(valid_pairs)}")
    
    # Split
    train_pairs, test_pairs = create_splits(valid_pairs, test_ratio=args.test_ratio)
    print(f"Train: {len(train_pairs)} | Test: {len(test_pairs)}")
    
    # Format to ChatML
    train_chatml = [format_to_chatml(p) for p in train_pairs]
    test_chatml = [format_to_chatml(p) for p in test_pairs]
    
    # Save
    save_jsonl(train_chatml, SPLITS_DIR / "train.jsonl")
    save_jsonl(test_chatml, SPLITS_DIR / "test.jsonl")
    print(f"\nSaved to {SPLITS_DIR}/train.jsonl and test.jsonl")
    
    # Also save test pairs with metadata (for evaluation)
    test_with_meta = []
    for pair, chatml in zip(test_pairs, test_chatml):
        test_with_meta.append({
            "question": pair["question"],
            "ground_truth": pair["answer"],
            "type": pair.get("type", "unknown"),
            "verse_id": pair.get("verse_id", ""),
            "messages": chatml["messages"],
        })
    
    test_meta_path = SPLITS_DIR / "test_with_metadata.json"
    with open(test_meta_path, "w", encoding="utf-8") as f:
        json.dump(test_with_meta, f, ensure_ascii=False, indent=2)
    print(f"Saved test metadata to {test_meta_path}")
    
    # Print sample
    print("\n--- Sample Training Example ---")
    sample = train_chatml[0]
    for msg in sample["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
        print(f"[{role}]: {content}\n")
    
    # Push to HuggingFace Hub
    if args.push_to_hub:
        try:
            from datasets import Dataset, DatasetDict
            
            train_ds = Dataset.from_list(train_chatml)
            test_ds = Dataset.from_list(test_chatml)
            ds_dict = DatasetDict({"train": train_ds, "test": test_ds})
            
            ds_dict.push_to_hub(args.push_to_hub)
            print(f"\n✅ Pushed to https://huggingface.co/datasets/{args.push_to_hub}")
        except ImportError:
            print("\n⚠️ Install 'datasets' library: pip install datasets")
        except Exception as e:
            print(f"\n❌ Push failed: {e}")


if __name__ == "__main__":
    main()