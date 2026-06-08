"""
01_scrape_verses.py
Scrapes all 700 Bhagavad Gita verses from the Vedic Scriptures API.
Saves raw verse data + chapter metadata.

API: https://vedicscriptures.github.io
Endpoints:
  - /chapters         -> chapter metadata (name, summary, verse count)
  - /slok/{ch}/{verse} -> full verse data (translations, commentaries)

Usage: python scripts/01_scrape_verses.py
Output: data/raw/all_verses.json, data/raw/chapters.json
"""

import json
import time
import requests
from pathlib import Path

BASE_URL = "https://vedicscriptures.github.io"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_chapters():
    """Fetch chapter metadata (names, summaries, verse counts)."""
    print("Fetching chapter metadata...")
    resp = requests.get(f"{BASE_URL}/chapters", timeout=30)
    resp.raise_for_status()
    chapters = resp.json()
    print(f"  Found {len(chapters)} chapters")
    return chapters


def fetch_verse(chapter: int, verse: int, retries: int = 3):
    """Fetch a single verse with retries."""
    url = f"{BASE_URL}/slok/{chapter}/{verse}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1} for {chapter}.{verse} (waiting {wait}s): {e}")
                time.sleep(wait)
            else:
                print(f"  FAILED {chapter}.{verse} after {retries} attempts: {e}")
                return None


def extract_english_content(verse_data: dict) -> dict:
    """
    Extract and organize English translations and commentaries
    from the raw verse JSON.
    
    Keys in raw data per scholar:
      - 'et' = English translation
      - 'ec' = English commentary
      - 'ht' = Hindi translation
      - 'hc' = Hindi commentary  
      - 'sc' = Sanskrit commentary
    """
    english_translations = {}
    english_commentaries = {}
    hindi_translations = {}
    hindi_commentaries = {}

    # Scholar key -> display name mapping (from the 'author' field)
    scholar_keys = [
        "siva", "purohit", "chinmay", "san", "adi",
        "gambir", "madhav", "anand", "rams", "raman",
        "abhinav", "sankar", "jaya", "vallabh", "ms",
        "srid", "dhan", "venkat", "puru", "neel", "prabhu", "tej"
    ]

    for key in scholar_keys:
        if key not in verse_data:
            continue
        scholar = verse_data[key]
        author = scholar.get("author", key)

        # English
        if "et" in scholar and scholar["et"]:
            english_translations[author] = scholar["et"].strip()
        if "ec" in scholar and scholar["ec"]:
            english_commentaries[author] = scholar["ec"].strip()

        # Hindi
        if "ht" in scholar and scholar["ht"]:
            hindi_translations[author] = scholar["ht"].strip()
        if "hc" in scholar and scholar["hc"]:
            hindi_commentaries[author] = scholar["hc"].strip()

    return {
        "english_translations": english_translations,
        "english_commentaries": english_commentaries,
        "hindi_translations": hindi_translations,
        "hindi_commentaries": hindi_commentaries,
    }


def compute_verse_richness(content: dict) -> int:
    """
    Compute a richness score based on total commentary length.
    Used later to identify philosophically significant verses.
    """
    total = 0
    for text in content["english_commentaries"].values():
        total += len(text.split())
    for text in content["hindi_commentaries"].values():
        total += len(text.split())  # approximate, Hindi word count
    return total


def scrape_all_verses(chapters: list) -> list:
    """Scrape all verses across all chapters."""
    all_verses = []
    total_verses = sum(ch["verses_count"] for ch in chapters)
    scraped = 0

    for ch in chapters:
        ch_num = ch["chapter_number"]
        v_count = ch["verses_count"]
        print(f"\nChapter {ch_num}: {ch.get('name_translation', '')} ({v_count} verses)")

        for v in range(1, v_count + 1):
            raw = fetch_verse(ch_num, v)
            if raw is None:
                continue

            content = extract_english_content(raw)
            richness = compute_verse_richness(content)

            verse_entry = {
                "id": f"BG{ch_num}.{v}",
                "chapter": ch_num,
                "verse": v,
                "chapter_name": ch.get("name_translation", ""),
                "chapter_meaning": ch.get("name_meaning", ""),
                "slok": raw.get("slok", ""),
                "transliteration": raw.get("transliteration", ""),
                "english_translations": content["english_translations"],
                "english_commentaries": content["english_commentaries"],
                "hindi_translations": content["hindi_translations"],
                "hindi_commentaries": content["hindi_commentaries"],
                "richness_score": richness,
            }
            all_verses.append(verse_entry)
            scraped += 1

            if scraped % 50 == 0:
                print(f"  Progress: {scraped}/{total_verses} verses scraped")

            # Be polite to the API - small delay between requests
            time.sleep(0.3)

    return all_verses


def print_summary(verses: list):
    """Print dataset summary statistics."""
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Total verses scraped: {len(verses)}")

    # Count English content availability
    has_eng_trans = sum(1 for v in verses if v["english_translations"])
    has_eng_comm = sum(1 for v in verses if v["english_commentaries"])
    has_hin_trans = sum(1 for v in verses if v["hindi_translations"])
    has_hin_comm = sum(1 for v in verses if v["hindi_commentaries"])

    print(f"Verses with English translations: {has_eng_trans}")
    print(f"Verses with English commentaries: {has_eng_comm}")
    print(f"Verses with Hindi translations:   {has_hin_trans}")
    print(f"Verses with Hindi commentaries:   {has_hin_comm}")

    # Richness distribution
    scores = sorted([v["richness_score"] for v in verses], reverse=True)
    print(f"\nRichness scores (commentary word count):")
    print(f"  Top 10 verses:  {scores[:10]}")
    print(f"  Median:         {scores[len(scores)//2]}")
    print(f"  Verses with score > 500: {sum(1 for s in scores if s > 500)}")
    print(f"  Verses with score > 200: {sum(1 for s in scores if s > 200)}")


def main():
    # Fetch chapter metadata
    chapters = fetch_chapters()

    # Save chapter metadata (includes summaries useful for Type 3/4 QA)
    chapters_path = RAW_DIR / "chapters.json"
    with open(chapters_path, "w", encoding="utf-8") as f:
        json.dump(chapters, f, ensure_ascii=False, indent=2)
    print(f"Saved chapter metadata to {chapters_path}")

    # Scrape all verses
    verses = scrape_all_verses(chapters)

    # Save all verses
    verses_path = RAW_DIR / "all_verses.json"
    with open(verses_path, "w", encoding="utf-8") as f:
        json.dump(verses, f, ensure_ascii=False, indent=2)
    print(f"\nSaved all verses to {verses_path}")

    # Print summary
    print_summary(verses)

    # Save top rich verses list (for Type 5 QA generation)
    rich_verses = sorted(verses, key=lambda v: v["richness_score"], reverse=True)
    rich_ids = [v["id"] for v in rich_verses[:200]]
    rich_path = RAW_DIR / "rich_verses_top200.json"
    with open(rich_path, "w", encoding="utf-8") as f:
        json.dump(rich_ids, f, indent=2)
    print(f"Saved top 200 rich verse IDs to {rich_path}")


if __name__ == "__main__":
    main()