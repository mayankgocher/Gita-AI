"""
02_generate_qa.py
Generates QA pairs from scraped Bhagavad Gita verse data using Gemini API.

Generates 6 types of QA pairs:
  Type 1: Verse Explanation (~700 pairs - every verse)
  Type 2: Life Situation Mapping (~400-500 pairs)
  Type 3: Thematic Query (~80 pairs)
  Type 4: Narrative Context (~100 pairs)
  Type 5: Multi-Perspective (~150-200 pairs - rich verses only)
  Type 6: Rephrased Variations (~200-300 pairs)

Prerequisites:
  - data/raw/all_verses.json (from 01_scrape_verses.py)
  - data/raw/chapters.json
  - .env file with GEMINI_API_KEY

Usage: 
  python scripts/02_generate_qa.py
  
  # Or generate specific types only:
  python scripts/02_generate_qa.py --types 1 2 3

  # Resume from where you left off:
  python scripts/02_generate_qa.py --types 1 --resume

Output: data/processed/qa_pairs_all.json
"""

import json
import time
import argparse
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Load .env file
load_dotenv()

# ─── Configuration ───────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_DIR = Path("data/raw")

MODEL = "gemini-2.5-flash-lite"
DELAY_BETWEEN_CALLS = 0.3  # minimal delay, Gemini paid tier is generous

client = None  # initialized in main()
call_count = 0


# ─── Life situations for Type 2 ─────────────────────────────────
LIFE_SITUATIONS = [
    "I am extremely stressed about my exam results and can't stop overthinking",
    "I feel lost and confused about which career path to choose",
    "Someone I trusted deeply has betrayed me and I can't forgive them",
    "I keep comparing myself to my more successful friends and feel worthless",
    "I am dealing with the death of a close family member and can't cope",
    "I feel lazy and unmotivated even though I know I should be working hard",
    "I am angry at my boss for being unfair but I can't express it",
    "My parents want me to do engineering but I want to pursue arts",
    "I did something wrong in the past and guilt is eating me alive",
    "I feel jealous of my colleague who got promoted over me",
    "I am afraid of failing and this fear is paralyzing me from taking action",
    "I feel detached from everything and nothing excites me anymore",
    "I am in a toxic relationship but I am too attached to leave",
    "I am overwhelmed by too many responsibilities and don't know where to start",
    "I have achieved a lot but still feel empty inside",
    "I want to stand up for what is right but I am afraid of consequences",
    "I keep procrastinating on important tasks and waste time on distractions",
    "I feel peer pressure to drink and party but I don't want to",
    "I had a huge fight with my best friend and I don't know if I should apologize",
    "I am facing a moral dilemma at work — should I report my colleague's mistake",
    "I can't stop worrying about the future and my financial security",
    "I feel like I am not good enough no matter how much I achieve",
    "I lost my job and feel like a complete failure",
    "I am struggling with addiction and feel powerless to stop",
    "Everyone around me seems happy but I feel deeply lonely",
    "I want to start something new but I am afraid of leaving my comfort zone",
    "I feel intense hatred toward someone who wronged me years ago",
    "I am confused between following my heart and being practical",
    "I have too much pride and ego and it's ruining my relationships",
    "I feel like giving up on life — nothing seems worth the effort anymore",
    "My hard work is never recognized while others take credit easily",
    "I want inner peace but my mind is constantly restless",
    "I am torn between family duty and personal ambition",
    "I feel suffocated by societal expectations about marriage and career",
    "I am grieving the end of a long relationship and can't move on",
    "I feel purposeless — I don't know why I exist or what I should do",
    "I want to help others but I don't even know how to help myself",
    "I am scared of death and what happens after",
    "I judge people too quickly and I know it's wrong but I can't help it",
    "I feel spiritually disconnected even though I want to believe in something",
    "I have anger issues and I lash out at people I love",
    "I feel controlled by my desires and cravings",
    "I want to meditate but my mind won't stay still for even a minute",
    "I feel like the world is unfair and good people always suffer",
    "I am anxious about a job interview tomorrow and can't sleep",
    "I feel disrespected by younger people and it makes me angry",
    "I sacrificed a lot for my family but nobody appreciates it",
    "I feel stuck in life — same routine, no growth, no excitement",
    "I am worried about my children's future in this uncertain world",
    "I made a big financial loss and I blame myself for being greedy",
]

# ─── Themes for Type 3 ──────────────────────────────────────────
GITA_THEMES = [
    "Karma Yoga — the path of selfless action",
    "Bhakti Yoga — the path of devotion",
    "Jnana Yoga — the path of knowledge",
    "Dhyana Yoga — the path of meditation",
    "Controlling anger and its consequences",
    "Desire, attachment and their dangers",
    "The three Gunas — Sattva, Rajas, Tamas",
    "Nature of Dharma and righteous duty",
    "Death, rebirth and the immortality of the soul",
    "Nature of the soul (Atman)",
    "Detachment and renunciation (Vairagya)",
    "Surrender to God (Sharanagati)",
    "Divine vs demoniac qualities in humans",
    "Food and its connection to the three Gunas",
    "Types of faith and worship",
    "Nature of God and His manifestations (Vibhuti)",
    "Maya — illusion and the material world",
    "Samkhya philosophy — Purusha and Prakriti",
    "Renunciation vs action — which is better",
    "Qualities of a Sthitaprajna (steady-minded person)",
    "The role of a Guru and the importance of learning",
    "Equality and equanimity in all situations",
    "Fear and how to overcome it",
    "Ego and pride — obstacles on the spiritual path",
    "Compassion, kindness and non-violence (Ahimsa)",
    "The cosmic form (Vishwaroop) and its significance",
    "Types of sacrifice (Yajna) described in the Gita",
    "Free will vs destiny in the Gita",
    "Leadership and decision-making lessons",
    "Mind control and dealing with a restless mind",
    "Work-life balance and duty vs personal desire",
    "The concept of Nishkama Karma — desireless action",
    "What happens after death according to the Gita",
    "How to deal with grief and loss",
    "The importance of consistency and discipline (Abhyasa)",
    "Self-doubt and Arjuna's crisis of confidence",
    "Types of knowledge — higher vs lower",
    "Friendship and loyalty in the Gita",
    "Time (Kala) and its supreme power",
    "Contentment and gratitude",
]

# ─── Narrative moments for Type 4 ───────────────────────────────
NARRATIVE_MOMENTS = [
    {"chapter": 1, "verses": "1-47", "event": "Arjuna sees his relatives on both sides of the battlefield and collapses with grief, refusing to fight"},
    {"chapter": 2, "verses": "1-10", "event": "Krishna rebukes Arjuna for his weakness and begins teaching — the transition from despair to wisdom"},
    {"chapter": 2, "verses": "11-30", "event": "Krishna reveals the immortality of the soul — the foundational teaching that the soul cannot be killed"},
    {"chapter": 2, "verses": "54-72", "event": "Arjuna asks 'What does a steady-minded person look like?' and Krishna describes the Sthitaprajna"},
    {"chapter": 3, "verses": "36-43", "event": "Arjuna asks 'What compels a person to commit sin?' and Krishna explains the role of desire and anger"},
    {"chapter": 4, "verses": "1-8", "event": "Krishna reveals that He incarnates age after age whenever dharma declines — the Avatara doctrine"},
    {"chapter": 4, "verses": "34-42", "event": "Krishna emphasizes the importance of seeking a Guru and gaining knowledge"},
    {"chapter": 7, "verses": "1-7", "event": "Krishna begins revealing His supreme divine nature and how everything rests in Him"},
    {"chapter": 9, "verses": "26-34", "event": "Krishna says even a leaf or flower offered with devotion is accepted — the most accessible teaching on Bhakti"},
    {"chapter": 10, "verses": "20-42", "event": "Krishna lists His divine manifestations — I am the Himalayas among mountains, Ganga among rivers"},
    {"chapter": 11, "verses": "1-20", "event": "Arjuna requests to see Krishna's cosmic form and is granted divine vision"},
    {"chapter": 11, "verses": "21-34", "event": "Arjuna sees the terrifying Vishwaroop — all beings rushing into Krishna's mouths — and asks 'Who are You?'"},
    {"chapter": 11, "verses": "35-55", "event": "Arjuna is overwhelmed with fear and begs Krishna to return to His normal form"},
    {"chapter": 12, "verses": "13-20", "event": "Krishna describes the qualities of His most dear devotee"},
    {"chapter": 16, "verses": "1-6", "event": "Krishna lists divine qualities vs demoniac qualities — the internal battle in every human"},
    {"chapter": 18, "verses": "57-66", "event": "Krishna's final and most intimate teaching — 'Surrender unto Me alone, I shall liberate you from all sins'"},
    {"chapter": 18, "verses": "72-78", "event": "The conclusion — Sanjaya's final words about the glory of the dialogue between Krishna and Arjuna"},
]


# ─── Prompt Templates ───────────────────────────────────────────

TYPE1_PROMPT = """You are an expert on the Bhagavad Gita with deep knowledge of all major commentaries.

Given the following verse data, generate a high-quality Q&A pair.

VERSE: {verse_id} (Chapter {chapter}, Verse {verse})
CHAPTER: {chapter_name} — {chapter_meaning}
SANSKRIT: {slok}
TRANSLITERATION: {transliteration}

ENGLISH TRANSLATIONS:
{english_translations}

ENGLISH COMMENTARIES:
{english_commentaries}

HINDI COMMENTARIES (use for deeper insight, but write answer in English):
{hindi_commentaries}

INSTRUCTIONS:
- Generate ONE Q&A pair where the question is a natural way someone would ask about this verse
- Vary the question style: "What does verse X.Y mean?", "Explain BG X.Y", "What is the teaching of Chapter X Verse Y?", "What does Krishna say in verse X.Y?" etc.
- The answer MUST include:
  1. The Sanskrit shloka (copy exactly)
  2. The transliteration
  3. A clear explanation synthesizing multiple translations
  4. ONE scholar's unique insight woven in naturally (name the scholar)
  5. Cross-reference to 1 related verse if applicable
- Answer length: 150-250 words
- Tone: knowledgeable teacher, not academic paper

OUTPUT FORMAT (strictly follow this):
QUESTION: <your question here>
ANSWER: <your answer here>"""

TYPE2_PROMPT = """You are a wise Bhagavad Gita life counselor who maps real-life problems to specific Gita verses.

LIFE SITUATION: {situation}

Here are ALL the verse summaries from the Bhagavad Gita (use these to find the most relevant verses):
{verse_summaries}

INSTRUCTIONS:
- Identify 1-3 specific verses that DIRECTLY address this life situation
- Generate a Q&A pair where:
  - The QUESTION is the life situation phrased as a personal question (first person, casual, emotional)
  - The ANSWER maps the situation to specific verses with:
    1. The Sanskrit shloka of the primary verse
    2. Clear explanation of how the verse applies
    3. ONE scholar's practical insight (name the scholar)
    4. Actionable takeaway the person can apply today
- Answer length: 200-300 words
- Tone: warm, empathetic, like a wise friend — not preachy

OUTPUT FORMAT (strictly follow this):
QUESTION: <personal question here>
ANSWER: <your answer here>"""

TYPE3_PROMPT = """You are a Bhagavad Gita thematic expert.

THEME: {theme}

Here are ALL the verse summaries from the Bhagavad Gita:
{verse_summaries}

INSTRUCTIONS:
- Identify ALL verses across the 18 chapters that discuss this theme
- Generate a Q&A pair where:
  - The QUESTION asks about this theme naturally: "What does the Gita say about X?", "Which verses discuss X?", "How does the Gita address X?"
  - The ANSWER clusters 3-5 most important verses with:
    1. Specific chapter:verse citations for each
    2. Sanskrit shloka for the MOST important verse on this theme
    3. Brief explanation of each verse's contribution to the theme
    4. How the teaching evolves across chapters (if applicable)
- Answer length: 250-350 words
- Organize from most important to least important verse

OUTPUT FORMAT (strictly follow this):
QUESTION: <your question here>
ANSWER: <your answer here>"""

TYPE4_PROMPT = """You are a Bhagavad Gita storyteller who knows the Mahabharata narrative deeply.

NARRATIVE MOMENT: {event}
CHAPTER: {chapter}, VERSES: {verses}

Here are the relevant verses with their translations:
{verse_data}

INSTRUCTIONS:
- Generate a Q&A pair about this narrative moment where:
  - The QUESTION asks about the story/context: "What was happening when...", "Why did Arjuna...", "What led to..."
  - The ANSWER provides:
    1. The narrative context (what happened before this moment)
    2. The emotional state of the characters
    3. Key Sanskrit shloka from this section
    4. Why this moment matters in the overall Gita narrative
- Answer length: 200-300 words
- Tone: engaging storyteller, not dry summary

OUTPUT FORMAT (strictly follow this):
QUESTION: <your question here>
ANSWER: <your answer here>"""

TYPE5_PROMPT = """You are a comparative philosophy scholar specializing in Bhagavad Gita commentaries.

VERSE: {verse_id} (Chapter {chapter}, Verse {verse})
SANSKRIT: {slok}
TRANSLITERATION: {transliteration}

ENGLISH TRANSLATIONS FROM DIFFERENT SCHOLARS:
{english_translations}

ENGLISH COMMENTARIES:
{english_commentaries}

HINDI COMMENTARIES (translate insights to English):
{hindi_commentaries}

INSTRUCTIONS:
- This is a philosophically rich verse with diverse interpretations
- Generate a Q&A pair where:
  - The QUESTION asks for a deep explanation: "Explain verse X.Y in depth", "What are the different interpretations of verse X.Y?", "What is the deeper meaning of BG X.Y?"
  - The ANSWER provides:
    1. Sanskrit shloka and transliteration
    2. The common/surface meaning all scholars agree on
    3. 2-3 DISTINCT philosophical interpretations from named scholars
    4. How these different readings lead to different practical implications
- Answer length: 300-400 words
- Name every scholar you reference

OUTPUT FORMAT (strictly follow this):
QUESTION: <your question here>
ANSWER: <your answer here>"""

TYPE6_PROMPT = """You are generating alternative phrasings for existing Bhagavad Gita Q&A pairs.

ORIGINAL QUESTION: {original_question}
ORIGINAL ANSWER: {original_answer}

INSTRUCTIONS:
- Generate 2 REPHRASED versions of the question that someone might naturally ask
- Include variations like:
  - Hinglish: "Verse 2.47 ka matlab kya hai?"
  - Colloquial: "What's the deal with that famous karma verse?"
  - Keyword search: "karmanyevadhikaraste meaning"
  - Specific: "What did Krishna tell Arjuna about action and results?"
- Keep the SAME answer for all (copy exactly)

OUTPUT FORMAT (strictly follow this):
QUESTION_1: <rephrased question 1>
QUESTION_2: <rephrased question 2>
ANSWER: <copy original answer exactly>"""


# ─── Helper Functions ────────────────────────────────────────────

def call_gemini(prompt: str, max_retries: int = 3) -> str:
    """Call Gemini API with retry logic."""
    global call_count
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 4096,
                },
            )
            call_count += 1
            time.sleep(DELAY_BETWEEN_CALLS)
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower():
                wait = 15 * (attempt + 1)
                print(f"    Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3)
                else:
                    return None
    return None


def parse_qa_response(response: str) -> list[dict]:
    """Parse the LLM response into Q&A pair(s)."""
    if not response:
        return []
    
    pairs = []
    
    # Handle Type 6 (multiple questions, one answer)
    if "QUESTION_1:" in response and "QUESTION_2:" in response:
        try:
            q1_start = response.index("QUESTION_1:") + len("QUESTION_1:")
            q2_start = response.index("QUESTION_2:")
            a_start = response.index("ANSWER:") + len("ANSWER:")
            
            q1 = response[q1_start:q2_start].strip()
            q2 = response[q2_start + len("QUESTION_2:"):response.index("ANSWER:")].strip()
            answer = response[a_start:].strip()
            
            if q1 and answer:
                pairs.append({"question": q1, "answer": answer})
            if q2 and answer:
                pairs.append({"question": q2, "answer": answer})
        except ValueError:
            pass
        return pairs
    
    # Handle standard single Q&A
    try:
        if "QUESTION:" in response and "ANSWER:" in response:
            q_start = response.index("QUESTION:") + len("QUESTION:")
            a_marker = response.index("ANSWER:")
            question = response[q_start:a_marker].strip()
            answer = response[a_marker + len("ANSWER:"):].strip()
            
            if question and answer:
                pairs.append({"question": question, "answer": answer})
    except ValueError:
        pass
    
    return pairs


def format_translations(translations: dict, max_scholars: int = 5) -> str:
    """Format translations dict into readable string for prompt."""
    if not translations:
        return "None available"
    items = list(translations.items())[:max_scholars]
    return "\n".join(f"- {author}: {text[:500]}" for author, text in items)


def format_commentaries(commentaries: dict, max_scholars: int = 3) -> str:
    """Format commentaries dict into readable string for prompt."""
    if not commentaries:
        return "None available"
    items = list(commentaries.items())[:max_scholars]
    return "\n".join(f"- {author}: {text[:800]}" for author, text in items)


def build_verse_summaries(verses: list) -> str:
    """Build a compact summary of all verses for Type 2/3 prompts."""
    summaries = []
    for v in verses:
        # Get the shortest English translation as summary
        if v["english_translations"]:
            first_trans = list(v["english_translations"].values())[0]
            summary = first_trans[:200]
        else:
            summary = "(no English translation available)"
        summaries.append(f"BG {v['chapter']}.{v['verse']}: {summary}")
    return "\n".join(summaries)


# ─── Generator Functions ────────────────────────────────────────

def generate_type1(verses: list, resume: bool = False) -> list:
    """Type 1: Verse Explanation — one per verse."""
    print("\n📖 Generating Type 1: Verse Explanations")
    pairs = []
    skip_ids = set()
    
    # Resume: load existing intermediate and skip already generated verses
    if resume:
        intermediate_path = PROCESSED_DIR / "qa_pairs_type1_intermediate.json"
        if intermediate_path.exists():
            with open(intermediate_path, "r", encoding="utf-8") as f:
                pairs = json.load(f)
            skip_ids = {p["verse_id"] for p in pairs if "verse_id" in p}
            print(f"  Resuming: loaded {len(pairs)} existing pairs, skipping {len(skip_ids)} verses")
    
    for i, v in enumerate(verses):
        if v["id"] in skip_ids:
            continue
        
        prompt = TYPE1_PROMPT.format(
            verse_id=v["id"],
            chapter=v["chapter"],
            verse=v["verse"],
            chapter_name=v["chapter_name"],
            chapter_meaning=v["chapter_meaning"],
            slok=v["slok"],
            transliteration=v["transliteration"],
            english_translations=format_translations(v["english_translations"]),
            english_commentaries=format_commentaries(v["english_commentaries"]),
            hindi_commentaries=format_commentaries(v["hindi_commentaries"]),
        )
        
        response = call_gemini(prompt)
        new_pairs = parse_qa_response(response)
        
        for p in new_pairs:
            p["type"] = "verse_explanation"
            p["verse_id"] = v["id"]
        pairs.extend(new_pairs)
        
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(verses)} verses | {len(pairs)} pairs generated")
            # Save intermediate results
            _save_intermediate(pairs, "type1")
    
    # Final save
    _save_intermediate(pairs, "type1")
    print(f"  ✅ Type 1 complete: {len(pairs)} pairs")
    return pairs


def generate_type2(verses: list) -> list:
    """Type 2: Life Situation Mapping."""
    print("\n🧘 Generating Type 2: Life Situation Mapping")
    pairs = []
    
    # Build compact verse summaries (truncated to fit context)
    verse_summaries = build_verse_summaries(verses)
    
    # Split summaries into chunks if too large (context limit)
    # Use ~first 300 most important verses for summary
    important_verses = [v for v in verses if v["richness_score"] > 50]
    if len(important_verses) < 200:
        important_verses = verses  # fallback to all
    summary_text = build_verse_summaries(important_verses[:350])
    
    for i, situation in enumerate(LIFE_SITUATIONS):
        prompt = TYPE2_PROMPT.format(
            situation=situation,
            verse_summaries=summary_text,
        )
        
        response = call_gemini(prompt)
        new_pairs = parse_qa_response(response)
        
        for p in new_pairs:
            p["type"] = "life_situation"
            p["situation"] = situation
        pairs.extend(new_pairs)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(LIFE_SITUATIONS)} situations | {len(pairs)} pairs")
            _save_intermediate(pairs, "type2")
    
    print(f"  ✅ Type 2 complete: {len(pairs)} pairs")
    return pairs


def generate_type3(verses: list) -> list:
    """Type 3: Thematic Query."""
    print("\n🎯 Generating Type 3: Thematic Queries")
    pairs = []
    
    summary_text = build_verse_summaries(verses[:400])
    
    for i, theme in enumerate(GITA_THEMES):
        prompt = TYPE3_PROMPT.format(
            theme=theme,
            verse_summaries=summary_text,
        )
        
        response = call_gemini(prompt)
        new_pairs = parse_qa_response(response)
        
        for p in new_pairs:
            p["type"] = "thematic"
            p["theme"] = theme
        pairs.extend(new_pairs)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(GITA_THEMES)} themes | {len(pairs)} pairs")
    
    print(f"  ✅ Type 3 complete: {len(pairs)} pairs")
    return pairs


def generate_type4(verses: list) -> list:
    """Type 4: Narrative Context."""
    print("\n📜 Generating Type 4: Narrative Context")
    pairs = []
    
    # Build a lookup for quick verse access
    verse_lookup = {(v["chapter"], v["verse"]): v for v in verses}
    
    for i, moment in enumerate(NARRATIVE_MOMENTS):
        # Get verses for this narrative moment
        ch = moment["chapter"]
        verse_range = moment["verses"]
        
        # Parse verse range (e.g., "1-47" or "54-72")
        if "-" in verse_range:
            start, end = map(int, verse_range.split("-"))
        else:
            start = end = int(verse_range)
        
        # Collect verse data for this range (limit to avoid context overflow)
        relevant_verses = []
        for v_num in range(start, min(end + 1, start + 10)):  # max 10 verses
            v = verse_lookup.get((ch, v_num))
            if v:
                trans = list(v["english_translations"].values())
                trans_text = trans[0][:300] if trans else "N/A"
                relevant_verses.append(f"BG {ch}.{v_num}: {v['slok'][:100]}\nTranslation: {trans_text}")
        
        verse_data = "\n\n".join(relevant_verses)
        
        prompt = TYPE4_PROMPT.format(
            event=moment["event"],
            chapter=ch,
            verses=verse_range,
            verse_data=verse_data,
        )
        
        response = call_gemini(prompt)
        new_pairs = parse_qa_response(response)
        
        for p in new_pairs:
            p["type"] = "narrative"
            p["chapter"] = ch
            p["verses"] = verse_range
        pairs.extend(new_pairs)
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(NARRATIVE_MOMENTS)} moments | {len(pairs)} pairs")
    
    print(f"  ✅ Type 4 complete: {len(pairs)} pairs")
    return pairs


def generate_type5(verses: list) -> list:
    """Type 5: Multi-Perspective — only for rich verses."""
    print("\n🔬 Generating Type 5: Multi-Perspective (rich verses)")
    
    # Select top verses by richness score
    rich_verses = sorted(verses, key=lambda v: v["richness_score"], reverse=True)
    top_rich = [v for v in rich_verses if v["richness_score"] > 100][:200]
    
    print(f"  Found {len(top_rich)} philosophically rich verses")
    pairs = []
    
    for i, v in enumerate(top_rich):
        prompt = TYPE5_PROMPT.format(
            verse_id=v["id"],
            chapter=v["chapter"],
            verse=v["verse"],
            slok=v["slok"],
            transliteration=v["transliteration"],
            english_translations=format_translations(v["english_translations"], max_scholars=6),
            english_commentaries=format_commentaries(v["english_commentaries"], max_scholars=3),
            hindi_commentaries=format_commentaries(v["hindi_commentaries"], max_scholars=2),
        )
        
        response = call_gemini(prompt)
        new_pairs = parse_qa_response(response)
        
        for p in new_pairs:
            p["type"] = "multi_perspective"
            p["verse_id"] = v["id"]
        pairs.extend(new_pairs)
        
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(top_rich)} verses | {len(pairs)} pairs")
            _save_intermediate(pairs, "type5")
    
    print(f"  ✅ Type 5 complete: {len(pairs)} pairs")
    return pairs


def generate_type6(existing_pairs: list, count: int = 250) -> list:
    """Type 6: Rephrased Variations of existing pairs."""
    print("\n🔄 Generating Type 6: Rephrased Variations")
    
    # Sample from Type 1 and Type 2 pairs (most useful to rephrase)
    candidates = [p for p in existing_pairs if p["type"] in ("verse_explanation", "life_situation")]
    if len(candidates) > count:
        candidates = random.sample(candidates, count)
    
    pairs = []
    
    for i, original in enumerate(candidates):
        prompt = TYPE6_PROMPT.format(
            original_question=original["question"],
            original_answer=original["answer"],
        )
        
        response = call_gemini(prompt)
        new_pairs = parse_qa_response(response)
        
        for p in new_pairs:
            p["type"] = "rephrased"
            p["original_type"] = original["type"]
            if "verse_id" in original:
                p["verse_id"] = original["verse_id"]
        pairs.extend(new_pairs)
        
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(candidates)} variations | {len(pairs)} pairs")
    
    print(f"  ✅ Type 6 complete: {len(pairs)} pairs")
    return pairs


# ─── Utilities ───────────────────────────────────────────────────

def _save_intermediate(pairs: list, label: str):
    """Save intermediate results to avoid data loss."""
    path = PROCESSED_DIR / f"qa_pairs_{label}_intermediate.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)


def save_all_pairs(all_pairs: list):
    """Save final combined QA pairs."""
    # Remove intermediate metadata, keep clean
    clean_pairs = []
    for p in all_pairs:
        clean = {
            "question": p["question"],
            "answer": p["answer"],
            "type": p["type"],
        }
        # Keep optional metadata
        for key in ["verse_id", "theme", "situation", "chapter"]:
            if key in p:
                clean[key] = p[key]
        clean_pairs.append(clean)
    
    path = PROCESSED_DIR / "qa_pairs_all.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(clean_pairs)} QA pairs to {path}")
    
    # Print distribution
    from collections import Counter
    type_dist = Counter(p["type"] for p in clean_pairs)
    print("\nDistribution:")
    for t, count in type_dist.most_common():
        print(f"  {t}: {count}")


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Gita QA pairs")
    parser.add_argument("--types", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Which QA types to generate (1-6)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume Type 1 from where it left off (skips already generated verses)")
    args = parser.parse_args()
    
    # Initialize Gemini client
    global client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env!")
        return
    client = genai.Client(api_key=api_key)
    print(f"✅ Gemini client initialized (model: {MODEL})")
    
    # Load scraped data
    verses_path = RAW_DIR / "all_verses.json"
    if not verses_path.exists():
        print(f"❌ {verses_path} not found. Run 01_scrape_verses.py first!")
        return
    
    with open(verses_path, "r", encoding="utf-8") as f:
        verses = json.load(f)
    print(f"Loaded {len(verses)} verses from {verses_path}")
    
    # Load chapters for context
    chapters_path = RAW_DIR / "chapters.json"
    if chapters_path.exists():
        with open(chapters_path, "r", encoding="utf-8") as f:
            chapters = json.load(f)
        # Add chapter info to verses
        ch_info = {ch["chapter_number"]: ch for ch in chapters}
        for v in verses:
            ch = ch_info.get(v["chapter"], {})
            v["chapter_name"] = ch.get("name_translation", "")
            v["chapter_meaning"] = ch.get("name_meaning", "")
    
    # Generate requested types
    all_pairs = []
    
    if 1 in args.types:
        all_pairs.extend(generate_type1(verses, resume=args.resume))
    
    if 2 in args.types:
        all_pairs.extend(generate_type2(verses))
    
    if 3 in args.types:
        all_pairs.extend(generate_type3(verses))
    
    if 4 in args.types:
        all_pairs.extend(generate_type4(verses))
    
    if 5 in args.types:
        all_pairs.extend(generate_type5(verses))
    
    if 6 in args.types:
        # Type 6 needs existing pairs from other types
        if all_pairs:
            all_pairs.extend(generate_type6(all_pairs))
        else:
            # Load previously generated pairs
            prev_path = PROCESSED_DIR / "qa_pairs_all.json"
            if prev_path.exists():
                with open(prev_path) as f:
                    prev_pairs = json.load(f)
                all_pairs.extend(generate_type6(prev_pairs))
            else:
                print("  ⚠️ No existing pairs for Type 6. Skipping.")
    
    # Save everything
    if all_pairs:
        save_all_pairs(all_pairs)
    else:
        print("No pairs generated!")
    
    print(f"\n📊 Total Gemini API calls made: {call_count}")


if __name__ == "__main__":
    main()