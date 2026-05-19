import feedparser
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from huggingface_hub import InferenceClient
from bs4 import BeautifulSoup
import trafilatura
import os
import time
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

missing = []

if not BOT_TOKEN:
    missing.append("TELEGRAM_BOT_TOKEN")

if not CHAT_ID:
    missing.append("TELEGRAM_CHAT_ID")

if missing:
    raise RuntimeError(
        f"""
Missing environment variables:

{', '.join(missing)}
"""
    )

print("\n========== ENV VARIABLES LOADED ==========")

print("GROQ:", bool(GROQ_API_KEY))
print("SAMBANOVA:", bool(SAMBANOVA_API_KEY))
print("HF:", bool(HF_TOKEN))
print("BOT_TOKEN:", bool(BOT_TOKEN))
print("CHAT_ID:", bool(CHAT_ID))

# =========================================================
# CONFIGURATION - PERFORMANCE OPTIMIZATIONS
# =========================================================

MAX_ARTICLES_TO_PROCESS = None  # NO LIMIT - process all articles
MAX_WORKERS = 5  # Parallel processing threads
ARTICLE_FETCH_TIMEOUT = 8  # Reduced from 15
URL_EXTRACTION_TIMEOUT = 8  # Reduced from 15
AI_SLEEP_TIME = 1  # Reduced from 2
TELEGRAM_SLEEP_TIME = 2  # Reduced from 3

# Local news bonus
LOCAL_NEWS_BONUS = 2.0  # Bonus points for Kushinagar/Gorakhpur news

# =========================================================
# AI CLIENTS
# =========================================================

groq_client = None
samba_client = None
hf_client = None

# =========================================================
# GROQ CLIENT
# =========================================================

if GROQ_API_KEY:
    try:
        groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        print("✅ GROQ READY")
    except Exception as e:
        print("❌ GROQ INIT ERROR:", e)

# =========================================================
# SAMBANOVA CLIENT
# =========================================================

if SAMBANOVA_API_KEY:
    try:
        samba_client = OpenAI(
            api_key=SAMBANOVA_API_KEY,
            base_url="https://api.sambanova.ai/v1"
        )
        print("✅ SAMBANOVA READY")
    except Exception as e:
        print("❌ SAMBANOVA INIT ERROR:", e)

# =========================================================
# HF CLIENT
# =========================================================

if HF_TOKEN:
    try:
        hf_client = InferenceClient(
            token=HF_TOKEN
        )
        print("✅ HF READY")
    except Exception as e:
        print("❌ HF INIT ERROR:", e)

# =========================================================
# MODEL LISTS
# =========================================================

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

SAMBANOVA_MODELS = [
    "Meta-Llama-3.3-70B-Instruct"
]

HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

# =========================================================
# TELEGRAM FUNCTION
# =========================================================

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    # Split into 4000-char chunks so no content is cut off
    chunk_size = 4000
    chunks = [message[i:i+chunk_size] for i in range(0, len(message), chunk_size)]

    for chunk in chunks:
        payload = {
            "chat_id": CHAT_ID,
            "text": chunk
        }
        try:
            response = requests.post(
                url,
                data=payload,
                timeout=30
            )
            print("\n========== TELEGRAM RESPONSE ==========")
            print(response.text)
            if len(chunks) > 1:
                time.sleep(1)  # small delay between chunks
        except Exception as e:
            print("\n❌ TELEGRAM ERROR:")
            print(e)

# =========================================================
# CLEAN URL
# =========================================================

def clean_url(url):
    try:
        return url.split("&")[0]
    except:
        return url

# =========================================================
# SHORT URL
# =========================================================

def shorten_url(long_url):
    try:
        api_url = (
            f"https://tinyurl.com/api-create.php?url={long_url}"
        )
        response = requests.get(
            api_url,
            timeout=10
        )
        if response.status_code == 200:
            return response.text.strip()
        return long_url
    except Exception as e:
        print("\n❌ SHORT LINK ERROR:")
        print(e)
        return long_url

# =========================================================
# EXTRACT REAL URL - OPTIMIZED
# =========================================================

def extract_real_url(google_url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(
            google_url,
            headers=headers,
            timeout=URL_EXTRACTION_TIMEOUT  # Reduced timeout
        )
        soup = BeautifulSoup(
            response.text,
            "lxml"
        )
        canonical = soup.find(
            "link",
            rel="canonical"
        )
        if canonical and canonical.get("href"):
            real_url = canonical["href"]
            if "news.google.com" not in real_url:
                return real_url
        return google_url
    except Exception as e:
        print(f"\n❌ REAL URL EXTRACTION ERROR: {e}")
        return google_url

# =========================================================
# FETCH FULL ARTICLE - OPTIMIZED
# =========================================================

def fetch_full_article(url):
    try:
        downloaded = trafilatura.fetch_url(
            url,
            timeout=ARTICLE_FETCH_TIMEOUT  # Reduced timeout
        )
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False
        )
        if text and len(text) > 300:
            return text[:5000]
        return None
    except Exception as e:
        print(f"\n❌ ARTICLE FETCH ERROR: {e}")
        return None

# =========================================================
# PROCESS SINGLE ENTRY - FOR PARALLEL PROCESSING
# =========================================================

def process_single_entry(entry, category):
    """Process a single news entry with URL extraction and article fetch"""
    try:
        published = entry.get("published_parsed")
        
        if not published:
            return None
        
        # Article publish date
        published_date = datetime(
            *published[:6]
        ).date()
        
        # Today's date (UTC)
        today = datetime.utcnow().date()
        
        # Yesterday's date
        yesterday = today - timedelta(days=1)
        
        # ONLY TODAY + YESTERDAY NEWS
        if published_date == today or published_date == yesterday:
            
            real_link = extract_real_url(entry.link)
            article_text = fetch_full_article(real_link)
            
            if not article_text:
                article_text = entry.title
            
            return {
                "category": category,
                "headline": entry.title,
                "link": real_link,
                "article": article_text
            }
        
        return None
        
    except Exception as e:
        print(f"❌ ENTRY PROCESSING ERROR: {e}")
        return None

# =========================================================
# AI GENERATION
# =========================================================

def generate_ai_response(
    prompt,
    headline,
    category,
    article_text
):
    # =====================================================
    # GROQ
    # =====================================================
    if groq_client:
        for model_name in GROQ_MODELS:
            try:
                print(f"\n🔥 USING GROQ: {model_name}")
                response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.8,
                    max_tokens=900
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"\n❌ GROQ FAILED: {model_name}")
                print(e)

    # =====================================================
    # SAMBANOVA
    # =====================================================
    if samba_client:
        for model_name in SAMBANOVA_MODELS:
            try:
                print(f"\n🔥 USING SAMBANOVA: {model_name}")
                response = samba_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.8,
                    max_tokens=900
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"\n❌ SAMBANOVA FAILED: {model_name}")
                print(e)

    # =====================================================
    # HUGGING FACE
    # =====================================================
    if hf_client:
        for model_name in HF_MODELS:
            try:
                print(f"\n🔥 USING HF: {model_name}")
                response = hf_client.chat_completion(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.8,
                    max_tokens=900
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"\n❌ HF FAILED: {model_name}")
                print(e)

    return "AI generation failed for this news."

# =========================================================
# RSS FEEDS - LOCAL NEWS FIRST
# =========================================================

RSS_FEEDS = {
    # LOCAL NEWS FIRST - PRIORITY
    "Kushinagar": "https://news.google.com/rss/headlines/section/geo/Kushinagar?hl=en-IN&gl=IN&ceid=IN:en",
    "Gorakhpur": "https://news.google.com/rss/headlines/section/geo/Gorakhpur?hl=en-IN&gl=IN&ceid=IN:en",
    
    # OTHER NEWS
    "India": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "Technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "World": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
    "Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "Entertainment": "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-IN&gl=IN&ceid=IN:en",
}

# =========================================================
# FETCH NEWS WITH PARALLEL PROCESSING
# =========================================================

all_news = []

print("\n========== FETCHING NEWS (PARALLEL) ==========\n")

for category, url in RSS_FEEDS.items():
    print(f"Fetching: {category}")
    
    try:
        feed = feedparser.parse(url)
        
        # No limit - process all entries from feed
        entries = feed.entries
        
        # Process entries in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            process_func = partial(process_single_entry, category=category)
            futures = [executor.submit(process_func, entry) for entry in entries]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        all_news.append(result)
                except Exception as e:
                    print(f"❌ FUTURE ERROR: {e}")
        
        print(f"✅ {category}: {len([n for n in all_news if n['category'] == category])} articles")
        
    except Exception as e:
        print(f"❌ RSS ERROR: {category}")
        print(e)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(all_news)

if df.empty:
    print("\n❌ NO NEWS FOUND")
    exit()

# =========================================================
# REMOVE DUPLICATES
# =========================================================

df["headline_clean"] = (
    df["headline"]
    .str.lower()
    .str.strip()
)

df.drop_duplicates(
    subset=["headline_clean"],
    inplace=True
)

df.drop(
    columns=["headline_clean"],
    inplace=True
)

print(f"\nTOTAL UNIQUE NEWS: {len(df)}")

# =========================================================
# PRIORITIZE LOCAL NEWS
# =========================================================

# Separate local and other news
local_news = df[df['category'].isin(['Kushinagar', 'Gorakhpur'])]
other_news = df[~df['category'].isin(['Kushinagar', 'Gorakhpur'])]

print(f"LOCAL NEWS (Kushinagar/Gorakhpur): {len(local_news)}")
print(f"OTHER NEWS: {len(other_news)}")

# NO LIMIT - process ALL news (local first, then others)
local_limit = len(local_news)   # All local news
other_limit = len(other_news)   # All other news

df_to_process = pd.concat([
    local_news,
    other_news
])

print(f"\n📊 PROCESSING {len(df_to_process)} ARTICLES (Local: {local_limit}, Other: {other_limit})")

# =========================================================
# RESULTS
# =========================================================

results = []

print("\n========== AI ANALYSIS STARTED ==========\n")

# =========================================================
# ANALYZE NEWS
# =========================================================

for index, row in df_to_process.iterrows():
    headline = row["headline"]
    category = row["category"]
    link = row["link"]
    article_text = row["article"]
    
    is_local = category in ['Kushinagar', 'Gorakhpur']

    print("\n" + "=" * 80)
    print(f"CATEGORY: {category} {'🏠 LOCAL' if is_local else ''}")
    print(f"HEADLINE: {headline}")
    print("=" * 80)

    prompt = f"""
You are a world-class viral strategist,
Hindi news writer,
Facebook content creator,
Instagram reel expert,
and emotional storytelling analyst.

Analyze this news carefully.
(avoid boring scoring! Don't explanation anything about scoring just give the scores)

HEADLINE:
{headline}

FULL ARTICLE:
{article_text}

CATEGORY: {category}
{"🏠 LOCAL NEWS - Kushinagar/Gorakhpur area - Give extra importance!" if is_local else ""}

STRICT FORMAT:

Emotion Score: number/10
Virality Score: number/10
Political Toxicity: number/10

HINDI FACEBOOK ARTICLE:
(one viral line hook with italic font and 150-550 words detailed Hindi article and ENDING CTA:one engagement line in hindi)

HASHTAGS:
(viral english hashtags only + #karanpatelkushinagar {"+ #Kushinagar #Gorakhpur" if is_local else ""})

"""

    try:
        ai_result = generate_ai_response(
            prompt,
            headline,
            category,
            article_text
        )

        print("\n========== AI RESPONSE ==========\n")
        print(ai_result[:500] + "...")  # Print only first 500 chars

        emotion_match = re.search(
            r"Emotion Score:\s*(\d+)",
            ai_result
        )

        virality_match = re.search(
            r"Virality Score:\s*(\d+)",
            ai_result
        )

        toxicity_match = re.search(
            r"Political Toxicity:\s*(\d+)",
            ai_result
        )

        emotion_score = int(
            emotion_match.group(1)
        ) if emotion_match else 5

        virality_score = int(
            virality_match.group(1)
        ) if virality_match else 5

        toxicity_score = int(
            toxicity_match.group(1)
        ) if toxicity_match else 0

        final_score = (
            virality_score * 0.5 +
            emotion_score * 0.35 -
            toxicity_score * 0.15
        )
        
        # ADD BONUS FOR LOCAL NEWS
        if is_local:
            final_score += LOCAL_NEWS_BONUS
            print(f"🏠 LOCAL NEWS BONUS: +{LOCAL_NEWS_BONUS}")

        final_score = round(final_score, 2)
        
        print(f"📊 FINAL SCORE: {final_score}")

        if final_score >= 5:
            result_row = {
                "category": category,
                "headline": headline,
                "link": link,
                "emotion_score": emotion_score,
                "virality_score": virality_score,
                "toxicity_score": toxicity_score,
                "final_score": final_score,
                "ai_analysis": ai_result,
                "is_local": is_local
            }
            results.append(result_row)

            # =====================================================
            # SEND TO TELEGRAM IMMEDIATELY - timeout se pehle hi
            # deliver ho jaye, end ka wait nahi
            # =====================================================
            sent_count = len(results)
            cleaned_link = clean_url(link)
            short_link = shorten_url(cleaned_link)
            local_badge = "🏠 KUSHINAGAR/GORAKHPUR LOCAL NEWS\n" if is_local else ""

            telegram_message = f"""
🔥 VIRAL NEWS #{sent_count}

{local_badge}📰 {headline}

📂 Category: {category}

📈 Viral Score: {final_score}
❤️ Emotion: {emotion_score}/10
🚀 Virality: {virality_score}/10
⚠️ Toxicity: {toxicity_score}/10

🔗 Link:
{short_link}

==========================

{ai_result}

==========================

📲 FOLLOW KARAN PATEL KUSHINAGAR ON :

📸 Instagram:
https://www.instagram.com/karan_patel_kushinagar

▶️ YouTube:
https://youtube.com/@karan_patel_kushinagar

📘 Facebook:
https://www.facebook.com/share/1D627R2qFV/

📲 WHATSAPP BRODCAST:
https://whatsapp.com/channel/0029Vb6Aycg3AzNOqOekVE0S

"""
            send_telegram_message(telegram_message)
            print(f"✅ SENT IMMEDIATELY #{sent_count} {'🏠 LOCAL' if is_local else ''}")
            time.sleep(TELEGRAM_SLEEP_TIME)

        time.sleep(AI_SLEEP_TIME)

    except Exception as e:
        print("\n❌ AI ANALYSIS ERROR:")
        print(e)

# =========================================================
# SAVE CSV AT END (summary only - news already sent above)
# =========================================================

results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.sort_values(by="final_score", ascending=False)

    csv_path = os.path.join(SCRIPT_DIR, "viral_reel_news.csv")
    results_df.to_csv(csv_path, index=False)

    local_count = results_df['is_local'].sum()
    print(f"\n✅ DONE — Total sent: {len(results_df)} articles (Local: {local_count})")
    print(f"✅ CSV SAVED: {csv_path}")

else:
    print("\n❌ NO HIGH VIRAL NEWS FOUND")

