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

    payload = {

        "chat_id": CHAT_ID,

        "text": message[:4000]
    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=30
        )

        print("\n========== TELEGRAM RESPONSE ==========")
        print(response.text)

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
# EXTRACT REAL URL
# =========================================================

def extract_real_url(google_url):

    try:

        headers = {
            "User-Agent":
            "Mozilla/5.0"
        }

        response = requests.get(
            google_url,
            headers=headers,
            timeout=15
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

        print("\n❌ REAL URL EXTRACTION ERROR:")
        print(e)

        return google_url

# =========================================================
# FETCH FULL ARTICLE
# =========================================================

def fetch_full_article(url):

    try:

        downloaded = trafilatura.fetch_url(url)

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

        print("\n❌ ARTICLE FETCH ERROR:")
        print(e)

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

                    max_tokens=900
                )

                content = response.choices[0].message.content

                if content:
                    return content

            except Exception as e:

                print(f"\n❌ HF FAILED: {model_name}")
                print(e)

    # =====================================================
    # FALLBACK
    # =====================================================

    print("\n⚠️ ALL AI FAILED")

    return f"""
Emotion Score: 5/10
Virality Score: 5/10
Political Toxicity: 0/10

HOOK:
Breaking news everyone is discussing.

HINDI FACEBOOK ARTICLE:
📰 {headline}

📂 Category: {category}

{article_text[:1200]}

ENDING CTA:
आप इस खबर पर क्या सोचते हैं?

HASHTAGS:
#BreakingNews #Trending #ViralNews #karanpatelkushinagar

SOCIAL MEDIA IMAGE PROMPT:
Create 1080x1350 breaking news poster with Hindi typography and branding:
KARAN PATEL KUSHINAGAR
"""

# =========================================================
# RSS FEEDS
# =========================================================

RSS_FEEDS = {

    "India":
        "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",

    "Technology":
        "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",

    "World":
        "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",

    "Business":
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",

    "Entertainment":
        "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-IN&gl=IN&ceid=IN:en",

    "Gorakhpur":
        "https://news.google.com/rss/headlines/section/geo/Gorakhpur?hl=en-IN&gl=IN&ceid=IN:en",

    "Kushinagar":
        "https://news.google.com/rss/headlines/section/geo/Kushinagar?hl=en-IN&gl=IN&ceid=IN:en"
}

# =========================================================
# FETCH NEWS
# =========================================================

all_news = []

print("\n========== FETCHING NEWS ==========\n")

for category, url in RSS_FEEDS.items():

    print(f"Fetching: {category}")

    try:

        feed = feedparser.parse(url)

        for entry in feed.entries:

            try:

                    published = entry.get("published_parsed")
                    
                    if not published:
                        continue
                    
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
                    
                        real_link = extract_real_url(
                            entry.link
                        )
                    
                        article_text = fetch_full_article(
                            real_link
                        )
                    
                        if not article_text:
                            article_text = entry.title
                    
                        all_news.append({
                    
                            "category": category,
                    
                            "headline": entry.title,
                    
                            "link": real_link,
                    
                            "article": article_text
                        })

            except Exception as e:

                print("❌ ENTRY ERROR:", e)

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

print("\nTOTAL UNIQUE NEWS:", len(df))

# =========================================================
# RESULTS
# =========================================================

results = []

print("\n========== AI ANALYSIS STARTED ==========\n")

# =========================================================
# ANALYZE NEWS
# =========================================================

for index, row in df.iterrows():

    headline = row["headline"]
    category = row["category"]
    link = row["link"]
    article_text = row["article"]

    print("\n" + "=" * 80)
    print(f"CATEGORY: {category}")
    print(f"HEADLINE: {headline}")
    print("=" * 80)

    prompt = f"""
You are a world-class viral strategist,
Hindi news writer,
Facebook content creator,
Instagram reel expert,
and emotional storytelling analyst.

Analyze this news carefully.
(aviod boring scoring explanation just give the scores)

HEADLINE:
{headline}

FULL ARTICLE:
{article_text}

STRICT FORMAT:

Emotion Score: number/10
Virality Score: number/10
Political Toxicity: number/10

HOOK:
(one viral line)

HINDI FACEBOOK ARTICLE:
(150-550 words detailed Hindi article)

ENDING CTA:
(one engagement line in hindi)

HASHTAGS:
(viral english hashtags only + #karanpatelkushinagar)

SOCIAL MEDIA IMAGE PROMPT:
(Create detailed 1080x1350 poster prompt with brading Karan Patel Kushinagar)
"""

    try:

        ai_result = generate_ai_response(
            prompt,
            headline,
            category,
            article_text
        )

        print("\n========== AI RESPONSE ==========\n")
        print(ai_result)

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

        final_score = round(final_score, 2)

        if final_score >= 5:

            results.append({

                "category": category,

                "headline": headline,

                "link": link,

                "emotion_score": emotion_score,

                "virality_score": virality_score,

                "toxicity_score": toxicity_score,

                "final_score": final_score,

                "ai_analysis": ai_result
            })

        time.sleep(2)

    except Exception as e:

        print("\n❌ AI ANALYSIS ERROR:")
        print(e)

# =========================================================
# FINAL RESULTS
# =========================================================

results_df = pd.DataFrame(results)

if not results_df.empty:

    results_df = results_df.sort_values(
        by="final_score",
        ascending=False
    )

    results_df = results_df.head(10)

    csv_path = os.path.join(
        SCRIPT_DIR,
        "viral_reel_news.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    print("\n========== SENDING NEWS ==========\n")

    for i, row in results_df.iterrows():

        cleaned_link = clean_url(
            row["link"]
        )

        short_link = shorten_url(
            cleaned_link
        )

        telegram_message = f"""
🔥 VIRAL NEWS #{i+1}

📰 {row['headline']}

📂 Category: {row['category']}

📈 Viral Score: {row['final_score']}
❤️ Emotion: {row['emotion_score']}/10
🚀 Virality: {row['virality_score']}/10
⚠️ Toxicity: {row['toxicity_score']}/10

🔗 Link:
{short_link}

==========================

{row['ai_analysis'][:2500]}
"""

        send_telegram_message(
            telegram_message
        )

        print(f"✅ SENT NEWS #{i+1}")

        time.sleep(3)

    print("\n✅ CSV SAVED:")
    print(csv_path)

else:

    print("\n❌ NO HIGH VIRAL NEWS FOUND")

"""
