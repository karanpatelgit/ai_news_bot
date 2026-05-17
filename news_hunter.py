# =========================================================
# NEWS HUNTER BOT
# =========================================================

import feedparser
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from openai import OpenAI
from huggingface_hub import InferenceClient

import os
import time
import re
import requests

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

# =========================================================
# API TOKENS
# =========================================================

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
# HUGGING FACE CLIENT
# =========================================================

if HF_TOKEN:

    try:

        hf_client = InferenceClient(
            token=HF_TOKEN
        )

        print("✅ HUGGING FACE READY")

    except Exception as e:

        print("❌ HF INIT ERROR:", e)

# =========================================================
# MODEL FALLBACK LISTS
# =========================================================

GROQ_MODELS = [

    "llama-3.3-70b-versatile",

    "llama-3.1-8b-instant"
]

SAMBANOVA_MODELS = [

    "Meta-Llama-3.3-70B-Instruct"
]

HF_MODELS = [

    "meta-llama/Meta-Llama-3-8B-Instruct",

    "mistralai/Mistral-7B-Instruct-v0.2"
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
# SHORT LINK
# =========================================================

def shorten_url(long_url):

    try:

        api_url = (
            f"https://is.gd/create.php?format=simple&url={long_url}"
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

def extract_real_url(google_link):

    try:

        response = requests.get(

            google_link,

            allow_redirects=True,

            timeout=10
        )

        return response.url

    except Exception as e:

        print("\n❌ REAL URL ERROR:")
        print(e)

        return google_link

# =========================================================
# AI GENERATION FUNCTION
# =========================================================

def generate_ai_response(prompt):

    # =====================================================
    # TRY GROQ MODELS
    # =====================================================

    if groq_client:

        for model_name in GROQ_MODELS:

            try:

                print(f"\n🔥 USING GROQ MODEL: {model_name}")

                response = groq_client.chat.completions.create(

                    model=model_name,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.8,

                    max_tokens=700
                )

                return response.choices[0].message.content

            except Exception as e:

                print(f"\n❌ GROQ MODEL FAILED: {model_name}")
                print(e)

    # =====================================================
    # TRY SAMBANOVA MODELS
    # =====================================================

    if samba_client:

        for model_name in SAMBANOVA_MODELS:

            try:

                print(f"\n🔥 USING SAMBANOVA MODEL: {model_name}")

                response = samba_client.chat.completions.create(

                    model=model_name,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    temperature=0.8,

                    max_tokens=700
                )

                return response.choices[0].message.content

            except Exception as e:

                print(f"\n❌ SAMBANOVA MODEL FAILED: {model_name}")
                print(e)

    # =====================================================
    # TRY HUGGING FACE MODELS
    # =====================================================

    if hf_client:

        for model_name in HF_MODELS:

            try:

                print(f"\n🔥 USING HF MODEL: {model_name}")

                response = hf_client.chat_completion(

                    model=model_name,

                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],

                    max_tokens=700
                )

                return response.choices[0].message.content

            except Exception as e:

                print(f"\n❌ HF MODEL FAILED: {model_name}")
                print(e)

    return "AI GENERATION FAILED"

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
        "https://news.google.com/rss/headlines/section/geo/Kushinagar?hl=en-IN&gl=IN&ceid=IN:en",

    "Uttar Pradesh":
        "https://news.google.com/rss/headlines/section/geo/Uttar+Pradesh?hl=en-IN&gl=IN&ceid=IN:en"
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

                published_date = datetime(
                    *published[:6],
                    tzinfo=timezone.utc
                )

                now = datetime.now(timezone.utc)

                difference = now - published_date

                # ONLY TODAY + YESTERDAY
                if difference <= timedelta(days=2):

                    all_news.append({

                        "category": category,

                        "headline": entry.title,

                        "link": entry.link
                    })

            except Exception as e:

                print("❌ DATE FILTER ERROR:", e)

    except Exception as e:

        print(f"❌ ERROR FETCHING {category}")
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
# RESULTS STORAGE
# =========================================================

results = []

print("\n========== AI VIRAL ANALYSIS STARTED ==========\n")

# =========================================================
# ANALYZE NEWS
# =========================================================

for index, row in df.iterrows():

    headline = row["headline"]

    category = row["category"]

    link = row["link"]

    print("\n" + "=" * 80)
    print(f"CATEGORY: {category}")
    print(f"HEADLINE: {headline}")
    print("=" * 80)

    prompt = f"""
You are a world-class viral content strategist,
facebook article writer,
Instagram reel expert,
YouTube Shorts expert,
and emotional storytelling analyst.

Analyze this news headline carefully.

HEADLINE:
{headline}

Your job:
- detect emotional power
- detect viral potential
- detect audience curiosity
- detect controversy
- detect shareability

Avoid boring analysis.

Give scores STRICTLY in this format:

Emotion Score: number/10
Virality Score: number/10
Political Toxicity: number/10

Then continue with:

HOOK:
(one ultra-viral opening line)

HINDI FACEBOOK ARTICLE:
(Write a professional engaging Facebook article in Hindi between 150-350 words)

ENDING CTA:
(one audience engagement line)

HASHTAGS:
(only trending hashtags + #karanpatelkushinagar)

SOCIAL MEDIA IMAGE PROMPT:
(Create detailed image prompt for 1080x1350 social media poster)
"""

    try:

        ai_result = generate_ai_response(prompt)

        print("\n========== AI RESPONSE ==========\n")
        print(ai_result)

        # =================================================
        # EXTRACT SCORES
        # =================================================

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
        ) if emotion_match else 0

        virality_score = int(
            virality_match.group(1)
        ) if virality_match else 0

        toxicity_score = int(
            toxicity_match.group(1)
        ) if toxicity_match else 0

        # =================================================
        # FINAL SCORE
        # =================================================

        final_score = (

            virality_score * 0.5 +

            emotion_score * 0.35 -

            toxicity_score * 0.15
        )

        final_score = round(final_score, 2)

        print("\n========== SCORES ==========")

        print("Emotion Score:", emotion_score)

        print("Virality Score:", virality_score)

        print("Political Toxicity:", toxicity_score)

        print("Final Viral Score:", final_score)

        # =================================================
        # FILTER
        # =================================================

        if final_score >= 3:

            print("\n✅ HIGH VIRAL POTENTIAL DETECTED")

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

        else:

            print("\n❌ LOW VIRAL POTENTIAL")

        time.sleep(2)

    except Exception as e:

        print("\n❌ AI ERROR:")
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

    # TOP 10 ONLY
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

        real_link = extract_real_url(
            row["link"]
        )

        short_link = shorten_url(
            real_link
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

{row['ai_analysis'][:2000]}
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
        time.sleep(3)

    print("\n✅ CSV SAVED:")
    print(csv_path)

else:

    print("\n❌ NO HIGH VIRAL NEWS FOUND")
{row['ai_analysis'][:2000]}
"""

        send_telegram_message(
            telegram_message
        )

        print(f"✅ Sent news #{i+1}")

        time.sleep(3)

    print("\n✅ CSV SAVED:")
    print(csv_path)

else:

    print("\n❌ NO HIGH-VIRAL NEWS FOUND")
"""


