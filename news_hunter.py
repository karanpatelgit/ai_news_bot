import feedparser
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import time
import re
import requests

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([HF_TOKEN, BOT_TOKEN, CHAT_ID]):
    raise RuntimeError("Missing HF_TOKEN / TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")

print("✅ ENV LOADED")

# =========================================================
# HUGGING FACE CLIENT (FIXED)
# =========================================================

client = InferenceClient(
    provider="hf-inference",   # 🔥 IMPORTANT FIX
    token=HF_TOKEN
)

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"  # ✅ WORKING MODEL

# =========================================================
# TELEGRAM
# =========================================================

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message[:3500]
    }

    try:
        r = requests.post(url, data=payload, timeout=30)
        print("Telegram:", r.text)
    except Exception as e:
        print("Telegram error:", e)

# =========================================================
# SHORT URL
# =========================================================

def shorten_url(url):
    try:
        r = requests.get(
            f"https://tinyurl.com/api-create.php?url={url}",
            timeout=10
        )
        return r.text if r.status_code == 200 else url
    except:
        return url

# =========================================================
# RSS FEEDS
# =========================================================

RSS_FEEDS = {
    "India": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "Tech": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "World": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
    "Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
}

# =========================================================
# FETCH NEWS
# =========================================================

all_news = []

for cat, url in RSS_FEEDS.items():
    print(f"Fetching {cat}")
    feed = feedparser.parse(url)

    for entry in feed.entries[:15]:
        all_news.append({
            "category": cat,
            "headline": entry.title,
            "link": entry.link
        })

df = pd.DataFrame(all_news).drop_duplicates(subset=["headline"])

print("Total news:", len(df))

# =========================================================
# ANALYSIS
# =========================================================

results = []

for i, row in df.iterrows():

    headline = row["headline"]
    link = row["link"]
    category = row["category"]

    print("\n", "="*60)
    print("HEADLINE:", headline)

    prompt = f"""
You are a viral Hindi Facebook content writer.

Analyze this headline:

{headline}

Return format:

Emotion Score: X/10
Virality Score: X/10
Political Toxicity: X/10

HOOK:
(one emotional hook)

HINDI ARTICLE:
(emotional viral Hindi story)

HASHTAGS:
(trending hashtags)
"""

    try:
        response = client.chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        text = response.choices[0].message.content
        print("\nAI OUTPUT:\n", text)

        # ---------------- SCORES ----------------
        emotion = int(re.search(r"Emotion Score:\s*(\d+)", text).group(1) or 0)
        virality = int(re.search(r"Virality Score:\s*(\d+)", text).group(1) or 0)
        toxicity = int(re.search(r"Political Toxicity:\s*(\d+)", text).group(1) or 0)

        final_score = round((virality * 0.5) + (emotion * 0.35) - (toxicity * 0.15), 2)

        if final_score >= 1:

            results.append({
                "category": category,
                "headline": headline,
                "link": link,
                "emotion": emotion,
                "virality": virality,
                "toxicity": toxicity,
                "score": final_score,
                "ai": text
            })

            print("✅ SAVED")
        else:
            print("❌ SKIPPED")

        time.sleep(2)

    except Exception as e:
        print("AI ERROR:", e)

# =========================================================
# SEND TOP RESULTS
# =========================================================

if results:

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:10]

    for idx, r in enumerate(results):

        short = shorten_url(r["link"])

        msg = f"""
🔥 VIRAL NEWS #{idx+1}

📰 {r['headline']}

📊 Score: {r['score']}
❤️ Emotion: {r['emotion']}/10
🚀 Virality: {r['virality']}/10
⚠️ Toxicity: {r['toxicity']}/10

🔗 {short}

━━━━━━━━━━━━━━
{r['ai'][:1200]}
"""

        send_telegram_message(msg)
        time.sleep(4)

    print("✅ DONE")

else:
    print("❌ No viral news found")

    print("\n❌ NO VIRAL NEWS FOUND")
