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
    raise RuntimeError(f"\nMissing environment variables:\n{', '.join(missing)}")

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

if GROQ_API_KEY:
    try:
        groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        print("✅ GROQ READY")
    except Exception as e:
        print("❌ GROQ INIT ERROR:", e)

if SAMBANOVA_API_KEY:
    try:
        samba_client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")
        print("✅ SAMBANOVA READY")
    except Exception as e:
        print("❌ SAMBANOVA INIT ERROR:", e)

if HF_TOKEN:
    try:
        hf_client = InferenceClient(token=HF_TOKEN)
        print("✅ HF READY")
    except Exception as e:
        print("❌ HF INIT ERROR:", e)

GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
SAMBANOVA_MODELS = ["Meta-Llama-3.3-70B-Instruct"]
HF_MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message[:4000]}
    try:
        response = requests.post(url, data=payload, timeout=30)
        print("\n========== TELEGRAM RESPONSE ==========")
        print(response.text)
    except Exception as e:
        print("\n❌ TELEGRAM ERROR:", e)

def clean_url(url):
    try:
        return url.split("&")[0]
    except:
        return url

def shorten_url(long_url):
    try:
        api_url = f"https://tinyurl.com/api-create.php?url={long_url}"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            return response.text.strip()
        return long_url
    except Exception as e:
        print("\n❌ SHORT LINK ERROR:", e)
        return long_url

def extract_real_url(google_url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(google_url, headers=headers, timeout=15)
        # CHANGED: 'lxml' se 'html.parser' kiya taaki GitHub runner pe error na aaye
        soup = BeautifulSoup(response.text, "html.parser")
        canonical = soup.find("link", rel="canonical")
        if canonical and canonical.get("href"):
            real_url = canonical["href"]
            if "news.google.com" not in real_url:
                return real_url
        return google_url
    except Exception as e:
        return google_url

def fetch_full_article(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text and len(text) > 300:
            return text[:5000]
        return None
    except Exception as e:
        return None

# Thread worker function to process individual RSS entries in parallel
def process_single_entry(entry, category):
    try:
        published = entry.get("published_parsed")
        if not published:
            return None
        
        published_date = datetime(*published[:6]).date()
        today = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        
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
    except Exception as e:
        pass
    return None

# =========================================================
# AI GENERATION
# =========================================================

def generate_ai_response(prompt, headline, category, article_text):
    if groq_client:
        for model_name in GROQ_MODELS:
            try:
                print(f"\n🔥 USING GROQ: {model_name}")
                response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=900
                )
                content = response.choices[0].message.content
                if content: return content
            except Exception as e:
                print(f"❌ GROQ FAILED: {model_name}\n{e}")

    if samba_client:
        for model_name in SAMBANOVA_MODELS:
            try:
                print(f"\n🔥 USING SAMBANOVA: {model_name}")
                response = samba_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=900
                )
                content = response.choices[0].message.content
                if content: return content
            except Exception as e:
                print(f"❌ SAMBANOVA FAILED: {model_name}\n{e}")

    if hf_client:
        for model_name in HF_MODELS:
            try:
                print(f"\n🔥 USING HF: {model_name}")
                response = hf_client.chat_completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=900
                )
                content = response.choices[0].message.content
                if content: return content
            except Exception as e:
                print(f"❌ HF FAILED: {model_name}\n{e}")

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
आप इस खबर पर what सोचते हैं?

HASHTAGS:
#BreakingNews #Trending #ViralNews #karanpatelkushinagar
"""

# =========================================================
# RSS FEEDS
# =========================================================

RSS_FEEDS = {
    "India": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "Technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "World": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
    "Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "Entertainment": "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-IN&gl=IN&ceid=IN:en",
    "Gorakhpur": "https://news.google.com/rss/headlines/section/geo/Gorakhpur?hl=en-IN&gl=IN&ceid=IN:en",
    "Kushinagar": "https://news.google.com/rss/headlines/section/geo/Kushinagar?hl=en-IN&gl=IN&ceid=IN:en"
}

# =========================================================
# FETCH NEWS (PARALLEL MULTI-THREADED EXTRACTION)
# =========================================================

all_news = []
tasks = []

print("\n========== FETCHING NEWS (FAST MULTI-THREADED) ==========\n")

# Phase 1: Rapidly parse RSS metadata
for category, url in RSS_FEEDS.items():
    print(f"Fetching RSS: {category}")
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            tasks.append((entry, category))
    except Exception as e:
        print(f"❌ RSS ERROR: {category} - {e}")

# Phase 2: Run web scraping threads concurrently to prevent workflow timeout
print(f"\nTotal items found to check: {len(tasks)}. Extracting content parallelly...")
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = [executor.submit(process_single_entry, entry, cat) for entry, cat in tasks]
    for future in as_completed(futures):
        res = future.result()
        if res:
            all_news.append(res)

# =========================================================
# DATAFRAME & DE-DUPLICATION
# =========================================================

df = pd.DataFrame(all_news)
if df.empty:
    print("\n❌ NO NEWS FOUND")
    exit()

df["headline_clean"] = df["headline"].str.lower().str.strip()
df.drop_duplicates(subset=["headline_clean"], inplace=True)
df.drop(columns=["headline_clean"], inplace=True)

print("\nTOTAL UNIQUE NEWS ARTICLES PREPARED:", len(df))

# =========================================================
# AI ANALYSIS
# =========================================================

results = []
print("\n========== AI ANALYSIS STARTED ==========\n")

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
You are a world-class viral strategist, Hindi news writer, Facebook content creator, Instagram reel expert, and emotional storytelling analyst.
Analyze this news carefully. Avoid boring scoring explanations! Just output raw integers directly for ratings.

HEADLINE: {headline}
FULL ARTICLE: {article_text}

STRICT FORMAT:
Emotion Score: number/10
Virality Score: number/10
Political Toxicity: number/10

HINDI FACEBOOK ARTICLE:
(one viral line hook with italic font and 150-550 words detailed Hindi article and ENDING CTA:one engagement line in hindi)

HASHTAGS:
(viral english hashtags only + #karanpatelkushinagar)
"""

    try:
        ai_result = generate_ai_response(prompt, headline, category, article_text)

        print("\n========== AI RESPONSE ==========\n")
        print(ai_result)

        emotion_match = re.search(r"Emotion Score:\s*(\d+)", ai_result)
        virality_match = re.search(r"Virality Score:\s*(\d+)", ai_result)
        toxicity_match = re.search(r"Political Toxicity:\s*(\d+)", ai_result)

        # CHANGED: Fixed the logic bug here so toxicity scores read cleanly without bugs
        emotion_score = int(emotion_match.group(1)) if emotion_match else 5
        virality_score = int(virality_match.group(1)) if virality_match else 5
        toxicity_score = int(toxicity_match.group(1)) if toxicity_match else 0

        final_score = (virality_score * 0.5) + (emotion_score * 0.35) - (toxicity_score * 0.15)
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

        # RESTORED: 2-second delay to safely guard your API rate limits
        time.sleep(2)

    except Exception as e:
        print("\n❌ AI ANALYSIS ERROR:", e)

# =========================================================
# FINAL RESULTS & TELEGRAM DELIVERY
# =========================================================

results_df = pd.DataFrame(results)

if not results_df.empty:
    results_df = results_df.sort_values(by="final_score", ascending=False)
    results_df = results_df.head(10)

    csv_path = os.path.join(SCRIPT_DIR, "viral_reel_news.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n========== SENDING NEWS TO TELEGRAM ==========\n")

    for i, row in results_df.iterrows():
        cleaned_link = clean_url(row["link"])
        short_link = shorten_url(cleaned_link)

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

==========================

📲 FOLLOW ON :
📸 Instagram: https://www.instagram.com/karan_patel_kushinagar
▶️ YouTube: https://youtube.com/@karan_patel_kushinagar
📘 Facebook: https://www.facebook.com/share/1D627R2qFV/
📲 WHATSAPP BROADCAST: https://whatsapp.com/channel/0029Vb6Aycg3AzNOqOekVE0S
"""
        send_telegram_message(telegram_message)
        print(f"✅ SENT NEWS #{i+1}")

        # RESTORED: 3-second delay for Telegram messaging intervals
        time.sleep(3)

    print("\n✅ CSV SAVED:", csv_path)
else:
    print("\n❌ NO HIGH VIRAL NEWS FOUND ABOVE SCORE THRESHOLD")

