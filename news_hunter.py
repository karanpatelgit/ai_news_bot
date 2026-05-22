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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =========================================================
# CONFIG
# =========================================================

MAX_WORKERS = 3
MAX_ARTICLES_TO_PROCESS = 25
AI_SLEEP_TIME = 1
TELEGRAM_SLEEP_TIME = 1
LOCAL_NEWS_BONUS = 2.0

# =========================================================
# AI CLIENTS
# =========================================================

groq_client = None
samba_client = None
hf_client = None
openrouter_client = None

# =========================================================
# GROQ
# =========================================================

if GROQ_API_KEY:
    try:
        groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        print("✅ GROQ READY")
    except Exception as e:
        print("❌ GROQ ERROR:", e)

# =========================================================
# SAMBANOVA
# =========================================================

if SAMBANOVA_API_KEY:
    try:
        samba_client = OpenAI(
            api_key=SAMBANOVA_API_KEY,
            base_url="https://api.sambanova.ai/v1"
        )
        print("✅ SAMBANOVA READY")
    except Exception as e:
        print("❌ SAMBANOVA ERROR:", e)

# =========================================================
# HF
# =========================================================

if HF_TOKEN:
    try:
        hf_client = InferenceClient(token=HF_TOKEN)
        print("✅ HF READY")
    except Exception as e:
        print("❌ HF ERROR:", e)


# =========================================================
# OPENROUTER CLIENT
# =========================================================


if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

        print("✅ OPENROUTER READY")

    except Exception as e:
        print("❌ OPENROUTER INIT ERROR:", e)

# =========================================================
# MODELS
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
OPENROUTER_MODELS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-32b:free",
    "google/gemma-3-27b-it:free"
]

# =========================================================
# TELEGRAM
# =========================================================

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

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

            print("✅ TELEGRAM SENT")
            print(response.text)

        except Exception as e:
            print("❌ TELEGRAM ERROR:", e)

# =========================================================
# URL HELPERS
# =========================================================

def clean_url(url):
    try:
        return url.split("&")[0]
    except:
        return url


def shorten_url(long_url):
    try:
        api_url = f"https://tinyurl.com/api-create.php?url={long_url}"

        response = requests.get(
            api_url,
            timeout=10
        )

        if response.status_code == 200:
            return response.text.strip()

        return long_url

    except Exception as e:
        print("❌ SHORT URL ERROR:", e)
        return long_url

# =========================================================
# REAL URL EXTRACTION
# =========================================================

def extract_real_url(google_url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            google_url,
            headers=headers,
            timeout=6
        )

        soup = BeautifulSoup(response.text, "lxml")

        canonical = soup.find("link", rel="canonical")

        if canonical and canonical.get("href"):
            real_url = canonical["href"]

            if "news.google.com" not in real_url:
                return real_url

        return google_url

    except Exception as e:
        print("❌ REAL URL ERROR:", e)
        return google_url

# =========================================================
# ARTICLE FETCH
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
        print(f"❌ ARTICLE FETCH ERROR: {e}")
        return None

# =========================================================
# PROCESS ENTRY
# =========================================================

def process_single_entry(entry, category):
    try:
        published = entry.get("published_parsed")

        if not published:
            return None

        published_datetime = datetime(
            *published[:6],
            tzinfo=timezone.utc
        )

        now = datetime.now(timezone.utc)

        time_diff = now - published_datetime

        if timedelta(0) <= time_diff <= timedelta(hours=8):

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
        print("❌ ENTRY PROCESS ERROR:", e)
        return None

# =========================================================
# AI GENERATION
# =========================================================

def generate_ai_response(prompt):

    # =====================================================
    # GROQ
    # =====================================================

    if groq_client:
        for model_name in GROQ_MODELS:

            try:
                print(f"🔥 USING GROQ: {model_name}")

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

                content = response.choices[0].message.content

                if content:
                    return content

            except Exception as e:
                print("❌ GROQ FAILED:", e)

    # =====================================================
    # SAMBANOVA
    # =====================================================

    if samba_client:
        for model_name in SAMBANOVA_MODELS:

            try:
                print(f"🔥 USING SAMBANOVA: {model_name}")

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

                content = response.choices[0].message.content

                if content:
                    return content

            except Exception as e:
                print("❌ SAMBANOVA FAILED:", e)

    # =====================================================
    # HUGGINGFACE
    # =====================================================

    if hf_client:
        for model_name in HF_MODELS:

            try:
                print(f"🔥 USING HF: {model_name}")

                response = hf_client.chat_completion(
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

                content = response.choices[0].message.content

                if content:
                    return content

            except Exception as e:
                print("❌ HF FAILED:", e)

    # =====================================================
    # OPENROUTER
    # =====================================================

    if openrouter_client:
        for model_name in OPENROUTER_MODELS:

            try:
                print(f"🔥 USING OPENROUTER: {model_name}")

                response = openrouter_client.chat.completions.create(
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

                content = response.choices[0].message.content

                if content:
                    return content

            except Exception as e:
                print(f"❌ OPENROUTER FAILED: {model_name}")
                print(e)

    # =====================================================
    # SMART FALLBACK FORMATTER
    # =====================================================

    print("\n⚠️ USING SMART FALLBACK FORMATTER")

    try:

        # Extract title
        title_match = re.search(
            r"NEWS TITLE:\s*(.+)",
            prompt
        )

        # Extract category
        category_match = re.search(
            r"CATEGORY:\s*(.+)",
            prompt
        )

        # Extract article
        article_match = re.search(
            r"FULL ARTICLE:\s*(.+?)SOURCE LINK:",
            prompt,
            re.DOTALL
        )

        # Extract link
        link_match = re.search(
            r"SOURCE LINK:\s*(.+)",
            prompt
        )

        headline = (
            title_match.group(1).strip()
            if title_match else "Breaking News"
        )

        category = (
            category_match.group(1).strip()
            if category_match else "News"
        )

        article_text = (
            article_match.group(1).strip()
            if article_match else headline
        )

        link = (
            link_match.group(1).strip()
            if link_match else ""
        )

        # Clean article
        article_text = article_text.replace("\n", " ")

        # Split sentences
        sentences = re.split(
            r'(?<=[.!?]) +',
            article_text
        )

        important_lines = []

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) > 40:
                important_lines.append(sentence)

            if len(important_lines) >= 4:
                break

        # Create summary
        summary = "\n\n".join([
            f"• {line}"
            for line in important_lines
        ])

        # Generate hashtags
        hashtags = (
            f"#{category.replace(' ', '')} "
            "#BreakingNews "
            "#IndiaNews "
            "#ViralNews"
        )

        # Final fallback post
        fallback_post = f"""
🔥 BREAKING NEWS

📰 {headline}

📂 Category: {category}

━━━━━━━━━━━━━━━

{summary}

━━━━━━━━━━━━━━━

🤔 आपकी क्या राय है?

📢 ऐसी ही खबरों के लिए follow करें।

🔗 Source:
{link}

{hashtags}

Emotion Score: 6/10
Virality Score: 6/10
Political Toxicity: 1/10
"""

        return fallback_post

    except Exception as e:

        print("❌ SMART FALLBACK ERROR:", e)

        return """
🔥 BREAKING NEWS

News update available.

Emotion Score: 5/10
Virality Score: 5/10
Political Toxicity: 0/10
"""
# =========================================================
# RSS FEEDS
# =========================================================

RSS_FEEDS = {
    "Kushinagar": "https://news.google.com/rss/headlines/section/geo/Kushinagar?hl=en-IN&gl=IN&ceid=IN:en",
    "Gorakhpur": "https://news.google.com/rss/headlines/section/geo/Gorakhpur?hl=en-IN&gl=IN&ceid=IN:en",
    "India": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "Technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "World": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
    "Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "Entertainment": "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-IN&gl=IN&ceid=IN:en"
}

# =========================================================
# FETCH NEWS
# =========================================================

all_news = []

print("========== FETCHING NEWS ==========")

for category, url in RSS_FEEDS.items():

    try:
        print(f"Fetching: {category}")

        feed = feedparser.parse(url)

        entries = feed.entries[:10]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            process_func = partial(
                process_single_entry,
                category=category
            )

            futures = [
                executor.submit(process_func, entry)
                for entry in entries
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()

                    if result:
                        all_news.append(result)

                except Exception as e:
                    print("❌ FUTURE ERROR:", e)

        print(f"✅ {category} DONE")

    except Exception as e:
        print(f"❌ RSS ERROR ({category}):", e)

# =========================================================
# DATAFRAME
# =========================================================

if not all_news:
    print("❌ NO NEWS FOUND")
    exit()

df = pd.DataFrame(all_news)

# REMOVE DUPLICATES

df["headline_clean"] = (
    df["headline"]
    .str.lower()
    .str.strip()
)

df.drop_duplicates(
    subset=["headline_clean"],
    inplace=True
)

df.drop(columns=["headline_clean"], inplace=True)

print(f"TOTAL UNIQUE NEWS: {len(df)}")

# =========================================================
# PRIORITY NEWS
# =========================================================

local_news = df[
    df['category'].isin(['Kushinagar', 'Gorakhpur'])
]

other_news = df[
    ~df['category'].isin(['Kushinagar', 'Gorakhpur'])
]

df_to_process = pd.concat([
    local_news,
    other_news
]).head(MAX_ARTICLES_TO_PROCESS)

print(f"PROCESSING: {len(df_to_process)} ARTICLES")

# =========================================================
# RESULTS
# =========================================================

results = []

# =========================================================
# MAIN LOOP
# =========================================================

for index, row in df_to_process.iterrows():

    headline = row["headline"]
    category = row["category"]
    link = row["link"]
    article_text = row["article"]

    is_local = category in ['Kushinagar', 'Gorakhpur']

    print("=" * 80)
    print(f"CATEGORY: {category}")
    print(f"HEADLINE: {headline}")
    print("=" * 80)

    prompt = f"""
You are an elite viral content writer and social media journalist.

Write a HIGH-ENGAGEMENT Hindi viral social media article.

NEWS TITLE:
{headline}

CATEGORY:
{category}

FULL ARTICLE:
{article_text}

SOURCE LINK:
{link}

INSTRUCTIONS:
- Use conversational Hindi
- Highly engaging tone
- Emotional storytelling
- Facebook monetization optimized
- Mobile friendly
- Use emojis strategically
- Add curiosity and suspense
- Avoid misinformation
- Add hashtags
- Add CTA

OUTPUT FORMAT:

1. Viral Headline
2. Opening Hook
3. Main Article
4. Key Highlights
5. Public Reaction
6. Engagement Question
7. CTA
8. Hashtags

At the end output EXACTLY:

Emotion Score: X/10
Virality Score: X/10
Political Toxicity: X/10
"""

    try:

        ai_result = generate_ai_response(prompt)

        emotion_match = re.search(
            r"Emotion Score:\s*(\d+)/10",
            ai_result
        )

        virality_match = re.search(
            r"Virality Score:\s*(\d+)/10",
            ai_result
        )

        toxicity_match = re.search(
            r"Political Toxicity:\s*(\d+)/10",
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

        if is_local:
            final_score += LOCAL_NEWS_BONUS

        final_score = round(final_score, 2)

        print(f"FINAL SCORE: {final_score}")

        if final_score >= 5:

            results.append({
                "category": category,
                "headline": headline,
                "link": link,
                "emotion_score": emotion_score,
                "virality_score": virality_score,
                "toxicity_score": toxicity_score,
                "final_score": final_score,
                "ai_analysis": ai_result,
                "is_local": is_local
            })

            cleaned_link = clean_url(link)
            short_link = shorten_url(cleaned_link)

            telegram_message = f"""
🔥 VIRAL NEWS

📰 {headline}

📂 Category: {category}

📈 Viral Score: {final_score}
❤️ Emotion: {emotion_score}/10
🚀 Virality: {virality_score}/10
⚠️ Toxicity: {toxicity_score}/10

🔗 Link:
{short_link}

====================

{ai_result}

====================

📸 Instagram:
https://www.instagram.com/karan_patel_kushinagar

▶️ YouTube:
https://youtube.com/@karan_patel_kushinagar

📘 Facebook:
https://www.facebook.com/share/1D627R2qFV/
"""

            send_telegram_message(telegram_message)

            print("✅ SENT TO TELEGRAM")

            time.sleep(TELEGRAM_SLEEP_TIME)

        time.sleep(AI_SLEEP_TIME)

    except Exception as e:
        print("❌ AI ANALYSIS ERROR:", e)

# =========================================================
# SAVE CSV
# =========================================================

results_df = pd.DataFrame(results)

if not results_df.empty:

    results_df = results_df.sort_values(
        by="final_score",
        ascending=False
    )

    csv_path = os.path.join(
        SCRIPT_DIR,
        "viral_reel_news.csv"
    )

    results_df.to_csv(csv_path, index=False)

    print(f"✅ CSV SAVED: {csv_path}")
    print(f"✅ TOTAL SENT: {len(results_df)}")

else:
    print("❌ NO HIGH VIRAL NEWS FOUND")

