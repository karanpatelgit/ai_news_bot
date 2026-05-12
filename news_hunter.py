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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

missing = []

if not HF_TOKEN:
    missing.append("HF_TOKEN")

if not BOT_TOKEN:
    missing.append("TELEGRAM_BOT_TOKEN")

if not CHAT_ID:
    missing.append("TELEGRAM_CHAT_ID")

if missing:
    raise RuntimeError(
        f"""
Missing environment variables:

{', '.join(missing)}

Create .env file in same folder as script.

Example:

HF_TOKEN=your_huggingface_token
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
"""
    )

print("\n========== ENV VARIABLES LOADED ==========")
print("HF_TOKEN:", bool(HF_TOKEN))
print("BOT_TOKEN:", bool(BOT_TOKEN))
print("CHAT_ID:", bool(CHAT_ID))

# =========================================================
# HUGGING FACE CLIENT
# =========================================================

client = InferenceClient(
    token=HF_TOKEN
)

# =========================================================
# TELEGRAM FUNCTION
# =========================================================

def send_telegram_message(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message
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

        print("\nTELEGRAM ERROR:")
        print(e)

# =========================================================
# SHORT LINK FUNCTION
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

        print("\nSHORT LINK ERROR:")
        print(e)

        return long_url
# =========================================================
# RSS NEWS FEEDS
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
    "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-IN&gl=IN&ceid=IN:en"
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

        for entry in feed.entries[:5]:

            all_news.append({
                "category": category,
                "headline": entry.title,
                "link": entry.link
            })

    except Exception as e:

        print(f"Error fetching {category}")
        print(e)

# =========================================================
# DATAFRAME
# =========================================================

df = pd.DataFrame(all_news)

df.drop_duplicates(
    subset=["headline"],
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
You are a world-class viral content strategist, facebook article writer,
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

Avoid boring analysis.(don't mention your explanations )

Give scores STRICTLY in this format:

Emotion Score: number/10
Virality Score: number/10
Political Toxicity: number/10

Then continue with:

HOOK:
(one ultra-viral opening line)

SHORT REEL SCRIPT:
(4-6 powerful lines)

ENDING CTA:
(one audience engagement line)

HINDI FACEBOOK ARTICLE:
(simple emotional spoken Hindi but lenghty  script for maximum emotional impact and starts with creative hook )

HASHTAGS:
(only trending hashtags)
"""

    try:

        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=700
        )

        ai_result = response.choices[0].message.content

        print("\n========== AI RESPONSE ==========\n")
        print(ai_result)

        # =====================================================
        # EXTRACT SCORES
        # =====================================================

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

        # =====================================================
        # FINAL VIRAL SCORE
        # =====================================================

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

        # =====================================================
        # FILTER BEST NEWS ONLY
        # =====================================================

        if (
            final_score >= 3
        ):

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

            print("\n❌ REJECTED (LOW VIRAL POTENTIAL)")

        time.sleep(2)

    except Exception as e:

        print("\nAI ERROR:")
        print(e)

# =========================================================
# CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

# =========================================================
# SAVE RESULTS
# =========================================================

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

    print("\n========== TOP VIRAL NEWS ==========\n")

    for i, row in results_df.iterrows():

        print(f"""
🔥 HEADLINE:
{row['headline']}

📈 VIRAL SCORE:
{row['final_score']}

❤️ EMOTION:
{row['emotion_score']}/10

🚀 VIRALITY:
{row['virality_score']}/10
""")
short_link = shorten_url(row['link'])

    # =====================================================
    # SEND BEST NEWS TO TELEGRAM
    # =====================================================
print("\n========== SENDING ALL TOP VIRAL NEWS ==========\n")

for i, row in results_df.iterrows():

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

    send_telegram_message(telegram_message)

    print(f"✅ Sent news #{i+1}")

    time.sleep(3)  # avoid Telegram flood limit

    print("\n✅ CSV SAVED:")
    print(csv_path)

else:

    print("\n❌ NO HIGH-VIRAL NEWS FOUND TODAY")
