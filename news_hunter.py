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

Required:
HF_TOKEN
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
"""
    )

print("\n========== ENV VARIABLES ==========")
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
# TELEGRAM MESSAGE FUNCTION
# =========================================================

def send_telegram_message(message):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message[:3500]
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

        api_url = f"https://tinyurl.com/api-create.php?url={long_url}"

        response = requests.get(
            api_url,
            timeout=20
        )

        if response.status_code == 200:

            return response.text

        else:

            return long_url

    except Exception as e:

        print("\nSHORT LINK ERROR:")
        print(e)

        return long_url

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

        for entry in feed.entries[:20]:

            all_news.append({

                "category": category,

                "headline": entry.title,

                "link": entry.link
            })

    except Exception as e:

        print(f"\nERROR FETCHING {category}")
        print(e)

# =========================================================
# CREATE DATAFRAME
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

print("\n========== AI ANALYSIS STARTED ==========\n")

# =========================================================
# ANALYZE NEWS
# =========================================================

for index, row in df.iterrows():

    headline = row["headline"]
    category = row["category"]
    link = row["link"]

    print("\n" + "=" * 80)
    print(f"HEADLINE: {headline}")
    print("=" * 80)

    prompt = f"""
You are an expert viral Facebook content writer,
Hindi emotional storytelling writer,
and social media engagement strategist.

Analyze this news headline.

HEADLINE:
{headline}

Give output STRICTLY in this format:

Emotion Score: number/10
Virality Score: number/10
Political Toxicity: number/10

HOOK:
(one short powerful emotional hook)

HINDI FACEBOOK ARTICLE:
(write a proper emotional Hindi Facebook article.
It should feel natural,
engaging,
human-written,
emotionally strong,
shareable,
and suitable for viral Facebook posting.)

HASHTAGS:
(only trending hashtags)
"""

    try:

        response = client.chat_completion(

            model="mistralai/Mistral-7B-Instruct-v0.2",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            max_tokens=600
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
        # FINAL SCORE
        # =====================================================

        final_score = (
            virality_score * 0.5 +
            emotion_score * 0.35 -
            toxicity_score * 0.15
        )

        final_score = round(final_score, 2)

        print("\n========== SCORES ==========")
        print("Emotion:", emotion_score)
        print("Virality:", virality_score)
        print("Toxicity:", toxicity_score)
        print("Final Score:", final_score)

        # =====================================================
        # SAVE NEWS
        # =====================================================

        if final_score >= 1:

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

            print("\n✅ SAVED")

        else:

            print("\n❌ REJECTED")

        time.sleep(2)

    except Exception as e:

        print("\nAI ERROR:")
        print(e)

# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

# =========================================================
# SORT + SEND
# =========================================================

if not results_df.empty:

    results_df = results_df.sort_values(
        by="final_score",
        ascending=False
    )

    # TOP 10 NEWS
    results_df = results_df.head(10)

    print("\nTOTAL TOP NEWS:", len(results_df))

    # SAVE CSV
    results_df.to_csv(
        "viral_facebook_news.csv",
        index=False
    )

    print("\n========== SENDING TOP NEWS ==========\n")

    # =====================================================
    # SEND ONE BY ONE
    # =====================================================

    for index, row in results_df.iterrows():

        try:

            # CREATE SHORT LINK
            short_link = shorten_url(
                row['link']
            )

            telegram_message = f"""
🔥 VIRAL NEWS #{index + 1}

📰 HEADLINE:
{row['headline']}

📈 FINAL SCORE:
{row['final_score']}

❤️ EMOTION:
{row['emotion_score']}/10

🚀 VIRALITY:
{row['virality_score']}/10

⚠️ TOXICITY:
{row['toxicity_score']}/10

🔗 LINK:
{short_link}

━━━━━━━━━━━━━━━━━━

{row['ai_analysis'][:1500]}
"""

            print(f"\nSENDING NEWS #{index + 1}")

            send_telegram_message(
                telegram_message
            )

            print(f"✅ NEWS #{index + 1} SENT")

            # avoid telegram flood limit
            time.sleep(5)

        except Exception as e:

            print(f"\nERROR SENDING NEWS #{index + 1}")
            print(e)

    print("\n✅ ALL TOP NEWS SENT")

else:

    print("\n❌ NO VIRAL NEWS FOUND")
