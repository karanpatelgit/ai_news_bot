import feedparser
import pandas as pd
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import time
import re
import requests
from PIL import Image
from io import BytesIO

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
        "text": message
    }

    try:

        response = requests.post(
            url,
            data=payload,
            timeout=30
        )

        print("\n========== TELEGRAM MESSAGE ==========")
        print(response.text)

    except Exception as e:

        print("\nTELEGRAM MESSAGE ERROR:")
        print(e)

# =========================================================
# TELEGRAM PHOTO FUNCTION
# =========================================================

def send_telegram_photo(photo_path, caption):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    try:

        with open(photo_path, "rb") as photo:

            files = {
                "photo": photo
            }

            data = {
                "chat_id": CHAT_ID,
                "caption": caption[:1000]
            }

            response = requests.post(
                url,
                files=files,
                data=data
            )

        print("\n========== TELEGRAM PHOTO ==========")
        print(response.text)

    except Exception as e:

        print("\nTELEGRAM PHOTO ERROR:")
        print(e)

# =========================================================
# AI IMAGE GENERATION
# =========================================================

def generate_news_image(headline, index):

    try:

        image_prompt = f"""
Create a cinematic breaking news thumbnail.

Headline:
{headline}

Style:
- ultra realistic
- emotional
- dramatic lighting
- social media viral thumbnail
- youtube shorts style
- instagram reel style
- highly engaging
- modern digital art
- 4k quality
"""

        image = client.text_to_image(
            image_prompt,
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )

        image_path = f"news_image_{index}.png"

        image.save(image_path)

        print(f"\n✅ IMAGE SAVED: {image_path}")

        return image_path

    except Exception as e:

        print("\nIMAGE GENERATION ERROR:")
        print(e)

        return None

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

        # FETCH MORE NEWS
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
You are a world-class viral content strategist,
Instagram reel expert,
YouTube Shorts expert,
and emotional storytelling analyst.

Analyze this news headline carefully.

HEADLINE:
{headline}

Your task:
- detect emotional power
- detect virality
- detect curiosity
- detect audience engagement
- detect controversy
- detect shareability

Give scores STRICTLY in this format:

Emotion Score: number/10
Virality Score: number/10
Political Toxicity: number/10

Then continue with:

HOOK:
(short ultra viral opening)

SHORT REEL SCRIPT:
(4-6 emotionally engaging lines)

ENDING CTA:
(one engagement line)

HINDI REEL SCRIPT:
(lengthy emotional Hindi spoken script)

CAPTION:
(short emotional caption)

HASHTAGS:
(viral hashtags only)
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
        # FILTER
        # =====================================================

        if (
            final_score >= 3 or
            virality_score >= 4 or
            emotion_score >= 5
        ):

            print("\n✅ SAVED")

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
# SAVE + SEND
# =========================================================

if not results_df.empty:

    results_df = results_df.sort_values(
        by="final_score",
        ascending=False
    )

    # TOP 10 NEWS
    results_df = results_df.head(10)

    results_df.to_csv(
        "viral_reel_news.csv",
        index=False
    )

    print("\n========== TOP 10 VIRAL NEWS ==========\n")

    # =====================================================
    # SEND TOP 10 NEWS
    # =====================================================

    for index, row in results_df.iterrows():

        print(f"\n🔥 SENDING NEWS #{index + 1}")

        telegram_message = f"""
🔥 VIRAL NEWS #{index + 1}

📰 HEADLINE:
{row['headline']}

📂 CATEGORY:
{row['category']}

📈 FINAL SCORE:
{row['final_score']}

❤️ EMOTION:
{row['emotion_score']}/10

🚀 VIRALITY:
{row['virality_score']}/10

⚠️ TOXICITY:
{row['toxicity_score']}/10

🔗 LINK:
{row['link']}

==================================================

{row['ai_analysis'][:3000]}
"""

        # =================================================
        # GENERATE AI IMAGE
        # =================================================

        image_path = generate_news_image(
            row['headline'],
            index
        )

        # =================================================
        # SEND IMAGE + MESSAGE
        # =================================================

        if image_path:

            send_telegram_photo(
                image_path,
                telegram_message
            )

        else:

            send_telegram_message(
                telegram_message
            )

        print(f"✅ NEWS #{index + 1} SENT")

        # Avoid Telegram flood limit
        time.sleep(5)

    print("\n✅ ALL TOP 10 NEWS SENT")

else:

    print("\n❌ NO VIRAL NEWS FOUND")

    print("\n❌ NO HIGH-VIRAL NEWS FOUND TODAY")
