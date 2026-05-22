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
from difflib import SequenceMatcher

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

MAX_WORKERS = 5
AI_SLEEP_TIME = 1
TELEGRAM_SLEEP_TIME = 1
LOCAL_NEWS_BONUS = 2.5

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
# DUPLICATE DETECTION
# =========================================================

def is_similar(a, b, threshold=0.7):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

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
    # SMART FALLBACK FORMATTER (FIXED & CASE-INSENSITIVE)
    # =====================================================

    print("\n⚠️ USING SMART FALLBACK FORMATTER")

    try:
        # Re-extract values cleanly using case-insensitive flags (re.IGNORECASE)
        title_match = re.search(r"(?:news title|title):\s*(.+)", prompt, re.IGNORECASE)
        category_match = re.search(r"category:\s*(.+)", prompt, re.IGNORECASE)
        article_match = re.search(r"(?:full article|article):\s*(.+?)(?:source link|link):", prompt, re.DOTALL | re.IGNORECASE)
        link_match = re.search(r"(?:source link|link):\s*(.+)", prompt, re.IGNORECASE)

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

        # Clean article text formatting
        article_text = article_text.replace("\n", " ").strip()

        # Split sentences dynamically to build the summary bullet points
        sentences = re.split(r'(?<=[.!?]) +', article_text)
        important_lines = []

        for sentence in sentences:
            sentence = sentence.strip()
            # If the article text is short (like a headline), catch everything available
            if len(sentence) > 10:
                important_lines.append(sentence)
            if len(important_lines) >= 5:
                break

        # Fallback if no lines were processed cleanly
        if not important_lines:
            important_lines = [headline]

        # Create structured summary bullet points
        summary = "\n\n".join([
            f"• {line}"
            for line in important_lines
        ])

        # Generate monetization hashtags
        hashtags = (
            f"#{category.replace(' ', '')} "
            "#BreakingNews "
            "#IndiaNews "
            "#ViralNews"
        )

        # Final structurally complete fallback post block
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
        print("❌ SMART FALLBACK CRITICAL ERROR:", e)
        # Final emergency string backup so the process never pushes empty structures
        return f"""
🔥 BREAKING NEWS

📰 {headline if 'headline' in locals() else 'New Update'}

📂 Category: {category if 'category' in locals() else 'News'}

━━━━━━━━━━━━━━━

• {headline if 'headline' in locals() else 'Click source link below to read full breaking news coverage.'}

━━━━━━━━━━━━━━━

🤔 आपकी क्या राय है?

📢 ऐसी ही खबरों के लिए follow करें।

🔗 Source:
{link if 'link' in locals() else ''}

#BreakingNews #ViralNews
Emotion Score: 5/10
Virality Score: 5/10
Political Toxicity: 0/10
"""

# =========================================================
# RSS FEEDS
# =========================================================

RSS_FEEDS = {
    # LOCAL
    "Kushinagar": "https://news.google.com/rss/headlines/section/geo/Kushinagar?hl=en-IN&gl=IN&ceid=IN:en",
    "Gorakhpur": "https://news.google.com/rss/headlines/section/geo/Gorakhpur?hl=en-IN&gl=IN&ceid=IN:en",

    # INDIA
    "India": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",

    # HIGH PRIORITY
    "India Politics": "https://news.google.com/rss/search?q=india+politics&hl=en-IN&gl=IN&ceid=IN:en",
    "India Pakistan": "https://news.google.com/rss/search?q=india+pakistan&hl=en-IN&gl=IN&ceid=IN:en",
    "India China": "https://news.google.com/rss/search?q=india+china&hl=en-IN&gl=IN&ceid=IN:en",
    "Breaking": "https://news.google.com/rss/search?q=breaking+india&hl=en-IN&gl=IN&ceid=IN:en",
    "Crime": "https://news.google.com/rss/search?q=india+crime&hl=en-IN&gl=IN&ceid=IN:en",
    "Defense": "https://news.google.com/rss/search?q=india+army+defense&hl=en-IN&gl=IN&ceid=IN:en",
    "AI": "https://news.google.com/rss/search?q=artificial+intelligence+india&hl=en-IN&gl=IN&ceid=IN:en",
    "Business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "World": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",

    # REAL SOURCES
    "Reuters World": "https://feeds.reuters.com/Reuters/worldNews",
    "BBC World": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC India": "http://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",

    # REDDIT SIGNALS
    "Reddit India": "https://www.reddit.com/r/india/.rss",
    "Reddit WorldNews": "https://www.reddit.com/r/worldnews/.rss",
    "Reddit Geopolitics": "https://www.reddit.com/r/geopolitics/.rss"
}

# =========================================================
# YOUTUBE TREND SIGNALS
# =========================================================

YOUTUBE_FEEDS = {
    "AajTak": "https://www.youtube.com/feeds/videos.xml?channel_id=UCt4t-jeY85JegMlZ-E5UWtA",
    "ANI": "https://www.youtube.com/feeds/videos.xml?channel_id=UCV-JlfdG9A8ZkNTJbVnHGkg",
    "IndiaTV": "https://www.youtube.com/feeds/videos.xml?channel_id=UCttspZesZIDEwwpVIgoZtWQ",
    "TV9 Bharatvarsh": "https://www.youtube.com/feeds/videos.xml?channel_id=UCP0uG-mcMImgKnJz-VsQxKQ"
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

        entries = feed.entries[:20]

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
# DATAFRAME VALIDATION
# =========================================================

if not all_news:
    print("❌ NO NEWS FOUND")
    exit()

# =========================================================
# GENERATE DYNAMIC TREND SIGNALS (NEW COMPONENT)
# =========================================================
print("========== GATHERING TREND SIGNALS ==========")
trending_keywords = {}

# 1. Harvest hot tokens from recent YouTube uploads
for channel_name, yt_url in YOUTUBE_FEEDS.items():
    try:
        print(f"Analyzing YouTube Trends: {channel_name}")
        yt_feed = feedparser.parse(yt_url)
        for entry in yt_feed.entries[:10]:  
            words = re.findall(r'\w+', entry.title.lower())
            for word in words:
                if len(word) > 3:  # Drop common tiny filler words
                    trending_keywords[word] = trending_keywords.get(word, 0) + 1.5
    except Exception as e:
        print(f"⚠️ YouTube Trend Extraction Error ({channel_name}):", e)

# 2. Cross-Coverage Analysis (Track high repetition across all collected headlines)
headline_frequencies = {}
for news in all_news:
    words = re.findall(r'\w+', news["headline"].lower())
    for word in words:
        if len(word) > 3:
            headline_frequencies[word] = headline_frequencies.get(word, 0) + 1.0


# =========================================================
# REMOVE DUPLICATES
# =========================================================

filtered_news = []

for news in all_news:

    duplicate_found = False

    for existing in filtered_news:

        if is_similar(
            news["headline"],
            existing["headline"]
        ):
            duplicate_found = True
            break

    if not duplicate_found:
        filtered_news.append(news)

print(f"UNIQUE NEWS ITEMS TO PROFILE: {len(filtered_news)}")

# =========================================================
# RESULTS
# =========================================================

results = []

# =========================================================
# MAIN LOOP
# =========================================================

for row in filtered_news:

    headline = row["headline"]
    category = row["category"]
    link = row["link"]
    article_text = row["article"]

    is_local = category in [
        'Kushinagar',
        'Gorakhpur'
    ]

    print("=" * 80)
    print(f"CATEGORY: {category}")
    print(f"HEADLINE: {headline}")
    print("=" * 80)

    # =====================================================
    # SMART SCORING WITH INTEGRATED TREND FACTOR
    # =====================================================

# =====================================================
    # SMART SCORING WITH TARGETED AUDIENCE KEYWORDS
    # =====================================================

    headline_lower = headline.lower()
    headline_words = re.findall(r'\w+', headline_lower)

    india_score = 0
    geo_score = 0
    emotion_bonus = 0
    controversy_bonus = 0
    local_bonus = 0
    trending_score = 0 

    # Dynamic YouTube + Cross-Coverage Trends
    for word in headline_words:
        if word in trending_keywords:
            trending_score += trending_keywords[word]
        if word in headline_frequencies:
            trending_score += (headline_frequencies[word] * 0.20)

    trending_score = min(round(trending_score, 2), 5.0)

    # 1. Local Audience Keyword Matching
    LOCAL_WORDS = [
    # --- CORE DISTRICTS & MICRO-REGIONS (English & Hindi) ---
    "kushinagar", "gorakhpur", "hata", "padrauna", "maharajganj", "deoria", "basti", "ghaziapur", 
    "azamgarh", "lucknow", "varanasi", "ballia", "mau", "jaunpur", "chauri chaura", "gonda", 
    "bahraich", "ayodhya", "siddharthnagar", "sant kabir nagar", "sultanpur", "amethi", "mirzapur", 
    "ghazipur", "ballia", "chandauli", "sonbhadra", "bhojpur", "purvanchal", "uttar pradesh", "up news",
    "कुशीनगर", "गोरखपुर", "हाटा", "पडरौना", "महराजगंज", "देवरिया", "पूर्वांचल", "उत्तर प्रदेश", "यूपी", 
    "बस्ती", "गाजीपुर", "बनारस", "मऊ", "बलिया", "जौनपुर", "चौरी चौरा", "अयोध्या", "सिद्धार्थनगर",
    
    # --- KUSHINAGAR & GORAKHPUR SUB-DIVISIONS / BLOCKS / TOWNS ---
    "kasia", "tamkuhi", "tamkuhi raj", "khadda", "captainganj", "ramkola", "seorahi", "fazilnagar", 
    "sukrauli", "dudhai", "neboua", "nautanwa", "anandnagar", "pharenda", "campierganj", "pipraich", 
    "gola", "khajni", "bansgaon", "sahjanwa", "barhalganj", "kauriram", "belaon", "urwa", "chargaon",
    "कसया", "तमकुही", "खड्डा", "कप्तानगंज", "रामकोला", "सेवरही", "फाजिलनगर", "सुकरौली", "नौतनवा", 
    "कैम्पियरगंज", "पिपराइच", "गोला", "खजनी", "बांसगांव", "सहजनवा", "बड़हलगंज", "कौड़ीराम", "चरगाँव",
    
    # --- LOCAL VILLAGES, CHOWKS & HISTORICAL STATIONS ---
    "kasya", "bodh gaya", "buddha", "parnirvana", "thuthibari", "sonauli", "chhitauni", "salempur", 
    "barhaj", "rudrapur", "gaura", "bhatni", "lar", "patherdeva", "tarkulwa", "desahi", "baitalpur", 
    "gauri bazar", "bhaluani", "kaptanganj", "siswa", "nichlaul", "mithauna", "धानी", "सिसवा", "निचलौल",
    "मीठाउना", "ठूठीबारी", "सनौली", "छितौनी", "सलेमपुर", "बरहज", "रुद्रपुर", "भाटपार", "भटनी", "लार",
    
    # --- REGIONAL FIGURES, SURNAMES & LOCAL CONTEXTS ---
    "yogi", "adityanath", "yogi adityanath", "gorakhnath", "peeth", "mahant", "योगी", "आदित्यनाथ", 
    "गोरखनाथ", "पीठ", "महंत", "up police", "gkp", "stf", "upstf", "dial 112", "daroga", "thana", 
    "kotwali", "chowki", "sp office", "dm office", "vda", "gda", "nagarpalika", "pradhan", "bdo",
    "दारोगा", "थाना", "कोतवाली", "चौकी", "प्रधान", "बीडीओ", "सिपाही", "दबंग", "मुख्तार", "अंसारी",
    
    # --- GEOGRAPHICAL & HIGH-TRAFFIC LOCAL TERM VARIANTS ---
    "bhojpuri", "bhojpuriya", "purvanchali", "ghaghra", "gandak", "rapti", "rohin", "ami", "kuwano", 
    "sarayu", "boating accident", "floods", "purvanchal expressway", "gorakhpur link", "aiims gorakhpur", 
    "brd medical college", "mmmut", "ddu", "kushinagar international airport", "buddha relics",
    "भोजपुरी", "पूर्वांचली", "घाघरा", "गंडक", "राप्ती", "सरयू", "बाढ़", "एक्सप्रेसवे", "एम्स", "बीआरडी",
    
    # --- EXTENSIVE UP/PURVANCHAL ADMINISTRATIVE TOWNS & BLOCKS ---
    "menhdawal", "khalilabad", "hainsarbazar", "bangaon", "harraiya", "captainganj", "rudhauli", 
    "bhanpur", "itwa", "domariaganj", "shohratgarh", "naugarh", "bansi", "alapur", "jalalpur", 
    "tanda", "akbarpur", "bhiti", "katehari", "maharajganj", "sagri", "gopalpur", "azamgarh city", 
    "nizamabad", "phoolpur", "didarganj", "lalganj", "mehnagar", "mubarakpur", "ghosi", "madhuban", 
    "mohammadabad", "kopaganj", "rasra", "bansdih", "sikanderpur", "phephna", "bairia", "zahoorabad", 
    "saidpur", "zamania", "muhammadabad", "jakhanian", "kerakat", "machhlishahr", "mariahu", "shahganj", 
    "mungra badshahpur", "malhani", "pindra", "shivpur", "cantt", "ajagara", "rohania", "sevapuri",
    "खलीलाबाद", "हरैया", "रुधौली", "इटवा", "डुमरियागंज", "शोहरतगढ़", "नौगढ़", "बांसी", "जलालपुर", 
    "टांडा", "अकबरपुर", "सगरी", "निज़ामबाद", "फूलपुर", "दीदारगंज", "लालगंज", "मऊनाथ", "घोषी", "मधुबन",
    "रसड़ा", "बांसडीह", "सिकंदरपुर", "फेफना", "बैरिया", "जमानिया", "जखनिया", "केराकत", "मछलीशहर", "मड़ियाहू",
    
    # --- HYPER-LOCAL VILLAGE / SMALL TOWN CONNECTIONS ---
    "asuran", "golghar", "buxipur", "mohaddipur", "taramandal", "medical road", "singharia", 
    "bargadwa", "shahpur", "tiwaripur", "rajendra nagar", "shastri nagar", "civil lines", "naushad", 
    "transport nagar", "gita press", "surajkund", "gorakhnath temple", "ramgarhtal", "leela",
    "असुरन", "गोलघर", "बक्शीपुर", "मोहद्दीपुर", "तारामंडल", "बरगदवा", "शाहपुर", "तिवारीपुर", "नौसड़", 
    "गीता प्रेस", "सूरजकुंड", "रामगढ़ताल", "गोरखनाथ मंदिर", "कौवाबाग", "पादरी बाजार", "झरना टोला",
    
    # --- ADDITIONAL LOCAL IDENTIFIERS TO GUARANTEE 300+ CAPTURE ---
    "kasya raj", "matha kaur", "ramabhar", "stupa", "wat thai", "linh son", "burmese temple", 
    "tibetan monastery", "shahi masjid", "kurmaghat", "narayani river", "choti gandak", "bansi river", 
    "ghaghra bridge", "dohrighat", "barhaj ghat", "ayodhya ram mandir route", "purvanchal handloom", 
    "terracotta", "bhathat", "jangal kauria", "pali", "sahjanwa industrial", "gida", "unwal", 
    "piprauli", "khorabar", "bramhpur", "gayanpur", "gopalpur chowk", "paniyara", "partawal", 
    "shyamdeurawa", "bhagwanpur", "bridgemanganj", "kolhui", "purandarpur", "lakhima", "sonauli border", 
    "belahiya", "valmiki nagar", "bagaha", "narkatiaganj", "chanpatia", "bettiah", "motihari"
    ]
    for keyword in LOCAL_WORDS:
        if keyword in headline_lower:
            local_bonus += 2.0  # Heavy boost for their home region

    # 2. National & Political High-Priority Matching
    HIGH_PRIORITY_KEYWORDS = [
    # --- CENTRAL GOVERNMENT & BJP APPARATUS ---
    "modi", "narendra modi", "bjp", "amit shah", "jp nadda", "rajnath singh", "nirmala sitharaman", 
    "s jaishankar", "nitin gadkari", "yogi", "pmo", "pmo india", "nda", "rss", "mohan bhagwat", 
    "lok sabha", "rajya sabha", "parliament", "sansad", "cabinet decision", "central govt",
    "मोदी", "नरेन्द्र मोदी", "भाजपा", "अमित शाह", "नड्डा", "राजनाथ", "जयशंकर", "गडकरी", "पीएमओ", 
    "संसद", "लोकसभा", "राज्यसभा", "मंत्रिमंडल", "कैबिनेट", "केंद्र सरकार", "आरएसएस", "मोहन भागवत",
    
    # --- OPPOSITION ALLIANCE & STATE LEADERSHIP ---
    "rahul gandhi", "priyanka gandhi", "sonia gandhi", "congress", "inc", "kharge", "akhilesh", 
    "yadav", "dimple yadav", "kejriwal", "aap", "mamata banerjee", "tmc", "stalin", "nitish kumar", 
    "jdu", "rjd", "tejashwi", "lalu yadav", "mayawati", "bsp", "india alliance", "mva", "sharad pawar",
    "राहुल गांधी", "प्रियंका", "कांग्रेस", "खड़गे", "अखिलेश यादव", "डिंपल", "केजरीवाल", "आप", "ममता", 
    "नीतीश कुमार", "तेजस्वी", "लालू यादव", "मायावती", "बसपा", "सपा", "विपक्ष", "इंडिया गठबंधन",
    
    # --- ELECTIONS, SEATS & CAMPAIGNS ---
    "election", "chunav", "voting", "matdan", "evm", "vvpat", "booth capturing", "exit poll", 
    "opinion poll", "by-poll", "voter id", "nomination", "rally", "jan sabha", "code of conduct",
    "चुनाव", "मतदान", "वोटिंग", "ईवीएम", "एग्जिट पोल", "नामांकन", "रैली", "जनसभा", "आचार संहिता",
    
    # --- LAW, JUDICIARY & INVESTIGATIVE AGENCIES ---
    "police", "arrest", "court", "supreme court", "high court", "cj", "ed", "cbi", "raid", "encounter", 
    "jail", "custody", "bailable", "non-bailable", "fir", "charge sheet", "summons", "interrogation", 
    "stf", "nia", "income tax", "it raid", "judgement", "verdict", "anticipatory bail", "warrant",
    "पुलिस", "गिरफ्तार", "कोर्ट", "सुप्रीम कोर्ट", "हाई कोर्ट", "जज", "ईडी", "सीबीआई", "छापा", "रेड", 
    "एनकाउंटर", "जेल", "हिरासत", "जमानत", "एफआईआर", "चार्जशीट", "समन", "पूछताछ", "एनआईए", "वारंट",
    
    # --- MILITARY, DEFENSE & WARFARE ---
    "pakistan", "china", "army", "attack", "war", "israel", "gaza", "iran", "border", "loc", "lac", 
    "iaf", "navy", "surgical strike", "drone", "missile", "agniveer", "agnipath", "iaf fighter", 
    "sukhoi", "rafale", "drdo", "indian army", "bsh", "crpf", "bsf", "itbp", "nsg", "terrorist", 
    "encounter", "infiltration", "commanders", "nuclear", "submarines", "aircraft carrier",
    "पाकिस्तान", "चीन", "सेना", "हमला", "युद्ध", "इजरायल", "गाजा", "ईरान", "बॉर्डर", "एलओसी", "राफेल", 
    "अग्निवीर", "सर्जिकल स्ट्राइक", "ड्रोन", "मिसाइल", "आतंकवादी", "घुसपैठ", "परमाणु", "फौजी", "जवान",
    
    # --- GLOBAL POWER PLAYERS & CHANNELS ---
    "trump", "donald trump", "putin", "russia", "biden", "america", "us news", "white house", 
    "kremlin", "pentagon", "united nations", "unsc", "nato", "zelensky", "ukraine war", "xi jinping", 
    "beijing", "washington", "london", "downing street", "bbc", "reuters", "al jazeera",
    "ट्रम्प", "पुतिन", "रूस", "बिडेन", "अमेरिका", "व्हाइट हाउस", "नाटो", "यूक्रेन", "शी जिनपिंग",
    
    # --- SCHEMES, ECONOMIC REFORMS & PUBLIC INFRASTRUCTURE ---
    "gst", "income tax slab", "budget", "rbi", "repo rate", "inflation", "gdp", "share market", 
    "nifty", "sensex", "sebi", "ipo", "crypto", "bitcoin", "vande bharat", "bullet train", "nhai", 
    "highway", "toll plaza", "ayushman bharat", "pm kisan", "ration card", "freebie", "subsidy",
    "बजट", "महंगाई", "शेयर बाजार", "सेंसेक्स", "निफ्टी", "क्रिप्टो", "वंदे भारत", "हाइवे", "टोल प्लाजा",
    
    # --- NATIONAL COMPETITIVE EXAMS, LEAKS & SYSTEMIC SCAMS ---
    "neet", "paper leak", "scam", "railway", "rrb", "ssc", "exam cancel", "leak", "fraud", "bribe", 
    "cgl", "upsc", "paper लीक", "घोटाला", "nta", "answer key", "cutoff", "result out", "cheating", 
    "dummy candidate", "solvers gang", "marks rigging", "re-exam", "grace marks", "uproar",
    "नीट", "पेपर लीक", "घोटाला", "रेलवे", "परीक्षा रद्द", "धांधली", "रिजल्ट", "धोखाधड़ी", "रिश्वत"
    ]
    for keyword in HIGH_PRIORITY_KEYWORDS:
        if keyword in headline_lower:
            india_score += 1.5

    # 3. Emotional & Sensational Triggers
    EMOTIONAL_WORDS = [
    # --- HIGH-VELOCITY ENGLISH SENSATIONALISM ---
    "shock", "fear", "warning", "danger", "viral", "anger", "violence", "protest", "ebola", "virus", 
    "outbreak", "alert", "breaking", "urgent", "hazard", "panic", "expose", "mystery", "miracle", 
    "tragedy", "disaster", "catastrophe", "scandal", "shameful", "horrific", "brutal", "deadly", 
    "conspiracy", "covert", "hidden camera", "caught on camera", "live stream", "unbelievable", 
    "bizarre", "myth", "haunted", "incident", "accident", "clash", "riot", "tension", "stampede",
    
    # --- INTENSE HINDI CLICK-THROUGH HOOKS (High Facebook Click-Through Rate) ---
    "सावधान", "बड़ी खबर", "खुलासा", "हादसा", "मौत", "हड़कंप", "सनसनी", "खौफ", "शर्मनाक", "चमत्कार", 
    "धोखा", "तबाही", "खतरा", "चेतावनी", "लाइव वीडियो", "सच्चाई", "मच गया बवाल", "रोंगटे खड़े", "दहशत",
    "सीसीटीवी", "शर्मसार", "भयंकर", "दर्दनाक", "बड़ी लापरवाही", "सनसनीखेज", "कांप उठी रूह", "देखें वीडियो",
    "खौफनाक", "दिल दहला देने वाला", "अंधविश्वास", "बवाल", "हंगामा", "आक्रोश", "गुस्सा", "मातम", "चीख-पुकार",
    "खूनी खेल", "वारदात", "भंडाफोड़", "काली करतूत", "सच्चा चेहरा", "गुप्त", "रहस्य", "अजीबोगरीब",
    
    # --- SOCIAL CRIME, OUTRAGE & INVESTIGATION ---
    "क्राइम", "मर्डर", "हादसा", "भूकंप", "आग लगी", "लूट", "गैंग", "बलात्कार", "गैंगस्टर", "माफिया", 
    "बदमाश", "गोलीबारी", "मर्डर केस", "फिरौती", "किडनैपिंग", "अपहरण", "डकैती", "चोरी", "फिरौती", 
    "शूटर", "सुपारी", "शव बरामद", "लाश", "हत्याकांड", "गोली मारकर हत्या", "चाकूबाजी", "एसिड अटैक",
    "रेप", "गैंगरेप", "उत्पीड़न", "घूसखोर", "रंगे हाथ", "ब्लैकमेल", "साइबर ठगी", "हनीट्रैप", "फर्जीवाड़ा",
    "ठग", "जालसाज", "माफिया राज", "बुलडोजर", "कुर्की", "गैंगवार", "हिस्ट्रीशीटर", "इनामी बदमाश",
    
    # --- ACCIDENTS, NATURAL HAZARDS & EMERGENCIES ---
    "blast", "explosion", "fire", "derailment", "train crash", "plane crash", "collision", "head-on", 
    "hit and run", "drowning", "lightning", "earthquake", "cloudburst", "landslide", "heatwave", 
    "cylinder blast", "short circuit", "building collapse", "bridge collapse", "poisonous", "gas leak",
    "धमाका", "विस्फोट", "आगजनी", "ट्रेन हादसा", "पटरी से उतरी", "टक्कर", "हिट एंड रन", "डूबने से मौत",
    "आकाशीय बिजली", "बादल फटा", "लैंडस्लाइड", "सिलेंडर ब्लास्ट", "शॉर्ट सर्किट", "जहरीली गैस", "दम घुटने",
    
    # --- REEL/VIRAL INTERNET ENGAGEMENT & OUTRAGE SENSORS ---
    "trending", "viral reel", "facebook trend", "instagram viral", "twitter fight", "boycott", 
    "trolled", "public bashing", "slapped", "abused", "viral audio", "leaked video", "crying video", 
    "suicide note", "last video", "stunt video", "speeding car", "fight video", "road rage", 
    "drunk driving", "high speed", "reckless", "suspension", "terminated", "action taken",
    "ट्रेंडिंग", "वायरल रील", "ट्विटर वॉर", "बॉयकाट", "ट्रोल", "पब्लिक ने पीटा", "थप्पड़", "गालीगलौज",
    "वायरल ऑडियो", "लीक वीडियो", "रोते हुए", "स्टंट वीडियो", "रोड रेज", "नशे में धुत", "सस्पेंड", "बर्खास्त"
    ]
    for word in EMOTIONAL_WORDS:
        if word in headline_lower:
            emotion_bonus += 1.2

    if category in ["India Politics", "India Pakistan", "India China", "Defense"]:
        geo_score += 3

    if category in ["Crime", "Breaking"]:
        controversy_bonus += 2

  # =====================================================
    # PRE-AI SCORE GATE (UPDATED WEIGHT DISTRIBUTION)
    # =====================================================

    raw_score = (
        india_score * 1.3 +
        geo_score * 1.0 +
        trending_score * 1.5 +  # Prioritizes active YouTube signals
        emotion_bonus * 1.2 +
        controversy_bonus * 1.0 +
        local_bonus * 1.5       # Strong bias toward your UP audience footprint
    )

    if is_local or category in ['Kushinagar', 'Gorakhpur']:
        raw_score += LOCAL_NEWS_BONUS

    raw_score = round(raw_score, 2)

    print(f"📊 PRE SCORE (Dynamic Gate Analysis): {raw_score}")

    # =====================================================
    # HARD GATE: PRE-AI LIMIT (Set to 5.5 to catch localized trends)
    # =====================================================
    if raw_score < 5.5:
        print(f"⏭️ SKIPPED LOW PRE-SCORE: {headline} ({raw_score})")
        continue

    print("✅ PRE-SCORE PASSED — RUNNING ENGINE")
    
    prompt = f"""
You are an elite viral  news content writer, Facebook monetization strategist, and social media journalist in Hindi Language .

Create a high-engagement, monetization-focused Hindi social media article optimized for Facebook, Instagram, Telegram, LinkedIn, X/Twitter, and YouTube community posts. The article should feel human-written, emotionally engaging, trustworthy, shareable, curiosity-driven, and tailored for social media.

Write in conversational, simple Hindi with a natural storytelling tone and mobile-friendly formatting. Use curiosity hooks, emotional storytelling, suspense, short punchy paragraphs, and relevant emojis. Avoid robotic language, repetition, fake info, clickbait, overly formal style, and long paragraphs.

Structure the output as follows:

Viral headline: Emotional, curiosity-based, clickable, and Facebook-optimized.
Opening hook (2–4 lines): Grabs attention immediately, builds suspense or emotion.
Main article (400–600 words): Explain what happened, why it matters, hidden angles, social reactions, impact on people, and possible future effects. Maintain suspense and emotional flow with short paragraphs.
Key highlights: 3–7 important bullet points.
Public reaction: Summarize social media and public responses.
Engagement section: Ask emotionally engaging questions like “आपकी क्या राय है?” or “क्या सरकार को कार्रवाई करनी चाहिए?”
Call to action: Encourage comments, shares, follows, saves, and joining Telegram/WhatsApp channels.
Hashtags: Use trending, niche, viral Hindi, and topic-specific tags.
Follow retention and monetization rules: Start with a strong hook, keep curiosity rising, optimize for mobile, use emotional language strategically, and maximize Facebook reach and engagement.
Note : never mention the section heading like - public reactioon, engagement section etc.
Ensure factual accuracy without exaggeration or misinformation, clearly separating facts from opinions.
IMPORTANT:
- Generate COMPLETE output
- Never stop midway
- Never cut article incomplete
- Finish all sections completely

Take your input data as :
Title: {headline}
Category: {category}
Full Article: {article_text}
Source Link: {link}
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

       # =====================================================
        # NEW BALANCED FINAL SCORE MATRIX (NO COMPRESSION)
        # =====================================================
        # AI directly points dega: max 4.0 points (2.0 + 2.0)
        ai_contribution = (virality_score * 0.20) + (emotion_score * 0.20)
        
        # Aapke core factors ka seedha real weight
        final_score = (
            ai_contribution +
            trending_score +          # Seedha trend score judega (Max 5.0)
            (india_score * 0.5) +     # Direct contribution
            (geo_score * 0.5) +
            (emotion_bonus * 0.3) +
            (controversy_bonus * 0.3) -
            (toxicity_score * 0.2)
        )

        if is_local or category in ['Kushinagar', 'Gorakhpur']:
            final_score += LOCAL_NEWS_BONUS

        final_score = round(final_score, 2)

        # =====================================================
        # FIXED FINAL GATE: NOW SET TO 6.5
        # =====================================================
        if final_score < 6.5:
            print(f"⏭️ SKIPPED LOW FINAL SCORE: {headline} ({final_score})")
            continue

        print(f"🚀 FINAL SCORE PASSED ({final_score}) — SENDING TELEGRAM")

        # =====================================================
        # CLEAN URL
        # =====================================================

        cleaned_link = clean_url(link)
        short_link = shorten_url(cleaned_link)

        # =====================================================
        # TELEGRAM MESSAGE
        # =====================================================

        telegram_message = f"""
🔥 VIRAL NEWS

📰 {headline}

📂 Category: {category}

📈 Viral Score: {final_score}
❤️ Emotion: {emotion_score}/10
🚀 Virality: {virality_score}/10
🔥 Trend Weight: {trending_score}/5.0
⚠️ Toxicity: {toxicity_score}/10

====================

{ai_result}

====================

🔗 News Source Link:
{short_link}

Follow us on:

📸 Instagram:
https://www.instagram.com/karan_patel_kushinagar

▶️ YouTube:
https://youtube.com/@karan_patel_kushinagar

📘 Facebook:
https://www.facebook.com/share/1D627R2qFV/
"""

        # =====================================================
        # SEND TELEGRAM
        # =====================================================

        send_telegram_message(telegram_message)

        print("✅ SENT TO TELEGRAM")

        # =====================================================
        # SAVE RESULTS
        # =====================================================

        results.append({
            "category": category,
            "headline": headline,
            "link": link,
            "short_link": short_link,
            "emotion_score": emotion_score,
            "virality_score": virality_score,
            "toxicity_score": toxicity_score,
            "trending_score": trending_score,
            "final_score": final_score,
            "ai_result": ai_result
        })

        time.sleep(TELEGRAM_SLEEP_TIME)
        time.sleep(AI_SLEEP_TIME)

    except Exception as e:
        print("❌ AI ANALYSIS ERROR:", e)
        continue

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
