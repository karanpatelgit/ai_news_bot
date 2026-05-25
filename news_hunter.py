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
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from difflib import SequenceMatcher
import random

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "").strip()
SAMBANOVA_API_KEY  = os.environ.get("SAMBANOVA_API_KEY", "").strip()
HF_TOKEN           = os.environ.get("HF_TOKEN", "").strip()

BOT_TOKEN          = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID            = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_CHAT_IDS  = os.environ.get("TELEGRAM_CHAT_IDS", "").strip()

CANVA_EMAIL        = os.environ.get("CANVA_EMAIL", "").strip()
CANVA_PASSWORD     = os.environ.get("CANVA_PASSWORD", "").strip()

print("=" * 50)
print("🔑 KEY STATUS:")
print(f"   GROQ       : {'✅ SET' if GROQ_API_KEY else '❌ MISSING'}")
print(f"   SAMBANOVA  : {'✅ SET' if SAMBANOVA_API_KEY else '❌ MISSING'}")
print(f"   HF_TOKEN   : {'✅ SET' if HF_TOKEN else '❌ MISSING'}")
print(f"   OPENROUTER : {'✅ SET — ' + OPENROUTER_API_KEY[:8] + '...' if OPENROUTER_API_KEY else '❌ MISSING'}")
print(f"   BOT_TOKEN  : {'✅ SET' if BOT_TOKEN else '❌ MISSING'}")
print(f"   CHAT_ID    : {'✅ SET' if CHAT_ID else '❌ MISSING'}")
print(f"   TELEG_CHAT : {'✅ SET' if TELEGRAM_CHAT_IDS else '❌ MISSING'}")
print(f"   CANVA      : {'✅ SET' if CANVA_EMAIL and CANVA_PASSWORD else '❌ MISSING'}")
print("=" * 50)

# =========================================================
# CONFIG
# =========================================================

MAX_WORKERS           = 6
AI_SLEEP_TIME         = 1
TELEGRAM_SLEEP_TIME   = 1
LOCAL_NEWS_BONUS      = 2.5
SCRAPER_TIME_WINDOW_H = 10
SCRAPER_MAX_PER_SITE  = 8
RSS_MAX_PER_FEED      = 20

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

def random_headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "hi-IN,hi;q=0.9,en-IN;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/",
        "DNT": "1",
    }

# =========================================================
# AI CLIENTS
# =========================================================

groq_client       = None
samba_client      = None
hf_client         = None
openrouter_client = None

if GROQ_API_KEY:
    try:
        groq_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        print("✅ GROQ READY")
    except Exception as e:
        print("❌ GROQ ERROR:", e)

if SAMBANOVA_API_KEY:
    try:
        samba_client = OpenAI(api_key=SAMBANOVA_API_KEY, base_url="https://api.sambanova.ai/v1")
        print("✅ SAMBANOVA READY")
    except Exception as e:
        print("❌ SAMBANOVA ERROR:", e)

if HF_TOKEN:
    try:
        hf_client = InferenceClient(token=HF_TOKEN)
        print("✅ HF READY")
    except Exception as e:
        print("❌ HF ERROR:", e)

if OPENROUTER_API_KEY:
    try:
        openrouter_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        print("✅ OPENROUTER READY")
    except Exception as e:
        print("❌ OPENROUTER INIT ERROR:", e)

# =========================================================
# MODELS
# =========================================================

GROQ_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
SAMBANOVA_MODELS = ["Meta-Llama-3.3-70B-Instruct"]
HF_MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct"]

OPENROUTER_MODELS = [
    "openrouter/auto",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.5-flash:free",
    "google/gemini-2.5-pro:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "deepseek/deepseek-chat:free"
]

# =========================================================
# DUPLICATE DETECTION
# =========================================================

def is_similar(a, b, threshold=0.7):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

# =========================================================
# TELEGRAM
# =========================================================

def _parse_chat_ids(raw: str):
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]

def send_telegram_message(message):
    url        = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    chunk_size = 4000
    chunks     = [message[i:i+chunk_size] for i in range(0, len(message), chunk_size)]

    chat_ids = _parse_chat_ids(TELEGRAM_CHAT_IDS)
    if not chat_ids and CHAT_ID:
        chat_ids = [CHAT_ID]

    if not chat_ids:
        print("❌ TELEGRAM ERROR: No CHAT_ID(s) configured")
        return

    for target_chat_id in chat_ids:
        for chunk in chunks:
            payload = {"chat_id": target_chat_id, "text": chunk}
            try:
                requests.post(url, data=payload, timeout=30)
                print(f"✅ TELEGRAM SENT -> chat_id={target_chat_id}")
            except Exception as e:
                print(f"❌ TELEGRAM ERROR -> chat_id={target_chat_id}:", e)

# =========================================================
# SEND TELEGRAM PHOTO
# =========================================================

def send_telegram_photo(photo_path, caption):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    chat_ids = _parse_chat_ids(TELEGRAM_CHAT_IDS)

    if not chat_ids and CHAT_ID:
        chat_ids = [CHAT_ID]

    if not chat_ids:
        print("❌ NO CHAT IDs FOUND")
        return

    for chat_id in chat_ids:

        try:

            with open(photo_path, "rb") as photo:

                response = requests.post(
                    url,
                    data={
                        "chat_id": chat_id,
                        "caption": caption[:1024]
                    },
                    files={
                        "photo": photo
                    },
                    timeout=60
                )

            print(f"✅ PHOTO SENT -> {chat_id}")

        except Exception as e:

            print(f"❌ TELEGRAM PHOTO ERROR ({chat_id}):", e)

# =========================================================
# SEND ARTICLE WITH INLINE ✅ GENERATE IMAGE BUTTON  ← NEW
# =========================================================

def send_article_with_button(chat_id, article_index, message_text):

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    keyboard = {
        "inline_keyboard": [[
            {
                "text": "🖼️ Generate Canva Image",
                "callback_data": f"canva_{article_index}"
            }
        ]]
    }

    # Split long messages — Telegram max is 4096 chars
    chunk_size = 3800
    chunks     = [message_text[i:i+chunk_size] for i in range(0, len(message_text), chunk_size)]

    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
        }
        # Only attach button to the LAST chunk so it appears at the bottom
        if i == len(chunks) - 1:
            payload["reply_markup"] = json.dumps(keyboard)

        try:
            requests.post(url, data=payload, timeout=30)
            print(f"✅ ARTICLE #{article_index} CHUNK {i+1} SENT WITH BUTTON -> {chat_id}")
        except Exception as e:
            print(f"❌ SEND WITH BUTTON ERROR: {e}")

# =========================================================
# ANSWER CALLBACK QUERY (removes Telegram spinner)  ← NEW
# =========================================================

def answer_callback(callback_query_id, text=""):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/answerCallbackQuery",
            data={"callback_query_id": callback_query_id, "text": text},
            timeout=10
        )
    except Exception as e:
        print("❌ ANSWER CALLBACK ERROR:", e)

# =========================================================
# IN-MEMORY ARTICLE STORE  ← NEW
# Holds finished articles so button clicks can retrieve them
# { article_index: { headline, category, link, ai_result, ... } }
# =========================================================

article_store = {}

# =========================================================
# CANVA IMAGE GENERATOR (called on button click only)  ← NEW
# =========================================================

def generate_and_send_canva_image(article_index):

    from canva_ai import generate_news_poster

    article = article_store.get(article_index)

    if not article:
        send_telegram_message(f"❌ Article #{article_index} not found in memory.")
        return

    headline  = article["headline"]
    category  = article["category"]
    link      = article["link"]
    ai_result = article["ai_result"]
    score     = article["final_score"]

    # Tell user we're starting
    send_telegram_message(
        f"⏳ Generating Canva image for article #{article_index}...\n\n"
        f"📰 {headline}\n\n"
        f"This may take ~2 minutes. Please wait! 🙏"
    )

    visual_prompt = f"""
Breaking Indian news poster.

Headline:
{headline}

Category:
{category}

News Summary:
{ai_result[:500]}

Style:
Ultra realistic
Hindi breaking news
Indian newsroom
Cinematic lighting
Viral Instagram post
Red and black theme
1080x1350 portrait
"""

    try:

        print(f"🎨 GENERATING CANVA POSTER FOR ARTICLE #{article_index}...")

        image_path = generate_news_poster(visual_prompt)

        if image_path and os.path.exists(image_path):

            caption = (
                f"🖼️ Canva AI Poster — Article #{article_index}\n\n"
                f"📰 {headline}\n"
                f"📂 {category}\n"
                f"📈 Viral Score: {score}\n"
                f"🔗 {link}\n\n"
                f"Follow us on:\n"
                f"📸 Instagram: https://www.instagram.com/karan_patel_kushinagar\n"
                f"▶️ YouTube: https://youtube.com/@karan_patel_kushinagar\n"
                f"📘 Facebook: https://www.facebook.com/share/1D627R2qFV/"
            )

            send_telegram_photo(image_path, caption)

            print(f"✅ CANVA IMAGE SENT FOR ARTICLE #{article_index}")

        else:

            send_telegram_message(
                f"❌ Image generation failed for article #{article_index}. Please try again."
            )

    except Exception as e:

        print(f"❌ CANVA ERROR for article #{article_index}: {e}")

        send_telegram_message(
            f"❌ Canva error for article #{article_index}: {str(e)[:200]}"
        )

# =========================================================
# TELEGRAM BUTTON CLICK LISTENER  ← NEW
# Polls Telegram for button taps and triggers Canva
# =========================================================

def listen_for_button_clicks(stop_event, timeout_minutes=30):

    print(f"\n========== LISTENING FOR BUTTON CLICKS ({timeout_minutes} min) ==========")
    print("Tap 🖼️ Generate Canva Image on any article in Telegram.")
    print("Listening will auto-stop after timeout.\n")

    last_update_id = None
    deadline       = time.time() + (timeout_minutes * 60)

    while not stop_event.is_set() and time.time() < deadline:

        try:

            url    = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            params = {
                "timeout": 10,
                "allowed_updates": ["callback_query"]
            }

            if last_update_id:
                params["offset"] = last_update_id + 1

            r       = requests.get(url, params=params, timeout=20)
            updates = r.json().get("result", [])

            for update in updates:

                last_update_id = update["update_id"]
                cb             = update.get("callback_query")

                if cb:

                    data  = cb.get("data", "")
                    cb_id = cb["id"]

                    if data.startswith("canva_"):

                        article_index = int(data.split("_")[1])

                        print(f"\n✅ Button clicked! Article #{article_index} selected for Canva.")

                        answer_callback(cb_id, "🖼️ Generating image... please wait!")

                        # Run Canva in background thread so listener keeps working
                        t = threading.Thread(
                            target=generate_and_send_canva_image,
                            args=(article_index,),
                            daemon=True
                        )
                        t.start()

        except Exception as e:

            print("LISTENER ERROR:", e)
            time.sleep(5)

        time.sleep(1)

    print("\n⏹️ Listener stopped.")

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
        api_url  = f"https://tinyurl.com/api-create.php?url={long_url}"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            return response.text.strip()
        return long_url
    except Exception as e:
        print("❌ SHORT URL ERROR:", e)
        return long_url

# =========================================================
# REAL URL EXTRACTION (Google redirect resolver)
# =========================================================

def extract_real_url(google_url):
    try:
        response  = requests.get(google_url, headers=random_headers(), timeout=6, allow_redirects=True)
        final_url = response.url
        if "news.google.com" not in final_url:
            return final_url
        soup      = BeautifulSoup(response.text, "lxml")
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
# ARTICLE FETCH (trafilatura deep extract)
# =========================================================

def fetch_full_article(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if text and len(text) > 300:
            return text[:6000]
        return None
    except Exception as e:
        print(f"❌ ARTICLE FETCH ERROR: {e}")
        return None

# =========================================================
# RSS ENTRY PROCESSOR
# =========================================================

def process_single_entry(entry, category, time_window_h=8):
    try:
        published = entry.get("published_parsed")
        if not published:
            return None
        published_datetime = datetime(*published[:6], tzinfo=timezone.utc)
        now       = datetime.now(timezone.utc)
        time_diff = now - published_datetime

        if timedelta(0) <= time_diff <= timedelta(hours=time_window_h):
            real_link    = extract_real_url(entry.link)
            article_text = fetch_full_article(real_link)
            if not article_text:
                article_text = entry.title
            return {
                "source":   "rss",
                "category": category,
                "headline": entry.title,
                "link":     real_link,
                "article":  article_text
            }
        return None
    except Exception as e:
        print("❌ ENTRY PROCESS ERROR:", e)
        return None

# =========================================================
# LOCAL SCRAPER
# =========================================================

SITE_SELECTORS = {
    "amarujala.com":      ["h3 a", "h2 a", ".list-content a", ".news-listing a"],
    "bhaskar.com":        ["h3 a", "h2 a", ".lead-news a", ".list-story a", ".card a"],
    "livehindustan.com":  ["h3 a", "h2 a", ".track-data a", ".story-list a"],
    "jagran.com":         ["h2 a", "h3 a", ".highlight a", ".section-news-block a"],
    "patrika.com":        ["h2 a", "h3 a", ".news-article a", ".article-card a"],
    "news18.com":         ["h3 a", "h2 a", ".story_list a", ".jsx-story a"],
    "etvbharat.com":      ["h3 a", "h2 a", ".post-title a", ".news-detail a"],
    "zeenews.india.com":  ["h2 a", "h3 a", ".swiper-slide a", ".news_box a"],
    "tv9hindi.com":       ["h2 a", "h3 a", ".media-heading a", ".article-list a"],
    "aajtak.in":          ["h2 a", "h3 a", ".story__title a", ".strylist a"],
    "abplive.com":        ["h2 a", "h3 a", ".article__title a", ".news-item a"],
    "indiatv.in":         ["h2 a", "h3 a", ".article-list a", ".newslist a"],
    "oneindia.com":       ["h3 a", "h2 a", ".article-list a", ".story-card a"],
    "webdunia.com":       ["h3 a", "h2 a", ".wd-news a", ".content-list a"],
    "gorakhpurlive.com":  ["h2 a", "h3 a", "article a", ".entry-title a"],
    "purvanchalnews.com": ["h2 a", "h3 a", "article a", ".post-title a"],
    "uptak.in":           ["h2 a", "h3 a", ".story a", ".card-title a"],
    "gaonconnection.com": ["h2 a", "h3 a", ".story-card a", ".post-title a"],
    "jansatta.com":       ["h2 a", "h3 a", ".articles a", ".story-list a"],
}

GENERIC_SELECTORS = ["h1 a", "h2 a", "h3 a", "h4 a", "article a", ".headline a", ".title a"]

def _get_selectors(url):
    for domain, selectors in SITE_SELECTORS.items():
        if domain in url:
            return selectors + GENERIC_SELECTORS
    return GENERIC_SELECTORS

def _is_valid_headline(text):
    text = text.strip()
    if len(text) < 15 or len(text) > 300:
        return False
    skip_patterns = [
        "होम", "menu", "home", "login", "register", "advertisement",
        "follow us", "subscribe", "newsletter", "copyright", "privacy",
        "terms", "contact", "about", "javascript", "click here",
        "read more", "और पढ़ें", "ताजा खबर", "सभी खबरें", "मुख्य पृष्ठ",
        "photo gallery", "video", "podcast",
    ]
    low = text.lower()
    for pat in skip_patterns:
        if pat in low:
            return False
    return True

def scrape_local_site(site_name, site_url, max_items=SCRAPER_MAX_PER_SITE):
    results = []
    try:
        html = None
        for attempt in range(2):
            try:
                resp = requests.get(
                    site_url,
                    headers=random_headers(),
                    timeout=12,
                    allow_redirects=True
                )
                if resp.status_code == 200:
                    html = resp.text
                    break
                else:
                    print(f"⚠️ [{site_name}] HTTP {resp.status_code} attempt {attempt+1}")
                    time.sleep(2)
            except Exception as fetch_err:
                print(f"⚠️ [{site_name}] Fetch attempt {attempt+1} error: {fetch_err}")
                time.sleep(2)

        if not html:
            print(f"❌ [{site_name}] Could not fetch page — skipping")
            return results

        soup       = BeautifulSoup(html, "lxml")
        selectors  = _get_selectors(site_url)
        seen_texts = set()
        links_found = []

        for selector in selectors:
            tags = soup.select(selector)
            for tag in tags:
                href = tag.get("href", "").strip()
                text = tag.get_text(separator=" ", strip=True)

                if not href or not _is_valid_headline(text):
                    continue
                if text in seen_texts:
                    continue

                if href.startswith("http"):
                    full_url = href
                elif href.startswith("//"):
                    full_url = "https:" + href
                elif href.startswith("/"):
                    from urllib.parse import urlparse
                    parsed   = urlparse(site_url)
                    full_url = f"{parsed.scheme}://{parsed.netloc}{href}"
                else:
                    continue

                seen_texts.add(text)
                links_found.append((text, full_url))
                if len(links_found) >= max_items:
                    break
            if len(links_found) >= max_items:
                break

        print(f"🔍 [{site_name}] Found {len(links_found)} candidate headlines")

        for headline, article_url in links_found:
            article_text = fetch_full_article(article_url)
            if not article_text or len(article_text.strip()) < 80:
                article_text = headline

            results.append({
                "source":   "scraper",
                "category": site_name,
                "headline": headline,
                "link":     article_url,
                "article":  article_text
            })

    except Exception as e:
        print(f"❌ [{site_name}] Scraper error: {e}")

    print(f"✅ [{site_name}] Scraped {len(results)} articles")
    return results

# =========================================================
# RSS FEEDS
# =========================================================

RSS_FEEDS = {
    "Kushinagar":
        "https://news.google.com/rss/headlines/section/geo/Kushinagar?hl=en-IN&gl=IN&ceid=IN:en",
    "Gorakhpur":
        "https://news.google.com/rss/headlines/section/geo/Gorakhpur?hl=en-IN&gl=IN&ceid=IN:en",
    "Maharajganj":
        "https://news.google.com/rss/headlines/section/geo/Maharajganj?hl=en-IN&gl=IN&ceid=IN:en",
    "Deoria":
        "https://news.google.com/rss/headlines/section/geo/Deoria?hl=en-IN&gl=IN&ceid=IN:en",
    "Basti":
        "https://news.google.com/rss/headlines/section/geo/Basti,Uttar Pradesh?hl=en-IN&gl=IN&ceid=IN:en",
    "Azamgarh":
        "https://news.google.com/rss/headlines/section/geo/Azamgarh?hl=en-IN&gl=IN&ceid=IN:en",
    "Varanasi":
        "https://news.google.com/rss/headlines/section/geo/Varanasi?hl=en-IN&gl=IN&ceid=IN:en",
    "Lucknow":
        "https://news.google.com/rss/headlines/section/geo/Lucknow?hl=en-IN&gl=IN&ceid=IN:en",
    "Ayodhya":
        "https://news.google.com/rss/headlines/section/geo/Ayodhya?hl=en-IN&gl=IN&ceid=IN:en",
    "Prayagraj":
        "https://news.google.com/rss/headlines/section/geo/Prayagraj?hl=en-IN&gl=IN&ceid=IN:en",
    "GNews_Kushinagar_HI":
        "https://news.google.com/rss/search?q=कुशीनगर&hl=hi-IN&gl=IN&ceid=IN:hi",
    "GNews_Gorakhpur_HI":
        "https://news.google.com/rss/search?q=गोरखपुर&hl=hi-IN&gl=IN&ceid=IN:hi",
    "GNews_UP_HI":
        "https://news.google.com/rss/search?q=उत्तर+प्रदेश&hl=hi-IN&gl=IN&ceid=IN:hi",
    "GNews_Purvanchal_HI":
        "https://news.google.com/rss/search?q=पूर्वांचल&hl=hi-IN&gl=IN&ceid=IN:hi",
    "GNews_YogiUP":
        "https://news.google.com/rss/search?q=yogi+uttar+pradesh&hl=en-IN&gl=IN&ceid=IN:en",
    "GNews_UPCrime":
        "https://news.google.com/rss/search?q=UP+crime+uttar+pradesh&hl=en-IN&gl=IN&ceid=IN:en",
    "GNews_UPPolice":
        "https://news.google.com/rss/search?q=UP+police+arrest+encounter&hl=en-IN&gl=IN&ceid=IN:en",
    "India":
        "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "India Politics":
        "https://news.google.com/rss/search?q=india+politics&hl=en-IN&gl=IN&ceid=IN:en",
    "India Pakistan":
        "https://news.google.com/rss/search?q=india+pakistan&hl=en-IN&gl=IN&ceid=IN:en",
    "India China":
        "https://news.google.com/rss/search?q=india+china&hl=en-IN&gl=IN&ceid=IN:en",
    "Breaking":
        "https://news.google.com/rss/search?q=breaking+india&hl=en-IN&gl=IN&ceid=IN:en",
    "Crime":
        "https://news.google.com/rss/search?q=india+crime&hl=en-IN&gl=IN&ceid=IN:en",
    "Defense":
        "https://news.google.com/rss/search?q=india+army+defense&hl=en-IN&gl=IN&ceid=IN:en",
    "AI":
        "https://news.google.com/rss/search?q=artificial+intelligence+india&hl=en-IN&gl=IN&ceid=IN:en",
    "Business":
        "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "World":
        "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
    "Entertainment":
        "https://news.google.com/rss/headlines/section/topic/ENTERTAINMENT?hl=en-IN&gl=IN&ceid=IN:en",
    "Sports":
        "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-IN&gl=IN&ceid=IN:en",
    "Health":
        "https://news.google.com/rss/headlines/section/topic/HEALTH?hl=en-IN&gl=IN&ceid=IN:en",
    "Technology":
        "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "Modi":
        "https://news.google.com/rss/search?q=narendra+modi&hl=en-IN&gl=IN&ceid=IN:en",
    "NEET_Exam":
        "https://news.google.com/rss/search?q=NEET+exam+india&hl=en-IN&gl=IN&ceid=IN:en",
    "India_Weather":
        "https://news.google.com/rss/search?q=india+weather+flood+rain&hl=en-IN&gl=IN&ceid=IN:en",
    "AmarUjala_National":   "https://www.amarujala.com/rss/india-news.xml",
    "AmarUjala_UP_RSS":     "https://www.amarujala.com/rss/uttar-pradesh.xml",
    "AmarUjala_Crime_RSS":  "https://www.amarujala.com/rss/crime.xml",
    "AmarUjala_Politics":   "https://www.amarujala.com/rss/politics.xml",
    "AmarUjala_World":      "https://www.amarujala.com/rss/world.xml",
    "AmarUjala_Business":   "https://www.amarujala.com/rss/business.xml",
    "AmarUjala_Sports":     "https://www.amarujala.com/rss/sports.xml",
    "AmarUjala_Entertainment": "https://www.amarujala.com/rss/entertainment.xml",
    "AmarUjala_Tech":       "https://www.amarujala.com/rss/technology.xml",
    "Jagran_National":      "https://www.jagran.com/rss/news-national.xml",
    "Jagran_UP_RSS":        "https://www.jagran.com/rss/news-uttar-pradesh.xml",
    "Jagran_World":         "https://www.jagran.com/rss/news-world.xml",
    "Jagran_Crime":         "https://www.jagran.com/rss/news-crime.xml",
    "Jagran_Business":      "https://www.jagran.com/rss/news-business.xml",
    "Jagran_Sports":        "https://www.jagran.com/rss/news-sports.xml",
    "Jagran_Entertainment": "https://www.jagran.com/rss/news-entertainment.xml",
    "Jagran_Tech":          "https://www.jagran.com/rss/news-technology.xml",
    "Hindustan_National":   "https://feed.livehindustan.com/rss/3127",
    "Hindustan_UP_RSS":     "https://feed.livehindustan.com/rss/3130",
    "Hindustan_World":      "https://feed.livehindustan.com/rss/3128",
    "Hindustan_Crime":      "https://feed.livehindustan.com/rss/3131",
    "Hindustan_Sports":     "https://feed.livehindustan.com/rss/3133",
    "Hindustan_Entertainment": "https://feed.livehindustan.com/rss/3132",
    "Hindustan_Business":   "https://feed.livehindustan.com/rss/3134",
    "Bhaskar_National":     "https://www.bhaskar.com/rss-feed/1061/",
    "Bhaskar_UP_RSS":       "https://www.bhaskar.com/rss-feed/1063/",
    "Bhaskar_World":        "https://www.bhaskar.com/rss-feed/1065/",
    "Bhaskar_Crime":        "https://www.bhaskar.com/rss-feed/1068/",
    "Bhaskar_Sports":       "https://www.bhaskar.com/rss-feed/1071/",
    "Bhaskar_Business":     "https://www.bhaskar.com/rss-feed/1064/",
    "Bhaskar_Entertainment":"https://www.bhaskar.com/rss-feed/1072/",
    "NDTV_India":           "https://feeds.feedburner.com/ndtvkhabar-samachar",
    "NDTV_World":           "https://feeds.feedburner.com/ndtvkhabar-duniya",
    "NDTV_Crime":           "https://feeds.feedburner.com/ndtvkhabar-crime",
    "NDTV_Politics":        "https://feeds.feedburner.com/ndtvkhabar-politics",
    "NDTV_Business":        "https://feeds.feedburner.com/ndtvkhabar-business",
    "NDTV_Sports":          "https://feeds.feedburner.com/ndtvkhabar-sports",
    "ABP_National":         "https://www.abplive.com/feed",
    "ABP_UP_RSS":           "https://www.abplive.com/states/up-uk/feed",
    "Patrika_National":     "https://api.patrika.com/rss/india-news",
    "Patrika_UP_RSS":       "https://api.patrika.com/rss/uttar-pradesh-news",
    "Patrika_Crime":        "https://api.patrika.com/rss/crime-news",
    "Patrika_World":        "https://api.patrika.com/rss/world-news",
    "Patrika_Business":     "https://api.patrika.com/rss/business-news",
    "News18_Hindi_National":"https://hindi.news18.com/rss/khabar/india.xml",
    "News18_Hindi_UP_RSS":  "https://hindi.news18.com/rss/khabar/uttar-pradesh.xml",
    "News18_Hindi_Crime":   "https://hindi.news18.com/rss/khabar/crime.xml",
    "News18_Hindi_Politics":"https://hindi.news18.com/rss/khabar/politics-and-nation.xml",
    "News18_Hindi_World":   "https://hindi.news18.com/rss/khabar/world.xml",
    "News18_Hindi_Business":"https://hindi.news18.com/rss/khabar/business.xml",
    "News18_Hindi_Sports":  "https://hindi.news18.com/rss/khabar/sports.xml",
    "News18_Hindi_Ent":     "https://hindi.news18.com/rss/khabar/entertainment.xml",
    "Reuters World":        "https://feeds.reuters.com/Reuters/worldNews",
    "BBC World":            "http://feeds.bbci.co.uk/news/world/rss.xml",
    "BBC India":            "http://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
    "Al Jazeera":           "https://www.aljazeera.com/xml/rss/all.xml",
    "Reddit India":         "https://www.reddit.com/r/india/.rss",
    "Reddit WorldNews":     "https://www.reddit.com/r/worldnews/.rss",
    "Reddit Geopolitics":   "https://www.reddit.com/r/geopolitics/.rss",
    "GaonConnection_RSS":   "https://www.gaonconnection.com/feed",
    "KrishiJagran_RSS":     "https://hindi.krishijagran.com/feed",
    "SarkariExam_RSS":      "https://www.sarkariresult.com/feed/",
}

# =========================================================
# LOCAL SCRAPERS
# =========================================================

LOCAL_SCRAPERS = {
    "AmarUjala_Kushinagar": "https://www.amarujala.com/uttar-pradesh/kushinagar",
    "AmarUjala_Gorakhpur":  "https://www.amarujala.com/uttar-pradesh/gorakhpur",
    "AmarUjala_Deoria":     "https://www.amarujala.com/uttar-pradesh/deoria",
    "AmarUjala_Maharajganj":"https://www.amarujala.com/uttar-pradesh/maharajganj",
    "AmarUjala_Basti":      "https://www.amarujala.com/uttar-pradesh/basti",
    "AmarUjala_UP":         "https://www.amarujala.com/uttar-pradesh",
    "AmarUjala_Crime":      "https://www.amarujala.com/crime",
    "AmarUjala_National":   "https://www.amarujala.com/india-news",
    "Bhaskar_Kushinagar":   "https://www.bhaskar.com/local/uttar-pradesh/kushinagar/",
    "Bhaskar_Gorakhpur":    "https://www.bhaskar.com/local/uttar-pradesh/gorakhpur/",
    "Bhaskar_Deoria":       "https://www.bhaskar.com/local/uttar-pradesh/deoria/",
    "Bhaskar_Maharajganj":  "https://www.bhaskar.com/local/uttar-pradesh/maharajganj/",
    "Bhaskar_Basti":        "https://www.bhaskar.com/local/uttar-pradesh/basti/",
    "Bhaskar_UP":           "https://www.bhaskar.com/local/uttar-pradesh/",
    "Bhaskar_Crime":        "https://www.bhaskar.com/crime/",
    "Bhaskar_National":     "https://www.bhaskar.com/national/",
    "Hindustan_Kushinagar": "https://www.livehindustan.com/uttar-pradesh/kushinagar",
    "Hindustan_Gorakhpur":  "https://www.livehindustan.com/uttar-pradesh/gorakhpur",
    "Hindustan_Deoria":     "https://www.livehindustan.com/uttar-pradesh/deoria",
    "Hindustan_Maharajganj":"https://www.livehindustan.com/uttar-pradesh/maharajganj",
    "Hindustan_Basti":      "https://www.livehindustan.com/uttar-pradesh/basti",
    "Hindustan_UP":         "https://www.livehindustan.com/uttar-pradesh",
    "Hindustan_Crime":      "https://www.livehindustan.com/crime",
    "Jagran_Gorakhpur":     "https://www.jagran.com/uttar-pradesh/gorakhpur-city-news-hindi.html",
    "Jagran_Kushinagar":    "https://www.jagran.com/uttar-pradesh/kushinagar-news-hindi.html",
    "Jagran_Deoria":        "https://www.jagran.com/uttar-pradesh/deoria-news-hindi.html",
    "Jagran_Maharajganj":   "https://www.jagran.com/uttar-pradesh/maharajganj-news-hindi.html",
    "Jagran_Basti":         "https://www.jagran.com/uttar-pradesh/basti-news-hindi.html",
    "Jagran_UP":            "https://www.jagran.com/uttar-pradesh/",
    "Jagran_Crime":         "https://www.jagran.com/crime/",
    "Jagran_National":      "https://www.jagran.com/news/national.html",
    "Patrika_Gorakhpur":    "https://www.patrika.com/gorakhpur-news/",
    "Patrika_Deoria":       "https://www.patrika.com/deoria-news/",
    "Patrika_UP":           "https://www.patrika.com/uttar-pradesh-news/",
    "Patrika_Crime":        "https://www.patrika.com/crime/",
    "Patrika_National":     "https://www.patrika.com/national-news/",
    "News18_UP":            "https://hindi.news18.com/uttar-pradesh/",
    "News18_Gorakhpur":     "https://hindi.news18.com/news/uttar-pradesh/gorakhpur/",
    "News18_Kushinagar":    "https://hindi.news18.com/news/uttar-pradesh/kushinagar/",
    "News18_Crime":         "https://hindi.news18.com/crime/",
    "News18_National":      "https://hindi.news18.com/india/",
    "ETVBharat_UP":         "https://www.etvbharat.com/hindi/uttar-pradesh",
    "ETVBharat_Gorakhpur":  "https://www.etvbharat.com/hindi/uttar-pradesh/state/gorakhpur",
    "ETVBharat_Kushinagar": "https://www.etvbharat.com/hindi/uttar-pradesh/state/kushinagar",
    "ETVBharat_Crime":      "https://www.etvbharat.com/hindi/crime",
    "ZeeUPUK":              "https://zeenews.india.com/hindi/india/up-uttarakhand",
    "ZeeUP":                "https://zeenews.india.com/hindi/tags/uttar-pradesh-news.html",
    "ZeeGorakhpur":         "https://zeenews.india.com/hindi/tags/gorakhpur-news.html",
    "ZeeCrime":             "https://zeenews.india.com/hindi/crime",
    "TV9_UP":               "https://www.tv9hindi.com/state/uttar-pradesh",
    "TV9_Gorakhpur":        "https://www.tv9hindi.com/tag/gorakhpur",
    "TV9_Kushinagar":       "https://www.tv9hindi.com/tag/kushinagar",
    "TV9_Crime":            "https://www.tv9hindi.com/crime",
    "AajTak_UP":            "https://www.aajtak.in/uttar-pradesh",
    "AajTak_Gorakhpur":     "https://www.aajtak.in/topic/gorakhpur",
    "AajTak_Kushinagar":    "https://www.aajtak.in/topic/kushinagar",
    "AajTak_Crime":         "https://www.aajtak.in/crime",
    "ABP_UP":               "https://www.abplive.com/states/up-uk",
    "ABP_Gorakhpur":        "https://www.abplive.com/topic/gorakhpur",
    "ABP_Kushinagar":       "https://www.abplive.com/topic/kushinagar",
    "IndiaTV_UP":           "https://www.indiatv.in/uttar-pradesh",
    "IndiaTV_Gorakhpur":    "https://www.indiatv.in/topic/gorakhpur",
    "IndiaTV_Kushinagar":   "https://www.indiatv.in/topic/kushinagar",
    "OneIndia_UP":          "https://hindi.oneindia.com/news/uttar-pradesh/",
    "OneIndia_Gorakhpur":   "https://hindi.oneindia.com/topic/gorakhpur/",
    "Webdunia_UP":          "https://hindi.webdunia.com/regional-hindi-news/uttar-pradesh/",
    "GorakhpurLive":        "https://gorakhpurlive.com/",
    "PurvanchalNews":       "https://purvanchalnews.com/",
    "UPTak":                "https://www.uptak.in/",
    "GaonConnection":       "https://www.gaonconnection.com/",
    "JanSatta_UP":          "https://www.jansatta.com/national/uttar-pradesh/",
    "UPPolice":             "https://uppolice.gov.in/",
    "PIB_UP":               "https://pib.gov.in/statewise.aspx?state=U",
    "IMD":                  "https://mausam.imd.gov.in/",
    "FloodAlerts":          "https://cwc.gov.in/flood-forecast-monitoring",
    "SarkariResult":        "https://www.sarkariresult.com/",
    "UPPSC":                "https://uppsc.up.nic.in/",
    "UPBoard":              "https://upmsp.edu.in/",
    "KrishiJagran":         "https://www.krishijagran.com/",
    "GaonConn_Agri":        "https://www.gaonconnection.com/kheti-kisani",
    "NHM_UP":               "https://upnrhm.gov.in/",
    "HealthDeptUP":         "http://up-health.in/",
}

# =========================================================
# YOUTUBE FEEDS
# =========================================================

YOUTUBE_FEEDS = {
    "AajTak":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCt4t-jeY85JegMlZ-E5UWtA",
    "ANI":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCV-JlfdG9A8ZkNTJbVnHGkg",
    "IndiaTV":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCttspZesZIDEwwpVIgoZtWQ",
    "TV9 Bharatvarsh":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCP0uG-mcMImgKnJz-VsQxKQ",
    "ABPNews":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCIvaH_4U3A3fkHPCB8-7ExQ",
    "ZeeNews":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCnGP-dm7gOtPQFSj8yEMnIg",
    "NDTV":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCZFMm1mMw0F81Z37aaEzTUA",
    "News18India":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UC6ebRQXZfOxPBqcMJEBmxEA",
    "RepublicBharat":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCFOfOOFBBHq5aOeT3GQjxSA",
    "SudarshannNews":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCGfGOxFJOfRFL6KzVkzT6eQ",
    "PrabhatKhabar":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCz_fTf5rObyAd9DQKsEAipA",
    "LallantopNews":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCx8Z14PpntdaxCt2hakbQLQ",
    "Samachar_Plus":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCPvnVbqIwWMVwWKbOJf7vlQ",
    "JanTak_TV":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCpKkGSV1yvA2JraTFYvOBrQ",
    "GorakhpurNews18":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCHExuHVAMnm0PgXxjCxGqjA",
    "PurvanchalSamachar":
        "https://www.youtube.com/feeds/videos.xml?channel_id=UCHfS6t7ZA3wJxRmAGMMIrjw",
}

# =========================================================
# FETCH NEWS — RSS FEEDS
# =========================================================

all_news = []
print("\n========== FETCHING RSS FEEDS ==========")

for category, url in RSS_FEEDS.items():
    try:
        print(f"📡 RSS: {category}")
        feed    = feedparser.parse(url)
        entries = feed.entries[:RSS_MAX_PER_FEED]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            process_func = partial(process_single_entry, category=category)
            futures = [executor.submit(process_func, entry) for entry in entries]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        all_news.append(result)
                except Exception as e:
                    print("❌ FUTURE ERROR:", e)

        print(f"  ✅ {category} RSS DONE — total so far: {len(all_news)}")
    except Exception as e:
        print(f"  ❌ RSS ERROR ({category}):", e)

# =========================================================
# FETCH NEWS — LOCAL SCRAPERS (parallel)
# =========================================================

print(f"\n========== SCRAPING {len(LOCAL_SCRAPERS)} LOCAL SITES ==========")

def _scrape_wrapper(args):
    site_name, site_url = args
    return scrape_local_site(site_name, site_url, max_items=SCRAPER_MAX_PER_SITE)

scraper_items = list(LOCAL_SCRAPERS.items())

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    scrape_futures = {executor.submit(_scrape_wrapper, item): item[0] for item in scraper_items}
    for future in as_completed(scrape_futures):
        site_name = scrape_futures[future]
        try:
            scraped = future.result()
            all_news.extend(scraped)
        except Exception as e:
            print(f"❌ SCRAPER FUTURE ERROR [{site_name}]: {e}")

print(f"\n✅ TOTAL RAW NEWS (RSS + Scrapers): {len(all_news)}")

if not all_news:
    print("❌ NO NEWS FOUND — exiting")
    exit()

# =========================================================
# TREND SIGNALS — YouTube feeds
# =========================================================

print("\n========== GATHERING TREND SIGNALS ==========")
trending_keywords = {}

for channel_name, yt_url in YOUTUBE_FEEDS.items():
    try:
        print(f"📺 YouTube Trends: {channel_name}")
        yt_feed = feedparser.parse(yt_url)
        for entry in yt_feed.entries[:10]:
            words = re.findall(r'\w+', entry.title.lower())
            for word in words:
                if len(word) > 3:
                    trending_keywords[word] = trending_keywords.get(word, 0) + 1.5
    except Exception as e:
        print(f"⚠️ YouTube Trend Error ({channel_name}):", e)

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
        if is_similar(news["headline"], existing["headline"]):
            duplicate_found = True
            break
    if not duplicate_found:
        filtered_news.append(news)

print(f"\n✅ UNIQUE NEWS ITEMS AFTER DEDUP: {len(filtered_news)}")

# =========================================================
# AI GENERATION (WITH EXPONENTIAL BACKOFF FOR 429 ERRORS)
# =========================================================

def generate_ai_response(prompt, headline="", category="", article_text="", link=""):

    if groq_client:
        for model_name in GROQ_MODELS:
            for attempt in range(3):
                try:
                    print(f"🔥 USING GROQ: {model_name}")
                    response = groq_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                        max_tokens=2048
                    )
                    result = response.choices[0].message.content
                    if result:
                        return result
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate_limit" in err_str:
                        wait = 5 * (2 ** attempt)
                        print(f"⚠️ GROQ 429 Rate Limit. Sleeping {wait}s...")
                        time.sleep(wait)
                    elif "402" in err_str:
                        print("❌ GROQ 402 (Insufficient Credits) — skipping entirely.")
                        break
                    else:
                        print("❌ GROQ FAILED:", e)
                        break

    if samba_client:
        for model_name in SAMBANOVA_MODELS:
            for attempt in range(3):
                try:
                    print(f"🔥 USING SAMBANOVA: {model_name}")
                    response = samba_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.8,
                        max_tokens=2048
                    )
                    result = response.choices[0].message.content
                    if result:
                        return result
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate_limit" in err_str:
                        wait = 5 * (2 ** attempt)
                        print(f"⚠️ SAMBANOVA 429 Rate Limit. Sleeping {wait}s...")
                        time.sleep(wait)
                    else:
                        print("❌ SAMBANOVA FAILED:", e)
                        break

    if hf_client:
        for model_name in HF_MODELS:
            try:
                print(f"🔥 USING HF: {model_name}")
                response = hf_client.chat_completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_tokens=2048
                )
                result = response.choices[0].message.content
                if result:
                    return result
            except Exception as e:
                if "402" in str(e):
                    print("❌ HF FAILED: 402 Payment Required — skipping HF entirely.")
                    break
                print("❌ HF FAILED:", e)

    if not OPENROUTER_API_KEY:
        print("⚠️ OPENROUTER_API_KEY not set — skipping OpenRouter")
    else:
        for model_name in OPENROUTER_MODELS:
            try:
                print(f"🔥 USING OPENROUTER: {model_name}")
                or_resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/",
                        "X-Title": "News Hunter Bot Matrix"
                    },
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.8,
                        "max_tokens": 2048
                    },
                    timeout=60
                )
                if or_resp.status_code == 200:
                    or_data    = or_resp.json()
                    or_content = or_data["choices"][0]["message"]["content"]
                    if or_content and or_content.strip():
                        print(f"✅ OPENROUTER SUCCESS: {model_name}")
                        return or_content
                    else:
                        print(f"⚠️ OPENROUTER empty response: {model_name}")
                elif or_resp.status_code == 404:
                    print(f"❌ OPENROUTER 404 (model removed): {model_name} — skipping")
                else:
                    print(f"❌ OPENROUTER HTTP {or_resp.status_code}: {model_name}")
                    print(or_resp.text[:300])
            except Exception as e:
                print(f"❌ OPENROUTER FAILED: {model_name} — {e}")

    # SMART FALLBACK FORMATTER
    print("\n⚠️ USING SMART FALLBACK FORMATTER")
    try:
        if not headline or headline.strip() == "":
            title_match = re.search(r"(?:news title|title):\s*(.+)", prompt, re.IGNORECASE)
            headline = title_match.group(1).strip() if title_match else "Breaking News"

        if not category or category.strip() == "":
            category_match = re.search(r"category:\s*(.+)", prompt, re.IGNORECASE)
            category = category_match.group(1).strip() if category_match else "News"

        if not article_text or len(article_text.strip()) < 50:
            article_match = re.search(
                r"(?:full article|article):\s*(.+?)(?:source link|link):",
                prompt, re.DOTALL | re.IGNORECASE
            )
            article_text = article_match.group(1).strip() if article_match else headline

        if not link or link.strip() == "":
            link_match = re.search(r"(?:source link|link):\s*(.+)", prompt, re.IGNORECASE)
            link = link_match.group(1).strip() if link_match else ""

        clean_text     = re.sub(r'\s+', ' ', article_text).strip()
        sentences      = re.split(r'(?<=[.!?।])\s+|;\s*', clean_text)
        important_lines = []

        for sentence in sentences:
            sentence = sentence.strip()
            if any(x in sentence.lower() for x in [
                "click here", "subscribe", "follow us", "read more",
                "advertisement", "copyright"
            ]):
                continue
            if len(sentence) > 30 and sentence.lower() not in headline.lower():
                important_lines.append(sentence)
            if len(important_lines) >= 10:
                break

        if len(important_lines) < 10:
            chunks = [clean_text[i:i+85].strip() for i in range(0, min(len(clean_text), 900), 85)]
            for chunk in chunks:
                if len(chunk) > 25 and chunk not in important_lines:
                    important_lines.append(chunk)
                if len(important_lines) >= 10:
                    break

        backup_points = [
            "इस महत्वपूर्ण घटनाक्रम के बाद संबंधित विभागों और प्रशासनिक टीमों को अलर्ट मोड पर रखा गया है।",
            "ग्राउंड जीरो पर मौजूद सुरक्षा बल और जांच एजेंसियां मामले की हर एक कड़ी को बारीकी से खंगाल रही हैं।",
            "स्थानीय प्रशासन ने आम जनता से किसी भी प्रकार की अफवाहों पर ध्यान न देने की अपील जारी की है।",
            "आधिकारिक सूत्रों के अनुसार, आने वाले कुछ घंटों में इस विषय पर एक विस्तृत प्रेस ब्रीफिंग की जा सकती है।",
            "इस संवेदनशील मामले पर देश-दुनिया के बड़े राजनीतिक और रणनीतिक विश्लेषकों की नजरें लगातार टिकी हुई हैं।"
        ]
        for pt in backup_points:
            if len(important_lines) >= 10:
                break
            important_lines.append(pt)

        summary  = "\n\n".join([f"• {line}" for line in important_lines[:12]])
        hashtags = f"#{category.replace(' ', '')} #BreakingNews #IndiaNews #ViralNews #FlashNews"

        return f"""
🔥 BREAKING NEWS
📰 {headline}
📂 Category: {category}
━━━━━━━━━━━━━━━
🌍 REAL SCRATCHED INSIGHTS:
{summary}
━━━━━━━━━━━━━━━
🤔 आपकी क्या राय है? इस पूरे मामले पर कमेंट बॉक्स में अपनी प्रतिक्रिया जरूर साझा करें।
📢 ऐसी ही देश-दुनिया और आपके इलाके की सबसे तेज और सटीक खबरों के लिए तुरंत follow करें।
🔗 Source:
{link}
{hashtags}
Emotion Score: 6/10
Virality Score: 6/10
Political Toxicity: 1/10
"""
    except Exception as e:
        print("❌ SMART FALLBACK CRITICAL ERROR:", e)
        return f"""
🔥 BREAKING NEWS
📰 {headline}
📂 Category: {category}
━━━━━━━━━━━━━━━
• {headline}
• ग्राउंड जीरो से आ रही हर पल की लाइव रिपोर्टिंग के लिए हमारे साथ बने रहें।
━━━━━━━━━━━━━━━
🤔 आपकी क्या राय है?
📢 ऐसी ही खबरों के लिए follow करें।
🔗 Source:
{link}
#BreakingNews #ViralNews
Emotion Score: 5/10
Virality Score: 5/10
Political Toxicity: 0/10
"""

# =========================================================
# KEYWORD BANKS
# =========================================================

LOCAL_WORDS = [
    "kushinagar", "gorakhpur", "hata", "padrauna", "maharajganj", "deoria", "basti", "ghaziapur",
    "azamgarh", "lucknow", "varanasi", "ballia", "mau", "jaunpur", "chauri chaura", "gonda",
    "bahraich", "ayodhya", "siddharthnagar", "sant kabir nagar", "sultanpur", "amethi", "mirzapur",
    "ghazipur", "ballia", "chandauli", "sonbhadra", "bhojpur", "purvanchal", "uttar pradesh", "up news",
    "कुशीनगर", "गोरखपुर", "हाटा", "पडरौना", "महराजगंज", "देवरिया", "पूर्वांचल", "उत्तर प्रदेश", "यूपी",
    "बस्ती", "गाजीपुर", "बनारस", "मऊ", "बलिया", "जौनपुर", "चौरी चौरा", "अयोध्या", "सिद्धार्थनगर",
    "kasia", "tamkuhi", "tamkuhi raj", "khadda", "captainganj", "ramkola", "seorahi", "fazilnagar",
    "sukrauli", "dudhai", "neboua", "nautanwa", "anandnagar", "pharenda", "campierganj", "pipraich",
    "gola", "khajni", "bansgaon", "sahjanwa", "barhalganj", "kauriram", "belaon", "urwa", "chargaon",
    "कसया", "तमकुही", "खड्डा", "कप्तानगंज", "रामकोला", "सेवरही", "फाजिलनगर", "सुकरौली", "नौतनवा",
    "कैम्पियरगंज", "पिपराइच", "गोला", "खजनी", "बांसगांव", "सहजनवा", "बड़हलगंज", "कौड़ीराम", "चरगाँव",
    "kasya", "bodh gaya", "buddha", "parnirvana", "thuthibari", "sonauli", "chhitauni", "salempur",
    "barhaj", "rudrapur", "gaura", "bhatni", "lar", "patherdeva", "tarkulwa", "desahi", "baitalpur",
    "gauri bazar", "bhaluani", "kaptanganj", "siswa", "nichlaul", "mithauna", "धानी", "सिसवा", "निचलौल",
    "मीठाउना", "ठूठीबारी", "सनौली", "छितौनी", "सलेमपुर", "बरहज", "रुद्रपुर", "भाटपार", "भटनी", "लार",
    "yogi", "adityanath", "yogi adityanath", "gorakhnath", "peeth", "mahant", "योगी", "आदित्यनाथ",
    "गोरखनाथ", "पीठ", "महंत", "up police", "gkp", "stf", "upstf", "dial 112", "daroga", "thana",
    "kotwali", "chowki", "sp office", "dm office", "vda", "gda", "nagarpalika", "pradhan", "bdo",
    "दारोगा", "थाना", "कोतवाली", "चौकी", "प्रधान", "बीडीओ", "सिपाही", "दबंग", "मुख्तार", "अंसारी",
    "bhojpuri", "bhojpuriya", "purvanchali", "ghaghra", "gandak", "rapti", "rohin", "ami", "kuwano",
    "sarayu", "boating accident", "floods", "purvanchal expressway", "gorakhpur link", "aiims gorakhpur",
    "brd medical college", "mmmut", "ddu", "kushinagar international airport", "buddha relics",
    "भोजपुरी", "पूर्वांचली", "घाघरा", "गंडक", "राप्ती", "सरयू", "बाढ़", "एक्सप्रेसवे", "एम्स", "बीआरडी",
    "menhdawal", "khalilabad", "hainsarbazar", "bangaon", "harraiya", "captainganj", "rudhauli",
    "bhanpur", "itwa", "domariaganj", "shohratgarh", "naugarh", "bansi", "alapur", "jalalpur",
    "tanda", "akbarpur", "bhiti", "katehari", "maharajganj", "sagri", "gopalpur", "azamgarh city",
    "nizamabad", "phoolpur", "didarganj", "lalganj", "mehnagar", "mubarakpur", "ghosi", "madhuban",
    "mohammadabad", "kopaganj", "rasra", "bansdih", "sikanderpur", "phephna", "bairia", "zahoorabad",
    "saidpur", "zamania", "muhammadabad", "jakhanian", "kerakat", "machhlishahr", "mariahu", "shahganj",
    "mungra badshahpur", "malhani", "pindra", "shivpur", "cantt", "ajagara", "rohania", "sevapuri",
    "खलीलाबाद", "हरैया", "रुधौली", "इटवा", "डुमरियागंज", "शोहरतगढ़", "नौगढ़", "बांसी", "जलालपुर",
    "टांडा", "अकबरपुर", "सगरी", "निज़ामबाद", "फूलपुर", "दीदारगंज", "लालगंज", "मऊनाथ", "घोषी", "मधुबन",
    "रसड़ा", "बांसडीह", "सिकंदरपुर", "फेफना", "बैरिया", "जमानिया", "जखनिया", "केराकत", "मछलीशहर", "मड़ियाहू",
    "asuran", "golghar", "buxipur", "mohaddipur", "taramandal", "medical road", "singharia",
    "bargadwa", "shahpur", "tiwaripur", "rajendra nagar", "shastri nagar", "civil lines", "naushad",
    "transport nagar", "gita press", "surajkund", "gorakhnath temple", "ramgarhtal", "leela",
    "असुरन", "गोलघर", "बक्शीपुर", "मोहद्दीपुर", "तारामंडल", "बरगदवा", "शाहपुर", "तिवारीपुर", "नौसड़",
    "गीता प्रेस", "सूरजकुंड", "रामगढ़ताल", "गोरखनाथ मंदिर", "कौवाबाग", "पादरी बाजार", "झरना टोला",
    "kasya raj", "matha kaur", "ramabhar", "stupa", "wat thai", "linh son", "burmese temple",
    "tibetan monastery", "shahi masjid", "kurmaghat", "narayani river", "choti gandak", "bansi river",
    "ghaghra bridge", "dohrighat", "barhaj ghat", "ayodhya ram mandir route", "purvanchal handloom",
    "terracotta", "bhathat", "jangal kauria", "pali", "sahjanwa industrial", "gida", "unwal",
    "piprauli", "khorabar", "bramhpur", "gayanpur", "gopalpur chowk", "paniyara", "partawal",
    "shyamdeurawa", "bhagwanpur", "bridgemanganj", "kolhui", "purandarpur", "lakhima", "sonauli border",
    "belahiya", "valmiki nagar", "bagaha", "narkatiaganj", "chanpatia", "bettiah", "motihari",
    "बेलहिया", "बाल्मीकि नगर", "बगाहा", "नरकटियागंज", "बेतिया", "मोतीहारी", "पनियरा", "परतावल", "श्यामदेउरवा"
]

HIGH_PRIORITY_KEYWORDS = [
    "modi", "narendra modi", "bjp", "amit shah", "jp nadda", "rajnath singh", "nirmala sitharaman",
    "s jaishankar", "nitin gadkari", "yogi", "pmo", "pmo india", "nda", "rss", "mohan bhagwat",
    "lok sabha", "rajya sabha", "parliament", "sansad", "cabinet decision", "central govt",
    "मोदी", "नरेन्द्र मोदी", "भाजपा", "अमित शाह", "नड्डा", "राजनाथ", "जयशंकर", "गडकरी", "पीएमओ",
    "संसद", "लोकसभा", "राज्यसभा", "मंत्रिमंडल", "कैबिनेट", "केंद्र सरकार", "आरएसएस", "मोहन भागवत",
    "rahul gandhi", "priyanka gandhi", "sonia gandhi", "congress", "inc", "kharge", "akhilesh",
    "yadav", "dimple yadav", "kejriwal", "aap", "mamata banerjee", "tmc", "stalin", "nitish kumar",
    "jdu", "rjd", "tejashwi", "lalu yadav", "mayawati", "bsp", "india alliance", "mva", "sharad pawar",
    "राहुल गांधी", "प्रियंका", "कांग्रेस", "खड़गे", "अखिलेश यादव", "डिंपल", "केजरीवाल", "आप", "ममता",
    "नीतीश कुमार", "तेजस्वी", "लालू यादव", "मायावती", "बसपा", "सपा", "विपक्ष", "इंडिया गठबंधन",
    "election", "chunav", "voting", "matdan", "evm", "vvpat", "booth capturing", "exit poll",
    "opinion poll", "by-poll", "voter id", "nomination", "rally", "jan sabha", "code of conduct",
    "चुनाव", "मतदान", "वोटिंग", "ईवीएम", "एग्जिट पोल", "नामांकन", "रैली", "जनसभा", "आचार संहिता",
    "police", "arrest", "court", "supreme court", "high court", "cj", "ed", "cbi", "raid", "encounter",
    "jail", "custody", "bailable", "non-bailable", "fir", "charge sheet", "summons", "interrogation",
    "stf", "nia", "income tax", "it raid", "judgement", "verdict", "anticipatory bail", "warrant",
    "पुलिस", "गिरफ्तार", "कोर्ट", "सुप्रीम कोर्ट", "हाई कोर्ट", "जज", "ईडी", "सीबीआई", "छापा", "रेड",
    "एनकाउंटर", "जेल", "हिरासत", "जमानत", "एफआईआर", "चार्जशीट", "समन", "पूछताछ", "एनआईए", "वारंट",
    "pakistan", "china", "army", "attack", "war", "israel", "gaza", "iran", "border", "loc", "lac",
    "iaf", "navy", "surgical strike", "drone", "missile", "agniveer", "agnipath", "iaf fighter",
    "sukhoi", "rafale", "drdo", "indian army", "bsh", "crpf", "bsf", "itbp", "nsg", "terrorist",
    "encounter", "infiltration", "commanders", "nuclear", "submarines", "aircraft carrier",
    "चीन", "सेना", "हमला", "युद्ध", "इजरायल", "गाजा", "ईरान", "बॉर्डर", "एलओसी", "राफेल",
    "अग्निवीर", "सर्जिकल स्ट्राइक", "ड्रोन", "मिसाइल", "आतंकवादी", "घुसपैठ", "परमाणु", "फौजी", "जवान",
    "trump", "donald trump", "putin", "russia", "biden", "america", "us news", "white house",
    "kremlin", "pentagon", "united nations", "unsc", "nato", "zelensky", "ukraine war", "xi jinping",
    "beijing", "washington", "london", "downing street", "bbc", "reuters", "al jazeera",
    "ट्रम्प", "पुतिन", "रूस", "बिडेन", "अमेरिका", "व्हाइट हाउस", "नाटो", "यूक्रेन", "शी जिनपिंग",
    "gst", "income tax slab", "budget", "rbi", "repo rate", "inflation", "gdp", "share market",
    "nifty", "sensex", "sebi", "ipo", "crypto", "bitcoin", "vande bharat", "bullet train", "nhai",
    "highway", "toll plaza", "ayushman bharat", "pm kisan", "ration card", "freebie", "subsidy",
    "बजट", "महंगाई", "शेयर बाजार", "सेंसेक्स", "निफ्टी", "क्रिप्टो", "वंदे भारत", "हाइवे", "टोल प्लाजा",
    "neet", "paper leak", "scam", "railway", "rrb", "ssc", "exam cancel", "leak", "fraud", "bribe",
    "cgl", "upsc", "paper लीक", "घोटाला", "nta", "answer key", "cutoff", "result out", "cheating",
    "dummy candidate", "solvers gang", "marks rigging", "re-exam", "grace marks", "uproar",
    "नीट", "पेपर लीक", "घोटाला", "रेलवे", "परीक्षा रद्द", "धांधली", "रिजल्ट", "धोखाधड़ी", "रिश्वत"
]

EMOTIONAL_WORDS = [
    "shock", "fear", "warning", "danger", "viral", "anger", "violence", "protest", "ebola", "virus",
    "outbreak", "alert", "breaking", "urgent", "hazard", "panic", "expose", "mystery", "miracle",
    "tragedy", "disaster", "catastrophe", "scandal", "shameful", "horrific", "brutal", "deadly",
    "conspiracy", "covert", "hidden camera", "caught on camera", "live stream", "unbelievable",
    "bizarre", "myth", "haunted", "incident", "accident", "clash", "riot", "tension", "stampede",
    "सावधान", "बड़ी खबर", "खुलासा", "हादसा", "मौत", "हड़कंप", "सनसनी", "खौफ", "शर्मनाक", "चमत्कार",
    "धोखा", "तबाही", "खतरा", "चेतावनी", "लाइव वीडियो", "सच्चाई", "मच गया बवाल", "रोंगटे खड़े", "दहशत",
    "सीसीटीवी", "शर्मसार", "भयंकर", "दर्दनाक", "बड़ी लापरवाही", "सनसनीखेज", "कांप उठी रूह", "देखें वीडियो",
    "खौफनाक", "दिल दहला देने वाला", "अंधविश्वास", "बवाल", "हंगामा", "आक्रोश", "गुस्सा", "मातम", "चीख-पुकार",
    "खूनी खेल", "वारदात", "भंडाफोड़", "काली करतूत", "सच्चा चेहरा", "गुप्त", "रहस्य", "अजीबोगरीब",
    "क्राइम", "मर्डर", "हादसा", "भूकंप", "आग लगी", "लूट", "गैंग", "बलात्कार", "गैंगस्टर", "माफिया",
    "बदमाश", "गोलीबारी", "मर्डर केस", "फिरौती", "किडनैपिंग", "अपहरण", "डकैती", "चोरी", "फिरौती",
    "शूटर", "सुपारी", "शव बरामद", "लाश", "हत्याकांड", "गोली मारकर हत्या", "चाकूबाजी", "एसिड अटैक",
    "रेप", "गैंगरेप", "उत्पीड़न", "घूसखोर", "रंगे हाथ", "ब्लैकमेल", "साइबर ठगी", "हनीट्रैप", "फर्जीवाड़ा",
    "ठग", "जालसाज", "माफिया राज", "बुलडोजर", "कुर्की", "गैंगवार", "हिस्ट्रीशीटर", "इनामी बदमाश",
    "blast", "explosion", "fire", "derailment", "train crash", "plane crash", "collision", "head-on",
    "hit and run", "drowning", "lightning", "earthquake", "cloudburst", "landslide", "heatwave",
    "cylinder blast", "short circuit", "building collapse", "bridge collapse", "poisonous", "gas leak",
    "धमाका", "विस्फोट", "आगजनी", "ट्रेन हादसा", "पटरी से उतरी", "टक्कर", "हिट एंड रन", "डूबने से मौत",
    "आकाशीय बिजली", "बादल फटा", "लैंडस्लाइड", "सिलेंडर ब्लास्ट", "शॉर्ट सर्किट", "जहरीली गैस", "दम घुटने",
    "trending", "viral reel", "facebook trend", "instagram viral", "twitter fight", "boycott",
    "trolled", "public bashing", "slapped", "abused", "viral audio", "leaked video", "crying video",
    "suicide note", "last video", "stunt video", "speeding car", "fight video", "road rage",
    "drunk driving", "high speed", "reckless", "suspension", "terminated", "action taken",
    "ट्रेंडिंग", "वायरल रील", "ट्विटर वॉर", "बॉयकाट", "ट्रोल", "पब्लिक ने पीटा", "थप्पड़", "गालीगलौज",
    "वायरल ऑडियो", "लीक वीडियो", "रोते हुए", "स्टंट वीडियो", "रोड रेज", "नशे में धुत", "सस्पेंड", "बर्खास्त",
    "emotional", "heartbreaking", "justice", "cheated", "exposed", "fake news", "mafia encounter",
    "भावुक", "दिल टूट गया", "न्याय", "इंसाफ", "धोखेबाज", "फर्जी वीडियो", "सच्चा खुलासा", "कच्चा चिट्ठा"
]

# =========================================================
# MAIN SCORING + AI LOOP
# =========================================================

results      = []
article_idx  = 0   # counter for article_store keys

HYPER_LOCAL_SCRAPER_KEYS = {
    k for k in LOCAL_SCRAPERS
    if any(loc in k.lower() for loc in [
        "kushinagar", "gorakhpur", "maharajganj", "deoria", "basti",
        "purvanchal", "gorakhpurlive", "purvanchalnews", "uptak"
    ])
}

for row in filtered_news:

    headline     = row["headline"]
    category     = row["category"]
    link         = row["link"]
    article_text = row["article"]
    source       = row.get("source", "rss")

    is_local = (
        category in ['Kushinagar', 'Gorakhpur', 'Maharajganj', 'Deoria', 'Basti',
                     'GNews_Kushinagar_HI', 'GNews_Gorakhpur_HI', 'GNews_Purvanchal_HI']
        or category in HYPER_LOCAL_SCRAPER_KEYS
        or any(loc in category.lower() for loc in [
            "kushinagar", "gorakhpur", "maharajganj", "deoria", "basti", "purvanchal"
        ])
    )

    print("=" * 80)
    print(f"SOURCE  : {source.upper()} | CATEGORY: {category}")
    print(f"HEADLINE: {headline}")
    print("=" * 80)

    headline_lower = headline.lower()
    headline_words = re.findall(r'\w+', headline_lower)

    india_score        = 0
    geo_score          = 0
    emotion_bonus      = 0
    controversy_bonus  = 0
    local_bonus        = 0
    trending_score     = 0

    for word in headline_words:
        if word in trending_keywords:
            trending_score += trending_keywords[word]
        if word in headline_frequencies:
            trending_score += (headline_frequencies[word] * 0.20)

    trending_score = min(round(trending_score, 2), 5.0)

    for keyword in LOCAL_WORDS:
        if keyword in headline_lower:
            local_bonus += 2.0

    for keyword in HIGH_PRIORITY_KEYWORDS:
        if keyword in headline_lower:
            india_score += 1.5

    for word in EMOTIONAL_WORDS:
        if word in headline_lower:
            emotion_bonus += 1.2

    if category in ["India Politics", "India Pakistan", "India China", "Defense",
                    "GNews_YogiUP", "GNews_UPCrime", "GNews_UPPolice"]:
        geo_score += 3

    if category in ["Crime", "Breaking"]:
        controversy_bonus += 2

    if source == "scraper" and category in HYPER_LOCAL_SCRAPER_KEYS:
        local_bonus += 1.5

    raw_score = (
        india_score        * 1.3 +
        geo_score          * 1.0 +
        trending_score     * 1.5 +
        emotion_bonus      * 1.2 +
        controversy_bonus  * 1.0 +
        local_bonus        * 1.5
    )

    if is_local:
        raw_score += LOCAL_NEWS_BONUS

    raw_score = round(raw_score, 2)
    print(f"📊 PRE SCORE: {raw_score}")

    if raw_score < 8.0:
        print(f"⏭️ SKIPPED LOW PRE-SCORE: {headline} ({raw_score})")
        continue

    print("✅ PRE-SCORE PASSED — RUNNING AI ENGINE")

    prompt = f"""
You are an elite viral news content writer, Facebook monetization strategist, and social media journalist in Hindi Language.
Create a high-engagement, monetization-focused Hindi social media article optimized for social networks.

Structure the output as follows:
Viral headline: Emotional, curiosity-based, Opening hook (2–3 lines) with Grabs attention.
Main article (300–400 words): Short paragraphs explaining the event details.
Key highlights: 3–5 important bullet points other than main article.
Public reaction(2-4): Summary of internet viewpoints(2-3 sentence) and Open-ended questions(2-3 question).
Call to action: Direct users to social handles named "karan_patel_kushinagar".

Take your input data as:
Title: {headline}
Category: {category}
Full Article: {article_text}
Source Link: {link}
"""

    try:

        ai_result = generate_ai_response(
            prompt=prompt,
            headline=headline,
            category=category,
            article_text=article_text,
            link=link
        )

        emotion_match  = re.search(r"Emotion Score:\s*(\d+)/10", ai_result)
        virality_match = re.search(r"Virality Score:\s*(\d+)/10", ai_result)
        toxicity_match = re.search(r"Political Toxicity:\s*(\d+)/10", ai_result)

        emotion_score  = int(emotion_match.group(1))  if emotion_match  else 5
        virality_score = int(virality_match.group(1)) if virality_match else 5
        toxicity_score = int(toxicity_match.group(1)) if toxicity_match else 0

        ai_contribution = (virality_score * 0.20) + (emotion_score * 0.20)

        final_score = (
            ai_contribution         +
            trending_score          +
            (india_score     * 0.5) +
            (geo_score       * 0.5) +
            (emotion_bonus   * 0.3) +
            (controversy_bonus * 0.3) -
            (toxicity_score  * 0.2)
        )

        if is_local:
            final_score += LOCAL_NEWS_BONUS

        final_score = round(final_score, 2)

        if final_score < 8.5:
            print(f"⏭️ SKIPPED LOW FINAL SCORE: {headline} ({final_score})")
            continue

        print(f"🚀 FINAL SCORE PASSED ({final_score}) — SENDING TELEGRAM")

        cleaned_link = clean_url(link)
        short_link   = shorten_url(cleaned_link)

        telegram_message = f"""
🔥 VIRAL NEWS
📰 {headline}
📂 Category: {category}
🗂️ Source Type: {source.upper()}
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
📸 Instagram: https://www.instagram.com/karan_patel_kushinagar  
▶️ YouTube: https://youtube.com/@karan_patel_kushinagar
📘 Facebook: https://www.facebook.com/share/1D627R2qFV/
"""

        # =====================================================
        # STORE ARTICLE IN MEMORY FOR BUTTON CLICK  ← NEW
        # =====================================================

        article_idx += 1

        article_store[article_idx] = {
            "headline":      headline,
            "category":      category,
            "link":          short_link,
            "emotion_score": emotion_score,
            "virality_score":virality_score,
            "toxicity_score":toxicity_score,
            "trending_score":trending_score,
            "final_score":   final_score,
            "ai_result":     ai_result,
            "source":        source,
        }

        # =====================================================
        # SEND TO ALL CHAT IDs WITH 🖼️ GENERATE IMAGE BUTTON
        # ← NEW: replaces the old send_telegram_message call
        # =====================================================

        chat_ids = _parse_chat_ids(TELEGRAM_CHAT_IDS)
        if not chat_ids and CHAT_ID:
            chat_ids = [CHAT_ID]

        for cid in chat_ids:
            send_article_with_button(cid, article_idx, telegram_message)

        print(f"✅ ARTICLE #{article_idx} SENT TO TELEGRAM WITH BUTTON")

        results.append({
            "source":         source,
            "category":       category,
            "headline":       headline,
            "link":           link,
            "short_link":     short_link,
            "emotion_score":  emotion_score,
            "virality_score": virality_score,
            "toxicity_score": toxicity_score,
            "trending_score": trending_score,
            "final_score":    final_score,
            "ai_result":      ai_result
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
    results_df = results_df.sort_values(by="final_score", ascending=False)
    csv_path   = os.path.join(SCRIPT_DIR, "viral_reel_news.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV SAVED: {csv_path}")
    print(f"✅ TOTAL SENT: {len(results_df)}")
else:
    print("\n❌ NO HIGH VIRAL NEWS FOUND")

# =========================================================
# START LISTENING FOR 🖼️ BUTTON CLICKS  ← NEW
# Waits 30 minutes for you to tap buttons in Telegram
# =========================================================

if article_store:
    print(f"\n✅ {len(article_store)} articles sent. Waiting for your image button taps...")
    stop_event = threading.Event()
    try:
        listen_for_button_clicks(stop_event, timeout_minutes=30)
    except KeyboardInterrupt:
        print("\n🛑 Stopped. Goodbye!")
        stop_event.set()
else:
    print("\n❌ No articles passed scoring — nothing to listen for.")
