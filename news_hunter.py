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
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from difflib import SequenceMatcher
import random
import io
import urllib.parse

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

print("=" * 50)
print("🔑 KEY STATUS:")
print(f"   GROQ       : {'✅ SET' if GROQ_API_KEY else '❌ MISSING'}")
print(f"   SAMBANOVA  : {'✅ SET' if SAMBANOVA_API_KEY else '❌ MISSING'}")
print(f"   HF_TOKEN   : {'✅ SET' if HF_TOKEN else '❌ MISSING'}")
print(f"   OPENROUTER : {'✅ SET — ' + OPENROUTER_API_KEY[:8] + '...' if OPENROUTER_API_KEY else '❌ MISSING'}")
print(f"   BOT_TOKEN  : {'✅ SET' if BOT_TOKEN else '❌ MISSING'}")
print(f"   CHAT_ID    : {'✅ SET' if CHAT_ID else '❌ MISSING'}")
print(f"   TELEG_CHAT : {'✅ SET' if TELEGRAM_CHAT_IDS else '❌ MISSING'}")
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

# Personal Branding Info — used in image prompt generator
PERSONAL_BRAND = {
    "name":        "Karan Patel",
    "handle":      "@karan_patel_kushinagar",
    "instagram":   "https://www.instagram.com/karan_patel_kushinagar",
    "youtube":     "https://youtube.com/@karan_patel_kushinagar",
    "facebook":    "https://www.facebook.com/share/1D627R2qFV/",
    "tagline":     "कुशीनगर की आवाज़ | सच्ची खबर, सबसे पहले",
    "logo_desc":   "minimalist 'KP' monogram logo in gold on dark background",
    "color_theme": "deep navy blue (#0D1B2A) with gold (#D4AF37) accents and white text",
}

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
# TELEGRAM — TEXT ONLY
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
# CLEAN RSS FEEDS (LOW DUPLICATES + HIGH VIRAL POTENTIAL)
# =========================================================

RSS_FEEDS = {

    # =====================================================
    # HYPERLOCAL / PURVANCHAL
    # =====================================================

    "Kushinagar_HI":
        "https://news.google.com/rss/search?q=कुशीनगर&hl=hi-IN&gl=IN&ceid=IN:hi",

    "Gorakhpur_HI":
        "https://news.google.com/rss/search?q=गोरखपुर&hl=hi-IN&gl=IN&ceid=IN:hi",

    "Deoria_HI":
        "https://news.google.com/rss/search?q=देवरिया&hl=hi-IN&gl=IN&ceid=IN:hi",

    "Maharajganj_HI":
        "https://news.google.com/rss/search?q=महराजगंज&hl=hi-IN&gl=IN&ceid=IN:hi",

    "Basti_HI":
        "https://news.google.com/rss/search?q=बस्ती+उत्तर+प्रदेश&hl=hi-IN&gl=IN&ceid=IN:hi",

    "Purvanchal_HI":
        "https://news.google.com/rss/search?q=पूर्वांचल&hl=hi-IN&gl=IN&ceid=IN:hi",

    # =====================================================
    # UTTAR PRADESH TRENDING
    # =====================================================

    "UP_Breaking":
        "https://news.google.com/rss/search?q=उत्तर+प्रदेश+ब्रेकिंग&hl=hi-IN&gl=IN&ceid=IN:hi",

    "UP_Politics":
        "https://news.google.com/rss/search?q=उत्तर+प्रदेश+राजनीति&hl=hi-IN&gl=IN&ceid=IN:hi",

    "UP_Crime":
        "https://news.google.com/rss/search?q=उत्तर+प्रदेश+अपराध&hl=hi-IN&gl=IN&ceid=IN:hi",

    "UP_Weather":
        "https://news.google.com/rss/search?q=यूपी+मौसम+बारिश+आंधी&hl=hi-IN&gl=IN&ceid=IN:hi",

    # =====================================================
    # INDIA TRENDING
    # =====================================================

    "India_Breaking":
        "https://news.google.com/rss/search?q=india+breaking&hl=en-IN&gl=IN&ceid=IN:en",

    "India_Politics":
        "https://news.google.com/rss/search?q=india+politics&hl=en-IN&gl=IN&ceid=IN:en",

    "India_China":
        "https://news.google.com/rss/search?q=india+china&hl=en-IN&gl=IN&ceid=IN:en",

    "India_Pakistan":
        "https://news.google.com/rss/search?q=india+pakistan&hl=en-IN&gl=IN&ceid=IN:en",

    # =====================================================
    # VIRAL CATEGORIES
    # =====================================================

    "Defense":
        "https://news.google.com/rss/search?q=india+army+missile+defense&hl=en-IN&gl=IN&ceid=IN:en",

    "AI":
        "https://news.google.com/rss/search?q=artificial+intelligence&hl=en-IN&gl=IN&ceid=IN:en",

    "World":
        "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",

    # =====================================================
    # PREMIUM RSS SOURCES
    # =====================================================

    "AmarUjala_UP_RSS":
        "https://www.amarujala.com/rss/uttar-pradesh.xml",

    "Jagran_UP_RSS":
        "https://www.jagran.com/rss/news-uttar-pradesh.xml",

    "Hindustan_UP_RSS":
        "https://feed.livehindustan.com/rss/3130",

    "Bhaskar_UP_RSS":
        "https://www.bhaskar.com/rss-feed/1063/",

    "ABP_UP_RSS":
        "https://www.abplive.com/states/up-uk/feed",

    "News18_Hindi_UP_RSS":
        "https://hindi.news18.com/rss/khabar/uttar-pradesh.xml",

    # =====================================================
    # GLOBAL + GEOPOLITICS
    # =====================================================

    "Reuters_World":
        "https://feeds.reuters.com/Reuters/worldNews",

    "BBC_India":
        "http://feeds.bbci.co.uk/news/world/asia/india/rss.xml",

    "Al_Jazeera":
        "https://www.aljazeera.com/xml/rss/all.xml",

    # =====================================================
    # REDDIT SIGNALS (OPTIONAL BUT GOOD)
    # =====================================================

    "Reddit_India":
        "https://www.reddit.com/r/india/.rss",

    "Reddit_WorldNews":
        "https://www.reddit.com/r/worldnews/.rss",

    "Reddit_Geopolitics":
        "https://www.reddit.com/r/geopolitics/.rss",
}


# =========================================================
# CLEAN LOCAL SCRAPERS
# =========================================================

LOCAL_SCRAPERS = {

    # =====================================================
    # TOP UP SOURCES
    # =====================================================

    "AmarUjala_UP":
        "https://www.amarujala.com/uttar-pradesh",

    "Bhaskar_UP":
        "https://www.bhaskar.com/local/uttar-pradesh/",

    "Hindustan_UP":
        "https://www.livehindustan.com/uttar-pradesh",

    "Jagran_UP":
        "https://www.jagran.com/uttar-pradesh/",

    "News18_UP":
        "https://hindi.news18.com/uttar-pradesh/",

    "TV9_UP":
        "https://www.tv9hindi.com/state/uttar-pradesh",

    "AajTak_UP":
        "https://www.aajtak.in/uttar-pradesh",

    "UPTak":
        "https://www.uptak.in/",

    # =====================================================
    # LOCAL HYPERLOCAL BONUS
    # =====================================================

    "GorakhpurLive":
        "https://gorakhpurlive.com/",

    "PurvanchalNews":
        "https://purvanchalnews.com/",
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
                        "HTTP-Referer": "https://github.com/karanpatelgit/ai_news_bot",
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
# AI IMAGE PROMPT GENERATOR
# Produces a ready-to-paste Midjourney / DALL-E / Ideogram
# prompt for a 1080 × 1350 px professional news post.
# Sent as a SEPARATE Telegram message chunk after the article.
# =========================================================

def generate_image_prompt(
    headline: str,
    category: str,
    article_text: str,
    final_score: float,
    emotion_score: int,
    short_link: str,
) -> str:
    """
    Uses the AI cascade to write a detailed image-generation prompt.
    Returns the raw prompt string (not an image).
    """

    # ── Extract a 1–2 sentence news summary for the prompt ──────────────
    clean_text = re.sub(r'\s+', ' ', article_text or headline).strip()
    sentences  = re.split(r'(?<=[.!?।])\s+', clean_text)
    summary_for_prompt = " ".join(sentences[:2])[:300]

    # ── Decide mood / colour palette based on category ──────────────────
    mood_map = {
        "Crime":         ("dark red warning atmosphere, crime scene yellow tape",         "crimson and dark red"),
        "Breaking":      ("urgent breaking news red-alert broadcast style",               "red, white, electric blue"),
        "Defense":       ("military patriotic Indian tricolor",                           "saffron, white, green, navy"),
        "India Politics":("formal political rally stage with Indian flags",               "deep blue, saffron, white"),
        "India Pakistan":("tense border atmosphere, Indian Army soldiers",                "olive green, desert beige"),
        "Sports":        ("vibrant stadium crowd energy, dynamic action lighting",        "gold, electric blue, white"),
        "Entertainment": ("Bollywood glam spotlight, glamorous neon lights",              "pink, gold, deep purple"),
        "Health":        ("clean medical professional environment, calming",              "mint green, white, light blue"),
        "Business":      ("corporate futuristic city skyline financial district",         "navy blue, gold, silver"),
        "Technology":    ("futuristic digital circuit glowing neon cyber aesthetic",      "electric blue, neon green, black"),
        "World":         ("global world map hologram geopolitical",                       "blue, silver, white"),
        "Kushinagar":    ("rural UP landscape golden fields sunrise, village community",  "golden yellow, green, earthy brown"),
        "Gorakhpur":     ("city of Gorakhpur urban streetscape, Gorakhnath temple hint",  "saffron, maroon, white"),
    }

    # Fall back: check for partial category matches
    mood_desc, palette = "dramatic news broadcast professional studio", "deep navy, gold, white"
    for key, (m, p) in mood_map.items():
        if key.lower() in category.lower():
            mood_desc, palette = m, p
            break

    # ── Emotional intensity cue ──────────────────────────────────────────
    if emotion_score >= 8:
        intensity = "high-drama cinematic lighting, bold shadows, urgent tension"
    elif emotion_score >= 5:
        intensity = "balanced editorial lighting, confident and impactful"
    else:
        intensity = "clean calm informational style, professional neutrality"

    # ── Build the meta-prompt for the AI to refine ──────────────────────
    meta_prompt = f"""
You are a world-class creative director specialising in viral social media news post design.
Generate ONE highly detailed image-generation prompt for Midjourney / DALL-E 3 / Ideogram
to create a 1080 × 1350 px portrait professional news post image.

NEWS CONTEXT:
- Headline  : {headline}
- Category  : {category}
- Summary   : {summary_for_prompt}
- Mood cue  : {mood_desc}
- Palette   : {palette}
- Intensity : {intensity}

PERSONAL BRANDING to embed visually in the prompt:
- Creator name  : {PERSONAL_BRAND["name"]}
- Handle        : {PERSONAL_BRAND["handle"]}
- Tagline       : {PERSONAL_BRAND["tagline"]}
- Logo hint     : {PERSONAL_BRAND["logo_desc"]}
- Color theme   : {PERSONAL_BRAND["color_theme"]}

MANDATORY DESIGN REQUIREMENTS you must include in the prompt:
1. Canvas: portrait 1080×1350 px, no cropping
2. Top strip: bold "BREAKING NEWS" or category label in white on coloured ribbon
3. Centre: dramatic visual scene that represents the news (photorealistic, no text in the scene)
4. Lower third: dark gradient overlay with bold Devanagari + English headline text placeholder
5. Bottom bar: brand colours ({PERSONAL_BRAND["color_theme"]}) with watermark "{PERSONAL_BRAND["handle"]}" and tagline
6. Typography style: bold condensed sans-serif for English, Noto Sans Devanagari style for Hindi
7. Style: ultra-HD photorealistic editorial magazine cover, 8K quality, professional post-processing
8. NO fake logos, NO real brand marks, NO copyrighted symbols

Return ONLY the image prompt — no explanation, no markdown, no preamble. 
The prompt must be one long paragraph of 120–180 words.
""".strip()

    print("🎨 GENERATING AI IMAGE PROMPT...")
    image_prompt = generate_ai_response(
        prompt=meta_prompt,
        headline=headline,
        category=category,
        article_text=article_text,
        link=""
    )

    # ── Fallback static prompt if AI fails ──────────────────────────────
    if not image_prompt or len(image_prompt.strip()) < 60:
        image_prompt = (
            f"Ultra-HD photorealistic professional social media news post, portrait 1080x1350px. "
            f"Top: bold red ribbon strip with white text 'BREAKING NEWS — {category}'. "
            f"Centre: dramatic {mood_desc} scene representing the news event, cinematic {intensity}, "
            f"colour palette of {palette}. Lower third: dark semi-transparent gradient overlay with "
            f"bold white condensed headline placeholder text. "
            f"Bottom bar in {PERSONAL_BRAND['color_theme']} colours featuring watermark "
            f"'{PERSONAL_BRAND['handle']}' in gold and tagline '{PERSONAL_BRAND['tagline']}' "
            f"in small white text. {PERSONAL_BRAND['logo_desc']} in bottom-right corner. "
            f"Style: editorial magazine cover quality, 8K, professional post-processing, no fake logos."
        )

    return image_prompt.strip()


def build_image_prompt_message(
    headline: str,
    category: str,
    article_text: str,
    final_score: float,
    emotion_score: int,
    short_link: str,
    article_idx: int,
) -> str:
    """
    Wraps the raw image prompt in a formatted Telegram message block.
    """
    prompt_text = generate_image_prompt(
        headline=headline,
        category=category,
        article_text=article_text,
        final_score=final_score,
        emotion_score=emotion_score,
        short_link=short_link,
    )

    message = f"""
🎨 AI IMAGE PROMPT — Article #{article_idx}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 Target Size : 1080 × 1350 px (Portrait / Instagram Post)
📂 Category    : {category}
📰 Headline    : {headline}
📈 Viral Score : {final_score}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🖼️ PASTE THIS PROMPT IN MIDJOURNEY / DALL-E 3 / IDEOGRAM:

{prompt_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 TIPS:
• Midjourney   → paste prompt + add  --ar 4:5 --v 6 --style raw
• DALL-E 3     → paste as-is in ChatGPT image mode
• Ideogram     → paste prompt, select "Realistic" style, 1080×1350
• Canva AI     → use as text-to-image prompt, set canvas 1080×1350

🔗 News Link: {short_link}
📸 {PERSONAL_BRAND["instagram"]}
""".strip()

    return message

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
    "gauri bazar", "bhaluani", "kaptanganj", "siswa", "nichlaul", "mithauna",
    "yogi", "adityanath", "yogi adityanath", "gorakhnath", "peeth", "mahant", "योगी", "आदित्यनाथ",
    "गोरखनाथ", "पीठ", "महंत", "up police", "gkp", "stf", "upstf", "dial 112", "daroga", "thana",
    "kotwali", "chowki", "sp office", "dm office", "vda", "gda", "nagarpalika", "pradhan", "bdo",
    "दारोगा", "थाना", "कोतवाली", "चौकी", "प्रधान", "बीडीओ",
    "bhojpuri", "bhojpuriya", "purvanchali", "ghaghra", "gandak", "rapti", "rohin", "ami", "kuwano",
    "sarayu", "boating accident", "floods", "purvanchal expressway", "gorakhpur link", "aiims gorakhpur",
    "brd medical college", "mmmut", "ddu", "kushinagar international airport", "buddha relics",
    "भोजपुरी", "पूर्वांचली", "घाघरा", "गंडक", "राप्ती", "सरयू", "बाढ़", "एक्सप्रेसवे", "एम्स", "बीआरडी",
]

HIGH_PRIORITY_KEYWORDS = [
    "modi", "narendra modi", "bjp", "amit shah", "jp nadda", "rajnath singh", "nirmala sitharaman",
    "s jaishankar", "nitin gadkari", "yogi", "pmo", "pmo india", "nda", "rss", "mohan bhagwat",
    "lok sabha", "rajya sabha", "parliament", "sansad", "cabinet decision", "central govt",
    "मोदी", "नरेन्द्र मोदी", "भाजपा", "अमित शाह", "नड्डा", "राजनाथ", "जयशंकर", "गडकरी", "पीएमओ",
    "संसद", "लोकसभा", "राज्यसभा", "मंत्रिमंडल", "कैबिनेट", "केंद्र सरकार", "आरएसएस", "मोहन भागवत",
    "rahul gandhi", "priyanka gandhi", "sonia gandhi", "congress", "inc", "kharge", "akhilesh",
    "yadav", "dimple yadav", "kejriwal", "aap", "mamata banerjee", "tmc", "stalin", "nitish kumar",
    "jdu", "rjd", "tejashwi", "lalu yadav", "mayawati", "bsp", "india alliance",
    "राहुल गांधी", "प्रियंका", "कांग्रेस", "खड़गे", "अखिलेश यादव", "केजरीवाल", "ममता",
    "election", "chunav", "voting", "matdan", "evm", "vvpat",
    "police", "arrest", "court", "supreme court", "high court", "ed", "cbi", "raid", "encounter",
    "jail", "custody", "fir", "charge sheet", "warrant",
    "पुलिस", "गिरफ्तार", "कोर्ट", "सुप्रीम कोर्ट", "हाई कोर्ट", "ईडी", "सीबीआई", "छापा", "रेड",
    "pakistan", "china", "army", "attack", "war", "israel", "gaza", "border", "loc", "lac",
    "iaf", "navy", "surgical strike", "drone", "missile", "agniveer",
    "चीन", "सेना", "हमला", "युद्ध", "इजरायल", "गाजा", "बॉर्डर", "राफेल", "ड्रोन", "मिसाइल",
    "trump", "putin", "russia", "america", "united nations", "nato",
    "gst", "income tax slab", "budget", "rbi", "repo rate", "inflation", "gdp", "share market",
    "nifty", "sensex", "neet", "paper leak", "scam", "railway", "exam cancel",
    "नीट", "पेपर लीक", "घोटाला", "रेलवे", "परीक्षा रद्द",
]

EMOTIONAL_WORDS = [
    "shock", "fear", "warning", "danger", "viral", "anger", "violence", "protest", "alert",
    "breaking", "urgent", "panic", "expose", "mystery", "miracle", "tragedy", "disaster",
    "catastrophe", "scandal", "shameful", "horrific", "brutal", "deadly", "conspiracy",
    "सावधान", "बड़ी खबर", "खुलासा", "हादसा", "मौत", "हड़कंप", "सनसनी", "खौफ", "शर्मनाक",
    "धोखा", "तबाही", "खतरा", "चेतावनी", "सच्चाई", "मच गया बवाल", "दहशत", "भयंकर", "दर्दनाक",
    "blast", "explosion", "fire", "derailment", "train crash", "collision", "drowning",
    "earthquake", "cloudburst", "landslide", "heatwave", "building collapse",
    "धमाका", "विस्फोट", "आगजनी", "ट्रेन हादसा", "टक्कर", "डूबने से मौत", "भूकंप", "बाढ़",
    "trending", "viral reel", "boycott", "trolled", "viral audio", "leaked video",
    "ट्रेंडिंग", "वायरल रील", "बॉयकाट", "वायरल ऑडियो", "लीक वीडियो",
]

# =========================================================
# MAIN SCORING + AI LOOP
# =========================================================

results      = []
article_idx  = 0

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
Create a high-engagement, monetization-focused Hindi social media article optimized for social networks. Sentence repetition is strictly prohibited.

Structure the output as follows:
Viral headline: Emotional, curiosity-based, Opening hook (2–3 lines) with Grabs attention.
Main article (300–400 words): Short paragraphs explaining the event details.
Key highlights: 3–5 important bullet points in other words than main article.
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

        article_idx += 1

        # =================================================
        # CHUNK 1 — FULL ARTICLE (text)
        # =================================================

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
📸 Instagram: {PERSONAL_BRAND["instagram"]}
▶️ YouTube: {PERSONAL_BRAND["youtube"]}
📘 Facebook: {PERSONAL_BRAND["facebook"]}
""".strip()

        send_telegram_message(telegram_message)
        print(f"✅ CHUNK 1 (ARTICLE) SENT — Article #{article_idx}")

        time.sleep(TELEGRAM_SLEEP_TIME)

        # =================================================
        # CHUNK 2 — AI IMAGE PROMPT
        # Generates a professional 1080×1350 px prompt
        # with news context + personal branding embedded
        # =================================================

        print(f"🎨 GENERATING IMAGE PROMPT for article #{article_idx}...")
        image_prompt_message = build_image_prompt_message(
            headline=headline,
            category=category,
            article_text=article_text,
            final_score=final_score,
            emotion_score=emotion_score,
            short_link=short_link,
            article_idx=article_idx,
        )

        send_telegram_message(image_prompt_message)
        print(f"✅ CHUNK 2 (IMAGE PROMPT) SENT — Article #{article_idx}")

        # =================================================
        # STORE RESULT
        # =================================================

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
            "ai_result":      ai_result,
        })

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
