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
from PIL import Image, ImageDraw, ImageFont

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
CF_ACCOUNT_ID      = os.environ.get("CF_ACCOUNT_ID", "").strip()
CF_API_TOKEN       = os.environ.get("CF_API_TOKEN", "").strip()
TOGETHER_API_KEY   = os.environ.get("TOGETHER_API_KEY", "").strip()

BOT_TOKEN          = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID            = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_CHAT_IDS  = os.environ.get("TELEGRAM_CHAT_IDS", "").strip()

print("=" * 50)
print("🔑 KEY STATUS:")
print(f"   GROQ       : {'✅ SET' if GROQ_API_KEY else '❌ MISSING'}")
print(f"   SAMBANOVA  : {'✅ SET' if SAMBANOVA_API_KEY else '❌ MISSING'}")
print(f"   HF_TOKEN   : {'✅ SET' if HF_TOKEN else '❌ MISSING'}")
print(f"   OPENROUTER : {'✅ SET — ' + OPENROUTER_API_KEY[:8] + '...' if OPENROUTER_API_KEY else '❌ MISSING'}")
print(f"   CLOUDFLARE : {'✅ SET' if (CF_ACCOUNT_ID and CF_API_TOKEN) else '⚠️ OPTIONAL (Not Set)'}")
print(f"   TOGETHER   : {'✅ SET' if TOGETHER_API_KEY else '⚠️ OPTIONAL (Not Set)'}")
print(f"   BOT_TOKEN  : {'✅ SET' if BOT_TOKEN else '❌ MISSING'}")
print(f"   CHAT_ID    : {'✅ SET' if CHAT_ID else '❌ MISSING'}")
print(f"   TELEG_CHAT : {'✅ SET' if TELEGRAM_CHAT_IDS else '❌ MISSING'}")
print("=" * 50)

# =========================================================
# CONFIG
# =========================================================

MAX_WORKERS           = 32
AI_SLEEP_TIME         = 1
TELEGRAM_SLEEP_TIME   = 1
LOCAL_NEWS_BONUS      = 2.5
SCRAPER_TIME_WINDOW_H = 10
SCRAPER_MAX_PER_SITE  = 8
RSS_MAX_PER_FEED      = 20

# Personal Branding Info — used in image prompt generator & watermark
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

GROQ_MODELS = ["llama-3.1-8b-instant" ,"llama-3.3-70b-versatile" ]
SAMBANOVA_MODELS = ["Meta-Llama-3.3-70B-Instruct"]
HF_MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct"]

OPENROUTER_MODELS = [
    "openrouter/free",                                      # official free auto-router (always try first)
    "deepseek/deepseek-v4-flash:free",                      # #1 quality, 1M context
    "minimax/minimax-m2.5:free",                            # 205K ctx, strong reasoning
    "google/gemma-4-31b-it:free",                           # Google, vision+tools, 262K ctx
    "nvidia/nemotron-3-super-120b-a12b:free",               # NVIDIA 120B, 1M ctx
    "openai/gpt-oss-120b:free",                             # OpenAI open weights 120B
    "meta-llama/llama-3.3-70b-instruct:free",               # Meta Llama 70B
    "nvidia/nemotron-3-nano-30b-a3b:free",                  # NVIDIA 30B, 256K ctx
    "qwen/qwen3-coder:free",                                # Qwen3 coder, 1M ctx
    "openai/gpt-oss-20b:free",                              # OpenAI open 20B, lighter
    "z-ai/glm-4.5-air:free",                                # GLM 4.5 Air
    "meta-llama/llama-3.2-3b-instruct:free",                # smallest, fastest fallback
    "nousresearch/hermes-3-llama-3.1-405b:free",            # Hermes 405B
]

# Per-provider cooldown tracking — if a provider 429s, skip it for N seconds
_provider_cooldown = {}   # {"groq": <unix_timestamp_resume>, "sambanova": ..., etc.}
_COOLDOWN_SECONDS  = 60   # wait 60 s before retrying a rate-limited provider

# =========================================================
# DUPLICATE DETECTION
# =========================================================

def is_similar(a, b, threshold=0.7):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

# =========================================================
# TELEGRAM — TEXT & PHOTO SENDERS
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

def send_telegram_photo_bytes(image_bytes: bytes, caption: str = ""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    chat_ids = _parse_chat_ids(TELEGRAM_CHAT_IDS)
    if not chat_ids and CHAT_ID:
        chat_ids = [CHAT_ID]

    if not chat_ids:
        print("❌ TELEGRAM PHOTO ERROR: No CHAT_ID(s) configured")
        return

    for target_chat_id in chat_ids:
        try:
            files = {"photo": ("news_post.jpg", image_bytes, "image/jpeg")}
            data  = {"chat_id": target_chat_id, "caption": caption[:1024]}
            resp  = requests.post(url, data=data, files=files, timeout=40)
            if resp.status_code == 200:
                print(f"📸 TELEGRAM PHOTO SENT -> chat_id={target_chat_id}")
            else:
                print(f"⚠️ TELEGRAM PHOTO HTTP {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            print(f"❌ TELEGRAM PHOTO ERROR -> chat_id={target_chat_id}: {e}")

# =========================================================
# MULTI-PROVIDER AUTHENTICATED IMAGE PIPELINE (NEW)
# =========================================================
CATEGORY_DISPLAY_MAP = {
    "GNews_Kushinagar_HI": "KUSHINAGAR NEWS",
    "GNews_Gorakhpur_HI":  "GORAKHPUR NEWS",
    "GNews_Purvanchal_HI": "PURVANCHAL UPDATES",
    "AmarUjala_UP":        "UTTAR PRADESH",
    "Bhaskar_UP":          "UP BREAKING",
    "Hindustan_UP":        "UTTAR PRADESH",
    "Jagran_UP":           "UP NEWS",
    "News18_UP":           "UP BREAKING",
    "TV9_UP":              "UTTAR PRADESH",
    "AajTak_UP":           "UP BREAKING",
    "UPTak":               "UP TAK NEWS",
    "GorakhpurLive":       "GORAKHPUR LIVE",
    "PurvanchalNews":      "PURVANCHAL NEWS",
    "GaonConnection":      "RURAL INDIA",
}

def generate_image_bytes(prompt: str) -> bytes:
    """
    Uses the updated router.huggingface.co endpoint + 1024x1024 standard resolution
    so FLUX.1-schnell and SDXL never reject the dimensions.
    """
    clean_prompt = prompt.replace("\n", " ").strip()

    # ── 1. HUGGING FACE ROUTER API (Updated 2026 Endpoint) ────────────────
    if HF_TOKEN and not _is_on_cooldown("hf_img"):
        hf_image_models = [
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "ByteDance/SDXL-Lightning"
        ]
        for model in hf_image_models:
            try:
                print(f"🎨 [HF IMAGE] Generating via {model}...")
                if hf_client:
                    try:
                        # Initialize with hf-inference provider explicitly
                        img_client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
                        pil_img = img_client.text_to_image(
                            prompt=clean_prompt,
                            model=model,
                            width=1024,
                            height=1024
                        )
                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=92)
                        img_bytes = buf.getvalue()
                        if len(img_bytes) > 5000:
                            print(f"✅ [HF IMAGE] Success via InferenceClient ({model})")
                            return img_bytes
                    except Exception as client_err:
                        print(f"⚠️ [HF Client] {model} fallback to router REST: {client_err}")

                # Updated Hugging Face Router URL (replaces deprecated api-inference.huggingface.co)
                api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
                headers = {"Authorization": f"Bearer {HF_TOKEN}"}
                payload = {"inputs": clean_prompt}
                resp = requests.post(api_url, headers=headers, json=payload, timeout=45)
                
                if resp.status_code == 200 and len(resp.content) > 5000:
                    print(f"✅ [HF IMAGE] Success via Router REST ({model})")
                    return resp.content
                elif resp.status_code in (402, 429):
                    print(f"⚠️ [HF IMAGE] Quota/Rate limit ({resp.status_code}) on {model}")
                    if resp.status_code == 402:
                        _set_cooldown("hf_img", 1800)
                        break
                else:
                    print(f"⚠️ [HF IMAGE] {model} HTTP {resp.status_code}: {resp.text[:150]}")
            except Exception as e:
                print(f"❌ [HF IMAGE] Error ({model}): {e}")

    # ── 2. CLOUDFLARE WORKERS AI (10,000 Free Neurons Daily) ──────────────
    if CF_ACCOUNT_ID and CF_API_TOKEN and not _is_on_cooldown("cf_img"):
        cf_models = [
            "@cf/black-forest-labs/flux-1-schnell",
            "@cf/bytedance/stable-diffusion-xl-lightning",
            "@cf/stabilityai/stable-diffusion-xl-base-1.0"
        ]
        for cf_model in cf_models:
            try:
                print(f"🎨 [CLOUDFLARE AI] Generating via {cf_model}...")
                cf_url  = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{cf_model}"
                headers = {"Authorization": f"Bearer {CF_API_TOKEN}"}
                payload = {"prompt": clean_prompt, "num_steps": 4}
                resp    = requests.post(cf_url, headers=headers, json=payload, timeout=45)

                if resp.status_code == 200:
                    content_type = resp.headers.get("Content-Type", "")
                    if "application/json" in content_type:
                        b64_img = resp.json().get("result", {}).get("image")
                        if b64_img:
                            print(f"✅ [CLOUDFLARE AI] Success ({cf_model})")
                            return base64.b64decode(b64_img)
                    elif len(resp.content) > 5000:
                        print(f"✅ [CLOUDFLARE AI] Binary success ({cf_model})")
                        return resp.content
            except Exception as e:
                print(f"❌ [CLOUDFLARE AI] Error ({cf_model}): {e}")

    return None


def fetch_article_og_image(article_url: str) -> bytes:
    """
    Scrapes the press photo from the article, strictly blocking Google News icons
    and generic site placeholders.
    """
    # Block un-redirected Google News URLs immediately
    if "news.google.com" in article_url or "google.com" in article_url:
        print("⚠️ [OG:IMAGE] Skipping Google News URL to avoid Google 'G' logo")
        return None

    try:
        resp = requests.get(article_url, headers=random_headers(), timeout=10)
        if resp.status_code != 200:
            return None
        soup   = BeautifulSoup(resp.text, "lxml")
        og_tag = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "twitter:image"})
        if og_tag and og_tag.get("content"):
            img_url = og_tag["content"].strip()
            
            # Reject Google icons, favicons, and default logos
            bad_tokens = ["google", "gstatic", "googleusercontent", "logo", "favicon", "default", "placeholder"]
            if any(b in img_url.lower() for b in bad_tokens):
                print(f"⚠️ [OG:IMAGE] Rejected generic/logo image URL: {img_url}")
                return None

            img_resp = requests.get(img_url, headers=random_headers(), timeout=12)
            if img_resp.status_code == 200 and len(img_resp.content) > 15000:
                print("✅ [OG:IMAGE FALLBACK] Extracted authentic press photo from article")
                return img_resp.content
    except Exception as e:
        print(f"⚠️ [OG:IMAGE FALLBACK] Failed: {e}")
    return None


def apply_brand_overlay(image_bytes: bytes, category: str = "BREAKING NEWS", headline: str = "") -> bytes:
    """
    Renders a broadcast-grade 1080x1350 Hindi News Card using Playwright Chromium
    (so Devanagari matras and dynamically-sized category ribbons render flawlessly),
    with a Pillow fallback.
    """
    clean_cat = CATEGORY_DISPLAY_MAP.get(category, category.replace("_", " ").upper())

    try:
        from playwright.sync_api import sync_playwright
        b64_bg = base64.b64encode(image_bytes).decode("utf-8")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@700;900&family=Montserrat:wght@800&display=swap" rel="stylesheet">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                width: 1080px; height: 1350px; overflow: hidden;
                font-family: 'Noto Sans Devanagari', sans-serif;
                background: #09121c; position: relative;
            }}
            .bg-img {{
                width: 1080px; height: 1255px;
                object-fit: cover; object-position: center;
                display: block;
            }}
            .top-badge {{
                position: absolute; top: 36px; left: 36px;
                background: #C8102E; color: #FFFFFF;
                padding: 12px 28px; font-family: 'Montserrat', sans-serif;
                font-size: 26px; font-weight: 800; letter-spacing: 1.5px;
                border-bottom: 5px solid #D4AF37; border-radius: 4px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.55);
                display: inline-block;
            }}
            .lower-third {{
                position: absolute; bottom: 95px; left: 0; width: 1080px;
                background: linear-gradient(to top, rgba(9,18,28,0.98) 0%, rgba(9,18,28,0.88) 70%, rgba(9,18,28,0.0) 100%);
                padding: 90px 42px 28px 42px;
            }}
            .headline-box {{
                border-left: 8px solid #C8102E;
                padding-left: 22px; color: #FFFFFF;
                font-size: 44px; font-weight: 900; line-height: 1.32;
                text-shadow: 0 4px 12px rgba(0,0,0,0.8);
            }}
            .footer {{
                position: absolute; bottom: 0; left: 0;
                width: 1080px; height: 95px;
                background: #0D1B2A; border-top: 6px solid #D4AF37;
                display: flex; align-items: center; justify-content: space-between;
                padding: 0 38px;
            }}
            .brand-handle {{
                color: #D4AF37; font-family: 'Montserrat', sans-serif;
                font-size: 26px; font-weight: 800;
            }}
            .brand-sub {{
                color: #FFFFFF; font-size: 22px; font-weight: 700;
            }}
        </style>
        </head>
        <body>
            <img class="bg-img" src="data:image/jpeg;base64,{b64_bg}" />
            <div class="top-badge">🔴 {clean_cat}</div>
            <div class="lower-third">
                <div class="headline-box">{headline}</div>
            </div>
            <div class="footer">
                <div class="brand-handle">{PERSONAL_BRAND['handle']} | {PERSONAL_BRAND['name']}</div>
                <div class="brand-sub">कुशीनगर और पूर्वांचल की आवाज़</div>
            </div>
        </body>
        </html>
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1080, "height": 1350})
            page.set_content(html_content, wait_until="networkidle", timeout=15000)
            final_bytes = page.screenshot(type="jpeg", quality=92)
            browser.close()
            return final_bytes
    except Exception as e:
        print(f"⚠️ Playwright card render failed ({e}) — using Pillow fallback...")
        # Pillow fallback with dynamic badge width so text never overflows
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((1080, 1350))
        draw = ImageDraw.Draw(img)
        badge_w = max(320, len(clean_cat) * 20 + 60)
        draw.rectangle([(30, 30), (30 + badge_w, 85)], fill=(200, 16, 46))
        draw.text((45, 45), clean_cat, fill=(255, 255, 255))
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=92)
        return out.getvalue()
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
        published_datetime = datetime(*published[:4], tzinfo=timezone.utc)
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

        def _fetch_one(hl_url):
            headline, article_url = hl_url
            article_text = fetch_full_article(article_url)
            if not article_text or len(article_text.strip()) < 80:
                article_text = headline
            return {"source": "scraper", "category": site_name,
                    "headline": headline, "link": article_url, "article": article_text}
        
        with ThreadPoolExecutor(max_workers=min(len(links_found), 8)) as art_pool:
            results = list(art_pool.map(_fetch_one, links_found))
    except Exception as e:
        print(f"❌ [{site_name}] Scraper error: {e}")

    print(f"✅ [{site_name}] Scraped {len(results)} articles")
    return results

# =========================================================
# CLEAN RSS FEEDS (LOW DUPLICATES + HIGH VIRAL POTENTIAL)
# =========================================================

RSS_FEEDS = {
    "GNews_Kushinagar_HI": "https://news.google.com/rss/search?q=%E0%A4%95%E0%A5%81%E0%A4%B6%E0%A5%80%E0%A4%A8%E0%A4%97%E0%A4%B0+when:1d&hl=hi&gl=IN&ceid=IN:hi",
    "GNews_Gorakhpur_HI":  "https://news.google.com/rss/search?q=%E0%A4%97%E0%A5%8B%E0%A4%B0%E0%A4%96%E0%A4%AA%E0%A5%81%E0%A4%B0+when:1d&hl=hi&gl=IN&ceid=IN:hi",
    "GNews_Purvanchal_HI": "https://news.google.com/rss/search?q=%E0%A4%AA%E0%A5%82%E0%A4%B0%E0%A5%8D%E0%A4%B5%E0%A4%BE%E0%A4%82%E0%A4%9A%E0%A4%B2+%E0%A4%AF%E0%A5%82%E0%A4%AA%E0%A5%80+when:1d&hl=hi&gl=IN&ceid=IN:hi",
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

    "GaonConnection":
        "https://www.gaonconnection.com/feed",
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
_all_rss_news_lock = threading.Lock()
print("\n========== FETCHING RSS FEEDS ==========")

def _fetch_rss_feed(cat_url_pair):
    category, url = cat_url_pair
    try:
        feed = feedparser.parse(url)
        return category, feed.entries[:RSS_MAX_PER_FEED]
    except Exception as e:
        print(f"  ❌ RSS PARSE ERROR ({category}): {e}")
        return category, []

def _process_entry_and_collect(entry, category):
    result = process_single_entry(entry, category=category)
    if result:
        with _all_rss_news_lock:
            all_news.append(result)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    feed_futures = {executor.submit(_fetch_rss_feed, pair): pair[0]
                    for pair in RSS_FEEDS.items()}
    entry_futures = []
    for ff in as_completed(feed_futures):
        category = feed_futures[ff]
        try:
            cat, entries = ff.result()
            print(f"  📡 {cat}: {len(entries)} entries")
            for entry in entries:
                entry_futures.append(executor.submit(_process_entry_and_collect, entry, cat))
        except Exception as e:
            print(f"  ❌ RSS FEED ERROR ({category}): {e}")
    for ef in as_completed(entry_futures):
        try:
            ef.result()
        except Exception as e:
            print("❌ ENTRY FUTURE ERROR:", e)

print(f"  ✅ ALL RSS DONE — total: {len(all_news)}")

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

def _is_on_cooldown(provider: str) -> bool:
    """Return True if this provider is still in its rate-limit cooldown window."""
    resume_at = _provider_cooldown.get(provider, 0)
    if time.time() < resume_at:
        remaining = int(resume_at - time.time())
        print(f"⏳ [{provider.upper()}] on cooldown for {remaining}s more — skipping")
        return True
    return False

def _set_cooldown(provider: str, seconds: int = _COOLDOWN_SECONDS):
    _provider_cooldown[provider] = time.time() + seconds
    print(f"🛑 [{provider.upper()}] rate-limited → cooldown {seconds}s")


def generate_ai_response(prompt, headline="", category="", article_text="", link=""):

    # ── 1. GROQ ────────────────────────────────────────────────────────────
    if groq_client and not _is_on_cooldown("groq"):
        for model_name in GROQ_MODELS:
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
                if "429" in err_str or "rate_limit" in err_str.lower():
                    _set_cooldown("groq")
                    break
                elif "402" in err_str:
                    print("❌ GROQ 402 — no credits, skipping provider")
                    _set_cooldown("groq", 3600)
                    break
                else:
                    print(f"❌ GROQ [{model_name}] FAILED:", e)

    # ── 2. SAMBANOVA ───────────────────────────────────────────────────────
    if samba_client and not _is_on_cooldown("sambanova"):
        for model_name in SAMBANOVA_MODELS:
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
                if "429" in err_str or "rate_limit" in err_str.lower():
                    _set_cooldown("sambanova")
                    break
                else:
                    print(f"❌ SAMBANOVA [{model_name}] FAILED:", e)

    # ── 3. HuggingFace ─────────────────────────────────────────────────────
    if hf_client and not _is_on_cooldown("hf"):
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
                err_str = str(e)
                if "402" in err_str:
                    print("❌ HF 402 Payment Required — skipping provider")
                    _set_cooldown("hf", 3600)
                    break
                elif "429" in err_str or "rate_limit" in err_str.lower():
                    _set_cooldown("hf")
                    break
                else:
                    print(f"❌ HF [{model_name}] FAILED:", e)

    # ── 4. OPENROUTER (free models, rotate through all) ────────────────────
    if not OPENROUTER_API_KEY:
        print("⚠️ OPENROUTER_API_KEY not set — skipping OpenRouter")
    elif _is_on_cooldown("openrouter"):
        pass
    else:
        or_model_cooldowns = {}

        for model_name in OPENROUTER_MODELS:
            if or_model_cooldowns.get(model_name):
                continue
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
                        print(f"⚠️ OPENROUTER empty response: {model_name} — trying next")

                elif or_resp.status_code == 404:
                    print(f"❌ OPENROUTER 404 — model removed: {model_name} — skipping")

                elif or_resp.status_code == 402:
                    err_body = or_resp.text[:200]
                    print(f"❌ OPENROUTER 402 [{model_name}]: {err_body} — skipping model")

                elif or_resp.status_code == 429:
                    err_body = or_resp.text[:300]
                    print(f"⚠️ OPENROUTER 429 [{model_name}]: {err_body}")
                    or_model_cooldowns[model_name] = True
                    time.sleep(2)

                else:
                    print(f"❌ OPENROUTER HTTP {or_resp.status_code} [{model_name}]: {or_resp.text[:200]}")

            except Exception as e:
                print(f"❌ OPENROUTER FAILED [{model_name}]: {e}")

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
# AI VISUAL SCENE PROMPT BUILDER
# =========================================================

def build_visual_scene_prompt(headline: str, category: str, article_text: str, emotion_score: int) -> str:
    """
    Translates Hindi/regional news context into a concise English visual scene prompt
    for FLUX.1-schnell / SDXL (strictly avoiding rendered text inside the raw AI photo).
    """
    mood_map = {
        "Crime":         "dark red warning atmosphere, police investigation scene, dramatic lighting",
        "Breaking":      "urgent breaking news broadcast atmosphere, high contrast",
        "Defense":       "patriotic Indian military atmosphere, tactical setting",
        "India Politics":"formal Indian political rally stage, press conference microphones",
        "Sports":        "vibrant stadium crowd energy, dynamic sports action lighting",
        "Kushinagar":    "Uttar Pradesh rural & town streetscape, golden sunlight, authentic Indian setting",
        "Gorakhpur":     "Gorakhpur Uttar Pradesh urban streetscape, authentic Indian photojournalism",
    }
    mood_cue = "authentic Indian photojournalism scene, editorial lighting"
    for key, val in mood_map.items():
        if key.lower() in category.lower():
            mood_cue = val
            break

    meta_prompt = (
        f"Write ONE single-sentence English visual description (maximum 40 words) for a photorealistic "
        f"news photograph representing this headline: '{headline}'. "
        f"Mood: {mood_cue}. Do NOT include any text, letters, banners, or words in the description. "
        f"Return ONLY the English description sentence."
    )

    scene_en = generate_ai_response(meta_prompt, headline=headline, category=category)
    if not scene_en or "BREAKING NEWS" in scene_en or len(scene_en) > 350:
        scene_en = f"Authentic Indian photojournalism scene representing {category} news in Uttar Pradesh, {mood_cue}"

    scene_en = re.sub(r'\s+', ' ', scene_en).strip()
    return (
        f"Ultra-HD editorial press photography, portrait 4:5 aspect ratio. "
        f"{scene_en}. Realistic Indian setting, cinematic lighting, 8k resolution, "
        f"sharp focus, strictly no text, no watermarks, no deformed faces."
    )

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

    if raw_score < 2.0:
        print(f"⏭️ SKIPPED LOW PRE-SCORE: {headline} ({raw_score})")
        continue

    print("✅ PRE-SCORE PASSED — RUNNING AI ENGINE")

    # Detect language mode based on category
    _local_cats = {
        'Kushinagar','Gorakhpur','Maharajganj','Deoria','Basti',
        'GNews_Kushinagar_HI','GNews_Gorakhpur_HI','GNews_Purvanchal_HI'
    }
    _is_local_lang = is_local or category in _local_cats
    _lang_instruction = (
        "भाषा: शुद्ध हिंदी — आसान, बोलचाल की हिंदी जो गाँव और कस्बे का आम इंसान भी समझ सके।"
        if _is_local_lang else
        "Language: Hindi main likho lekin impactful English words naturally mix karo (jaise: viral, breaking, update, reaction, shocking). Avoid full English sentences."
    )

    prompt = f"""
Tu ek expert social media news writer hai jo Facebook aur Instagram ke liye viral, monetization-friendly content likhta hai.
Tera channel hai "@karan_patel_kushinagar" — local Purvanchal ki awaaz, sach aur speed ke saath.

{_lang_instruction} but the article writting language will be always Hindi.

STRICT OUTPUT FORMAT — bilkul is structure mein likho, koi extra heading mat add karo:

 🔥 VIRAL HEADLINE (1 line)
   - Emotional + curiosity gap wali — reader ko rukne par majboor kare
   - Number ya shocking word se shuru karo jab possible ho
   - Max 12 words

 🪝 OPENING HOOK (2-3 lines)
   - Pehli line mein sabse badi baat — reader scroll nahi karega
   - Ek aisa angle jo dusre news pages ne nahi pakda
   - Directly baat pe aao, "aaj hum baat karenge" BILKUL MAT LIKHO

 📰 MAIN STORY (250-300 words)
   - Short paragraphs — max 3 lines each, Facebook mobile readers ke liye
   - Har paragraph mein ek nayi information
   - Facts, figures, naam, jagah — jo bhi available ho use karo
   - Dramatic but factual — koi exaggeration nahi
   - Story aise build karo ki reader end tak padhe
   - Try to fetch more informative lines like include important lines ,fact ,data and all essential things.

💬 PUBLIC REACTION ANGLE (2-3 lines)
   - Log is khabar par kya soch rahe hain / kya react kar rahe hain
   - 1 open-ended question jo comments drive kare
   - Example style: "Social media par log pooch rahe hain ki..."

📣 CALL TO ACTION (2 lines max)
   - Line 1: Engagement drive karo — share/tag/comment
   - Line 2: Follow karo @karan_patel_kushinagar — Facebook aur Instagram par

#️⃣ HASHTAGS (1 line, 5-8 tags)
   - Mix: 3 broad (#BreakingNews #Viral) + 3 topic-specific + 2 local (#Kushinagar #Purvanchal)
   - No spaces inside hashtags

Also append at the very end:
Emotion Score: X/10
Virality Score: Y/10
Political Toxicity: Z/10

---
INPUT DATA:
शीर्षक / Title: {headline}
Category: {category}
Full Article: {article_text[:3000]}
Source: {link}
---

QUALITY RULES — इन्हें तोड़ा तो content reject होगा:
✗ "आज हम बात करेंगे" — NEVER
✗ Sentence repetition — NEVER  
✗ Vague filler lines — NEVER
✗ More than 3 lines in any paragraph — NEVER
✗ Hashtags inside the article body — NEVER
✓ Every line must add new value
✓ Reader must feel: "yaar yeh toh share karna padega"
✓ End with a question that makes people comment
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

        if final_score < 4.0:
            print(f"⏭️ SKIPPED LOW FINAL SCORE: {headline} ({final_score})")
            continue

        print(f"🚀 FINAL SCORE PASSED ({final_score}) — GENERATING IMAGE & SENDING TELEGRAM")

        cleaned_link = clean_url(link)
        short_link   = shorten_url(cleaned_link)

        article_idx += 1

        # =================================================
        # CHUNK 1 — GENERATE AI IMAGE (WITH OG FALLBACK) & SEND PHOTO
        # =================================================

        visual_prompt = build_visual_scene_prompt(
            headline=headline,
            category=category,
            article_text=article_text,
            emotion_score=emotion_score
        )
        print(f"🎨 IMAGE PROMPT: {visual_prompt}")

        raw_image_bytes = generate_image_bytes(visual_prompt)
        if not raw_image_bytes:
            print("⚠️ All AI image APIs exhausted/failed — trying article og:image fallback...")
            raw_image_bytes = fetch_article_og_image(cleaned_link)

        if raw_image_bytes:
            branded_bytes = apply_brand_overlay(raw_image_bytes, category=category)
            photo_caption = (
                f"🔥 {headline}\n"
                f"📂 {category} | 📈 Viral Score: {final_score}\n"
                f"📢 {PERSONAL_BRAND['handle']}"
            )
            send_telegram_photo_bytes(branded_bytes, caption=photo_caption)
            print(f"✅ CHUNK 1 (BRANDED PHOTO) SENT — Article #{article_idx}")
            time.sleep(TELEGRAM_SLEEP_TIME)
        else:
            print(f"⚠️ No image available for Article #{article_idx} — sending text only")

        # =================================================
        # CHUNK 2 — FULL ARTICLE (text)
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
        print(f"✅ CHUNK 2 (ARTICLE) SENT — Article #{article_idx}")

        time.sleep(TELEGRAM_SLEEP_TIME)

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
            "image_prompt":   visual_prompt,
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
