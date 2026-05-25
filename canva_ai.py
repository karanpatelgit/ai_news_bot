from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import os
import time

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

CANVA_EMAIL = os.getenv("CANVA_EMAIL")
CANVA_PASSWORD = os.getenv("CANVA_PASSWORD")

# =========================================================
# CANVA AI NEWS POSTER GENERATOR
# =========================================================

def generate_news_poster(article_text):

    # =====================================================
    # AI PROMPT CREATION
    # =====================================================

    prompt = f"""
Create an ultra realistic Indian breaking news Instagram poster.

NEWS ARTICLE:
{article_text}

INSTRUCTIONS:
- Make it look like a professional Indian news media graphic
- Highly informative visual
- Cinematic newsroom style
- Dramatic lighting
- Viral social media look
- Add realistic environment according to the news
- If crime -> police, sirens, tension
- If politics -> politicians, crowd, parliament style
- If flood/disaster -> realistic disaster scene
- If international -> geopolitical newsroom graphics
- Add Hindi news aesthetic
- Strong emotional storytelling
- Bold headline composition
- Include breaking news feeling
- Clean typography area
- Mobile friendly
- Professional composition
- Hyper realistic
- No watermark
- Modern news design
- Instagram optimized

STYLE:
- Red + black + yellow news palette
- High contrast
- Sharp details
- Studio quality
- Realistic humans
- Realistic Indian atmosphere
- News channel visual quality

SIZE:
1080x1350 portrait Instagram post
"""

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False
        )

        context = browser.new_context()

        page = context.new_page()

        print("🚀 Opening Canva AI Generator...")

        page.goto("https://www.canva.com/ai-image-generator/")

        time.sleep(8)

        # =================================================
        # LOGIN
        # =================================================

        if "login" in page.url.lower():

            print("🔐 Logging into Canva...")

            page.fill(
                'input[type="email"]',
                CANVA_EMAIL
            )

            page.click(
                'button[type="submit"]'
            )

            time.sleep(3)

            page.fill(
                'input[type="password"]',
                CANVA_PASSWORD
            )

            page.click(
                'button[type="submit"]'
            )

            time.sleep(10)

        # =================================================
        # ENTER AI PROMPT
        # =================================================

        print("✍️ Sending article to Canva AI...")

        textarea = page.locator("textarea").first

        textarea.click()

        textarea.fill(prompt)

        time.sleep(2)

        # =================================================
        # GENERATE IMAGE
        # =================================================

        print("🎨 Generating AI News Poster...")

        generate_button = page.locator(
            "button"
        ).filter(
            has_text="Generate"
        ).first

        generate_button.click()

        # =================================================
        # WAIT FOR GENERATION
        # =================================================

        print("⏳ Waiting for Canva AI generation...")

        time.sleep(40)

        print("✅ AI poster generated")

        # =================================================
        # OPEN GENERATED IMAGE
        # =================================================

        try:

            generated_image = page.locator("img").nth(0)

            generated_image.click()

            print("✅ Opened generated poster")

        except Exception as e:

            print("❌ Could not open image:", e)

        time.sleep(5)

        # =================================================
        # DOWNLOAD IMAGE
        # =================================================

        try:

            print("📥 Downloading poster...")

            page.keyboard.press("Control+Shift+E")

            time.sleep(5)

            download_button = page.locator(
                "text=Download"
            ).last

            download_button.click()

            print("✅ Download started")

        except Exception as e:

            print("❌ Download failed:", e)

        time.sleep(15)

        browser.close()

        print("🔥 NEWS POSTER COMPLETED")


# =========================================================
# EXAMPLE USAGE
# =========================================================

article = """
गोरखपुर में देर रात बड़ा सड़क हादसा हुआ।
तेज रफ्तार ट्रक और बस की टक्कर में कई लोग घायल हो गए।
घटना के बाद इलाके में अफरा-तफरी मच गई।
पुलिस और प्रशासन मौके पर पहुंच गया।
स्थानीय लोगों ने राहत कार्य शुरू किया।
"""

generate_news_poster(article)
