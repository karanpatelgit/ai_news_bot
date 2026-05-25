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

    os.makedirs("downloads", exist_ok=True)

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=False
        )

        context = browser.new_context(
            accept_downloads=True
        )

        page = context.new_page()

        print("🚀 Opening Canva AI Generator...")

        page.goto("https://www.canva.com/ai-image-generator/")

        time.sleep(8)

        # =================================================
        # LOGIN
        # =================================================

        if "login" in page.url.lower() or page.locator('input[type="email"]').count() > 0:

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

            page.goto("https://www.canva.com/ai-image-generator/")

            time.sleep(6)

        # =================================================
        # ENTER AI PROMPT
        # =================================================

        print("✍️ Sending article to Canva AI...")

        textarea = page.locator("textarea").first

        textarea.wait_for(state="visible", timeout=15000)

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

        time.sleep(60)

        print("✅ AI poster generated")

        # =================================================
        # DOWNLOAD IMAGE — METHOD 1: keyboard shortcut
        # =================================================

        save_path = "downloads/news.png"
        downloaded = False

        try:

            print("📥 Downloading poster...")

            with page.expect_download(timeout=30000) as download_info:

                page.keyboard.press("Control+Shift+E")

                time.sleep(5)

                download_button = page.locator(
                    "button"
                ).filter(
                    has_text="Download"
                ).last

                download_button.click()

            download = download_info.value

            download.save_as(save_path)

            downloaded = True

            print(f"✅ IMAGE SAVED: {save_path}")

        except Exception as e:

            print(f"⚠️ Download method 1 failed: {e}")

        # =================================================
        # DOWNLOAD IMAGE — METHOD 2: direct button fallback
        # =================================================

        if not downloaded:

            try:

                print("📥 Trying alternate download method...")

                with page.expect_download(timeout=30000) as download_info:

                    page.locator(
                        "button"
                    ).filter(
                        has_text="Download"
                    ).last.click()

                download = download_info.value

                download.save_as(save_path)

                downloaded = True

                print(f"✅ IMAGE SAVED (method 2): {save_path}")

            except Exception as e:

                print(f"❌ Both download methods failed: {e}")

        browser.close()

        if downloaded:
            return save_path

        return None
