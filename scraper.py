from playwright.sync_api import sync_playwright
import os
import json
from datetime import datetime

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def capture_page(page, session_dir, step_num, label, suspected_pattern=None):
    """Screenshot + text + basic DOM metadata for one flow step."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"step{step_num:02d}_{label}_{timestamp}"

    screenshot_path = os.path.join(session_dir, f"{filename_base}.png")
    page.screenshot(path=screenshot_path, full_page=True)

    text_content = page.inner_text("body")
    text_path = os.path.join(session_dir, f"{filename_base}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    # Grab clickable elements + their bounding boxes — useful later for
    # "tiny button" / "hidden link" visual pattern detection
    elements = page.eval_on_selector_all(
        "button, a, input[type=submit]",
        """els => els.map(e => {
            const r = e.getBoundingClientRect();
            return {
                text: e.innerText?.trim().slice(0, 80),
                tag: e.tagName,
                x: r.x, y: r.y, width: r.width, height: r.height,
                visible: r.width > 0 && r.height > 0
            };
        })"""
    )

    metadata = {
        "step": step_num,
        "label": label,
        "url": page.url,
        "timestamp": timestamp,
        "suspected_pattern": suspected_pattern,  # e.g. "roach_motel", "confirmshaming"
        "clickable_elements": elements,
    }
    meta_path = os.path.join(session_dir, f"{filename_base}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Captured: {filename_base}")


def run_session(site_name, start_url):
    """One full flow capture for one site — becomes one 'flow sequence' sample."""
    session_dir = os.path.join(OUTPUT_DIR, site_name)
    os.makedirs(session_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(start_url)

        step = 1
        capture_page(page, session_dir, step, "landing"); step += 1
        input("Navigate to product page, then press Enter...")
        capture_page(page, session_dir, step, "product"); step += 1
        input("Add to cart, then press Enter...")
        capture_page(page, session_dir, step, "cart"); step += 1
        input("Go to checkout, then press Enter...")
        capture_page(page, session_dir, step, "checkout", suspected_pattern="hidden_costs?"); step += 1
        input("Reach payment screen, then press Enter...")
        capture_page(page, session_dir, step, "payment"); step += 1
        input("Navigate to account/subscription settings, then press Enter...")
        capture_page(page, session_dir, step, "settings"); step += 1
        input("Find cancel-subscription flow — press Enter at EACH step you take...")
        capture_page(page, session_dir, step, "cancel_step", suspected_pattern="roach_motel?"); step += 1

        while True:
            cont = input("Another cancel-flow step? (y/n): ")
            if cont.lower() != "y":
                break
            capture_page(page, session_dir, step, "cancel_step", suspected_pattern="roach_motel?")
            step += 1

        browser.close()


if __name__ == "__main__":
    # Add real URLs — clean ones, no ad-tracking params
    sites = {
        #"amazon_in": "https://www.amazon.in/",
        #"flipkart": "https://www.flipkart.com/",
        #"byjus": "https://byjus.com/",
        #"myntra":"https://www.croma.com/"
        #"pepperfry": "https://www.pepperfry.com/",
        #"snapdeal": "https://www.snapdeal.com/",
        #"snapdeal": "https://www.snapdeal.com/",

        #"ajio": "https://www.ajio.com/",
        #"nykaa": "https://www.nykaa.com/",
        #"urbanladder": "https://www.urbanladder.com/",
        #'JioMart': "https://www.jiomart.com/",
        #'Blinkit' : "https://blinkit.com/",
        #'Zepto' : "https://www.zepto.com/",
        #'PharmEasy' : "https://pharmeasy.in/",
        #'Lenskart' : "https://www.lenskart.com/",
        #'Moglix': "https://www.moglix.com/",
        #'shopclues':'https://www.shopclues.com/',
        #"Purplle": "https://www.purplle.com/",
        #"Zivame": "https://www.zivame.com/",
        #"Clovia": "https://www.clovia.com/",
        #"LimeRoad": "https://www.limeroad.com/",
        #"Bewakoof": "https://www.bewakoof.com/",
        #"Tata 1mg": "https://www.1mg.com/",
        #"Netmeds": "https://www.netmeds.com/",
        #"HealthKart": "https://www.healthkart.com/",
        #"Techjockey": "https://www.techjockey.com/",
        "The Souled Store": "https://www.thesouledstore.com/",

    }
    for name, url in sites.items():
        run_session(name, url)