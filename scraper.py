from playwright.sync_api import sync_playwright
import os
from datetime import datetime

# Where captures get saved
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def capture_page(page, label):
    """Screenshot + extract visible text for the current page state."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{label}_{timestamp}"

    # Screenshot
    screenshot_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    page.screenshot(path=screenshot_path, full_page=True)

    # Extract visible text
    text_content = page.inner_text("body")
    text_path = os.path.join(OUTPUT_DIR, f"{filename_base}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    print(f"Captured: {filename_base}")

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        page.goto("https://www.amazon.in/?&tag=googhydrabk1-21&ref=pd_sl_9ie4f5jnmt_e&adgrpid=155259815193&hvpone=&hvptwo=&hvadid=815461296170&hvpos=&hvnetw=g&hvrand=17035789890440978007&hvqmt=e&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=20453&hvtargid=kwd-19029053386&hydadcr=5621_2502646&mcid=5e8480ad24973e2487148e05e9f70ad5&hvocijid=17035789890440978007--&hvexpln=nav&gad_source=1")  # replace this
        capture_page(page, "landing")

        input("Navigate to product page, then press Enter...")
        capture_page(page, "product")

        input("Add to cart, then press Enter...")
        capture_page(page, "cart")

        input("Go to checkout, then press Enter...")
        capture_page(page, "checkout")

        input("Reach payment screen, then press Enter...")
        capture_page(page, "payment")

        input("Now navigate to account/subscription settings, then press Enter...")
        capture_page(page, "settings")

        input("Try to find the cancel-subscription flow, press Enter at each step...")
        capture_page(page, "cancel_step1")

        input("Continue cancel flow, press Enter...")
        capture_page(page, "cancel_step2")

        browser.close()

if __name__ == "__main__":
    run()