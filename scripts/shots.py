"""Capture screenshots of every page for the README."""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("reports/figures"); OUT.mkdir(parents=True, exist_ok=True)
PAGES = [("", "01_home"), ("Game_Explorer", "02_game_explorer"),
         ("Player_Lab", "03_player_lab"), ("Shot_Intelligence", "04_shot_intelligence"),
         ("Drive_Intelligence", "05_drive_intelligence"), ("Pick_and_Roll", "06_pick_and_roll"),
         ("Spatial_Analysis", "07_spatial_analysis"),
         ("Defensive_Intelligence", "08_defensive_intelligence"),
         ("Tracking_Lab", "09_tracking_lab"),
         ("Why_Did_This_Happen", "10_why_did_this_happen"),
         ("Scouting_Report", "11_scouting_report"),
         ("Season_Explorer", "12_season_explorer"),
         ("Data_Quality", "13_data_quality")]

with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                          args=["--no-sandbox", "--disable-dev-shm-usage"])
    pg = b.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1.5)
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    for slug, name in PAGES:
        url = f"http://localhost:8501/{slug}" if slug else "http://localhost:8501"
        pg.goto(url, wait_until="networkidle", timeout=120000)
        try:
            pg.wait_for_selector("[data-testid='stAppViewContainer']", timeout=30000)
        except Exception:
            pass
        time.sleep(7)
        pg.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
        # Streamlit virtualises its scroll container, so full_page capture is a
        # no-op; scroll and take a second frame instead.
        pg.evaluate("""() => {
            const el = document.querySelector('section.main, [data-testid="stMain"]')
                     || document.scrollingElement;
            el.scrollTo(0, 1250);
        }""")
        time.sleep(3)
        pg.screenshot(path=str(OUT / f"{name}_lower.png"), full_page=False)
        print("captured", name)
    b.close()
    if errors:
        print("PAGE ERRORS:", errors[:10])
