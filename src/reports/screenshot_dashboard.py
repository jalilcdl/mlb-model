"""
Capture a real screenshot of the running dashboard's Game Detail tab.

Drives the actual Streamlit app in Chrome via Playwright -- not a mockup. Uses
the machine's installed Chrome (channel="chrome") so there is no separate
~120MB browser download.

Streamlit needs real waiting rather than a fixed sleep: it streams the page
over a websocket and renders progressively, and tab panels are lazy -- the
Game Detail content does not exist in the DOM until its tab is clicked. So this
clicks the tab, then waits on concrete selectors (the team logo, the diagnostics
heading) before capturing.

    python -m src.reports.screenshot_dashboard
    python -m src.reports.screenshot_dashboard --url http://localhost:8501 --out shot.png
"""
import argparse
import sys
from pathlib import Path

from src import config

DEFAULT_URL = "http://localhost:8501"
DEFAULT_OUT = config.PROCESSED_DIR / "game_detail_screenshot.png"


def _select_game(page, game, index, timeout_ms):
    """Pick a specific matchup from the Game Detail selectbox.

    Streamlit's selectbox renders its options in a detached popover, so the
    option elements do not exist until the control is opened. After choosing,
    Streamlit reruns the script, so we wait for the header to actually show the
    requested teams rather than screenshotting the previous game's page.
    """
    page.click('[data-testid="stSelectbox"]')
    page.wait_for_selector('[role="option"]', timeout=timeout_ms)

    # The option list is VIRTUALIZED (only ~11 of the day's games are in the DOM
    # at once), so a matchup lower down the slate cannot simply be clicked, and
    # the widget's type-to-filter does not receive synthetic keystrokes here.
    # Keyboard navigation is deterministic: the highlight starts on the current
    # selection (index 0 on first open), so ArrowDown exactly `index` times lands
    # on the wanted game regardless of what is rendered.
    for _ in range(index):
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(40)
    page.keyboard.press("Enter")
    away, home = [t.strip() for t in game.split("@")]
    # Header renders the codes as separate markdown headings.
    page.wait_for_selector(f'h3:has-text("{away}")', timeout=timeout_ms)
    page.wait_for_selector(f'h3:has-text("{home}")', timeout=timeout_ms)
    page.wait_for_timeout(1200)


def capture(url=DEFAULT_URL, out_path=DEFAULT_OUT, tab="Game Detail", game=None, game_index=0,
            width=1600, height=1200, timeout_ms=120_000):
    from playwright.sync_api import sync_playwright

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=2)  # retina-ish, readable when shared
        page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

        # Wait for Streamlit itself to mount, then for the first run to finish.
        page.wait_for_selector('[data-testid="stAppViewContainer"], .stApp', timeout=timeout_ms)
        page.wait_for_selector(f'[role="tab"]:has-text("{tab}")', timeout=timeout_ms)

        page.click(f'[role="tab"]:has-text("{tab}")')

        # Game Detail specifics: the logo image and the diagnostics section only
        # exist once the panel has actually rendered.
        page.wait_for_selector('img[src*="espncdn"]', timeout=timeout_ms)
        if game:
            _select_game(page, game, game_index, timeout_ms)
        page.wait_for_selector('text=Model diagnostics', timeout=timeout_ms)
        page.wait_for_selector('text=Team comparison', timeout=timeout_ms)

        # Let late elements (plotly chart, images decoding) settle.
        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass  # networkidle rarely settles on a websocket app; not fatal
        page.wait_for_timeout(1500)

        # full_page=True is NOT enough here. Streamlit scrolls inside
        # section[data-testid="stMain"], so the <html> element stays exactly one
        # viewport tall and a "full page" capture silently crops everything below
        # the fold (the team comparison and diagnostics panels). Measure the real
        # inner scroll height and grow the viewport to fit, then re-measure once
        # because the taller viewport reflows the layout.
        for _ in range(3):
            content_h = page.evaluate(
                """() => {
                    const m = document.querySelector('section[data-testid="stMain"]');
                    return m ? m.scrollHeight : document.documentElement.scrollHeight;
                }"""
            )
            target = int(content_h) + 120  # padding so nothing sits flush at the edge
            if abs(target - height) < 40:
                break
            height = target
            page.set_viewport_size({"width": width, "height": height})
            page.wait_for_timeout(900)

        page.screenshot(path=str(out_path), full_page=True)

        info = page.evaluate("""() => {
            const imgs = [...document.querySelectorAll('img')].filter(i => (i.src||'').includes('espncdn'));
            const sel = document.querySelector('[data-testid="stSelectbox"]');
            return {
                selected: sel ? (sel.innerText||'').split('\\n').pop().trim() : null,
                logos: imgs.map(i => ({file: i.src.split('/').pop(), ok: i.complete && i.naturalWidth > 0})),
                hasRecommendation: document.body.innerText.includes('Model makes'),
                hasComparison: document.body.innerText.includes('Team comparison'),
                hasDiagnostics: document.body.innerText.includes('Model diagnostics'),
                hasStars: document.body.innerText.includes('\\u2605'),
            };
        }""")
        browser.close()
    return out_path, info


def main():
    ap = argparse.ArgumentParser(description="Screenshot the dashboard's Game Detail tab.")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tab", default="Game Detail")
    args = ap.parse_args()
    try:
        path, info = capture(args.url, args.out, tab=args.tab)
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print(f"Is the dashboard running at {args.url}? Start it with run_dashboard.bat")
        sys.exit(1)
    print(f"saved: {path}")
    print(f"size:  {path.stat().st_size:,} bytes")
    print("content verified in-page:")
    for k in ("hasRecommendation", "hasComparison", "hasDiagnostics", "hasStars"):
        print(f"  {k}: {info[k]}")
    for lg in info["logos"]:
        print(f"  logo {lg['file']}: loaded={lg['ok']}")


if __name__ == "__main__":
    main()
