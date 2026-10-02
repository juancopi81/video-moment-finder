#!/usr/bin/env python3
"""Optional browser validation using an installed Playwright/Chromium, without network content."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, chromium_sandbox=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        # The managed browser disallows file URLs. This is authorized generated
        # document content, supplied through the browser API without navigation.
        page.set_content(args.html.read_text(encoding="utf-8"), wait_until="load")
        page.screenshot(path=str(args.output_dir / "desktop-top.png"))
        checks = {"title": page.title(), "chromium_sandbox": True}
        if page.locator("#batch-size").count():
            assert page.locator("#pair-matrix span").count() == 36
            assert page.locator("#pair-matrix .positive").count() == 6
            page.locator("#batch-size").fill("6")
            assert page.locator("#pair-matrix span").count() == 144
            assert "120 different-image" in page.locator("#matrix-description").inner_text()
            page.locator("#batch-size").fill("3")
            checks["matrix_at_3_and_6_originals"] = True
        if page.locator("summary").count():
            page.locator("summary").first.focus()
            page.keyboard.press("Enter")
            assert page.locator("details").first.get_attribute("open") is not None
            checks["keyboard_hint_reveal"] = True
        images = page.locator("img")
        for index in range(images.count()):
            images.nth(index).scroll_into_view_if_needed()
            images.nth(index).evaluate("img => img.loading = 'eager'")
        page.wait_for_function("[...document.images].every(x => x.complete && x.naturalWidth > 0)")
        checks["embedded_images_loaded"] = images.count()
        page.screenshot(path=str(args.output_dir / "desktop-full.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("scrollTo(0, 0)")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(args.output_dir / "mobile-top.png"))
        page.screenshot(path=str(args.output_dir / "mobile-full.png"), full_page=True)
        checks["mobile_no_horizontal_overflow"] = True
        assert not errors, errors
        checks["page_errors"] = errors
        browser.close()
        (args.output_dir / "browser-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        print(json.dumps(checks))


if __name__ == "__main__":
    main()
