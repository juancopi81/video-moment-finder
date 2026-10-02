#!/usr/bin/env python3
"""Optional browser validation using an installed Playwright/Chromium, without network content."""
import argparse
import csv
import io
import itertools
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
        if page.locator("#review-controls").count():
            total = page.locator(".flashcard").count()
            assert page.locator(".flashcard:visible").count() == 1
            assert page.locator("#previous-card").is_disabled()
            if total > 1:
                page.locator("#next-card").click()
                assert page.locator("#review-status").inner_text() == f"Card 2 of {total}"
                assert page.locator(".flashcard:visible details[open]").count() == 0
                page.locator("#previous-card").click()
            page.locator("#toggle-list").click()
            assert page.locator(".flashcard:visible").count() == total
            page.locator("#toggle-list").click()
            with page.expect_download(timeout=10000) as event:
                page.locator('a[download="vmf-flashcards.csv"]').click()
            download = event.value
            download.save_as(str(args.output_dir / "downloaded-flashcards.csv"))
            rows = list(csv.reader(io.StringIO((args.output_dir / "downloaded-flashcards.csv").read_text())))
            assert rows[0] == ["Front", "Back", "Tags"] and len(rows) == total + 1
            checks["flashcard_navigation_and_csv_download"] = total
        if page.locator("#lab-data").count():
            data = json.loads(page.locator("#lab-data").text_content())
            controls = data["controls"]
            visited = set()
            for choice in itertools.product(*(c["options"] for c in controls)):
                selection = {c["id"]: option["id"] for c, option in zip(controls, choice)}
                for key, value in selection.items():
                    page.locator(f'[data-control="{key}"]').select_option(value)
                match = next(s for s in data["states"] if s["when"] == selection)
                assert page.locator("#states [data-state]:visible").count() == 1
                assert page.locator("#states [data-state]:visible").get_attribute("data-state") == match["id"]
                visited.add(match["id"])
            page.locator("#pin").click()
            assert page.locator("#pinned [data-state]").get_attribute("data-state") == match["id"]
            page.locator("#learner-note").fill("Check the changed assumption before applying the source rule.")
            with page.expect_download(timeout=10000) as event:
                page.locator("#download").click()
            event.value.save_as(str(args.output_dir / "downloaded-comparison.txt"))
            assert "Check the changed assumption" in (args.output_dir / "downloaded-comparison.txt").read_text()
            page.locator("#reset").click()
            assert page.locator("#pinned [data-state]").get_attribute("data-state") == data["baseline"]
            assert page.locator("#states [data-state]:visible").get_attribute("data-state") == data["baseline"]
            checks["assumption_lab_cases_pin_reset_download"] = len(visited)
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
