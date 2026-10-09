"""Boundary tests for portable archive safety and reproducibility."""

import copy
import json
import os
import shutil
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

from build_package import build_archive
from validate_package import PLUGIN_NAME, validate_archive


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        original = Path(__file__).resolve().parents[2] / "plugins" / PLUGIN_NAME
        self.manifest = json.loads((original / "plugin.json").read_text())
        self.write_manifest()
        shutil.copyfile(original / "mcp.json", self.source / "mcp.json")
        shutil.copytree(original / "assets", self.source / "assets")
        (self.source / "README.md").write_text("# Fixture package\n", encoding="utf-8")
        (self.source / "LICENSE").write_text("Fixture only\n", encoding="utf-8")
        for skill in ("study-guide", "flashcards", "tutor", "assumption-lab", "presentation", "setup"):
            target = self.source / "skills" / skill / "SKILL.md"
            target.parent.mkdir(parents=True)
            target.write_text(f"---\nname: {skill}\ndescription: Use for the fixture workflow.\n---\nFixture.\n", encoding="utf-8")
        self.archive = self.root / "package.zip"

    def write_manifest(self):
        (self.source / "plugin.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def build_and_check(self):
        build_archive(self.source, self.archive)
        return validate_archive(self.archive)

    def test_valid_archive_is_not_mistaken_for_submission_ready(self):
        report = self.build_and_check()
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["submission_ready"])
        self.assertIn("tutor", report["skills"])
        self.assertTrue(report["submission_gaps"])
        self.assertFalse(validate_archive(self.archive, submission=True)["valid"])

    def test_bytes_are_reproducible_across_source_mtime_and_mode(self):
        self.build_and_check()
        first = self.archive.read_bytes()
        os.utime(self.source / "README.md", (1720000000, 1720000000))
        (self.source / "README.md").chmod(0o600)
        build_archive(self.source, self.archive)
        self.assertEqual(first, self.archive.read_bytes())

    def test_path_traversal_is_rejected_without_extraction(self):
        self.build_and_check()
        with zipfile.ZipFile(self.archive, "a") as bundle:
            bundle.writestr(f"{PLUGIN_NAME}/../../escape.txt", "unsafe")
        self.assertFalse(validate_archive(self.archive)["valid"])
        self.assertFalse((self.root / "escape.txt").exists())

    def test_symlinks_rejected_during_build_and_in_archive(self):
        (self.source / "assets" / "linked.png").symlink_to(self.source / "assets" / "logo.png")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            build_archive(self.source, self.archive)
        (self.source / "assets" / "linked.png").unlink()
        self.build_and_check()
        with zipfile.ZipFile(self.archive, "a") as bundle:
            info = zipfile.ZipInfo(f"{PLUGIN_NAME}/assets/linked.png")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(info, "logo.png")
        self.assertFalse(validate_archive(self.archive)["valid"])

    def test_hidden_personal_binding_is_rejected(self):
        (self.source / ".app.json").write_text('{"apps": ["private"]}', encoding="utf-8")
        self.assertFalse(self.build_and_check()["valid"])

    def test_shadowed_compatibility_binding_is_rejected(self):
        overlay = self.source / ".codex-plugin" / "plugin.json"
        overlay.parent.mkdir()
        legacy = copy.deepcopy(self.manifest)
        legacy["extensions"]["com.openai"]["apps"] = "./.app.json"
        overlay.write_text(json.dumps(legacy), encoding="utf-8")
        self.assertFalse(self.build_and_check()["valid"])

    def test_credentials_and_signed_urls_are_rejected(self):
        for text in ("https://example.test/frame.png?X-Amz-Signature=123", 'Bearer abcdefghijklmnopqrstuvwxyz1234567890'):
            with self.subTest(text=text):
                (self.source / "README.md").write_text(text, encoding="utf-8")
                self.assertFalse(self.build_and_check()["valid"])

    def test_missing_markdown_target_is_rejected(self):
        (self.source / "README.md").write_text("[Missing](examples/gone.html)\n", encoding="utf-8")
        self.assertFalse(self.build_and_check()["valid"])

    def test_review_case_count_and_types_are_checked(self):
        review = self.manifest["extensions"]["com.openai"]["review"]["test_cases"]
        review["positive"].append(copy.deepcopy(review["positive"][0]))
        review["negative"][0]["prompt"] = ["Not a string"]
        self.write_manifest()
        errors = self.build_and_check()["errors"]
        self.assertTrue(any("Exactly 5" in error for error in errors))
        self.assertTrue(any("required fields" in error for error in errors))

    def test_malformed_metadata_is_rejected_without_crashing(self):
        for field in ("interface", "review", "publication"):
            with self.subTest(field=field):
                original = copy.deepcopy(self.manifest)
                self.manifest["extensions"]["com.openai"][field] = ["invalid"]
                self.write_manifest()
                self.assertFalse(self.build_and_check()["valid"])
                self.manifest = original

    def test_small_listing_logo_is_rejected(self):
        self.manifest["extensions"]["com.openai"]["interface"]["logo"] = "./assets/composer-icon.png"
        self.write_manifest()
        self.assertTrue(any("256" in error for error in self.build_and_check()["errors"]))

    def test_mcp_headers_cannot_embed_credentials(self):
        config = json.loads((self.source / "mcp.json").read_text())
        config["mcpServers"][PLUGIN_NAME]["headers"] = {"Authorization": "private"}
        (self.source / "mcp.json").write_text(json.dumps(config), encoding="utf-8")
        self.assertFalse(self.build_and_check()["valid"])

    def test_onboarding_must_exist_and_stay_inside_the_package(self):
        self.manifest["extensions"]["com.openai"]["onboardingSkill"] = "../outside/SKILL.md"
        self.write_manifest()
        self.assertFalse(self.build_and_check()["valid"])
        self.manifest["extensions"]["com.openai"]["onboardingSkill"] = "./skills/setup/SKILL.md"
        self.write_manifest()
        (self.source / "skills/setup/SKILL.md").unlink()
        self.assertFalse(self.build_and_check()["valid"])

    def test_output_cannot_be_inside_source(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            build_archive(self.source, self.source / "package.zip")


if __name__ == "__main__":
    unittest.main()
