"""Repeatable checks for Brandwise's deterministic parts (no AI service needed).

Run from the BRANDWISE folder with the virtual environment active:
    python tests\\run_checks.py
"""
import os
import sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app import rules, imagecheck  # noqa: E402

results = []


def check(name, condition):
    results.append((name, bool(condition)))
    print(("PASS  " if condition else "FAIL  ") + name)


def cats(findings):
    return [f["category"] for f in findings]


def make_image(bg, box, size=(800, 400)):
    """A test image: a background colour with a coloured block covering about 40% of it."""
    img = Image.new("RGB", size, bg)
    ImageDraw.Draw(img).rectangle([0, 0, int(size[0] * 0.4), size[1]], fill=box)
    return img


print("--- Rule checks ---")
clean = rules.check_rules("Managing money should feel calm.", "twitter")
check("Clean text produces no findings", clean == [])

risky = rules.check_rules("Finwise offers risk-free investing for everyone.", "twitter")
check("Banned word 'risk-free' is found", "banned_word" in cats(risky))
check("Missing regulatory disclaimer is found", "missing_disclaimer" in cats(risky))

disc = ("Finwise is a financial technology company, not an FDIC-insured bank. "
        "Banking services provided by partner banks, Members FDIC.")
with_disc = rules.check_rules("Investing made simple. " + disc, "twitter")
check("No disclaimer finding when the disclaimer is present", "missing_disclaimer" not in cats(with_disc))

check("Preferred term 'cheap' is found", "preferred_term" in cats(rules.check_rules("A cheap way to save.", "twitter")))
check("Twitter length limit is enforced", "channel_limit" in cats(rules.check_rules("x" * 300, "twitter")))
check("Email subject length limit is enforced", "channel_limit" in cats(rules.check_rules("y" * 100, "email_subject")))

print("--- Colour and contrast maths ---")
black, white = (0, 0, 0), (255, 255, 255)
check("Black on white has contrast 21:1", abs(imagecheck.contrast_ratio(black, white) - 21.0) < 0.1)
check("Same colour has contrast 1:1", abs(imagecheck.contrast_ratio(white, white) - 1.0) < 0.01)
check("Grey is treated as neutral", imagecheck.is_neutral((128, 128, 128)))
check("Orange is not neutral", not imagecheck.is_neutral((228, 87, 46)))

print("--- Alt text checks ---")
cfg = imagecheck.load_config()
check("Missing alt text is a high-severity finding",
      imagecheck.alt_text_findings("", cfg)[0]["severity"] == "high")
check("A file name used as alt text is flagged", len(imagecheck.alt_text_findings("IMG_1234.png", cfg)) >= 1)
check("'Image of ...' alt text is flagged", len(imagecheck.alt_text_findings("Image of a cat on a sofa", cfg)) >= 1)
check("Good alt text passes",
      imagecheck.alt_text_findings("Finwise banner reading: managing money should feel calm.", cfg) == [])

print("--- Image checks ---")
low = make_image("#8FA3BF", "#A9BAD1")
low_findings, _ = imagecheck.image_findings(low, "A banner for the Finwise launch campaign")
check("Low-contrast image gets a contrast finding", "image_contrast" in cats(low_findings))

off = make_image("#E4572E", "#14213D")
off_findings, _ = imagecheck.image_findings(off, "A banner for the Finwise launch campaign")
check("Off-brand colour gets a palette finding", "image_palette" in cats(off_findings))

good = make_image("#14213D", "#D4AF37")
good_findings, _ = imagecheck.image_findings(good, "A banner for the Finwise launch campaign")
check("On-brand, high-contrast image has no colour findings",
      "image_palette" not in cats(good_findings) and "image_contrast" not in cats(good_findings))

small_findings, _ = imagecheck.image_findings(make_image("#14213D", "#D4AF37", (300, 200)), "A small banner image")
check("Small image gets a size finding", "image_size" in cats(small_findings))

try:
    imagecheck.decode_data_url("not an image")
    check("Invalid image data is rejected", False)
except ValueError:
    check("Invalid image data is rejected", True)

passed = sum(1 for _, ok in results if ok)
print(f"\n{passed} of {len(results)} checks passed.")
sys.exit(0 if passed == len(results) else 1)