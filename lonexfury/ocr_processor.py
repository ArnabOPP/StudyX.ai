"""
ocr_processor.py — Image & Handwritten Notes OCR Pipeline
==========================================================
Converts images of handwritten notes, screenshots, and scanned pages
into embedding-ready text chunks.

Supports:
  - Handwritten notes (photos from phone)
  - Scanned textbook pages
  - Whiteboard photos
  - Diagram images with labels
  - Images extracted from PDFs (passed from pdf_processor.py)

Pipeline:
  1. Load image → preprocess for OCR accuracy
     (grayscale → denoise → deskew → adaptive threshold)
  2. Run Tesseract OCR (with Hindi + English language packs)
  3. Post-process: clean artifacts, fix common OCR errors
  4. Detect if image contains a diagram (low text density → flag for visual)
  5. Chunk text into embedding-ready blocks
  6. Return (metadata, chunks)

Dependencies:
  pip install pillow pytesseract opencv-python-headless numpy langdetect
  # Also: install Tesseract binary + language packs
  # Windows: https://github.com/UB-Mannheim/tesseract/wiki
  # Linux:   sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-ben
"""

import os
import re
import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict


# ─────────────────────────────────────────────
# TESSERACT PATH (Windows — no PATH config needed)
# ─────────────────────────────────────────────
import pytesseract as _pt, platform as _pl
if _pl.system() == "Windows":
    _pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHUNK_MAX_CHARS  = 800        # slightly smaller for OCR (noisier text)
CHUNK_VERSION    = "1.0"
EMBEDDING_MODEL  = "paraphrase-multilingual-mpnet-base-v2"

# Tesseract language string — add 'ben' for Bengali if pack installed
TESSERACT_LANGS  = "eng+hin"

# If extracted text is below this density, image is likely a diagram
DIAGRAM_TEXT_DENSITY_THRESHOLD = 0.15   # chars per pixel (very rough)
MIN_TEXT_FOR_CHUNK = 30                 # skip images with almost no text

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def clean_ocr_text(text: str) -> str:
    """
    Post-process Tesseract output.
    Fixes common OCR artifacts from handwritten/scanned content.
    Conservative — never corrupts valid text.
    """
    # Remove null bytes and control chars
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Safe replacements only — these never corrupt valid English
    replacements = {
        r'[|]{2,}': '||',      # normalize pipes
        r'-\s*\n\s*': '',      # join hyphenated line breaks ("multi-\nlingual" → "multilingual")
        r'\n{3,}':  '\n\n',    # max 2 newlines
        r'[ \t]{2,}': ' ',     # multiple spaces → single space
        r'\u00ad': '',         # soft hyphens
        r'\ufeff': '',         # BOM
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Remove lines that are just symbols or very short garbage
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Keep line if it has enough alphanumeric content
        alpha_ratio = sum(c.isalnum() for c in stripped) / max(len(stripped), 1)
        if len(stripped) >= 2 and alpha_ratio >= 0.3:
            cleaned.append(stripped)
        elif stripped in {'.', ',', '-', '_', '|', '/', '\\', '=', '+'}:
            pass  # drop pure symbol lines
        elif stripped:
            cleaned.append(stripped)

    return '\n'.join(cleaned).strip()


def is_diagram_image(text: str, img_width: int, img_height: int) -> bool:
    """Heuristic: if very little text relative to image size, it's likely a diagram."""
    pixel_count  = img_width * img_height
    text_density = len(text) / max(pixel_count, 1)
    return text_density < DIAGRAM_TEXT_DENSITY_THRESHOLD and len(text) < 200


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_image(img_path: str):
    """
    Preprocess image for maximum OCR accuracy.
    Steps: resize → grayscale → denoise → deskew → adaptive threshold.
    Returns preprocessed PIL Image.
    """
    try:
        import cv2
        import numpy as np
        from PIL import Image

        img = cv2.imread(img_path)
        if img is None:
            # Fallback: try PIL for formats cv2 might miss
            pil_img = Image.open(img_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # ── 1. Upscale small images (OCR needs min ~300 DPI) ──────────
        min_dim = 1200
        if max(h, w) < min_dim:
            scale = min_dim / max(h, w)
            img   = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

        # ── 2. Grayscale ──────────────────────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── 3. Denoise (fastNlMeansDenoising works well for photos) ───
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # ── 4. Deskew ─────────────────────────────────────────────────
        coords = np.column_stack(np.where(denoised < 200))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                (h2, w2) = denoised.shape[:2]
                center   = (w2 // 2, h2 // 2)
                M        = cv2.getRotationMatrix2D(center, angle, 1.0)
                denoised = cv2.warpAffine(denoised, M, (w2, h2),
                                          flags=cv2.INTER_CUBIC,
                                          borderMode=cv2.BORDER_REPLICATE)

        # ── 5. Adaptive threshold (handles uneven lighting in photos) ─
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21, C=10
        )

        return Image.fromarray(thresh)

    except ImportError:
        # cv2 not available → basic PIL preprocessing
        from PIL import Image, ImageFilter, ImageEnhance
        img = Image.open(img_path).convert("L")     # grayscale
        img = ImageEnhance.Contrast(img).enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)
        return img
    except Exception as e:
        print(f"  ⚠️  Preprocessing failed ({e}) — using raw image")
        from PIL import Image
        return Image.open(img_path).convert("L")


# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────

def run_tesseract(img_path: str) -> Tuple[str, int, int]:
    """
    Run Tesseract on an image file with adaptive PSM selection.
    Returns (raw_text, img_width, img_height).

    PSM strategy for academic content:
      PSM 6  = uniform block of text (typed pages, clean notes)
      PSM 11 = sparse text (handwritten notes, mixed diagrams+text)
      PSM 4  = single column (lecture slides, textbook pages)
    We try all three and pick the one that returns the most text.
    """
    import pytesseract
    from PIL import Image

    base_config = f"--oem 3 -l {TESSERACT_LANGS} "
    psm_configs = [
        base_config + "--psm 6",   # uniform block — best for typed text
        base_config + "--psm 11",  # sparse text — best for handwritten notes
        base_config + "--psm 4",   # single column — best for slides/textbooks
    ]

    preprocessed = preprocess_image(img_path)
    w, h         = preprocessed.size

    best_text = ""
    for config in psm_configs:
        try:
            text = pytesseract.image_to_string(preprocessed, config=config)
            if len(text.strip()) > len(best_text.strip()):
                best_text = text
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError(
                "Tesseract not found. Install from:\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  Linux:   sudo apt install tesseract-ocr tesseract-ocr-hin"
            )
        except Exception:
            continue

    return best_text, w, h


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def build_chunks(
    text:      str,
    img_path:  str,
    file_meta: Dict,
    is_diagram: bool = False,
) -> List[Dict]:
    """Chunk OCR text into embedding-ready blocks."""
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    chunks      = []
    path        = Path(img_path)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    buffer    = ""
    chunk_idx = 0

    def emit(buf: str):
        nonlocal chunk_idx
        buf = buf.strip()
        if len(buf) < MIN_TEXT_FOR_CHUNK:
            return
        chunks.append({
            "original_text": buf,
            "english_text":  buf,
            "metadata": {
                "source_type":     "image",
                "file_name":       path.name,
                "source_url":      img_path,
                "title":           file_meta.get("title", path.stem),
                "is_diagram":      str(is_diagram),
                "chunk_index":     str(chunk_idx),
                "content_hash":    content_hash(buf),
                "chunk_version":   CHUNK_VERSION,
                "embedding_model": EMBEDDING_MODEL,
                "ingested_at":     ingested_at,
            },
        })
        chunk_idx += 1

    for sent in sentences:
        if not sent.strip():
            continue
        if len(buffer) + len(sent) > CHUNK_MAX_CHARS:
            emit(buffer)
            buffer = sent + " "
        else:
            buffer += sent + " "

    if buffer.strip():
        emit(buffer)

    return chunks


# ─────────────────────────────────────────────
# PUBLIC API — single image
# ─────────────────────────────────────────────

def process_image(img_path: str, title: str = "") -> Tuple[Dict, List[Dict]]:
    """
    Process a single image file through OCR pipeline.
    Returns (metadata, chunks).
    """
    path = Path(img_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix}")

    print(f"\n🖼️   OCR: {path.name}")

    raw_text, w, h = run_tesseract(img_path)
    cleaned        = clean_ocr_text(raw_text)
    diagram_flag   = is_diagram_image(cleaned, w, h)

    if diagram_flag:
        print(f"  📊  Detected as diagram (low text density) — flagged for visual pipeline")
    else:
        print(f"  ✅  Extracted {len(cleaned)} chars of text")

    file_meta = {
        "title":      title or path.stem,
        "file_name":  path.name,
        "img_width":  w,
        "img_height": h,
    }

    chunks = build_chunks(cleaned, img_path, file_meta, is_diagram=diagram_flag)

    metadata = {
        "source_type": "image",
        "file_name":   path.name,
        "title":       file_meta["title"],
        "source_url":  img_path,
        "is_diagram":  diagram_flag,
        "text_length": len(cleaned),
        "chunk_count": len(chunks),
    }

    return metadata, chunks


# ─────────────────────────────────────────────
# PUBLIC API — batch (called by processor.py)
# ─────────────────────────────────────────────

def process_image_batch(
    img_paths: List[str],
    title:     str = "",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process multiple images (e.g., all images from a PDF).
    Returns (all_metadata, all_chunks).
    """
    all_meta   = []
    all_chunks = []
    diagrams   = []

    for img_path in img_paths:
        try:
            meta, chunks = process_image(img_path, title=title)
            all_meta.append(meta)
            all_chunks.extend(chunks)
            if meta["is_diagram"]:
                diagrams.append(img_path)
        except Exception as e:
            print(f"  ⚠️  Skipped {img_path}: {e}")

    print(f"\n  ✅  OCR batch complete: {len(all_chunks)} chunks from {len(img_paths)} images")
    if diagrams:
        print(f"  📊  {len(diagrams)} diagram images flagged (stored separately for visual context)")

    return all_meta, all_chunks


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_processor.py <image_path_or_directory>")
        print("       Supports: .png .jpg .jpeg .webp .bmp .tiff")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_dir():
        img_paths = [
            str(p) for p in target.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        print(f"📂  Found {len(img_paths)} images in {target}")
        all_meta, all_chunks = process_image_batch(img_paths)
    else:
        meta, all_chunks = process_image(str(target))
        all_meta = [meta]

    print("\n" + "=" * 55)
    print("  📊  RESULTS")
    print("=" * 55)
    for m in all_meta:
        diag = " [DIAGRAM]" if m.get("is_diagram") else ""
        print(f"  {m['file_name']}{diag} → {m['chunk_count']} chunks")

    for i, chunk in enumerate(all_chunks[:3]):
        print(f"\n{'─' * 55}")
        print(f"  CHUNK {i}  |  {chunk['metadata']['file_name']}")
        print(f"{'─' * 55}")
        print(f"  {chunk['english_text'][:300]}")

    export = input("\n💾  Export to JSON? (y/N): ").strip().lower()
    if export == "y":
        out_name = target.stem + "_ocr_chunks.json"
        with open(out_name, "w", encoding="utf-8") as f:
            json.dump({"metadata": all_meta, "chunks": all_chunks},
                      f, ensure_ascii=False, indent=2)
        print(f"  ✅  Saved to '{out_name}'")