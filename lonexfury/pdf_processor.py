"""
pdf_processor.py — Document Ingestion Pipeline
===============================================
Handles PDF, Word (.docx), and plain text files.
Extracts text, tables, images, and metadata.
Outputs embedding-ready chunks identical in structure to youtube_scraper.py.

Pipeline:
  1. Detect file type
  2. Extract text page-by-page (PyMuPDF for PDF, python-docx for Word)
  3. Extract tables separately (pdfplumber) — formatted as markdown
  4. Extract embedded images → save for OCR pipeline
  5. Detect section headings → use as chapter markers
  6. Chunk into ~CHUNK_MAX_CHARS blocks with overlap
  7. Return (metadata, chunks)

Dependencies:
  pip install pymupdf pdfplumber python-docx pillow langdetect
"""

import os
import re
import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CHUNK_MAX_CHARS    = 1000
CHUNK_OVERLAP_SEGS = 2
CHUNK_VERSION      = "1.0"
EMBEDDING_MODEL    = "paraphrase-multilingual-mpnet-base-v2"

# Heading detection — lines matching these are section markers
# Rules are intentionally strict to avoid false positives on body text
HEADING_PATTERNS = [
    r"^(chapter|unit|section|module|topic|part)\s+[\dIVXivx]+",  # "Chapter 3", "Part One"
    r"^\d+[\.\d]*\s+[A-Z][A-Z\s]{3,40}$",      # "1.2 LAWS OF THERMODYNAMICS" — must end with caps
    r"^[A-Z][A-Z\s]{4,35}$",                    # ALL CAPS, 5-36 chars, no lowercase at all
]

# Lines containing these words are never headings (body text false positives)
HEADING_BLACKLIST = [
    r"\bthe\b", r"\band\b", r"\bwas\b", r"\bwere\b", r"\bhave\b",
    r"\bthis\b", r"\bthat\b", r"\bwith\b", r"\bfrom\b", r"\byour\b",
    r"\bnever\b", r"\bbecause\b", r"\bwould\b", r"\bcould\b",
]

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_heading(line: str) -> bool:
    line = line.strip()
    if len(line) < 3 or len(line) > 80:
        return False
    # Reject if line contains common body-text words (case-insensitive)
    line_lower = line.lower()
    for bl in HEADING_BLACKLIST:
        if re.search(bl, line_lower):
            return False
    for pattern in HEADING_PATTERNS:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False


def clean_text(text: str) -> str:
    """Remove excessive whitespace, fix common PDF extraction artifacts."""
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    # Max 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove page numbers (standalone digits on a line)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Remove headers/footers repeated every page (heuristic: lines < 80 chars at top)
    return text.strip()


def table_to_markdown(table: List[List[Any]]) -> str:
    """Convert pdfplumber table (list of rows) to markdown table string."""
    if not table or not table[0]:
        return ""
    rows = []
    for i, row in enumerate(table):
        cells = [str(c).strip() if c is not None else "" for c in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(rows)


# ─────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────

def extract_pdf(file_path: str) -> Tuple[List[Dict], List[str], Dict]:
    """
    Extract text and tables from PDF.
    Returns:
      pages      : list of {page_num, text, heading, tables}
      image_paths: list of saved image file paths (for OCR pipeline)
      file_meta  : basic file metadata
    """
    import fitz  # PyMuPDF
    import pdfplumber

    path = Path(file_path)
    pages      = []
    image_paths = []

    # ── PyMuPDF: text + images ─────────────────────────────────────────
    doc = fitz.open(file_path)
    file_meta = {
        "title":      doc.metadata.get("title", path.stem) or path.stem,
        "author":     doc.metadata.get("author", ""),
        "page_count": doc.page_count,
        "file_name":  path.name,
        "file_size":  os.path.getsize(file_path),
    }

    img_dir = path.parent / f"__images_{path.stem}__"

    for page_num in range(doc.page_count):
        page      = doc[page_num]
        raw_text  = page.get_text("text")
        text      = clean_text(raw_text)

        # Detect heading from first non-empty line
        heading = ""
        for line in text.split("\n"):
            line = line.strip()
            if line and is_heading(line):
                heading = line
                break

        # Extract embedded images
        img_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(img_list):
            try:
                xref  = img_info[0]
                pix   = fitz.Pixmap(doc, xref)
                if pix.width < 50 or pix.height < 50:
                    continue                         # skip tiny decorative images
                img_dir.mkdir(exist_ok=True)
                img_path = img_dir / f"p{page_num+1}_img{img_idx}.png"
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(str(img_path))
                image_paths.append(str(img_path))
            except Exception:
                pass

        pages.append({
            "page_num": page_num + 1,
            "text":     text,
            "heading":  heading,
            "tables":   [],          # filled by pdfplumber below
        })

    doc.close()

    # ── pdfplumber: table extraction ──────────────────────────────────
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, plpage in enumerate(pdf.pages):
                tables = plpage.extract_tables()
                if tables:
                    md_tables = [table_to_markdown(t) for t in tables if t]
                    pages[page_num]["tables"] = md_tables
    except Exception as e:
        print(f"  ⚠️  pdfplumber table extraction failed: {e}")

    return pages, image_paths, file_meta


# ─────────────────────────────────────────────
# DOCX EXTRACTION
# ─────────────────────────────────────────────

def extract_docx(file_path: str) -> Tuple[List[Dict], List[str], Dict]:
    """
    Extract text and tables from Word document.
    Treats each paragraph group as a "page" equivalent.
    """
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    path     = Path(file_path)
    doc      = Document(file_path)
    pages    = []
    images   = []

    file_meta = {
        "title":      doc.core_properties.title or path.stem,
        "author":     doc.core_properties.author or "",
        "page_count": None,          # Word doesn't expose page count easily
        "file_name":  path.name,
        "file_size":  os.path.getsize(file_path),
    }

    # Collect all block elements in order
    current_section  = []
    current_heading  = ""
    section_idx      = 0

    def flush_section():
        nonlocal section_idx
        if current_section:
            text = "\n".join(current_section).strip()
            if text:
                pages.append({
                    "page_num": section_idx + 1,
                    "text":     clean_text(text),
                    "heading":  current_heading,
                    "tables":   [],
                })
                section_idx += 1

    for elem in doc.element.body:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag == "p":
            # Paragraph
            from docx.oxml.ns import qn
            para_text = "".join(
                node.text for node in elem.iter(qn("w:t")) if node.text
            ).strip()
            if not para_text:
                continue

            # Check if heading style
            style_name = ""
            pPr = elem.find(f"{{{elem.nsmap.get('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')}}}pPr")
            if pPr is not None:
                pStyle = pPr.find(f"{{{elem.nsmap.get('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')}}}pStyle")
                if pStyle is not None:
                    style_name = pStyle.get(f"{{{elem.nsmap.get('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')}}}val", "")

            if "Heading" in style_name or is_heading(para_text):
                flush_section()
                current_heading = para_text
                current_section = [para_text]
            else:
                current_section.append(para_text)

        elif tag == "tbl":
            # Table — convert to markdown inline
            rows = []
            for tr in elem.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"):
                row = []
                for tc in tr.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                    cell_text = "".join(
                        t.text for t in tc.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                        if t.text
                    ).strip()
                    row.append(cell_text)
                if row:
                    rows.append(row)
            if rows:
                md = table_to_markdown(rows)
                current_section.append(md)
                if pages:
                    pages[-1]["tables"].append(md)

    flush_section()

    return pages, images, file_meta


# ─────────────────────────────────────────────
# TXT EXTRACTION
# ─────────────────────────────────────────────

def extract_txt(file_path: str) -> Tuple[List[Dict], List[str], Dict]:
    path = Path(file_path)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    file_meta = {
        "title":      path.stem,
        "author":     "",
        "page_count": 1,
        "file_name":  path.name,
        "file_size":  os.path.getsize(file_path),
    }

    pages = [{
        "page_num": 1,
        "text":     clean_text(content),
        "heading":  "",
        "tables":   [],
    }]
    return pages, [], file_meta


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def build_chunks(
    pages:     List[Dict],
    file_meta: Dict,
    source_path: str,
) -> List[Dict]:
    """
    Chunk page content into ~CHUNK_MAX_CHARS blocks.

    Overlap strategy:
      - Each chunk's text is CLEAN — starts at a sentence boundary, no mid-word cuts.
      - The last 2 sentences of each chunk are stored as 'overlap_context' metadata.
      - RAG retrieval uses overlap_context to bridge chunk boundaries during search.
      - This means chunks never start mid-sentence or mid-word.

    Tables are emitted as separate standalone chunks.
    """
    ingested_at  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    chunk_idx    = 0
    chunks       = []

    # Active sentence buffer — list of (sentence_str, page_num, heading_str)
    sent_buffer: List[Tuple[str, int, str]] = []
    buf_len      = 0
    overlap_ctx  = ""    # last 2 sentences of previous chunk (for metadata only)

    def emit_chunk_from_sents(sents: List[Tuple[str, int, str]], overlap: str):
        nonlocal chunk_idx
        if not sents:
            return
        text    = " ".join(s[0] for s in sents).strip()
        page    = sents[0][1]
        heading = sents[0][2]
        if not text or len(text) < 10:
            return
        chunks.append({
            "original_text": text,
            "english_text":  text,
            "metadata": {
                "source_type":     "document",
                "file_name":       file_meta["file_name"],
                "source_url":      source_path,
                "title":           file_meta["title"],
                "author":          file_meta.get("author", ""),
                "page_num":        str(page),
                "heading":         heading,
                "chunk_type":      "text",
                "chunk_index":     str(chunk_idx),
                "overlap_context": overlap,   # bridge to previous chunk for RAG
                "content_hash":    content_hash(text),
                "chunk_version":   CHUNK_VERSION,
                "embedding_model": EMBEDDING_MODEL,
                "ingested_at":     ingested_at,
            },
        })
        chunk_idx += 1

    def emit_table_chunk(text: str, page: int, heading: str):
        nonlocal chunk_idx
        text = text.strip()
        if not text:
            return
        chunks.append({
            "original_text": text,
            "english_text":  text,
            "metadata": {
                "source_type":     "document",
                "file_name":       file_meta["file_name"],
                "source_url":      source_path,
                "title":           file_meta["title"],
                "author":          file_meta.get("author", ""),
                "page_num":        str(page),
                "heading":         heading,
                "chunk_type":      "table",
                "chunk_index":     str(chunk_idx),
                "overlap_context": "",
                "content_hash":    content_hash(text),
                "chunk_version":   CHUNK_VERSION,
                "embedding_model": EMBEDDING_MODEL,
                "ingested_at":     ingested_at,
            },
        })
        chunk_idx += 1

    def flush_buffer():
        """Emit current buffer as a chunk, compute overlap for next chunk.

        If the last item in the buffer doesn't end with sentence-ending
        punctuation (indicating it was cut mid-thought), carry it forward
        to the next chunk so the chunk boundary lands cleanly.
        """
        nonlocal overlap_ctx, chunk_idx, sent_buffer, buf_len
        if not sent_buffer:
            return

        # Back up: if last sentence ends without terminal punctuation,
        # it's likely a mid-thought cut — carry it to next chunk.
        carry_forward: List[Tuple[str, int, str]] = []
        emit_sents = sent_buffer[:]

        # Only back up if we have multiple sentences (don't empty the chunk)
        while len(emit_sents) > 1:
            last = emit_sents[-1][0].rstrip()
            if last.endswith((".", "!", "?", '"', "”", "’", ":", ";")):
                break
            carry_forward.insert(0, emit_sents.pop())

        emit_chunk_from_sents(emit_sents, overlap_ctx)
        # Overlap = last 2 emitted sentences (metadata only for next chunk)
        overlap_sents = emit_sents[-2:]
        overlap_ctx   = " ".join(s[0] for s in overlap_sents)
        # Next chunk starts with the carry-forward fragments (if any)
        sent_buffer   = carry_forward
        buf_len       = sum(len(s[0]) + 1 for s in sent_buffer)

    # ── Pre-process: merge sentences that split across page boundaries ────
    # PDFs often break mid-sentence at page boundaries. If page N ends without
    # sentence-ending punctuation, glue its last fragment to page N+1's first.
    pending_fragment = ""    # carries the broken end of previous page

    for page_idx, page in enumerate(pages):
        current_page = page["page_num"]
        current_head = page.get("heading", "")

        # ── Tables: emit as standalone chunks ─────────────────────────
        for table_md in page.get("tables", []):
            if len(table_md.strip()) > 10:
                emit_table_chunk(
                    f"Table from page {current_page}:\n{table_md}",
                    current_page, current_head,
                )

        # ── Text: split into sentences, accumulate cleanly ─────────────
        text = page["text"]
        if not text:
            continue

        # Glue any pending fragment from previous page onto this page's start
        if pending_fragment:
            text = pending_fragment + " " + text
            pending_fragment = ""

        # Detect if THIS page ends mid-sentence (no terminal punctuation)
        # If so, peek the last fragment and hold it for next page
        text_stripped = text.rstrip()
        ends_cleanly  = text_stripped.endswith((".", "!", "?", '"', "”", "’"))

        # Only hold a fragment if there's a next page to glue to
        if not ends_cleanly and page_idx < len(pages) - 1:
            # Find last sentence boundary; everything after it is the fragment
            last_boundary = max(
                text_stripped.rfind(". "),
                text_stripped.rfind("! "),
                text_stripped.rfind("? "),
                text_stripped.rfind(".\n"),
                text_stripped.rfind("!\n"),
                text_stripped.rfind("?\n"),
            )
            if last_boundary > 0 and last_boundary > len(text_stripped) - 200:
                # Fragment is short enough to plausibly be a broken sentence
                pending_fragment = text_stripped[last_boundary + 2:].strip()
                text             = text_stripped[: last_boundary + 1]

        # Split on sentence-ending punctuation followed by whitespace, or newlines
        raw_sents = re.split(r'(?<=[.!?])\s+|\n', text)

        for raw in raw_sents:
            sent = raw.strip()
            if not sent or len(sent) < 2:
                continue

            # Single sentence longer than limit → flush then hard-split at words
            if len(sent) > CHUNK_MAX_CHARS:
                flush_buffer()
                words   = sent.split()
                segment = ""
                for word in words:
                    if len(segment) + len(word) + 1 > CHUNK_MAX_CHARS:
                        if segment.strip():
                            sent_buffer.append((segment.strip(), current_page, current_head))
                            buf_len += len(segment)
                            flush_buffer()
                        segment = word + " "
                    else:
                        segment += word + " "
                if segment.strip():
                    sent_buffer.append((segment.strip(), current_page, current_head))
                    buf_len += len(segment)
                continue

            # Would adding this sentence overflow the chunk?
            if buf_len + len(sent) + 1 > CHUNK_MAX_CHARS and sent_buffer:
                flush_buffer()

            sent_buffer.append((sent, current_page, current_head))
            buf_len += len(sent) + 1

    # If pending fragment leftover after last page, append it as final content
    if pending_fragment:
        sent_buffer.append((pending_fragment, pages[-1]["page_num"], ""))
        buf_len += len(pending_fragment)

    # Flush any remaining sentences
    if sent_buffer:
        flush_buffer()

    return chunks


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def is_scanned_pdf(pages: List[Dict], threshold: int = 50) -> bool:
    """
    Heuristic: if pages average fewer than `threshold` chars of extracted text,
    the PDF is likely scanned (text baked into page images).
    """
    if not pages:
        return False
    total_text = sum(len(p.get("text", "")) for p in pages)
    avg_chars  = total_text / len(pages)
    return avg_chars < threshold


def render_pdf_to_images(file_path: str, dpi: int = 200) -> List[str]:
    """
    Render each PDF page as a PNG image (for scanned PDFs).
    Returns list of image file paths.
    """
    import fitz
    path     = Path(file_path)
    img_dir  = path.parent / f"__rendered_{path.stem}__"
    img_dir.mkdir(exist_ok=True)
    img_paths = []

    doc = fitz.open(file_path)
    zoom = dpi / 72
    mat  = fitz.Matrix(zoom, zoom)

    for page_num in range(doc.page_count):
        page = doc[page_num]
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        img_path = img_dir / f"page_{page_num+1:03d}.png"
        pix.save(str(img_path))
        img_paths.append(str(img_path))

    doc.close()
    return img_paths


def process_document(file_path: str) -> Tuple[Dict, List[Dict]]:
    """
    Main entry point. Process any supported document into structured chunks.
    Returns (metadata, chunks).
    Raises ValueError for unsupported types, RuntimeError on extraction failure.

    For PDFs detected as scanned (low text density), automatically renders
    each page as an image and routes through OCR pipeline.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    print(f"\n📄  Processing: {path.name}")
    print(f"  📁  Type: {ext.upper().lstrip('.')}")
    print(f"  📦  Size: {os.path.getsize(file_path) / 1024:.1f} KB\n")

    # ── Extract ────────────────────────────────────────────────────────
    if ext == ".pdf":
        pages, image_paths, file_meta = extract_pdf(file_path)

        # Auto-detect scanned PDFs and route to OCR fallback
        if is_scanned_pdf(pages):
            print(f"  🔍  Low text density detected — appears to be a scanned PDF")
            print(f"  🖼️   Rendering pages as images for OCR pipeline…")
            try:
                rendered_pages   = render_pdf_to_images(file_path)
                image_paths.extend(rendered_pages)
                file_meta["scanned"] = True
                print(f"  ✅  Rendered {len(rendered_pages)} pages for OCR")
            except Exception as e:
                print(f"  ⚠️  Page rendering failed: {e}")

    elif ext in {".docx", ".doc"}:
        pages, image_paths, file_meta = extract_docx(file_path)
    elif ext == ".txt":
        pages, image_paths, file_meta = extract_txt(file_path)
    else:
        raise ValueError(f"Handler missing for {ext}")

    if not pages:
        raise RuntimeError("No content extracted from document.")

    print(f"  ✅  Extracted {len(pages)} sections")
    if image_paths:
        print(f"  🖼️   Found {len(image_paths)} images → queued for OCR")

    # ── Chunk ──────────────────────────────────────────────────────────
    print("  📦  Building chunks…")
    chunks = build_chunks(pages, file_meta, file_path)
    print(f"  ✅  {len(chunks)} chunks ready\n")

    metadata = {
        "source_type": "document",
        "file_name":   file_meta["file_name"],
        "title":       file_meta["title"],
        "author":      file_meta.get("author", ""),
        "page_count":  file_meta.get("page_count"),
        "source_url":  file_path,
        "image_paths": image_paths,    # passed to OCR processor
        "chunk_count": len(chunks),
    }

    return metadata, chunks


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <file_path>")
        sys.exit(1)

    try:
        meta, chunks = process_document(sys.argv[1])
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"\n❌  {e}")
        sys.exit(1)

    print("=" * 55)
    print("  📊  RESULTS")
    print("=" * 55)
    print(f"  Title      : {meta['title']}")
    print(f"  Pages      : {meta['page_count']}")
    print(f"  Chunks     : {meta['chunk_count']}")
    print(f"  Images     : {len(meta['image_paths'])}")
    print()

    for i, chunk in enumerate(chunks[:5]):
        print(f"{'─' * 55}")
        print(f"  CHUNK {i}  |  Page {chunk['metadata']['page_num']}"
              f"  |  {chunk['metadata']['heading'][:40]}")
        print(f"{'─' * 55}")
        print(f"  {chunk['english_text'][:300]}…\n")

    if len(chunks) > 5:
        print(f"  … and {len(chunks) - 5} more chunks\n")

    export = input("💾  Export to JSON? (y/N): ").strip().lower()
    if export == "y":
        out_file = Path(sys.argv[1]).stem + "_chunks.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"metadata": meta, "chunks": chunks}, f, ensure_ascii=False, indent=2)
        print(f"  ✅  Saved to '{out_file}'")