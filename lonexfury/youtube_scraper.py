"""
youtube_scraper.py — YouTube Ingestion Pipeline
================================================
Scrapes any YouTube video into structured, embedding-ready chunks.

What's stored per chunk:
  - original_text   : transcript in source language
  - english_text    : Google-translated or Whisper-translated English
  - language        : detected ISO language code (e.g. 'hi', 'en')
  - chunk_index     : position in video (for neighbour retrieval)
  - timestamp       : [HH:MM:SS] of chunk start
  - start_seconds   : raw float for range queries
  - source_type     : always 'youtube'
  - video_id, title, channel, duration_s, upload_date, source_url

Pipeline:
  1. Validate URL → extract video ID
  2. Fetch metadata via yt-dlp (incl. chapters, view/like counts)
  3. Cloud transcript API (instant, free) — tries all API versions
  4. Fallback → download audio → Faster-Whisper (GPU if available)
  5. If non-English:
       a. Try Google Translate per-segment (free, accurate, no audio needed)
          — with retry on rate limit
       b. Fallback → Whisper translate pass on audio
  6. Chunk original segments into ~CHUNK_MAX_CHARS blocks
     Map English onto each chunk by timestamp range
  7. Return (metadata, chunks)

Changes vs v1.4:
  - WHISPER_MODEL_SIZE: small → medium (better translation quality on GPU)
  - CHUNK_MAX_CHARS: 600 → 1000 (better for longer videos)
  - EMBEDDING_MODEL constant added — metadata no longer hardcoded to old model
  - Language detection samples min(50, total) segments instead of just 15
  - Google Translate: retry with backoff on rate limit / connection error
  - Segment duration stored in each segment dict
  - embed_text field removed from chunks (database.py embeds english_text directly)
  - CLI JSON export defaults to Enter=n (no longer forces a keypress)

Dependencies:
  pip install yt-dlp youtube-transcript-api faster-whisper imageio-ffmpeg deep-translator langdetect
"""

import os
import re
import sys
import time
import json
import hashlib
import threading
from datetime import datetime, timezone
from typing import Optional, List, Tuple
import imageio_ffmpeg
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

# Whisper model — auto-selected at runtime based on GPU availability:
#   GPU present → "medium" (better quality, fast on CUDA)
#   CPU only    → "small"  (medium on CPU = 10-15 min per video, too slow)
# Override by setting env var: WHISPER_MODEL_SIZE=large-v2
WHISPER_MODEL_SIZE  = os.getenv("WHISPER_MODEL_SIZE", "auto")
WHISPER_BEAM_SIZE   = 1
WHISPER_CPU_THREADS = 4
CHUNK_MAX_CHARS     = 1000
CHUNK_OVERLAP_SEGS  = 2

CHUNK_VERSION = "1.6"

# Must match database.py EMBEDDING_MODEL — stored in chunk metadata for traceability
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

TRANSCRIPT_LANGUAGES = ["en", "hi", "es", "fr", "de", "pt", "ja", "ko", "zh-Hans", "zh-Hant"]
ENGLISH_CODES        = {"en", "en-US", "en-GB", "en-AU", "en-CA", "en-IN"}

GTRANS_BATCH_SIZE  = 10       # reduced: Hindi segments are long, 20 risks 5000-char API limit
GTRANS_MAX_RETRIES = 3
GTRANS_RETRY_DELAY = 5.0

# yt-dlp headers — prevents blocking at captive portals and rate-limited networks
YTDLP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────

def clean_translated_text(text: str) -> str:
    """
    Fix Google Translate artifacts — conservatively.

    Root cause of word merges: Google Translate strips spaces at XML tag
    boundaries. Fix is applied at the SOURCE (tags include trailing space)
    so this cleaner only handles residual safe cases:

      1. Remove stray XML tag remnants from failed extractions
      2. Fix missing space AFTER punctuation (e.g. "Ok.What" → "Ok. What")
         — safe because punctuation never legitimately touches a letter
      3. Normalize multiple spaces → single space
      4. Strip leading/trailing whitespace

    We deliberately do NOT regex-replace inside words — that causes more
    damage than the original missing-space problem.
    """
    if not text:
        return text

    # 1. Remove stray XML tag remnants
    text = re.sub(r'</?s\d+>', '', text)

    # 2. Fix missing space after sentence-ending punctuation
    #    e.g. "Ok.What" → "Ok. What"  |  "done!He" → "done! He"
    #    Only triggers when punctuation is directly followed by a letter —
    #    safe because "₹1.5" won't match (digit after dot, not letter)
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)

    # 3. Normalize whitespace
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.strip()

    return text


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def extract_video_id(url: str) -> Optional[str]:
    """Extract the 11-character YouTube video ID from any valid YouTube URL format.
    Returns None for non-YouTube URLs (Instagram, Facebook, Twitter/X etc.) —
    those still work fine via yt-dlp but don't have a video_id field.
    """
    patterns = [
        r'(?:youtube\.com|music\.youtube\.com)/shorts/([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com|music\.youtube\.com)/live/([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com|music\.youtube\.com)/embed/([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com|music\.youtube\.com)/v/([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com|music\.youtube\.com)/watch\?.*?v=([a-zA-Z0-9_-]{11})',
        r'youtu\.be/([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_supported_url(url: str) -> bool:
    """
    Returns True if the URL is from a platform yt-dlp can handle.
    Covers YouTube, Instagram Reels, Facebook videos, Twitter/X, and
    any other yt-dlp-supported site (1800+ platforms).
    """
    supported_domains = [
        "youtube.com", "youtu.be", "music.youtube.com",
        "instagram.com", "facebook.com", "fb.watch",
        "twitter.com", "x.com", "t.co",
        "vimeo.com", "dailymotion.com", "twitch.tv",
    ]
    url_lower = url.lower()
    return any(domain in url_lower for domain in supported_domains)


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to '[HH:MM:SS]'."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


def content_hash(text: str) -> str:
    """SHA-256 hash of text — detects when chunk content changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _Spinner:
    """CLI spinner for blocking I/O."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str):
        self._msg     = message
        self._running = False
        self._thread  = None

    def _spin(self):
        i = 0
        while self._running:
            sys.stdout.write(f"\r  {self.FRAMES[i % len(self.FRAMES)]}  {self._msg}  ")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write("\r" + " " * (len(self._msg) + 10) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self._running = True
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._running = False
        self._thread.join()


# ─────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────

def fetch_video_metadata(url: str, ffmpeg_path: str) -> dict:
    """Fetch title, channel, duration, chapters, upload date — no media download.
    Uses a real browser User-Agent to avoid blocks at captive-portal networks.
    Socket timeout of 30s prevents indefinite hangs on slow connections.
    """
    opts = {
        "quiet":           True,
        "no_warnings":     True,
        "skip_download":   True,
        "ffmpeg_location": ffmpeg_path,
        "http_headers":    YTDLP_HEADERS,
        "socket_timeout":  30,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    chapters = []
    for ch in (info.get("chapters") or []):
        chapters.append({
            "title":      ch.get("title", ""),
            "start_time": ch.get("start_time", 0),
            "end_time":   ch.get("end_time", 0),
        })

    return {
        "title":       info.get("title",       "Untitled"),
        "channel":     info.get("uploader",    "Unknown"),
        "description": info.get("description", ""),
        "duration_s":  info.get("duration",    0),
        "upload_date": info.get("upload_date", ""),
        "source_url":  url,
        "source_type": "youtube",
        "video_id":    info.get("id", extract_video_id(url)),
        "chapters":    chapters,
        "view_count":  info.get("view_count",  0),
        "like_count":  info.get("like_count",  0),
    }


# ─────────────────────────────────────────────
# CLOUD TRANSCRIPT  (v0.x / v1.x / v2.x safe)
# ─────────────────────────────────────────────

def _parse_segments(raw) -> List[dict]:
    """
    Normalise across all youtube-transcript-api versions.
    Returns list of {'text', 'start', 'duration'} dicts.
    Duration is stored for tighter timestamp boundary mapping.
    """
    if hasattr(raw, "to_raw_data"):
        raw = raw.to_raw_data()
    result = []
    for seg in raw:
        if isinstance(seg, dict):
            text     = seg.get("text",     "").strip()
            start    = float(seg.get("start",    0))
            duration = float(seg.get("duration", 0))
        else:
            text     = getattr(seg, "text",     "").strip()
            start    = float(getattr(seg, "start",    0.0))
            duration = float(getattr(seg, "duration", 0.0))
        if text:
            result.append({"text": text, "start": start, "duration": duration})
    return result


def fetch_cloud_transcript(video_id: str) -> Optional[List[dict]]:
    """Try every known youtube-transcript-api calling style. Returns segments or None."""
    from youtube_transcript_api import YouTubeTranscriptApi

    # Strategy 1: v1.x instance with language preference
    try:
        api    = YouTubeTranscriptApi()
        result = _parse_segments(api.fetch(video_id, languages=TRANSCRIPT_LANGUAGES))
        if result:
            return result
    except Exception as e:
        print(f"  ℹ️  fetch(languages) failed: {e}")

    # Strategy 2: v1.x instance without preference
    try:
        api    = YouTubeTranscriptApi()
        result = _parse_segments(api.fetch(video_id))
        if result:
            return result
    except Exception as e:
        print(f"  ℹ️  fetch() failed: {e}")

    # Strategy 3: v1.x list() → find best track
    try:
        api   = YouTubeTranscriptApi()
        tlist = api.list(video_id)
        track = None
        try:
            track = tlist.find_manually_created_transcript(TRANSCRIPT_LANGUAGES)
        except Exception:
            pass
        if track is None:
            try:
                track = tlist.find_generated_transcript(TRANSCRIPT_LANGUAGES)
            except Exception:
                pass
        if track is None:
            track = next(iter(tlist), None)
        if track is not None:
            result = _parse_segments(track.fetch())
            if result:
                return result
    except Exception as e:
        print(f"  ℹ️  list() strategy failed: {e}")

    # Strategy 4: v0.x class method
    try:
        from youtube_transcript_api import YouTubeTranscriptApi as YTA
        result = _parse_segments(YTA.get_transcript(video_id, languages=TRANSCRIPT_LANGUAGES))
        if result:
            return result
    except Exception as e:
        print(f"  ℹ️  get_transcript() failed: {e}")

    return None


# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────

def detect_language(segments: List[dict]) -> str:
    """
    Detect dominant language from transcript segments.
    Samples up to 50 segments spread evenly across the full video
    (not just the first 50) — prevents misdetection when intros are
    in English but the rest of the video is in Hindi/Bengali.
    Falls back to 'en' if langdetect is not available.
    """
    try:
        from langdetect import detect
        total = len(segments)
        if total == 0:
            return "en"
        # Evenly spaced sample indices across full video
        step        = max(1, total // 50)
        sample_segs = segments[::step][:50]
        sample      = " ".join(s["text"] for s in sample_segs)
        return detect(sample)
    except Exception:
        return "en"


# ─────────────────────────────────────────────
# GOOGLE TRANSLATE  (primary translation path)
# ─────────────────────────────────────────────

def translate_segments_google(
    segments: List[dict],
    src_lang:  str,
) -> Optional[List[dict]]:
    """
    Translate segments using Google Translate (free, via deep-translator).
    Preserves original timestamps and durations.
    Returns None if deep-translator not installed.

    Improvements vs v1.4:
      - Retry with exponential backoff on rate limit / connection error
        (GTRANS_MAX_RETRIES attempts, GTRANS_RETRY_DELAY seconds between)
      - XML tag separator robust against mangling (unchanged)
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return None

    lang_map = {
        "zh-cn": "zh-CN", "zh-tw": "zh-TW", "zh": "zh-CN", "jw": "jd",
    }
    src = lang_map.get(src_lang, src_lang)

    translated_segs = []
    total           = len(segments)
    t_start         = time.time()

    print(f"  🌐  Google Translate: {total} segments ({src} → en)…")

    for batch_start in range(0, total, GTRANS_BATCH_SIZE):
        batch = segments[batch_start : batch_start + GTRANS_BATCH_SIZE]
        texts = [s["text"] for s in batch]

        # Wrap each segment with XML tags + trailing space INSIDE the tag.
        # The trailing space ensures that even if Google strips spaces at tag
        # boundaries, the words from adjacent segments won't merge.
        # e.g. <s0>take the lemon </s0><s1>Take, take</s1>
        #                         ^-- this space prevents "lemonTake" merge
        tagged = "".join(f"<s{i}>{t} </s{i}>" for i, t in enumerate(texts))

        parts      = None
        last_error = None

        for attempt in range(GTRANS_MAX_RETRIES):
            try:
                translator = GoogleTranslator(source=src, target="en")
                translated = translator.translate(tagged)

                extracted = []
                for i in range(len(batch)):
                    m = re.search(rf"<s{i}>(.*?)</s{i}>", translated, re.DOTALL)
                    extracted.append(m.group(1).strip() if m else None)

                if any(p is None for p in extracted):
                    # Tags mangled — translate individually
                    extracted = []
                    for t in texts:
                        try:
                            extracted.append(
                                GoogleTranslator(source=src, target="en").translate(t)
                            )
                        except Exception:
                            extracted.append(t)

                parts = extracted
                break   # success — exit retry loop

            except Exception as e:
                last_error = e
                if attempt < GTRANS_MAX_RETRIES - 1:
                    wait = GTRANS_RETRY_DELAY * (attempt + 1)
                    print(f"\n  ⚠️  Translate attempt {attempt+1} failed ({e}) "
                          f"— retrying in {wait:.0f}s…")
                    time.sleep(wait)

        if parts is None:
            print(f"\n  ⚠️  All retries failed ({last_error}) — keeping original text")
            parts = texts

        for seg, eng in zip(batch, parts):
            raw = (eng or seg["text"]).strip()
            translated_segs.append({
                "text":     clean_translated_text(raw),
                "start":    seg["start"],
                "duration": seg.get("duration", 0),
            })

        done   = min(batch_start + GTRANS_BATCH_SIZE, total)
        filled = int(done / total * 30)
        bar    = "█" * filled + "░" * (30 - filled)
        print(f"  [{bar}] {done}/{total}", end="\r")

    print(f"\n  ✅  Google Translate done — {total} segs in {time.time()-t_start:.1f}s\n")
    return translated_segs


# ─────────────────────────────────────────────
# WHISPER MODEL — GPU-first, CPU fallback
# ─────────────────────────────────────────────

_whisper_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    """
    Load Whisper once and reuse. Probes GPU immediately after load to catch
    missing CUDA DLL errors (cublas64_12.dll etc.) inside the try/except.
    Falls back to CPU automatically.
    """
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    print(f"⚡  Loading Whisper '{WHISPER_MODEL_SIZE}' model…")
    print(f"  ℹ️  First run downloads weights — tiny≈75MB · small≈244MB · medium≈769MB")
    print(f"  ℹ️  Download progress appears below. Already cached = loads instantly.\n")

    try:
        import torch
        if torch.cuda.is_available():
            gpu        = torch.cuda.get_device_name(0)
            model_size = WHISPER_MODEL_SIZE if WHISPER_MODEL_SIZE != "auto" else "medium"
            print(f"  🎮  GPU detected: {gpu} — loading {model_size} float16…")
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            import numpy as np
            probe = np.zeros(16000, dtype=np.float32)
            list(model.transcribe(probe, beam_size=1)[0])
            _whisper_model = model
            print(f"\n  ✅  Whisper {model_size} loaded on GPU ({gpu}).\n")
            return _whisper_model
        else:
            print("  ℹ️  CUDA not available — using CPU.")
    except ImportError:
        print("  ℹ️  PyTorch not installed — using CPU.")
    except Exception as e:
        print(f"  ⚠️  GPU load/probe failed ({e}) — falling back to CPU.")

    # CPU path: use "small" by default (medium = 10-15 min per video on CPU)
    model_size = WHISPER_MODEL_SIZE if WHISPER_MODEL_SIZE != "auto" else "small"
    print(f"  💻  Loading Whisper {model_size} on CPU (INT8 · {WHISPER_CPU_THREADS} threads)…")
    print(f"  ℹ️  CPU tip: 'small' ~1-3 min/video. Set WHISPER_MODEL_SIZE=medium for better quality.")
    _whisper_model = WhisperModel(
        model_size, device="cpu",
        compute_type="int8", cpu_threads=WHISPER_CPU_THREADS,
    )
    print(f"\n  ✅  Whisper {model_size} loaded on CPU.\n")
    return _whisper_model


# ─────────────────────────────────────────────
# WHISPER TRANSCRIPTION
# ─────────────────────────────────────────────

def _run_whisper(audio_path: str, task: str = "transcribe") -> Tuple[List[dict], str]:
    """
    Run Whisper on an audio file. task = 'transcribe' or 'translate'.
    Returns (segments, detected_language).
    Segments include 'duration' field for tighter timestamp mapping.
    """
    model = get_whisper_model()

    segments_gen, info = model.transcribe(
        audio_path,
        task=task,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=True,
    )

    detected_lang = info.language
    print(f"  🌐  Language: '{detected_lang}' (confidence: {info.language_probability:.0%})\n")

    segments = []
    count    = 0
    t_start  = time.time()

    for seg in segments_gen:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({
            "text":     text,
            "start":    seg.start,
            "duration": seg.end - seg.start,
        })
        count += 1
        if count % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  ✏️  Seg {count:>4}  |  {seconds_to_timestamp(seg.start)}"
                  f"  |  {elapsed:.0f}s  |  …{text[:55]}")

    print(f"\n  ✅  {task.capitalize()} done — {count} segments in "
          f"{time.time() - t_start:.1f}s\n")
    return segments, detected_lang


def transcribe_via_whisper(url: str, ffmpeg_path: str) -> Tuple[List[dict], List[dict], str]:
    """Download audio and transcribe. Returns (original_segs, english_segs, language)."""
    print("⚠️  Cloud transcript unavailable — switching to local Whisper…")

    tmp_stem = "__yt_scraper_tmp__"
    tmp_mp3  = f"{tmp_stem}.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_stem,
        "ffmpeg_location": ffmpeg_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3", "preferredquality": "192",
        }],
        "quiet": True, "no_warnings": True,
        "http_headers":   YTDLP_HEADERS,
        "socket_timeout": 60,
    }

    print("📥  Downloading audio…")
    with _Spinner("Downloading via yt-dlp"):
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    if not os.path.exists(tmp_mp3):
        raise FileNotFoundError(f"Audio download failed — '{tmp_mp3}' not found.")
    print("  ✅  Audio downloaded.\n")

    print("🎙️  Transcribing (original language)…\n")
    original_segs, detected_lang = _run_whisper(tmp_mp3, task="transcribe")

    if detected_lang in ENGLISH_CODES or detected_lang == "en":
        print("  ℹ️  Source is English — skipping translation.\n")
        english_segs = original_segs
    else:
        print(f"🌍  Translating '{detected_lang}' → English…\n")
        english_segs = translate_segments_google(original_segs, detected_lang)
        if english_segs is None:
            print("  ℹ️  deep-translator not available — using Whisper translate…\n")
            english_segs, _ = _run_whisper(tmp_mp3, task="translate")

    if os.path.exists(tmp_mp3):
        os.remove(tmp_mp3)
        print("🗑️  Temporary audio deleted.")

    return original_segs, english_segs, detected_lang


# ─────────────────────────────────────────────
# TRANSLATION FOR CLOUD TRANSCRIPTS
# ─────────────────────────────────────────────

def _translate_cloud_segments(
    cloud_segs:  List[dict],
    language:    str,
    url:         str,
    ffmpeg_path: str,
    metadata:    dict,
) -> Tuple[List[dict], str]:
    """
    Translate cloud transcript to English.
    Priority: Google Translate → Whisper audio fallback.
    Returns (english_segments, translation_model_name).
    """
    print(f"  🌍  Translating cloud transcript ('{language}' → en)…\n")
    english_segs = translate_segments_google(cloud_segs, language)

    if english_segs is not None:
        return english_segs, "google-translate"

    print("  ℹ️  Google Translate unavailable — downloading audio for Whisper…\n")
    tmp_stem = "__yt_scraper_tmp__"
    tmp_mp3  = f"{tmp_stem}.mp3"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": tmp_stem,
        "ffmpeg_location": ffmpeg_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3", "preferredquality": "192",
        }],
        "quiet": True, "no_warnings": True,
        "http_headers":   YTDLP_HEADERS,
        "socket_timeout": 60,
    }
    print("📥  Downloading audio for Whisper translation…")
    with _Spinner("Downloading"):
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    if os.path.exists(tmp_mp3):
        whisper_segs, _ = _run_whisper(tmp_mp3, task="translate")
        os.remove(tmp_mp3)
        return whisper_segs, f"whisper-{WHISPER_MODEL_SIZE}"
    else:
        print("  ⚠️  Audio download failed — keeping original as English.")
        return cloud_segs, "none"


# ─────────────────────────────────────────────
# CHUNKING  v1.5
# ─────────────────────────────────────────────

def build_chunks(
    original_segs: List[dict],
    english_segs:  List[dict],
    metadata:      dict,
    language:      str,
) -> List[dict]:
    """
    Chunk transcript into ~CHUNK_MAX_CHARS blocks with timestamp-aligned English.

    Changes vs v1.4:
      - embed_text field removed — database.py embeds english_text directly,
        so the prefix-heavy embed_text was dead weight and misleading
      - segment 'duration' used for tighter end boundary on last segment
      - EMBEDDING_MODEL constant referenced instead of hardcoded string
    """
    ingested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    chapters    = metadata.get("chapters", [])
    duration_s  = float(metadata.get("duration_s", 0))

    def _get_chapter(start_s: float) -> str:
        for ch in reversed(chapters):
            if start_s >= ch["start_time"]:
                return ch["title"]
        return ""

    # ── Step 1: chunk original segments, record time ranges ──────────────
    orig_chunks: List[Tuple[str, float, float, int]] = []
    buffer       = ""
    chunk_start  = 0.0
    chunk_idx    = 0
    overlap_tail: List[dict] = []
    total        = len(original_segs)

    for i, seg in enumerate(original_segs):
        if not buffer:
            chunk_start = seg["start"]
            if overlap_tail:
                buffer = " ".join(s["text"] for s in overlap_tail) + " "

        buffer  += seg["text"] + " "
        is_last  = (i == total - 1)

        if len(buffer) >= CHUNK_MAX_CHARS or is_last:
            if i + 1 < total:
                chunk_end = original_segs[i + 1]["start"]
            else:
                # Use segment duration if available, else video duration
                seg_end   = seg["start"] + seg.get("duration", 0)
                chunk_end = max(seg_end, duration_s) if duration_s > 0 else seg_end

            orig_chunks.append((buffer.strip(), chunk_start, chunk_end, chunk_idx))
            chunk_idx   += 1
            overlap_tail = original_segs[max(0, i - CHUNK_OVERLAP_SEGS + 1) : i + 1]
            buffer       = ""
            chunk_start  = 0.0

    # ── Step 2: sort English segments by timestamp ────────────────────────
    eng_sorted         = sorted(english_segs, key=lambda s: s["start"])
    OVERLAP_LOOKBACK_S = CHUNK_OVERLAP_SEGS * 3.0

    def _strip_leading_fragment(text: str) -> str:
        """
        Drop a short leading fragment left over from the previous chunk.
        e.g. "Is. But these look like local lemons..." → "But these look..."
        Only drops if the text before the first sentence boundary is ≤ 5 words.
        """
        match = re.search(r'[.!?]\s+', text)
        if not match:
            return text
        first_sentence = text[:match.start()].strip()
        if len(first_sentence.split()) <= 5:
            return text[match.end():].strip()
        return text

    def _collect_english(start_s: float, end_s: float, is_first: bool) -> str:
        """
        Collect English segments whose timestamps fall in [look_from, end_s).
        Non-first chunks look back OVERLAP_LOOKBACK_S to include overlap text,
        then strip any leading fragment already covered by the previous chunk.
        Segments are joined with a space and cleaned to fix translation artifacts.
        """
        look_from = start_s if is_first else max(0.0, start_s - OVERLAP_LOOKBACK_S)
        matched   = [s["text"] for s in eng_sorted if look_from <= s["start"] < end_s]
        if matched:
            # Join with space — each segment may or may not end with punctuation
            # Add space between segments to prevent word merges at boundaries
            text = " ".join(t.rstrip() for t in matched).strip()
            text = clean_translated_text(text)
            return text if is_first else _strip_leading_fragment(text)
        if eng_sorted:
            closest = min(eng_sorted, key=lambda s: abs(s["start"] - start_s))
            return clean_translated_text(closest["text"].strip())
        return ""

    # ── Step 3: assemble chunks ───────────────────────────────────────────
    chunks    = []
    seen_hashes: set = set()

    for chunk_pos, (orig_text, start_s, end_s, cidx) in enumerate(orig_chunks):
        ts       = seconds_to_timestamp(start_s)
        eng_text = _collect_english(start_s, end_s, is_first=(chunk_pos == 0))
        chapter  = _get_chapter(start_s)

        # Hash on original text (always present) to avoid empty-string collisions
        hash_source = orig_text if orig_text else eng_text
        chash       = content_hash(hash_source)

        # Skip exact duplicate chunks (can occur if translation returns identical text)
        if chash in seen_hashes:
            continue
        seen_hashes.add(chash)

        chunks.append({
            "original_text": orig_text,
            "english_text":  eng_text,
            "metadata": {
                "source_type":       "youtube",
                "video_id":          metadata["video_id"],
                "source_url":        metadata["source_url"],
                "title":             metadata["title"],
                "channel":           metadata["channel"],
                "duration_s":        str(metadata["duration_s"]),
                "upload_date":       metadata.get("upload_date", ""),
                "view_count":        str(metadata.get("view_count", 0)),
                "language":          language,
                "translation_model": metadata.get("translation_model", "none"),
                "chunk_index":       str(cidx),
                "timestamp":         ts,
                "start_seconds":     str(round(start_s, 2)),
                "end_seconds":       str(round(end_s, 2)),
                "chapter":           chapter,
                "content_hash":      chash,
                "chunk_version":     CHUNK_VERSION,
                "embedding_model":   EMBEDDING_MODEL,
                "ingested_at":       ingested_at,
            },
        })

    return chunks


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def scrape_youtube(url: str) -> Tuple[dict, List[dict]]:
    """
    Main entry point. Scrape any supported video URL into structured chunks.
    Supports YouTube, Instagram Reels, Facebook, Twitter/X, Vimeo, etc.
    Returns (metadata, chunks). Raises ValueError or RuntimeError on failure.
    """
    video_id = extract_video_id(url)
    if not video_id and not is_supported_url(url):
        raise ValueError(f"Not a supported video URL: {url!r}")

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    # ── Metadata ──────────────────────────────────────────────────────────
    print("📋  Fetching video metadata…")
    with _Spinner("Contacting YouTube"):
        metadata = fetch_video_metadata(url, ffmpeg_path)

    print(f"  🎬  {metadata['title']}")
    print(f"  📺  {metadata['channel']}  •  "
          f"Duration: {seconds_to_timestamp(metadata['duration_s'])}")
    if metadata["chapters"]:
        print(f"  📑  {len(metadata['chapters'])} chapters detected")
    print()

    # ── Cloud transcript (YouTube only — other platforms go straight to Whisper) ──
    print("🔍  Checking for cloud transcript…")
    cloud_segs = fetch_cloud_transcript(video_id) if video_id else None

    if cloud_segs:
        print(f"  ✅  Cloud transcript: {len(cloud_segs)} segments.\n")

        language = detect_language(cloud_segs)

        if language not in ENGLISH_CODES and language != "en":
            english_segs, translation_model = _translate_cloud_segments(
                cloud_segs, language, url, ffmpeg_path, metadata
            )
            metadata["translation_model"] = translation_model
            original_segs = cloud_segs
        else:
            original_segs = cloud_segs
            english_segs  = cloud_segs
            metadata["translation_model"] = "none"

    else:
        # ── Whisper fallback ─────────────────────────────────────────────
        original_segs, english_segs, language = transcribe_via_whisper(url, ffmpeg_path)
        metadata["translation_model"] = (
            "google-translate" if language not in ENGLISH_CODES else "none"
        )

    metadata["language"] = language

    if not original_segs:
        raise RuntimeError("All transcript methods returned no data.")

    print("📦  Building chunks…")
    chunks = build_chunks(original_segs, english_segs, metadata, language)
    print(f"  ✅  {len(chunks)} chunks ready.\n")

    return metadata, chunks


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  🎬  YouTube Scraper  v1.6")
    print("=" * 55)

    url = input("\n🔗  Paste a video URL (YouTube / Instagram / Facebook / Twitter): ").strip()

    try:
        meta, chunks = scrape_youtube(url)
    except (ValueError, RuntimeError, FileNotFoundError) as err:
        print(f"\n❌  {err}")
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 55)
    print("  📊  RESULTS")
    print("=" * 55)
    print(f"  Title       : {meta['title']}")
    print(f"  Channel     : {meta['channel']}")
    print(f"  Language    : {meta.get('language', 'unknown')}")
    print(f"  Translation : {meta.get('translation_model', 'none')}")
    print(f"  Chunks      : {len(chunks)}")
    if meta.get("chapters"):
        print(f"  Chapters    : {len(meta['chapters'])}")
    print()

    # ── Show all chunks ───────────────────────────────────────────────────
    for i, c in enumerate(chunks):
        ch_label = f"  |  📑 {c['metadata']['chapter']}" if c['metadata'].get('chapter') else ""
        print(f"{'─' * 55}")
        print(f"  CHUNK {i}  |  {c['metadata']['timestamp']}{ch_label}")
        print(f"{'─' * 55}")
        print(f"  [Original]\n  {c['original_text']}")
        print()
        print(f"  [English]\n  {c['english_text']}")
        print()

    # ── Export option — Enter defaults to n ──────────────────────────────
    print("=" * 55)
    export = input("💾  Export full data to JSON? (y/N): ").strip().lower()
    if export == "y":
        safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", meta["title"][:40])
        filename   = f"{meta['video_id']}_{safe_title}.json"
        output     = {
            "metadata": {k: v for k, v in meta.items() if k != "chapters"},
            "chapters": meta.get("chapters", []),
            "chunks":   chunks,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  ✅  Saved to '{filename}'")