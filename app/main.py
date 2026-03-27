from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter, ImageEnhance
import tempfile
import os
import httpx
import subprocess
import yt_dlp
import uuid
import json
import asyncio
import hashlib
import time
import re
from io import BytesIO
from difflib import SequenceMatcher

import pytesseract

app = FastAPI(title="Auction Worker")

APP_VERSION = "async-v17"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES")

DEEPGRAM_URL = (
    "https://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&language=pt-BR"
    "&smart_format=true"
    "&utterances=true"
    "&punctuate=true"
    "&numerals=true"
)

JOB_DIR = Path("/tmp/jobs")
JOB_DIR.mkdir(exist_ok=True)

FRAME_DIR = Path("/tmp/frames")
FRAME_DIR.mkdir(exist_ok=True)

VIDEO_CACHE_DIR = Path("/tmp/video-cache")
VIDEO_CACHE_DIR.mkdir(exist_ok=True)

ROOT_DIR = Path("/app")
FALLBACK_COOKIES_FILE = ROOT_DIR / "cookies.txt"
DENO_PATH = Path("/usr/local/bin/deno")
PRE_BOUNDARY_OFFSETS = [1.5, 0.5]
PANEL_CATEGORIES = [
    "Bezerro",
    "Garrote",
    "Vaca",
    "Boi magro",
    "Toruno",
    "Novilha",
    "Bezerra fêmea",
    "Vaca prenha",
    "Vaca parida",
    "Cruz industrial",
    "Macho(s) Nelore",
    "Macho(s) Crz. Ind.",
    "Macho(s) Nelore e Crz Ind",
    "Macho(s) Anel",
    "Macho(s) Tricross",
    "Macho(s) Touruno",
    "Macho(s) Crz Ind",
    "Nelore",
    "Macho(s) Cruzado",
    "Anel",
    "Cruzado",
    "Touruno",
]
PANEL_LAYOUT_TEMPLATES = {
    "correa_green_bar_v1": {
        "lot": (0.00, 0.74, 0.17, 0.995),
        "info": (0.15, 0.78, 0.68, 0.95),
        "price": (0.67, 0.75, 0.995, 0.95),
        "price_focus": (0.76, 0.75, 0.995, 0.95),
        "price_top": (0.73, 0.74, 0.995, 0.885),
    },
    "generic_bottom_bar_v1": {
        "lot": (0.00, 0.72, 0.19, 0.995),
        "info": (0.18, 0.76, 0.70, 0.95),
        "price": (0.68, 0.73, 0.995, 0.95),
        "price_focus": (0.77, 0.73, 0.995, 0.95),
        "price_top": (0.73, 0.72, 0.995, 0.88),
    },
}


class TranscriptRequest(BaseModel):
    video_url: str
    video_id: str
    job_id: str | None = None


class FrameRequest(BaseModel):
    video_url: str
    timestamp: float
    video_id: str | None = None
    worker_job_id: str | None = None


class FrameBatchRequest(BaseModel):
    video_url: str
    video_id: str | None = None
    worker_job_id: str | None = None
    start_time: float = 0
    end_time: float | None = None
    interval_seconds: float = 15
    max_frames: int = 80


class FrameBoundaryRequest(BaseModel):
    video_url: str
    video_id: str | None = None
    worker_job_id: str | None = None
    start_time: float = 0
    end_time: float | None = None
    interval_seconds: float = 8
    max_frames: int = 300
    full_diff_threshold: int = 10
    focus_diff_threshold: int = 12
    min_gap_seconds: float = 20
    focus_left: float = 0.45
    focus_top: float = 0.10
    focus_right: float = 0.98
    focus_bottom: float = 0.90


class PanelOcrSupportFrame(BaseModel):
    frame_file: str | None = None
    frame_url: str | None = None
    timestamp: float | None = None
    offset_seconds: float | None = None
    label: str | None = None


class PanelOcrRequest(BaseModel):
    frame_file: str | None = None
    frame_url: str | None = None
    timestamp: float | None = None
    support_frames: list[PanelOcrSupportFrame] = []
    categories: list[str] | None = None
    layout_hint: str | None = None


class PanelOcrBatchItem(BaseModel):
    frame_file: str | None = None
    frame_url: str | None = None
    timestamp: float | None = None
    support_frames: list[PanelOcrSupportFrame] = []
    categories: list[str] | None = None
    layout_hint: str | None = None
    metadata: dict | None = None


class PanelOcrBatchRequest(BaseModel):
    items: list[PanelOcrBatchItem]


def job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def save_job(job_id: str, data: dict):
    with open(job_path(job_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_job(job_id: str):
    path = job_path(job_id)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_cookies_file(output_dir: Path) -> Path | None:
    if YOUTUBE_COOKIES and YOUTUBE_COOKIES.strip():
        cookie_file = output_dir / "cookies.txt"
        cookie_file.write_text(YOUTUBE_COOKIES, encoding="utf-8")
        return cookie_file

    if FALLBACK_COOKIES_FILE.exists():
        return FALLBACK_COOKIES_FILE

    return None


def build_ydl_opts(output_template: str | None, cookie_file: Path | None, format_selector: str) -> dict:
    ydl_opts = {
        "format": format_selector,
        "quiet": True,
        "noplaylist": True,
        "restrictfilenames": True,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["default"]
            }
        },
        "remote_components": {"ejs:github"},
    }

    if output_template:
        ydl_opts["outtmpl"] = output_template

    if DENO_PATH.exists():
        ydl_opts["js_runtimes"] = {
            "deno": {
                "path": str(DENO_PATH)
            }
        }

    if cookie_file:
        ydl_opts["cookiefile"] = str(cookie_file)

    return ydl_opts


def find_downloaded_file(output_dir: Path, prefix: str) -> Path | None:
    candidates = [
        p for p in output_dir.iterdir()
        if p.is_file() and p.name.startswith(prefix)
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def get_visual_cache_key(video_url: str, video_id: str | None) -> str:
    if video_id and video_id.strip():
        return video_id.strip()
    return hashlib.sha1(video_url.encode("utf-8")).hexdigest()[:16]


def wait_for_cached_file(file_prefix: str, lock_path: Path, timeout_seconds: int = 600) -> Path | None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        cached = find_downloaded_file(VIDEO_CACHE_DIR, file_prefix)
        if cached and cached.exists():
            return cached
        if not lock_path.exists():
            break
        time.sleep(1)
    return find_downloaded_file(VIDEO_CACHE_DIR, file_prefix)


def download_visual_video(video_url: str, video_id: str | None) -> Path:
    cache_key = get_visual_cache_key(video_url, video_id)
    file_prefix = f"visual_{cache_key}."
    lock_path = VIDEO_CACHE_DIR / f"visual_{cache_key}.lock"

    cached = find_downloaded_file(VIDEO_CACHE_DIR, file_prefix)
    if cached and cached.exists():
        return cached

    lock_fd = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        output_template = str(VIDEO_CACHE_DIR / f"{file_prefix}%(ext)s")
        cookie_file = write_cookies_file(VIDEO_CACHE_DIR)

        format_attempts = [
            "bestvideo[height<=360][ext=mp4]/best[height<=360][ext=mp4]/bestvideo[height<=360]/best[height<=360]",
            "worstvideo[ext=mp4]/worst[ext=mp4]/worstvideo/worst",
            "bestvideo/best",
        ]

        errors = []

        for format_selector in format_attempts:
            try:
                ydl_opts = build_ydl_opts(output_template, cookie_file, format_selector)
                ydl_opts["overwrites"] = False
                ydl_opts["quiet"] = True
                ydl_opts["noprogress"] = True

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])

                downloaded = find_downloaded_file(VIDEO_CACHE_DIR, file_prefix)
                if downloaded and downloaded.exists():
                    return downloaded

                errors.append(f"{format_selector}: download sem arquivo gerado")
            except Exception as e:
                errors.append(f"{format_selector}: {str(e)}")

        raise RuntimeError(
            "Erro ao baixar vídeo para amostragem visual. Tentativas: " + " | ".join(errors)
        )
    except FileExistsError:
        waited = wait_for_cached_file(file_prefix, lock_path)
        if waited and waited.exists():
            return waited
        raise RuntimeError("Timeout aguardando vídeo em cache para amostragem visual.")
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass


def download_audio(video_url: str, output_dir: Path) -> Path:
    output_template = str(output_dir / "audio.%(ext)s")
    cookie_file = write_cookies_file(output_dir)

    format_attempts = [
        "bestaudio/best",
        "bestaudio*",
        "best",
    ]

    errors = []

    for format_selector in format_attempts:
        try:
            ydl_opts = build_ydl_opts(output_template, cookie_file, format_selector)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            downloaded = find_downloaded_file(output_dir, "audio.")
            if downloaded and downloaded.exists():
                return downloaded

            errors.append(f"{format_selector}: download sem arquivo gerado")
        except Exception as e:
            errors.append(f"{format_selector}: {str(e)}")

    raise RuntimeError(
        "Erro ao baixar áudio com yt-dlp. Tentativas: " + " | ".join(errors)
    )


def convert_to_wav(input_file: Path, output_dir: Path) -> Path:
    output_file = output_dir / "audio.wav"

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(output_file),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Erro ffmpeg: {result.stderr}")

    if not output_file.exists():
        raise RuntimeError("Falha na conversão para WAV.")

    return output_file


def get_video_stream_url(video_url: str, output_dir: Path) -> str:
    """
    Usa yt-dlp para resolver a URL real do stream de vídeo.
    Isso evita passar a página do YouTube diretamente para o ffmpeg.
    """
    cookie_file = write_cookies_file(output_dir)

    format_attempts = [
        "bestvideo[ext=mp4]/best[ext=mp4]/bestvideo/best"
    ]

    errors = []

    for format_selector in format_attempts:
        try:
            ydl_opts = build_ydl_opts(None, cookie_file, format_selector)
            ydl_opts["skip_download"] = True

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

            # Caso venha formato único
            if info.get("url"):
                return info["url"]

            # Caso venha lista de formatos solicitados
            requested_formats = info.get("requested_formats") or []
            for fmt in requested_formats:
                if fmt.get("url"):
                    return fmt["url"]

            # Fallback para formatos disponíveis
            formats = info.get("formats") or []
            for fmt in reversed(formats):
                if fmt.get("vcodec") != "none" and fmt.get("url"):
                    return fmt["url"]

            errors.append(f"{format_selector}: nenhuma stream URL encontrada")
        except Exception as e:
            errors.append(f"{format_selector}: {str(e)}")

    raise RuntimeError(
        "Erro ao resolver stream de vídeo com yt-dlp. Tentativas: " + " | ".join(errors)
    )


def extract_frame_from_stream(stream_url: str, timestamp: float, frame_path: Path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            stream_url,
            "-threads",
            "1",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-loglevel",
            "error",
            str(frame_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Erro ffmpeg ao extrair frame: {result.stderr}")

    if not frame_path.exists():
        raise RuntimeError("Frame não foi gerado.")


def get_stream_duration_seconds(stream_url: str) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            stream_url,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    try:
        return float((result.stdout or "").strip())
    except Exception:
        return None


def build_frame_timestamps(
    *,
    duration_seconds: float | None,
    start_time: float,
    end_time: float | None,
    interval_seconds: float,
    max_frames: int,
) -> list[float]:
    safe_interval = max(1.0, float(interval_seconds or 15))
    safe_max_frames = max(1, min(int(max_frames or 80), 300))
    safe_start = max(0.0, float(start_time or 0))

    effective_end = end_time
    if effective_end is None and duration_seconds is not None:
        effective_end = duration_seconds

    if effective_end is not None:
        effective_end = max(safe_start, float(effective_end))

    timestamps: list[float] = []
    current = safe_start

    while len(timestamps) < safe_max_frames:
        if effective_end is not None and current > effective_end:
            break
        timestamps.append(round(current, 2))
        current += safe_interval

    return timestamps


def build_public_frame_url(base_url: str, frame_name: str) -> str:
    return base_url.rstrip("/") + f"/frame/file/{frame_name}"


def unique_sorted_timestamps(values: list[float]) -> list[float]:
    seen = set()
    ordered: list[float] = []
    for value in sorted(values):
        rounded = round(float(value), 2)
        if rounded in seen:
            continue
        seen.add(rounded)
        ordered.append(rounded)
    return ordered


def build_preboundary_timestamps(
    *,
    boundary_timestamp: float,
    start_time: float,
    offsets: list[float] | None = None,
) -> list[float]:
    offsets = offsets or PRE_BOUNDARY_OFFSETS
    safe_boundary = max(0.0, float(boundary_timestamp or 0))
    safe_start = max(0.0, float(start_time or 0))
    timestamps = [
        safe_boundary - float(offset)
        for offset in offsets
        if (safe_boundary - float(offset)) >= safe_start
    ]

    if not timestamps and safe_boundary > safe_start:
        timestamps = [max(safe_start, safe_boundary - 0.5)]

    return unique_sorted_timestamps(timestamps)


def build_montage_image(
    frames: list[dict],
    montage_path: Path,
):
    if not frames:
        raise RuntimeError("Nenhum frame de suporte disponível para montar o painel de análise.")

    target_height = 220
    padding = 8
    caption_height = 22
    font = ImageFont.load_default()
    prepared: list[tuple[Image.Image, str, Path]] = []

    try:
        for frame in frames:
            frame_path = Path(frame["frame_path"])
            with Image.open(frame_path) as source:
                rgb = source.convert("RGB")
                ratio = target_height / max(1, rgb.height)
                width = max(1, int(rgb.width * ratio))
                resized = rgb.resize((width, target_height))
                prepared.append((resized, frame["label"], frame_path))

        total_width = sum(image.width for image, _, _ in prepared) + padding * (len(prepared) + 1)
        total_height = target_height + caption_height + (padding * 2)
        canvas = Image.new("RGB", (total_width, total_height), color=(245, 246, 248))
        draw = ImageDraw.Draw(canvas)

        x = padding
        for image, label, _ in prepared:
            canvas.paste(image, (x, padding + caption_height))
            draw.rectangle(
                [(x, padding), (x + image.width, padding + caption_height - 2)],
                fill=(32, 35, 42),
            )
            draw.text((x + 6, padding + 4), label, fill=(255, 255, 255), font=font)
            x += image.width + padding

        canvas.save(montage_path, format="JPEG", quality=92)
    finally:
        for image, _, _ in prepared:
            image.close()


def build_support_bundle(
    *,
    video_path: str,
    boundary_timestamp: float,
    start_time: float,
    request_base_url: str,
) -> dict:
    support_timestamps = build_preboundary_timestamps(
        boundary_timestamp=boundary_timestamp,
        start_time=start_time,
    )

    support_frames: list[dict] = []
    for timestamp in support_timestamps:
        frame_name = f"frame_{uuid.uuid4().hex}.jpg"
        frame_path = FRAME_DIR / frame_name
        extract_frame_from_stream(video_path, timestamp, frame_path)
        offset = round(float(boundary_timestamp) - float(timestamp), 2)
        support_frames.append({
            "timestamp": round(timestamp, 2),
            "offset_seconds": offset,
            "frame_file": frame_name,
            "frame_path": str(frame_path),
            "frame_url": build_public_frame_url(request_base_url, frame_name),
            "label": f"T-{offset:.1f}s",
        })
    if not support_frames:
        raise RuntimeError("Nenhum frame de suporte disponível para análise do lote.")

    support_frames.sort(key=lambda frame: frame["offset_seconds"])
    primary_frame = support_frames[0]
    fallback_frames = support_frames[1:]

    return {
        "support_frames": [
            {
                "timestamp": frame["timestamp"],
                "offset_seconds": frame["offset_seconds"],
                "frame_file": frame["frame_file"],
                "frame_url": frame["frame_url"],
                "label": frame["label"],
            }
            for frame in fallback_frames
        ],
        "analysis_frame_file": primary_frame["frame_file"],
        "analysis_frame_url": primary_frame["frame_url"],
        "analysis_timestamp": primary_frame["timestamp"],
        "analysis_offsets_seconds": [frame["offset_seconds"] for frame in support_frames],
        "primary_frame_file": primary_frame["frame_file"],
        "primary_frame_url": primary_frame["frame_url"],
        "primary_frame_timestamp": primary_frame["timestamp"],
        "support_frame_count": len(fallback_frames),
    }


def open_frame_image(frame_file: str | None, frame_url: str | None) -> Image.Image:
    if frame_file:
        local_path = FRAME_DIR / frame_file
        if local_path.exists():
            with Image.open(local_path) as image:
                return image.convert("RGB")

    if frame_url:
        response = httpx.get(frame_url, timeout=60)
        response.raise_for_status()
        with Image.open(BytesIO(response.content)) as image:
            return image.convert("RGB")

    raise RuntimeError("Nenhum frame_file ou frame_url válido foi informado para OCR.")


def crop_by_ratio(
    image: Image.Image,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> Image.Image:
    width, height = image.size
    box = (
        int(width * left),
        int(height * top),
        int(width * right),
        int(height * bottom),
    )
    return image.crop(box)


def average_region_rgb(
    image: Image.Image,
    left: float,
    top: float,
    right: float,
    bottom: float,
) -> tuple[float, float, float]:
    region = crop_by_ratio(image, left, top, right, bottom).convert("RGB")
    pixels = list(region.getdata())
    if not pixels:
        return (0.0, 0.0, 0.0)
    red = sum(pixel[0] for pixel in pixels) / len(pixels)
    green = sum(pixel[1] for pixel in pixels) / len(pixels)
    blue = sum(pixel[2] for pixel in pixels) / len(pixels)
    return (red, green, blue)


def detect_panel_layout(image: Image.Image, layout_hint: str | None = None) -> str:
    if layout_hint and layout_hint in PANEL_LAYOUT_TEMPLATES:
        return layout_hint

    bottom_band = average_region_rgb(image, 0.05, 0.79, 0.95, 0.95)
    left_badge = average_region_rgb(image, 0.00, 0.74, 0.16, 0.99)

    bottom_red, bottom_green, bottom_blue = bottom_band
    badge_red, badge_green, badge_blue = left_badge

    green_bar_detected = (
        bottom_green > (bottom_red + 18)
        and bottom_green > (bottom_blue + 10)
    )
    light_badge_detected = (
        badge_red > 170
        and badge_green > 170
        and badge_blue > 170
    )

    if green_bar_detected and light_badge_detected:
        return "correa_green_bar_v1"

    return "generic_bottom_bar_v1"


def enhance_for_ocr(image: Image.Image, *, scale: int = 3, threshold: int | None = None) -> Image.Image:
    grayscale = ImageOps.grayscale(image)
    grayscale = ImageOps.autocontrast(grayscale)
    grayscale = grayscale.resize((grayscale.width * scale, grayscale.height * scale))
    grayscale = grayscale.filter(ImageFilter.MedianFilter(size=3))
    grayscale = ImageEnhance.Contrast(grayscale).enhance(2.2)
    if threshold is not None:
        grayscale = grayscale.point(lambda p: 255 if p >= threshold else 0)
    return grayscale


def ocr_text_variants(image: Image.Image, configs: list[str], thresholds: list[int | None]) -> list[str]:
    values: list[str] = []
    for threshold in thresholds:
        prepared = enhance_for_ocr(image, threshold=threshold)
        for config in configs:
            try:
                text = pytesseract.image_to_string(prepared, lang="por", config=config)
            except pytesseract.TesseractNotFoundError as exc:
                raise RuntimeError("Tesseract OCR não está disponível no container.") from exc
            cleaned = " ".join((text or "").replace("\n", " ").split())
            if cleaned:
                values.append(cleaned)
    unique: list[str] = []
    seen = set()
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(value.strip())
    return unique


def parse_digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


def parse_money_number(text: str) -> float | None:
    normalized = text.upper().replace("R$", " ").replace("RS", " ").replace("S$", " ")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("O", "0")
    matches = re.findall(r"(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d{3,6})", normalized)
    for match in matches:
        candidate = match
        if "," in candidate and "." in candidate:
            candidate = candidate.replace(".", "").replace(",", ".")
        elif "," in candidate:
            candidate = candidate.replace(",", ".")
        else:
            candidate = candidate.replace(".", "")
        try:
            amount = float(candidate)
            if 100 <= amount <= 20000:
                return round(amount, 2)
        except Exception:
            continue

    compact_matches = re.findall(r"(\d{4,6})(\d{2})", normalized)
    for integer_part, decimal_part in compact_matches:
        try:
            amount = float(f"{integer_part}.{decimal_part}")
            if 100 <= amount <= 20000:
                return round(amount, 2)
        except Exception:
            continue
    return None


def parse_lot_value(texts: list[str]) -> str:
    for text in texts:
        digits = parse_digits(text)
        if 1 <= len(digits) <= 3:
            return digits.zfill(3)
    return ""


def parse_info_value(texts: list[str]) -> dict:
    best = {
        "quantidade_animais": "",
        "categoria_animal": "",
        "peso_medio_kg": "",
        "raw": "",
    }

    for text in texts:
        cleaned = text.replace("|", " ").replace("_", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        quantity_match = re.search(r"^(\d{1,3})\b", cleaned)
        weight_match = re.search(r"(\d{2,4})\s*kg\b", cleaned, flags=re.IGNORECASE)
        quantity = quantity_match.group(1) if quantity_match else ""
        weight = weight_match.group(1) if weight_match else ""

        category = ""
        if quantity_match and weight_match:
            category = cleaned[quantity_match.end():weight_match.start()]
        elif quantity_match:
            category = cleaned[quantity_match.end():]
        category = category.replace("-", " ").strip(" -")
        category = re.sub(r"\s+", " ", category)

        score = int(bool(quantity)) + int(bool(weight)) + int(bool(category))
        best_score = (
            int(bool(best["quantidade_animais"])) +
            int(bool(best["peso_medio_kg"])) +
            int(bool(best["categoria_animal"]))
        )
        if score > best_score:
            best = {
                "quantidade_animais": quantity,
                "categoria_animal": category,
                "peso_medio_kg": weight,
                "raw": cleaned,
            }

    return best


def normalize_category(value: str, categories: list[str]) -> str:
    candidate = re.sub(r"\s+", " ", (value or "").strip())
    if not candidate:
        return ""

    candidate_folded = (
        candidate.lower()
        .replace("ç", "c")
        .replace("ã", "a")
        .replace("á", "a")
        .replace("é", "e")
        .replace("ê", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("ú", "u")
    )

    best_match = ""
    best_score = 0.0
    for category in categories:
        category_folded = (
            category.lower()
            .replace("ç", "c")
            .replace("ã", "a")
            .replace("á", "a")
            .replace("é", "e")
            .replace("ê", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ô", "o")
            .replace("ú", "u")
        )
        score = SequenceMatcher(None, candidate_folded, category_folded).ratio()
        if score > best_score:
            best_match = category
            best_score = score

    if best_score >= 0.72:
        return best_match
    return candidate


def format_brl(value: float | None) -> str:
    if value is None:
        return ""
    whole = f"{value:,.2f}"
    whole = whole.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {whole}"


def format_decimal_brl(value: float | None) -> str:
    if value is None:
        return ""
    return f"R$ {value:.2f}".replace(".", ",")


def extract_panel_fields(
    image: Image.Image,
    categories: list[str],
    layout_hint: str | None = None,
) -> dict:
    layout_id = detect_panel_layout(image, layout_hint=layout_hint)
    template = PANEL_LAYOUT_TEMPLATES[layout_id]

    lot_crop = crop_by_ratio(image, *template["lot"])
    info_crop = crop_by_ratio(image, *template["info"])
    price_crop = crop_by_ratio(image, *template["price"])
    price_focus_crop = crop_by_ratio(image, *template["price_focus"])
    price_top_crop = crop_by_ratio(image, *template["price_top"])

    lot_texts = ocr_text_variants(
        lot_crop,
        configs=[
            "--psm 7 -c tessedit_char_whitelist=0123456789",
            "--psm 8 -c tessedit_char_whitelist=0123456789",
        ],
        thresholds=[None, 170, 200],
    )
    info_texts = ocr_text_variants(
        info_crop,
        configs=[
            "--psm 7",
            "--psm 6",
        ],
        thresholds=[None, 180, 210],
    )
    price_texts = ocr_text_variants(
        price_crop,
        configs=[
            "--psm 7",
            "--psm 6",
        ],
        thresholds=[None, 150, 185],
    )
    price_focus_texts = ocr_text_variants(
        price_focus_crop,
        configs=[
            "--psm 7",
            "--psm 8",
        ],
        thresholds=[None, 150, 185],
    )
    price_top_texts = ocr_text_variants(
        price_top_crop,
        configs=[
            "--psm 7 -c tessedit_char_whitelist=0123456789R$.,",
            "--psm 8 -c tessedit_char_whitelist=0123456789R$.,",
        ],
        thresholds=[None, 135, 155, 175, 195],
    )

    info_values = parse_info_value(info_texts)
    price_values = [parse_money_number(text) for text in (price_top_texts + price_focus_texts + price_texts)]
    price_values = [value for value in price_values if value is not None]
    price = max(price_values) if price_values else None
    weight_value = int(info_values["peso_medio_kg"]) if info_values["peso_medio_kg"].isdigit() else None
    price_per_kg = round(price / weight_value, 2) if price and weight_value else None

    return {
        "lote": parse_lot_value(lot_texts),
        "quantidade_animais": info_values["quantidade_animais"],
        "categoria_animal": normalize_category(info_values["categoria_animal"], categories),
        "peso_medio_kg": info_values["peso_medio_kg"],
        "preco_compra_rs": format_brl(price),
        "preco_kg_rs": format_decimal_brl(price_per_kg),
        "is_visual_candidate": bool(parse_lot_value(lot_texts)),
        "frame_kind": "lot_panel_deterministic_ocr",
        "observacao": "",
        "layout_id": layout_id,
        "_ocr_debug": {
            "layout_id": layout_id,
            "lot_texts": lot_texts[:3],
            "info_texts": info_texts[:3],
            "price_top_texts": price_top_texts[:3],
            "price_texts": price_texts[:3],
            "price_focus_texts": price_focus_texts[:3],
        },
    }


def choose_consensus(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def parse_currency_field(value: str) -> float | None:
    if not value:
        return None
    return parse_money_number(value)


def build_ocr_consensus(frame_results: list[dict]) -> dict:
    lot = choose_consensus([item["lote"] for item in frame_results])
    quantity = choose_consensus([item["quantidade_animais"] for item in frame_results])
    category = choose_consensus([item["categoria_animal"] for item in frame_results])
    weight = choose_consensus([item["peso_medio_kg"] for item in frame_results])

    valid_prices: list[float] = []
    for item in frame_results:
        amount = parse_currency_field(item["preco_compra_rs"])
        if amount is None:
            continue
        valid_prices.append(amount)
    best_price = max(valid_prices) if valid_prices else None
    layout_id = choose_consensus([item.get("layout_id", "") for item in frame_results])

    weight_value = int(weight) if weight.isdigit() else None
    price_per_kg = round(best_price / weight_value, 2) if best_price and weight_value else None

    populated_fields = sum(
        1 for value in [lot, quantity, category, weight, best_price]
        if value not in ("", None)
    )
    confidence = round(min(0.99, 0.45 + (0.11 * populated_fields)), 2) if lot else 0.0

    return {
        "status": "ok",
        "is_visual_candidate": bool(lot),
        "frame_kind": "lot_panel_deterministic_ocr",
        "lote": lot,
        "quantidade_animais": quantity,
        "categoria_animal": category,
        "peso_medio_kg": weight,
        "preco_compra_rs": format_brl(best_price),
        "preco_kg_rs": format_decimal_brl(price_per_kg),
        "comprador": "",
        "regiao_destino": "",
        "confidence": confidence,
        "observacao": f"ocr_consensus_frames={len(frame_results)}; layout_id={layout_id or 'desconhecido'}",
        "layout_id": layout_id,
        "frame_results": frame_results,
    }


async def run_panel_ocr(
    *,
    frame_file: str | None,
    frame_url: str | None,
    timestamp: float | None,
    support_frames: list[PanelOcrSupportFrame],
    categories: list[str] | None,
    layout_hint: str | None,
) -> dict:
    resolved_categories = categories or PANEL_CATEGORIES
    sources = [{
        "frame_file": frame_file,
        "frame_url": frame_url,
        "timestamp": timestamp,
        "label": "primary_frame",
    }]
    for frame in support_frames:
        sources.append({
            "frame_file": frame.frame_file,
            "frame_url": frame.frame_url,
            "timestamp": frame.timestamp,
            "label": frame.label or "support_frame",
        })

    frame_results: list[dict] = []
    for source in sources:
        if not source["frame_file"] and not source["frame_url"]:
            continue
        image = await asyncio.to_thread(
            open_frame_image,
            source["frame_file"],
            source["frame_url"],
        )
        try:
            extracted = await asyncio.to_thread(
                extract_panel_fields,
                image,
                resolved_categories,
                layout_hint,
            )
        finally:
            image.close()

        frame_results.append({
            "frame_file": source["frame_file"] or "",
            "frame_url": source["frame_url"] or "",
            "timestamp": source["timestamp"],
            "label": source["label"],
            **extracted,
        })

    consensus = build_ocr_consensus(frame_results)
    consensus["version"] = APP_VERSION
    return consensus


def average_hash(image: Image.Image, hash_size: int = 8) -> tuple[int, ...]:
    resized = image.convert("L").resize((hash_size, hash_size))
    pixels = list(resized.getdata())
    mean = sum(pixels) / max(1, len(pixels))
    return tuple(1 if pixel >= mean else 0 for pixel in pixels)


def hamming_distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(1 for left, right in zip(a, b) if left != right)


def clamp_region(
    width: int,
    height: int,
    left_ratio: float,
    top_ratio: float,
    right_ratio: float,
    bottom_ratio: float,
) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(width * left_ratio)))
    top = max(0, min(height - 1, int(height * top_ratio)))
    right = max(left + 1, min(width, int(width * right_ratio)))
    bottom = max(top + 1, min(height, int(height * bottom_ratio)))
    return left, top, right, bottom


def compute_frame_hashes(
    frame_path: Path,
    *,
    focus_left: float,
    focus_top: float,
    focus_right: float,
    focus_bottom: float,
) -> dict:
    with Image.open(frame_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        region = clamp_region(
            width,
            height,
            focus_left,
            focus_top,
            focus_right,
            focus_bottom,
        )
        focus = rgb.crop(region)
        return {
            "size": {"width": width, "height": height},
            "focus_region_pixels": {
                "left": region[0],
                "top": region[1],
                "right": region[2],
                "bottom": region[3],
            },
            "full_hash": average_hash(rgb),
            "focus_hash": average_hash(focus),
        }


async def transcribe_with_deepgram(audio_path: Path):
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY não configurada.")

    file_size = audio_path.stat().st_size

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
    }

    async with httpx.AsyncClient(timeout=600) as client:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        response = await client.post(
            DEEPGRAM_URL,
            headers=headers,
            content=audio_bytes,
        )

    if response.status_code >= 400:
        raise RuntimeError(
            f"Deepgram HTTP {response.status_code} | "
            f"size={file_size} bytes | "
            f"body={response.text}"
        )

    return response.json()


def normalize_segments(deepgram_response):
    results = deepgram_response.get("results", {})
    utterances = results.get("utterances", [])

    segments = []

    for utt in utterances:
        text = (utt.get("transcript") or "").strip()

        if not text:
            continue

        segments.append({
            "start": utt.get("start"),
            "end": utt.get("end"),
            "text": text,
        })

    return segments


async def process_job(job_id: str, video_url: str, video_id: str):
    try:
        save_job(job_id, {
            "job_id": job_id,
            "video_id": video_id,
            "status": "processing",
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            downloaded = await asyncio.to_thread(download_audio, video_url, tmp)
            wav = await asyncio.to_thread(convert_to_wav, downloaded, tmp)
            deepgram = await transcribe_with_deepgram(wav)
            segments = normalize_segments(deepgram)

            save_job(job_id, {
                "job_id": job_id,
                "video_id": video_id,
                "status": "finished",
                "has_transcript": len(segments) > 0,
                "downloaded_file": downloaded.name,
                "wav_file": wav.name,
                "segments": segments,
            })

    except Exception as e:
        save_job(job_id, {
            "job_id": job_id,
            "video_id": video_id,
            "status": "failed",
            "error": str(e),
        })


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "cookies_env_configured": bool(YOUTUBE_COOKIES and YOUTUBE_COOKIES.strip()),
        "cookies_file_exists": FALLBACK_COOKIES_FILE.exists(),
        "deno_exists": DENO_PATH.exists(),
        "deno_path": str(DENO_PATH),
    }


@app.post("/transcript/start")
async def transcript_start(payload: TranscriptRequest, background_tasks: BackgroundTasks):
    job_id = payload.job_id or str(uuid.uuid4())

    save_job(job_id, {
        "job_id": job_id,
        "video_id": payload.video_id,
        "status": "queued",
    })

    background_tasks.add_task(
        process_job,
        job_id,
        payload.video_url,
        payload.video_id,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "version": APP_VERSION,
    }


@app.get("/transcript/status/{job_id}")
async def transcript_status(job_id: str):
    job = load_job(job_id)

    if not job:
        return {
            "job_id": job_id,
            "status": "not_found",
            "version": APP_VERSION,
        }

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "version": APP_VERSION,
    }


@app.get("/transcript/result/{job_id}")
async def transcript_result(job_id: str):
    job = load_job(job_id)

    if not job:
        return {
            "job_id": job_id,
            "status": "not_found",
            "version": APP_VERSION,
        }

    job["version"] = APP_VERSION
    return job


@app.post("/frame/extract")
async def frame_extract(payload: FrameRequest, request: Request):
    try:
        video_path = await asyncio.to_thread(
            download_visual_video,
            payload.video_url,
            payload.video_id
        )

        frame_name = f"frame_{uuid.uuid4().hex}.jpg"
        frame_path = FRAME_DIR / frame_name

        await asyncio.to_thread(
            extract_frame_from_stream,
            str(video_path),
            payload.timestamp,
            frame_path
        )

        frame_url = str(request.base_url).rstrip("/") + f"/frame/file/{frame_name}"

        return {
            "status": "ok",
            "frame_file": frame_name,
            "frame_url": frame_url,
            "timestamp": payload.timestamp,
            "video_url": payload.video_url,
            "video_id": payload.video_id,
            "worker_job_id": payload.worker_job_id,
            "version": APP_VERSION,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "timestamp": payload.timestamp,
                "video_url": payload.video_url,
                "video_id": payload.video_id,
                "worker_job_id": payload.worker_job_id,
                "version": APP_VERSION,
            },
        )


@app.post("/frame/sample")
async def frame_sample(payload: FrameBatchRequest, request: Request):
    try:
        video_path = await asyncio.to_thread(
            download_visual_video,
            payload.video_url,
            payload.video_id
        )

        duration_seconds = await asyncio.to_thread(
            get_stream_duration_seconds,
            str(video_path)
        )

        timestamps = build_frame_timestamps(
            duration_seconds=duration_seconds,
            start_time=payload.start_time,
            end_time=payload.end_time,
            interval_seconds=payload.interval_seconds,
            max_frames=payload.max_frames,
        )

        frames = []
        for timestamp in timestamps:
            frame_name = f"frame_{uuid.uuid4().hex}.jpg"
            frame_path = FRAME_DIR / frame_name

            await asyncio.to_thread(
                extract_frame_from_stream,
                str(video_path),
                timestamp,
                frame_path
            )

            frame_url = str(request.base_url).rstrip("/") + f"/frame/file/{frame_name}"
            frames.append({
                "timestamp": timestamp,
                "frame_file": frame_name,
                "frame_url": frame_url,
            })

        return {
            "status": "ok",
            "video_url": payload.video_url,
            "video_id": payload.video_id,
            "worker_job_id": payload.worker_job_id,
            "duration_seconds": duration_seconds,
            "sampled_frames": len(frames),
            "frames": frames,
            "cached_video": video_path.name,
            "version": APP_VERSION,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "video_url": payload.video_url,
                "video_id": payload.video_id,
                "worker_job_id": payload.worker_job_id,
                "version": APP_VERSION,
            },
        )


@app.post("/frame/boundaries")
async def frame_boundaries(payload: FrameBoundaryRequest, request: Request):
    try:
        video_path = await asyncio.to_thread(
            download_visual_video,
            payload.video_url,
            payload.video_id
        )

        duration_seconds = await asyncio.to_thread(
            get_stream_duration_seconds,
            str(video_path)
        )

        timestamps = build_frame_timestamps(
            duration_seconds=duration_seconds,
            start_time=payload.start_time,
            end_time=payload.end_time,
            interval_seconds=payload.interval_seconds,
            max_frames=payload.max_frames,
        )

        request_base_url = str(request.base_url)
        scanned_frames = []
        boundary_candidates = []
        previous = None

        for timestamp in timestamps:
            frame_name = f"frame_{uuid.uuid4().hex}.jpg"
            frame_path = FRAME_DIR / frame_name

            await asyncio.to_thread(
                extract_frame_from_stream,
                str(video_path),
                timestamp,
                frame_path
            )

            hashes = await asyncio.to_thread(
                compute_frame_hashes,
                frame_path,
                focus_left=payload.focus_left,
                focus_top=payload.focus_top,
                focus_right=payload.focus_right,
                focus_bottom=payload.focus_bottom,
            )

            frame_url = build_public_frame_url(request_base_url, frame_name)

            full_diff = None
            focus_diff = None
            is_candidate = False
            reason = "initial_frame"
            score = 0

            if previous is not None:
                full_diff = hamming_distance(previous["hashes"]["full_hash"], hashes["full_hash"])
                focus_diff = hamming_distance(previous["hashes"]["focus_hash"], hashes["focus_hash"])
                score = max(full_diff, focus_diff)
                is_candidate = (
                    full_diff >= payload.full_diff_threshold
                    or focus_diff >= payload.focus_diff_threshold
                )
                reason = "visual_change" if is_candidate else "stable"
            else:
                is_candidate = False
                score = 0

            frame_info = {
                "timestamp": round(timestamp, 2),
                "frame_file": frame_name,
                "frame_url": frame_url,
                "full_diff": full_diff,
                "focus_diff": focus_diff,
                "change_score": score,
                "is_boundary_candidate": is_candidate,
                "reason": reason,
                "focus_region_pixels": hashes["focus_region_pixels"],
                "image_size": hashes["size"],
            }

            scanned_frames.append(frame_info)

            if is_candidate:
                analysis_frame = previous["frame"] if previous and reason == "visual_change" else frame_info
                candidate_info = {
                    **frame_info,
                    "boundary_type": "panel_disappeared_or_transition",
                    "analysis_timestamp": analysis_frame["timestamp"],
                    "analysis_frame_file": analysis_frame["frame_file"],
                    "analysis_frame_url": analysis_frame["frame_url"],
                    "boundary_timestamp": frame_info["timestamp"],
                    "boundary_frame_file": frame_info["frame_file"],
                    "boundary_frame_url": frame_info["frame_url"],
                    "analysis_single_frame_file": analysis_frame["frame_file"],
                    "analysis_single_frame_url": analysis_frame["frame_url"],
                }
                if (
                    boundary_candidates
                    and (timestamp - boundary_candidates[-1]["timestamp"]) < payload.min_gap_seconds
                ):
                    if score > (boundary_candidates[-1].get("change_score") or 0):
                        boundary_candidates[-1] = candidate_info
                else:
                    boundary_candidates.append(candidate_info)

            previous = {
                "hashes": hashes,
                "frame": frame_info,
            }

        enriched_candidates = []
        for candidate in boundary_candidates:
            enriched = dict(candidate)
            try:
                support_bundle = await asyncio.to_thread(
                    build_support_bundle,
                    video_path=str(video_path),
                    boundary_timestamp=float(candidate["boundary_timestamp"]),
                    start_time=float(payload.start_time or 0),
                    request_base_url=request_base_url,
                )
                enriched.update(support_bundle)
                enriched["analysis_mode"] = "pre_boundary_montage"
                enriched["support_frame_count"] = len(support_bundle["support_frames"])
            except Exception as e:
                enriched["analysis_mode"] = "single_frame_fallback"
                enriched["support_frames"] = []
                enriched["support_frame_count"] = 0
                enriched["support_generation_error"] = str(e)
            enriched_candidates.append(enriched)

        return {
            "status": "ok",
            "video_url": payload.video_url,
            "video_id": payload.video_id,
            "worker_job_id": payload.worker_job_id,
            "duration_seconds": duration_seconds,
            "sampled_frames": len(scanned_frames),
            "boundary_candidates": len(enriched_candidates),
            "candidates": enriched_candidates,
            "frames": scanned_frames,
            "cached_video": video_path.name,
            "version": APP_VERSION,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "video_url": payload.video_url,
                "video_id": payload.video_id,
                "worker_job_id": payload.worker_job_id,
                "version": APP_VERSION,
            },
        )


@app.post("/frame/panel-ocr")
async def frame_panel_ocr(payload: PanelOcrRequest):
    try:
        return await run_panel_ocr(
            frame_file=payload.frame_file,
            frame_url=payload.frame_url,
            timestamp=payload.timestamp,
            support_frames=payload.support_frames,
            categories=payload.categories,
            layout_hint=payload.layout_hint,
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "version": APP_VERSION,
            },
        )


@app.post("/frame/panel-ocr-batch")
async def frame_panel_ocr_batch(payload: PanelOcrBatchRequest):
    try:
        results: list[dict] = []
        for item in payload.items:
            try:
                consensus = await run_panel_ocr(
                    frame_file=item.frame_file,
                    frame_url=item.frame_url,
                    timestamp=item.timestamp,
                    support_frames=item.support_frames,
                    categories=item.categories,
                    layout_hint=item.layout_hint,
                )
                if item.metadata:
                    consensus["metadata"] = item.metadata
                results.append(consensus)
            except Exception as item_error:
                results.append({
                    "status": "failed",
                    "error": str(item_error),
                    "metadata": item.metadata or {},
                    "version": APP_VERSION,
                    "frame_results": [],
                })

        return {
            "status": "ok",
            "results": results,
            "count": len(results),
            "version": APP_VERSION,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error": str(e),
                "version": APP_VERSION,
            },
        )


@app.get("/frame/file/{filename}")
async def frame_file(filename: str):
    path = FRAME_DIR / filename
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "status": "not_found",
                "filename": filename,
                "version": APP_VERSION,
            },
        )
    return FileResponse(path)
