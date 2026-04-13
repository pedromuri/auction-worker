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
import unicodedata
from io import BytesIO
from collections import Counter
from difflib import SequenceMatcher

import pytesseract

app = FastAPI(title="Auction Worker")

APP_VERSION = "async-v41"

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
# The isolated price probe works better with a shorter, earlier tail:
# later frames are noisier while T-6s..T-3s usually contains the stable close price.
PRICE_PROBE_OFFSETS = [6.0, 5.0, 4.0, 3.0]
PRICE_PROBE_FALLBACK_OFFSETS = [60.0, 50.0, 40.0, 30.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0]
PRICE_PROBE_FORWARD_OFFSETS = [-4.0, -12.0, -20.0, -30.0, -40.0, -50.0, -60.0]
PRICE_PROBE_TIMESTAMP_JITTERS = [0.0, 0.25, -0.25]
PRICE_PROBE_PER_KG_MIN = 8.0
PRICE_PROBE_PER_KG_MAX = 23.5
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
PANEL_CATEGORY_PROFILES = {
    "default": PANEL_CATEGORIES,
    "correa_femeas_v1": [
        "Vaca",
        "Novilha",
        "Bezerra fêmea",
        "Vaca prenha",
        "Vaca parida",
        "Cruz industrial",
        "Anel",
        "Cruzado",
        "Nelore",
    ],
    "correa_machos_v1": [
        "Bezerro",
        "Garrote",
        "Boi magro",
        "Toruno",
        "Cruz industrial",
        "Macho(s) Nelore",
        "Macho(s) Crz. Ind.",
        "Macho(s) Nelore e Crz Ind",
        "Macho(s) Anel",
        "Macho(s) Tricross",
        "Macho(s) Touruno",
        "Macho(s) Crz Ind",
        "Macho(s) Cruzado",
        "Anel",
        "Cruzado",
        "Touruno",
        "Nelore",
    ],
}
PANEL_LAYOUT_TEMPLATES = {
    "correa_green_bar_v1": {
        "lot": (0.00, 0.74, 0.17, 0.995),
        "info": (0.15, 0.78, 0.68, 0.95),
        "info_detail": (0.17, 0.895, 0.66, 0.962),
        "price": (0.67, 0.75, 0.995, 0.95),
        "price_focus": (0.76, 0.75, 0.995, 0.95),
        "price_probe": (0.72, 0.745, 0.995, 0.885),
        "price_probe_focus": (0.80, 0.745, 0.995, 0.885),
        "price_probe_alt": (0.69, 0.72, 0.995, 0.90),
        "price_probe_alt_focus": (0.76, 0.72, 0.995, 0.90),
        "price_probe_alt2": (0.64, 0.70, 0.995, 0.91),
        "price_probe_alt2_focus": (0.72, 0.70, 0.995, 0.91),
    },
    "generic_bottom_bar_v1": {
        "lot": (0.00, 0.72, 0.19, 0.995),
        "info": (0.18, 0.76, 0.70, 0.95),
        "price": (0.68, 0.73, 0.995, 0.95),
        "price_focus": (0.77, 0.73, 0.995, 0.95),
        "price_probe": (0.73, 0.725, 0.995, 0.88),
        "price_probe_focus": (0.81, 0.725, 0.995, 0.88),
        "price_probe_alt": (0.70, 0.705, 0.995, 0.90),
        "price_probe_alt_focus": (0.78, 0.705, 0.995, 0.90),
        "price_probe_alt2": (0.66, 0.69, 0.995, 0.91),
        "price_probe_alt2_focus": (0.74, 0.69, 0.995, 0.91),
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
    include_support_bundle: bool = True
    include_frames: bool = True


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
    category_profile: str | None = None


class PanelOcrBatchItem(BaseModel):
    frame_file: str | None = None
    frame_url: str | None = None
    timestamp: float | None = None
    support_frames: list[PanelOcrSupportFrame] = []
    categories: list[str] | None = None
    layout_hint: str | None = None
    category_profile: str | None = None
    metadata: dict | None = None


class PanelOcrBatchRequest(BaseModel):
    items: list[PanelOcrBatchItem]


class PriceProbeItem(BaseModel):
    video_url: str
    video_id: str | None = None
    boundary_timestamp: float
    weight_hint: str | None = None
    layout_hint: str | None = None
    metadata: dict | None = None


class PriceProbeBatchRequest(BaseModel):
    items: list[PriceProbeItem]


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


def build_relative_timestamps(
    *,
    boundary_timestamp: float,
    offsets: list[float],
) -> list[float]:
    safe_boundary = max(0.0, float(boundary_timestamp or 0))
    timestamps = [max(0.0, safe_boundary - float(offset)) for offset in offsets]
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


def folded_text(value: str | None) -> str:
    candidate = (value or "").strip().lower()
    return (
        candidate
        .replace("ç", "c")
        .replace("ă", "a")
        .replace("á", "a")
        .replace("é", "e")
        .replace("ę", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ô", "o")
        .replace("ú", "u")
    )


def infer_panel_category_profile(
    explicit_profile: str | None = None,
    event_text: str | None = None,
    categories: list[str] | None = None,
) -> str:
    if explicit_profile and explicit_profile in PANEL_CATEGORY_PROFILES:
        return explicit_profile

    event_folded = folded_text(event_text)
    if "femea" in event_folded:
        return "correa_femeas_v1"
    if "macho" in event_folded:
        return "correa_machos_v1"

    if categories:
        category_folded = " ".join(folded_text(category) for category in categories)
        if "vaca" in category_folded or "novilha" in category_folded or "femea" in category_folded:
            return "correa_femeas_v1"
        if "macho" in category_folded or "garrote" in category_folded or "bezerro" in category_folded:
            return "correa_machos_v1"

    return "default"


def infer_panel_layout_hint(
    explicit_layout_hint: str | None = None,
    *,
    event_text: str | None = None,
    category_profile: str | None = None,
) -> str | None:
    if explicit_layout_hint and explicit_layout_hint in PANEL_LAYOUT_TEMPLATES:
        return explicit_layout_hint

    profile = (category_profile or "").strip()
    if profile.startswith("correa_"):
        return "correa_green_bar_v1"

    event_folded = folded_text(event_text)
    if "correa" in event_folded:
        return "correa_green_bar_v1"

    return None


def resolve_panel_categories(
    categories: list[str] | None,
    category_profile: str,
) -> list[str]:
    if category_profile and category_profile != "default":
        profile_categories = PANEL_CATEGORY_PROFILES.get(category_profile)
        if profile_categories:
            return profile_categories
    if categories:
        return categories
    return PANEL_CATEGORY_PROFILES.get(category_profile, PANEL_CATEGORIES)


def region_color_fraction(
    image: Image.Image,
    left: float,
    top: float,
    right: float,
    bottom: float,
    predicate,
) -> float:
    region = crop_by_ratio(image, left, top, right, bottom).convert("RGB")
    pixels = list(region.getdata())
    if not pixels:
        return 0.0
    matched = sum(1 for pixel in pixels if predicate(*pixel))
    return matched / len(pixels)


def assess_panel_visibility(image: Image.Image, layout_id: str) -> dict:
    template = PANEL_LAYOUT_TEMPLATES[layout_id]
    band_top = min(template["lot"][1], template["info"][1], template["price"][1])
    band_bottom = max(template["lot"][3], template["info"][3], template["price"][3])

    green_fraction = region_color_fraction(
        image,
        0.02,
        band_top,
        0.98,
        band_bottom,
        lambda r, g, b: g > (r + 22) and g > (b + 12) and g >= 70,
    )
    badge_bright_fraction = region_color_fraction(
        image,
        *template["lot"],
        lambda r, g, b: r >= 170 and g >= 170 and b >= 170,
    )
    yellow_fraction = region_color_fraction(
        image,
        *template.get("price_probe", template["price"]),
        lambda r, g, b: r >= 145 and g >= 115 and b <= 120 and r >= g,
    )

    panel_score = round((green_fraction * 3.0) + (badge_bright_fraction * 2.2) + (yellow_fraction * 1.4), 3)
    panel_visible = (
        (green_fraction >= 0.09 and badge_bright_fraction >= 0.05)
        or panel_score >= 0.42
    )

    return {
        "panel_visible": panel_visible,
        "panel_score": panel_score,
        "green_fraction": round(green_fraction, 4),
        "badge_bright_fraction": round(badge_bright_fraction, 4),
        "yellow_fraction": round(yellow_fraction, 4),
    }


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
    normalized = text.replace("R$", " ").replace("RS", " ").replace("S$", " ")
    normalized = normalized.replace(" ", "")
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
    return None


def enhance_price_probe_variants(image: Image.Image) -> list[Image.Image]:
    grayscale = ImageOps.grayscale(image)
    grayscale = ImageOps.autocontrast(grayscale)
    grayscale = grayscale.resize((grayscale.width * 4, grayscale.height * 4))
    grayscale = grayscale.filter(ImageFilter.MedianFilter(size=3))
    grayscale = ImageEnhance.Contrast(grayscale).enhance(3.0)
    grayscale = ImageEnhance.Sharpness(grayscale).enhance(2.0)

    variants = [grayscale]
    for threshold in (135, 155, 175, 195, 215):
        binary = grayscale.point(lambda p, t=threshold: 255 if p >= t else 0)
        variants.append(binary)
        variants.append(ImageOps.invert(binary))
    return variants


def extract_yellow_text_mask(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    mask = Image.new("L", rgb.size, 0)
    pixels_in = rgb.load()
    pixels_out = mask.load()
    width, height = rgb.size
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels_in[x, y]
            is_yellow = (
                red >= 150
                and green >= 125
                and blue <= 170
                and (red - blue) >= 25
                and (green - blue) >= 10
            )
            pixels_out[x, y] = 255 if is_yellow else 0
    mask = mask.resize((mask.width * 4, mask.height * 4))
    mask = mask.filter(ImageFilter.MedianFilter(size=3))
    mask = ImageEnhance.Contrast(mask).enhance(2.8)
    return mask


def extract_light_text_mask(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    mask = Image.new("L", rgb.size, 0)
    pixels_in = rgb.load()
    pixels_out = mask.load()
    width, height = rgb.size
    for y in range(height):
        for x in range(width):
            red, green, blue = pixels_in[x, y]
            is_light_text = (
                min(red, green, blue) >= 145
                and (max(red, green, blue) - min(red, green, blue)) <= 95
            )
            pixels_out[x, y] = 255 if is_light_text else 0
    mask = mask.resize((mask.width * 4, mask.height * 4))
    mask = mask.filter(ImageFilter.MedianFilter(size=3))
    mask = ImageEnhance.Contrast(mask).enhance(3.0)
    return mask


def ocr_info_detail_texts(image: Image.Image) -> list[str]:
    values: list[str] = []
    configs = [
        "--psm 7",
        "--psm 6",
    ]
    for threshold in (None, 150, 180, 210):
        prepared = enhance_for_ocr(image, threshold=threshold)
        for config in configs:
            try:
                text = pytesseract.image_to_string(prepared, lang="por", config=config)
            except pytesseract.TesseractNotFoundError as exc:
                raise RuntimeError("Tesseract OCR não está disponível no container.") from exc
            cleaned = " ".join((text or "").replace("\n", " ").split())
            if cleaned:
                values.append(cleaned)

    light_mask = extract_light_text_mask(image)
    for prepared in (light_mask, ImageOps.invert(light_mask)):
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


def ocr_price_probe_texts(image: Image.Image) -> list[str]:
    values: list[str] = []
    configs = [
        "--psm 7 -c tessedit_char_whitelist=0123456789.,",
        "--psm 8 -c tessedit_char_whitelist=0123456789.,",
        "--psm 13 -c tessedit_char_whitelist=0123456789.,",
    ]
    variants = enhance_price_probe_variants(image)
    yellow_mask = extract_yellow_text_mask(image)
    variants.append(yellow_mask)
    variants.append(ImageOps.invert(yellow_mask))
    for prepared in variants:
        for config in configs:
            try:
                text = pytesseract.image_to_string(prepared, lang="por", config=config)
            except pytesseract.TesseractNotFoundError as exc:
                raise RuntimeError("Tesseract OCR nÃ£o estÃ¡ disponÃ­vel no container.") from exc
            cleaned = " ".join((text or "").replace("\n", " ").split())
            if cleaned:
                values.append(cleaned)
    unique: list[str] = []
    seen = set()
    for value in values:
        cleaned = value.strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            unique.append(cleaned)
    return unique


def score_price_probe_text(text: str) -> float:
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return 0.0
    score = 1.0
    if re.fullmatch(r"\d{1,2}\.\d{3},\d{2}", cleaned):
        score += 5.0
    elif re.fullmatch(r"\d{4},\d{2}", cleaned):
        score += 4.0
    elif re.fullmatch(r"\d{4}", cleaned):
        score += 0.75
    elif re.fullmatch(r"\d{1,2}\.\d{3}", cleaned):
        score += 2.0
    if cleaned.startswith("7 "):
        score -= 1.0
    if len(re.findall(r"\d", cleaned)) > 7:
        score -= 1.5
    if any(marker in cleaned for marker in ("253", "533", "799", "252")):
        score -= 1.0
    return max(score, 0.25)


def extract_price_probe_fields(
    image: Image.Image,
    *,
    weight_hint: str | None = None,
    layout_hint: str | None = None,
) -> dict:
    layout_id = detect_panel_layout(image, layout_hint=layout_hint)
    template = PANEL_LAYOUT_TEMPLATES[layout_id]

    lot_crop = crop_by_ratio(image, *template["lot"])
    price_crop = crop_by_ratio(image, *template["price"])
    price_focus_crop = crop_by_ratio(image, *template["price_focus"])
    price_probe_crop = crop_by_ratio(image, *template.get("price_probe", template["price"]))
    price_probe_focus_crop = crop_by_ratio(image, *template.get("price_probe_focus", template["price_focus"]))
    price_probe_alt_crop = crop_by_ratio(image, *template.get("price_probe_alt", template.get("price_probe", template["price"])))
    price_probe_alt_focus_crop = crop_by_ratio(image, *template.get("price_probe_alt_focus", template.get("price_probe_focus", template["price_focus"])))
    price_probe_alt2_crop = crop_by_ratio(image, *template.get("price_probe_alt2", template.get("price_probe_alt", template.get("price_probe", template["price"]))))
    price_probe_alt2_focus_crop = crop_by_ratio(image, *template.get("price_probe_alt2_focus", template.get("price_probe_alt_focus", template.get("price_probe_focus", template["price_focus"]))))

    lot_texts = ocr_text_variants(
        lot_crop,
        configs=[
            "--psm 7 -c tessedit_char_whitelist=0123456789",
            "--psm 8 -c tessedit_char_whitelist=0123456789",
        ],
        thresholds=[None, 170, 200],
    )
    price_texts = ocr_price_probe_texts(price_crop)
    price_focus_texts = ocr_price_probe_texts(price_focus_crop)
    price_probe_texts = ocr_price_probe_texts(price_probe_crop)
    price_probe_focus_texts = ocr_price_probe_texts(price_probe_focus_crop)
    price_probe_alt_texts: list[str] = []
    price_probe_alt_focus_texts: list[str] = []
    price_probe_alt2_texts: list[str] = []
    price_probe_alt2_focus_texts: list[str] = []
    weight_value = int(weight_hint) if str(weight_hint or "").isdigit() else None

    def collect_candidates(
        source_groups: list[tuple[str, list[str], float]],
        candidate_scores: dict[float, float],
        candidate_counts: dict[float, int],
        candidate_sources: dict[float, set[str]],
    ) -> None:
        for source_name, texts, source_weight in source_groups:
            for text in texts:
                value = parse_money_number(text)
                if value is None:
                    continue
                if weight_value:
                    per_kg = value / weight_value
                    if not (PRICE_PROBE_PER_KG_MIN <= per_kg <= PRICE_PROBE_PER_KG_MAX):
                        continue
                value = round(value, 2)
                candidate_scores[value] = candidate_scores.get(value, 0.0) + (score_price_probe_text(text) * source_weight)
                candidate_counts[value] = candidate_counts.get(value, 0) + 1
                candidate_sources.setdefault(value, set()).add(source_name)

    candidate_scores: dict[float, float] = {}
    candidate_counts: dict[float, int] = {}
    candidate_sources: dict[float, set[str]] = {}
    collect_candidates([
        ("probe_focus", price_probe_focus_texts, 1.85),
        ("probe", price_probe_texts, 1.45),
        ("focus", price_focus_texts, 0.95),
        ("price", price_texts, 0.65),
    ], candidate_scores, candidate_counts, candidate_sources)

    used_alt_probe = False
    price_probe_alt_texts = ocr_price_probe_texts(price_probe_alt_crop)
    price_probe_alt_focus_texts = ocr_price_probe_texts(price_probe_alt_focus_crop)
    alt_candidate_scores: dict[float, float] = {}
    alt_candidate_counts: dict[float, int] = {}
    alt_candidate_sources: dict[float, set[str]] = {}
    collect_candidates([
        ("probe_alt_focus", price_probe_alt_focus_texts, 1.55),
        ("probe_alt", price_probe_alt_texts, 1.20),
        ("focus", price_focus_texts, 0.95),
        ("price", price_texts, 0.65),
    ], alt_candidate_scores, alt_candidate_counts, alt_candidate_sources)

    price_probe_alt2_texts = ocr_price_probe_texts(price_probe_alt2_crop)
    price_probe_alt2_focus_texts = ocr_price_probe_texts(price_probe_alt2_focus_crop)
    alt2_candidate_scores: dict[float, float] = {}
    alt2_candidate_counts: dict[float, int] = {}
    alt2_candidate_sources: dict[float, set[str]] = {}
    collect_candidates([
        ("probe_alt2_focus", price_probe_alt2_focus_texts, 1.45),
        ("probe_alt2", price_probe_alt2_texts, 1.10),
        ("probe_alt_focus", price_probe_alt_focus_texts, 1.00),
        ("probe_alt", price_probe_alt_texts, 0.85),
        ("focus", price_focus_texts, 0.95),
        ("price", price_texts, 0.65),
    ], alt2_candidate_scores, alt2_candidate_counts, alt2_candidate_sources)

    def best_candidate_tuple(
        scores: dict[float, float],
        counts: dict[float, int],
        sources: dict[float, set[str]],
    ) -> tuple[float, float, int, int] | None:
        if not scores:
            return None
        ranked = sorted(
            scores.items(),
            key=lambda item: (
                -item[1],
                -len(sources.get(item[0], set())),
                -counts.get(item[0], 0),
                item[0],
            ),
        )
        value, score = ranked[0]
        return (
            float(value),
            float(score),
            int(counts.get(value, 0)),
            len(sources.get(value, set())),
        )

    primary_best = best_candidate_tuple(candidate_scores, candidate_counts, candidate_sources)
    alt_best = best_candidate_tuple(alt_candidate_scores, alt_candidate_counts, alt_candidate_sources)
    alt2_best = best_candidate_tuple(alt2_candidate_scores, alt2_candidate_counts, alt2_candidate_sources)

    def merge_candidates(
        scores: dict[float, float],
        counts: dict[float, int],
        sources: dict[float, set[str]],
        scale: float = 1.0,
    ) -> None:
        nonlocal used_alt_probe
        if not scores:
            return
        used_alt_probe = True
        for value, score in scores.items():
            candidate_scores[value] = candidate_scores.get(value, 0.0) + (score * scale)
            candidate_counts[value] = candidate_counts.get(value, 0) + counts.get(value, 0)
            candidate_sources.setdefault(value, set()).update(sources.get(value, set()))

    if not candidate_scores:
        merge_candidates(alt_candidate_scores, alt_candidate_counts, alt_candidate_sources, scale=1.0)
    elif primary_best and alt_best:
        primary_value, _, primary_count, primary_source_count = primary_best
        alt_value, alt_score, alt_count, alt_source_count = alt_best
        if (
            alt_value > primary_value
            and (alt_value - primary_value) <= 900
            and alt_count >= max(1, primary_count - 1)
            and alt_source_count >= max(1, primary_source_count - 1)
        ) or (
            primary_value > (alt_value + 1400)
            and alt_count >= primary_count
            and alt_source_count >= primary_source_count
            and alt_score >= 0.75
        ):
            merge_candidates(alt_candidate_scores, alt_candidate_counts, alt_candidate_sources, scale=0.85)

    if not candidate_scores:
        merge_candidates(alt2_candidate_scores, alt2_candidate_counts, alt2_candidate_sources, scale=1.0)
    elif primary_best and alt2_best:
        primary_value, _, primary_count, primary_source_count = best_candidate_tuple(candidate_scores, candidate_counts, candidate_sources) or primary_best
        alt2_value, alt2_score, alt2_count, alt2_source_count = alt2_best
        if (
            alt2_value > primary_value
            and (alt2_value - primary_value) <= 900
            and alt2_count >= max(1, primary_count - 1)
        ) or (
            primary_value > (alt2_value + 1400)
            and alt2_count >= primary_count
            and alt2_source_count >= primary_source_count
            and alt2_score >= 0.75
        ):
            merge_candidates(alt2_candidate_scores, alt2_candidate_counts, alt2_candidate_sources, scale=0.70)

    best_value = None
    best_support = 0
    if candidate_scores:
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (
                -item[1],
                -len(candidate_sources.get(item[0], set())),
                -candidate_counts.get(item[0], 0),
                item[0],
            ),
        )
        best_value = ranked[0][0]
        best_support = candidate_counts.get(best_value, 0)
    return {
        "layout_id": layout_id,
        "lot_value": parse_lot_value(lot_texts),
        "price_value": best_value,
        "price_support": best_support,
        "lot_texts": lot_texts[:4],
        "price_texts": price_texts[:6],
        "price_focus_texts": price_focus_texts[:6],
        "price_probe_texts": price_probe_texts[:6],
        "price_probe_focus_texts": price_probe_focus_texts[:6],
        "price_probe_alt_texts": price_probe_alt_texts[:6],
        "price_probe_alt_focus_texts": price_probe_alt_focus_texts[:6],
        "price_probe_alt2_texts": price_probe_alt2_texts[:6],
        "price_probe_alt2_focus_texts": price_probe_alt2_focus_texts[:6],
        "used_alt_probe": used_alt_probe,
    }


def choose_price_probe_track(
    price_frames: list[dict],
    weight_value: int | None = None,
    lot_hint: str | None = None,
) -> float | None:
    def collect_valid_frames(enforce_lot_hint: bool) -> list[dict]:
        collected: list[dict] = []
        for frame in price_frames:
            value = frame.get("price_value")
            if value is None:
                continue
            if lot_hint and enforce_lot_hint:
                lot_value = str(frame.get("lot_value") or "").zfill(3)
                if lot_value and lot_value != str(lot_hint).zfill(3):
                    continue
            if weight_value:
                per_kg = value / weight_value
                if not (PRICE_PROBE_PER_KG_MIN <= per_kg <= PRICE_PROBE_PER_KG_MAX):
                    continue
            collected.append(frame)
        return collected

    valid_frames = collect_valid_frames(enforce_lot_hint=True)
    if not valid_frames and lot_hint:
        valid_frames = collect_valid_frames(enforce_lot_hint=False)

    if not valid_frames:
        return None

    # Build clusters from repeated visible values, then prefer the most stable late cluster.
    clustered: list[tuple[float, float, int, int, float, float]] = []
    used_frames: set[int] = set()
    ordered_frames = sorted(valid_frames, key=lambda item: float(item.get("timestamp") or 0))
    for frame in ordered_frames:
        frame_id = id(frame)
        if frame_id in used_frames:
            continue
        candidate = float(frame["price_value"])
        cluster_frames = [item for item in ordered_frames if abs(float(item["price_value"]) - candidate) <= 120]
        support = len(cluster_frames)
        if support < 2:
            continue
        for item in cluster_frames:
            used_frames.add(id(item))
        latest_frame = max(cluster_frames, key=lambda item: float(item.get("timestamp") or 0))
        latest_ts = float(latest_frame.get("timestamp") or 0)
        unique_timestamps = len({round(float(item.get("timestamp") or 0), 2) for item in cluster_frames})
        avg_frame_support = sum(float(item.get("price_support") or 1) for item in cluster_frames) / support
        avg_panel_score = sum(float(item.get("panel_score") or 0) for item in cluster_frames) / support
        cluster_value = float(latest_frame["price_value"])
        score = (
            (support * 240)
            + (unique_timestamps * 140)
            + (avg_frame_support * 120)
            + (avg_panel_score * 80)
            + (latest_ts / 10.0)
            + (min(cluster_value, 8000.0) / 200.0)
        )
        clustered.append((score, cluster_value, support, unique_timestamps, avg_frame_support, latest_ts))

    if clustered:
        clustered.sort(key=lambda item: (-item[0], -item[5], -item[1]))
        top_score, top_value, top_support, _, _, top_ts = clustered[0]
        if len(clustered) > 1:
            _, second_value, second_support, _, _, second_ts = clustered[1]
            if top_value > (second_value + 1000) and top_support <= second_support and top_ts <= (second_ts + 4):
                return second_value
            if top_value > (second_value + 1500) and top_support <= (second_support + 1) and top_ts <= (second_ts + 8):
                return second_value
            if second_value > (top_value + 250) and second_value <= (top_value + 800) and second_support >= (top_support - 1) and second_ts >= (top_ts - 12):
                return second_value
        later_lower_frames = [
            frame for frame in ordered_frames
            if float(frame.get("timestamp") or 0) > (top_ts + 4)
            and float(frame.get("price_value") or 0) < (top_value - 800)
        ]
        later_lower_values = sorted({float(frame.get("price_value") or 0) for frame in later_lower_frames})
        if len(later_lower_values) >= 3:
            max_lower = later_lower_values[-1]
            if max_lower >= max(100.0, top_value - 2800):
                return max_lower
        return top_value

    valid_frames.sort(key=lambda item: float(item.get("timestamp") or 0))
    tail = valid_frames[-5:] if len(valid_frames) > 5 else valid_frames

    scored: list[tuple[float, float, float, float]] = []
    for index, frame in enumerate(tail):
        candidate = float(frame["price_value"])
        frame_support = int(frame.get("price_support") or 1)
        support = sum(1 for item in valid_frames if abs(float(item["price_value"]) - candidate) <= 120)
        tail_support = sum(1 for item in tail if abs(float(item["price_value"]) - candidate) <= 120)
        later_frames = tail[index + 1:]
        future_penalty = sum(1 for item in later_frames if float(item["price_value"]) < (candidate - 120))
        recency_rank = index + 1
        score = (
            (frame_support * 260)
            + (tail_support * 150)
            + (support * 70)
            + (recency_rank * 5)
            - (future_penalty * 160)
        )
        scored.append((score, candidate, frame_support, recency_rank))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1] if scored else None


def parse_lot_value(texts: list[str], category_profile: str = "default") -> str:
    candidate_scores: dict[str, int] = {}
    for text in texts:
        digits = parse_digits(text)
        if not (1 <= len(digits) <= 3):
            continue
        normalized = digits.lstrip("0") or "0"
        if len(normalized) > 3 or normalized == "0":
            continue
        candidate_scores[normalized] = candidate_scores.get(normalized, 0) + 1

    if not candidate_scores:
        return ""

    ranked_candidates = sorted(
        candidate_scores.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    best_candidate, best_score = ranked_candidates[0]

    # In Correa female lots, OCR sometimes prepends a noisy leading digit
    # (for example 505 instead of 05). Prefer the shorter suffix variant when
    # it has nearly the same support across OCR variants.
    if category_profile == "correa_femeas_v1" and len(best_candidate) == 3:
        suffix_options = []
        for candidate, score in ranked_candidates[1:]:
            if len(candidate) >= len(best_candidate):
                continue
            if best_candidate.endswith(candidate.zfill(2)) or best_candidate.endswith(candidate):
                suffix_options.append((candidate, score))
        if suffix_options:
            suffix_candidate, suffix_score = sorted(
                suffix_options,
                key=lambda item: (-item[1], len(item[0]), item[0]),
            )[0]
            if suffix_score >= (best_score - 1):
                return suffix_candidate.zfill(3)

    return best_candidate.zfill(3)


def parse_info_value(texts: list[str]) -> dict:
    best = {
        "quantidade_animais": "",
        "categoria_animal": "",
        "peso_medio_kg": "",
        "composicao_lote": "",
        "bezerros_femeas": "",
        "bezerros_machos": "",
        "paridas": "",
        "solteiras": "",
        "prenhas": "",
        "raw": "",
    }

    for text in texts:
        cleaned = text.replace("|", " / ").replace("_", " ")
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

        composition = ""
        if weight_match:
            composition = cleaned[weight_match.end():]
        composition = composition.strip(" -/")
        composition = re.sub(r"\s+", " ", composition)

        standalone_composition = (
            not composition
            and not quantity
            and not weight
            and bool(
                re.search(r"\bPARIDAS?\b|\bSOLTEIRAS?\b|\bPRENHAS?\b", cleaned, flags=re.IGNORECASE)
                or re.search(r"\b\d{1,2}\s*[FM]\b", cleaned, flags=re.IGNORECASE)
            )
        )
        if standalone_composition:
            composition = cleaned

        bezerros_femeas_match = re.search(r"\b(\d{1,2})\s*F\b", composition, flags=re.IGNORECASE)
        bezerros_machos_match = re.search(r"\b(\d{1,2})\s*M\b", composition, flags=re.IGNORECASE)
        paridas_match = re.search(r"\b(\d{1,2})\s*PARIDAS?\b", composition, flags=re.IGNORECASE)
        solteiras_match = re.search(r"\b(\d{1,2})\s*SOLTEIRAS?\b", composition, flags=re.IGNORECASE)
        prenhas_match = re.search(r"\b(\d{1,2})\s*PRENHAS?\b", composition, flags=re.IGNORECASE)

        composition_fields = {
            "composicao_lote": composition,
            "bezerros_femeas": bezerros_femeas_match.group(1) if bezerros_femeas_match else "",
            "bezerros_machos": bezerros_machos_match.group(1) if bezerros_machos_match else "",
            "paridas": paridas_match.group(1) if paridas_match else "",
            "solteiras": solteiras_match.group(1) if solteiras_match else "",
            "prenhas": prenhas_match.group(1) if prenhas_match else "",
        }

        score = (
            int(bool(quantity)) +
            int(bool(weight)) +
            int(bool(category)) +
            int(bool(composition_fields["composicao_lote"]))
        )
        best_score = (
            int(bool(best["quantidade_animais"])) +
            int(bool(best["peso_medio_kg"])) +
            int(bool(best["categoria_animal"])) +
            int(bool(best["composicao_lote"]))
        )
        if score > best_score:
            best = {
                "quantidade_animais": quantity,
                "categoria_animal": category,
                "peso_medio_kg": weight,
                "composicao_lote": composition_fields["composicao_lote"],
                "bezerros_femeas": composition_fields["bezerros_femeas"],
                "bezerros_machos": composition_fields["bezerros_machos"],
                "paridas": composition_fields["paridas"],
                "solteiras": composition_fields["solteiras"],
                "prenhas": composition_fields["prenhas"],
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


def canonical_category_value(value: str) -> str:
    folded = folded_text(value)
    folded = unicodedata.normalize("NFKD", folded).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", folded).strip()


def normalize_category(
    value: str,
    categories: list[str],
    category_profile: str = "default",
) -> str:
    candidate = re.sub(r"\s+", " ", (value or "").strip())
    if not candidate:
        return ""

    candidate_folded = canonical_category_value(candidate)
    if not candidate_folded:
        return ""

    if category_profile == "correa_femeas_v1":
        if "prenha" in candidate_folded:
            return "Vaca prenha"
        if "parida" in candidate_folded:
            return "Vaca parida"
        if "novilh" in candidate_folded:
            return "Novilha"
        if "bezer" in candidate_folded:
            return "Bezerra femea"
        if "anel" in candidate_folded:
            return "Anel"
        if "cruz" in candidate_folded or "crz" in candidate_folded or "industrial" in candidate_folded or "tricross" in candidate_folded:
            return "Cruz industrial"
        if "touruno" in candidate_folded or "toruno" in candidate_folded:
            return "Cruzado"
        if "nelore" in candidate_folded:
            return "Nelore"
        if "vaca" in candidate_folded or "boi" in candidate_folded or "garrote" in candidate_folded:
            return "Vaca"

    best_match = ""
    best_score = 0.0
    for category in categories:
        category_folded = canonical_category_value(category)
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
    category_profile: str = "default",
) -> dict:
    layout_id = detect_panel_layout(image, layout_hint=layout_hint)
    template = PANEL_LAYOUT_TEMPLATES[layout_id]

    lot_crop = crop_by_ratio(image, *template["lot"])
    info_crop = crop_by_ratio(image, *template["info"])
    info_detail_crop = crop_by_ratio(image, *template["info_detail"]) if template.get("info_detail") else None
    price_crop = crop_by_ratio(image, *template["price"])
    price_focus_crop = crop_by_ratio(image, *template["price_focus"])

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
    info_detail_texts: list[str] = []
    if info_detail_crop is not None:
        info_detail_texts = ocr_info_detail_texts(info_detail_crop)
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

    info_values = parse_info_value(info_texts)
    info_detail_values = parse_info_value(info_detail_texts)
    for field in (
        "composicao_lote",
        "bezerros_femeas",
        "bezerros_machos",
        "paridas",
        "solteiras",
        "prenhas",
    ):
        if info_detail_values.get(field):
            info_values[field] = info_detail_values[field]
    price_values = [parse_money_number(text) for text in (price_focus_texts + price_texts)]
    price_values = [value for value in price_values if value is not None]
    price = max(price_values) if price_values else None
    weight_value = int(info_values["peso_medio_kg"]) if info_values["peso_medio_kg"].isdigit() else None
    price_per_kg = round(price / weight_value, 2) if price and weight_value else None

    lote_value = parse_lot_value(lot_texts, category_profile=category_profile)

    return {
        "lote": lote_value,
        "quantidade_animais": info_values["quantidade_animais"],
        "categoria_animal": normalize_category(info_values["categoria_animal"], categories, category_profile),
        "peso_medio_kg": info_values["peso_medio_kg"],
        "composicao_lote": info_values["composicao_lote"],
        "bezerros_femeas": info_values["bezerros_femeas"],
        "bezerros_machos": info_values["bezerros_machos"],
        "paridas": info_values["paridas"],
        "solteiras": info_values["solteiras"],
        "prenhas": info_values["prenhas"],
        "preco_compra_rs": format_brl(price),
        "preco_kg_rs": format_decimal_brl(price_per_kg),
        "is_visual_candidate": bool(lote_value),
        "frame_kind": "lot_panel_deterministic_ocr",
        "observacao": "",
        "layout_id": layout_id,
        "_ocr_debug": {
            "layout_id": layout_id,
            "lot_texts": lot_texts[:3],
            "info_texts": info_texts[:3],
            "info_detail_texts": info_detail_texts[:5],
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


def post_validate_consensus_lot(
    lot: str,
    *,
    category_profile: str = "default",
) -> str:
    if not lot:
        return ""
    if category_profile == "correa_femeas_v1":
        try:
            lot_number = int(lot)
        except ValueError:
            return lot
        # In the current Correa female family, OCR sometimes hallucinates
        # inflated 3-digit lots such as 205/505/935. Keep the family-wide
        # validation conservative and reject these outliers from the visual
        # consensus so they do not become false positives downstream.
        if lot_number > 150:
            return ""
    return lot


def build_ocr_consensus(frame_results: list[dict], *, category_profile: str = "default") -> dict:
    lot = choose_consensus([item["lote"] for item in frame_results])
    lot = post_validate_consensus_lot(lot, category_profile=category_profile)
    quantity = choose_consensus([item["quantidade_animais"] for item in frame_results])
    category = choose_consensus([item["categoria_animal"] for item in frame_results])
    weight = choose_consensus([item["peso_medio_kg"] for item in frame_results])
    composition = choose_consensus([item.get("composicao_lote", "") for item in frame_results])
    bezerros_femeas = choose_consensus([item.get("bezerros_femeas", "") for item in frame_results])
    bezerros_machos = choose_consensus([item.get("bezerros_machos", "") for item in frame_results])
    paridas = choose_consensus([item.get("paridas", "") for item in frame_results])
    solteiras = choose_consensus([item.get("solteiras", "") for item in frame_results])
    prenhas = choose_consensus([item.get("prenhas", "") for item in frame_results])

    valid_prices: list[float] = []
    for item in frame_results:
        amount = parse_currency_field(item["preco_compra_rs"])
        if amount is None:
            continue
        valid_prices.append(amount)
    best_price = max(valid_prices) if valid_prices else None
    layout_id = choose_consensus([item.get("layout_id", "") for item in frame_results])
    info_detail_debug: list[str] = []
    for item in frame_results:
        debug_payload = item.get("_ocr_debug") or {}
        for value in debug_payload.get("info_detail_texts", [])[:3]:
            cleaned = " ".join(str(value or "").split()).strip()
            if cleaned and cleaned not in info_detail_debug:
                info_detail_debug.append(cleaned)
            if len(info_detail_debug) >= 3:
                break
        if len(info_detail_debug) >= 3:
            break

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
        "composicao_lote": composition,
        "bezerros_femeas": bezerros_femeas,
        "bezerros_machos": bezerros_machos,
        "paridas": paridas,
        "solteiras": solteiras,
        "prenhas": prenhas,
        "preco_compra_rs": format_brl(best_price),
        "preco_kg_rs": format_decimal_brl(price_per_kg),
        "comprador": "",
        "regiao_destino": "",
        "confidence": confidence,
        "observacao": (
            f"ocr_consensus_frames={len(frame_results)}; layout_id={layout_id or 'desconhecido'}"
            + (
                f"; info_detail_debug={' | '.join(info_detail_debug)}"
                if info_detail_debug and not composition
                else ""
            )
        ),
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
    category_profile: str | None = None,
    metadata: dict | None = None,
) -> dict:
    event_text = ""
    if metadata:
        event_text = str(metadata.get("evento") or metadata.get("Evento") or "").strip()
    resolved_category_profile = infer_panel_category_profile(
        explicit_profile=category_profile,
        event_text=event_text,
        categories=categories,
    )
    resolved_layout_hint = infer_panel_layout_hint(
        layout_hint,
        event_text=event_text,
        category_profile=resolved_category_profile,
    )
    resolved_categories = resolve_panel_categories(categories, resolved_category_profile)
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
                resolved_layout_hint,
                resolved_category_profile,
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

    consensus = build_ocr_consensus(frame_results, category_profile=resolved_category_profile)
    consensus["category_profile"] = resolved_category_profile
    consensus["observacao"] = f"{consensus.get('observacao', '')}; category_profile={resolved_category_profile}".strip("; ")
    consensus["version"] = APP_VERSION
    return consensus


async def run_price_probe(
    *,
    video_url: str,
    video_id: str | None,
    boundary_timestamp: float,
    weight_hint: str | None,
    layout_hint: str | None,
    lot_hint: str | None = None,
    video_path: Path | None = None,
) -> dict:
    resolved_video_path = video_path or await asyncio.to_thread(download_visual_video, video_url, video_id)
    frame_paths: list[Path] = []
    price_frames: list[dict] = []
    layout_candidates: list[str] = []
    frame_errors: list[dict] = []
    event_text = ""
    if layout_hint and layout_hint not in PANEL_LAYOUT_TEMPLATES:
        layout_hint = None

    # Metadata-driven family hint keeps Correa variants in the same visual family
    # without hardcoding one template per individual leilao.
    resolved_layout_hint = layout_hint
    if lot_hint:
        # no-op placeholder to keep local variable grouping readable
        pass

    async def collect_probe_frames(offsets: list[float], window_label: str) -> list[dict]:
        timestamps = build_relative_timestamps(
            boundary_timestamp=float(boundary_timestamp),
            offsets=offsets,
        )
        collected_frames: list[dict] = []
        for timestamp in timestamps:
            base_timestamp = float(timestamp)
            offset_seconds = round(max(0.0, float(boundary_timestamp) - base_timestamp), 2)
            extracted_frame = None
            last_error = None

            for jitter in PRICE_PROBE_TIMESTAMP_JITTERS:
                probe_timestamp = max(0.0, base_timestamp + jitter)
                frame_name = f"frame_{uuid.uuid4().hex}.jpg"
                frame_path = FRAME_DIR / frame_name
                try:
                    await asyncio.to_thread(
                        extract_frame_from_stream,
                        str(resolved_video_path),
                        probe_timestamp,
                        frame_path,
                    )
                    frame_paths.append(frame_path)

                    with Image.open(frame_path) as source:
                        rgb = source.convert("RGB")
                        extracted = await asyncio.to_thread(
                            extract_price_probe_fields,
                            rgb,
                            weight_hint=weight_hint,
                            layout_hint=resolved_layout_hint,
                        )
                        layout_id = extracted.get("layout_id", "")
                        panel_visibility = assess_panel_visibility(rgb, layout_id)
                    if layout_id:
                        layout_candidates.append(layout_id)

                    extracted_frame = {
                        "timestamp": probe_timestamp,
                        "base_timestamp": base_timestamp,
                        "offset_seconds": round(max(0.0, float(boundary_timestamp) - probe_timestamp), 2),
                        "layout_id": layout_id,
                        "lot_value": extracted.get("lot_value", ""),
                        "price_value": extracted.get("price_value"),
                        "lot_texts": extracted.get("lot_texts", []),
                        "price_texts": extracted.get("price_texts", []),
                        "price_focus_texts": extracted.get("price_focus_texts", []),
                        "price_probe_alt_used": extracted.get("used_alt_probe", False),
                        "timestamp_jitter": jitter,
                        "window_label": window_label,
                        **panel_visibility,
                    }

                    if extracted_frame["price_value"] is not None:
                        break
                    if extracted_frame["price_texts"] or extracted_frame["price_focus_texts"]:
                        break
                except Exception as frame_error:
                    last_error = frame_error

            if extracted_frame is not None:
                collected_frames.append(extracted_frame)
            elif last_error is not None:
                frame_errors.append({
                    "timestamp": base_timestamp,
                    "offset_seconds": offset_seconds,
                    "window_label": window_label,
                    "error": str(last_error),
                })
        return collected_frames

    try:
        close_frames = await collect_probe_frames(PRICE_PROBE_OFFSETS, "close_window")
        price_frames.extend(close_frames)

        close_visible_frames = [frame for frame in close_frames if frame.get("panel_visible")]
        close_prices = [frame.get("price_value") for frame in close_visible_frames if frame.get("price_value") is not None]

        if not close_visible_frames or not close_prices:
            price_frames.extend(await collect_probe_frames(PRICE_PROBE_FALLBACK_OFFSETS, "early_fallback"))

        if close_visible_frames and close_visible_frames[-1].get("panel_visible"):
            price_frames.extend(await collect_probe_frames(PRICE_PROBE_FORWARD_OFFSETS, "forward_window"))

        weight_value = int(weight_hint) if str(weight_hint or "").isdigit() else None

        visible_frames = [frame for frame in price_frames if frame.get("panel_visible")]
        best_price = choose_price_probe_track(
            visible_frames or price_frames,
            weight_value=weight_value,
            lot_hint=lot_hint,
        )
        price_per_kg = round(best_price / weight_value, 2) if best_price and weight_value else None
        panel_last_seen_at = max(
            (float(frame.get("timestamp")) for frame in price_frames if frame.get("panel_visible")),
            default=None,
        )

        return {
            "status": "ok",
            "preco_compra_rs": format_brl(best_price),
            "preco_kg_rs": format_decimal_brl(price_per_kg),
            "layout_id": choose_consensus(layout_candidates),
            "price_frames": price_frames,
            "frame_errors": frame_errors,
            "panel_last_seen_at": panel_last_seen_at,
            "version": APP_VERSION,
        }
    finally:
        for frame_path in frame_paths:
            try:
                frame_path.unlink(missing_ok=True)
            except Exception:
                pass


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


def model_dump_compat(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def price_probe_job_id(item: PriceProbeItem) -> str:
    metadata = item.metadata or {}
    payload = {
        "app_version": APP_VERSION,
        "video_url": item.video_url,
        "video_id": item.video_id or "",
        "boundary_timestamp": round(float(item.boundary_timestamp), 3),
        "weight_hint": item.weight_hint or "",
        "layout_hint": item.layout_hint or "",
        "lote": str(metadata.get("lote") or "").strip(),
        "tracking_id": str(metadata.get("tracking_id") or "").strip(),
        "execution_id": str(metadata.get("execution_id") or "").strip(),
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return f"price-probe-{digest}"


async def process_price_probe_job(job_id: str, item_data: dict):
    metadata = item_data.get("metadata") or {}
    try:
        save_job(job_id, {
            "job_id": job_id,
            "status": "processing",
            "metadata": metadata,
            "version": APP_VERSION,
        })

        probe = await run_price_probe(
            video_url=str(item_data.get("video_url") or "").strip(),
            video_id=(str(item_data.get("video_id") or "").strip() or None),
            boundary_timestamp=float(item_data.get("boundary_timestamp") or 0),
            weight_hint=(str(item_data.get("weight_hint") or "").strip() or None),
            layout_hint=(str(item_data.get("layout_hint") or "").strip() or None),
            lot_hint=(str(metadata.get("lote") or "").strip() or None),
        )
        if metadata:
            probe["metadata"] = metadata

        save_job(job_id, {
            "job_id": job_id,
            "status": "finished",
            "metadata": metadata,
            "result": probe,
            "version": APP_VERSION,
        })
    except Exception as e:
        save_job(job_id, {
            "job_id": job_id,
            "status": "failed",
            "metadata": metadata,
            "error": str(e),
            "version": APP_VERSION,
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
            if payload.include_support_bundle:
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
            else:
                enriched["analysis_mode"] = "single_frame_boundary_scan"
                enriched["support_frames"] = []
                enriched["support_frame_count"] = 0
                enriched["analysis_offsets_seconds"] = []
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
            "frames": scanned_frames if payload.include_frames else [],
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
            category_profile=payload.category_profile,
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
                    category_profile=item.category_profile,
                    metadata=item.metadata,
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


@app.post("/frame/price-probe-batch")
async def frame_price_probe_batch(payload: PriceProbeBatchRequest, background_tasks: BackgroundTasks):
    try:
        results: list[dict] = []
        for item in payload.items:
            metadata = item.metadata or {}
            force_sync = str(metadata.get("force_sync") or "").strip().lower() in {"1", "true", "yes", "sync"}
            job_id = price_probe_job_id(item)
            job = load_job(job_id)

            if force_sync:
                probe = await run_price_probe(
                    video_url=item.video_url,
                    video_id=item.video_id,
                    boundary_timestamp=float(item.boundary_timestamp),
                    weight_hint=item.weight_hint,
                    layout_hint=item.layout_hint,
                    lot_hint=(str(metadata.get("lote") or "").strip() or None),
                )
                if metadata:
                    probe["metadata"] = metadata
                probe["job_id"] = job_id
                probe["version"] = APP_VERSION
                save_job(job_id, {
                    "job_id": job_id,
                    "status": "finished",
                    "metadata": metadata,
                    "result": probe,
                    "version": APP_VERSION,
                })
                results.append(probe)
                continue

            if job and job.get("status") == "finished":
                result = dict(job.get("result") or {})
                result["job_id"] = job_id
                result["metadata"] = metadata or result.get("metadata") or {}
                result["version"] = APP_VERSION
                results.append(result)
                continue

            if job and job.get("status") == "failed":
                results.append({
                    "status": "failed",
                    "error": str(job.get("error") or ""),
                    "job_id": job_id,
                    "metadata": metadata,
                    "version": APP_VERSION,
                })
                continue

            status = "queued"
            if job and job.get("status") in {"queued", "processing"}:
                status = job.get("status")
            else:
                item_data = model_dump_compat(item)
                save_job(job_id, {
                    "job_id": job_id,
                    "status": "queued",
                    "metadata": metadata,
                    "version": APP_VERSION,
                })
                background_tasks.add_task(process_price_probe_job, job_id, item_data)

            results.append({
                "status": status,
                "job_id": job_id,
                "metadata": metadata,
                "version": APP_VERSION,
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
