from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
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

app = FastAPI(title="Auction Worker")

APP_VERSION = "async-v13"

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

            frame_url = str(request.base_url).rstrip("/") + f"/frame/file/{frame_name}"

            full_diff = None
            focus_diff = None
            is_candidate = False
            reason = "initial_frame"
            score = 0

            if previous is not None:
                full_diff = hamming_distance(previous["full_hash"], hashes["full_hash"])
                focus_diff = hamming_distance(previous["focus_hash"], hashes["focus_hash"])
                score = max(full_diff, focus_diff)
                is_candidate = (
                    full_diff >= payload.full_diff_threshold
                    or focus_diff >= payload.focus_diff_threshold
                )
                reason = "visual_change" if is_candidate else "stable"
            else:
                is_candidate = True
                score = 64

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
                if (
                    boundary_candidates
                    and (timestamp - boundary_candidates[-1]["timestamp"]) < payload.min_gap_seconds
                ):
                    if score > (boundary_candidates[-1].get("change_score") or 0):
                        boundary_candidates[-1] = frame_info
                else:
                    boundary_candidates.append(frame_info)

            previous = hashes

        return {
            "status": "ok",
            "video_url": payload.video_url,
            "video_id": payload.video_id,
            "worker_job_id": payload.worker_job_id,
            "duration_seconds": duration_seconds,
            "sampled_frames": len(scanned_frames),
            "boundary_candidates": len(boundary_candidates),
            "candidates": boundary_candidates,
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
