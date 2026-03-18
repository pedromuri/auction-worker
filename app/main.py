from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os
import httpx
import subprocess
import yt_dlp
import uuid
import json
import asyncio

app = FastAPI(title="Auction Worker")

APP_VERSION = "async-v9"

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


def build_ydl_opts(output_template: str, cookie_file: Path | None, format_selector: str) -> dict:
    ydl_opts = {
        "format": format_selector,
        "outtmpl": output_template,
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


def download_video_for_frame(video_url: str, output_dir: Path) -> Path:
    output_template = str(output_dir / "video.%(ext)s")
    cookie_file = write_cookies_file(output_dir)

    format_attempts = [
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "bestvideo+bestaudio/best",
        "best",
    ]

    errors = []

    for format_selector in format_attempts:
        try:
            ydl_opts = build_ydl_opts(output_template, cookie_file, format_selector)
            ydl_opts["merge_output_format"] = "mp4"

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            downloaded = find_downloaded_file(output_dir, "video.")
            if downloaded and downloaded.exists():
                return downloaded

            errors.append(f"{format_selector}: download sem arquivo gerado")
        except Exception as e:
            errors.append(f"{format_selector}: {str(e)}")

    raise RuntimeError(
        "Erro ao baixar vídeo para frame com yt-dlp. Tentativas: " + " | ".join(errors)
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


def extract_frame_from_video(input_video: Path, timestamp: float, frame_path: Path):
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(input_video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Erro ffmpeg ao extrair frame: {result.stderr}")

    if not frame_path.exists():
        raise RuntimeError("Falha ao gerar frame.")


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
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            downloaded_video = await asyncio.to_thread(download_video_for_frame, payload.video_url, tmp)

            frame_name = f"frame_{uuid.uuid4().hex}.jpg"
            frame_path = FRAME_DIR / frame_name

            await asyncio.to_thread(
                extract_frame_from_video,
                downloaded_video,
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
        return {
            "status": "failed",
            "error": str(e),
            "timestamp": payload.timestamp,
            "video_url": payload.video_url,
            "video_id": payload.video_id,
            "worker_job_id": payload.worker_job_id,
            "version": APP_VERSION,
        }


@app.get("/frame/file/{filename}")
async def frame_file(filename: str):
    path = FRAME_DIR / filename
    if not path.exists():
        return {"status": "not_found", "filename": filename}
    return FileResponse(path)
