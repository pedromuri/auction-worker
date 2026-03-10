from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os
import httpx
import subprocess
import yt_dlp
import uuid
import json

app = FastAPI(title="Auction Worker")

APP_VERSION = "async-v4"

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

ROOT_DIR = Path("/app")
FALLBACK_COOKIES_FILE = ROOT_DIR / "cookies.txt"


class TranscriptRequest(BaseModel):
    video_url: str
    video_id: str
    job_id: str | None = None


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
    """
    Prioridade:
    1) variável de ambiente YOUTUBE_COOKIES
    2) arquivo /app/cookies.txt
    3) nenhum cookie
    """
    if YOUTUBE_COOKIES and YOUTUBE_COOKIES.strip():
        cookie_file = output_dir / "cookies.txt"
        cookie_file.write_text(YOUTUBE_COOKIES, encoding="utf-8")
        return cookie_file

    if FALLBACK_COOKIES_FILE.exists():
        return FALLBACK_COOKIES_FILE

    return None


def download_audio(video_url: str, output_dir: Path) -> Path:
    output_template = str(output_dir / "audio.%(ext)s")
    cookie_file = write_cookies_file(output_dir)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        },
    }

    if cookie_file:
        ydl_opts["cookiefile"] = str(cookie_file)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar áudio com yt-dlp: {str(e)}")

    files = list(output_dir.glob("audio.*"))

    if not files:
        raise RuntimeError("Falha ao baixar áudio: nenhum arquivo foi gerado.")

    return files[0]


def convert_to_wav(input_file: Path, output_dir: Path) -> Path:
    output_file = output_dir / "audio.wav"

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
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
            "text": text
        })

    return segments


async def process_job(job_id: str, video_url: str, video_id: str):
    try:
        save_job(job_id, {
            "job_id": job_id,
            "video_id": video_id,
            "status": "processing"
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            downloaded = download_audio(video_url, tmp)
            wav = convert_to_wav(downloaded, tmp)
            deepgram = await transcribe_with_deepgram(wav)
            segments = normalize_segments(deepgram)

            save_job(job_id, {
                "job_id": job_id,
                "video_id": video_id,
                "status": "finished",
                "has_transcript": len(segments) > 0,
                "downloaded_file": downloaded.name,
                "wav_file": wav.name,
                "segments": segments
            })

    except Exception as e:
        save_job(job_id, {
            "job_id": job_id,
            "video_id": video_id,
            "status": "failed",
            "error": str(e)
        })


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "cookies_env_configured": bool(YOUTUBE_COOKIES and YOUTUBE_COOKIES.strip()),
        "cookies_file_exists": FALLBACK_COOKIES_FILE.exists(),
    }


@app.post("/transcript/start")
async def transcript_start(payload: TranscriptRequest, background_tasks: BackgroundTasks):
    job_id = payload.job_id or str(uuid.uuid4())

    save_job(job_id, {
        "job_id": job_id,
        "video_id": payload.video_id,
        "status": "queued"
    })

    background_tasks.add_task(
        process_job,
        job_id,
        payload.video_url,
        payload.video_id
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "version": APP_VERSION
    }


@app.get("/transcript/status/{job_id}")
async def transcript_status(job_id: str):
    job = load_job(job_id)

    if not job:
        return {
            "job_id": job_id,
            "status": "not_found",
            "version": APP_VERSION
        }

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "version": APP_VERSION
    }


@app.get("/transcript/result/{job_id}")
async def transcript_result(job_id: str):
    job = load_job(job_id)

    if not job:
        return {
            "job_id": job_id,
            "status": "not_found",
            "version": APP_VERSION
        }

    job["version"] = APP_VERSION
    return job
