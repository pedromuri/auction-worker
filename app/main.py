from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from pytube import YouTube
import tempfile
import os
import httpx

app = FastAPI(title="Auction Worker")

APP_VERSION = "debug-v2"

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_URL = (
    "https://api.deepgram.com/v1/listen"
    "?model=nova-2"
    "&language=pt-BR"
    "&smart_format=true"
    "&utterances=true"
    "&punctuate=true"
    "&numerals=true"
)


class TranscriptRequest(BaseModel):
    video_url: str
    video_id: str
    job_id: str


def download_audio(video_url: str, output_dir: Path) -> Path:
    yt = YouTube(video_url)

    stream = (
        yt.streams
        .filter(only_audio=True)
        .order_by("abr")
        .desc()
        .first()
    )

    if not stream:
        raise RuntimeError("Não foi possível localizar um stream de áudio do vídeo.")

    stream.download(
        output_path=str(output_dir),
        filename="audio"
    )

    downloaded_files = list(output_dir.glob("audio.*"))
    if not downloaded_files:
        raise RuntimeError("Áudio não encontrado após download.")

    return downloaded_files[0]


def guess_content_type(audio_path: Path) -> str:
    ext = audio_path.suffix.lower()

    if ext == ".mp3":
        return "audio/mpeg"
    if ext in [".mp4", ".m4a"]:
        return "audio/mp4"
    if ext == ".webm":
        return "audio/webm"
    if ext == ".wav":
        return "audio/wav"
    if ext == ".ogg":
        return "audio/ogg"

    return "application/octet-stream"


async def transcribe_with_deepgram(audio_path: Path) -> dict:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY não configurada.")

    content_type = guess_content_type(audio_path)
    file_size = audio_path.stat().st_size

    if file_size == 0:
        raise RuntimeError("Arquivo de áudio vazio.")

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type,
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
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
            f"content_type={content_type} | "
            f"file={audio_path.name} | "
            f"size={file_size} bytes | "
            f"body={response.text}"
        )

    return response.json()


def normalize_segments(deepgram_response: dict) -> list[dict]:
    results = deepgram_response.get("results", {})
    utterances = results.get("utterances", [])

    segments = []
    for utt in utterances:
        text = (utt.get("transcript") or "").strip()
        if not text:
            continue

        segments.append(
            {
                "start": utt.get("start"),
                "end": utt.get("end"),
                "text": text,
            }
        )

    return segments


@app.get("/health")
async def health():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/transcript")
async def transcript(payload: TranscriptRequest):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            audio_path = download_audio(payload.video_url, tmp_path)
            deepgram_response = await transcribe_with_deepgram(audio_path)
            segments = normalize_segments(deepgram_response)

            return {
                "has_transcript": len(segments) > 0,
                "job_id": payload.job_id,
                "video_id": payload.video_id,
                "version": APP_VERSION,
                "downloaded_file": audio_path.name,
                "segments": segments,
            }

    except Exception as e:
        return {
            "has_transcript": False,
            "job_id": payload.job_id,
            "video_id": payload.video_id,
            "version": APP_VERSION,
            "error": str(e),
        }
