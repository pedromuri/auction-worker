from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import tempfile
import os
import httpx
import subprocess
import yt_dlp

app = FastAPI(title="Auction Worker")

APP_VERSION = "debug-v4"

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

    output_template = str(output_dir / "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    files = list(output_dir.glob("audio.*"))

    if not files:
        raise RuntimeError("Falha ao baixar áudio com yt-dlp.")

    return files[0]


def convert_to_wav(input_file: Path, output_dir: Path) -> Path:

    output_file = output_dir / "audio.wav"

    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_file),
            "-ar", "16000",
            "-ac", "1",
            str(output_file)
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Erro ffmpeg: {result.stderr}")

    if not output_file.exists():
        raise RuntimeError("Falha na conversão para WAV.")

    return output_file


async def transcribe_with_deepgram(audio_path: Path) -> dict:

    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY não configurada.")

    file_size = audio_path.stat().st_size

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav",
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
            f"size={file_size} bytes | "
            f"body={response.text}"
        )

    return response.json()


def normalize_segments(deepgram_response: dict):

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
    return {
        "status": "ok",
        "version": APP_VERSION
    }


@app.post("/transcript")
async def transcript(payload: TranscriptRequest):

    downloaded_name = None
    wav_name = None

    try:

        with tempfile.TemporaryDirectory() as tmpdir:

            tmp_path = Path(tmpdir)

            downloaded_audio = download_audio(payload.video_url, tmp_path)
            downloaded_name = downloaded_audio.name

            wav_audio = convert_to_wav(downloaded_audio, tmp_path)
            wav_name = wav_audio.name

            deepgram_response = await transcribe_with_deepgram(wav_audio)

            segments = normalize_segments(deepgram_response)

            return {
                "has_transcript": len(segments) > 0,
                "job_id": payload.job_id,
                "video_id": payload.video_id,
                "version": APP_VERSION,
                "downloaded_file": downloaded_name,
                "wav_file": wav_name,
                "segments": segments,
            }

    except Exception as e:

        return {
            "has_transcript": False,
            "job_id": payload.job_id,
            "video_id": payload.video_id,
            "version": APP_VERSION,
            "downloaded_file": downloaded_name,
            "wav_file": wav_name,
            "error": str(e),
        }
