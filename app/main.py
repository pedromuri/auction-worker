from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from pytube import YouTube
import tempfile
import os
import httpx

app = FastAPI(title="Auction Worker")

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


async def transcribe_with_deepgram(audio_path: Path) -> dict:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY não configurada.")

    headers = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/octet-stream",
}
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        with open(audio_path, "rb") as f:
            response = await client.post(
                DEEPGRAM_URL,
                headers=headers,
                content=f.read(),
            )

    if response.status_code >= 400:
        raise RuntimeError(f"Erro Deepgram: {response.status_code} - {response.text}")

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
    return {"status": "ok"}


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
                "segments": segments,
            }

    except Exception as e:
        return {
            "has_transcript": False,
            "job_id": payload.job_id,
            "video_id": payload.video_id,
            "error": str(e),
        }
