import yt_dlp


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
