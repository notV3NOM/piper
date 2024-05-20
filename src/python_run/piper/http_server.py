#!/usr/bin/env python3
import argparse
import io
import logging
import wave
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

from . import PiperVoice
from .download import ensure_voice_exists, find_voice, get_voices

_LOGGER = logging.getLogger()

app = FastAPI()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    #
    parser.add_argument("-m", "--model", required=True, help="Path to Onnx model file")
    parser.add_argument("-c", "--config", help="Path to model config file")
    #
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w", "--noise_w", type=float, help="Phoneme width noise"
    )
    #
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    #
    parser.add_argument(
        "--sentence-silence",
        "--sentence_silence",
        type=float,
        default=0.0,
        help="Seconds of silence after each sentence",
    )
    #
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        action="append",
        default=[str(Path.cwd())],
        help="Data directory to check for downloaded models (default: current directory)",
    )
    parser.add_argument(
        "--download-dir",
        "--download_dir",
        help="Directory to download voices into (default: first data dir)",
    )
    #
    parser.add_argument(
        "--update-voices",
        action="store_true",
        help="Download latest voices.json during startup",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    if not args.download_dir:
        # Download to first data directory by default
        args.download_dir = args.data_dir[0]

    # Download voice if file doesn't exist
    model_path = Path(args.model)
    if not model_path.exists():
        # Load voice info
        voices_info = get_voices(args.download_dir, update_voices=args.update_voices)

        # Resolve aliases for backwards compatibility with old voice names
        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(args.model, args.data_dir, args.download_dir, voices_info)
        args.model, args.config = find_voice(args.model, args.data_dir)

    # Load voice
    voice = PiperVoice.load(args.model, config_path=args.config, use_cuda=args.cuda)
    synthesize_args = {
        "speaker_id": args.speaker,
        "length_scale": args.length_scale,
        "noise_scale": args.noise_scale,
        "noise_w": args.noise_w,
        "sentence_silence": args.sentence_silence,
    }

    @app.post("/")
    async def app_synthesize(request: Request):
        text = (await request.body()).decode("utf-8").strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        _LOGGER.debug("Synthesizing text: %s", text)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            voice.synthesize(text, wav_file, **synthesize_args)

        wav_io.seek(0)
        return StreamingResponse(wav_io, media_type="audio/wav")

    @app.get("/")
    async def app_synthesize_get(text: str = ""):
        text = text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        _LOGGER.debug("Synthesizing text: %s", text)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            voice.synthesize(text, wav_file, **synthesize_args)

        wav_io.seek(0)
        return StreamingResponse(wav_io, media_type="audio/wav")

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
