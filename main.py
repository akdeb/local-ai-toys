# main.py
import argparse
import asyncio
import base64
import json
import logging
import sys
from contextlib import asynccontextmanager

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mlx_lm.generate import generate as generate_text
from mlx_lm.utils import load as load_llm

from mlx_audio.stt.models.whisper import Model as Whisper
from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.tts.generate import load_audio as load_tts_audio

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(
        self,
        silence_threshold=0.03,
        silence_duration=1.5,
        input_sample_rate=16_000,
        output_sample_rate=24_000,
        streaming_interval=3,
        frame_duration_ms=30,
        stt_model="mlx-community/whisper-large-v3-turbo",
        llm_model="Qwen/Qwen2.5-0.5B-Instruct-4bit",
        tts_model="mlx-community/csm-1b-fp16",
        tts_ref_audio: str | None = None,
        tts_ref_text: str | None = None,
        tts_ref_audio_seconds: int | None = None,
    ):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.streaming_interval = streaming_interval
        self.frame_duration_ms = frame_duration_ms

        self.stt_model = stt_model
        self.llm_model = llm_model
        self.tts_model = tts_model
        self._tts_ref_audio_path = tts_ref_audio
        self.tts_ref_text = tts_ref_text
        self.tts_ref_audio = None  # Will be loaded as mx.array in init_models
        self.tts_ref_audio_seconds = tts_ref_audio_seconds

        self.mlx_lock = asyncio.Lock()

    async def init_models(self):
        logger.info(f"Loading text generation model: {self.llm_model}")
        self.llm, self.tokenizer = await asyncio.to_thread(
            lambda: load_llm(self.llm_model)
        )

        logger.info(f"Loading text-to-speech model: {self.tts_model}")
        self.tts = await asyncio.to_thread(lambda: load_tts(self.tts_model))

        logger.info(f"Loading speech-to-text model: {self.stt_model}")
        self.stt = Whisper.from_pretrained(self.stt_model)

        # Load reference audio for voice cloning if provided
        if self._tts_ref_audio_path:
            logger.info(f"Loading TTS reference audio: {self._tts_ref_audio_path}")
            self.tts_ref_audio = load_tts_audio(
                self._tts_ref_audio_path,
                sample_rate=self.tts.sample_rate,
                segment_duration=self.tts_ref_audio_seconds,
            )
            # Auto-transcribe ref_audio if ref_text not provided
            if not self.tts_ref_text:
                logger.info("Transcribing reference audio...")
                result = self.stt.generate(self.tts_ref_audio)
                self.tts_ref_text = result.text.strip()
                logger.info(f"Reference text: {self.tts_ref_text}")

        # ✅ Warm up TTS once at startup to avoid first-utterance garble
        logger.info("Warming up TTS...")
        async with self.mlx_lock:
            await asyncio.to_thread(self._warmup_tts)
        logger.info("TTS warmup done.")

    def _warmup_tts(self):
        generate_kwargs = {
            "sample_rate": self.output_sample_rate,
            "stream": True,
            "streaming_interval": self.streaming_interval,
            "verbose": False,
        }
        if self.tts_ref_audio is not None:
            generate_kwargs["ref_audio"] = self.tts_ref_audio
        if self.tts_ref_text:
            generate_kwargs["ref_text"] = self.tts_ref_text

        # Generate a short utterance and discard output (forces MLX allocations/JIT/kernels)
        for _chunk in self.tts.generate("testing", **generate_kwargs):
            pass

    async def transcribe(self, audio_bytes: bytes) -> str:
        audio = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        async with self.mlx_lock:
            result = await asyncio.to_thread(self.stt.generate, mx.array(audio))
        return result.text.strip()

    async def generate_response(self, text: str) -> str:
        def _get_llm_response(llm, tokenizer, messages, *, verbose=False):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return generate_text(llm, tokenizer, prompt, verbose=verbose).strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. You always respond with short "
                    "sentences and never use punctuation like parentheses or colons "
                    "that wouldn't appear in conversational speech."
                ),
            },
            {"role": "user", "content": text},
        ]
        async with self.mlx_lock:
            response_text = await asyncio.to_thread(
                _get_llm_response, self.llm, self.tokenizer, messages, verbose=False
            )
        return response_text

    async def synthesize_speech(self, text: str, cancel_event: asyncio.Event = None):
        """
        Generator that yields audio chunks (as int16 PCM bytes) for the given text.
        Can be cancelled by setting cancel_event.
        """
        audio_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _tts_stream():
            generate_kwargs = {
                "sample_rate": self.output_sample_rate,
                "stream": True,
                "streaming_interval": self.streaming_interval,
                "verbose": False,
            }
            if self.tts_ref_audio is not None:
                generate_kwargs["ref_audio"] = self.tts_ref_audio
            if self.tts_ref_text:
                generate_kwargs["ref_text"] = self.tts_ref_text

            for chunk in self.tts.generate(text, **generate_kwargs):
                if cancel_event and cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(audio_queue.put_nowait, chunk.audio)
            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        async with self.mlx_lock:
            tts_task = asyncio.create_task(asyncio.to_thread(_tts_stream))
            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    if cancel_event and cancel_event.is_set():
                        break

                    # ✅ Convert float audio to int16 safely (clip to avoid overflow garble)
                    audio_np = np.asarray(chunk, dtype=np.float32)
                    audio_np = np.clip(audio_np, -1.0, 1.0)
                    audio_int16 = (audio_np * 32767.0).astype(np.int16)
                    yield audio_int16.tobytes()
            finally:
                await tts_task


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f"Client connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f"Client disconnected. Total connections: {len(self.active_connections)}"
        )


pipeline: VoicePipeline = None
manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = VoicePipeline(
        stt_model=app.state.stt_model,
        tts_model=app.state.tts_model,
        llm_model=app.state.llm_model,
        tts_ref_audio=app.state.tts_ref_audio,
        tts_ref_text=app.state.tts_ref_text,
        tts_ref_audio_seconds=app.state.tts_ref_audio_seconds,
        silence_threshold=app.state.silence_threshold,
        silence_duration=app.state.silence_duration,
        streaming_interval=app.state.streaming_interval,
        output_sample_rate=app.state.output_sample_rate,
    )
    await pipeline.init_models()
    logger.info("Voice pipeline initialized")
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Voice Pipeline WebSocket Server", lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for voice communication.

    Protocol:
    - Client sends: {"type": "audio", "data": "<base64 encoded int16 PCM audio>"}
    - Client sends: {"type": "end_of_speech"} to signal end of utterance
    - Server sends: {"type": "transcription", "text": "..."}
    - Server sends: {"type": "response", "text": "..."}
    - Server sends: {"type": "audio", "data": "<base64 encoded int16 PCM audio>"}
    - Server sends: {"type": "audio_end"}
    """
    await manager.connect(websocket)

    audio_buffer = bytearray()
    cancel_event = asyncio.Event()
    current_tts_task = None

    # ✅ Prebuffer to avoid initial underrun/garble
    PREBUFFER_MS = 300
    PREBUFFER_BYTES = int(pipeline.output_sample_rate * (PREBUFFER_MS / 1000.0) * 2)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")

            if msg_type == "audio":
                audio_data = base64.b64decode(message["data"])
                audio_buffer.extend(audio_data)

                # If user is speaking while we're TTS-ing, cancel current TTS
                if current_tts_task and not current_tts_task.done():
                    cancel_event.set()
                    try:
                        await current_tts_task
                    except asyncio.CancelledError:
                        pass
                    cancel_event.clear()
                    current_tts_task = None

            elif msg_type == "end_of_speech":
                if audio_buffer:
                    logger.info("Processing audio...")

                    transcription = await pipeline.transcribe(bytes(audio_buffer))
                    audio_buffer.clear()

                    if transcription:
                        logger.info(f"Transcribed: {transcription}")
                        await websocket.send_text(
                            json.dumps({"type": "transcription", "text": transcription})
                        )

                        logger.info("Generating response...")
                        response_text = await pipeline.generate_response(transcription)
                        logger.info(f"Response: {response_text}")

                        await websocket.send_text(
                            json.dumps({"type": "response", "text": response_text})
                        )

                        cancel_event.clear()

                        async def stream_audio():
                            buffered = bytearray()
                            started = False

                            async for audio_chunk in pipeline.synthesize_speech(
                                response_text, cancel_event
                            ):
                                if cancel_event.is_set():
                                    break

                                if not started:
                                    buffered.extend(audio_chunk)
                                    if len(buffered) < PREBUFFER_BYTES:
                                        continue

                                    # Send the buffered audio as the first payload
                                    await websocket.send_text(
                                        json.dumps(
                                            {
                                                "type": "audio",
                                                "data": base64.b64encode(
                                                    bytes(buffered)
                                                ).decode("utf-8"),
                                            }
                                        )
                                    )
                                    buffered.clear()
                                    started = True
                                else:
                                    await websocket.send_text(
                                        json.dumps(
                                            {
                                                "type": "audio",
                                                "data": base64.b64encode(
                                                    audio_chunk
                                                ).decode("utf-8"),
                                            }
                                        )
                                    )

                            await websocket.send_text(json.dumps({"type": "audio_end"}))

                        current_tts_task = asyncio.create_task(stream_audio())

            elif msg_type == "cancel":
                if current_tts_task and not current_tts_task.done():
                    cancel_event.set()
                audio_buffer.clear()

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


def main():
    if len(sys.argv) >= 6 and sys.argv[1:5] == ["-B", "-S", "-I", "-c"]:
        code = sys.argv[5]
        if isinstance(code, str) and code.startswith("from multiprocessing."):
            exec(code, {"__name__": "__main__"})
            return

    parser = argparse.ArgumentParser(description="Voice Pipeline WebSocket Server")
    parser.add_argument(
        "--stt_model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="STT model",
    )
    parser.add_argument(
        "--tts_model", type=str, default="mlx-community/csm-1b-fp16", help="TTS model"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="LLM model",
    )
    parser.add_argument(
        "--tts_ref_audio",
        type=str,
        default=None,
        help="Optional reference audio WAV path for voice cloning",
    )
    parser.add_argument(
        "--tts_ref_audio_seconds",
        type=int,
        default=15,
        help="Seconds of reference audio to use (CSM has a max context; keep this small)",
    )
    parser.add_argument(
        "--tts_ref_text",
        type=str,
        default=None,
        help="Optional reference caption for the reference audio",
    )
    parser.add_argument(
        "--silence_duration", type=float, default=1.5, help="Silence duration"
    )
    parser.add_argument(
        "--silence_threshold", type=float, default=0.03, help="Silence threshold"
    )
    parser.add_argument(
        "--streaming_interval", type=int, default=3, help="Streaming interval"
    )
    parser.add_argument(
        "--output_sample_rate",
        type=int,
        default=24_000,
        help="Output sample rate for TTS audio",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    app.state.stt_model = args.stt_model
    app.state.tts_model = args.tts_model
    app.state.llm_model = args.llm_model
    app.state.tts_ref_audio = args.tts_ref_audio
    app.state.tts_ref_text = args.tts_ref_text
    app.state.tts_ref_audio_seconds = args.tts_ref_audio_seconds
    app.state.silence_threshold = args.silence_threshold
    app.state.silence_duration = args.silence_duration
    app.state.streaming_interval = args.streaming_interval
    app.state.output_sample_rate = args.output_sample_rate

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
