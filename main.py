# main.py
import argparse
import asyncio
import base64
import json
import logging
import re
import sys
from contextlib import asynccontextmanager

import mlx.core as mx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from mlx_lm.generate import generate_step
from mlx_lm.utils import load as load_llm

from mlx_audio.stt.models.whisper import Model as Whisper
from tts import ChatterboxTTS, CSMTTS, KokoroTTS

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
        # TTS backend selection
        tts_backend: str = "kokoro",
        # Kokoro-specific
        tts_voice: str = "af_heart",
        tts_lang_code: str = "a",
        tts_speed: float = 1.0,
        # CSM-specific
        tts_ref_audio: str | None = None,
        tts_ref_text: str | None = None,
        tts_ref_audio_seconds: int = 15,
    ):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.streaming_interval = streaming_interval
        self.frame_duration_ms = frame_duration_ms

        self.stt_model_id = stt_model
        self.llm_model = llm_model
        
        # TTS config
        self.tts_backend = tts_backend
        self.tts_voice = tts_voice
        self.tts_lang_code = tts_lang_code
        self.tts_speed = tts_speed
        self.tts_ref_audio = tts_ref_audio
        self.tts_ref_text = tts_ref_text
        self.tts_ref_audio_seconds = tts_ref_audio_seconds

        self.mlx_lock = asyncio.Lock()

    async def init_models(self):
        logger.info(f"Loading text generation model: {self.llm_model}")
        self.llm, self.tokenizer = await asyncio.to_thread(
            lambda: load_llm(self.llm_model)
        )

        logger.info(f"Loading speech-to-text model: {self.stt_model_id}")
        self.stt = Whisper.from_pretrained(self.stt_model_id)

        # Initialize TTS backend
        logger.info(f"Loading TTS backend: {self.tts_backend}")
        if self.tts_backend == "kokoro":
            self.tts = KokoroTTS(
                model_id="mlx-community/Kokoro-82M-bf16",
                voice=self.tts_voice,
                lang_code=self.tts_lang_code,
                speed=self.tts_speed,
                output_sample_rate=self.output_sample_rate,
            )
        elif self.tts_backend == "csm":
            self.tts = CSMTTS(
                model_id="mlx-community/csm-1b-fp16",
                ref_audio_path=self.tts_ref_audio,
                ref_text=self.tts_ref_text,
                ref_audio_seconds=self.tts_ref_audio_seconds,
                output_sample_rate=self.output_sample_rate,
                streaming_interval=self.streaming_interval,
                stt_model=self.stt,
            )
        elif self.tts_backend == "chatterbox":
            self.tts = ChatterboxTTS(
                model_id="mlx-community/chatterbox-turbo-4bit",
                ref_audio_path=self.tts_ref_audio,
                output_sample_rate=self.output_sample_rate,
            )
        else:
            raise ValueError(f"Unknown TTS backend: {self.tts_backend}")

        await asyncio.to_thread(self.tts.load)

        # ✅ Warm up TTS once at startup
        logger.info("Warming up TTS...")
        async with self.mlx_lock:
            await asyncio.to_thread(self.tts.warmup)
        logger.info("TTS warmup done.")

    async def transcribe(self, audio_bytes: bytes) -> str:
        audio = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        async with self.mlx_lock:
            result = await asyncio.to_thread(self.stt.generate, mx.array(audio))
        return result.text.strip()

    async def generate_response_sentences(self, text: str, cancel_event: asyncio.Event = None):
        """
        Async generator that yields sentences one at a time as the LLM generates them.
        Implements sentence-by-sentence alternation for lower latency.
        """
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

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Sentence boundary pattern
        sentence_end_pattern = re.compile(r'(?<=[.!?])\s+')

        buffer = ""
        full_response = ""

        def _make_sampler(temperature: float):
            def _sampler(logits: mx.array) -> mx.array:
                if temperature == 0.0:
                    return mx.argmax(logits, axis=-1)
                return mx.random.categorical(logits / temperature)

            return _sampler

        # Build eos id set robustly across tokenizer implementations
        eos_ids = set()
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            eos_ids.add(int(eos_token_id))
        eos_token_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if eos_token_ids:
            eos_ids.update(int(x) for x in eos_token_ids)

        token_generator = None
        finished = False

        while True:
            if cancel_event and cancel_event.is_set():
                break

            # If buffer already contains a sentence from a previous overrun, emit it without generating.
            parts = sentence_end_pattern.split(buffer, maxsplit=1)
            if len(parts) > 1:
                sentence = parts[0].strip()
                buffer = parts[1] if len(parts) > 1 else ""
                if sentence:
                    yield sentence
                continue

            if finished:
                break

            sentence_to_yield = None

            # Generate until we hit a sentence boundary, but ONLY while holding the MLX lock.
            async with self.mlx_lock:
                if token_generator is None:
                    prompt_tokens = self.tokenizer.encode(prompt)
                    prompt_tokens = mx.array(prompt_tokens)
                    token_generator = generate_step(
                        prompt_tokens,
                        self.llm,
                        max_tokens=512,
                        sampler=_make_sampler(0.7),
                    )

                while True:
                    try:
                        token, _logprobs = next(token_generator)
                    except StopIteration:
                        finished = True
                        break

                    # Handle both mx.array and plain int returns
                    if hasattr(token, 'item'):
                        token_id = int(token.item())
                    else:
                        token_id = int(token)

                    if eos_ids and token_id in eos_ids:
                        finished = True
                        break

                    token_str = self.tokenizer.decode([token_id])
                    buffer += token_str
                    full_response += token_str

                    parts = sentence_end_pattern.split(buffer, maxsplit=1)
                    if len(parts) > 1:
                        sentence_to_yield = parts[0].strip()
                        buffer = parts[1] if len(parts) > 1 else ""
                        break

            if sentence_to_yield:
                yield sentence_to_yield

        if buffer.strip():
            yield buffer.strip()

        logger.info(f"Full response: {full_response.strip()}")

    async def generate_response(self, text: str) -> str:
        """Legacy method - generates full response at once."""
        sentences = []
        async for sentence in self.generate_response_sentences(text):
            sentences.append(sentence)
        return " ".join(sentences)

    async def synthesize_speech(self, text: str, cancel_event: asyncio.Event = None):
        """
        Generator that yields audio chunks (as int16 PCM bytes) for the given text.
        Can be cancelled by setting cancel_event.
        """
        audio_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _tts_stream():
            # TTS wrapper already returns int16 PCM bytes
            for audio_bytes in self.tts.generate(text):
                if cancel_event and cancel_event.is_set():
                    break
                loop.call_soon_threadsafe(audio_queue.put_nowait, audio_bytes)
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
                    yield chunk
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
        llm_model=app.state.llm_model,
        tts_backend=app.state.tts_backend,
        tts_voice=app.state.tts_voice,
        tts_lang_code=app.state.tts_lang_code,
        tts_speed=app.state.tts_speed,
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

                        cancel_event.clear()

                        # ✅ Sentence-by-sentence alternation:
                        # Generate sentence 1 → Speak sentence 1 → Generate sentence 2 → Speak sentence 2 → ...
                        async def stream_sentences():
                            sentence_num = 0
                            async for sentence in pipeline.generate_response_sentences(
                                transcription, cancel_event
                            ):
                                if cancel_event.is_set():
                                    break

                                sentence_num += 1
                                logger.info(f"Sentence {sentence_num}: {sentence}")

                                # Send the sentence text to client
                                await websocket.send_text(
                                    json.dumps({"type": "response", "text": sentence})
                                )

                                # Immediately TTS this sentence
                                buffered = bytearray()
                                started = False

                                async for audio_chunk in pipeline.synthesize_speech(
                                    sentence, cancel_event
                                ):
                                    if cancel_event.is_set():
                                        break

                                    if not started:
                                        buffered.extend(audio_chunk)
                                        if len(buffered) < PREBUFFER_BYTES:
                                            continue

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

                                # Flush any remaining buffered audio for this sentence
                                if buffered:
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

                            await websocket.send_text(json.dumps({"type": "audio_end"}))

                        current_tts_task = asyncio.create_task(stream_sentences())

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
        "--llm_model",
        type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="LLM model",
    )
    parser.add_argument(
        "--tts_backend",
        type=str,
        default="kokoro",
        choices=["kokoro", "csm", "chatterbox"],
        help="TTS backend to use: kokoro, csm, or chatterbox",
    )
    # Kokoro-specific args
    parser.add_argument(
        "--tts_voice",
        type=str,
        default="af_heart",
        help="[Kokoro] Voice to use (e.g. af_heart, af_bella, am_adam)",
    )
    parser.add_argument(
        "--tts_lang_code",
        type=str,
        default="a",
        help="[Kokoro] Language code (a=American English, b=British English)",
    )
    parser.add_argument(
        "--tts_speed",
        type=float,
        default=1.0,
        help="TTS speech speed multiplier",
    )
    # CSM-specific args
    parser.add_argument(
        "--tts_ref_audio",
        type=str,
        default=None,
        help="[CSM] Reference audio WAV path for voice cloning",
    )
    parser.add_argument(
        "--tts_ref_text",
        type=str,
        default=None,
        help="[CSM] Reference text (auto-transcribed if not provided)",
    )
    parser.add_argument(
        "--tts_ref_audio_seconds",
        type=int,
        default=15,
        help="[CSM] Seconds of reference audio to use",
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
    app.state.llm_model = args.llm_model
    app.state.tts_backend = args.tts_backend
    app.state.tts_voice = args.tts_voice
    app.state.tts_lang_code = args.tts_lang_code
    app.state.tts_speed = args.tts_speed
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
