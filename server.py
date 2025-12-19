import argparse
import asyncio
import base64
import json
import logging

import mlx.core as mx
import numpy as np
from websockets.asyncio.server import serve
from mlx_lm.generate import generate as generate_text
from mlx_lm.utils import load as load_llm

from mlx_audio.stt.models.whisper import Model as Whisper
from mlx_audio.tts.utils import load_model as load_tts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoicePipeline:
    def __init__(
        self,
        stt_model="mlx-community/whisper-large-v3-turbo",
        # llm_model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        llm_model="mistralai/Ministral-3-3B-Reasoning-2512-GGUF",
        tts_model="mlx-community/csm-1b",
        output_sample_rate=24_000,
    ):
        self.stt_model = stt_model
        self.llm_model = llm_model
        self.tts_model = tts_model
        self.output_sample_rate = output_sample_rate
        self.mlx_lock = asyncio.Lock()

    async def init_models(self):
        logger.info(f"Loading LLM: {self.llm_model}")
        self.llm, self.tokenizer = await asyncio.to_thread(
            lambda: load_llm(self.llm_model)
        )
        logger.info(f"Loading TTS: {self.tts_model}")
        self.tts = await asyncio.to_thread(lambda: load_tts(self.tts_model))
        logger.info(f"Loading STT: {self.stt_model}")
        self.stt = Whisper.from_pretrained(self.stt_model)
        logger.info("All models loaded")

    async def transcribe(self, audio_bytes: bytes) -> str:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        async with self.mlx_lock:
            result = await asyncio.to_thread(self.stt.generate, mx.array(audio))
        return result.text.strip()

    async def generate_response(self, text: str) -> str:
        def _get_response(llm, tokenizer, messages):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return generate_text(llm, tokenizer, prompt, verbose=False).strip()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful voice assistant. Respond with short sentences.",
            },
            {"role": "user", "content": text},
        ]
        async with self.mlx_lock:
            return await asyncio.to_thread(
                _get_response, self.llm, self.tokenizer, messages
            )

    async def synthesize_speech(self, text: str):
        audio_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _tts_stream():
            for chunk in self.tts.generate(
                text,
                sample_rate=self.output_sample_rate,
                stream=True,
                streaming_interval=3,
                verbose=False,
            ):
                loop.call_soon_threadsafe(audio_queue.put_nowait, chunk.audio)
            loop.call_soon_threadsafe(audio_queue.put_nowait, None)

        async with self.mlx_lock:
            tts_task = asyncio.create_task(asyncio.to_thread(_tts_stream))
            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break
                    audio_np = np.array(chunk, dtype=np.float32)
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    yield audio_int16.tobytes()
            finally:
                await tts_task


pipeline: VoicePipeline = None


async def handle_client(websocket):
    logger.info("Client connected")
    audio_buffer = bytearray()

    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "audio":
                audio_data = base64.b64decode(data["data"])
                audio_buffer.extend(audio_data)

            elif msg_type == "end_of_speech":
                if audio_buffer:
                    logger.info("Transcribing...")
                    transcription = await pipeline.transcribe(bytes(audio_buffer))
                    audio_buffer.clear()

                    if transcription:
                        logger.info(f"User: {transcription}")
                        await websocket.send(
                            json.dumps({"type": "transcription", "text": transcription})
                        )

                        logger.info("Generating response...")
                        response = await pipeline.generate_response(transcription)
                        logger.info(f"Assistant: {response}")
                        await websocket.send(
                            json.dumps({"type": "response", "text": response})
                        )

                        logger.info("Synthesizing speech...")
                        async for audio_chunk in pipeline.synthesize_speech(response):
                            await websocket.send(
                                json.dumps({
                                    "type": "audio",
                                    "data": base64.b64encode(audio_chunk).decode(),
                                })
                            )
                        await websocket.send(json.dumps({"type": "audio_end"}))
                        logger.info("Response complete")

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info("Client disconnected")


async def main(args):
    global pipeline
    pipeline = VoicePipeline(
        stt_model=args.stt_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
    )
    await pipeline.init_models()

    logger.info(f"Server starting on ws://{args.host}:{args.port}")
    async with serve(handle_client, args.host, args.port):
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Pipeline Server")
    parser.add_argument("--stt_model", default="mlx-community/whisper-large-v3-turbo")
    # parser.add_argument("--llm_model", default="mlx-community/Ministral-3-3B-Instruct-2512-4bit")
    parser.add_argument("--llm_model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    # parser.add_argument("--llm_model", default="mistralai/Ministral-3-3B-Reasoning-2512-GGUF")
    parser.add_argument("--tts_model", default="mlx-community/csm-1b-fp16")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    asyncio.run(main(args))
