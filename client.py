import argparse
import asyncio
import base64
import json
import threading

import numpy as np
import pyaudio
import webrtcvad
from websockets.asyncio.client import connect

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
OUTPUT_SAMPLE_RATE = 24000


class VoiceClient:
    def __init__(self, server_url: str, vad_aggressiveness: int = 3):
        self.server_url = server_url
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.audio = pyaudio.PyAudio()
        
        self.mic_muted = False
        self.running = True
        self.audio_buffer = bytearray()
        self.speech_started = False
        self.silence_frames = 0
        self.silence_threshold = 30  # ~900ms of silence to end speech

    def start_mic_stream(self):
        return self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
        )

    def start_speaker_stream(self):
        return self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
        )

    async def run(self):
        print(f"Connecting to {self.server_url}...")
        async with connect(self.server_url) as ws:
            print("Connected! Speak into your microphone.")
            print("Press Ctrl+C to exit.\n")

            receive_task = asyncio.create_task(self.receive_loop(ws))
            send_task = asyncio.create_task(self.send_loop(ws))

            try:
                await asyncio.gather(receive_task, send_task)
            except asyncio.CancelledError:
                pass

    async def send_loop(self, ws):
        mic_stream = self.start_mic_stream()
        loop = asyncio.get_running_loop()

        try:
            while self.running:
                if self.mic_muted:
                    await asyncio.sleep(0.01)
                    continue

                audio_data = await loop.run_in_executor(
                    None, mic_stream.read, FRAME_SIZE, False
                )

                is_speech = self.vad.is_speech(audio_data, SAMPLE_RATE)

                if is_speech:
                    if not self.speech_started:
                        self.speech_started = True
                        print("🎤 Listening...")
                    self.audio_buffer.extend(audio_data)
                    self.silence_frames = 0
                elif self.speech_started:
                    self.audio_buffer.extend(audio_data)
                    self.silence_frames += 1

                    if self.silence_frames >= self.silence_threshold:
                        print("⏳ Processing...")
                        self.mic_muted = True
                        
                        await ws.send(json.dumps({
                            "type": "audio",
                            "data": base64.b64encode(bytes(self.audio_buffer)).decode(),
                        }))
                        await ws.send(json.dumps({"type": "end_of_speech"}))
                        
                        self.audio_buffer.clear()
                        self.speech_started = False
                        self.silence_frames = 0

        finally:
            mic_stream.stop_stream()
            mic_stream.close()

    async def receive_loop(self, ws):
        speaker_stream = self.start_speaker_stream()

        try:
            async for message in ws:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "transcription":
                    print(f"📝 You: {data['text']}")

                elif msg_type == "response":
                    print(f"🤖 Assistant: {data['text']}")

                elif msg_type == "audio":
                    audio_bytes = base64.b64decode(data["data"])
                    speaker_stream.write(audio_bytes)

                elif msg_type == "audio_end":
                    print("✅ Ready\n")
                    self.mic_muted = False

        finally:
            speaker_stream.stop_stream()
            speaker_stream.close()

    def cleanup(self):
        self.running = False
        self.audio.terminate()


async def main(args):
    client = VoiceClient(args.server, args.vad_level)
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        client.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Client")
    parser.add_argument("--server", default="ws://localhost:8765")
    parser.add_argument("--vad_level", type=int, default=3, choices=[0, 1, 2, 3],
                        help="VAD aggressiveness (0-3, higher = more aggressive)")
    args = parser.parse_args()

    asyncio.run(main(args))
