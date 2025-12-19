import numpy as np
from typing import Generator

from mlx_audio.tts.utils import load_model as load_tts

from .base import BaseTTS


class KokoroTTS(BaseTTS):
    """Kokoro TTS backend with voice selection."""

    def __init__(
        self,
        model_id: str = "mlx-community/Kokoro-82M-bf16",
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
        output_sample_rate: int = 24_000,
    ):
        self.model_id = model_id
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.output_sample_rate = output_sample_rate

        self.model = None

    def load(self) -> None:
        """Load the Kokoro model."""
        self.model = load_tts(self.model_id)

    def generate(self, text: str) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text."""
        for chunk in self.model.generate(
            text,
            voice=self.voice,
            speed=self.speed,
            lang_code=self.lang_code,
        ):
            audio_np = np.asarray(chunk.audio, dtype=np.float32)
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767.0).astype(np.int16)
            yield audio_int16.tobytes()

    def warmup(self) -> None:
        """Warm up the TTS model."""
        for _ in self.generate("Hello."):
            pass

    @property
    def sample_rate(self) -> int:
        return self.output_sample_rate
