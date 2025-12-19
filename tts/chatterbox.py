import numpy as np
from typing import Generator, Optional

from mlx_audio.tts.utils import load_model as load_tts

from .base import BaseTTS


class ChatterboxTTS(BaseTTS):
    """Chatterbox Turbo TTS backend with voice cloning support."""

    def __init__(
        self,
        model_id: str = "mlx-community/chatterbox-turbo-4bit",
        ref_audio_path: Optional[str] = None,
        output_sample_rate: int = 24_000,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
    ):
        self.model_id = model_id
        self.ref_audio_path = ref_audio_path
        self.output_sample_rate = output_sample_rate
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        self.model = None

    def load(self) -> None:
        """Load the Chatterbox model and prepare conditionals if ref audio provided."""
        self.model = load_tts(self.model_id)

        if self.ref_audio_path:
            self.model.prepare_conditionals(self.ref_audio_path)

    def generate(self, text: str) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text."""
        for chunk in self.model.generate(
            text,
            ref_audio=None,  # Already prepared via prepare_conditionals
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
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
