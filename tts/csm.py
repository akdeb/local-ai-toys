import numpy as np
from typing import Generator, Optional

from mlx_audio.tts.utils import load_model as load_tts
from mlx_audio.tts.generate import load_audio as load_tts_audio

from .base import BaseTTS


class CSMTTS(BaseTTS):
    """CSM (Sesame) TTS backend with voice cloning support."""

    def __init__(
        self,
        model_id: str = "mlx-community/csm-1b-fp16",
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_audio_seconds: int = 15,
        output_sample_rate: int = 24_000,
        streaming_interval: int = 3,
        stt_model=None,
    ):
        self.model_id = model_id
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.ref_audio_seconds = ref_audio_seconds
        self.output_sample_rate = output_sample_rate
        self.streaming_interval = streaming_interval
        self.stt_model = stt_model

        self.model = None
        self.ref_audio = None

    def load(self) -> None:
        """Load the CSM model and reference audio if provided."""
        self.model = load_tts(self.model_id)

        if self.ref_audio_path:
            self.ref_audio = load_tts_audio(
                self.ref_audio_path,
                sample_rate=self.model.sample_rate,
                segment_duration=self.ref_audio_seconds,
            )
            if not self.ref_text and self.stt_model:
                import mlx.core as mx
                result = self.stt_model.generate(self.ref_audio)
                self.ref_text = result.text.strip()

    def generate(self, text: str) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text."""
        generate_kwargs = {
            "sample_rate": self.output_sample_rate,
            "stream": True,
            "streaming_interval": self.streaming_interval,
            "verbose": False,
            "split_pattern": r"[.!?]+",  # Split on sentence endings to avoid duration limit
        }
        if self.ref_audio is not None:
            generate_kwargs["ref_audio"] = self.ref_audio
        if self.ref_text:
            generate_kwargs["ref_text"] = self.ref_text

        for chunk in self.model.generate(text, **generate_kwargs):
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
