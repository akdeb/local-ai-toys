import numpy as np
from typing import Generator, Optional

from mlx_audio.tts.utils import load_model as load_tts

from .base import BaseTTS


class VoxCPMTTS(BaseTTS):
    """VoxCPM TTS backend with voice cloning support."""

    # Chunk size in samples (~100ms at 44.1kHz = 4410 samples)
    CHUNK_SAMPLES = 4410

    def __init__(
        self,
        model_id: str = "mlx-community/VoxCPM1.5-fp16",
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        output_sample_rate: int = 44_100,
        max_tokens: int = 4096,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
    ):
        self.model_id = model_id
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.output_sample_rate = output_sample_rate
        self.max_tokens = max_tokens
        self.inference_timesteps = inference_timesteps
        self.cfg_value = cfg_value

        self.model = None
        self._ref_audio = None

    def load(self) -> None:
        """Load the VoxCPM model and prepare reference audio if provided."""
        self.model = load_tts(self.model_id)

        # Pre-load reference audio if provided
        if self.ref_audio_path:
            import librosa
            self._ref_audio, _ = librosa.load(
                self.ref_audio_path,
                sr=self.output_sample_rate,
                mono=True,
            )
            import mlx.core as mx
            self._ref_audio = mx.array(self._ref_audio)

    def generate(self, text: str) -> Generator[bytes, None, None]:
        """Generate audio chunks for the given text."""
        for chunk in self.model.generate(
            text,
            max_tokens=self.max_tokens,
            ref_text=self.ref_text,
            ref_audio=self._ref_audio,
            inference_timesteps=self.inference_timesteps,
            cfg_value=self.cfg_value,
        ):
            audio_np = np.asarray(chunk.audio, dtype=np.float32)
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767.0).astype(np.int16)

            # Chunk the audio to avoid WebSocket message size limits
            for i in range(0, len(audio_int16), self.CHUNK_SAMPLES):
                audio_chunk = audio_int16[i : i + self.CHUNK_SAMPLES]
                yield audio_chunk.tobytes()

    def warmup(self) -> None:
        """Warm up the TTS model."""
        for _ in self.generate("Hello."):
            pass

    @property
    def sample_rate(self) -> int:
        return self.output_sample_rate
