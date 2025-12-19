from abc import ABC, abstractmethod
from typing import Generator, Any


class BaseTTS(ABC):
    """Base class for TTS backends."""

    @abstractmethod
    def load(self) -> None:
        """Load the TTS model."""
        pass

    @abstractmethod
    def generate(self, text: str) -> Generator[bytes, None, None]:
        """
        Generate audio for the given text.
        
        Yields:
            Audio chunks as int16 PCM bytes.
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the TTS model to avoid first-utterance latency."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the output sample rate."""
        pass
