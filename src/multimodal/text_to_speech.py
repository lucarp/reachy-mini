"""Text-to-speech using Piper for fast, natural-sounding synthesis."""

import logging
import wave
from pathlib import Path
from typing import Optional, Generator
from piper import PiperVoice
from piper.voice import SynthesisConfig as PiperSynthesisConfig
from ..utils.config import TextToSpeechConfig

logger = logging.getLogger(__name__)


class PiperTTS:
    """Piper-based text-to-speech with streaming support."""

    def __init__(self, config: TextToSpeechConfig):
        """Initialize Piper TTS.

        Args:
            config: Text-to-speech configuration
        """
        self.config = config
        self.voice = None

        logger.info("Initializing Piper TTS")
        self._load_voice()

    def _load_voice(self):
        """Load Piper voice model."""
        try:
            if not self.config.voice_path:
                raise ValueError("Voice path not configured. Set PIPER_VOICE_PATH in .env")

            voice_path = Path(self.config.voice_path)

            if not voice_path.exists():
                raise FileNotFoundError(f"Voice model not found: {voice_path}")

            # Load voice
            self.voice = PiperVoice.load(str(voice_path))

            logger.info(f"Loaded Piper voice: {self.config.voice} from {voice_path}")

        except Exception as e:
            logger.error(f"Error loading Piper voice: {e}")
            raise

    def _get_synthesis_config(self) -> PiperSynthesisConfig:
        """Get synthesis configuration.

        Returns:
            PiperSynthesisConfig instance
        """
        syn_config = self.config.synthesis_config

        return PiperSynthesisConfig(
            volume=syn_config.volume,
            length_scale=syn_config.length_scale,
            noise_scale=syn_config.noise_scale,
            noise_w_scale=syn_config.noise_w_scale,
        )

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
    ) -> bool:
        """Synthesize text to audio file.

        Args:
            text: Text to synthesize
            output_path: Path to save WAV file

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Get synthesis config
            syn_config = self._get_synthesis_config()

            # Synthesize to WAV file
            with wave.open(str(output_path), "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)

            logger.info(f"Synthesized text to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error synthesizing to file: {e}")
            return False

    def synthesize_stream(
        self,
        text: str,
    ) -> Generator[bytes, None, None]:
        """Synthesize text to audio stream.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks as bytes
        """
        try:
            syn_config = self._get_synthesis_config()

            # Stream audio chunks
            for chunk in self.voice.synthesize(text, syn_config=syn_config):
                # chunk.audio_int16_bytes contains raw PCM audio
                yield chunk.audio_int16_bytes

        except Exception as e:
            logger.error(f"Error in streaming synthesis: {e}")
            yield b""

    def get_audio_format(self) -> dict:
        """Get audio format information.

        Returns:
            Dictionary with sample_rate, sample_width, channels
        """
        # Piper typically outputs 16-bit PCM
        return {
            "sample_rate": self.config.sample_rate,
            "sample_width": 2,  # 16-bit = 2 bytes
            "channels": 1,      # Mono
        }
