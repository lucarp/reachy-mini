"""Speech-to-text using WhisperX for fast, accurate transcription."""

import logging
import numpy as np
import whisperx
from typing import Optional, Dict, Any
from ..utils.config import SpeechToTextConfig

logger = logging.getLogger(__name__)


class WhisperXTranscriber:
    """WhisperX-based speech transcription with word-level timestamps."""

    def __init__(self, config: SpeechToTextConfig):
        """Initialize WhisperX transcriber.

        Args:
            config: Speech-to-text configuration
        """
        self.config = config
        self.model = None
        self.align_model = None
        self.align_metadata = None

        logger.info("Initializing WhisperX transcriber")
        self._load_models()

    def _load_models(self):
        """Load WhisperX models."""
        try:
            # Load transcription model
            self.model = whisperx.load_model(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )

            logger.info(
                f"Loaded WhisperX model: {self.config.model} "
                f"(device: {self.config.device}, compute_type: {self.config.compute_type})"
            )

            # Load alignment model for word-level timestamps
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.config.language,
                device=self.config.device,
            )

            logger.info(f"Loaded alignment model for language: {self.config.language}")

        except Exception as e:
            logger.error(f"Error loading WhisperX models: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Transcribe audio to text with word-level timestamps.

        Args:
            audio: Audio data as numpy array
            sample_rate: Audio sample rate (default: 16000)

        Returns:
            Dictionary with transcription results:
            {
                "text": str,              # Full transcription
                "segments": List[dict],   # Segments with timestamps
                "language": str,          # Detected language
                "word_segments": List[dict]  # Word-level timestamps
            }
        """
        try:
            # Transcribe with WhisperX
            result = self.model.transcribe(
                audio,
                batch_size=self.config.batch_size,
                language=self.config.language,
            )

            # Align for word-level timestamps
            aligned_result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.config.device,
                return_char_alignments=False,
            )

            # Extract full text
            full_text = " ".join(
                segment.get("text", "").strip()
                for segment in aligned_result["segments"]
            )

            return {
                "text": full_text,
                "segments": aligned_result["segments"],
                "language": result.get("language", self.config.language),
                "word_segments": aligned_result.get("word_segments", []),
            }

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {
                "text": "",
                "segments": [],
                "language": self.config.language,
                "word_segments": [],
                "error": str(e),
            }

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcription results dictionary
        """
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path)

            # Transcribe
            return self.transcribe(audio)

        except Exception as e:
            logger.error(f"Error transcribing file {audio_path}: {e}")
            return {
                "text": "",
                "segments": [],
                "language": self.config.language,
                "error": str(e),
            }

    def cleanup(self):
        """Clean up models to free memory."""
        import gc
        import torch

        del self.model
        del self.align_model
        del self.align_metadata

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("WhisperX models cleaned up")
