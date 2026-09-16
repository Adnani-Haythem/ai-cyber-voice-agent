"""Speech recognition utilities for files and microphone input."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional


class ASREngine:
    """Speech-to-text helper built on SpeechRecognition."""

    def __init__(self, model_name: str = "base", language: str = "en-US") -> None:
        self.model_name = model_name
        self.language = language
        self.available = False
        self.backend = "speech_recognition"
        self.fallback_backend = "sounddevice"
        self.last_error: Optional[str] = None
        self._sr = None
        self._sd = None
        self._np = None
        self._recognizer = None

        try:
            import speech_recognition as sr

            self._sr = sr
            self._recognizer = sr.Recognizer()
            self.available = True
        except Exception as exc:
            self.last_error = str(exc)

        try:
            import sounddevice as sd

            self._sd = sd
        except Exception:
            self._sd = None

        try:
            import numpy as np

            self._np = np
        except Exception:
            self._np = None

    def status(self) -> dict:
        return {
            "available": self.available,
            "backend": self.backend,
            "fallback_backend": self.fallback_backend,
            "language": self.language,
            "model_name": self.model_name,
            "last_error": self.last_error,
        }

    def transcribe_file(self, audio_path: str) -> Optional[str]:
        """Transcribe an audio file to text."""

        if not self.available or self._sr is None or self._recognizer is None:
            self.last_error = "speech_recognition is not available"
            return None

        source_path = Path(audio_path)
        if not source_path.exists():
            self.last_error = f"Audio file not found: {audio_path}"
            return None

        normalized_path = self._prepare_audio_file(source_path)
        if normalized_path is None:
            return None

        try:
            with self._sr.AudioFile(str(normalized_path)) as source:
                audio_data = self._recognizer.record(source)

            transcript = self._recognizer.recognize_google(
                audio_data,
                language=self.language,
            )
            return transcript.strip()
        except self._sr.UnknownValueError:
            self.last_error = "Could not understand the audio"
            return None
        except self._sr.RequestError as exc:
            self.last_error = f"Speech recognition request failed: {exc}"
            return None
        except Exception as exc:
            self.last_error = f"Transcription failed: {exc}"
            return None
        finally:
            if normalized_path != source_path and normalized_path.exists():
                try:
                    normalized_path.unlink()
                except OSError:
                    pass

    def transcribe_microphone(self, duration: int = 5, sample_rate: int = 16000) -> Optional[str]:
        """Record from microphone and transcribe it."""

        if self.available and self._sr is not None and self._recognizer is not None:
            try:
                with self._sr.Microphone(sample_rate=sample_rate) as source:
                    self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self._recognizer.record(source, duration=duration)
                return self._recognizer.recognize_google(audio_data, language=self.language).strip()
            except Exception as exc:
                self.last_error = f"Microphone recognition failed: {exc}"

        return self.record_microphone(duration=duration, sample_rate=sample_rate)

    def record_microphone(self, duration: int = 5, sample_rate: int = 16000) -> Optional[str]:
        """Record with sounddevice when SpeechRecognition's microphone path is unavailable."""

        if self._sd is None or self._np is None:
            self.last_error = "sounddevice and numpy are required for microphone fallback"
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = Path(temp_file.name)
        try:
            channels = 1
            frames = int(duration * sample_rate)
            recording = self._sd.rec(frames, samplerate=sample_rate, channels=channels, dtype="float32")
            self._sd.wait()

            scaled = self._np.clip(recording, -1.0, 1.0)
            pcm_audio = (scaled * 32767).astype(self._np.int16)

            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_audio.tobytes())

            return self.transcribe_file(str(temp_path))
        except Exception as exc:
            self.last_error = f"sounddevice recording failed: {exc}"
            return None
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _prepare_audio_file(self, source_path: Path) -> Optional[Path]:
        supported_extensions = {".wav", ".aiff", ".aif", ".flac"}
        if source_path.suffix.lower() in supported_extensions:
            return source_path

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.last_error = (
                f"Unsupported audio format: {source_path.suffix}. Install ffmpeg or upload WAV/FLAC/AIFF."
            )
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            converted_path = Path(temp_file.name)
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(converted_path),
        ]

        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                self.last_error = completed.stderr.strip() or "Audio conversion failed"
                try:
                    converted_path.unlink()
                except OSError:
                    pass
                return None
            return converted_path
        except Exception as exc:
            self.last_error = f"Audio conversion failed: {exc}"
            try:
                converted_path.unlink()
            except OSError:
                pass
            return None


if __name__ == "__main__":
    engine = ASREngine()
    print(engine.status())