"""Text-to-speech utilities with pyttsx3 primary and gTTS fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class TTSEngine:
    """Speak text through the local TTS engine, with gTTS as a fallback."""

    def __init__(self, rate: int = 175, volume: float = 0.9, language: str = "en") -> None:
        self.rate = rate
        self.volume = volume
        self.language = language
        self.available = False
        self.backend = "none"
        self.last_error: Optional[str] = None
        self._engine = None

        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            self.available = True
            self.backend = "pyttsx3"
        except Exception as exc:
            self.last_error = str(exc)

        try:
            from gtts import gTTS

            self._gtts = gTTS
            if not self.available:
                self.available = True
                self.backend = "gtts"
        except Exception as exc:
            self._gtts = None
            if self.last_error is None:
                self.last_error = str(exc)

    def status(self) -> dict:
        return {
            "available": self.available,
            "backend": self.backend,
            "rate": self.rate,
            "volume": self.volume,
            "language": self.language,
            "last_error": self.last_error,
        }

    def speak(self, text: str) -> bool:
        """Speak text aloud."""

        if not text:
            self.last_error = "No text provided"
            return False

        if self._engine is not None:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
                return True
            except Exception as exc:
                self.last_error = f"pyttsx3 playback failed: {exc}"

        if self._gtts is None:
            if self.last_error is None:
                self.last_error = "No TTS backend available"
            return False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_path = Path(temp_file.name)
        try:
            self._gtts(text=text, lang=self.language).save(str(temp_path))
            return self._play_audio_file(temp_path)
        except Exception as exc:
            self.last_error = f"gTTS playback failed: {exc}"
            return False
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def save_to_file(self, text: str, filename: str = "response.wav") -> bool:
        """Save spoken audio to a file."""

        if not text:
            self.last_error = "No text provided"
            return False

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._engine is not None:
            try:
                self._engine.save_to_file(text, str(output_path))
                self._engine.runAndWait()
                return True
            except Exception as exc:
                self.last_error = f"pyttsx3 save failed: {exc}"

        if self._gtts is None:
            return False

        target_path = output_path if output_path.suffix.lower() == ".mp3" else output_path.with_suffix(".mp3")
        try:
            self._gtts(text=text, lang=self.language).save(str(target_path))
            return True
        except Exception as exc:
            self.last_error = f"gTTS save failed: {exc}"
            return False

    def _play_audio_file(self, audio_path: Path) -> bool:
        try:
            if os.name == "nt":
                os.startfile(str(audio_path))
                return True

            if shutil.which("open") is not None:
                subprocess.Popen(["open", str(audio_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True

            for player in ("ffplay", "mpg123", "vlc", "cvlc", "xdg-open"):
                executable = shutil.which(player)
                if executable is None:
                    continue
                command = [executable, str(audio_path)]
                if player == "ffplay":
                    command = [executable, "-nodisp", "-autoexit", str(audio_path)]
                subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        except Exception as exc:
            self.last_error = f"Audio playback failed: {exc}"
            return False

        self.last_error = "No local audio player found for gTTS fallback"
        return False


if __name__ == "__main__":
    engine = TTSEngine()
    print(engine.status())