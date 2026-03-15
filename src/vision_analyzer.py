import base64
import io
import requests
from typing import Optional

from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


class VisionAnalyzer:
    """Analyzes images using an Ollama vision model and returns text descriptions."""

    def __init__(self,
                 model_name: str = "gemma3",
                 ollama_base_url: str = "http://localhost:11434",
                 max_image_dimension: int = 1024):
        self._model_name = model_name
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._max_image_dimension = max_image_dimension

        self._prompt = (
            "Describe this image in detail. Include:\n"
            "- What is happening in the scene (activities, events)\n"
            "- People: how many, what they're doing\n"
            "- Setting and location (indoor/outdoor, type of place, landmarks)\n"
            "- Objects and notable features\n"
            "- Time of day and weather if visible\n"
            "- Any text visible in the image\n"
            "- The mood or atmosphere\n"
            "Provide a natural, descriptive paragraph."
        )

    def _resize_image(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        if w <= self._max_image_dimension and h <= self._max_image_dimension:
            return image
        ratio = min(self._max_image_dimension / w, self._max_image_dimension / h)
        new_size = (int(w * ratio), int(h * ratio))
        return image.resize(new_size, Image.LANCZOS)

    def _image_to_base64(self, file_path: str) -> str:
        with Image.open(file_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img = self._resize_image(img)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def analyze(self, file_path: str) -> Optional[str]:
        try:
            image_b64 = self._image_to_base64(file_path)

            response = requests.post(
                f"{self._ollama_base_url}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": self._prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                    }
                },
                timeout=120
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip() or None

        except Exception as e:
            print(f"  Warning: Vision analysis failed for {file_path}: {e}")
            return None
