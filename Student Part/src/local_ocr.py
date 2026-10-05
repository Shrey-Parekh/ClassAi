from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class LocalOcr:
    def __init__(self, prefer_tesseract: bool = False):
        self.prefer_tesseract = prefer_tesseract

    def image_to_text(self, image: Image.Image) -> str:
        try:
            engine = get_rapidocr_engine()
        except ImportError as exc:
            raise RuntimeError(
                "Local OCR requires RapidOCR. Install dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        processed = preprocess_for_ocr(image)
        result = engine(np.array(processed))
        texts = getattr(result, "txts", None) or ()
        if not texts:
            return ""
        return "\n".join(str(text) for text in texts).strip()


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.SHARPEN)
    return gray.convert("RGB")


@lru_cache(maxsize=1)
def get_rapidocr_engine():
    from rapidocr import RapidOCR

    return RapidOCR()
