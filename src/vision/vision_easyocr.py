from typing import Optional

import easyocr

from src.utils import parse_ocr_response
from src.vision.base import BaseVision

class EasyOCR(BaseVision):
    def __init__(self):
        self._reader = easyocr.Reader(['en'])

    def get_vision_completion(self, bytes_image: bytes) -> Optional[str]:
        """Get text detection within image from easy OCR API."""

        # Use the reader to read text from the bytes
        detections = self._reader.readtext(bytes_image)
        detections_concat = ''.join([text for (_, text, _) in detections])
        
        result = parse_ocr_response(response=detections_concat)

        return result