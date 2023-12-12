from typing import Optional

from cv2.typing import MatLike
import easyocr

from src.vision._base import BaseVision
from src.utils import (
    encode_image,
    parse_ocr_response)

class EasyOCR(BaseVision):
    """
    Easy OCR API connection.
    """
    def __init__(self):
        self._reader = easyocr.Reader(['en'])

    def _process_image(self, image: MatLike) -> bytes:
        """
        Transform open-cv image object into bytes.
        """

        bytes_image = encode_image(image=image)

        return bytes_image

    def get_completion(self, image: MatLike) -> Optional[str]:
        """
        Get text detection within image from easy OCR API.
        """

        bytes_image = self._process_image(image=image)

        # Use the reader to read text from the bytes
        detections = self._reader.readtext(bytes_image)
        detections_concat = ''.join([text for (_, text, _) in detections])
        
        result = parse_ocr_response(response=detections_concat)

        return result