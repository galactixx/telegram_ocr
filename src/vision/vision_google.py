import os
from typing import Optional

from cv2.typing import MatLike
from google.cloud import vision

from src.vision._base import BaseVision
from src.utils import (
    encode_image, 
    parse_ocr_response)

class GoogleVision(BaseVision):
    """
    Google AI vision API connection.
    """
    def __init__(self):
        
        # Retrieve API key
        self._api_key = os.environ.get("GOOGLE_API_KEY")
        
        # The API key is blank or not set
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY is not set or is blank")

        self._client = vision.ImageAnnotatorClient(
            client_options={'api_key': self._api_key})
        
    def _process_image(self, image: MatLike) -> bytes:
        """
        Transform open-cv image object into bytes.
        """

        bytes_image = encode_image(image=image)

        return bytes_image

    def get_completion(self, image: MatLike) -> Optional[str]:
        """
        Get text detection within image from Google AI vision API.
        """

        bytes_image = self._process_image(image=image)

        # Retrieve all detected text in image
        image = vision.Image(content=bytes_image)
        response = self._client.text_detection(image=image)
        detections = response.text_annotations

        # Sort and retrieve longest detected text
        segments = detections[1:]
        sorted_segments = sorted(segments, key=lambda x: len(x.description), reverse=True)
        result = parse_ocr_response(response=sorted_segments[0].description)

        if response.error.message:
            return None
        return result