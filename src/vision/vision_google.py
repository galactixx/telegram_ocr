import os
from typing import Optional

from google.cloud import vision

from src.utils import parse_ocr_response
from src.vision._base import BaseVision

class GoogleVision(BaseVision):
    """Google AI vision API connection."""
    def __init__(self):
        
        # Retrieve API key
        self._api_key = os.environ['GOOGLE_API_KEY']
        
        # The API key is blank or not set
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY is not set or is blank")

        self._client = vision.ImageAnnotatorClient(
            client_options={'api_key': os.environ['GOOGLE_API_KEY']})

    def get_completion(self, bytes_image: bytes) -> Optional[str]:
        """Get text detection within image from Google AI vision API."""

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