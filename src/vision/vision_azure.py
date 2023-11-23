import time
import os
from typing import Optional
import io

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

from src.utils import parse_ocr_response
from src.vision._base import BaseVision 

class AzureVision(BaseVision):
    """Microsoft Azure AI vision API connection."""
    def __init__(self):
        self._key = os.environ['AZURE_API_KEY']
        self._endpoint = os.environ['AZURE_ENDPOINT']

        # The API key is blank or not set
        if not self._key:
            raise ValueError("AZURE_API_KEY is not set or is blank")
        
        # The endpoint is blank or not set
        if not self._endpoint:
            raise ValueError("AZURE_ENDPOINT is not set or is blank")

        self._client = ComputerVisionClient(
            self._endpoint, CognitiveServicesCredentials(self._key))

    def get_completion(self, bytes_image: bytes) -> Optional[str]:
        """Get text detection within image from Azure vision API."""

        read_response = self._client.read_in_stream(io.BytesIO(bytes_image), raw=True)

        # Get the operation location (URL with an ID at the end)
        read_operation_location = read_response.headers["Operation-Location"]
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results 
        while True:
            read_result = self._client.get_read_result(operation_id)
            if read_result.status.lower() not in ['notstarted', 'running']:
                break
            time.sleep(1)

        # Print results, line by line
        if read_result.status == OperationStatusCodes.succeeded:
            detections_concat = ''.join(
                [line.text for text_result in read_result.analyze_result.read_results
                 for line in text_result.lines])
            result = parse_ocr_response(response=detections_concat)
            return result