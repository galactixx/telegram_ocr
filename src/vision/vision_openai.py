import os
from typing import Optional

from openai import OpenAI

from src.vision._base import BaseVision
from src.vision._models import OpenAIModels
from src.utils import parse_ocr_response

class OpenAIVision(BaseVision):
    """OpenAI vision API connection."""
    def __init__(self,
                 model_name: OpenAIModels = OpenAIModels.GPT_4_VISION,
                 temperature: float = 1.0):
        self._model_name = model_name
        self._temperature = temperature

        # Retrieve API key
        self._api_key = os.environ['OPENAI_API_KEY']

        # The API key is blank or not set
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is not set or is blank")

        # Instantiate a client object for interacting with the OpenAI API
        self._client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        if not isinstance(self._model_name, OpenAIModels):
            raise ValueError(f'{model_name.value} is not a valid model name for OpenAI API')

    def get_vision_completion(self, prompt: str, base64_image: str) -> Optional[str]:
        """Get prompt vision completion for image from OpenAI vision API."""

        response = self._client.chat.completions.create(
            model=self._model_name.value,
            messages=[
            {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
            }
        ])
        detected_text = parse_ocr_response(response=response.choices[0].message.content)
        return detected_text