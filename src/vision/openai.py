import os

from openai import OpenAI

from src.vision.base import BaseInterface
from src.vision.models import OpenAIModels

class OpenAIInterface(BaseInterface):
    """Simple interface for OpenAI LLM."""
    def __init__(self,
                 model_name: OpenAIModels = OpenAIModels.GPT_4_VISION,
                 temperature: float = 1.0):
        self.model_name = model_name
        self.temperature = temperature

        # Retrieve API key
        self._open_ai_key = os.environ['OPENAI_API_KEY']

        # The API key is blank or not set
        if not self._open_ai_key:
            raise ValueError("OPENAI_API_KEY is not set or is blank")

        # Instantiate a client object for interacting with the OpenAI API
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        if not isinstance(self.model_name, OpenAIModels):
            raise ValueError(f'{model_name.value} is not a valid model name for OpenAI API')
        
    def get_vision_completion(self, prompt: str, base64_image: str) -> str:
        """Get prompt vision completion for image from OpenAI API."""

        # Chat completion
        response = self.client.chat.completions.create(
            model=self.model_name.value,
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
        
        return response.choices[0].message.content