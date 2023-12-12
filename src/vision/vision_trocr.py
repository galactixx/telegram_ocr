import os

from PIL import Image
import cv2
from cv2.typing import MatLike
from torch import Tensor
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel)

from src.utils import parse_ocr_response
from src.vision._base import BaseVision
from src.vision._models import (
    LocalModels,
    ModelsDirectory)

class TrOCR(BaseVision):
    """
    TrOCR local model inference.
    """
    def __init__(
        self,
        model_name: LocalModels = LocalModels.TROCR_LARGE_STR,
        model_directory: ModelsDirectory = ModelsDirectory.TROCR
    ):
        self._model_name = model_name
        self._model_directory = model_directory

        # Get local pretrained models
        local_model_path = os.path.join(self._model_directory.value, self._model_name.value)
        current_model_path = self._model_name.value if not os.path.exists(local_model_path) else local_model_path

        self._processor = TrOCRProcessor.from_pretrained(current_model_path)
        self._model = VisionEncoderDecoderModel.from_pretrained(current_model_path)

        if not os.path.exists(local_model_path):

            self._processor.save_pretrained(local_model_path)
            self._model.save_pretrained(local_model_path)

    def _process_image(self, image: MatLike) -> Tensor:
        """
        Transform open-cv image object into tensor.
        """

        # Convert matlive open-cv object into RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Turn rgb image into pixel values
        pixel_values = self._processor(images=image_pil, return_tensors="pt").pixel_values
        
        return pixel_values

    def get_completion(self, image: MatLike) -> str:
        """
        Get text detection within image from trocr local model.
        """

        pixel_values = self._process_image(image=image)

        # Run inference on image
        output = self._model.generate(pixel_values)
        response = self._processor.batch_decode(output, skip_special_tokens=True)[0]

        result = parse_ocr_response(response=response)
        return result