from typing import Any

from cv2.typing import MatLike
from abc import (
    ABC,
    abstractmethod)

class BaseVision(ABC):
    @abstractmethod
    def _process_image(self, image: MatLike) -> Any:
        pass

    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        pass