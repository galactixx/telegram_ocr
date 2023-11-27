from typing import Any

from abc import ABC, abstractmethod
from cv2.typing import MatLike

class BaseVision(ABC):
    @abstractmethod
    def _process_image(self, image: MatLike) -> Any:
        pass

    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        pass