from abc import ABC, abstractmethod

class BaseVision(ABC):
    """Base interface for vision api."""

    @abstractmethod
    def get_vision_completion(self, prompt: str) -> str:
        """Get prompt completion from vision API."""
        pass