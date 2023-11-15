from abc import ABC, abstractmethod

class BaseInterface(ABC):
    """Base interface for LLM."""

    @abstractmethod
    def get_vision_completion(self, prompt: str) -> str:
        """Get prompt completion from LLM."""
        pass