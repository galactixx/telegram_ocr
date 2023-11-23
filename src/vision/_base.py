from abc import ABC, abstractmethod

class BaseVision(ABC):

    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        pass