from enum import Enum

class OpenAIModels(Enum):
    """All OpenAI models available through API"""
    GPT_4_VISION = 'gpt-4-vision-preview'

class LocalModels(Enum):
    """All local models available."""
    TROCR_LARGE_STR = 'microsoft/trocr-large-str'
    TROCR_LARGE_STAGE_1 = 'microsoft/trocr-large-stage1'
    TROCR_LARGE_PRINTED = 'microsoft/trocr-large-printed'
    TROCR_LARGE_HAND_WRITTEN = 'microsoft/trocr-large-handwritten'

class ModelsDirectory(Enum):
    """All local models directories."""
    TROCR = './models/trocr'