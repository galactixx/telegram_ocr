import os

from src.media.loader import MediaLoader
from src.media.parser import MediaParser
from src.vision.vision_openai import OpenAIVision
from examples._evals import ocr_evaluation

def example_ocr() -> str:
    """
    Example of how OpenAI OCR functions can be used.
    """

    # Instantiate OpenAI vision
    openai = OpenAIVision()

    # Retrieve all sample media
    media_directory = "./examples/images"

    sample_media = os.listdir(media_directory)

    for media in sample_media:
        media_path = os.path.join(media_directory, media)

        # Instantiate media parser
        media_parser = MediaParser(
            media_loader=MediaLoader(media_path=media_path)
        )
        media_parser.remove_small_contours()

        response = openai.get_completion(
            prompt='What are the largest characters in this image? Only output the text in the image.',
            image=media_parser.image
        )
        
        # OCR evaluation
        ocr_evaluation(image_name=media, prediction=response)

if __name__ == "__main__":
    example_ocr()