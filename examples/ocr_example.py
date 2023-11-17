import os

from src.media.loader import MediaLoader
from src.media.parser import MediaParser
from src.vision.openai import OpenAIInterface
from src.utils import encode_image

def example_ocr() -> str:
    """Example of how OCR functions can be used."""

    # Instantiate OpenAI vision
    openai = OpenAIInterface()

    # Retrieve all sample media
    media_directory = "./examples/images"

    sample_media = os.listdir(media_directory)

    for media in sample_media:
        media_path = os.path.join(media_directory, media)

        # Instantiate media parser
        media_parser = MediaParser(
            media_loader=MediaLoader(media_path=media_path))
        media_parser.remove_small_contours()

        # After image processing, encode image to base64 representation
        base64_image = encode_image(image=media_parser.image)

        response = openai.get_vision_completion(
            prompt='What are the largest characters in this image? Only output the text in the image.',
            base64_image=base64_image)
        print(f'Text detected in {media}: {response}')

if __name__ == "__main__":
    example_ocr()