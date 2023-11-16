import os

from src.media.parser import MediaParser
from src.vision.openai import OpenAIInterface
from src.utils import encode_image

def example_ocr() -> str:
    """Example of how OCR functions can be used."""

    # Instantiate OpenAI vision
    openai = OpenAIInterface()

    # Retrieve all sample images
    images_directory = "./examples/images"
    sample_images = os.listdir(images_directory)

    for image in sample_images:
        image_path = os.path.join(images_directory, image)

        # Instantiate media parser
        media_parser = MediaParser(image_path=image_path)
        media_parser.remove_small_contours()

        # After image processing, encode image to base64 representation
        base64_image = encode_image(image=media_parser.image)

        response = openai.get_vision_completion(
            prompt='What are the largest characters in this image? Only output the text in the image.',
            base64_image=base64_image)
        print(f'Text detected in {image}: {response}')

if __name__ == "__main__":
    example_ocr()