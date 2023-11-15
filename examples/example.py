from src.media.parser import MediaParser
from src.vision.openai import OpenAIInterface
from src.utils import encode_image

def example_ocr() -> str:
    """Example of how OCR functions can be used."""

    # Instantiate OpenAI vision
    openai = OpenAIInterface()

    # Instantiate media parser
    media_parser = MediaParser(image_path='test-photo-5.jpg')
    media_parser.remove_small_contours()

    # After image processing, encode image to base64 representation
    base64_image = encode_image(image=media_parser.image)

    response = openai.get_vision_completion(
        prompt='What are the largest characters in this image? Only output the text in the image.',
        base64_image=base64_image)
    return response

if __name__ == "__main__":
    response = example_ocr()
    print(response)