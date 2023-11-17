import numpy as np
import cv2
from cv2.typing import MatLike

from src.media.loader import MediaLoader

DEFAULT_KERNEL = np.ones((3, 3), np.uint8)

class MediaParser:
    """
    Helper methods when parsing images using cv2.
    """
    def __init__(self, media_loader: MediaLoader):
        self._media_loader = media_loader

        # Load in image path into cv2 and do basic pre-processing
        self.image = self._basic_preprocessing()

        # Image processing
        self.image = self._image_processing()

    def _white_pixels_percent(self, image: MatLike) -> float:
        """Calculate percentage of white pixels in image."""

        # Count the number of white pixels
        number_of_white_pix = cv2.countNonZero(image)

        # Calculate the total number of pixels
        total_pix = image.shape[0] * image.shape[1]

        # Calculate the percentage of white pixels
        percentage = (number_of_white_pix / total_pix) * 100
        return percentage

    def _do_inversion(self, image: MatLike) -> MatLike:
        """Invert the image only if it is white characters on black background."""

        # Calculate the average pixel intensity
        average_intensity = np.mean(image)

        # Determine the thresholding type based on the average intensity
        if average_intensity < 145:
            return cv2.bitwise_not(image)
        return image

    def _image_processing_white_characters(self) -> MatLike:
        """Image pre-processing to highlight white characters."""

        # Generate threshold to highlight white characters
        _, thresh = cv2.threshold(self.image, 245, 255, cv2.THRESH_BINARY)

        # Closing on threshold
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, DEFAULT_KERNEL)

        # Invert black and white
        result = self._do_inversion(image=closing)

        # Erode image
        eroded = cv2.erode(result, DEFAULT_KERNEL, iterations=1)
        return eroded
    
    def _image_processing_black_characters(self) -> MatLike:
        """Image pre-processing to highlight black characters."""

        # Apply thresholding if your image is not binary
        _, thresh = cv2.threshold(self.image, 135, 255, cv2.THRESH_BINARY_INV)

        # Invert the binary image
        inverted_image = self._do_inversion(image=thresh)

        # Erode image
        eroded = cv2.erode(inverted_image, DEFAULT_KERNEL, iterations=1)
        return eroded
    
    def _basic_preprocessing(self) -> MatLike:
        """Basic image pre-processing."""

        # Apply grey scale and gaussian blur to image
        grey = cv2.cvtColor(self._media_loader.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 0)
        return blur

    def _image_processing(self) -> MatLike:
        """Core image pre-processing."""

        # Generate threshold to highlight white & black characters
        image_white = self._image_processing_white_characters()

        # Check percentage of pixels that are white
        if self._white_pixels_percent(image=image_white) > 93.0:
            image_black = self._image_processing_black_characters()
            return image_black
        
        return image_white
     
    def remove_small_contours(self) -> None:
        """If needed, remove small contours from image."""

        # Find contours and hierarchy
        contours, _ = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through contours
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 1000:
                cv2.drawContours(self.image, [contour], 0, (255), -1)