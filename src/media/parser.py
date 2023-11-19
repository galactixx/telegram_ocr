import numpy as np
import cv2
from cv2.typing import MatLike

from src.media.loader import MediaLoader

DEFAULT_KERNEL = np.ones((2, 2), np.uint8)
WHITE_THRESHOLD = (240, 240, 240)
LIGHT_GREY_THRESHOLD = (150, 150, 150)

class MediaParser:
    """
    Helper methods when parsing images using cv2.
    """
    def __init__(self,
                 media_loader: MediaLoader,
                 white_pixel_threshold: int = 0,
                 light_grey_pixel_threshold: int = 30):
        self._media_loader = media_loader
        self._white_pixel_threshold = white_pixel_threshold
        self._light_grey_pixel_threshold = light_grey_pixel_threshold
        
        self._white_pixel_percentage = self._calculate_white_pixels_percentage(
            image=self._media_loader.image,
            threshold=WHITE_THRESHOLD)
        self._light_grey_pixel_percentage = self._calculate_white_pixels_percentage(
            image=self._media_loader.image,
            threshold=LIGHT_GREY_THRESHOLD)

        # Basic image pre-processing
        self.image = self._basic_preprocessing(image=self._media_loader.image)

        # Image pre-processing
        self.image = self._image_processing()

    def _calculate_white_pixels_percentage(self, image: MatLike, threshold: tuple) -> float:
        """Calculate the percentage of white or near-white pixels using NumPy."""

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a boolean mask where white or near-white pixels are True
        mask = np.all(img_rgb >= threshold, axis=-1)
        white_pixels_percentage = np.sum(mask) / mask.size * 100
        return white_pixels_percentage

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
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Opening on threshold
        eroded = cv2.erode(thresh, DEFAULT_KERNEL, iterations=3)
        dilated = cv2.dilate(eroded, DEFAULT_KERNEL, iterations=4)

        # Invert black and white and erode image
        inverted = self._do_inversion(image=dilated)
        return inverted
 
    def _image_processing_black_characters(self) -> MatLike:
        """Image pre-processing to highlight black characters."""

        # Apply thresholding if your image is not binary
        _, thresh = cv2.threshold(self.image, 135, 255, cv2.THRESH_BINARY_INV)

        # Invert the binary image and erode image
        inverted = self._do_inversion(image=thresh)
        eroded = cv2.erode(inverted, DEFAULT_KERNEL, iterations=1)
        return eroded
    
    def _clahe_transformation(self, image: MatLike) -> MatLike:
        """Clahe transformation on l-channel in image to increase contrast."""

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
            
        # Convert the image to LAB color space and split channels
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L-channel and merge channels
        cl = clahe.apply(l)
        lab_image = cv2.merge((cl, a, b))
        return lab_image

    def _basic_preprocessing(self, image: MatLike) -> MatLike:
        """Basic image pre-processing."""

        # Do clahe transformation only if there are no white/near-white pixels
        if (self._white_pixel_percentage == self._white_pixel_threshold and
            self._light_grey_pixel_percentage < self._light_grey_pixel_threshold):
            image = self._clahe_transformation(image=image)
            print('Here')

        # Apply grey scale and gaussian blur to image
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 0)
        return blur

    def _image_processing(self) -> MatLike:
        """Core image pre-processing."""

        if self._light_grey_pixel_percentage > self._light_grey_pixel_threshold:
            image = self._image_processing_black_characters()
        else:
            image = self._image_processing_white_characters()
        
        return image

    def remove_small_contours(self) -> None:
        """If needed, remove small contours from image."""

        # Find contours and hierarchy
        contours, _ = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through contours
        for contour in contours:
            area = cv2.contourArea(contour)

            if area < 1000:
                cv2.drawContours(self.image, [contour], 0, (255), -1)