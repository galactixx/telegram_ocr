from typing import Tuple

import numpy as np
import cv2
from cv2.typing import MatLike
from numpy.typing import NDArray

from src.media.loader import MediaLoader

WHITE_THRESHOLD = (240, 240, 240)
LIGHT_GREY_THRESHOLD = (150, 150, 150)

class MediaParser:
    """
    Image pre-processing methods for OCR analysis.
    """
    def __init__(self,
                 media_loader: MediaLoader,
                 pixel_threshold: int = 15000,
                 inversion_threshold: int = 145,
                 white_pixel_threshold: float = 0.0,
                 light_grey_pixel_threshold: float = 0.30,
                 contour_area_threshold: float = 0.008,
                 contour_area_number_threshold: int = 100,
                 contour_alignment_threshold: float = 0.15,
                 contour_alignment_deviation: float = 1.50):
        self._media_loader = media_loader
        self._pixel_threshold = pixel_threshold
        self._inversion_threshold = inversion_threshold
        self._white_pixel_threshold = white_pixel_threshold
        self._light_grey_pixel_threshold = light_grey_pixel_threshold
        self._contour_area_threshold = contour_area_threshold
        self._contour_area_number_threshold = contour_area_number_threshold
        self._contour_alignment_threshold = contour_alignment_threshold
        self._contour_alignment_deviation = contour_alignment_deviation

        self._image_width = self._media_loader.image.shape[0]
        self._image_height = self._media_loader.image.shape[1]
        self._image_area = self._image_width * self._image_height
        self._center_y = self._image_width // 2

        self._white_pixel_percentage = self._calculate_pixels_percentage(
            image=self._media_loader.image,
            threshold=WHITE_THRESHOLD)
        self._light_grey_pixel_percentage = self._calculate_pixels_percentage(
            image=self._media_loader.image,
            threshold=LIGHT_GREY_THRESHOLD)

        self._erosion_iterations, self._kernel = self._kernel_choice()

        # Basic image pre-processing
        self.image = self._basic_preprocessing(image=self._media_loader.image)

        # Image pre-processing
        self.image = self._image_processing()

    def _kernel_choice(self) -> Tuple[int, NDArray]:
        """Decide size of kernel and erosion iterations depending on size of image (pixels)."""
        if self._image_area * self._white_pixel_percentage < self._pixel_threshold:
            return 3, np.ones((2, 2), np.uint8)
        else:
            return 4, np.ones((3, 3), np.uint8)

    def _calculate_pixels_percentage(self, image: MatLike, threshold: tuple) -> float:
        """Calculate the percentage of white or near-white pixels using NumPy."""

        # Convert the image to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create a boolean mask where white or near-white pixels are True
        mask = np.all(img_rgb >= threshold, axis=-1)
        white_pixels_percentage = np.sum(mask) / mask.size
        return white_pixels_percentage

    def _do_inversion(self, image: MatLike) -> MatLike:
        """Invert the image only if it is white characters on black background."""

        # Calculate the average pixel intensity
        average_intensity = np.mean(image)

        # Determine the thresholding type based on the average intensity
        if average_intensity < self._inversion_threshold:
            return cv2.bitwise_not(image)
        return image

    def _image_processing_white_characters(self) -> MatLike:
        """Image pre-processing to highlight white characters."""

        # Generate threshold to highlight white characters
        _, thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Opening on threshold
        eroded = cv2.erode(thresh, self._kernel, iterations=self._erosion_iterations)
        dilated = cv2.dilate(eroded, self._kernel, iterations=9)

        # Invert black and white and erode image
        inverted = self._do_inversion(image=dilated)
        return inverted
 
    def _image_processing_black_characters(self) -> MatLike:
        """Image pre-processing to highlight black characters."""

        # Apply thresholding if your image is not binary
        _, thresh = cv2.threshold(self.image, 135, 255, cv2.THRESH_BINARY_INV)

        # Invert the binary image and erode image
        inverted = self._do_inversion(image=thresh)
        eroded = cv2.erode(inverted, self._kernel, iterations=1)
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
        if len(contours) >= self._contour_area_number_threshold:
            for contour in contours:
                area = cv2.contourArea(contour)

                if (area / self._image_area) < self._contour_area_threshold:
                    cv2.drawContours(self.image, [contour], 0, (255), -1)

    def realign_and_center_contours(self) -> None:
        """Re-align and center specific contours that are noticeably not centered."""

        def contour_mid_point_difference(y: float, h: float) -> float:
            return (y + y + h) // 2 - self._center_y

        # Find contours and hierarchy
        contour_diffs = []
        contours, _ = cv2.findContours(self.image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            _, y, _, h = cv2.boundingRect(contour)
            contour_diffs.append(
                contour_mid_point_difference(y=y, h=h)
            )

        contour_diffs_std = np.std(contour_diffs)
        contour_diffs_mean = np.mean(contour_diffs)

        # Center of the image and iterate through contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_mid_point_diff = contour_mid_point_difference(y=y, h=h)

            if (abs(contour_mid_point_diff) / self._image_width > self._contour_alignment_threshold and
                (contour_mid_point_diff - contour_diffs_mean) / contour_diffs_std > self._contour_alignment_deviation):

                character = self.image[y:y+h, x:x+w].copy()
                cv2.drawContours(self.image, [contour], 0, (255), -1)

                # Calculate new position and re-place
                new_y = self._center_y - h // 2
                self.image[new_y:new_y+h, x:x+w] = character

        cv2.imshow('image', self.image)
        cv2.waitKey(0)