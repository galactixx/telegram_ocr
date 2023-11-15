import numpy as np
import cv2
from cv2.typing import MatLike

class MediaParser:
    """
    Helper methods when parsing media (mp4 or images) using cv2.
    """
    def __init__(self, image_path: str):
        self._image_path = image_path

        # Load in image path into cv2 and convert to gray scale
        self.image = cv2.imread(self._image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Median image metrics
        self._pixel_variance = self._calculate_variance_of_pixel_intensities()

        # Image processing
        self.image = self._image_processing()

        # All contours in relevant image
        self._contours = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._contours = self._contours[0] if len(self._contours) == 2 else self._contours[1]
        self._contour_areas = [cv2.contourArea(c) for c in self._contours]

    def _threshold_type(self, image: MatLike) -> int:
        """Invert the image only if it is white characters on black background."""

        # Calculate the average pixel intensity
        average_intensity = np.mean(image)

        # Determine the thresholding type based on the average intensity
        if average_intensity < 128:
            threshold_type = cv2.THRESH_BINARY
        else:
            threshold_type = cv2.THRESH_BINARY_INV
        return threshold_type
    
    def _calculate_variance_of_pixel_intensities(self) -> float:
        """Calculation of variance of pixel intensities for image."""
        _, std_dev = cv2.meanStdDev(self.image)
        variance = std_dev[0]**2

        # Normalizing by the number of pixels
        num_pixels = self.image.shape[0] * self.image.shape[1]
        normalized_variance = variance / num_pixels

        return normalized_variance[0]

    def _image_processing(self) -> MatLike:
        """Basic image pre-processing on pixel array of image."""
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(self.image, (5, 5), 0)

        # Threshold type to use
        threshold_type = self._threshold_type(image=blurred)

        # Otsu thresholding 
        if self._pixel_variance < 0.001 or self._pixel_variance > 0.002:
            _, thresh = cv2.threshold(blurred, 120, 255, threshold_type + cv2.THRESH_OTSU)

        # Adaptive thresholding
        else:
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, 11, 2)

        return thresh
    
    def remove_small_contours(self, min_percentage_of_total: float = 0.10, area_percentile: float = 90.0) -> None:
        """If needed, remove small contours from image."""

        # Calculate the threshold area based on the percentage
        percent_of_total_threshold_area = sum(self._contour_areas) * min_percentage_of_total

        # Determine the contour area threshold based on the specified percentile
        percentile_threshold_area = np.percentile(self._contour_areas, area_percentile)

        # Iterate through contours and determine if far enough below average to be removed
        for c in self._contours:
            area = cv2.contourArea(c)

            if area < percent_of_total_threshold_area and area < percentile_threshold_area:
                cv2.drawContours(self.image, [c], -1, (0,0,0), -1)