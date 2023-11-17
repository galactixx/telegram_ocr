import cv2
from cv2.typing import MatLike

FRAME_TIME = 0

class MediaLoader:
    """
    Load in .jpg/.png/.mp4 file and standardize to image.
    """
    def __init__(self, media_path: str):
        self._media_path = media_path
        self._media_path_lower = media_path.lower()

        if self._media_path_lower.endswith('.mp4'):
            self.image = self._convert_mp4_to_jpg()
        elif (self._media_path_lower.endswith('.jpg') or 
              self._media_path_lower.endswith('.png')):
            self.image = self._load_image()
        else:
            self.image = None
            print('Extension of image path is not recognized...')

    def _load_image(self) -> MatLike:
        """Load image as MatLike object."""
        return cv2.imread(self._media_path)

    def _convert_mp4_to_jpg(self) -> MatLike:
        """Convert mp4 to jpg for ease in image processing."""
        cap = cv2.VideoCapture(self._media_path)

        # Check if the video opened successfully
        if not cap.isOpened():
            print("Error: Could not open video.")

        # Get frames info
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(FRAME_TIME * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, image = cap.read()

        # Check if the frame was read successfully
        if ret:
            cap.release()
            return image