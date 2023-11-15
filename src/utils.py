import os
import json
from typing import Tuple
import base64

import cv2
from cv2.typing import MatLike

with open('config.json', 'r') as f:
    config = json.load(f)

def _clean_channel(channel: str) -> str:
    return channel.replace('@', '')

def load_api_info() -> Tuple[int, str, str, str]:
    """Load in all API info from both config file and environment variables."""

    # Config variables
    TELEGRAM_CHANNEL = config['telegram_channel']

    # Environment variables
    TELEGRAM_APP_ID = os.environ['TELEGRAM_APP_ID']
    TELEGRAM_APP_HASH = os.environ['TELEGRAM_APP_HASH']
    TELEGRAM_PHONE_NUMBER = os.environ['TELEGRAM_PHONE_NUMBER']
    return TELEGRAM_APP_ID, TELEGRAM_APP_HASH, TELEGRAM_PHONE_NUMBER, TELEGRAM_CHANNEL

def source_data_directory(channel: str) -> None:
    """Return source data directory."""
    return f"{config['data_dir']}/{_clean_channel(channel=channel)}"

def source_data_directories(channel: str) -> None:
    """Create source data directories for parsed data."""

    path = source_data_directory(channel=channel)
    os.makedirs(path, exist_ok=True)

def encode_image(image: MatLike) -> str:
    """Given MatLike object from cv2, return base64 representation of image."""

    # Convert the image to bytes
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    
    # Convert the bytes to a base64 string
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64