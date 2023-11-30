import os
import json
import base64
import re
from typing import Optional

import numpy as np
from dataclasses import dataclass
import cv2
from cv2.typing import MatLike

@dataclass
class TelegramInfo:
    app_id: int
    app_hash: str
    phone_number: str
    channel: str
    channel_to_send: str
    channel_keywords: list

with open('config.json', 'r') as f:
    config = json.load(f)

def clean_channel(channel: str) -> str:
    return channel.replace('@', '')

def load_api_info() -> TelegramInfo:
    """Load in all API info from both config file and environment variables."""

    # Config variables
    TELEGRAM_CHANNEL = config['telegram_channel']
    TELEGRAM_CHANNEL_TO_SEND = config['telegram_channel_to_send']
    TELEGRAM_CHANNEL_KEYWORDS = config['telegram_keywords']

    # Environment variables
    TELEGRAM_APP_ID = os.environ['TELEGRAM_APP_ID']
    TELEGRAM_APP_HASH = os.environ['TELEGRAM_APP_HASH']
    TELEGRAM_PHONE_NUMBER = os.environ['TELEGRAM_PHONE_NUMBER']

    return TelegramInfo(
        app_id=TELEGRAM_APP_ID,
        app_hash=TELEGRAM_APP_HASH,
        phone_number=TELEGRAM_PHONE_NUMBER,
        channel=TELEGRAM_CHANNEL,
        channel_to_send=TELEGRAM_CHANNEL_TO_SEND,
        channel_keywords=TELEGRAM_CHANNEL_KEYWORDS
    )

def source_data_directory(channel: str) -> None:
    """Return source data directory."""

    return f"{config['data_dir']}/{clean_channel(channel=channel)}"

def source_data_directories(channel: str) -> None:
    """Create source data directories for parsed data."""

    path = source_data_directory(channel=channel)
    os.makedirs(path, exist_ok=True)

def encode_image(image: MatLike) -> bytes:
    """Given MatLike object from cv2, return bytes representation of image."""

    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()
    return img_bytes

def encode_image_base64(image: MatLike) -> bytes:
    """Given MatLike object from cv2, return base64 representation of image."""

    img_bytes = encode_image(image=image)
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def parse_ocr_response(response: str) -> Optional[str]:
    """Logic to parse and clean the OCR response."""
    
    # Remove non-letter characters
    cleaned_response = re.sub('[^A-Za-z]+', '', response).strip().upper()

    # Ignore reponse if length is more than 5 letters
    if len(cleaned_response) in range(1, 6):
        return cleaned_response