import asyncio

from src.vision.openai import OpenAIInterface
from src.telegram.telegram import TelegramOCR
from src.utils import (
    load_api_info,
    source_data_directories)

(TELEGRAM_APP_ID, TELEGRAM_APP_HASH,
 TELEGRAM_PHONE_NUMBER, TELEGRAM_CHANNEL) = load_api_info()

if __name__ == "__main__":

    # Generate directories
    source_data_directories(channel=TELEGRAM_CHANNEL)

    # Telegram instantiation
    telegram = TelegramOCR(
        telegram_app_id=TELEGRAM_APP_ID,
        telegram_app_hash=TELEGRAM_APP_HASH,
        telegram_phone_number=TELEGRAM_PHONE_NUMBER,
        openai_vision=OpenAIInterface()
    )

    # Download any images sent by account in question
    asyncio.run(telegram.stream_images_in_messages(telegram_channel=TELEGRAM_CHANNEL))