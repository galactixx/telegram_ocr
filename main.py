import asyncio

from src.vision.openai import OpenAIInterface
from src.telegram.telegram import TelegramOCR
from src.utils import (
    load_api_info,
    source_data_directories)

# Load in telegram api info
telegram_info = load_api_info()

if __name__ == "__main__":

    # Generate directories
    source_data_directories(channel=telegram_info.channel)

    # Telegram instantiation
    telegram = TelegramOCR(
        telegram_app_id=telegram_info.app_id,
        telegram_app_hash=telegram_info.app_hash,
        telegram_phone_number=telegram_info.phone_number,
        openai_vision=OpenAIInterface()
    )

    # Download any images sent by account in question
    asyncio.run(telegram.stream_images_in_messages(
        telegram_channel=telegram_info.channel,
        telegram_channel_to_send=telegram_info.channel_to_send))