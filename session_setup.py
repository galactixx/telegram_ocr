import asyncio

from src.vision.openai import OpenAIInterface
from src.telegram.telegram import TelegramOCR
from src.utils import load_api_info

# Load in telegram api info
telegram_info = load_api_info()

if __name__ == "__main__":
    telegram = TelegramOCR(
        telegram_app_id=telegram_info.app_id,
        telegram_app_hash=telegram_info.app_hash,
        telegram_phone_number=telegram_info.phone_number,
        openai_vision=OpenAIInterface()
    )

    # Set-up session for future use
    asyncio.run(telegram.set_up_initial_authorization())