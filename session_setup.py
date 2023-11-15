import asyncio

from src.telegram.telegram import TelegramOCR
from src.utils import load_api_info

(TELEGRAM_APP_ID, TELEGRAM_APP_HASH,
 TELEGRAM_PHONE_NUMBER, TELEGRAM_CHANNEL) = load_api_info()

if __name__ == "__main__":
    telegram = TelegramOCR(
        telegram_app_id=TELEGRAM_APP_ID,
        telegram_app_hash=TELEGRAM_APP_HASH,
        telegram_phone_number=TELEGRAM_PHONE_NUMBER
    )

    # Set-up session for future use
    asyncio.run(telegram.set_up_initial_authorization())