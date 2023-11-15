from telethon.tl.types import MessageMediaPhoto
from telethon import TelegramClient, events

from src.utils import source_data_directory

class TelegramOCR:
    """
    Interact with Telegram to retrieve a variety of data based on a single crypto account.
    """
    def __init__(
        self,
        telegram_app_id: int,
        telegram_app_hash: str,
        telegram_phone_number: str):
        self._telegram_app_id = telegram_app_id
        self._telegram_app_hash = telegram_app_hash
        self._telegram_phone_number = telegram_phone_number

        # Telegram client instance
        self._client = TelegramClient(
            'crypto_ocr',
            self._telegram_app_id,
            self._telegram_app_hash)
        
    async def set_up_initial_authorization(self) -> None:
        """Connect to the Telegram server and set-up initial authorization."""
        await self._client.connect()

        # If you're not authorized, log in
        if not await self._client.is_user_authorized():
            await self._client.send_code_request(self._telegram_phone_number)
            await self._client.sign_in(self._telegram_phone_number, input('Enter the code: '))

        await self._client.disconnect()

    async def stream_images_in_messages(self, telegram_channel: str) -> None:
        """Stream messages from Telegram channel and process images."""
        path_source_data_image = source_data_directory(channel=telegram_channel)

        await self._client.connect()

        channel_entity = await self._client.get_entity(telegram_channel)

        @self._client.on(events.NewMessage(chats=channel_entity))
        async def handler(event):
            """Ensure that the instance of media is an image"""
            print(event.message)

            if event.message.media and isinstance(event.message.media, MessageMediaPhoto):
                photo = event.message.media.photo

                # Download the photo data
                photo_path = await self._client.download_media(
                    photo,
                    f'{path_source_data_image}/{event.message.id}.jpg')
                
        await self._client.run_until_disconnected()