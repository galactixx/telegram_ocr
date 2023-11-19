from telethon import events, TelegramClient
from telethon.tl.types import (
    MessageMediaDocument,
    MessageMediaPhoto)

from src.media.parser import MediaParser
from src.media.loader import MediaLoader
from src.vision.openai import OpenAIVision
from src.utils import (
    clean_channel,
    encode_image, 
    parse_ocr_response,
    source_data_directory)

class TelegramOCR:
    """
    Interact with Telegram to retrieve a variety of data based on a single crypto account.
    """
    def __init__(
        self,
        telegram_app_id: int,
        telegram_app_hash: str,
        telegram_phone_number: str,
        openai_vision: OpenAIVision):
        self._telegram_app_id = telegram_app_id
        self._telegram_app_hash = telegram_app_hash
        self._telegram_phone_number = telegram_phone_number
        self.openai_vision = openai_vision

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

    async def stream_images_in_messages(
        self,
        telegram_channel: str,
        telegram_channel_to_send: str,
        telegram_channel_keywords: list) -> None:
        """Stream messages from Telegram channel and process images."""

        path_source_data_image = source_data_directory(channel=telegram_channel)

        await self._client.connect()

        channel_entity = await self._client.get_entity(telegram_channel)

        @self._client.on(events.NewMessage(chats=channel_entity))
        async def handler(event):
            """Telegram media handler."""

            # Message info
            message = event.message
            message_media = message.media
            if message_media:
                media_path = None

                # If media is a photo or mp4 file
                if isinstance(message_media, MessageMediaPhoto):
                    media_path = await self._client.download_media(
                        message_media.photo,
                        f'{path_source_data_image}/{message.id}.jpg')
                elif isinstance(message_media, MessageMediaDocument):
                    if 'video/mp4' in message_media.document.mime_type:
                        media_path = await self._client.download_media(
                            message_media,
                            f'{path_source_data_image}/{message.id}.mp4')

                # Media parser instantiation
                if media_path is not None:
                    media_parser = MediaParser(
                        media_loader=MediaLoader(media_path=media_path))
                    if media_parser.image is not None:
                        media_parser.remove_small_contours()

                        # After image processing, encode image to base64 representation
                        base64_image = encode_image(image=media_parser.image)

                        response = self.openai_vision.get_vision_completion(
                            prompt='What are the largest characters in this image? Only output the text in the image.',
                            base64_image=base64_image)
                        
                        # Check if any keywords are in text of message
                        do_keywords = any(i for i in telegram_channel_keywords if i in message.text)
                        response_parsed = parse_ocr_response(response=response)

                        # Send parsed response to Telegram channel
                        if do_keywords:
                            if response_parsed is not None:
                                message = f'Text parsed in message on {message.date}: {response_parsed}'
                                await self._client.send_message(telegram_channel_to_send, message)
                            else:
                                message_link = f"https://t.me/{clean_channel(channel=telegram_channel)}/{message.id}"
                                message = f"ISSUE WITH PARSING TEXT IN MESSAGE on {message.date}.\nPlease see link to message: {message_link}"
                                await self._client.send_message(telegram_channel_to_send, message)

        await self._client.run_until_disconnected()