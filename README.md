# telegram_ocr
This repository contains a specialized tool designed to scan and analyze images from cryptocurrency-focused Telegram channels. Utilizing advanced vision and Optical Character Recognition (OCR) methods, the tool efficiently parses images to extract information about specific crypto tokens.

This was done for a use case where I wanted to parse highly variable and unstructured text from specific images.

NOTE:
- The telegram connection has to be tested so that area of it is still a WIP (really easy but just have not done it).
- Some functions including the utils and image pre-processing is fine-tuned to my specific use-case. For example the ```parse_ocr_response``` function looks for responses that are of a specific length range.
- Thus, this repo could provide useful starter code for general use cases, but please be aware that much of the repo is tailored to my aim.
