# Whisper-FastAPI

Whisper-FastAPI is a very simple Python FastAPI interface for konele and OpenAI services. It is based on the `faster-whisper` project and provides an API for konele-like interface, where translations and transcriptions can be obtained by connecting over websockets or POST requests.

## Features

- **Translation and Transcription**: The application provides an API for konele service, where translations and transcriptions can be obtained by connecting over websockets or POST requests.
- **Language Support**: If the target language is English, then the application will translate any source language to English.
- **Websocket and POST Method Support**: The project supports a websocket (`/konele/ws`) and a POST method to `/konele/post`.
- **Audio Transcriptions**: The `/v1/audio/transcriptions` endpoint allows users to upload an audio file and receive transcription in response, with an optional `response_type` parameter. The `response_type` can be 'json', 'text', 'tsv', 'srt', and 'vtt'.
- **Simplified Chinese**: The traditional Chinese will be automatically convert to simplified Chinese for konele using `opencc` library.

## Usage

### Konele Voice Typing

For konele voice typing, you can use either the websocket endpoint or the POST method endpoint.

- **Websocket**: Connect to the websocket at `/konele/ws` and send audio data. The server will respond with the transcription or translation.
- **POST Method**: Send a POST request to `/konele/post` with the audio data in the body. The server will respond with the transcription or translation.

You can also use the demo I have created to quickly test the effect at <https://yongyuancv.cn/konele/ws> and <https://yongyuancv.cn/konele/post>

### OpenAI Whisper Service

To use the service that matches the structure of the OpenAI Whisper service, send a POST request to `/v1/audio/transcriptions` with an audio file. The server will respond with the transcription in the format specified by the `response_type` parameter.

You can also use the demo I have created to quickly test the effect at <https://yongyuancv.cn/v1/audio/transcriptions>

My demo is using the large-v2 model on RTX3060.

## Getting Started

To run the application, you need to have Python installed on your machine. You can then clone the repository and install the required dependencies.

```bash
git clone https://github.com/heimoshuiyu/whisper-fastapi.git
cd whisper-fastapi
pip install -r requirements.txt
```

You can then run the application using the following command: (model will be download from huggingface if not exists in cache dir)

```bash
python main.py --host 0.0.0.0 --port 5000 --model large-v2
```

This will start the application on `http://<your-ip-address>:5000`.

## Limitation

Defect: Due to the synchronous nature of inference, this API can actually only handle one request at a time.
