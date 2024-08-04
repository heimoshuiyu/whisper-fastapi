import tqdm
import json
from fastapi.responses import StreamingResponse
import wave
import pydub
import io
import hashlib
import argparse
import uvicorn
from typing import Annotated, Any, BinaryIO, Literal, Generator, Tuple, Iterable
from fastapi import (
    File,
    HTTPException,
    Query,
    UploadFile,
    Form,
    FastAPI,
    Request,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from src.whisper_ctranslate2.whisper_ctranslate2 import Transcribe
from src.whisper_ctranslate2.writers import format_timestamp
from faster_whisper.transcribe import Segment, TranscriptionInfo
import opencc
from prometheus_fastapi_instrumentator import Instrumentator

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0", type=str)
parser.add_argument("--port", default=5000, type=int)
parser.add_argument("--model", default="large-v3", type=str)
parser.add_argument("--device", default="auto", type=str)
parser.add_argument("--cache_dir", default=None, type=str)
parser.add_argument("--local_files_only", default=False, type=bool)
parser.add_argument("--threads", default=4, type=int)
args = parser.parse_args()
app = FastAPI()
# Instrument your app with default metrics and expose the metrics
Instrumentator().instrument(app).expose(app, endpoint="/konele/metrics")
ccc = opencc.OpenCC("t2s.json")

print("Loading model...")
transcriber = Transcribe(
    model_path=args.model,
    device=args.device,
    device_index=0,
    compute_type="default",
    threads=args.threads,
    cache_directory=args.cache_dir,
    local_files_only=args.local_files_only,
)
print("Model loaded!")


# allow all cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def stream_writer(generator: Generator[dict[str, Any], Any, None]):
    for segment in generator:
        yield "data: " + json.dumps(segment, ensure_ascii=False) + "\n\n"
    yield "data: [DONE]\n\n"


def text_writer(generator: Generator[dict[str, Any], Any, None]):
    for segment in generator:
        yield segment["text"].strip() + "\n"


def tsv_writer(generator: Generator[dict[str, Any], Any, None]):
    yield "start\tend\ttext\n"
    for i, segment in enumerate(generator):
        start_time = str(round(1000 * segment["start"]))
        end_time = str(round(1000 * segment["end"]))
        text = segment["text"]
        yield f"{start_time}\t{end_time}\t{text}\n"


def srt_writer(generator: Generator[dict[str, Any], Any, None]):
    for i, segment in enumerate(generator):
        start_time = format_timestamp(
            segment["start"], decimal_marker=",", always_include_hours=True
        )
        end_time = format_timestamp(
            segment["end"], decimal_marker=",", always_include_hours=True
        )
        text = segment["text"]
        yield f"{i}\n{start_time} --> {end_time}\n{text}\n\n"


def vtt_writer(generator: Generator[dict[str, Any], Any, None]):
    yield "WEBVTT\n\n"
    for i, segment in enumerate(generator):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"]
        yield f"{start_time} --> {end_time}\n{text}\n\n"


def build_json_result(
    generator: Iterable[Segment],
    info:  dict,
) -> dict[str, Any]:
    segments = [i for i in generator]
    return info | {
        "text": "\n".join(i["text"] for i in segments),
        "segments": segments,
    }


def stream_builder(
    audio: BinaryIO,
    task: str,
    vad_filter: bool,
    language: str | None,
    initial_prompt: str = "",
    repetition_penalty: float = 1.0,
) -> Tuple[Iterable[dict], dict]:
    segments, info = transcriber.model.transcribe(
        audio=audio,
        language=language,
        task=task,
        initial_prompt=initial_prompt,
        word_timestamps=True,
        repetition_penalty=repetition_penalty,
    )
    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )
    def wrap():
        last_pos = 0
        with tqdm.tqdm(total=info.duration, unit="seconds", disable=True) as pbar:
            for segment in segments:
                start, end, text = segment.start, segment.end, segment.text
                pbar.update(end - last_pos)
                last_pos = end
                data = segment._asdict()
                if data.get('words') is not None:
                    data["words"] = [i._asdict() for i in data["words"]]
                data["text"] = ccc.convert(data["text"])
                yield data

    info_dict = info._asdict()
    if info_dict['transcription_options'] is not None:
        info_dict['transcription_options'] = info_dict['transcription_options']._asdict()
    if info_dict['vad_options'] is not None:
        info_dict['vad_options'] = info_dict['vad_options']._asdict()
    
    return wrap(), info_dict


@app.websocket("/k6nele/status")
@app.websocket("/konele/status")
async def konele_status(
    websocket: WebSocket,
):
    await websocket.accept()
    await websocket.send_json(dict(num_workers_available=1))
    await websocket.close()


@app.websocket("/k6nele/ws")
@app.websocket("/konele/ws")
async def konele_ws(
    websocket: WebSocket,
    task: Literal["transcribe", "translate"] = "transcribe",
    lang: str = "und",
    initial_prompt: str = "",
    vad_filter: bool = False,
    content_type: Annotated[str, Query(alias="content-type")] = "audio/x-raw",
):
    await websocket.accept()

    # convert lang code format (eg. en-US to en)
    lang = lang.split("-")[0]

    print("WebSocket client connected, lang is", lang)
    print("content-type is", content_type)
    data = b""
    while True:
        try:
            data += await websocket.receive_bytes()
            print("Received data:", len(data), data[-10:])
            if data[-3:] == b"EOS":
                print("End of speech")
                break
        except:
            break

    md5 = hashlib.md5(data).hexdigest()

    # create fake file for wave.open
    file_obj = io.BytesIO()

    if content_type.startswith("audio/x-flac"):
        pydub.AudioSegment.from_file(io.BytesIO(data), format="flac").export(
            file_obj, format="wav"
        )
    else:
        buffer = wave.open(file_obj, "wb")
        buffer.setnchannels(1)
        buffer.setsampwidth(2)
        buffer.setframerate(16000)
        buffer.writeframes(data)

    file_obj.seek(0)

    generator = stream_builder(
        audio=file_obj,
        task=task,
        vad_filter=vad_filter,
        language=None if lang == "und" else lang,
        initial_prompt=initial_prompt,
    )
    result = build_json_result(generator)

    text = result.get("text", "")
    print("result", text)

    await websocket.send_json(
        {
            "status": 0,
            "segment": 0,
            "result": {"hypotheses": [{"transcript": text}], "final": True},
            "id": md5,
        }
    )
    await websocket.close()


@app.post("/k6nele/post")
@app.post("/konele/post")
async def translateapi(
    request: Request,
    task: Literal["transcribe", "translate"] = "transcribe",
    lang: str = "und",
    initial_prompt: str = "",
    vad_filter: bool = False,
):
    content_type = request.headers.get("Content-Type", "")
    print("downloading request file", content_type)

    # convert lang code format (eg. en-US to en)
    lang = lang.split("-")[0]

    splited = [i.strip() for i in content_type.split(",") if "=" in i]
    info = {k: v for k, v in (i.split("=") for i in splited)}
    print(info)

    channels = int(info.get("channels", "1"))
    rate = int(info.get("rate", "16000"))

    body = await request.body()
    md5 = hashlib.md5(body).hexdigest()

    # create fake file for wave.open
    file_obj = io.BytesIO()

    if content_type.startswith("audio/x-flac"):
        pydub.AudioSegment.from_file(io.BytesIO(body), format="flac").export(
            file_obj, format="wav"
        )
    else:
        buffer = wave.open(file_obj, "wb")
        buffer.setnchannels(channels)
        buffer.setsampwidth(2)
        buffer.setframerate(rate)
        buffer.writeframes(body)

    file_obj.seek(0)

    generator = stream_builder(
        audio=file_obj,
        task=task,
        vad_filter=vad_filter,
        language=None if lang == "und" else lang,
        initial_prompt=initial_prompt,
    )
    result = build_json_result(generator)

    text = result.get("text", "")
    print("result", text)

    return {
        "status": 0,
        "hypotheses": [{"utterance": text}],
        "id": md5,
    }


@app.post("/v1/audio/transcriptions")
async def transcription(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    response_format: str = Form("json"),
    task: str = Form("transcribe"),
    language: str = Form("und"),
    vad_filter: bool = Form(False),
    repetition_penalty: float = Form(1.0),
):
    """Transcription endpoint

    User upload audio file in multipart/form-data format and receive transcription in response
    """

    # timestamp as filename, keep original extension
    generator, info = stream_builder(
        audio=io.BytesIO(file.file.read()),
        task=task,
        vad_filter=vad_filter,
        language=None if language == "und" else language,
        repetition_penalty=repetition_penalty,
    )

    # special function for streaming response (OpenAI API does not have this)
    if response_format == "stream":
        return StreamingResponse(
            stream_writer(generator),
            media_type="text/event-stream",
        )
    elif response_format == "json":
        return build_json_result(generator)
    elif response_format == "text":
        return StreamingResponse(text_writer(generator), media_type="text/plain")
    elif response_format == "tsv":
        return StreamingResponse(tsv_writer(generator), media_type="text/plain")
    elif response_format == "srt":
        return StreamingResponse(srt_writer(generator), media_type="text/plain")
    elif response_format == "vtt":
        return StreamingResponse(vtt_writer(generator), media_type="text/plain")

    raise HTTPException(400, "Invailed response_format")


uvicorn.run(app, host=args.host, port=args.port)
