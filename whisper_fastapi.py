import wave
import pydub
import io
import hashlib
import argparse
import uvicorn
from typing import Annotated, Any, Literal
from fastapi import File, Query, UploadFile, Form, FastAPI, Request, WebSocket, Response
from fastapi.middleware.cors import CORSMiddleware
from src.whisper_ctranslate2.whisper_ctranslate2 import Transcribe, TranscriptionOptions
from src.whisper_ctranslate2.writers import format_timestamp
import opencc
from prometheus_fastapi_instrumentator import Instrumentator

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0", type=str)
parser.add_argument("--port", default=5000, type=int)
parser.add_argument("--model", default="large-v2", type=str)
parser.add_argument("--cache_dir", default=None, type=str)
args = parser.parse_args()
app = FastAPI()
# Instrument your app with default metrics and expose the metrics
Instrumentator().instrument(app).expose(app, endpoint="/konele/metrics")
ccc = opencc.OpenCC("t2s.json")

print("Loading model...")
transcriber = Transcribe(
    model_path=args.model,
    device="auto",
    device_index=0,
    compute_type="default",
    threads=1,
    cache_directory=args.cache_dir,
    local_files_only=False,
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


def generate_tsv(result: dict[str, list[Any]]):
    tsv = "start\tend\ttext\n"
    for i, segment in enumerate(result["segments"]):
        start_time = str(round(1000 * segment["start"]))
        end_time = str(round(1000 * segment["end"]))
        text = segment["text"]
        tsv += f"{start_time}\t{end_time}\t{text}\n"
    return tsv


def generate_srt(result: dict[str, list[Any]]):
    srt = ""
    for i, segment in enumerate(result["segments"], start=1):
        start_time = format_timestamp(
            segment["start"], decimal_marker=",", always_include_hours=True
        )
        end_time = format_timestamp(
            segment["end"], decimal_marker=",", always_include_hours=True
        )
        text = segment["text"]
        srt += f"{i}\n{start_time} --> {end_time}\n{text}\n\n"
    return srt


def generate_vtt(result: dict[str, list[Any]]):
    vtt = "WEBVTT\n\n"
    for segment in result["segments"]:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"]
        vtt += f"{start_time} --> {end_time}\n{text}\n\n"
    return vtt


def get_options(*, initial_prompt=""):
    options = TranscriptionOptions(
        beam_size=5,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=True,
        temperature=[0.0, 1.0 + 1e-6, 0.2],
        suppress_tokens=[-1],
        word_timestamps=True,
        print_colors=False,
        prepend_punctuations="\"'“¿([{-",
        append_punctuations="\"'.。,，!！?？:：”)]}、",
        vad_filter=False,
        vad_threshold=None,
        vad_min_speech_duration_ms=None,
        vad_max_speech_duration_s=None,
        vad_min_silence_duration_ms=None,
        initial_prompt=initial_prompt,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        prompt_reset_on_temperature=False,
        suppress_blank=False,
    )
    return options


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

    options = get_options(initial_prompt=initial_prompt)

    result = transcriber.inference(
        audio=file_obj,
        task=task,
        language=lang if lang != "und" else None,  # type: ignore
        verbose=False,
        live=False,
        options=options,
    )
    text = result.get("text", "")
    text = ccc.convert(text)
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

    options = get_options(initial_prompt=initial_prompt)

    result = transcriber.inference(
        audio=file_obj,
        task=task,
        language=lang if lang != "und" else None,  # type: ignore
        verbose=False,
        live=False,
        options=options,
    )
    text = result.get("text", "")
    text = ccc.convert(text)
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
):
    """Transcription endpoint

    User upload audio file in multipart/form-data format and receive transcription in response
    """

    # timestamp as filename, keep original extension
    options = get_options(initial_prompt=prompt)

    result: Any = transcriber.inference(
        audio=io.BytesIO(file.file.read()),
        task="transcribe",
        language=None,  # type: ignore
        verbose=False,
        live=False,
        options=options,
    )

    if response_format == "json":
        return result
    elif response_format == "text":
        return Response(
            content="\n".join(s["text"] for s in result["segments"]),
            media_type="plain/text",
        )
    elif response_format == "tsv":
        return Response(content=generate_tsv(result), media_type="plain_text")
    elif response_format == "srt":
        return Response(content=generate_srt(result), media_type="plain_text")
    elif response_format == "vtt":
        return generate_vtt(result)

    return {"error": "Invalid response_format"}


uvicorn.run(app, host=args.host, port=args.port)
