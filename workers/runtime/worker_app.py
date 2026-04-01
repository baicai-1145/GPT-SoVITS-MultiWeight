import asyncio
import subprocess
import threading
import wave
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Generator, Optional, Union

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import (
    get_method_names as get_cut_method_names,
)
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()
cut_method_names = get_cut_method_names()


@dataclass
class WorkerRouteOptions:
    enable_control: bool = False
    enable_reference_audio_route: bool = False
    enable_hot_reload_routes: bool = False
    serial_inference: bool = True
    mode_label: str = "worker"
    metadata: dict[str, Any] = field(default_factory=dict)
    control_handler: Optional[Callable[[str], None]] = None

    def registered_routes(self) -> list[str]:
        routes = ["/tts", "/health", "/meta"]
        if self.enable_control:
            routes.append("/control")
        if self.enable_reference_audio_route:
            routes.append("/set_refer_audio")
        if self.enable_hot_reload_routes:
            routes.extend(["/set_gpt_weights", "/set_sovits_weights"])
        return routes


class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: Union[bool, int] = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    overlap_length: int = 2
    min_chunk_length: int = 16


def _normalize_lang(value: Optional[str]) -> Optional[str]:
    return value.lower() if isinstance(value, str) else value


def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    def handle_pack_ogg():
        with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
            audio_file.write(data)

    stack_size = 4096 * 4096
    try:
        threading.stack_size(stack_size)
        pack_ogg_thread = threading.Thread(target=handle_pack_ogg)
        pack_ogg_thread.start()
        pack_ogg_thread.join()
    except RuntimeError as exc:
        print(f"RuntimeError: {exc}")
        print("Changing the thread stack size is unsupported.")
    except ValueError as exc:
        print(f"ValueError: {exc}")
        print("The specified stack size is invalid.")

    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-vn",
            "-f",
            "adts",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def check_params(req: dict, tts_config: TTS_Config):
    text = req.get("text", "")
    text_lang = req.get("text_lang", "")
    ref_audio_path = req.get("ref_audio_path", "")
    media_type = req.get("media_type", "wav")
    prompt_lang = req.get("prompt_lang", "")
    text_split_method = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    if text_lang not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    if prompt_lang not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_split_method:{text_split_method} is not supported"},
        )
    return None


def create_tts_app(
    tts_pipeline: TTS,
    tts_config: TTS_Config,
    *,
    route_options: Optional[WorkerRouteOptions] = None,
) -> FastAPI:
    route_options = route_options or WorkerRouteOptions()
    if route_options.enable_control and route_options.control_handler is None:
        raise ValueError("control_handler is required when enable_control is True")

    app = FastAPI(title=f"GPT-SoVITS {route_options.mode_label}")
    inference_lock = asyncio.Lock()

    def build_meta_payload() -> dict[str, Any]:
        payload = {
            "mode": route_options.mode_label,
            "serial_inference": route_options.serial_inference,
            "routes": route_options.registered_routes(),
            "tts": tts_pipeline.get_runtime_meta(),
        }
        payload.update(route_options.metadata)
        return payload

    async def tts_handle(req: dict):
        req = dict(req)
        req["text_lang"] = _normalize_lang(req.get("text_lang"))
        req["prompt_lang"] = _normalize_lang(req.get("prompt_lang"))

        streaming_mode = req.get("streaming_mode", False)
        return_fragment = req.get("return_fragment", False)
        media_type = req.get("media_type", "wav")

        check_res = check_params(req, tts_config)
        if check_res is not None:
            return check_res

        if streaming_mode == 0:
            streaming_mode = False
            return_fragment = False
            fixed_length_chunk = False
        elif streaming_mode == 1:
            streaming_mode = False
            return_fragment = True
            fixed_length_chunk = False
        elif streaming_mode == 2:
            streaming_mode = True
            return_fragment = False
            fixed_length_chunk = False
        elif streaming_mode == 3:
            streaming_mode = True
            return_fragment = False
            fixed_length_chunk = True
        else:
            return JSONResponse(
                status_code=400,
                content={"message": "the value of streaming_mode must be 0, 1, 2, 3(int) or true/false(bool)"},
            )

        req["streaming_mode"] = streaming_mode
        req["return_fragment"] = return_fragment
        req["fixed_length_chunk"] = fixed_length_chunk

        print(f"{streaming_mode} {return_fragment} {fixed_length_chunk}")
        streaming_mode = streaming_mode or return_fragment

        async def run_streaming_generator():
            if route_options.serial_inference:
                await inference_lock.acquire()
            try:
                tts_generator = tts_pipeline.run(req)
                first_chunk = True
                stream_media_type = media_type
                for sr, chunk in tts_generator:
                    if first_chunk and stream_media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        stream_media_type = "raw"
                        first_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, stream_media_type).getvalue()
            finally:
                if route_options.serial_inference and inference_lock.locked():
                    inference_lock.release()

        if streaming_mode:
            return StreamingResponse(
                run_streaming_generator(),
                media_type=f"audio/{media_type}",
            )

        try:
            if route_options.serial_inference:
                async with inference_lock:
                    sr, audio_data = next(tts_pipeline.run(req))
            else:
                sr, audio_data = next(tts_pipeline.run(req))
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
        except Exception as exc:
            return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(exc)})

    @app.get("/health")
    async def health():
        payload = build_meta_payload()
        payload.update(tts_pipeline.get_health_status())
        ready = payload.get("ready", False)
        return JSONResponse(status_code=200 if ready else 503, content=payload)

    @app.get("/meta")
    async def meta():
        return JSONResponse(status_code=200, content=build_meta_payload())

    @app.get("/tts")
    async def tts_get_endpoint(
        text: str = None,
        text_lang: str = None,
        ref_audio_path: str = None,
        aux_ref_audio_paths: list = None,
        prompt_lang: str = None,
        prompt_text: str = "",
        top_k: int = 15,
        top_p: float = 1,
        temperature: float = 1,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        speed_factor: float = 1.0,
        fragment_interval: float = 0.3,
        seed: int = -1,
        media_type: str = "wav",
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        sample_steps: int = 32,
        super_sampling: bool = False,
        streaming_mode: Union[bool, int] = False,
        overlap_length: int = 2,
        min_chunk_length: int = 16,
    ):
        req = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": int(batch_size),
            "batch_threshold": float(batch_threshold),
            "speed_factor": float(speed_factor),
            "split_bucket": split_bucket,
            "fragment_interval": fragment_interval,
            "seed": seed,
            "media_type": media_type,
            "streaming_mode": streaming_mode,
            "parallel_infer": parallel_infer,
            "repetition_penalty": float(repetition_penalty),
            "sample_steps": int(sample_steps),
            "super_sampling": super_sampling,
            "overlap_length": int(overlap_length),
            "min_chunk_length": int(min_chunk_length),
        }
        return await tts_handle(req)

    @app.post("/tts")
    async def tts_post_endpoint(request: TTS_Request):
        req = request.dict()
        return await tts_handle(req)

    if route_options.enable_control:

        @app.get("/control")
        async def control(command: str = None):
            if command is None:
                return JSONResponse(status_code=400, content={"message": "command is required"})
            route_options.control_handler(command)

    if route_options.enable_reference_audio_route:

        @app.get("/set_refer_audio")
        async def set_refer_audio(refer_audio_path: str = None):
            try:
                tts_pipeline.set_ref_audio(refer_audio_path)
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"message": "set refer audio failed", "Exception": str(exc)},
                )
            return JSONResponse(status_code=200, content={"message": "success"})

    if route_options.enable_hot_reload_routes:

        @app.get("/set_gpt_weights")
        async def set_gpt_weights(weights_path: str = None):
            try:
                if weights_path in ["", None]:
                    return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
                tts_pipeline.init_t2s_weights(weights_path)
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"message": "change gpt weight failed", "Exception": str(exc)},
                )
            return JSONResponse(status_code=200, content={"message": "success"})

        @app.get("/set_sovits_weights")
        async def set_sovits_weights(weights_path: str = None):
            try:
                if weights_path in ["", None]:
                    return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
                tts_pipeline.init_vits_weights(weights_path)
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"message": "change sovits weight failed", "Exception": str(exc)},
                )
            return JSONResponse(status_code=200, content={"message": "success"})

    return app
