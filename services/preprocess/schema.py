import base64
import io
from typing import List, Optional

import torch
from pydantic import BaseModel, Field


class TensorPayload(BaseModel):
    storage: str = "torch.save+base64"
    dtype: str
    shape: List[int]
    data_b64: str


class TextSegmentRequest(BaseModel):
    text: str
    lang: str
    version: str = "v2"


class TextSegmentResponse(BaseModel):
    phones: List[int]
    norm_text: str
    bert_features: TensorPayload
    cache_hit: bool
    cache_key: str


class TextSegmentsRequest(BaseModel):
    text: str
    lang: str
    text_split_method: str


class TextSegmentsResponse(BaseModel):
    segments: List[str]
    cache_hit: bool
    cache_key: str


class TextPreprocessRequest(BaseModel):
    text: str
    lang: str
    text_split_method: str
    version: str = "v2"


class TextPreprocessResponse(BaseModel):
    segments: List[TextSegmentResponse]
    cache_hit: bool
    cache_key: str


class ReferencePreprocessRequest(BaseModel):
    ref_audio_path: str
    sampling_rate: int
    filter_length: int
    hop_length: int
    win_length: int
    is_half: bool = False
    need_v2_audio: bool = False


class ReferencePreprocessResponse(BaseModel):
    spec: TensorPayload
    raw_audio: TensorPayload
    raw_sr: int
    audio_16k: Optional[TensorPayload] = None
    hubert_feature: TensorPayload
    cache_hit: bool
    cache_key: str


def encode_tensor(tensor: torch.Tensor) -> TensorPayload:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), buffer)
    return TensorPayload(
        dtype=str(tensor.dtype),
        shape=list(tensor.shape),
        data_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def decode_tensor(payload: TensorPayload, device=None) -> torch.Tensor:
    if isinstance(payload, dict):
        payload = TensorPayload(**payload)
    buffer = io.BytesIO(base64.b64decode(payload.data_b64.encode("ascii")))
    tensor = torch.load(buffer, map_location="cpu", weights_only=False)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def build_text_segment_response(payload: dict) -> TextSegmentResponse:
    return TextSegmentResponse(
        phones=list(payload["phones"]),
        norm_text=payload["norm_text"],
        bert_features=encode_tensor(payload["bert_features"]),
        cache_hit=bool(payload.get("cache_hit", False)),
        cache_key=str(payload.get("cache_key", "")),
    )


def build_reference_response(payload: dict) -> ReferencePreprocessResponse:
    audio_16k = payload.get("audio_16k")
    return ReferencePreprocessResponse(
        spec=encode_tensor(payload["spec"]),
        raw_audio=encode_tensor(payload["raw_audio"]),
        raw_sr=int(payload["raw_sr"]),
        audio_16k=encode_tensor(audio_16k) if audio_16k is not None else None,
        hubert_feature=encode_tensor(payload["hubert_feature"]),
        cache_hit=bool(payload.get("cache_hit", False)),
        cache_key=str(payload.get("cache_key", "")),
    )
