import base64
import io
from typing import List

import torch
from pydantic import BaseModel


class TensorPayload(BaseModel):
    storage: str = "torch.save+base64"
    dtype: str
    shape: List[int]
    data_b64: str


class VocoderSynthesisRequest(BaseModel):
    version: str
    pred_spec: TensorPayload


class VocoderSynthesisResponse(BaseModel):
    version: str
    sample_rate: int
    audio: TensorPayload


class SuperResolveRequest(BaseModel):
    audio: TensorPayload
    sample_rate: int


class SuperResolveResponse(BaseModel):
    sample_rate: int
    audio: TensorPayload


def encode_tensor(tensor: torch.Tensor) -> TensorPayload:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu().contiguous(), buffer)
    return TensorPayload(
        dtype=str(tensor.dtype),
        shape=list(tensor.shape),
        data_b64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def decode_tensor(payload: TensorPayload | dict, device=None) -> torch.Tensor:
    if isinstance(payload, dict):
        payload = TensorPayload(**payload)
    buffer = io.BytesIO(base64.b64decode(payload.data_b64.encode("ascii")))
    tensor = torch.load(buffer, map_location="cpu", weights_only=False)
    if device is not None:
        tensor = tensor.to(device)
    return tensor
