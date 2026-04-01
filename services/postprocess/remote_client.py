import requests
import torch

from services.postprocess.schema import (
    SuperResolveRequest,
    VocoderSynthesisRequest,
    decode_tensor,
    encode_tensor,
)

VOCODER_CONFIGS = {
    "v3": {
        "sr": 24000,
        "T_ref": 468,
        "T_chunk": 934,
        "upsample_rate": 256,
        "overlapped_len": 12,
    },
    "v4": {
        "sr": 48000,
        "T_ref": 500,
        "T_chunk": 1000,
        "upsample_rate": 480,
        "overlapped_len": 12,
    },
}


class RemotePostprocessClient:
    backend_name = "remote_http"

    def __init__(self, base_url: str, timeout_seconds: float, device, is_half: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.device = torch.device(device)
        self.is_half = bool(is_half)

    @staticmethod
    def _model_to_dict(model):
        return model.model_dump() if hasattr(model, "model_dump") else model.dict()

    def _request(self, method: str, path: str, payload: dict | None = None):
        response = requests.request(
            method=method,
            url=f"{self.base_url}{path}",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def get_vocoder_config(self, version: str) -> dict:
        if version not in VOCODER_CONFIGS:
            raise ValueError(f"unsupported vocoder version: {version}")
        return dict(VOCODER_CONFIGS[version])

    def set_device(self, device):
        self.device = torch.device(device)

    def enable_half_precision(self, enable: bool = True):
        self.is_half = bool(enable)

    def get_health_status(self) -> dict:
        payload = self._request("GET", "/health")
        payload["backend"] = self.backend_name
        payload["base_url"] = self.base_url
        return payload

    def get_runtime_meta(self) -> dict:
        payload = self._request("GET", "/meta")
        payload["backend"] = self.backend_name
        payload["base_url"] = self.base_url
        payload["target_device"] = str(self.device)
        payload["target_is_half"] = self.is_half
        return payload

    def synthesize_vocoder(self, pred_spec: torch.Tensor, version: str) -> tuple[torch.Tensor, int]:
        request = VocoderSynthesisRequest(version=version, pred_spec=encode_tensor(pred_spec))
        payload = self._request("POST", "/postprocess/vocoder", self._model_to_dict(request))
        return decode_tensor(payload["audio"], self.device), int(payload["sample_rate"])

    def super_resolve(self, audio: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        request = SuperResolveRequest(audio=encode_tensor(audio), sample_rate=sample_rate)
        try:
            payload = self._request("POST", "/postprocess/sr", self._model_to_dict(request))
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 503:
                raise FileNotFoundError(response.json().get("detail", "super-resolution model is not available")) from exc
            raise
        return decode_tensor(payload["audio"], self.device), int(payload["sample_rate"])
