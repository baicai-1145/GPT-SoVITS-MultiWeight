import requests
import torch

from services.preprocess.schema import (
    ReferencePreprocessRequest,
    TextPreprocessRequest,
    TextSegmentRequest,
    TextSegmentsRequest,
    decode_tensor,
)


class RemotePreprocessClient:
    backend_name = "remote_http"
    bert_tokenizer = None
    bert_model = None
    cnhuhbert_model = None
    text_preprocessor = None

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

    def pre_segment_text(self, text: str, lang: str, text_split_method: str) -> dict:
        request = TextSegmentsRequest(text=text, lang=lang, text_split_method=text_split_method)
        payload = self._request("POST", "/preprocess/text/segments", self._model_to_dict(request))
        return {
            "segments": payload["segments"],
            "cache_hit": payload["cache_hit"],
            "cache_key": payload["cache_key"],
        }

    def preprocess_segment_text(self, text: str, lang: str, version: str) -> dict:
        request = TextSegmentRequest(text=text, lang=lang, version=version)
        payload = self._request("POST", "/preprocess/segment", self._model_to_dict(request))
        return {
            "phones": payload["phones"],
            "norm_text": payload["norm_text"],
            "bert_features": decode_tensor(payload["bert_features"], self.device),
            "cache_hit": payload["cache_hit"],
            "cache_key": payload["cache_key"],
        }

    def preprocess_text(self, text: str, lang: str, text_split_method: str, version: str) -> dict:
        request = TextPreprocessRequest(text=text, lang=lang, text_split_method=text_split_method, version=version)
        payload = self._request("POST", "/preprocess/text", self._model_to_dict(request))
        return {
            "segments": [
                {
                    "phones": item["phones"],
                    "norm_text": item["norm_text"],
                    "bert_features": decode_tensor(item["bert_features"], self.device),
                    "cache_hit": item["cache_hit"],
                    "cache_key": item["cache_key"],
                }
                for item in payload["segments"]
            ],
            "cache_hit": payload["cache_hit"],
            "cache_key": payload["cache_key"],
        }

    def preprocess_reference_audio(
        self,
        ref_audio_path: str,
        sampling_rate: int,
        filter_length: int,
        hop_length: int,
        win_length: int,
        is_half: bool,
        need_v2_audio: bool,
    ) -> dict:
        request = ReferencePreprocessRequest(
            ref_audio_path=ref_audio_path,
            sampling_rate=sampling_rate,
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            is_half=is_half,
            need_v2_audio=need_v2_audio,
        )
        payload = self._request("POST", "/preprocess/reference", self._model_to_dict(request))
        return {
            "spec": decode_tensor(payload["spec"], self.device),
            "raw_audio": decode_tensor(payload["raw_audio"], self.device),
            "raw_sr": payload["raw_sr"],
            "audio_16k": decode_tensor(payload["audio_16k"], self.device) if payload["audio_16k"] is not None else None,
            "hubert_feature": decode_tensor(payload["hubert_feature"], self.device),
            "cache_hit": payload["cache_hit"],
            "cache_key": payload["cache_key"],
        }
