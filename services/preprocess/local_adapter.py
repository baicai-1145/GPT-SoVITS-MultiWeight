import copy
import os
import threading
from typing import Dict, List

import librosa
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForMaskedLM, AutoTokenizer

from feature_extractor.cnhubert import CNHubert
from module.mel_processing import spectrogram_torch
from TTS_infer_pack.TextPreprocessor import TextPreprocessor


class LocalPreprocessAdapter:
    backend_name = "local"

    def __init__(self, bert_base_path: str, cnhuhbert_base_path: str, device, is_half: bool = False):
        self.bert_base_path = bert_base_path
        self.cnhuhbert_base_path = cnhuhbert_base_path
        self.device = torch.device(device)
        self.is_half = bool(is_half)
        self.bert_tokenizer = None
        self.bert_model = None
        self.cnhuhbert_model = None
        self.text_preprocessor = None
        self._lock = threading.RLock()
        self._segment_split_cache: Dict[str, dict] = {}
        self._segment_feature_cache: Dict[str, dict] = {}
        self._text_cache: Dict[str, dict] = {}
        self._reference_cache: Dict[str, dict] = {}
        self._resample_transform_dict: Dict[str, torchaudio.transforms.Resample] = {}
        self.init_bert_weights(self.bert_base_path)
        self.init_cnhuhbert_weights(self.cnhuhbert_base_path)

    def _cache_key(self, prefix: str, *parts) -> str:
        return prefix + "::" + "||".join(str(part) for part in parts)

    def _resample(self, audio_tensor: torch.Tensor, sr0: int, sr1: int):
        key = f"{sr0}-{sr1}-{self.device}"
        if key not in self._resample_transform_dict:
            self._resample_transform_dict[key] = torchaudio.transforms.Resample(sr0, sr1).to(self.device)
        return self._resample_transform_dict[key](audio_tensor)

    def _clear_caches(self):
        self._segment_split_cache.clear()
        self._segment_feature_cache.clear()
        self._text_cache.clear()
        self._reference_cache.clear()

    def _refresh_text_preprocessor(self):
        self.text_preprocessor = TextPreprocessor(self.bert_model, self.bert_tokenizer, self.device)

    def init_bert_weights(self, base_path: str):
        with self._lock:
            print(f"Loading BERT weights from {base_path}")
            self.bert_base_path = base_path
            self.bert_tokenizer = AutoTokenizer.from_pretrained(base_path)
            self.bert_model = AutoModelForMaskedLM.from_pretrained(base_path)
            self.bert_model = self.bert_model.eval().to(self.device)
            if self.is_half and str(self.device) != "cpu":
                self.bert_model = self.bert_model.half()
            self._refresh_text_preprocessor()
            self._clear_caches()

    def init_cnhuhbert_weights(self, base_path: str):
        with self._lock:
            print(f"Loading CNHuBERT weights from {base_path}")
            self.cnhuhbert_base_path = base_path
            self.cnhuhbert_model = CNHubert(base_path)
            self.cnhuhbert_model = self.cnhuhbert_model.eval().to(self.device)
            if self.is_half and str(self.device) != "cpu":
                self.cnhuhbert_model = self.cnhuhbert_model.half()
            self._clear_caches()

    def enable_half_precision(self, enable: bool = True):
        with self._lock:
            self.is_half = bool(enable)
            if str(self.device) == "cpu" and self.is_half:
                self.is_half = False
            if self.bert_model is not None:
                self.bert_model = self.bert_model.half() if self.is_half else self.bert_model.float()
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.half() if self.is_half else self.cnhuhbert_model.float()
            self._clear_caches()

    def set_device(self, device):
        with self._lock:
            self.device = torch.device(device)
            self._resample_transform_dict.clear()
            if self.bert_model is not None:
                self.bert_model = self.bert_model.to(self.device)
            if self.cnhuhbert_model is not None:
                self.cnhuhbert_model = self.cnhuhbert_model.to(self.device)
            if self.text_preprocessor is not None:
                self.text_preprocessor.device = self.device
            self._clear_caches()

    def get_health_status(self) -> dict:
        return {
            "ready": all(
                item is not None
                for item in [
                    self.bert_tokenizer,
                    self.bert_model,
                    self.cnhuhbert_model,
                    self.text_preprocessor,
                ]
            ),
            "backend": self.backend_name,
            "device": str(self.device),
            "is_half": self.is_half,
            "cache_sizes": {
                "text_segments": len(self._segment_split_cache),
                "segment_features": len(self._segment_feature_cache),
                "texts": len(self._text_cache),
                "references": len(self._reference_cache),
            },
        }

    def get_runtime_meta(self) -> dict:
        return {
            "backend": self.backend_name,
            "device": str(self.device),
            "is_half": self.is_half,
            "bert_base_path": self.bert_base_path,
            "cnhuhbert_base_path": self.cnhuhbert_base_path,
            "cache_sizes": self.get_health_status()["cache_sizes"],
        }

    def pre_segment_text(self, text: str, lang: str, text_split_method: str) -> dict:
        cache_key = self._cache_key("split", text, lang, text_split_method)
        with self._lock:
            cached = self._segment_split_cache.get(cache_key)
            if cached is not None:
                return {"segments": list(cached["segments"]), "cache_hit": True, "cache_key": cache_key}
            segments = self.text_preprocessor.pre_seg_text(text, lang, text_split_method)
            self._segment_split_cache[cache_key] = {"segments": list(segments)}
            return {"segments": list(segments), "cache_hit": False, "cache_key": cache_key}

    def preprocess_segment_text(self, text: str, lang: str, version: str) -> dict:
        cache_key = self._cache_key("segment", text, lang, version)
        with self._lock:
            cached = self._segment_feature_cache.get(cache_key)
            if cached is not None:
                return {
                    "phones": list(cached["phones"]),
                    "bert_features": cached["bert_features"],
                    "norm_text": cached["norm_text"],
                    "cache_hit": True,
                    "cache_key": cache_key,
                }
            phones, bert_features, norm_text = self.text_preprocessor.segment_and_extract_feature_for_text(
                text, lang, version
            )
            result = {
                "phones": list(phones) if phones is not None else None,
                "bert_features": bert_features,
                "norm_text": norm_text,
            }
            self._segment_feature_cache[cache_key] = result
            return {
                "phones": list(result["phones"]) if result["phones"] is not None else None,
                "bert_features": result["bert_features"],
                "norm_text": result["norm_text"],
                "cache_hit": False,
                "cache_key": cache_key,
            }

    def preprocess_text(self, text: str, lang: str, text_split_method: str, version: str) -> dict:
        normalized_text = self.text_preprocessor.replace_consecutive_punctuation(text)
        cache_key = self._cache_key("text", normalized_text, lang, text_split_method, version)
        with self._lock:
            cached = self._text_cache.get(cache_key)
            if cached is not None:
                return {
                    "segments": [
                        {
                            "phones": list(item["phones"]),
                            "bert_features": item["bert_features"],
                            "norm_text": item["norm_text"],
                        }
                        for item in cached["segments"]
                    ],
                    "cache_hit": True,
                    "cache_key": cache_key,
                }

        split_result = self.pre_segment_text(normalized_text, lang, text_split_method)
        segments = []
        for item_text in split_result["segments"]:
            item = self.preprocess_segment_text(item_text, lang, version)
            if item["phones"] is None or item["norm_text"] == "":
                continue
            segments.append(
                {
                    "phones": list(item["phones"]),
                    "bert_features": item["bert_features"],
                    "norm_text": item["norm_text"],
                }
            )
        with self._lock:
            self._text_cache[cache_key] = {"segments": copy.copy(segments)}
        return {"segments": segments, "cache_hit": False, "cache_key": cache_key}

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
        abs_path = os.path.abspath(ref_audio_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(abs_path)
        stat = os.stat(abs_path)
        cache_key = self._cache_key(
            "reference",
            abs_path,
            stat.st_mtime_ns,
            sampling_rate,
            filter_length,
            hop_length,
            win_length,
            is_half,
            need_v2_audio,
        )
        with self._lock:
            cached = self._reference_cache.get(cache_key)
            if cached is not None:
                return {
                    "spec": cached["spec"],
                    "raw_audio": cached["raw_audio"],
                    "raw_sr": cached["raw_sr"],
                    "audio_16k": cached["audio_16k"],
                    "hubert_feature": cached["hubert_feature"],
                    "cache_hit": True,
                    "cache_key": cache_key,
                }

        raw_audio, raw_sr = torchaudio.load(abs_path)
        raw_audio = raw_audio.to(self.device).float()
        audio = raw_audio
        if audio.shape[0] == 2:
            audio = audio.mean(0).unsqueeze(0)
        if raw_sr != sampling_rate:
            audio = self._resample(audio, raw_sr, sampling_rate)
        maxx = audio.abs().max()
        if maxx > 1:
            audio = audio / min(2, maxx)

        spec = spectrogram_torch(
            audio,
            filter_length,
            sampling_rate,
            hop_length,
            win_length,
            center=False,
        )
        if is_half:
            spec = spec.half()

        audio_16k = None
        if need_v2_audio:
            audio_16k = self._resample(audio, sampling_rate, 16000)
            if is_half:
                audio_16k = audio_16k.half()

        zero_wav = np.zeros(int(sampling_rate * 0.3), dtype=np.float16 if is_half else np.float32)
        with torch.no_grad():
            wav16k, _ = librosa.load(abs_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k).to(self.device)
            zero_wav_torch = torch.from_numpy(zero_wav).to(self.device)
            if is_half:
                wav16k = wav16k.half()
                zero_wav_torch = zero_wav_torch.half()
            wav16k = torch.cat([wav16k, zero_wav_torch])
            hubert_feature = self.cnhuhbert_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)

        result = {
            "spec": spec,
            "raw_audio": raw_audio,
            "raw_sr": raw_sr,
            "audio_16k": audio_16k,
            "hubert_feature": hubert_feature,
        }
        with self._lock:
            self._reference_cache[cache_key] = result
        return {
            "spec": result["spec"],
            "raw_audio": result["raw_audio"],
            "raw_sr": result["raw_sr"],
            "audio_16k": result["audio_16k"],
            "hubert_feature": result["hubert_feature"],
            "cache_hit": False,
            "cache_key": cache_key,
        }
