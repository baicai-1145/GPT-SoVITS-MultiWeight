import os

import torch

from BigVGAN.bigvgan import BigVGAN
from module.models import Generator
from tools.audio_sr import AP_BWE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRETRAINED_ROOT = os.path.join(PROJECT_ROOT, "GPT_SoVITS", "pretrained_models")
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


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"Attribute {item} not found") from exc

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super().__setitem__(key, value)
        super().__setattr__(key, value)


class LocalPostprocessAdapter:
    backend_name = "local"

    def __init__(self, device, is_half: bool = False):
        self.device = torch.device(device)
        self.is_half = bool(is_half)
        self.vocoders: dict[str, torch.nn.Module] = {}
        self.sr_model = None
        self.sr_model_not_exist = False

    def _apply_precision_and_device(self, model):
        model = model.eval()
        if self.is_half and str(self.device) != "cpu":
            model = model.half().to(self.device)
        else:
            model = model.to(self.device)
        return model

    def get_vocoder_config(self, version: str) -> dict:
        if version not in VOCODER_CONFIGS:
            raise ValueError(f"unsupported vocoder version: {version}")
        return dict(VOCODER_CONFIGS[version])

    def _load_vocoder(self, version: str):
        if version in self.vocoders:
            return self.vocoders[version]

        if version == "v3":
            model = BigVGAN.from_pretrained(
                os.path.join(PRETRAINED_ROOT, "models--nvidia--bigvgan_v2_24khz_100band_256x"),
                use_cuda_kernel=False,
            )
            model.remove_weight_norm()
        elif version == "v4":
            model = Generator(
                initial_channel=100,
                resblock="1",
                resblock_kernel_sizes=[3, 7, 11],
                resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsample_rates=[10, 6, 2, 2, 2],
                upsample_initial_channel=512,
                upsample_kernel_sizes=[20, 12, 4, 4, 4],
                gin_channels=0,
                is_bias=True,
            )
            model.remove_weight_norm()
            state_dict_g = torch.load(
                os.path.join(PRETRAINED_ROOT, "gsv-v4-pretrained", "vocoder.pth"),
                map_location="cpu",
                weights_only=False,
            )
            print("loading vocoder", model.load_state_dict(state_dict_g))
        else:
            raise ValueError(f"unsupported vocoder version: {version}")

        model = self._apply_precision_and_device(model)
        self.vocoders[version] = model
        return model

    def warmup_vocoder(self, version: str):
        return self._load_vocoder(version)

    def init_sr_model(self):
        if self.sr_model is not None:
            return self.sr_model
        try:
            self.sr_model = AP_BWE(self.device, DictToAttrRecursive)
            self.sr_model_not_exist = False
        except FileNotFoundError:
            self.sr_model_not_exist = True
            self.sr_model = None
        return self.sr_model

    def enable_half_precision(self, enable: bool = True):
        self.is_half = bool(enable)
        if str(self.device) == "cpu" and self.is_half:
            self.is_half = False
        for version, model in list(self.vocoders.items()):
            self.vocoders[version] = self._apply_precision_and_device(model)

    def set_device(self, device):
        self.device = torch.device(device)
        for version, model in list(self.vocoders.items()):
            self.vocoders[version] = self._apply_precision_and_device(model)
        if self.sr_model is not None:
            self.sr_model = self.sr_model.to(self.device)

    def get_health_status(self) -> dict:
        return {
            "ready": True,
            "backend": self.backend_name,
            "device": str(self.device),
            "is_half": self.is_half,
            "loaded_vocoders": sorted(self.vocoders.keys()),
            "sr_model_loaded": self.sr_model is not None,
            "sr_model_missing": self.sr_model_not_exist,
        }

    def get_runtime_meta(self) -> dict:
        return {
            "backend": self.backend_name,
            "device": str(self.device),
            "is_half": self.is_half,
            "loaded_vocoders": sorted(self.vocoders.keys()),
            "sr_model_loaded": self.sr_model is not None,
            "sr_model_missing": self.sr_model_not_exist,
            "supported_vocoder_versions": sorted(VOCODER_CONFIGS.keys()),
        }

    def synthesize_vocoder(self, pred_spec: torch.Tensor, version: str) -> tuple[torch.Tensor, int]:
        model = self._load_vocoder(version)
        config = self.get_vocoder_config(version)
        pred_spec = pred_spec.to(self.device)
        with torch.inference_mode():
            wav_gen = model(pred_spec)
            audio = wav_gen[0][0]
        return audio, config["sr"]

    def super_resolve(self, audio: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, int]:
        model = self.init_sr_model()
        if model is None:
            raise FileNotFoundError("super-resolution model is not available")
        audio = audio.to(self.device)
        audio_hr, sr = model(audio, sample_rate)
        audio_tensor = torch.from_numpy(audio_hr).to(self.device)
        return audio_tensor, sr
