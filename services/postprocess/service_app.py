import argparse
import os
import sys
import traceback

REEXEC_FLAG = "GPT_SOVITS_POSTPROCESS_LD_REEXEC"


def ensure_conda_runtime_libs():
    if os.environ.get(REEXEC_FLAG) == "1":
        return
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    conda_lib = os.path.join(conda_prefix, "lib")
    if not os.path.isdir(conda_lib):
        return
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    paths = [path for path in ld_library_path.split(":") if path]
    if paths[:1] == [conda_lib]:
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([conda_lib] + [path for path in paths if path != conda_lib])
    env[REEXEC_FLAG] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_conda_runtime_libs()

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "GPT_SoVITS"))

from services.postprocess.local_adapter import LocalPostprocessAdapter
from services.postprocess.schema import (
    SuperResolveRequest,
    SuperResolveResponse,
    VocoderSynthesisRequest,
    VocoderSynthesisResponse,
    decode_tensor,
    encode_tensor,
)


def load_service_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    server = raw.get("server") or {}
    postprocess = raw.get("postprocess") or {}
    missing = [key for key in ["device", "is_half"] if key not in postprocess]
    if missing:
        raise ValueError(f"postprocess config missing required keys: {', '.join(missing)}")

    return {
        "config_path": os.path.abspath(config_path),
        "host": server.get("host", "127.0.0.1"),
        "port": int(server.get("port", 9872)),
        "postprocess": dict(postprocess),
    }


def create_postprocess_app(adapter: LocalPostprocessAdapter, service_config: dict) -> FastAPI:
    app = FastAPI(title="GPT-SoVITS Shared Postprocess Service")

    @app.get("/health")
    def health():
        return adapter.get_health_status()

    @app.get("/meta")
    def meta():
        return {
            "service": "postprocess_backend",
            "config_path": service_config["config_path"],
            "host": service_config["host"],
            "port": service_config["port"],
            "postprocess": adapter.get_runtime_meta(),
        }

    @app.post("/postprocess/vocoder", response_model=VocoderSynthesisResponse)
    def postprocess_vocoder(request: VocoderSynthesisRequest):
        pred_spec = decode_tensor(request.pred_spec, adapter.device)
        audio, sample_rate = adapter.synthesize_vocoder(pred_spec, request.version)
        return VocoderSynthesisResponse(
            version=request.version,
            sample_rate=sample_rate,
            audio=encode_tensor(audio),
        )

    @app.post("/postprocess/sr", response_model=SuperResolveResponse)
    def postprocess_sr(request: SuperResolveRequest):
        audio = decode_tensor(request.audio, adapter.device)
        try:
            audio_hr, sample_rate = adapter.super_resolve(audio, request.sample_rate)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return SuperResolveResponse(sample_rate=sample_rate, audio=encode_tensor(audio_hr))

    return app


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS shared postprocess service")
    parser.add_argument(
        "-c",
        "--service-config",
        type=str,
        default="configs/services/postprocess.yaml",
        help="Path to postprocess service config yaml",
    )
    parser.add_argument("-a", "--bind-addr", type=str, default=None, help="Override bind address from config")
    parser.add_argument("-p", "--port", type=int, default=None, help="Override port from config")
    args = parser.parse_args()

    service_config = load_service_config(args.service_config)
    host = args.bind_addr or service_config["host"]
    port = args.port or service_config["port"]
    service_config["host"] = host
    service_config["port"] = port

    postprocess = service_config["postprocess"]
    adapter = LocalPostprocessAdapter(device=postprocess["device"], is_half=bool(postprocess["is_half"]))
    app = create_postprocess_app(adapter, service_config)

    try:
        if host == "None":
            host = None
        uvicorn.run(app=app, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
