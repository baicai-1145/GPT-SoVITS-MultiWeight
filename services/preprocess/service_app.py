import argparse
import os
import sys
import traceback

REEXEC_FLAG = "GPT_SOVITS_PREPROCESS_LD_REEXEC"


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
from fastapi import FastAPI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "GPT_SoVITS"))

from services.preprocess.local_adapter import LocalPreprocessAdapter
from services.preprocess.schema import (
    ReferencePreprocessRequest,
    TextPreprocessRequest,
    TextPreprocessResponse,
    TextSegmentRequest,
    TextSegmentsRequest,
    TextSegmentsResponse,
    build_reference_response,
    build_text_segment_response,
)


def load_service_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    server = raw.get("server") or {}
    frontend = raw.get("frontend") or {}

    def resolve_path(path_value: str) -> str:
        if path_value in [None, ""]:
            return path_value
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))

    missing = [key for key in ["device", "is_half", "bert_base_path", "cnhuhbert_base_path"] if key not in frontend]
    if missing:
        raise ValueError(f"frontend config missing required keys: {', '.join(missing)}")

    frontend = dict(frontend)
    frontend["bert_base_path"] = resolve_path(frontend.get("bert_base_path"))
    frontend["cnhuhbert_base_path"] = resolve_path(frontend.get("cnhuhbert_base_path"))

    return {
        "config_path": os.path.abspath(config_path),
        "host": server.get("host", "127.0.0.1"),
        "port": int(server.get("port", 9870)),
        "frontend": frontend,
    }


def create_preprocess_app(adapter: LocalPreprocessAdapter, service_config: dict) -> FastAPI:
    app = FastAPI(title="GPT-SoVITS Shared Preprocess Service")

    @app.get("/health")
    def health():
        return adapter.get_health_status()

    @app.get("/meta")
    def meta():
        return {
            "service": "preprocess_frontend",
            "config_path": service_config["config_path"],
            "host": service_config["host"],
            "port": service_config["port"],
            "frontend": adapter.get_runtime_meta(),
        }

    @app.post("/preprocess/text/segments", response_model=TextSegmentsResponse)
    def preprocess_text_segments(request: TextSegmentsRequest):
        payload = adapter.pre_segment_text(request.text, request.lang, request.text_split_method)
        return TextSegmentsResponse(**payload)

    @app.post("/preprocess/segment")
    def preprocess_segment(request: TextSegmentRequest):
        payload = adapter.preprocess_segment_text(request.text, request.lang, request.version)
        return build_text_segment_response(payload)

    @app.post("/preprocess/text", response_model=TextPreprocessResponse)
    def preprocess_text(request: TextPreprocessRequest):
        payload = adapter.preprocess_text(request.text, request.lang, request.text_split_method, request.version)
        return TextPreprocessResponse(
            segments=[build_text_segment_response(item) for item in payload["segments"]],
            cache_hit=bool(payload["cache_hit"]),
            cache_key=str(payload["cache_key"]),
        )

    @app.post("/preprocess/reference")
    def preprocess_reference(request: ReferencePreprocessRequest):
        payload = adapter.preprocess_reference_audio(
            ref_audio_path=request.ref_audio_path,
            sampling_rate=request.sampling_rate,
            filter_length=request.filter_length,
            hop_length=request.hop_length,
            win_length=request.win_length,
            is_half=request.is_half,
            need_v2_audio=request.need_v2_audio,
        )
        return build_reference_response(payload)

    return app


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS shared preprocess service")
    parser.add_argument(
        "-c",
        "--service-config",
        type=str,
        default="configs/services/frontend.yaml",
        help="Path to preprocess service config yaml",
    )
    parser.add_argument("-a", "--bind-addr", type=str, default=None, help="Override bind address from config")
    parser.add_argument("-p", "--port", type=int, default=None, help="Override port from config")
    args = parser.parse_args()

    service_config = load_service_config(args.service_config)
    host = args.bind_addr or service_config["host"]
    port = args.port or service_config["port"]
    service_config["host"] = host
    service_config["port"] = port

    frontend = service_config["frontend"]
    adapter = LocalPreprocessAdapter(
        bert_base_path=frontend["bert_base_path"],
        cnhuhbert_base_path=frontend["cnhuhbert_base_path"],
        device=frontend["device"],
        is_half=bool(frontend["is_half"]),
    )
    app = create_preprocess_app(adapter, service_config)

    try:
        if host == "None":
            host = None
        uvicorn.run(app=app, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
