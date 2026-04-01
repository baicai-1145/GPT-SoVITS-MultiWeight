import argparse
import os
import sys
import traceback

REEXEC_FLAG = "GPT_SOVITS_WORKER_LD_REEXEC"


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

    updated_paths = [conda_lib] + [path for path in paths if path != conda_lib]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(updated_paths)
    env[REEXEC_FLAG] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


ensure_conda_runtime_libs()

import uvicorn
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "GPT_SoVITS"))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from services.preprocess.local_adapter import LocalPreprocessAdapter
from services.preprocess.remote_client import RemotePreprocessClient
from services.postprocess.local_adapter import LocalPostprocessAdapter
from services.postprocess.remote_client import RemotePostprocessClient
from workers.runtime.worker_app import WorkerRouteOptions, create_tts_app


def load_worker_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}

    worker = raw_config.get("worker") or {}
    server = raw_config.get("server") or {}
    tts = raw_config.get("tts") or {}
    preprocess = raw_config.get("preprocess") or {}
    postprocess = raw_config.get("postprocess") or {}

    def resolve_path(path_value: str) -> str:
        if path_value in [None, ""]:
            return path_value
        if os.path.isabs(path_value):
            return path_value
        return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))

    missing_tts = [
        key
        for key in ["version", "device", "is_half", "t2s_weights_path", "vits_weights_path"]
        if key not in tts
    ]
    if missing_tts:
        raise ValueError(f"worker tts config missing required keys: {', '.join(missing_tts)}")

    tts = dict(tts)
    for key in ["t2s_weights_path", "vits_weights_path", "bert_base_path", "cnhuhbert_base_path"]:
        tts[key] = resolve_path(tts.get(key))

    return {
        "config_path": os.path.abspath(config_path),
        "worker_id": worker.get("worker_id") or os.path.splitext(os.path.basename(config_path))[0],
        "speaker_id": worker.get("speaker_id"),
        "description": worker.get("description", ""),
        "serial_inference": bool(worker.get("serial_inference", True)),
        "host": server.get("host", "127.0.0.1"),
        "port": int(server.get("port", 9880)),
        "tts": tts,
        "preprocess": {
            "mode": str(preprocess.get("mode", "local")).lower(),
            "base_url": preprocess.get("base_url"),
            "timeout_seconds": float(preprocess.get("timeout_seconds", 120.0)),
        },
        "postprocess": {
            "mode": str(postprocess.get("mode", "local")).lower(),
            "base_url": postprocess.get("base_url"),
            "timeout_seconds": float(postprocess.get("timeout_seconds", 120.0)),
        },
    }


def build_preprocess_adapter(worker_config: dict, tts_config: TTS_Config):
    preprocess = worker_config["preprocess"]
    mode = preprocess["mode"]
    if mode == "local":
        return LocalPreprocessAdapter(
            bert_base_path=tts_config.bert_base_path,
            cnhuhbert_base_path=tts_config.cnhuhbert_base_path,
            device=tts_config.device,
            is_half=tts_config.is_half,
        )
    if mode == "remote":
        base_url = preprocess.get("base_url")
        if not base_url:
            raise ValueError("remote preprocess mode requires preprocess.base_url")
        return RemotePreprocessClient(
            base_url=base_url,
            timeout_seconds=preprocess["timeout_seconds"],
            device=tts_config.device,
            is_half=tts_config.is_half,
        )
    raise ValueError(f"unsupported preprocess mode: {mode}")


def build_postprocess_adapter(worker_config: dict, tts_config: TTS_Config):
    postprocess = worker_config["postprocess"]
    mode = postprocess["mode"]
    if mode == "local":
        return LocalPostprocessAdapter(device=tts_config.device, is_half=tts_config.is_half)
    if mode == "remote":
        base_url = postprocess.get("base_url")
        if not base_url:
            raise ValueError("remote postprocess mode requires postprocess.base_url")
        return RemotePostprocessClient(
            base_url=base_url,
            timeout_seconds=postprocess["timeout_seconds"],
            device=tts_config.device,
            is_half=tts_config.is_half,
        )
    raise ValueError(f"unsupported postprocess mode: {mode}")


def build_worker_app(config_path: str):
    worker_config = load_worker_config(config_path)
    tts_config = TTS_Config({"custom": worker_config["tts"]})
    runtime_config_dir = os.path.join(PROJECT_ROOT, "configs", "workers", "runtime")
    os.makedirs(runtime_config_dir, exist_ok=True)
    tts_config.configs_path = os.path.join(runtime_config_dir, f"{worker_config['worker_id']}.tts_infer.yaml")
    tts_config.save_configs(tts_config.configs_path)
    print(tts_config)
    preprocess_adapter = build_preprocess_adapter(worker_config, tts_config)
    postprocess_adapter = build_postprocess_adapter(worker_config, tts_config)
    tts_pipeline = TTS(
        tts_config,
        main_weights_locked=True,
        preprocess_adapter=preprocess_adapter,
        postprocess_adapter=postprocess_adapter,
    )

    route_options = WorkerRouteOptions(
        enable_control=False,
        enable_reference_audio_route=False,
        enable_hot_reload_routes=False,
        serial_inference=worker_config["serial_inference"],
        mode_label="fixed_weight_worker",
        metadata={
            "worker_id": worker_config["worker_id"],
            "speaker_id": worker_config["speaker_id"],
            "description": worker_config["description"],
            "fixed_weights": True,
            "config_path": worker_config["config_path"],
            "preprocess": worker_config["preprocess"],
            "postprocess": worker_config["postprocess"],
        },
    )
    app = create_tts_app(tts_pipeline, tts_config, route_options=route_options)
    return app, worker_config


def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS fixed-weight worker")
    parser.add_argument(
        "-c",
        "--worker-config",
        type=str,
        required=True,
        help="Path to worker config yaml",
    )
    parser.add_argument("-a", "--bind-addr", type=str, default=None, help="Override bind address from config")
    parser.add_argument("-p", "--port", type=int, default=None, help="Override port from config")
    args = parser.parse_args()

    app, worker_config = build_worker_app(args.worker_config)
    host = args.bind_addr or worker_config["host"]
    port = args.port or worker_config["port"]

    try:
        if host == "None":
            host = None
        uvicorn.run(app=app, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
