"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`

### 命令控制

endpoint: `/control`

### 切换GPT模型

endpoint: `/set_gpt_weights`

### 切换Sovits模型

endpoint: `/set_sovits_weights`
"""

import argparse
import os
import signal
import sys
import traceback

import uvicorn

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from workers.runtime.worker_app import WorkerRouteOptions, create_tts_app

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default=9880, help="default: 9880")
args = parser.parse_args()

config_path = args.tts_config or "GPT_SoVITS/configs/tts_infer.yaml"
port = args.port
host = args.bind_addr
argv = sys.argv

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


APP = create_tts_app(
    tts_pipeline,
    tts_config,
    route_options=WorkerRouteOptions(
        enable_control=True,
        enable_reference_audio_route=True,
        enable_hot_reload_routes=True,
        serial_inference=False,
        mode_label="legacy_api_v2",
        metadata={
            "fixed_weights": False,
            "config_path": os.path.abspath(config_path),
        },
        control_handler=handle_control,
    ),
)


if __name__ == "__main__":
    try:
        if host == "None":
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
