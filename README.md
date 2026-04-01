<div align="center">

<h1>GPT-SoVITS-MultiWeight</h1>
Single-GPU multi-weight GPT-SoVITS inference with NVIDIA MPS.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<a href="https://trendshift.io/repositories/7033" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7033" alt="RVC-Boss%2FGPT-SoVITS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- img src="https://counter.seku.su/cmoe?name=gptsovits&theme=r34" /><br> -->

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![GitHub release](https://img.shields.io/github/v/release/RVC-Boss/gpt-sovits?style=for-the-badge&logo=github)](https://github.com/RVC-Boss/gpt-sovits/releases)

[![Train In Colab](https://img.shields.io/badge/Colab-Training-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/Colab-WebUI.ipynb)
[![Huggingface](https://img.shields.io/badge/免费在线体验-free_online_demo-yellow.svg?style=for-the-badge&logo=huggingface)](https://lj1995-gpt-sovits-proplus.hf.space/)
[![Image Size](https://img.shields.io/docker/image-size/xxxxrt666/gpt-sovits/latest?style=for-the-badge&logo=docker)](https://hub.docker.com/r/xxxxrt666/gpt-sovits)

[![简体中文](https://img.shields.io/badge/简体中文-阅读文档-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e)
[![English](https://img.shields.io/badge/English-Read%20Docs-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://rentry.co/GPT-SoVITS-guide#/)
[![Change Log](https://img.shields.io/badge/Change%20Log-View%20Updates-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/en/Changelog_EN.md)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)

**English** | [**中文简体**](./docs/cn/README.md) | [**日本語**](./docs/ja/README.md) | [**한국어**](./docs/ko/README.md) | [**Türkçe**](./docs/tr/README.md)

</div>

---

## Fork Focus

This fork is not centered on a generic WebUI deployment. Its primary goal is:

> run multiple fixed GPT-SoVITS weights on a single NVIDIA GPU, without vGPU, using process-level isolation plus NVIDIA MPS for controllable compute sharing.

Key points for this fork:

- Single card, multiple fixed-weight workers.
- No vGPU requirement.
- Use NVIDIA MPS for per-process compute partitioning, not full-card hard isolation.
- Shared frontend and postprocess components are reused across workers.
- Main model inference stays isolated per worker.

### What is shared

The following components are suitable for reuse or service sharing:

- text normalization and text split
- g2pw
- tokenizer / BERT
- CNHuBERT
- reference audio read-only preprocessing and read-only cache
- version-level vocoder
- SR model

### What stays isolated in MPS workers

The following parts are kept in dedicated fixed-weight workers and should not be shared as mutable runtime state:

- GPT / T2S main model
- SoVITS / VITS main model
- `prompt_cache`
- reference-audio context cache
- request-mode switch state
- any mutable inference-time model state

### Verified deployment target

- On a 24 GB GPU, this fork has been verified to run 8 fixed-weight workers stably with shared preprocess and postprocess services.
- Under that 24 GB / 8-worker setup, concurrent inference latency stays close across workers rather than diverging heavily.
- On RTX 4090, the target throughput is `RTF < 1`, which is sufficient for real-time streaming generation.

### Current support boundary

- Linux + CUDA only.
- NVIDIA GPU required.
- Compute capability `SM >= 3.5`.
- In practice, almost all currently used NVIDIA GPUs satisfy this requirement.

For architecture details, reuse boundaries, and rollout notes, see:

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)

## Features:

1. **Zero-shot TTS:** Input a 5-second vocal sample and experience instant text-to-speech conversion.

2. **Few-shot TTS:** Fine-tune the model with just 1 minute of training data for improved voice similarity and realism.

3. **Cross-lingual Support:** Inference in languages different from the training dataset, currently supporting English, Japanese, Korean, Cantonese and Chinese.

4. **WebUI Tools:** Integrated tools include voice accompaniment separation, automatic training set segmentation, Chinese ASR, and text labeling, assisting beginners in creating training datasets and GPT/SoVITS models.

**Check out our [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw) here!**

Unseen speakers few-shot fine-tuning demo:

https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb

**RTF(inference speed) of GPT-SoVITS v2 ProPlus**:
0.028 tested in 4060Ti, 0.014 tested in 4090 (1400words~=4min, inference time is 3.36s), 0.526 in M4 CPU. You can test our [huggingface demo](https://lj1995-gpt-sovits-proplus.hf.space/) (half H200) to experience high-speed inference .

请不要尬黑GPT-SoVITS推理速度慢，谢谢！

**User guide: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## Installation

> Note
>
> This README still contains a large amount of upstream GPT-SoVITS installation and WebUI content.
> For this fork, the maintained and verified path is Linux + CUDA + NVIDIA GPU.
> Windows, macOS, and non-CUDA deployment paths below should be treated as upstream reference, not this fork's primary support target.

For users in China, you can [click here](https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS-Official) to use AutoDL Cloud Docker to experience the full functionality online.

### Tested Environments

| Python Version | PyTorch Version  | Device        |
| -------------- | ---------------- | ------------- |
| Python 3.10    | PyTorch 2.5.1    | CUDA 12.4     |
| Python 3.11    | PyTorch 2.5.1    | CUDA 12.4     |
| Python 3.11    | PyTorch 2.7.0    | CUDA 12.8     |
| Python 3.9     | PyTorch 2.8.0dev | CUDA 12.8     |
| Python 3.9     | PyTorch 2.5.1    | Apple silicon |
| Python 3.11    | PyTorch 2.7.0    | Apple silicon |
| Python 3.9     | PyTorch 2.2.2    | CPU           |

### Windows

If you are a Windows user (tested with win>=10), you can [download the integrated package](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v3lora-20250228.7z?download=true) and double-click on _go-webui.bat_ to start GPT-SoVITS-WebUI.

**Users in China can [download the package here](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#KTvnO).**

Install the program by running the following commands:

```pwsh
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
pwsh -F install.ps1 --Device <CU126|CU128|CPU> --Source <HF|HF-Mirror|ModelScope> [--DownloadUVR5]
```

### Linux

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device <CU126|CU128|ROCM|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
```

### macOS

**Note: The models trained with GPUs on Macs result in significantly lower quality compared to those trained on other devices, so we are temporarily using CPUs instead.**

Install the program by running the following commands:

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device <MPS|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
```

### Install Manually

#### Install Dependences

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits

pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
```

#### Install FFmpeg

##### Conda Users

```bash
conda activate GPTSoVits
conda install ffmpeg
```

##### Ubuntu/Debian Users

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
```

##### Windows Users

Download and place [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) and [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) in the GPT-SoVITS root

Install [Visual Studio 2017](https://aka.ms/vs/17/release/vc_redist.x86.exe)

##### MacOS Users

```bash
brew install ffmpeg
```

### Running GPT-SoVITS with Docker

#### Docker Image Selection

Due to rapid development in the codebase and a slower Docker image release cycle, please:

- Check [Docker Hub](https://hub.docker.com/r/xxxxrt666/gpt-sovits) for the latest available image tags
- Choose an appropriate image tag for your environment
- `Lite` means the Docker image **does not include** ASR models and UVR5 models. You can manually download the UVR5 models, while the program will automatically download the ASR models as needed
- The appropriate architecture image (amd64/arm64) will be automatically pulled during Docker Compose
- Docker Compose will mount **all files** in the current directory. Please switch to the project root directory and **pull the latest code** before using the Docker image
- Optionally, build the image locally using the provided Dockerfile for the most up-to-date changes

#### Environment Variables

- `is_half`: Controls whether half-precision (fp16) is enabled. Set to `true` if your GPU supports it to reduce memory usage.

#### Shared Memory Configuration

On Windows (Docker Desktop), the default shared memory size is small and may cause unexpected behavior. Increase `shm_size` (e.g., to `16g`) in your Docker Compose file based on your available system memory.

#### Choosing a Service

The `docker-compose.yaml` defines two services:

- `GPT-SoVITS-CU126` & `GPT-SoVITS-CU128`: Full version with all features.
- `GPT-SoVITS-CU126-Lite` & `GPT-SoVITS-CU128-Lite`: Lightweight version with reduced dependencies and functionality.

To run a specific service with Docker Compose, use:

```bash
docker compose run --service-ports <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128>
```

#### Building the Docker Image Locally

If you want to build the image yourself, use:

```bash
bash docker_build.sh --cuda <12.6|12.8> [--lite]
```

#### Accessing the Running Container (Bash Shell)

Once the container is running in the background, you can access it using:

```bash
docker exec -it <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128> bash
```

## Credits

Special thanks to the following projects and contributors:

### Theoretical Research

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)

### Pretrained Models

- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### Text Frontend for Inference

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### WebUI Tools

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)

Thankful to @Naozumi520 for providing the Cantonese training set and for the guidance on Cantonese-related knowledge.

## Thanks to all contributors for their efforts

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
