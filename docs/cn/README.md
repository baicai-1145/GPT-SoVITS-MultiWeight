<div align="center">

<h1>GPT-SoVITS-MultiWeight</h1>
基于 NVIDIA MPS 的单卡多权重 GPT-SoVITS 推理工程.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/RVC-Boss/GPT-SoVITS)

<a href="https://trendshift.io/repositories/7033" target="_blank"><img src="https://trendshift.io/api/badge/repositories/7033" alt="RVC-Boss%2FGPT-SoVITS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[![Python](https://img.shields.io/badge/python-3.10--3.12-blue?style=for-the-badge&logo=python)](https://www.python.org)
[![GitHub release](https://img.shields.io/github/v/release/RVC-Boss/gpt-sovits?style=for-the-badge&logo=github)](https://github.com/RVC-Boss/gpt-sovits/releases)

[![Train In Colab](https://img.shields.io/badge/Colab-Training-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com/github/RVC-Boss/GPT-SoVITS/blob/main/Colab-WebUI.ipynb)
[![Huggingface](https://img.shields.io/badge/免费在线体验-free_online_demo-yellow.svg?style=for-the-badge&logo=huggingface)](https://lj1995-gpt-sovits-proplus.hf.space/)
[![Image Size](https://img.shields.io/docker/image-size/xxxxrt666/gpt-sovits/latest?style=for-the-badge&logo=docker)](https://hub.docker.com/r/xxxxrt666/gpt-sovits)

[![简体中文](https://img.shields.io/badge/简体中文-阅读文档-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e)
[![English](https://img.shields.io/badge/English-Read%20Docs-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://rentry.co/GPT-SoVITS-guide#/)
[![Change Log](https://img.shields.io/badge/Change%20Log-View%20Updates-blue?style=for-the-badge&logo=googledocs&logoColor=white)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/docs/en/Changelog_EN.md)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative)](https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE)


[**English**](../../README.md) | **中文简体** | [**日本語**](../ja/README.md) | [**한국어**](../ko/README.md) | [**Türkçe**](../tr/README.md)

</div>

---

## 本 Fork 定位

这个 fork 的重点不是通用 WebUI 部署，而是：

> 在单张 NVIDIA GPU 上，不依赖 vGPU，基于多进程固定权重 + NVIDIA MPS，实现多权重并发推理。

当前项目重点如下：

- 单卡部署，多固定权重 worker 并发推理。
- 不需要 vGPU。
- 使用 NVIDIA MPS 做进程级算力份额控制，不宣称整卡硬隔离。
- 共享前处理与后处理公共组件，减少重复加载。
- 主模型推理保持 worker 级隔离。

### 哪些组件会复用

以下模块适合做共享服务或只读复用：

- 文本清洗与切分
- g2pw
- tokenizer / BERT
- CNHuBERT
- 参考音频只读预处理与只读缓存
- 按版本共享的 vocoder
- SR 模型

### 哪些组件会独立成 MPS worker

以下部分保持在固定权重 worker 内独立运行，不共享可变推理态：

- GPT / T2S 主模型
- SoVITS / VITS 主模型
- `prompt_cache`
- 参考音频上下文缓存
- 请求模式切换状态
- 任何推理过程中会被修改的模型内部状态

### 当前已验证能力

- 在 24GB 显存卡上，当前方案可稳定运行 8 个固定权重 worker。
- 在 24GB / 8 worker 配置下，并发推理时各 worker 延迟基本一致，没有明显拉开。
- 在 RTX 4090 上，目标能力为 `RTF < 1`，可用于实时流式生成。

### 当前支持范围

- 目前仅支持 Linux + CUDA。
- 需要 NVIDIA GPU。
- 需要计算能力 `SM >= 3.5`。
- 基本上现在仍在使用的 NVIDIA 显卡都满足这个要求。

更多架构边界、复用拆分和实施步骤可参考：

- [ARCHITECTURE.md](../../ARCHITECTURE.md)
- [IMPLEMENTATION_PLAN.md](../../IMPLEMENTATION_PLAN.md)

## 功能

1. **零样本文本到语音 (TTS):** 输入 5 秒的声音样本, 即刻体验文本到语音转换.

2. **少样本 TTS:** 仅需 1 分钟的训练数据即可微调模型, 提升声音相似度和真实感.

3. **跨语言支持:** 支持与训练数据集不同语言的推理, 目前支持英语、日语、韩语、粤语和中文.

4. **WebUI 工具:** 集成工具包括声音伴奏分离、自动训练集分割、中文自动语音识别(ASR)和文本标注, 协助初学者创建训练数据集和 GPT/SoVITS 模型.

**查看我们的介绍视频 [demo video](https://www.bilibili.com/video/BV12g4y1m7Uw)**

未见过的说话者 few-shot 微调演示:

<https://github.com/RVC-Boss/GPT-SoVITS/assets/129054828/05bee1fa-bdd8-4d85-9350-80c060ab47fb>

**用户手册: [简体中文](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e) | [English](https://rentry.co/GPT-SoVITS-guide#/)**

## 安装

> 说明
>
> 本文档仍保留了较多上游 GPT-SoVITS 的安装与 WebUI 内容。
> 对本 fork 而言，当前维护和验证的主路径是 Linux + CUDA + NVIDIA GPU。
> 下方的 Windows、macOS、非 CUDA 路径更适合作为上游参考，而不是本 fork 当前的主要支持范围。

中国地区的用户可[点击此处](https://www.codewithgpu.com/i/RVC-Boss/GPT-SoVITS/GPT-SoVITS-Official)使用 AutoDL 云端镜像进行体验.

### 测试通过的环境

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

如果你是 Windows 用户 (已在 win>=10 上测试), 可以下载[整合包](https://huggingface.co/lj1995/GPT-SoVITS-windows-package/resolve/main/GPT-SoVITS-v3lora-20250228.7z?download=true), 解压后双击 go-webui.bat 即可启动 GPT-SoVITS-WebUI.

**中国地区的用户可以[在此处下载整合包](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/dkxgpiy9zb96hob4#KTvnO).**

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

**注: 在 Mac 上使用 GPU 训练的模型效果显著低于其他设备训练的模型, 所以我们暂时使用 CPU 进行训练.**

运行以下的命令来安装本项目:

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
bash install.sh --device <MPS|CPU> --source <HF|HF-Mirror|ModelScope> [--download-uvr5]
```

### 手动安装

#### 安装依赖

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits

pip install -r extra-req.txt --no-deps
pip install -r requirements.txt
```

#### 安装 FFmpeg

##### Conda 用户

```bash
conda activate GPTSoVits
conda install ffmpeg
```

##### Ubuntu/Debian 用户

```bash
sudo apt install ffmpeg
sudo apt install libsox-dev
```

##### Windows 用户

下载并将 [ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe) 和 [ffprobe.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe) 放置在 GPT-SoVITS 根目录下

安装 [Visual Studio 2017](https://aka.ms/vs/17/release/vc_redist.x86.exe) 环境

##### MacOS 用户

```bash
brew install ffmpeg
```

### 运行 GPT-SoVITS (使用 Docker)

#### Docker 镜像选择

由于代码库更新频繁, 而 Docker 镜像的发布周期相对较慢, 请注意：

- 前往 [Docker Hub](https://hub.docker.com/r/xxxxrt666/gpt-sovits) 查看最新可用的镜像标签(tags)
- 根据你的运行环境选择合适的镜像标签
- `Lite` Docker 镜像**不包含** ASR 模型和 UVR5 模型. 你可以自行下载 UVR5 模型, ASR 模型则会在需要时由程序自动下载
- 在使用 Docker Compose 时, 会自动拉取适配的架构镜像 (amd64 或 arm64)
- Docker Compose 将会挂载当前目录的**所有文件**, 请在使用 Docker 镜像前先切换到项目根目录并**拉取代码更新**
- 可选：为了获得最新的更改, 你可以使用提供的 Dockerfile 在本地构建镜像

#### 环境变量

- `is_half`：控制是否启用半精度(fp16). 如果你的 GPU 支持, 设置为 `true` 可以减少显存占用

#### 共享内存配置

在 Windows (Docker Desktop) 中, 默认共享内存大小较小, 可能导致运行异常. 请在 Docker Compose 文件中根据系统内存情况, 增大 `shm_size` (例如设置为 `16g`)

#### 选择服务

`docker-compose.yaml` 文件定义了两个主要服务类型：

- `GPT-SoVITS-CU126` 与 `GPT-SoVITS-CU128`：完整版, 包含所有功能
- `GPT-SoVITS-CU126-Lite` 与 `GPT-SoVITS-CU128-Lite`：轻量版, 依赖更少, 功能略有删减

如需使用 Docker Compose 运行指定服务, 请执行：

```bash
docker compose run --service-ports <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128>
```

#### 本地构建 Docker 镜像

如果你希望自行构建镜像, 请使用以下命令：

```bash
bash docker_build.sh --cuda <12.6|12.8> [--lite]
```

#### 访问运行中的容器 (Bash Shell)

当容器在后台运行时, 你可以通过以下命令进入容器：

```bash
docker exec -it <GPT-SoVITS-CU126-Lite|GPT-SoVITS-CU128-Lite|GPT-SoVITS-CU126|GPT-SoVITS-CU128> bash
```

## 致谢

特别感谢以下项目和贡献者:

### 理论研究

- [ar-vits](https://github.com/innnky/ar-vits)
- [SoundStorm](https://github.com/yangdongchao/SoundStorm/tree/master/soundstorm/s1/AR)
- [vits](https://github.com/jaywalnut310/vits)
- [TransferTTS](https://github.com/hcy71o/TransferTTS/blob/master/models.py#L556)
- [contentvec](https://github.com/auspicious3000/contentvec/)
- [hifi-gan](https://github.com/jik876/hifi-gan)
- [fish-speech](https://github.com/fishaudio/fish-speech/blob/main/tools/llama/generate.py#L41)
- [f5-TTS](https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/backbones/dit.py)
- [shortcut flow matching](https://github.com/kvfrans/shortcut-models/blob/main/targets_shortcut.py)

### 预训练模型

- [Chinese Speech Pretrain](https://github.com/TencentGameMate/chinese_speech_pretrain)
- [Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [eresnetv2](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)

### 推理用文本前端

- [paddlespeech zh_normalization](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization)
- [split-lang](https://github.com/DoodleBears/split-lang)
- [g2pW](https://github.com/GitYCC/g2pW)
- [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)
- [paddlespeech g2pw](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/g2pw)

### WebUI 工具

- [ultimatevocalremovergui](https://github.com/Anjok07/ultimatevocalremovergui)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [SubFix](https://github.com/cronrpc/SubFix)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [gradio](https://github.com/gradio-app/gradio)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [AP-BWE](https://github.com/yxlu-0102/AP-BWE)

感谢 @Naozumi520 提供粤语训练集, 并在粤语相关知识方面给予指导.

## 感谢所有贡献者的努力

<a href="https://github.com/RVC-Boss/GPT-SoVITS/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Boss/GPT-SoVITS" />
</a>
