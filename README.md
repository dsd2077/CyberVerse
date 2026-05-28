<h1 align="center">CyberVerse</h1>
<p align="center"><em>CyberVerse is an open-source <strong>real-time digital-human Agent framework</strong>. It uses WebRTC, persona memory, tools, RAG, and optional digital-human video capabilities to help you build AI agents centered on voice interaction.</em></p>

<p align="center">
  <a href="README.md"><strong>English</strong></a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg" alt="License: GPL v3"/></a>
  <a href="https://github.com/dsd2077/CyberVerse/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome"/></a>
  <a href="https://oosmetrics.com/repo/dsd2077/CyberVerse"><img src="https://api.oosmetrics.com/api/v1/badge/achievement/4795438a-70e7-4997-bd8a-93e7a13c8d81.svg" alt="oosmetrics: Top 1 in Streaming by velocity - 2026-05-12"/></a>
</p>

<p align="center">
  <a href="docs/assets/logo.png"><img src="docs/assets/logo.png" alt="CyberVerse logo" width="100%"/></a>
</p>

---

### One Photo. A Living Digital Human.

> Ever dreamed of having your own J.A.R.V.I.S. — an AI that truly sees you, hears you, and talks back in real time?
>
> Want to see someone you've lost again, hear their voice, watch them smile at you?
>
> Or maybe there's a character you've always wished you could bring to life?
>
> **Just one photo. CyberVerse makes them alive.**

## What is a Digital-Human Agent?

<p align="center">
  <a href="docs/assets/digital-human-agent.jpeg"><img src="docs/assets/digital-human-agent.jpeg" alt="CyberVerse digital-human Agent" width="100%"/></a>
</p>

## Demo
<p align="center"><em>The following characters are demo examples only. They are not bundled with CyberVerse and are not provided for commercial use.</em></p>

<p align="center">
  <a href="docs/assets/character1.png"><img src="docs/assets/character1.png" alt="CyberVerse character selection gallery" width="100%"/></a>
</p>

<p align="center">
  <a href="docs/assets/character2.png"><img src="docs/assets/character2.png" alt="CyberVerse character gallery examples" width="100%"/></a>
</p>

<div align="center">

| [![](docs/assets/爱丽丝.mov.png)](https://youtu.be/Lk88sew2x4o) | [![](docs/assets/丽娜.mov.png)](https://youtu.be/8jdQ3ThcwgA) |
|:---:|:---:|
| [**Alice — watch on YouTube**](https://youtu.be/Lk88sew2x4o) | [**Lina — watch on YouTube**](https://youtu.be/8jdQ3ThcwgA) |

| [![](docs/assets/小龙女.mov.png)](https://youtu.be/WjEHUYZx5Gs) |
|:---:|
| [**Xiaolongnü — watch on YouTube**](https://youtu.be/WjEHUYZx5Gs) |

</div>

## Features

### Realtime Voice Agent

Voice is CyberVerse's default interaction mode, designed for low-latency realtime conversations that can run for long sessions. Users can continuously talk with an Agent through a microphone, interrupt the model while it is speaking, and mix voice and text input in the same conversation turn.

Each character can have its own voice, welcome message, and personality configuration, and voice cloning is supported. Conversations support pause and resume; when `inference.avatar.enabled` is set to `false`, the platform runs in pure voice mode, publishes only the audio stream, requires no local Avatar GPU, and keeps the core voice experience intact.

### Audio/Video over WebRTC

The session pipeline is built on WebRTC and can choose direct P2P (embedded TURN / NAT traversal) or LiveKit SFU mode based on the deployment scenario, balancing low latency with connectivity in complex network environments.

In standard mode and supported omni sessions, the Agent can also receive user camera frames or screen-sharing frames as visual input, enabling face-to-face interaction that can listen and see instead of being limited to plain text context.

### PersonaAgent + SubAgent Tasks

CyberVerse uses a multi-agent architecture: PersonaAgent stays in the foreground to maintain fluid conversation, respond quickly to interruptions, and handle context switches; long-running work such as search, research, material organization, summarization, and HTML report generation is delegated to background SubAgents asynchronously.

This keeps complex tasks from slowing down voice turns. Users can keep speaking, ask follow-up questions, or adjust direction, and PersonaAgent can return the SubAgent result once it is ready.

### Character Memory and RAG

Each character's conversation history is persisted to local disk and automatically loaded when you re-enter a conversation, preserving continuity across sessions. You can also import knowledge bases, documents, and biographical material for a character; the system indexes them for retrieval-augmented generation, making answers better aligned with the character's background and persona.

### Optional Digital Human Video

When you have GPU resources and want the Agent to be visible, enable avatar inference: a single character reference image can drive realtime facial animation, lip-sync, and cached idle video playback through configurable backends such as FlashHead and LiveAct. If you do not have a GPU or do not need video yet, disable it to return to a pure voice Agent; the same character and persona configuration continues to work.

### Plugin-Based Stack

Brain, voice, hearing, tools, memory, and face are all replaceable modules. You can combine omni models, LLMs, TTS, ASR, embeddings, RAG, tool calls, and Avatar backends in `cyberverse_config.yaml`, then configure different vendors' API keys and service endpoints in the web UI at **`/settings`** to switch providers and model combinations by scenario.

## Quick Start

### Prerequisites

- Node 18+
- Go 1.25 (required: `protoc-gen-go`, `protoc-gen-go-grpc`)
- Conda
- Python 3.10+
- FFmpeg

> For pure voice sessions, no local avatar GPU is required. Runtime cost depends on the realtime voice/omni/LLM/TTS/ASR providers you configure.

To verify, use:

```bash
node --version
go version
protoc --version
ffmpeg -version
conda --version
```

### Step 1: Clone

```bash
git clone https://github.com/dsd2077/CyberVerse.git
cd CyberVerse
```

### Step 2: Create Python environment

```bash
conda create -n cyberverse python=3.10
conda activate cyberverse
```

### Step 3: Configure environment variables

```bash
cp infra/.env.example .env
```

Edit `.env` and fill in the supported API keys:

Alibaba Cloud Qwen-series models:

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

Or Volcengine Doubao-series models:

```env
DOUBAO_ACCESS_TOKEN=your_doubao_access_token
DOUBAO_APP_ID=your_doubao_app_id
```

Doubao Voice: follow the [Volcengine quick start](https://www.volcengine.com/docs/6561/2119699?lang=zh) to get **App ID** / **API Key**, then fill in `DOUBAO_APP_ID` / `DOUBAO_ACCESS_TOKEN`.

After the stack is running, you can change API keys and service endpoints from the web UI at **`/settings`** instead of editing `.env` only.

### Step 4: Create local config and enable voice-only mode

```bash
cp infra/cyberverse_config.example.yaml cyberverse_config.yaml
```

Edit `cyberverse_config.yaml`:

```yaml
inference:
  avatar:
    enabled: false
```

With `enabled: false`, CyberVerse runs as a pure voice agent assistant.


### Step 5: Install project dependencies

```bash
make setup
```

This installs the base editable package (`[dev,inference]`), generates gRPC stubs, and installs frontend dependencies.

Install the voice-agent extras used by the default config:

```bash
# all optional groups at once
pip install -e ".[all]"
```

### Step 6: Start services (3 terminals)

**Terminal 1** — Python inference server:

```bash
conda activate cyberverse
make inference
```

**Terminal 2** — Go API server:

```bash
make server
```

**Terminal 3** — Frontend:

```bash
make frontend
```

### Step 7: Verify

```bash
# Check API health
curl -s http://localhost:8080/api/v1/health
```

Open http://localhost:5173 in your browser.

## Optional: Full Digital-Human Video

If you want to drive realtime Avatar video with FlashHead or LiveAct, follow the steps below.

### Additional Requirements

- GPU with CUDA 12.8+
- PyTorch 2.8 (CUDA 12.8)
- FFmpeg with `libvpx` for video encoding
- Avatar model weights

Install PyTorch (CUDA 12.8):

```bash
pip3 install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Install vllm if you use LiveAct:

```bash
pip install vllm==0.11.0
```

### Download Model Weights

CyberVerse currently supports **FlashHead** and **LiveAct**; download only what you need. More models will continue to be added.

```bash
pip install "huggingface_hub[cli]"
```

#### FlashHead (SoulX-FlashHead)

| Model Component | Description | Link |
| :--- | :--- | :--- |
| `SoulX-FlashHead-1_3B` | 1.3B FlashHead weights | [Hugging Face](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B), [ModelScope](https://modelscope.cn/models/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | Audio feature extractor | [Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h), [ModelScope](https://modelscope.cn/models/facebook/wav2vec2-base-960h) |

```bash
# If you are in mainland China, you can use a mirror first:
# export HF_ENDPOINT=https://hf-mirror.com

hf download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir ./checkpoints/SoulX-FlashHead-1_3B

hf download facebook/wav2vec2-base-960h \
  --local-dir ./checkpoints/wav2vec2-base-960h
```

#### LiveAct (SoulX-LiveAct)

| ModelName | Download |
|-----------|----------|
| SoulX-LiveAct | [Hugging Face](https://huggingface.co/Soul-AILab/LiveAct), [ModelScope](https://modelscope.cn/models/Soul-AILab/LiveAct) |
| chinese-wav2vec2-base | [Hugging Face](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base), [ModelScope](https://modelscope.cn/models/TencentGameMate/chinese-wav2vec2-base) |

```bash
hf download Soul-AILab/LiveAct \
  --local-dir ./checkpoints/LiveAct

hf download TencentGameMate/chinese-wav2vec2-base \
  --local-dir ./checkpoints/chinese-wav2vec2-base
```

### Configure Avatar Inference

Set `enabled: true`, then update the model paths to match your local checkpoints:

```yaml
inference:
  avatar:
    enabled: true
    default: "flash_head"               # selects which avatar model to start; if set to live_act, fill the live_act section below
    runtime:
      cuda_visible_devices: 0      # shared GPU ID(s), e.g. 0,1 for multi-GPU
      world_size: 1                # shared GPU count, set to 2 for dual-GPU
    flash_head:
      checkpoint_dir: "./checkpoints/SoulX-FlashHead-1_3B"  # ← your path
      wav2vec_dir: "./checkpoints/wav2vec2-base-960h"        # ← your path
      model_type: "lite"           # "pro" for higher quality (needs more GPU)
      compile_model: true
      compile_vae: true
      dist_worker_main_thread: true
      infer_params:
        frame_num: 33
        motion_frames_latent_num: 2
        tgt_fps: 20
        sample_rate: 16000
        sample_shift: 5
        color_correction_strength: 1.0
        cached_audio_duration: 8
        num_heads: 12
        height: 512
        width: 512
    live_act:
      ckpt_dir: "./checkpoints/LiveAct"                     # ← your path
      wav2vec_dir: "./checkpoints/chinese-wav2vec2-base"   # ← your path
      seed: 42
      compile_wan_model: false
      compile_vae_decode: false
      dist_worker_main_thread: true
      default_prompt: "一个人在说话"
      infer_params:
        size: "320*480"
        fps: 20
        audio_cfg: 1.0
```

You can also adjust these options later in the web UI.

### SageAttention & FlashAttention (Optional)

```bash
# SageAttention (source build)
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # Optional
python setup.py install
```

```bash
# FlashAttention (optional)
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

> If compilation is slow, download a prebuilt wheel from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2) and `pip install <wheel>.whl`.

### Avatar Hardware Benchmarks

Realtime digital-human video requires GPU acceleration. Below are benchmarks for FlashHead and LiveAct avatar models:

| Model | Quality | GPU | Count | Resolution | FPS | Real-time? |
|-------|---------|-----|-------|------------|-----|------------|
| FlashHead 1.3B | Pro | RTX 5090 | 2 | 512×512 | 25+ | ✅ Yes |
| FlashHead 1.3B | Pro | RTX 5090 | 1 | 464x464 | 20 | ✅ Yes |
| FlashHead 1.3B | Pro | RTX PRO 6000 | 1 | 512×512 | 20 | ✅ Yes |
| FlashHead 1.3B | Pro | RTX 4090 | 1 | 512×512 | ~10.8 | ❌ No |
| FlashHead 1.3B | Lite | RTX 4090 | 1 | 512×512 | 25+ | ✅ Yes |
| LiveAct 18B | — | RTX PRO 6000 | 2 | 320×480 | 20 | ✅ Yes |
| LiveAct 18B | — | RTX PRO 6000 | 1 | 256×417 | 20 | ✅ Yes |

> **Pro** favors visual quality; **Lite** favors speed. The table reflects typical **quality–compute** balances — more GPU headroom lets you push higher quality; tighter hardware calls for lower settings (resolution, **Pro** vs **Lite**, etc.) to stay realtime.

When avatar inference is enabled, `make inference` reads `inference.avatar.default` from `cyberverse_config.yaml` and initializes exactly that one avatar model in the current inference process. Wait until you see:

- `Active avatar model initialized: <model_name>`
- `CyberVerse Inference Server started on port 50051`

## Remote Access Notes

When `streaming_mode: direct` uses the embedded TURN server, the browser must be able to reach the server's `8443/TCP`. If the page loads but audio/video never connects, or the server logs show `ICE connection state: failed` or `publish timeout waiting for connection`, first check whether your machine can reach port `8443` on the server:

```bash
nc -vz <server-ip> 8443
```

If `8443` is not reachable, the usual cause is a cloud security group, firewall, or NAT restriction. In that case, you can forward your local `8443` to the server through an SSH tunnel:

```bash
ssh -L 8443:127.0.0.1:8443 user@host -p port
```

After the tunnel is established, the browser will access the remote TURN service through local `127.0.0.1:8443`.

If you want the browser to connect to the remote server directly instead of through an SSH tunnel, set `pipeline.ice_public_ip` in `cyberverse_config.yaml` to the server's public IP or domain. If you are using an SSH tunnel, you can keep the default value (`127.0.0.1`).

## Roadmap

### 1. **Realtime Audio/Video Agent Platform**

Make voice-first realtime agents easy to run, customize, and embed.

- [x] Character CRUD with multiple reference images, active image, fixed/random display mode, optional face crop, tags, voice fields, personality, welcome message, and system prompt
- [x] Realtime voice sessions over WebRTC — direct P2P (embedded TURN) or LiveKit SFU
- [x] Pure voice sessions with `inference.avatar.enabled: false`
- [x] Pluggable modules (omni model, LLM, TTS, ASR, embedding, RAG, avatar); configure different vendors' API keys via YAML and UI settings
- [x] Session management: per-character chat history persisted to disk and loaded when a conversation starts
- [x] Voice cloning: supports Doubao voice cloning
- [x] Hybrid input: supports both voice and text in the same conversation
- [x] Voice interruption while the model is speaking, plus session pause and resume
- [x] User camera input and screen-sharing visual frames in standard mode and supported omni sessions
- [x] PersonaAgent and background SubAgent task execution
- [x] Import knowledge, documents, and biographical material for character-grounded RAG Q&A
- [ ] Embeddable for developers (Web component or SDK) to integrate self-hosted instances into their own sites
- [ ] Live streaming: audio/video output for broadcast-style use cases

### 2. **Realtime Digital-Human Calls**

When Avatar GPU resources are available, turn the voice Agent into a realtime video call.

- [x] Realtime avatar video driven from reference images via configurable avatar plugins (e.g. FlashHead, LiveAct)
- [x] Cached idle video playback for character presence
- [x] Audio/video synchronization for realtime speaking segments
- [ ] More avatar backends with different quality/latency/cost tradeoffs
- [ ] Better avatar deployment profiles for consumer GPU, workstation GPU, and cloud GPU environments

### 3. **Agent Network**

Connect multiple agents so they can communicate, collaborate, and form networks.

- [ ] Enable agent-to-agent communication
- [ ] Enable multi-agent collaboration and delegation
- [ ] Enable shared memory and shared knowledge between agents
- [ ] Build an open network of connected agents

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

## Acknowledgements

- [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) — Avatar model by Soul AI Lab

- [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) - Avatar model by Soul AI Lab
- [Pion](https://github.com/pion/webrtc) — Go WebRTC implementation
- [Linux.do](https://linux.do/)

## ❓ FAQ

### What is CyberVerse?
CyberVerse is an open-source real-time digital-human Agent framework. It uses WebRTC, persona memory, tools, RAG, and optional digital-human video capabilities to help you build AI agents centered on voice interaction.

### Key Features
| Feature | Description |
|---------|-------------|
| **Realtime Voice Agent** | Low-latency conversations with microphone input and interruptions |
| **Audio/Video over WebRTC** | P2P or LiveKit SFU mode for different deployment scenarios |
| **PersonaAgent + SubAgent** | Multi-agent architecture for fluid conversations and background tasks |
| **Character Memory & RAG** | Persistent conversation history and knowledge base integration |
| **Digital Human Video** | Optional GPU-powered avatar with lip-sync and facial animation |
| **Plugin-Based Stack** | Replaceable modules for Brain, voice, hearing, tools, memory, and face |

### How to Install
Follow Quick Start in README.md:
- Node 18+
- Go 1.25
- Conda
- Python 3.10+
- FFmpeg

### Supported Languages
- English
- Chinese (简体中文)
- Japanese (日本語)
- Korean (한국어)

### Requirements
| Component | Version |
|-----------|---------|
| Node.js | 18+ |
| Go | 1.25 |
| Python | 3.10+ |
| FFmpeg | Required |

### License
GPL v3 License

### Help Resources
- [Documentation](https://github.com/dsd2077/CyberVerse/tree/main/docs)
- [Issues](https://github.com/dsd2077/CyberVerse/issues)
- [Demo Videos](https://github.com/dsd2077/CyberVerse#demo)
