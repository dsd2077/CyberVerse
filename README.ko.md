# CyberVerse

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | **한국어**

### 사진 한 장으로 살아 움직이는 디지털 휴먼.

> 당신을 진짜로 보고, 듣고, 실시간으로 말을 건네는 나만의 J.A.R.V.I.S. 같은 AI를 꿈꿔본 적이 있나요?
> 그리운 사람을 다시 보고, 그 목소리를 듣고, 미소 짓는 모습을 볼 수 있다면 어떨까요?
> 혹은 늘 현실로 불러오고 싶었던 캐릭터가 있을지도 모릅니다.
>
> **사진 한 장이면 됩니다. CyberVerse가 그 존재를 살아 움직이게 합니다.**

CyberVerse는 실시간 영상 통화가 가능한 오픈소스 **디지털 휴먼 에이전트 플랫폼**입니다. 영상 통화하듯 얼굴을 보며 대화할 수 있는 AI 에이전트를 만들 수 있습니다.

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)

## 주요 기능

### 실시간 영상 통화

미리 녹화된 영상이 아닙니다. 턴 기반 대화도 아닙니다. 디지털 휴먼과 **시간 제한 없이** 저지연으로 실시간 영상 통화를 할 수 있으며, 첫 프레임은 **약 1.5초** 안에 표시됩니다. WebRTC 기반으로 P2P 스트리밍과 내장 TURN / NAT 트래버설을 제공합니다.

### 단순한 아바타가 아니라 Agent

각 디지털 휴먼은 단순히 대화할 수 있는 아바타가 아닙니다. 실제로 일을 해내는 AI입니다.

### 사진 한 장으로 생성

사진 한 장만 업로드하면 디지털 휴먼을 만들 수 있습니다. 업계 최신 디지털 휴먼 모델이 실시간 얼굴 애니메이션, 자연스러운 립싱크, 대기 중 호흡감을 제공하며, 3D 모델링이나 모션 캡처는 필요하지 않습니다.

### Agent를 자유롭게 조립

두뇌, 얼굴, 목소리, 귀까지 모든 구성 요소를 교체 가능한 플러그인으로 다룰 수 있습니다. YAML 설정을 통해 LLM, TTS 엔진, ASR 모델, 아바타 백엔드를 자유롭게 조합할 수 있습니다. 기본으로 GPT-4o, Doubao, OpenAI TTS, Whisper, FlashHead를 지원합니다.

## 데모

<div>
  <a href="https://youtu.be/Lk88sew2x4o"><img src="./docs/demo/爱丽丝.mov.png" alt="Alice Demo" width="48%"></a>
  <a href="https://youtu.be/8jdQ3ThcwgA"><img src="./docs/demo/丽娜.mov.png" alt="Lina Demo" width="48%"></a>
  <br />
  <a href="https://youtu.be/WjEHUYZx5Gs"><img src="./docs/demo/小龙女.mov.png" alt="Xiaolongnv Demo" width="48%"></a>
</div>

## 하드웨어 요구 사항

실시간 영상 대화에는 GPU 가속이 필요합니다. 아래는 FlashHead 및 LiveAct 아바타 모델의 벤치마크입니다.

| 모델 | 품질 | GPU | 수량 | 해상도 | FPS | 실시간 가능? |
|-------|---------|-----|-------|------------|-----|------------|
| FlashHead 1.3B | Pro | RTX 5090 | 2 | 512×512 | 25+ | ✅ 예 |
| FlashHead 1.3B | Pro | RTX 4090 | 1 | 512×512 | ~10.8 | ❌ 아니오 |
| FlashHead 1.3B | Lite | RTX 4090 | 1 | 512×512 | 25+ | ✅ 예 |
| LiveAct 18B | — | RTX PRO 6000 | 2 | 320×480 | 20 | ✅ 예 |

> **Pro**는 화질을, **Lite**는 속도를 우선합니다. 표의 구성은 일반적인 **화질–연산** 균형 예시입니다. 여유 연산이 있으면 화질을 더 올릴 수 있고, 부족하면 해상도·**Pro**/**Lite** 선택 등 화질 설정을 낮춰 실시간 동작을 유지하세요.

## 빠른 시작

### 사전 준비

- Python 3.10+
- Node 18+
- Go 1.22+
- PyTorch 2.8（CUDA 12.8）
- CUDA 12.8+ 지원 GPU
- FFmpeg(`libvpx` 포함, 영상 인코딩용)

### 1단계: 클론

```bash
git clone https://github.com/anthropics/CyberVerse.git
cd CyberVerse
```

### 2단계: Python 환경 만들기

```bash
conda create -n cyberverse python=3.10
conda activate cyberverse
```

### 3단계: 환경 변수 설정

```bash
cp infra/.env.example .env
```

`.env`를 편집해 API 키를 입력합니다.

```
DOUBAO_ACCESS_TOKEN=your_doubao_access_token   # ByteDance Doubao 음성 LLM
DOUBAO_APP_ID=your_doubao_app_id
```

Doubao Voice: [Volcengine 빠른 시작](https://www.volcengine.com/docs/6561/2119699?lang=zh)에서 **App ID** / **API Key**를 확인해 `DOUBAO_APP_ID` / `DOUBAO_ACCESS_TOKEN`에 넣습니다.

스택이 실행된 뒤에는 `.env`만 수정할 필요 없이, Web UI의 **`/settings`** 에서 이 값들과 다른 API 키 / 서비스 엔드포인트도 변경할 수 있습니다.

### 4단계: 모델 가중치 다운로드

CyberVerse는 현재 **FlashHead**와 **LiveAct** 두 가지 모델을 지원합니다. 필요한 것만 다운로드하면 됩니다. 향후 더 많은 백엔드를 지원할 예정입니다.

```bash
pip install "huggingface_hub[cli]"
```

#### FlashHead（SoulX-FlashHead）

| 모델 구성 요소 | 설명 | 링크 |
| :--- | :--- | :--- |
| `SoulX-FlashHead-1_3B` | 1.3B FlashHead 가중치 | [Hugging Face](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| `wav2vec2-base-960h` | 오디오 특징 추출기 | [Hugging Face](https://huggingface.co/facebook/wav2vec2-base-960h) |

```bash
# 중국 본토에서는 먼저 미러를 사용할 수 있습니다:
# export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B \
  --local-dir ./checkpoints/SoulX-FlashHead-1_3B

huggingface-cli download facebook/wav2vec2-base-960h \
  --local-dir ./checkpoints/wav2vec2-base-960h
```

#### LiveAct（SoulX-LiveAct）

| 모델명 | 다운로드 |
|-----------|----------|
| SoulX-LiveAct | [Hugging Face](https://huggingface.co/Soul-AILab/LiveAct), [ModelScope](https://modelscope.cn/models/Soul-AILab/LiveAct) |
| chinese-wav2vec2-base | [Hugging Face](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) |

```bash
huggingface-cli download Soul-AILab/LiveAct \
  --local-dir ./checkpoints/LiveAct

huggingface-cli download TencentGameMate/chinese-wav2vec2-base \
  --local-dir ./checkpoints/chinese-wav2vec2-base
```


### 5단계: 설정 업데이트

`cyberverse_config.yaml`을 열고 모델 경로를 로컬 checkpoint 경로에 맞게 수정합니다.

```yaml
inference:
  avatar:
    runtime:
      cuda_visible_devices: 0      # 공용 GPU ID. 멀티 GPU라면 0,1 등으로 설정
      world_size: 1                # 공용 GPU 수. 듀얼 GPU면 2로 설정
    flash_head:
      checkpoint_dir: "./checkpoints/SoulX-FlashHead-1_3B"  # ← 로컬 경로
      wav2vec_dir: "./checkpoints/wav2vec2-base-960h"        # ← 로컬 경로
      model_type: "lite"           # 더 높은 품질이 필요하면 "pro"(GPU 더 필요)
```

이 단계의 경로 수정은 잠시 건너뛰고, 나중에 Web UI에서 조정해도 됩니다.

### 6단계: SageAttention 및 FlashAttention 설치(선택 사항)

```bash
# SageAttention
pip install sageattention==2.2.0 --no-build-isolation
```

```bash
# FlashAttention(선택 사항)
pip install ninja
pip install flash_attn==2.8.0.post2 --no-build-isolation
```

> 컴파일이 오래 걸린다면 [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.0.post2)에서 미리 빌드된 wheel을 받아 `pip install <wheel>.whl`로 설치할 수 있습니다.



### 7단계: 프로젝트 의존성 설치

```bash
make setup
```

이 단계에서는 기본 editable package(`.[dev,inference]`)를 설치하고, gRPC stubs를 생성하며, 프런트엔드 의존성도 함께 설치합니다. 추가 Python 패키지가 필요하다면 전부 한 번에 설치할 수도 있고(용량 큼), [`pyproject.toml`](pyproject.toml)의 `[project.optional-dependencies]` 아래 extras 중 필요한 것만 골라 설치할 수도 있습니다.

```bash
# 모든 optional 그룹 한 번에 설치
pip install -e ".[all]"

# 또는 필요한 것만 선택. 예:
pip install -e ".[voice_llm,flash_head]"
pip install -e ".[live_act]"
```

### 8단계: 서비스 시작(터미널 3개)

**터미널 1** — Python 추론 서버:

```bash
conda activate cyberverse
make inference
```

`make inference`는 `cyberverse_config.yaml`의 `inference.avatar.default`를 읽고, 현재 추론 프로세스에서는 그 하나의 아바타 모델만 초기화합니다. 시작 로그에는 활성화된 아바타 모델 이름이 출력됩니다.

다음 로그가 나올 때까지 기다립니다.

- `Active avatar model initialized: <model_name>`
- `CyberVerse Inference Server started on port 50051`

**터미널 2** — Go API 서버:

```bash
make server
```

**터미널 3** — 프런트엔드:

```bash
make frontend
```

### 9단계: 확인

```bash
# API 상태 확인
curl -s http://localhost:8080/api/v1/health
```

브라우저에서 http://localhost:5173 를 열면 바로 사용할 수 있습니다.

## 로드맵

### **디지털 휴먼 제작 플랫폼**
캐릭터와 추론 설정을 구성하고, 실시간 디지털 휴먼 세션을 시작합니다.

- [x] 여러 참조 이미지, 활성 이미지, 고정/랜덤 표시 모드, 선택적 얼굴 크롭, 태그, 음성 필드, 성격, 환영 메시지, 시스템 프롬프트를 포함한 캐릭터 CRUD
- [x] 참조 이미지를 기반으로 구성 가능한 아바타 플러그인(FlashHead, LiveAct 등)을 통해 실시간 아바타 영상을 생성
- [x] WebRTC 기반 실시간 음성/영상. **직접** P2P(내장 TURN) 또는 **LiveKit** SFU(`pipeline.streaming_mode`) 지원. 세션은 **voice_llm** 및 **standard**(text → LLM → TTS → avatar) 파이프라인 지원
- [x] YAML로 교체 가능한 추론 스택(avatar, voice LLM, LLM, TTS, ASR), 아바타 파라미터 런치 UI, 기본 아바타 모델을 바꾸려면 일반적으로 추론 서비스 재시작 필요
- [x] API 키와 서비스 엔드포인트를 설정할 수 있는 Web **Settings**(`/settings`), 연결 테스트 및 영속화 지원
- [x] 캐릭터별 대화 기록을 디스크에 저장하고 REST API로 제공
- [x] 선택적 세션 영상 녹화(`cyberverse_config.yaml`의 `recording`)
- [ ] 캐릭터 기반 응답을 위한 지식 및 문서 가져오기(RAG)

### 2. **디지털 휴먼 에이전트**
디지털 휴먼을 메모리, 도구, 작업 수행 능력을 가진 에이전트로 발전시킵니다.

- [ ] 세션 간 장기 메모리 추가
- [ ] 도구 사용 및 function calling 추가
- [ ] 워크플로 실행 및 작업 완료 기능 추가

### 3. **에이전트 네트워크**
여러 에이전트를 연결해 서로 소통하고 협업하며 네트워크를 형성할 수 있게 합니다.

- [ ] agent-to-agent 통신 활성화
- [ ] 멀티 에이전트 협업 및 위임 활성화
- [ ] 에이전트 간 공유 메모리 및 공유 지식 활성화
- [ ] 연결된 에이전트의 개방형 네트워크 구축

## 라이선스

GNU General Public License v3.0. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

## 감사의 말

- [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) — Soul AI Lab의 아바타 모델

- [SoulX-LiveAct](https://github.com/Soul-AILab/SoulX-LiveAct) - Soul AI Lab의 아바타 모델
- [Pion](https://github.com/pion/webrtc) — Go WebRTC 구현체
