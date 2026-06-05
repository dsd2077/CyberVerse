# 변경 이력

## 2026-05-29~2026-06-05

Features:
- AutoDL 플랫폼 이미지 추가(심사 중)
- LiveAct에 FP4 GEMM 가속 추가, B 시리즈 GPU에서 추론 연산을 더 압축 가능 ([2cace72](https://github.com/dsd2077/CyberVerse/commit/2cace72))
- MuseTalk 모델 초기 구현 도입. 학습, 전처리, 오프라인/실시간 추론 스크립트 및 Gradio demo 포함 ([c826645](https://github.com/dsd2077/CyberVerse/commit/c826645))

Changes:
- PersonaAgent에서 `wait_for_more_input` 도구 제거. 표현이 불명확할 때 도구 호출 대신 자연스럽게 추가 질문 ([aa691ee](https://github.com/dsd2077/CyberVerse/commit/aa691ee))
- 예시 설정 `idle_strategy` 기본값을 `silent_inference`로 변경, 연속 추론 모드와 일치 ([7644430](https://github.com/dsd2077/CyberVerse/commit/7644430))

Bugs:
- LiveAct 멀티 GPU 분산 제어 broadcast가 GPU에서 busy-wait 하던 문제 수정. 제어 채널을 독립 `gloo` 프로세스 그룹으로 변경 ([PR #19](https://github.com/dsd2077/CyberVerse/pull/19), [54635a3](https://github.com/dsd2077/CyberVerse/commit/54635a3))

Others:
- README에 FAQ(QA) 섹션 추가. RTP 지표로 디지털 휴먼 끊김과 음영 지연 진단 가능 ([f7a1a4b](https://github.com/dsd2077/CyberVerse/commit/f7a1a4b))
- LiveAct FP4 GEMM 빌드/설치 안내 추가 (LightX2V / CUTLASS) ([ccbb978](https://github.com/dsd2077/CyberVerse/commit/ccbb978))
- 오디오 처리 시스템 의존성 추가: `libopus-dev`, `libopusfile-dev`, `libsoxr-dev`, `pkg-config` ([dba745b](https://github.com/dsd2077/CyberVerse/commit/dba745b))
- FlashAttention 설치 안내를 2.8.1로 업데이트 ([74d58e6](https://github.com/dsd2077/CyberVerse/commit/74d58e6))

## 2026-05-22~2026-05-29

이번 주에는 주로 실시간 디지털 휴먼 파이프라인의 Direct WebRTC 안정성, AV 동기화, idle 상태 디지털 휴먼 전략, 음성 컨텍스트 복구를 다뤘습니다.

### fix: 음성/영상 싱크 불일치 버그 수정

이전 버전에는 대화가 계속될수록 음성과 영상의 싱크가 점차 어긋나는 문제가 있었습니다. 이 문제를 고치기 위해 이번 주에는 `WebRTC` 프로토콜을 자세히 파고들었고, 원인을 정리했습니다. 원인은 꽤 복잡하지만, 크게 보면 다음 세 가지였습니다.

1. 송신 측 pacing 이 불안정함
2. turn 사이의 idle gap 처리가 잘못됨
3. 브라우저 receiver 상태가 turn 을 넘어 잔류함

이 세 가지 문제를 이해하려면, 먼저 CyberVerse 가 오디오와 비디오를 프론트엔드로 어떻게 보내는지 봐야 합니다. `cached_video` 모드에서 디지털 휴먼은 한 turn 씩 말합니다. 디지털 휴먼이 말하지 않을 때는 미리 생성해 둔 영상을 재생합니다. 그래서 이 모드를 `cached_video` 모드라고 부릅니다. 전체 비디오 스트림은 대략 다음과 같습니다.

```text
speaking -> idle -> speaking -> idle -> speaking
```

중간의 정지 시간은 고정되어 있지 않습니다. 몇 초일 수도 있고, 수십 초일 수도 있습니다. 문제는 여기에 있었습니다. 저는 동일한 Direct WebRTC media path 를 사용해 조각난 실시간 오디오와 비디오를 전달하고 있었지만, 이 media path 의 송신 리듬, RTP 타임라인, 브라우저 receiver 상태는 이전까지 "연속 라이브 스트림"처럼 처리되고 있었습니다.

첫 번째 원인은 송신 측 pacing 이 불안정했다는 점입니다. 이전에는 서버가 영상을 publish 할 때 "비디오 프레임 하나를 쓰고, frameDur 만큼 sleep" 하는 상대적인 리듬을 사용했습니다. 이 방식은 단순해 보이지만, 패킷 쓰기, 스케줄링, sleep 마다 오차가 생기고, 그 오차는 하나의 session 안에서 프레임마다 누적됩니다. 더 까다로운 점은 Opus 오디오 프레임이 20 ms 리듬으로 부드럽게 전송되지 않고, 비디오 프레임 단위로 묶여 함께 전송되었다는 것입니다. 실제로는 오디오도 비디오 프레임을 따라 묶음으로 전송되었습니다. 브라우저가 받는 arrival pattern 이 고르지 않아졌고, 비디오 jitter buffer 가 점차 커지면서 결국 화면이 음성보다 점점 뒤처지는 증상으로 나타났습니다.

두 번째 원인은 speaking turn 사이의 RTP idle gap 처리가 올바르지 않았다는 점입니다. 두 speaking turn 사이의 정지는 2초일 수도 있고, 2분일 수도 있습니다. 하지만 이전 RTP timestamp gap correction 은 최대 2초까지만 건너뛰었습니다. 브라우저 입장에서는 이것이 매우 비정상적인 신호가 됩니다. 실제 네트워크 도착 시간은 오래 지났는데, RTP media clock 은 2초만 전진했기 때문입니다. 브라우저의 video jitter buffer 는 이후 패킷을 심각하게 지연된 스트림으로 처리했고, `video_jb` 가 오염된 뒤에는 이후 재생에도 계속 영향을 주었습니다.

세 번째 원인은 브라우저 receiver 상태가 speaking turn 을 넘어 잔류한다는 점입니다. 프론트엔드가 WebRTC 화면에서 idle 영상으로 전환하더라도, 이는 시각 레이어의 전환일 뿐입니다. `RTCPeerConnection`, receiver, jitter buffer 는 리셋되지 않습니다. 즉, 어느 speaking turn 에서 비디오 jitter buffer 가 높아지면, 이후 turn 에서는 생성 측이 이미 정상으로 돌아왔더라도 그 비정상 상태를 계속 물려받을 수 있습니다. 그래서 로그에서 특정 turn 부터 명확한 drift 가 시작되고, 이후 여러 turn 에서도 `JBDelta(window)` 가 계속 높게 유지되는 모습을 볼 수 있었습니다.

처음에는 vibe coding 방식으로 이 문제를 해결하려고 했지만, 하루 종일 시도해도 해결하지 못했습니다. 음성/영상 싱크 불일치는 "주관적인" 판단이고, AI 는 이를 직접 인지할 수 없습니다. AI 가 아무리 논리적으로 추론해도 "싱크가 맞지 않는다"는 상태 자체를 감지할 수는 없습니다. 그래서 전체 파이프라인에 모니터링을 추가하고, 음성/영상 동기화 차이를 정량화하는 방법을 조사했습니다. [Commit ad470a7](https://github.com/dsd2077/CyberVerse/commit/ad470a7652ce39345941f304aaa091fe35519695). 실제 피드백 지표가 생기자, AI 에게도 튜닝할 방향이 생겼습니다.

원인이 명확해지자 해결은 간단해졌습니다. [Commit 0ad5730](https://github.com/dsd2077/CyberVerse/commit/0ad5730641d36e2de1ed575d71c3ca3cb510c997)

### feat: `silent_inference` 추론 모드 추가

CyberVerse 는 이전까지 계속 `cached_video` 모드를 사용했습니다. 이 모드에는 몇 가지 장점이 있습니다. 연산 자원을 절약할 수 있고, 디지털 휴먼 연출의 여지가 더 크며, 사용자가 더 강력한 영상 생성 모델로 보기 좋은 대기 영상을 많이 만들 수 있습니다. 하지만 단점도 있습니다. 인물 전체의 연속성이 충분하지 않습니다. "말하지 않음 -> 말함 -> 말하지 않음" 과정에서 화면이 끊겨 보입니다. 이 문제를 해결하기 위해 `silent_inference` 모드를 추가했습니다. 이 모드에서는 모델이 계속 추론을 수행합니다. 말할 때는 음성 오디오로 추론하고, 말하지 않을 때는 무음 오디오로 추론합니다. "말하지 않음 -> 말함 -> 말하지 않음" 과정이 매우 자연스럽게 이어지고, 프레임 점프가 발생하지 않습니다. 물론 이 모드의 단점은 말하지 않을 때의 화면이 계속 반복된다는 점입니다. 또한 이 화면은 제어할 수 없습니다. 무음으로 구동하든 랜덤 노이즈로 구동하든 큰 차이는 없습니다. [Commit 57512f7](https://github.com/dsd2077/CyberVerse/commit/57512f77b21f07ff2a8b7b3af184159eaa9883bc)

### fix: 음성 파이프라인 및 컨텍스트 복구

타임아웃 후 재연결되면 컨텍스트가 사라지는 문제를 수정했습니다. Qwen Omni 모델에는 5분 연결 타임아웃이 설정되어 있습니다. 타임아웃 이후에는 파일을 다시 보내 wake up 할 수 있습니다. 하지만 재연결 후 이전 turn 의 컨텍스트가 이어지지 않는 bug 가 있었습니다. [Commit 0e1c171](https://github.com/dsd2077/CyberVerse/commit/0e1c171c41fc28543f199df7629a14ac5b1078de)

### 엔지니어링 규칙 문서 조정

- AGENTS.md / Claude.md 에 Truth-First Reasoning Rules 를 추가하여 AI 가 자기 자신과 계속 논쟁하는 상황을 줄이도록 했습니다. 자세한 내용: [X](https://x.com/dsd2077/status/2059101010317189519?s=20)
- andrej-karpathy-skills 의 네 가지 원칙을 제거했습니다. 온라인 피드백에 따르면 이 프롬프트는 크게 유용하지 않은 듯하고, 이미 공식 동작에 흡수되었을 가능성이 있습니다.

[Commit e6788a8](https://github.com/dsd2077/CyberVerse/commit/e6788a843ed806d78a3268fe9fee1854855a8639)
