# Changelog

## 2026-05-29~2026-06-05

Features:
- Added AutoDL platform image (under review)
- LiveAct adds FP4 GEMM acceleration, enabling further inference compute compression on B-series GPUs ([2cace72](https://github.com/dsd2077/CyberVerse/commit/2cace72))
- Initial MuseTalk model implementation, including training, preprocessing, offline/real-time inference scripts, and a Gradio demo ([c826645](https://github.com/dsd2077/CyberVerse/commit/c826645))

Changes:
- PersonaAgent removes the `wait_for_more_input` tool; when input is unclear, it asks follow-up questions naturally instead of invoking a tool ([aa691ee](https://github.com/dsd2077/CyberVerse/commit/aa691ee))
- Example config `idle_strategy` default changed to `silent_inference`, aligned with continuous inference mode ([7644430](https://github.com/dsd2077/CyberVerse/commit/7644430))

Bugs:
- Fixed LiveAct multi-GPU distributed control broadcast busy-wait on GPU; control channel now uses a separate `gloo` process group ([PR #19](https://github.com/dsd2077/CyberVerse/pull/19), [54635a3](https://github.com/dsd2077/CyberVerse/commit/54635a3))

Others:
- README adds an FAQ (QA) section with RTP metrics for diagnosing digital human stuttering and A/V lag ([f7a1a4b](https://github.com/dsd2077/CyberVerse/commit/f7a1a4b))
- Added LiveAct FP4 GEMM build/install instructions (LightX2V / CUTLASS) ([ccbb978](https://github.com/dsd2077/CyberVerse/commit/ccbb978))
- Added audio processing system dependencies: `libopus-dev`, `libopusfile-dev`, `libsoxr-dev`, `pkg-config` ([dba745b](https://github.com/dsd2077/CyberVerse/commit/dba745b))
- FlashAttention install instructions updated to 2.8.1 ([74d58e6](https://github.com/dsd2077/CyberVerse/commit/74d58e6))

## 2026-05-22~2026-05-29

This week mainly focused on Direct WebRTC stability, AV sync, idle-state digital human strategy, and voice context recovery in the real-time digital human pipeline.

### fix: Fixed audio-video desynchronization

Previous versions had an issue where audio and video would gradually drift out of sync as a conversation continued. To fix this, I spent time this week digging into the `WebRTC` protocol and finally got a clear handle on the issue. The root cause is fairly complex, but at a high level it came down to three main factors:

1. Unstable sender-side pacing
2. Incorrect idle gap handling between turns
3. Browser receiver state persisting across turns

To understand these three issues, it is necessary to first look at how CyberVerse sends audio and video to the frontend. In `cached_video` mode, the digital human speaks turn by turn. When the digital human is not speaking, a pre-generated video is played instead, which is why this is called `cached_video` mode. The overall video stream looks roughly like this:

```text
speaking -> idle -> speaking -> idle -> speaking
```

The pauses in between are not fixed. They may last a few seconds, or they may last dozens of seconds. The problem was there: I was using the same Direct WebRTC media path to carry segmented real-time audio and video, but the sender pacing, RTP timeline, and browser receiver state for that media path were previously being handled more like a continuous livestream.

The first cause was unstable sender-side pacing. Previously, when the server published video, it used a relative rhythm of "write one video frame, then sleep for one frame duration." This looks simple, but every packet write, scheduling step, and sleep introduces error, and those errors accumulate frame by frame within a session. More importantly, Opus audio frames had previously been grouped with video frames when sent, instead of being sent smoothly at a 20 ms cadence. In practice, audio was being sent out in groups alongside video frames. The browser received an uneven arrival pattern, the video jitter buffer gradually grew, and the final symptom was video falling increasingly behind audio.

The second cause was incorrect RTP idle gap handling between speaking turns. The pause between two speaking turns might be 2 seconds, or it might be 2 minutes, but the previous RTP timestamp gap correction skipped at most 2 seconds. To the browser, this became a very abnormal signal: a long time had passed in real network arrival time, but the RTP media clock had advanced by only 2 seconds. The browser's video jitter buffer treated subsequent packets as a severely delayed stream, and once `video_jb` was polluted, it continued to affect later playback.

The third cause was that browser receiver state persisted across speaking turns. When the frontend switched from the WebRTC picture to the idle video, it was only a visual-layer switch. It did not reset the `RTCPeerConnection`, receiver, or jitter buffer. In other words, once one speaking turn had pushed the video jitter buffer high, later turns could continue inheriting that abnormal state even if the generation side had already recovered. This is why the logs showed obvious drift starting in one turn, while `JBDelta(window)` stayed high across later turns.

At first I tried to solve this with vibe coding, but after a full day of that it still was not fixed. Audio-video desynchronization is a "subjective" judgment, and AI cannot perceive it directly. No matter how the AI reasoned through the logic, it could not actually perceive the "out of sync" part. So I added monitoring across the whole pipeline and researched ways to quantify the audio-video sync delta. [Commit ad470a7](https://github.com/dsd2077/CyberVerse/commit/ad470a7652ce39345941f304aaa091fe35519695). Once there were real feedback metrics, the AI had a direction for tuning.

After the cause was clear, the fix became straightforward. [Commit 0ad5730](https://github.com/dsd2077/CyberVerse/commit/0ad5730641d36e2de1ed575d71c3ca3cb510c997)

### feat: Added `silent_inference` inference mode

CyberVerse had previously always used `cached_video` mode. This mode has some advantages: it saves compute, gives more room for digital-human orchestration, and lets users use stronger video generation models to create better-looking standby videos, even many of them. But it also has a downside: the character is not continuous enough. The transition from not speaking, to speaking, and back to not speaking creates a visible visual break. I added `silent_inference` mode to solve this problem. In this mode, the model keeps running inference continuously: it uses speech audio during speaking, and silent audio during idle periods. The transition from not speaking, to speaking, and back to not speaking becomes very continuous, with no frame jumps. Of course, the downside is that the idle picture keeps repeating, and that picture cannot really be controlled. Whether driven by silence or random noise, the result does not differ very much. [Commit 57512f7](https://github.com/dsd2077/CyberVerse/commit/57512f77b21f07ff2a8b7b3af184159eaa9883bc)

### fix: Voice pipeline and context recovery

Fixed the issue where context was lost after timeout reconnection. The Qwen Omni model has a 5-minute connection timeout. After the timeout, files can be sent again to wake it up. But there was a bug after reconnection: the context from the previous turn was not carried forward. [Commit 0e1c171](https://github.com/dsd2077/CyberVerse/commit/0e1c171c41fc28543f199df7629a14ac5b1078de)

### Engineering guidelines document updates

- Added Truth-First Reasoning Rules to AGENTS.md / Claude.md to reduce cases where AI argues with itself. See: [X](https://x.com/dsd2077/status/2059101010317189519?s=20)
- Removed the four andrej-karpathy-skills principles. According to online feedback, that prompt did not seem very useful and may already have been absorbed by the official behavior.

[Commit e6788a8](https://github.com/dsd2077/CyberVerse/commit/e6788a843ed806d78a3268fe9fee1854855a8639)
