# 更新日志

### 2026-05-29~2026-06-05

Features:
- 新增AutoDL平台镜像(审核中)
- LiveAct 新增 FP4 GEMM 加速，支持 B 系列 GPU 进一步压缩推理算力（[2cace72](https://github.com/dsd2077/CyberVerse/commit/2cace72)）
- 引入 MuseTalk 模型初始实现，含训练、预处理、离线/实时推理脚本与 Gradio demo（[c826645](https://github.com/dsd2077/CyberVerse/commit/c826645)）

Changes:
- PersonaAgent 移除 `wait_for_more_input` 工具，表达不清时直接自然追问，不再走工具调用（[aa691ee](https://github.com/dsd2077/CyberVerse/commit/aa691ee)）
- 示例配置 `idle_strategy` 默认改为 `silent_inference`，与连贯推理模式对齐（[7644430](https://github.com/dsd2077/CyberVerse/commit/7644430)）

Bugs:
- 修复 LiveAct 多卡分布式控制 broadcast 在 GPU 上 busy-wait 的问题，控制通道改走独立 `gloo` 进程组（[PR #19](https://github.com/dsd2077/CyberVerse/pull/19)，[54635a3](https://github.com/dsd2077/CyberVerse/commit/54635a3)）

Others:
- README 新增常见问题自检（QA）章节，引入 RTP 指标排查数字人卡顿与音画落后（[f7a1a4b](https://github.com/dsd2077/CyberVerse/commit/f7a1a4b)）
- 补充 LiveAct FP4 GEMM 编译安装说明（LightX2V / CUTLASS）（[ccbb978](https://github.com/dsd2077/CyberVerse/commit/ccbb978)）
- 补充音频处理系统依赖：`libopus-dev`、`libopusfile-dev`、`libsoxr-dev`、`pkg-config`（[dba745b](https://github.com/dsd2077/CyberVerse/commit/dba745b)）
- FlashAttention 安装说明更新至 2.8.1（[74d58e6](https://github.com/dsd2077/CyberVerse/commit/74d58e6)）

### 2026-05-22~2026-05-29
本周主要围绕实时数字人链路的 Direct WebRTC 稳定性、AV 同步、空闲态数字人策略和语音上下文恢复展开。

#### fix：修复音画不同步bug

之前版本存在一个问题，随着对话的持续进行，会逐渐出现音画不同步的情况。为了修复这个问题，这周认真啃了一下 `WebRTC` 协议。把这个问题搞清楚了。问题的原因比较复杂，总的来说主要是三个原因：

1. 发送端 pacing 不稳定
2. turn 间 idle gap 处理错误
3. 浏览器 receiver 状态跨 turn 残留

要搞懂这三个问题，必须先看一下 CyberVerse 是怎么发送音视频到前端的。在 `cached_video` 模式下，数字人说话是一轮一轮的；当数字人没有说话时，会播放提前生成的视频（所以叫 cached_video 模式）。整个视频流大概是这样：

```text
speaking -> idle -> speaking -> idle -> speaking
```

中间的停顿时长是不确定的，可能是几秒，也可能是几十秒。问题就出在这里：我用了同一个 Direct WebRTC media path 去承载一段一段的实时音视频，但这条 media path 的发送节奏、RTP 时间线和浏览器 receiver 状态，之前都更像是在按「连续直播流」来处理。

第一个原因是发送端 pacing 不稳定。之前服务端发布视频时使用的是「写一帧视频，然后 sleep 一个 frameDur」的相对节奏。这个写法看起来简单，但每次写包、调度、sleep 都会产生误差，而且误差会在一个 session 内逐帧累积。更麻烦的是，音频 Opus 帧之前是按视频帧分组一起写出去的，没有按 20ms 的节奏平滑发送，实际效果就是跟着视频帧一组一组发送出去。浏览器收到的 arrival pattern 不均匀，视频 jitter buffer 会被逐渐拉大，最后表现为画面越来越落后于声音。

第二个原因是 speaking turn 之间的 RTP idle gap 处理不正确。两个 speaking turn 之间可能停顿 2S，也可能停顿两分钟，但之前 RTP timestamp gap correction 最多只跳过 2 秒。对浏览器来说，这就变成了一个很异常的信号：真实网络到达时间已经过去很久，但 RTP media clock 只前进了 2 秒。浏览器的视频 jitter buffer 会把后续包当成严重异常的延迟流来处理，`video_jb` 被污染后，就会继续影响后面的播放。

第三个原因是浏览器 receiver 状态会跨 speaking turn 残留。前端从 WebRTC 画面切到 idle 视频，只是视觉层的切换，并不会重置 `RTCPeerConnection`、receiver 或 jitter buffer。也就是说，一旦某一轮 speaking turn 把视频 jitter buffer 拉高，后面几轮即使生成端已经恢复正常，也可能继续继承这个异常状态。这就是为什么日志里会看到某一轮开始明显漂移，后面几轮 `JBDelta(window)` 仍然保持很高。

一开始我尝试以vibe coding的方式来解决这个问题，但是vibe了整整一天都没搞定。因为音画不同是一件“主观”的判断，AI感知不到这个事情。所以无论AI无论怎么进行逻辑推理，它都感知不到“不同步”。于是乎我想到了给整个链路添加监控，并且调研了一些方法来量化“音画同步“的差值。[Commit ad470a7
](https://github.com/dsd2077/CyberVerse/commit/ad470a7652ce39345941f304aaa091fe35519695)。有了真实的反馈指标后，AI就有了调优的方向。

搞清楚原因，解决起来就简单了。[Commit 0ad5730
](https://github.com/dsd2077/CyberVerse/commit/0ad5730641d36e2de1ed575d71c3ca3cb510c997)

#### feat：新增 `silent_inference` 推理模式
CyberVerse 此前一直都是采用的 `cached_video` 模式，这种模式有一些好处：节省算力、对数字人的编排空间更大，用户可以使用更强大的视频生成模型来生成更好看的待机视频，而且可以生成很多很多。但这个模式也有一个不好的地方，就是整个人物不够连贯，不说话——>说话——>不说话，这个过程画面是有割裂的。所以我新增了 `silent_inference` 模式来解决这个问题。这个模型会一直进行推理，说话的时候用语音音频推理，不说话的时候用静音音频推理。不说话——>说话——>不说话，这个过程就会变得非常连贯。不会出现任何跳帧的情况。当然这个模式的缺点就是不说话时的画面会一直重复，而且这个画面是没有办法控制的，不管使用静音驱动还是随机噪音驱动，都不会有太大区别。[Commit 57512f7](https://github.com/dsd2077/CyberVerse/commit/57512f77b21f07ff2a8b7b3af184159eaa9883bc)

#### fix：语音链路与上下文恢复
修复超时重连后上下文丢失的问题。qwen omni模型有设置5分钟的超时连接，超过过后可以，再次发送文件进行唤醒。但是重连之后有个bug：没有上一轮回话的上下文带上。[Commit 0e1c171](https://github.com/dsd2077/CyberVerse/commit/0e1c171c41fc28543f199df7629a14ac5b1078de)

#### 工程规范文档调整
- 补充 Truth-First Reasoning Rules 到 AGENTS.md / Claude.md，用于抑制AI左右脑互搏的情况。详见：[X](https://x.com/dsd2077/status/2059101010317189519?s=20)
- 移除了andrej-karpathy-skills 四个原则，听网友说这个提示词没有什么卵用，可能已经被官方吸纳了。
[Commit e6788a8](https://github.com/dsd2077/CyberVerse/commit/e6788a843ed806d78a3268fe9fee1854855a8639)