import { ref, watch } from 'vue'
import { Room, RoomEvent, Track, type RemoteTrack } from 'livekit-client'

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error'

// ─── 诊断日志工具 ──────────────────────────────────────────────────────────
const T0 = performance.now()
function ts() {
  return `+${((performance.now() - T0) / 1000).toFixed(3)}s`
}

// 合并流状态（音频和视频都进同一个 <video> 元素）
let _videoEl: HTMLVideoElement | null = null
let _videoMST: MediaStreamTrack | null = null
let _audioMST: MediaStreamTrack | null = null

// 帧计数
let videoFrameCount = 0
let videoFirstFrameTime: number | null = null
let videoLastFrameWallMs: number | null = null
let audioPlayWallMs: number | null = null

// RVFC 段内 checkpoint
let rvfcCheckpointMedia: number | null = null
let rvfcCheckpointCurrentTime: number | null = null
let debugPollTimer: ReturnType<typeof setInterval> | null = null

// ─── 帧间抖动(Jitter)测量 ─────────────────────────────────────────────────
// 滑动窗口记录最近 N 帧的 wall-clock 到达时间，用于计算帧间隔分布
const JITTER_WINDOW = 120 // 保留最近 120 帧（约 4-5 秒 @25fps）
let frameArrivalTimes: number[] = []
let framePresentationTimes: number[] = [] // RVFC mediaTime

function resetJitterState() {
  frameArrivalTimes = []
  framePresentationTimes = []
}

function recordFrameArrival(wallMs: number, mediaTime?: number) {
  frameArrivalTimes.push(wallMs)
  if (frameArrivalTimes.length > JITTER_WINDOW) {
    frameArrivalTimes.shift()
  }
  if (mediaTime !== undefined) {
    framePresentationTimes.push(mediaTime)
    if (framePresentationTimes.length > JITTER_WINDOW) {
      framePresentationTimes.shift()
    }
  }
}

/** 计算帧间隔统计：均值、标准差、最大值、P95、卡顿次数 */
function computeJitterStats(): FrameJitterStats {
  const times = frameArrivalTimes
  if (times.length < 2) {
    return { meanIntervalMs: 0, stddevMs: 0, maxIntervalMs: 0, p95IntervalMs: 0, stutterCount: 0, windowSize: 0 }
  }
  const intervals: number[] = []
  for (let i = 1; i < times.length; i++) {
    intervals.push(times[i] - times[i - 1])
  }
  intervals.sort((a, b) => a - b)
  const mean = intervals.reduce((s, v) => s + v, 0) / intervals.length
  const variance = intervals.reduce((s, v) => s + (v - mean) ** 2, 0) / intervals.length
  const stddev = Math.sqrt(variance)
  const p95 = intervals[Math.floor(intervals.length * 0.95)]
  const maxInterval = intervals[intervals.length - 1]
  // 卡顿定义：帧间隔 > 均值 * 2 且 > 60ms（即明显超出预期帧率）
  const stutterThreshold = Math.max(mean * 2, 60)
  const stutterCount = intervals.filter(v => v > stutterThreshold).length
  return { meanIntervalMs: Math.round(mean * 10) / 10, stddevMs: Math.round(stddev * 10) / 10, maxIntervalMs: Math.round(maxInterval), p95IntervalMs: Math.round(p95), stutterCount, windowSize: intervals.length }
}

export type FrameJitterStats = {
  meanIntervalMs: number    // 平均帧间隔 (ms)
  stddevMs: number          // 帧间隔标准差 (ms) — 越大越卡
  maxIntervalMs: number     // 最大帧间隔 (ms)
  p95IntervalMs: number     // P95 帧间隔 (ms)
  stutterCount: number      // 卡顿次数（帧间隔 > 2倍均值 且 > 60ms）
  windowSize: number        // 统计窗口帧数
}

export type WebRTCNetworkStats = {
  roundTripTimeMs: number | null
  jitterMs: number | null
  packetsLost: number
  packetsReceived: number
  lossRate: number           // 丢包率 (0-1)
  bytesReceived: number
  framesDecoded: number
  framesDropped: number
  frameWidth: number
  frameHeight: number
  nackCount: number          // NACK 请求次数
  pliCount: number           // PLI 请求次数
  firCount: number           // FIR 请求次数
  jitterBufferDelayMs: number | null  // 抖动缓冲延迟
  jitterBufferEmittedCount: number
  codec: string
}

export type AVSyncDebugState = {
  sessionId: string
  connectionState: ConnectionState
  audioSubscribedAtMs: number | null
  videoSubscribedAtMs: number | null
  audioUnmutedAtMs: number | null
  videoUnmutedAtMs: number | null
  firstPlayAtMs: number | null
  videoFirstFrameAtMs: number | null
  lastVideoFrameAtMs: number | null
  fps: number
  videoCurrentTime: number
  readyState: number
  playbackRate: number
  decodedFrames: number
  droppedFrames: number
  totalFrames: number
  notes: string[]
  // ─── 新增量化指标 ───
  jitter: FrameJitterStats
  network: WebRTCNetworkStats | null
}

const emptyNetworkStats = (): WebRTCNetworkStats => ({
  roundTripTimeMs: null,
  jitterMs: null,
  packetsLost: 0,
  packetsReceived: 0,
  lossRate: 0,
  bytesReceived: 0,
  framesDecoded: 0,
  framesDropped: 0,
  frameWidth: 0,
  frameHeight: 0,
  nackCount: 0,
  pliCount: 0,
  firCount: 0,
  jitterBufferDelayMs: null,
  jitterBufferEmittedCount: 0,
  codec: '',
})

const emptyDebugState = (): AVSyncDebugState => ({
  sessionId: '',
  connectionState: 'disconnected',
  audioSubscribedAtMs: null,
  videoSubscribedAtMs: null,
  audioUnmutedAtMs: null,
  videoUnmutedAtMs: null,
  firstPlayAtMs: null,
  videoFirstFrameAtMs: null,
  lastVideoFrameAtMs: null,
  fps: 0,
  videoCurrentTime: 0,
  readyState: 0,
  playbackRate: 1,
  decodedFrames: 0,
  droppedFrames: 0,
  totalFrames: 0,
  notes: [],
  jitter: { meanIntervalMs: 0, stddevMs: 0, maxIntervalMs: 0, p95IntervalMs: 0, stutterCount: 0, windowSize: 0 },
  network: null,
})

function resetState() {
  _videoMST = null
  _audioMST = null
  videoFrameCount = 0
  videoFirstFrameTime = null
  videoLastFrameWallMs = null
  audioPlayWallMs = null
  rvfcCheckpointMedia = null
  rvfcCheckpointCurrentTime = null
  resetJitterState()
}

/**
 * 把音频和视频合并进同一个 <video> 元素的 MediaStream。
 * 这是保证音画同步的关键：浏览器对同一 MediaElement 内的 A/V 轨道
 * 会通过 RTCP SR 做原生 lipsync，视频卡顿时音频也会随之等待。
 */
function mergeCombinedStream() {
  const el = _videoEl
  if (!el || !_videoMST) return

  const tracks: MediaStreamTrack[] = [_videoMST]
  if (_audioMST) tracks.push(_audioMST)

  el.srcObject = new MediaStream(tracks)
  console.log(
    `[AVSync][${ts()}] ✅ 合并流已设置 → <video>: video=✓ audio=${!!_audioMST}` +
      ` (浏览器将原生同步音画)`
  )
}

function logIncomingTrack(kind: 'audio' | 'video', track: RemoteTrack) {
  const mediaTrack = track.mediaStreamTrack
  console.log(
    `[AVSync][${ts()}] ${kind === 'audio' ? '🔊 AUDIO' : '🎬 VIDEO'} 收到: sid=${track.sid}` +
      ` readyState=${mediaTrack.readyState} muted=${mediaTrack.muted} enabled=${mediaTrack.enabled}`
  )
}

type VideoFrameMeta = { mediaTime: number; presentedFrames: number }
type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback: (cb: (now: DOMHighResTimeStamp, meta: VideoFrameMeta) => void) => void
}

function attachVideoFrameCallback(elParam: HTMLVideoElement) {
  const el: HTMLVideoElement = elParam
  rvfcCheckpointMedia = null
  rvfcCheckpointCurrentTime = null

  if (!('requestVideoFrameCallback' in elParam)) {
    el.addEventListener('timeupdate', () => {
      if (el.currentTime > 0) {
        videoLastFrameWallMs = Date.now()
      }
      if (videoFirstFrameTime === null && el.currentTime > 0) {
        videoFirstFrameTime = performance.now()
        const delay =
          audioPlayWallMs !== null
            ? ((videoFirstFrameTime - audioPlayWallMs) / 1000).toFixed(3)
            : 'N/A'
        console.log(
          `[AVSync][${ts()}] 🎬 VIDEO 首帧 (timeupdate): currentTime=${el.currentTime.toFixed(3)}s` +
            ` | 比 audio.play 晚 ${delay}s`
        )
      }
    })
    return
  }

  const rvfc = (el as VideoWithRVFC).requestVideoFrameCallback.bind(el)
  const onFrame = (now: DOMHighResTimeStamp, meta: VideoFrameMeta) => {
    videoFrameCount++
    videoLastFrameWallMs = Date.now()
    recordFrameArrival(now, meta.mediaTime)

    if (videoFirstFrameTime === null) {
      videoFirstFrameTime = now
      const delay =
        audioPlayWallMs !== null
          ? ((now - audioPlayWallMs) / 1000).toFixed(3)
          : 'N/A'
      console.log(
        `[AVSync][${ts()}] 🎬 VIDEO 首帧: mediaTime=${meta.mediaTime.toFixed(3)}s` +
          ` presentedFrames=${meta.presentedFrames}` +
          ` | 比 audio.play 晚 ${delay}s`
      )
    }

    if (videoFrameCount > 0 && videoFrameCount % 50 === 0) {
      const ct = el.currentTime
      if (rvfcCheckpointMedia !== null && rvfcCheckpointCurrentTime !== null) {
        const dVm = meta.mediaTime - rvfcCheckpointMedia
        const dCt = ct - rvfcCheckpointCurrentTime
        const seg = dVm - dCt
        console.log(
          `[AVSync][${ts()}] 📊 每50帧: 段Δmedia=${dVm.toFixed(3)}s 段ΔcurrentTime=${dCt.toFixed(3)}s` +
            ` 段偏差(media-ct)=${seg.toFixed(3)}s | mediaTime=${meta.mediaTime.toFixed(3)}s`
        )
      }
      rvfcCheckpointMedia = meta.mediaTime
      rvfcCheckpointCurrentTime = ct
    } else if (rvfcCheckpointMedia === null) {
      rvfcCheckpointMedia = meta.mediaTime
      rvfcCheckpointCurrentTime = el.currentTime
    }

    rvfc(onFrame)
  }
  rvfc(onFrame)
}

// ──────────────────────────────────────────────────────────────────────────────

export function useWebRTC() {
  const videoElement = ref<HTMLVideoElement | null>(null)
  const connectionState = ref<ConnectionState>('disconnected')
  const error = ref<string>('')
  const debugState = ref<AVSyncDebugState>(emptyDebugState())

  let room: InstanceType<typeof Room> | null = null
  const pendingVideoTracks: RemoteTrack[] = []
  let networkStatsTimer: ReturnType<typeof setInterval> | null = null

  /** 通过 LiveKit Room 内部的 RTCPeerConnection 采集 WebRTC 统计 */
  async function pollNetworkStats() {
    if (!room) return
    try {
      // LiveKit Room 的 engine.subscriber.pc 是接收端 PeerConnection
      const pc = (room as any).engine?.subscriber?.pc as RTCPeerConnection | undefined
      if (!pc) return
      const stats = await pc.getStats()
      const net = emptyNetworkStats()

      stats.forEach((report) => {
        if (report.type === 'inbound-rtp' && report.kind === 'video') {
          net.framesDecoded = report.framesDecoded ?? 0
          net.framesDropped = report.framesDropped ?? 0
          net.frameWidth = report.frameWidth ?? 0
          net.frameHeight = report.frameHeight ?? 0
          net.packetsLost = report.packetsLost ?? 0
          net.packetsReceived = report.packetsReceived ?? 0
          net.bytesReceived = report.bytesReceived ?? 0
          net.jitterMs = report.jitter != null ? Math.round(report.jitter * 1000 * 10) / 10 : null
          net.nackCount = report.nackCount ?? 0
          net.pliCount = report.pliCount ?? 0
          net.firCount = report.firCount ?? 0
          net.jitterBufferDelayMs = report.jitterBufferDelay != null && report.jitterBufferEmittedCount
            ? Math.round((report.jitterBufferDelay / report.jitterBufferEmittedCount) * 1000 * 10) / 10
            : null
          net.jitterBufferEmittedCount = report.jitterBufferEmittedCount ?? 0
          if (net.packetsReceived > 0) {
            net.lossRate = Math.round((net.packetsLost / (net.packetsLost + net.packetsReceived)) * 10000) / 10000
          }
          // Find codec
          if (report.codecId) {
            const codecReport = stats.get(report.codecId)
            if (codecReport) {
              net.codec = codecReport.mimeType ?? ''
            }
          }
        }
        if (report.type === 'candidate-pair' && report.state === 'succeeded') {
          net.roundTripTimeMs = report.currentRoundTripTime != null
            ? Math.round(report.currentRoundTripTime * 1000 * 10) / 10
            : null
        }
      })
      debugState.value.network = net
    } catch {
      // getStats can fail during reconnection, ignore
    }
  }

  function pushNote(note: string) {
    const next = [...debugState.value.notes, `${new Date().toISOString()} ${note}`]
    debugState.value.notes = next.slice(-10)
  }

  function flushPendingVideoTracks() {
    const el = videoElement.value
    if (!el || pendingVideoTracks.length === 0) return
    const pending = pendingVideoTracks.splice(0, pendingVideoTracks.length)
    for (const track of pending) {
      if (track.kind === Track.Kind.Video) {
        console.log(`[AVSync][${ts()}] 🎬 VIDEO track flush: sid=${track.sid}`)
        _videoMST = track.mediaStreamTrack
        _videoEl = el
        mergeCombinedStream()
        attachVideoFrameCallback(el)
      }
    }
  }

  watch(videoElement, () => {
    flushPendingVideoTracks()
  })

  const isMuted = ref(false)

  const MIC_LEVEL_BARS = 16
  const micBarLevels = ref<number[]>(Array.from({ length: MIC_LEVEL_BARS }, () => 0))

  let micAudioContext: AudioContext | null = null
  let micAnalyser: AnalyserNode | null = null
  let micRafId = 0
  let micMediaSource: MediaStreamAudioSourceNode | null = null
  let micFreqBuffer: Uint8Array<ArrayBuffer> | null = null

  function stopMicMetering() {
    if (micRafId) {
      cancelAnimationFrame(micRafId)
      micRafId = 0
    }
    micMediaSource?.disconnect()
    micMediaSource = null
    micAnalyser?.disconnect()
    micAnalyser = null
    micFreqBuffer = null
    if (micAudioContext && micAudioContext.state !== 'closed') {
      void micAudioContext.close()
    }
    micAudioContext = null
    micBarLevels.value = Array.from({ length: MIC_LEVEL_BARS }, () => 0)
  }

  function attachMicMeter(mediaTrack: MediaStreamTrack) {
    stopMicMetering()
    if (mediaTrack.readyState !== 'live') {
      const once = () => {
        if (mediaTrack.readyState === 'live') attachMicMeter(mediaTrack)
      }
      mediaTrack.addEventListener('unmute', once, { once: true })
      return
    }

    try {
      const ctx = new AudioContext()
      micAudioContext = ctx
      const src = ctx.createMediaStreamSource(new MediaStream([mediaTrack]))
      micMediaSource = src
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 512
      analyser.smoothingTimeConstant = 0.65
      micAnalyser = analyser
      micFreqBuffer = new Uint8Array(analyser.frequencyBinCount)
      src.connect(analyser)

      const tick = () => {
        if (!micAnalyser || !micFreqBuffer) return
        micAnalyser.getByteFrequencyData(micFreqBuffer)
        const data = micFreqBuffer
        const n = data.length
        const start = 1
        const usable = Math.max(1, n - start)
        const binW = usable / MIC_LEVEL_BARS
        const next: number[] = []
        for (let i = 0; i < MIC_LEVEL_BARS; i++) {
          const lo = Math.floor(start + i * binW)
          const hi = Math.floor(start + (i + 1) * binW)
          let sum = 0
          for (let j = lo; j < hi; j++) {
            sum += data[j] ?? 0
          }
          const bins = Math.max(1, hi - lo)
          const avg = sum / bins / 255
          next.push(Math.min(1, avg ** 0.65 * 3.2))
        }
        micBarLevels.value = next
        micRafId = requestAnimationFrame(tick)
      }

      void ctx.resume().then(() => {
        micRafId = requestAnimationFrame(tick)
      })
    } catch (e) {
      console.warn('[useWebRTC] mic meter failed', e)
      stopMicMetering()
    }
  }

  function tryAttachLocalMic(r: InstanceType<typeof Room>) {
    const pub = r.localParticipant.getTrackPublication(Track.Source.Microphone)
    const mst = pub?.track?.mediaStreamTrack
    if (mst && mst.readyState === 'live') {
      attachMicMeter(mst)
    }
  }

  async function toggleMute() {
    if (!room || connectionState.value !== 'connected') return
    const next = !isMuted.value
    await room.localParticipant.setMicrophoneEnabled(!next)
    isMuted.value = next
    pushNote(`mic ${next ? 'muted' : 'unmuted'}`)
  }

  async function connect(livekitUrl: string, token: string) {
    if (connectionState.value === 'connecting' || connectionState.value === 'connected') {
      return
    }

    connectionState.value = 'connecting'
    debugState.value = {
      ...emptyDebugState(),
      connectionState: 'connecting',
      sessionId: token.slice(0, 16),
    }
    error.value = ''
    resetState()

    try {
      room = new Room({
        adaptiveStream: false,
        dynacast: false,
      })

      room.on(RoomEvent.LocalTrackPublished, (publication) => {
        if (publication.source === Track.Source.Microphone) {
          const mst = publication.track?.mediaStreamTrack
          if (mst) {
            attachMicMeter(mst)
          }
        }
      })

      room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        const now = Date.now()

        if (track.kind === Track.Kind.Video) {
          logIncomingTrack('video', track)
          debugState.value.videoSubscribedAtMs = now
          pushNote(`video track subscribed sid=${track.sid}`)
          track.mediaStreamTrack.onunmute = () => {
            debugState.value.videoUnmutedAtMs = Date.now()
            pushNote(`video track onunmute sid=${track.sid}`)
          }
          _videoMST = track.mediaStreamTrack
          if (videoElement.value) {
            _videoEl = videoElement.value
            mergeCombinedStream()
            attachVideoFrameCallback(videoElement.value)
          } else {
            console.log(`[AVSync][${ts()}] 🎬 VIDEO track 排队 (videoElement 未就绪)`)
            pendingVideoTracks.push(track)
          }
        }

        if (track.kind === Track.Kind.Audio) {
          logIncomingTrack('audio', track)
          debugState.value.audioSubscribedAtMs = now
          pushNote(`audio track subscribed sid=${track.sid}`)
          track.mediaStreamTrack.onunmute = () => {
            debugState.value.audioUnmutedAtMs = Date.now()
            pushNote(`audio track onunmute sid=${track.sid}`)
          }
          _audioMST = track.mediaStreamTrack

          const el = videoElement.value
          if (el) {
            el.addEventListener('play', () => {
              if (audioPlayWallMs === null) {
                audioPlayWallMs = performance.now()
                debugState.value.firstPlayAtMs = Date.now()
                console.log(
                  `[AVSync][${ts()}] ▶ 首次播放: currentTime=${el.currentTime.toFixed(3)}s`
                )
              }
            }, { once: true })
          }

          mergeCombinedStream()
        }
      })

      room.on(RoomEvent.Disconnected, () => {
        stopMicMetering()
        connectionState.value = 'disconnected'
        debugState.value.connectionState = 'disconnected'
        pushNote('room disconnected')
      })

      const connectPromise = room.connect(livekitUrl, token)
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Connection timeout')), 10000)
      )
      await Promise.race([connectPromise, timeoutPromise])

      connectionState.value = 'connected'
      debugState.value.connectionState = 'connected'
      pushNote('room connected')
      await room.localParticipant.setMicrophoneEnabled(true)
      tryAttachLocalMic(room)
      // 每秒采集一次 WebRTC 网络统计
      networkStatsTimer = setInterval(() => void pollNetworkStats(), 1000)
    } catch (e: unknown) {
      stopMicMetering()
      const msg = e instanceof Error ? e.message : 'Connection failed'
      error.value = msg
      connectionState.value = 'error'
      debugState.value.connectionState = 'error'
      pushNote(`connect error: ${msg}`)
    }
  }

  function disconnect() {
    stopMicMetering()
    if (networkStatsTimer) {
      clearInterval(networkStatsTimer)
      networkStatsTimer = null
    }
    pendingVideoTracks.length = 0

    // 清理 <video> 元素的合并流
    if (videoElement.value) {
      videoElement.value.srcObject = null
    }

    resetState()
    room?.removeAllListeners()
    room?.disconnect()
    room = null
    connectionState.value = 'disconnected'
    debugState.value.connectionState = 'disconnected'
    if (debugPollTimer) {
      window.clearInterval(debugPollTimer)
      debugPollTimer = null
    }
  }

  let lastPollFrames = 0
  let lastPollTimeMs = 0

  watch(videoElement, (el) => {
    if (!el) return
    if (debugPollTimer) {
      window.clearInterval(debugPollTimer)
      debugPollTimer = null
    }
    lastPollFrames = 0
    lastPollTimeMs = Date.now()
    debugPollTimer = window.setInterval(() => {
      const quality = typeof el.getVideoPlaybackQuality === 'function'
        ? el.getVideoPlaybackQuality()
        : null
      debugState.value.videoCurrentTime = el.currentTime
      debugState.value.readyState = el.readyState
      debugState.value.playbackRate = el.playbackRate
      const currentFrames = quality?.totalVideoFrames ?? videoFrameCount
      debugState.value.decodedFrames = currentFrames
      debugState.value.droppedFrames = quality?.droppedVideoFrames ?? 0
      debugState.value.totalFrames = currentFrames
      debugState.value.lastVideoFrameAtMs = videoLastFrameWallMs
      const now = Date.now()
      const dt = (now - lastPollTimeMs) / 1000
      if (dt > 0 && lastPollTimeMs > 0) {
        debugState.value.fps = Math.round((currentFrames - lastPollFrames) / dt)
      }
      lastPollFrames = currentFrames
      lastPollTimeMs = now
      if (videoFirstFrameTime !== null && debugState.value.videoFirstFrameAtMs === null) {
        debugState.value.videoFirstFrameAtMs = Date.now()
      }
      // 更新帧抖动统计
      debugState.value.jitter = computeJitterStats()
    }, 500)
    el.addEventListener('emptied', () => pushNote('video element emptied'))
    el.addEventListener('waiting', () => pushNote('video element waiting'))
    el.addEventListener('stalled', () => pushNote('video element stalled'))
    el.addEventListener('playing', () => pushNote('video element playing'))
    el.addEventListener(
      'ended',
      () => {
        if (debugPollTimer) {
          window.clearInterval(debugPollTimer)
          debugPollTimer = null
        }
      },
      { once: true }
    )
  })

  return {
    videoElement,
    connectionState,
    debugState,
    error,
    isMuted,
    micBarLevels,
    connect,
    disconnect,
    toggleMute,
  }
}
