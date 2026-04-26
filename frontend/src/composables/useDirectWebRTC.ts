import { ref, watch } from 'vue'
import type {
  ConnectionState,
  AVSyncDebugState,
  FrameJitterStats,
  WebRTCNetworkStats,
} from './useWebRTC'

// ─── Diagnostics ──────────────────────────────────────────────────────────
const T0 = performance.now()
function ts() {
  return `+${((performance.now() - T0) / 1000).toFixed(3)}s`
}

// Frame tracking state (module-level, same pattern as useWebRTC)
let videoFrameCount = 0
let videoFirstFrameTime: number | null = null
let videoLastFrameWallMs: number | null = null
let audioPlayWallMs: number | null = null
let rvfcCheckpointMedia: number | null = null
let rvfcCheckpointCurrentTime: number | null = null
let debugPollTimer: ReturnType<typeof setInterval> | null = null

// Jitter measurement
const JITTER_WINDOW = 120
let frameArrivalTimes: number[] = []

function resetJitterState() {
  frameArrivalTimes = []
}

function recordFrameArrival(wallMs: number) {
  frameArrivalTimes.push(wallMs)
  if (frameArrivalTimes.length > JITTER_WINDOW) {
    frameArrivalTimes.shift()
  }
}

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
  const stutterThreshold = Math.max(mean * 2, 60)
  const stutterCount = intervals.filter(v => v > stutterThreshold).length
  return {
    meanIntervalMs: Math.round(mean * 10) / 10,
    stddevMs: Math.round(stddev * 10) / 10,
    maxIntervalMs: Math.round(maxInterval),
    p95IntervalMs: Math.round(p95),
    stutterCount,
    windowSize: intervals.length,
  }
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
  videoFrameCount = 0
  videoFirstFrameTime = null
  videoLastFrameWallMs = null
  audioPlayWallMs = null
  rvfcCheckpointMedia = null
  rvfcCheckpointCurrentTime = null
  resetJitterState()
}

type VideoFrameMeta = { mediaTime: number; presentedFrames: number }
type VideoWithRVFC = HTMLVideoElement & {
  requestVideoFrameCallback: (cb: (now: DOMHighResTimeStamp, meta: VideoFrameMeta) => void) => void
}

function attachVideoFrameCallback(el: HTMLVideoElement) {
  rvfcCheckpointMedia = null
  rvfcCheckpointCurrentTime = null

  if (!('requestVideoFrameCallback' in el)) {
    (el as HTMLVideoElement).addEventListener('timeupdate', () => {
      if ((el as HTMLVideoElement).currentTime > 0) {
        videoLastFrameWallMs = Date.now()
      }
      if (videoFirstFrameTime === null && (el as HTMLVideoElement).currentTime > 0) {
        videoFirstFrameTime = performance.now()
      }
    })
    return
  }

  const rvfc = (el as VideoWithRVFC).requestVideoFrameCallback.bind(el)
  const onFrame = (now: DOMHighResTimeStamp, meta: VideoFrameMeta) => {
    videoFrameCount++
    videoLastFrameWallMs = Date.now()
    recordFrameArrival(now)

    if (videoFirstFrameTime === null) {
      videoFirstFrameTime = now
      console.log(
        `[DirectRTC][${ts()}] VIDEO first frame: mediaTime=${meta.mediaTime.toFixed(3)}s` +
          ` presentedFrames=${meta.presentedFrames}`
      )
    }

    if (videoFrameCount > 0 && videoFrameCount % 50 === 0) {
      const ct = el.currentTime
      if (rvfcCheckpointMedia !== null && rvfcCheckpointCurrentTime !== null) {
        const dVm = meta.mediaTime - rvfcCheckpointMedia
        const dCt = ct - rvfcCheckpointCurrentTime
        console.log(
          `[DirectRTC][${ts()}] per-50 frames: dMedia=${dVm.toFixed(3)}s dCT=${dCt.toFixed(3)}s`
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

export function useDirectWebRTC() {
  const videoElement = ref<HTMLVideoElement | null>(null)
  const connectionState = ref<ConnectionState>('disconnected')
  const error = ref<string>('')
  const debugState = ref<AVSyncDebugState>(emptyDebugState())

  let pc: RTCPeerConnection | null = null
  let sendSignaling: ((msg: any) => void) | null = null
  let localStream: MediaStream | null = null
  let networkStatsTimer: ReturnType<typeof setInterval> | null = null

  // Serialize signaling: queue operations so addIceCandidate waits for setRemoteDescription
  let signalingChain: Promise<void> = Promise.resolve()

  // TURN ICE servers received from server via webrtc_config
  let pendingIceServers: RTCIceServer[] | null = null

  const isMuted = ref(false)
  let sentWebrtcReady = false

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
      mediaTrack.addEventListener('unmute', () => attachMicMeter(mediaTrack), { once: true })
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
      console.warn('[DirectRTC] mic meter failed', e)
      stopMicMetering()
    }
  }

  async function requestMicrophone(reason: 'connect' | 'click') {
    if (!window.isSecureContext) {
      throw new Error(`Microphone requires HTTPS or localhost (origin=${window.location.origin})`)
    }
    try {
      localStream = await navigator.mediaDevices.getUserMedia({ audio: true })
      for (const track of localStream.getAudioTracks()) {
        // default to unmuted when acquiring
        track.enabled = true
        attachMicMeter(track)
      }
      isMuted.value = false
      pushNote(`mic acquired (${reason})`)
    } catch (e: any) {
      let userMsg: string = e?.message || String(e)
      if (e?.name === 'NotAllowedError') {
        userMsg =
          'Microphone access denied. ' +
          'On macOS: System Settings → Privacy & Security → Microphone → enable your browser, then click the mic button to retry. ' +
          'On other platforms: check browser site permissions and allow microphone for this page.'
        throw new Error(userMsg)
      }
      throw new Error(userMsg)
    }
  }

  function pushNote(note: string) {
    const next = [...debugState.value.notes, `${new Date().toISOString()} ${note}`]
    debugState.value.notes = next.slice(-10)
  }

  async function pollNetworkStats() {
    if (!pc) return
    try {
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

  /**
   * Connect to the server via Direct P2P WebRTC.
   * Only acquires microphone and sends webrtc_ready.
   * PeerConnection is created later when the server sends webrtc_offer.
   * @param signalingFn - function to send signaling messages via WebSocket
   */
  async function connect(signalingFn: (msg: any) => void) {
    if (connectionState.value === 'connecting' || connectionState.value === 'connected') {
      return
    }

    connectionState.value = 'connecting'
    debugState.value = { ...emptyDebugState(), connectionState: 'connecting' }
    error.value = ''
    resetState()
    sendSignaling = signalingFn
    pendingIceServers = null
    sentWebrtcReady = false

    // Try to acquire mic, but don't abort the connection if it fails.
    // The WebRTC session (AI video/audio) still works without a mic track.
    // The user can retry via the mic button once OS/browser permission is granted.
    if (!localStream) {
      try {
        await requestMicrophone('connect')
      } catch (micErr: any) {
        pushNote(`mic failed on connect: ${micErr?.name ?? micErr}`)
      }
    } else {
      for (const track of localStream.getAudioTracks()) {
        attachMicMeter(track)
      }
    }

    try {
      // Tell server we're ready for negotiation
      sendSignaling({ type: 'webrtc_ready' })
      sentWebrtcReady = true
      pushNote('sent webrtc_ready')
    } catch (e: unknown) {
      stopMicMetering()
      const msg = e instanceof Error ? e.message : 'Connection failed'
      error.value = msg
      connectionState.value = 'error'
      debugState.value.connectionState = 'error'
      pushNote(`connect error: ${msg}`)
    }
  }

  /**
   * Create PeerConnection with the given ICE servers and set up handlers.
   */
  function createPeerConnection(iceServers: RTCIceServer[]): RTCPeerConnection {
    const newPc = new RTCPeerConnection({ iceServers })

    // Handle remote tracks (video + audio from server)
    let videoMST: MediaStreamTrack | null = null
    let audioMST: MediaStreamTrack | null = null

    const mergeStream = (resumePlay = false) => {
      const el = videoElement.value
      if (!el || !videoMST) return
      const tracks: MediaStreamTrack[] = [videoMST]
      if (audioMST) tracks.push(audioMST)
      el.srcObject = new MediaStream(tracks)
      el.muted = false
      console.log(`[DirectRTC][${ts()}] merged stream set: video=Y audio=${!!audioMST}`)
      // When srcObject is replaced (e.g. audio track added after video), Chrome does
      // not automatically resume playback — the autoplay attribute only fires once.
      // Explicitly call play() so audio is heard from the first response.
      if (resumePlay) {
        el.play().catch((e) => {
          console.warn(`[DirectRTC][${ts()}] play() after audio merge failed:`, e)
        })
      }
    }

    newPc.ontrack = (event) => {
      const track = event.track
      const now = Date.now()

      if (track.kind === 'video') {
        console.log(`[DirectRTC][${ts()}] VIDEO track received`)
        debugState.value.videoSubscribedAtMs = now
        pushNote(`video track received`)
        track.onunmute = () => {
          debugState.value.videoUnmutedAtMs = Date.now()
          pushNote('video track unmuted')
        }
        videoMST = track
        if (videoElement.value) {
          mergeStream()
          attachVideoFrameCallback(videoElement.value)
        }
      }

      if (track.kind === 'audio') {
        console.log(`[DirectRTC][${ts()}] AUDIO track received`)
        debugState.value.audioSubscribedAtMs = now
        pushNote('audio track received')
        track.onunmute = () => {
          debugState.value.audioUnmutedAtMs = Date.now()
          pushNote('audio track unmuted')
        }
        audioMST = track

        const el = videoElement.value
        if (el) {
          el.addEventListener('play', () => {
            if (audioPlayWallMs === null) {
              audioPlayWallMs = performance.now()
              debugState.value.firstPlayAtMs = Date.now()
              console.log(`[DirectRTC][${ts()}] first play: currentTime=${el.currentTime.toFixed(3)}s`)
            }
          }, { once: true })
        }

        mergeStream(true /* resumePlay: audio track just added */)
      }
    }

    newPc.onicecandidate = (event) => {
      if (event.candidate) {
        sendSignaling?.({
          type: 'ice_candidate',
          candidate: event.candidate.candidate,
          sdp_mid: event.candidate.sdpMid,
          sdp_mline_index: event.candidate.sdpMLineIndex,
        })
      }
    }

    newPc.onconnectionstatechange = () => {
      // Guard against stale callbacks after a mic-retry reconnect.
      if (newPc !== pc) return
      const state = newPc.connectionState
      console.log(`[DirectRTC][${ts()}] connection state: ${state}`)
      pushNote(`connection: ${state}`)
      if (state === 'connected') {
        connectionState.value = 'connected'
        debugState.value.connectionState = 'connected'
      } else if (state === 'failed' || state === 'closed') {
        connectionState.value = 'disconnected'
        debugState.value.connectionState = 'disconnected'
      }
    }

    // Add microphone tracks
    if (localStream) {
      for (const track of localStream.getAudioTracks()) {
        newPc.addTrack(track, localStream)
      }
    }

    // Start network stats polling
    if (networkStatsTimer) clearInterval(networkStatsTimer)
    networkStatsTimer = setInterval(() => void pollNetworkStats(), 1000)

    return newPc
  }

  /**
   * Handle incoming signaling messages from the server (via WebSocket).
   * Operations are serialized so that addIceCandidate always waits for
   * a pending setRemoteDescription to complete first.
   */
  function handleSignaling(data: any) {
    signalingChain = signalingChain.then(async () => {
      if (data.type === 'webrtc_config') {
        // Save TURN/STUN ICE servers from server; PC will be created on webrtc_offer
        pendingIceServers = data.ice_servers || null
        console.log(`[DirectRTC][${ts()}] received webrtc_config, ice_servers:`, pendingIceServers)
        pushNote('received webrtc_config')
        return
      }

      if (data.type === 'webrtc_offer') {
        // Create PeerConnection now, using TURN ICE servers if available
        const iceServers: RTCIceServer[] = pendingIceServers || [{ urls: 'stun:stun.l.google.com:19302' }]
        console.log(`[DirectRTC][${ts()}] creating PeerConnection with ICE servers:`, iceServers)
        pc = createPeerConnection(iceServers)

        console.log(`[DirectRTC][${ts()}] received SDP offer, setting remote description...`)
        await pc.setRemoteDescription(new RTCSessionDescription({
          type: 'offer',
          sdp: data.sdp,
        }))
        const answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        sendSignaling?.({
          type: 'webrtc_answer',
          sdp: answer.sdp,
        })
        console.log(`[DirectRTC][${ts()}] SDP answer sent`)
        pushNote('SDP answer sent')
        return
      }

      if (data.type === 'ice_candidate' && data.candidate) {
        if (!pc) return
        console.log(`[DirectRTC][${ts()}] adding remote ICE candidate: ${data.candidate.substring(0, 80)}...`)
        try {
          await pc.addIceCandidate(new RTCIceCandidate({
            candidate: data.candidate,
            sdpMid: data.sdp_mid ?? undefined,
            sdpMLineIndex: data.sdp_mline_index ?? undefined,
          }))
          console.log(`[DirectRTC][${ts()}] ICE candidate added successfully`)
        } catch (e) {
          console.warn('[DirectRTC] addIceCandidate failed:', e)
        }
      }
    }).catch(e => {
      console.error('[DirectRTC] signaling chain error:', e)
    })
  }

  async function toggleMute() {
    // If mic not yet acquired, retry on user gesture.
    if (!localStream) {
      error.value = ''
      try {
        await requestMicrophone('click')
        if (sentWebrtcReady && sendSignaling && connectionState.value === 'connected') {
          // Connection already established without mic — close the current PC and
          // re-negotiate so the new mic track is included. Only safe when fully
          // connected (not mid-negotiation).
          pc?.close()
          pc = null
          sentWebrtcReady = false
          signalingChain = Promise.resolve()
          connectionState.value = 'connecting'
          debugState.value.connectionState = 'connecting'
          sendSignaling({ type: 'webrtc_ready' })
          sentWebrtcReady = true
          pushNote('mic acquired, reconnecting with audio')
        } else if (!sentWebrtcReady && sendSignaling) {
          sendSignaling({ type: 'webrtc_ready' })
          sentWebrtcReady = true
          pushNote('sent webrtc_ready (mic acquired on click)')
        }
      } catch (e: unknown) {
        stopMicMetering()
        const msg = e instanceof Error ? e.message : 'Microphone failed'
        error.value = msg
        pushNote(`mic retry failed: ${msg}`)
      }
      return
    }

    const next = !isMuted.value
    for (const track of localStream.getAudioTracks()) {
      track.enabled = !next
    }
    isMuted.value = next
    pushNote(`mic ${next ? 'muted' : 'unmuted'}`)
  }

  function disconnect() {
    stopMicMetering()
    if (networkStatsTimer) {
      clearInterval(networkStatsTimer)
      networkStatsTimer = null
    }
    sentWebrtcReady = false

    if (videoElement.value) {
      videoElement.value.srcObject = null
    }

    if (localStream) {
      localStream.getTracks().forEach(t => t.stop())
      localStream = null
    }

    resetState()
    pc?.close()
    pc = null
    sendSignaling = null
    signalingChain = Promise.resolve()
    pendingIceServers = null
    connectionState.value = 'disconnected'
    debugState.value.connectionState = 'disconnected'
    if (debugPollTimer) {
      window.clearInterval(debugPollTimer)
      debugPollTimer = null
    }
  }

  // Debug state polling (same pattern as useWebRTC)
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
      debugState.value.jitter = computeJitterStats()
    }, 500)
    el.addEventListener('emptied', () => pushNote('video element emptied'))
    el.addEventListener('waiting', () => pushNote('video element waiting'))
    el.addEventListener('stalled', () => pushNote('video element stalled'))
    el.addEventListener('playing', () => pushNote('video element playing'))
    el.addEventListener('ended', () => {
      if (debugPollTimer) {
        window.clearInterval(debugPollTimer)
        debugPollTimer = null
      }
    }, { once: true })
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
    handleSignaling,
  }
}
