import { computed, ref, shallowRef } from 'vue'

export type VisualSource = 'camera' | 'screen'

export interface VisualInputConfig {
  enabled: boolean
  frame_interval_ms: number
  max_width: number
  max_height: number
  jpeg_quality: number
  max_frame_bytes: number
  ws_max_message_bytes: number
  max_recent_frames: number
  frame_ttl_ms: number
}

const DEFAULT_CONFIG: VisualInputConfig = {
  enabled: true,
  frame_interval_ms: 1000,
  max_width: 1280,
  max_height: 720,
  jpeg_quality: 0.78,
  max_frame_bytes: 512 * 1024,
  ws_max_message_bytes: 1024 * 1024,
  max_recent_frames: 2,
  frame_ttl_ms: 10000,
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const value = String(reader.result || '')
      const comma = value.indexOf(',')
      resolve(comma >= 0 ? value.slice(comma + 1) : value)
    }
    reader.onerror = () => reject(reader.error || new Error('Failed to read frame'))
    reader.readAsDataURL(blob)
  })
}

function canvasToBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob | null> {
  return new Promise((resolve) => {
    canvas.toBlob(resolve, 'image/jpeg', quality)
  })
}

function targetSize(video: HTMLVideoElement, maxWidth: number, maxHeight: number, scale: number) {
  const sourceWidth = Math.max(1, video.videoWidth)
  const sourceHeight = Math.max(1, video.videoHeight)
  const fit = Math.min(maxWidth / sourceWidth, maxHeight / sourceHeight, 1) * scale
  return {
    width: Math.max(1, Math.round(sourceWidth * fit)),
    height: Math.max(1, Math.round(sourceHeight * fit)),
  }
}

export function useVisualInput(
  sendMessage: (message: Record<string, unknown>) => boolean,
  getConfig?: () => Partial<VisualInputConfig> | undefined,
) {
  const activeSource = ref<VisualSource | null>(null)
  const isStarting = ref(false)
  const error = ref('')
  const lastFrameAt = ref(0)
  const droppedFrames = ref(0)
  const previewStream = shallowRef<MediaStream | null>(null)

  let stream: MediaStream | null = null
  let video: HTMLVideoElement | null = null
  let canvas: HTMLCanvasElement | null = null
  let timer: ReturnType<typeof setInterval> | null = null
  let frameSeq = 0
  let captureInFlight = false

  const isCameraActive = computed(() => activeSource.value === 'camera')
  const isScreenActive = computed(() => activeSource.value === 'screen')

  function config(): VisualInputConfig {
    return { ...DEFAULT_CONFIG, ...(getConfig?.() || {}) }
  }

  function clearTimer() {
    if (timer) {
      clearInterval(timer)
      timer = null
    }
  }

  function stopTracks() {
    stream?.getTracks().forEach((track) => {
      track.onended = null
      track.stop()
    })
    stream = null
    previewStream.value = null
  }

  function cleanupVideo() {
    if (video) {
      video.pause()
      video.srcObject = null
    }
    video = null
    canvas = null
    captureInFlight = false
  }

  function stop(source?: VisualSource, notify = true) {
    const stoppedSource = activeSource.value
    if (source && stoppedSource && source !== stoppedSource) return
    clearTimer()
    stopTracks()
    cleanupVideo()
    activeSource.value = null
    if (notify && stoppedSource) {
      sendMessage({ type: 'visual_input_stop', source: stoppedSource })
    }
  }

  async function encodeFrame(cfg: VisualInputConfig): Promise<{ blob: Blob; width: number; height: number } | null> {
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
      return null
    }
    const cnv = canvas || document.createElement('canvas')
    canvas = cnv

    const scaleAttempts = [1, 0.82, 0.68, 0.55]
    const qualityAttempts = [
      Math.min(Math.max(cfg.jpeg_quality, 0.35), 0.92),
      0.7,
      0.6,
      0.5,
    ]

    for (const scale of scaleAttempts) {
      const { width, height } = targetSize(video, cfg.max_width, cfg.max_height, scale)
      cnv.width = width
      cnv.height = height
      const ctx = cnv.getContext('2d')
      if (!ctx) return null
      ctx.drawImage(video, 0, 0, width, height)

      for (const quality of qualityAttempts) {
        const blob = await canvasToBlob(cnv, quality)
        if (blob && blob.size <= cfg.max_frame_bytes) {
          return { blob, width, height }
        }
      }
    }
    return null
  }

  async function captureFrame() {
    if (!activeSource.value || captureInFlight) return
    captureInFlight = true
    try {
      const cfg = config()
      const encoded = await encodeFrame(cfg)
      if (!encoded) {
        droppedFrames.value += 1
        return
      }
      const data = await blobToBase64(encoded.blob)
      const sent = sendMessage({
        type: 'visual_frame',
        source: activeSource.value,
        mime: 'image/jpeg',
        data,
        width: encoded.width,
        height: encoded.height,
        timestamp_ms: Date.now(),
        frame_seq: ++frameSeq,
      })
      if (sent) {
        lastFrameAt.value = Date.now()
      }
    } catch (e) {
      droppedFrames.value += 1
      console.warn('[VisualInput] capture failed', e)
    } finally {
      captureInFlight = false
    }
  }

  async function start(source: VisualSource) {
    const cfg = config()
    if (!cfg.enabled) {
      error.value = '当前会话未启用视觉输入'
      return
    }
    if (!window.isSecureContext) {
      error.value = '摄像头和屏幕共享需要 HTTPS 或 localhost'
      return
    }
    isStarting.value = true
    error.value = ''
    stop(undefined, true)

    try {
      stream = source === 'camera'
        ? await navigator.mediaDevices.getUserMedia({ video: true })
        : await navigator.mediaDevices.getDisplayMedia({ video: true })
      previewStream.value = stream

      for (const track of stream.getVideoTracks()) {
        track.onended = () => stop(source, true)
      }

      video = document.createElement('video')
      video.muted = true
      video.playsInline = true
      video.srcObject = stream
      await video.play()

      activeSource.value = source
      frameSeq = 0
      sendMessage({ type: 'visual_input_start', source })
      await captureFrame()
      timer = setInterval(() => void captureFrame(), Math.max(250, cfg.frame_interval_ms))
    } catch (e: any) {
      stop(undefined, false)
      if (e?.name === 'NotAllowedError') {
        error.value = source === 'camera' ? '摄像头权限被拒绝' : '屏幕共享权限被拒绝'
      } else {
        error.value = e?.message || '无法启动视觉输入'
      }
    } finally {
      isStarting.value = false
    }
  }

  async function toggleCamera() {
    if (activeSource.value === 'camera') {
      stop('camera', true)
      return
    }
    await start('camera')
  }

  async function toggleScreen() {
    if (activeSource.value === 'screen') {
      stop('screen', true)
      return
    }
    await start('screen')
  }

  return {
    activeSource,
    isCameraActive,
    isScreenActive,
    isStarting,
    error,
    lastFrameAt,
    droppedFrames,
    previewStream,
    toggleCamera,
    toggleScreen,
    stop,
  }
}
