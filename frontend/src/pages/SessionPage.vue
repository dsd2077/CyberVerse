<script setup lang="ts">
import { ref, watchEffect, unref, onUnmounted, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import VideoPlayer from '../components/VideoPlayer.vue'
import ChatPanel from '../components/ChatPanel.vue'
import VoiceWaveform from '../components/VoiceWaveform.vue'
import { useWebRTC } from '../composables/useWebRTC'
import { useDirectWebRTC } from '../composables/useDirectWebRTC'
import { useChat } from '../composables/useChat'
import { useVisualInput, type VisualInputConfig } from '../composables/useVisualInput'
import { deleteSession } from '../services/api'

const router = useRouter()
const route = useRoute()
const sessionId = computed(() => route.params.id as string)

const videoPlayerRef = ref<InstanceType<typeof VideoPlayer> | null>(null)
const sessionVideoShellRef = ref<HTMLElement | null>(null)
const visualPreviewShellRef = ref<HTMLElement | null>(null)
const visualPreviewRef = ref<HTMLVideoElement | null>(null)
const visualPreviewPosition = ref<{ x: number; y: number } | null>(null)
const isVisualPreviewDragging = ref(false)
const elapsed = ref(0)
const clockMs = ref(Date.now())
let timer: ReturnType<typeof setInterval> | null = null
const showDiag = ref(false)
const isChatCollapsed = ref(false)

const streamingMode = (route.query.streaming_mode as string) || 'direct'
const sessionMode = computed<'voice_llm' | 'standard'>(() =>
  route.query.mode === 'standard' ? 'standard' : 'voice_llm'
)
const isStandardMode = computed(() => sessionMode.value === 'standard')

function parseVisualInputConfig(): Partial<VisualInputConfig> | undefined {
  const raw = route.query.visual_input
  if (!raw || Array.isArray(raw)) return undefined
  try {
    return JSON.parse(raw) as Partial<VisualInputConfig>
  } catch {
    return undefined
  }
}

const visualInputConfig = computed(() => parseVisualInputConfig())

// Both composables are called unconditionally (Vue requirement),
// but only the active one is wired up.
const lk = useWebRTC()
const dp = useDirectWebRTC()
const isDirectMode = streamingMode === 'direct'

const videoElement = isDirectMode ? dp.videoElement : lk.videoElement
const connectionState = isDirectMode ? dp.connectionState : lk.connectionState
const debugState = isDirectMode ? dp.debugState : lk.debugState
const isMuted = isDirectMode ? dp.isMuted : lk.isMuted
const micBarLevels = isDirectMode ? dp.micBarLevels : lk.micBarLevels
const toggleMute = isDirectMode ? dp.toggleMute : lk.toggleMute
const webrtcDisconnect = isDirectMode ? dp.disconnect : lk.disconnect

watchEffect(() => {
  const inst = videoPlayerRef.value
  const inner = inst?.videoRef
  videoElement.value = inner ? unref(inner) : null
})

const characterId = computed(() => (route.query.character_id as string) || '')

const {
  messages,
  currentTranscript,
  currentLLMResponse,
  avatarStatus,
  idleVideoUrl,
  idleVideoUrls,
  historyLoading,
  historyHasMore,
  sendText,
  connect: chatConnect,
  disconnect: chatDisconnect,
  loadHistory,
  registerSignalingHandler,
  sendSignaling,
  sendWSMessage,
  isConnected: chatConnected,
} = useChat(() => sessionId.value)

const visualInput = useVisualInput(
  (msg) => sendWSMessage(msg),
  () => visualInputConfig.value,
)
const canUseVisualInput = computed(() =>
  isStandardMode.value && chatConnected.value && (visualInputConfig.value?.enabled ?? true)
)
const visualPreviewStyle = computed(() => {
  const pos = visualPreviewPosition.value
  if (!pos) return undefined
  return {
    left: `${pos.x}px`,
    top: `${pos.y}px`,
    right: 'auto',
    bottom: 'auto',
  }
})

const VISUAL_PREVIEW_MARGIN = 12

type VisualPreviewDragState = {
  pointerId: number
  offsetX: number
  offsetY: number
  previewEl: HTMLElement
}

let visualPreviewDragState: VisualPreviewDragState | null = null

function clampValue(value: number, min: number, max: number): number {
  if (max < min) return min
  return Math.min(Math.max(value, min), max)
}

function clampVisualPreviewPosition(x: number, y: number, previewEl: HTMLElement): { x: number; y: number } {
  const shell = sessionVideoShellRef.value
  if (!shell) return { x, y }

  const shellRect = shell.getBoundingClientRect()
  const previewRect = previewEl.getBoundingClientRect()
  const maxX = shellRect.width - previewRect.width - VISUAL_PREVIEW_MARGIN
  const maxY = shellRect.height - previewRect.height - VISUAL_PREVIEW_MARGIN

  return {
    x: clampValue(x, VISUAL_PREVIEW_MARGIN, maxX),
    y: clampValue(y, VISUAL_PREVIEW_MARGIN, maxY),
  }
}

function updateVisualPreviewPosition(clientX: number, clientY: number) {
  const drag = visualPreviewDragState
  const shell = sessionVideoShellRef.value
  if (!drag || !shell) return

  const shellRect = shell.getBoundingClientRect()
  visualPreviewPosition.value = clampVisualPreviewPosition(
    clientX - shellRect.left - drag.offsetX,
    clientY - shellRect.top - drag.offsetY,
    drag.previewEl,
  )
}

function handleVisualPreviewPointerMove(event: PointerEvent) {
  if (!visualPreviewDragState || event.pointerId !== visualPreviewDragState.pointerId) return
  updateVisualPreviewPosition(event.clientX, event.clientY)
  event.preventDefault()
}

function stopVisualPreviewDragging(event?: PointerEvent) {
  if (event && visualPreviewDragState?.pointerId !== event.pointerId) return

  if (visualPreviewDragState) {
    try {
      visualPreviewDragState.previewEl.releasePointerCapture(visualPreviewDragState.pointerId)
    } catch {}
  }

  visualPreviewDragState = null
  isVisualPreviewDragging.value = false
  window.removeEventListener('pointermove', handleVisualPreviewPointerMove)
  window.removeEventListener('pointerup', stopVisualPreviewDragging)
  window.removeEventListener('pointercancel', stopVisualPreviewDragging)
}

function handleVisualPreviewPointerDown(event: PointerEvent) {
  if (event.button !== 0) return

  const shell = sessionVideoShellRef.value
  const previewEl = event.currentTarget as HTMLElement | null
  if (!shell || !previewEl) return

  const shellRect = shell.getBoundingClientRect()
  const previewRect = previewEl.getBoundingClientRect()
  visualPreviewDragState = {
    pointerId: event.pointerId,
    offsetX: event.clientX - previewRect.left,
    offsetY: event.clientY - previewRect.top,
    previewEl,
  }
  visualPreviewPosition.value = clampVisualPreviewPosition(
    previewRect.left - shellRect.left,
    previewRect.top - shellRect.top,
    previewEl,
  )
  isVisualPreviewDragging.value = true

  previewEl.setPointerCapture(event.pointerId)
  window.addEventListener('pointermove', handleVisualPreviewPointerMove)
  window.addEventListener('pointerup', stopVisualPreviewDragging)
  window.addEventListener('pointercancel', stopVisualPreviewDragging)
  event.preventDefault()
}

function keepVisualPreviewInBounds() {
  const pos = visualPreviewPosition.value
  const previewEl = visualPreviewShellRef.value
  if (!pos || !previewEl) return

  visualPreviewPosition.value = clampVisualPreviewPosition(pos.x, pos.y, previewEl)
}

watchEffect(() => {
  const el = visualPreviewRef.value
  if (!el) return
  const stream = visualInput.previewStream.value
  if (el.srcObject !== stream) {
    el.srcObject = stream
  }
  if (stream) {
    void el.play().catch(() => {})
  }
})

// Initialize idle video URLs from route query (if already cached at session creation)
const routeIdleUrls = route.query.idle_video_urls
  ? JSON.parse(route.query.idle_video_urls as string) as string[]
  : null
const routeIdleUrl = (route.query.idle_video_url as string) || ''
if (routeIdleUrls && routeIdleUrls.length > 0) {
  idleVideoUrls.value = routeIdleUrls
} else if (routeIdleUrl) {
  idleVideoUrls.value = [routeIdleUrl]
}

// Switch to webrtc only when fresh frames are actually arriving.
// The backend now delays avatar_status=speaking until the first video frame is
// about to be published, so we no longer need the "frozen last frame" fallback
// that previously showed a stale frame for seconds.
let _prevDisplayMode = ''
const displayMode = computed<'webrtc' | 'standby' | 'placeholder'>(() => {
  const lastFrameAt = debugState.value.lastVideoFrameAtMs
  const hasFreshRealtimeFrame = !!lastFrameAt && clockMs.value - lastFrameAt < 3000

  let result: 'webrtc' | 'standby' | 'placeholder'
  let reason = ''
  if (idleVideoUrl.value && avatarStatus.value !== 'speaking') {
    result = 'standby'
    reason = `avatarStatus=${avatarStatus.value}`
  } else if (hasFreshRealtimeFrame) {
    result = 'webrtc'
    reason = `fresh frame (${lastFrameAt ? (Date.now() - lastFrameAt) + 'ms ago' : ''})`
  } else if (idleVideoUrl.value) {
    result = 'standby'
    reason = `fallback (speaking but no fresh frame yet)`
  } else {
    result = 'placeholder'
    reason = 'no idle video'
  }
  if (result !== _prevDisplayMode) {
    console.log(`[switch] ${_prevDisplayMode || 'init'} → ${result} | reason: ${reason}`)
    _prevDisplayMode = result
  }
  return result
})

// Auto-connect on mount using session params from query
onMounted(async () => {
  window.addEventListener('resize', keepVisualPreviewInBounds)

  const startedAt = Date.now()

  await chatConnect()

  // Load conversation history for this character
  if (characterId.value) {
    loadHistory(characterId.value)
  }

  if (isDirectMode) {
    // Direct P2P WebRTC: register signaling handler then connect
    registerSignalingHandler((data: any) => dp.handleSignaling(data))
    await dp.connect((msg: any) => sendSignaling(msg))
  } else {
    // LiveKit mode
    const url = route.query.livekit_url as string
    const token = route.query.livekit_token as string
    if (url && token) {
      await lk.connect(url, token)
    }
  }

  timer = setInterval(() => {
    const now = Date.now()
    clockMs.value = now
    elapsed.value = Math.floor((now - startedAt) / 1000)
  }, 500)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
  stopVisualPreviewDragging()
  window.removeEventListener('resize', keepVisualPreviewInBounds)
  visualInput.stop(undefined, true)
})

async function handleDisconnect() {
  visualInput.stop(undefined, true)
  webrtcDisconnect()
  chatDisconnect()
  if (sessionId.value) {
    await deleteSession(sessionId.value).catch(() => {})
  }
  router.push('/characters')
}

function handleLoadMore() {
  if (characterId.value) {
    loadHistory(characterId.value)
  }
}

function toggleChatPanel() {
  isChatCollapsed.value = !isChatCollapsed.value
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}
</script>

<template>
  <div class="session-page" :class="{ 'chat-collapsed': isChatCollapsed }">
    <!-- Left: Video area (60%) -->
    <div ref="sessionVideoShellRef" class="session-video-shell">
      <VideoPlayer
        ref="videoPlayerRef"
        :display-mode="displayMode"
        :standby-src="idleVideoUrl"
        :standby-sources="idleVideoUrls"
        class="w-full flex-1 min-h-0"
      />

      <!-- Back button (top-left, glass) -->
      <button @click="handleDisconnect"
              class="absolute top-5 left-5 px-3 py-2 bg-black/70 backdrop-blur-sm rounded-cv-md text-sm text-cv-text hover:bg-black/90 transition-colors cursor-pointer z-10">
        ← 返回
      </button>

      <!-- FPS indicator (top-right, glass) — click to toggle diagnostics -->
      <button v-if="connectionState === 'connected'"
              @click="showDiag = !showDiag"
              class="absolute top-5 right-5 px-2.5 py-1.5 bg-black/70 backdrop-blur-sm rounded-cv-md text-xs font-mono text-cv-text z-10 cursor-pointer hover:bg-black/90 transition-colors">
        {{ debugState.fps }} FPS
        <span v-if="debugState.jitter.stutterCount > 0" class="ml-1 text-yellow-400">
          {{ debugState.jitter.stutterCount }} stutters
        </span>
      </button>

      <button
        v-if="isChatCollapsed"
        type="button"
        class="chat-expand-button"
        title="展开对话"
        aria-label="展开对话"
        @click="toggleChatPanel"
      >
        <svg class="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M10 3 5 8l5 5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>

      <!-- Diagnostics panel -->
      <div v-if="showDiag && connectionState === 'connected'"
           class="absolute top-14 right-5 w-80 max-h-[70vh] overflow-y-auto bg-black/85 backdrop-blur-md rounded-lg border border-white/10 p-3 text-[11px] font-mono text-cv-text z-20 space-y-2">
        <div class="text-xs font-semibold text-cv-accent mb-1">Frame Jitter</div>
        <div class="grid grid-cols-2 gap-x-3 gap-y-0.5">
          <span class="text-cv-text-muted">Mean interval</span>
          <span>{{ debugState.jitter.meanIntervalMs }} ms</span>
          <span class="text-cv-text-muted">Stddev</span>
          <span :class="debugState.jitter.stddevMs > 15 ? 'text-yellow-400' : ''">
            {{ debugState.jitter.stddevMs }} ms
          </span>
          <span class="text-cv-text-muted">P95</span>
          <span :class="debugState.jitter.p95IntervalMs > 80 ? 'text-red-400' : ''">
            {{ debugState.jitter.p95IntervalMs }} ms
          </span>
          <span class="text-cv-text-muted">Max</span>
          <span :class="debugState.jitter.maxIntervalMs > 100 ? 'text-red-400' : ''">
            {{ debugState.jitter.maxIntervalMs }} ms
          </span>
          <span class="text-cv-text-muted">Stutters</span>
          <span :class="debugState.jitter.stutterCount > 0 ? 'text-yellow-400' : ''">
            {{ debugState.jitter.stutterCount }} / {{ debugState.jitter.windowSize }} frames
          </span>
        </div>

        <div class="text-xs font-semibold text-cv-accent mt-2 mb-1">Playback</div>
        <div class="grid grid-cols-2 gap-x-3 gap-y-0.5">
          <span class="text-cv-text-muted">FPS</span>
          <span>{{ debugState.fps }}</span>
          <span class="text-cv-text-muted">Decoded</span>
          <span>{{ debugState.decodedFrames }}</span>
          <span class="text-cv-text-muted">Dropped</span>
          <span :class="debugState.droppedFrames > 0 ? 'text-red-400' : ''">
            {{ debugState.droppedFrames }}
          </span>
          <span class="text-cv-text-muted">Ready state</span>
          <span>{{ debugState.readyState }}</span>
          <span class="text-cv-text-muted">Display mode</span>
          <span>{{ displayMode }}</span>
        </div>

        <template v-if="debugState.network">
          <div class="text-xs font-semibold text-cv-accent mt-2 mb-1">Network (WebRTC)</div>
          <div class="grid grid-cols-2 gap-x-3 gap-y-0.5">
            <span class="text-cv-text-muted">RTT</span>
            <span>{{ debugState.network.roundTripTimeMs ?? '—' }} ms</span>
            <span class="text-cv-text-muted">Jitter (RTP)</span>
            <span>{{ debugState.network.jitterMs ?? '—' }} ms</span>
            <span class="text-cv-text-muted">Packet loss</span>
            <span :class="debugState.network.lossRate > 0.01 ? 'text-red-400' : ''">
              {{ debugState.network.packetsLost }} ({{ (debugState.network.lossRate * 100).toFixed(2) }}%)
            </span>
            <span class="text-cv-text-muted">NACK / PLI / FIR</span>
            <span>{{ debugState.network.nackCount }} / {{ debugState.network.pliCount }} / {{ debugState.network.firCount }}</span>
            <span class="text-cv-text-muted">Jitter buffer</span>
            <span>{{ debugState.network.jitterBufferDelayMs ?? '—' }} ms</span>
            <span class="text-cv-text-muted">Resolution</span>
            <span>{{ debugState.network.frameWidth }}x{{ debugState.network.frameHeight }}</span>
            <span class="text-cv-text-muted">Codec</span>
            <span>{{ debugState.network.codec || '—' }}</span>
          </div>
        </template>

        <div class="text-xs font-semibold text-cv-accent mt-2 mb-1">Notes</div>
        <div class="text-[10px] text-cv-text-muted space-y-0.5 max-h-24 overflow-y-auto">
          <div v-for="(note, i) in debugState.notes" :key="i">{{ note }}</div>
          <div v-if="!debugState.notes.length" class="italic">No events</div>
        </div>
      </div>

      <!-- Local mic input level (Web Audio analyser, not avatar state) -->
      <div class="absolute bottom-14 left-5 z-10 max-w-[min(100%,28rem)]">
        <VoiceWaveform
          type="user"
          label="麦克风输入"
          :levels="micBarLevels"
          :muted="isMuted"
        />
      </div>

      <div
        v-if="isStandardMode && visualInput.error.value"
        class="absolute bottom-24 left-1/2 -translate-x-1/2 z-10 max-w-[min(90vw,28rem)] px-3 py-2 bg-black/80 border border-red-400/30 text-red-100 text-xs rounded-cv-md"
      >
        {{ visualInput.error.value }}
      </div>

      <div
        v-if="visualInput.previewStream.value"
        class="visual-preview"
        :class="{ dragging: isVisualPreviewDragging }"
        :style="visualPreviewStyle"
        ref="visualPreviewShellRef"
        @pointerdown="handleVisualPreviewPointerDown"
      >
        <video
          ref="visualPreviewRef"
          class="visual-preview-video"
          :class="{ 'visual-preview-mirror': visualInput.isCameraActive.value }"
          autoplay
          muted
          playsinline
        />
        <button
          type="button"
          class="visual-preview-close"
          title="关闭视频输入"
          aria-label="关闭视频输入"
          @pointerdown.stop
          @click.stop="visualInput.stop()"
        >
          <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 4l8 8M12 4l-8 8" stroke-linecap="round" />
          </svg>
        </button>
      </div>

      <!-- Control bar (bottom center, floating) -->
      <div class="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 px-6 py-2.5 bg-black/70 backdrop-blur-xl rounded-2xl border border-white/10 shadow-[0_4px_16px_rgba(0,0,0,0.3)] z-10">
        <!-- User video input (standard mode only) -->
        <button
          v-if="isStandardMode"
          type="button"
          :disabled="!canUseVisualInput || visualInput.isStarting.value"
          :title="visualInput.isCameraActive.value ? '关闭摄像头输入' : '开启摄像头输入'"
          :aria-label="visualInput.isCameraActive.value ? '关闭摄像头输入' : '开启摄像头输入'"
          class="w-12 h-12 rounded-full flex items-center justify-center transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          :class="visualInput.isCameraActive.value ? 'bg-cv-accent text-white' : 'bg-white/10 text-cv-text hover:bg-white/16'"
          @click="visualInput.toggleCamera()"
        >
          <svg class="w-5 h-5" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6">
            <path d="M3.5 6.5A2.5 2.5 0 0 1 6 4h5a2.5 2.5 0 0 1 2.5 2.5v7A2.5 2.5 0 0 1 11 16H6a2.5 2.5 0 0 1-2.5-2.5v-7Z" />
            <path d="m13.5 8 3-2v8l-3-2" stroke-linecap="round" stroke-linejoin="round" />
            <path v-if="!visualInput.isCameraActive.value" d="M3 3l14 14" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
          </svg>
        </button>

        <!-- Mic button -->
        <button @click="toggleMute()"
                class="w-12 h-12 rounded-full flex items-center justify-center transition-colors cursor-pointer"
                :class="isMuted ? 'bg-cv-danger' : 'bg-cv-accent shadow-[0_2px_8px_rgba(59,130,246,0.3)]'">
          <svg class="w-5 h-5 text-white" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
            <rect x="7" y="2" width="6" height="10" rx="3" />
            <path d="M4 10a6 6 0 0012 0M10 16v2M7 18h6" stroke-linecap="round" />
            <path v-if="isMuted" d="M3 3l14 14" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
          </svg>
        </button>

        <!-- Timer -->
        <div class="flex items-center gap-2">
          <span class="w-1.5 h-1.5 rounded-full" :class="connectionState === 'connected' ? 'bg-cv-success' : 'bg-cv-danger'" />
          <span class="text-[11px] text-cv-text-muted font-mono">{{ formatTime(elapsed) }}</span>
        </div>

        <!-- Disconnect -->
        <button @click="handleDisconnect"
                class="w-10 h-10 rounded-full bg-cv-danger flex items-center justify-center text-white hover:bg-red-600 transition-colors cursor-pointer">
          <svg class="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 4l8 8M12 4l-8 8" stroke-linecap="round" />
          </svg>
        </button>
      </div>
    </div>

    <!-- Right: Chat panel (40%) -->
    <div class="session-chat-sidebar" :aria-hidden="isChatCollapsed" :inert="isChatCollapsed">
      <div class="session-chat-inner">
        <!-- Chat header -->
        <div class="h-[52px] shrink-0 flex items-center justify-between px-5 border-b border-cv-border-subtle">
          <div class="flex items-center gap-3 min-w-0">
            <button
              type="button"
              class="chat-toggle-button"
              title="收起对话"
              aria-label="收起对话"
              :aria-expanded="!isChatCollapsed"
              @click="toggleChatPanel"
            >
              <svg class="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8">
                <path d="M6 3l5 5-5 5" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            </button>
            <span class="text-base font-semibold text-cv-text truncate">对话</span>
          </div>
          <button class="text-[13px] text-cv-text-muted hover:text-cv-text transition-colors cursor-pointer">清空</button>
        </div>

        <!-- Messages (reuse ChatPanel) -->
        <ChatPanel
          :messages="messages"
          :current-transcript="currentTranscript"
          :current-l-l-m-response="currentLLMResponse"
          :avatar-status="avatarStatus"
          :history-loading="historyLoading"
          :history-has-more="historyHasMore"
          @send-text="sendText"
          @load-more="handleLoadMore"
          class="flex-1"
        />

        <!-- Footer hint -->
        <div class="h-6 flex items-center justify-center shrink-0">
          <span class="text-[11px] text-cv-text-muted">Shift+Enter 换行 · VoiceLLM 模式下可直接语音对话</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.session-page {
  position: relative;
  display: flex;
  width: 100%;
  height: 100vh;
  overflow: hidden;
  background: #000;
}

.session-video-shell {
  position: relative;
  flex: 1 1 auto;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  background: #000;
}

.session-chat-sidebar {
  flex: 0 0 clamp(360px, 40vw, 560px);
  width: clamp(360px, 40vw, 560px);
  min-width: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--color-cv-surface);
  border-left: 1px solid var(--color-cv-border-subtle);
  opacity: 1;
  transition:
    flex-basis 220ms ease,
    width 220ms ease,
    opacity 160ms ease,
    border-color 220ms ease;
}

.session-chat-inner {
  width: clamp(360px, 40vw, 560px);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.chat-collapsed .session-chat-sidebar {
  flex-basis: 0;
  width: 0;
  border-left-width: 0;
  border-left-color: transparent;
  opacity: 0;
  pointer-events: none;
}

.chat-toggle-button,
.chat-expand-button {
  width: 32px;
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  border-radius: 8px;
  color: var(--color-cv-text-secondary);
  background: transparent;
  transition:
    color 160ms ease,
    background-color 160ms ease;
  cursor: pointer;
}

.chat-toggle-button:hover,
.chat-expand-button:hover {
  color: var(--color-cv-text);
  background: var(--color-cv-hover);
}

.chat-expand-button {
  position: absolute;
  top: 64px;
  right: 20px;
  z-index: 10;
  color: var(--color-cv-text);
  background: rgba(0, 0, 0, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
}

.chat-expand-button:hover {
  background: rgba(0, 0, 0, 0.9);
}

.visual-preview {
  position: absolute;
  right: 20px;
  bottom: 88px;
  z-index: 12;
  width: clamp(132px, 18vw, 220px);
  aspect-ratio: 16 / 10;
  overflow: hidden;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  background: #05070a;
  box-shadow: 0 14px 42px rgba(0, 0, 0, 0.46);
  cursor: grab;
  touch-action: none;
  user-select: none;
}

.visual-preview.dragging {
  cursor: grabbing;
}

.visual-preview-video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.visual-preview-mirror {
  transform: scaleX(-1);
}

.visual-preview-close {
  position: absolute;
  top: 8px;
  right: 8px;
  width: 26px;
  height: 26px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  background: rgba(0, 0, 0, 0.58);
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 999px;
  cursor: pointer;
  transition: background-color 160ms ease;
}

.visual-preview-close:hover {
  background: rgba(0, 0, 0, 0.82);
}

@media (max-width: 900px) {
  .session-chat-sidebar,
  .session-chat-inner {
    width: min(82vw, 420px);
  }

  .session-chat-sidebar {
    flex-basis: min(82vw, 420px);
  }

  .visual-preview {
    right: 14px;
    bottom: 94px;
    width: min(42vw, 170px);
  }
}
</style>
