<script setup lang="ts">
import { ref, watchEffect, unref, onUnmounted, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import VideoPlayer from '../components/VideoPlayer.vue'
import ChatPanel from '../components/ChatPanel.vue'
import VoiceWaveform from '../components/VoiceWaveform.vue'
import { useWebRTC } from '../composables/useWebRTC'
import { useDirectWebRTC } from '../composables/useDirectWebRTC'
import { useChat } from '../composables/useChat'
import { deleteSession } from '../services/api'

const router = useRouter()
const route = useRoute()
const sessionId = computed(() => route.params.id as string)

const videoPlayerRef = ref<InstanceType<typeof VideoPlayer> | null>(null)
const elapsed = ref(0)
const clockMs = ref(Date.now())
let timer: ReturnType<typeof setInterval> | null = null
const showDiag = ref(false)

const streamingMode = (route.query.streaming_mode as string) || 'direct'

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
} = useChat(() => sessionId.value)

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
})

async function handleDisconnect() {
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

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = s % 60
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}
</script>

<template>
  <div class="h-screen flex bg-black overflow-hidden">
    <!-- Left: Video area (60%) -->
    <div class="relative flex-[3] flex flex-col min-h-0 bg-black">
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

      <!-- Control bar (bottom center, floating) -->
      <div class="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 px-6 py-2.5 bg-black/70 backdrop-blur-xl rounded-2xl border border-white/10 shadow-[0_4px_16px_rgba(0,0,0,0.3)] z-10">
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
    <div class="flex-[2] border-l border-cv-border-subtle flex flex-col bg-cv-surface">
      <!-- Chat header -->
      <div class="h-[52px] shrink-0 flex items-center justify-between px-5 border-b border-cv-border-subtle">
        <span class="text-base font-semibold text-cv-text">对话</span>
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
</template>
