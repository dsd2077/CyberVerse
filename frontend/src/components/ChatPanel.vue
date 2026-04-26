<script setup lang="ts">
import { ref, nextTick, watch, onMounted, onUnmounted, computed } from 'vue'
import type { ChatMessage, AvatarStatus } from '../composables/useChat'

const props = defineProps<{
  messages: ChatMessage[]
  currentTranscript: string
  currentLLMResponse: string
  avatarStatus: AvatarStatus
  historyLoading?: boolean
  historyHasMore?: boolean
}>()

const emit = defineEmits<{
  sendText: [text: string]
  loadMore: []
}>()

const inputText = ref('')
const messagesContainer = ref<HTMLElement | null>(null)
const sentinel = ref<HTMLElement | null>(null)
let observer: IntersectionObserver | null = null
let prevMessageCount = 0
let isLoadingHistory = false
let scrollHeightBeforeLoad = 0
let initialLoadDone = false

function handleSend() {
  const text = inputText.value.trim()
  if (!text) return
  emit('sendText', text)
  inputText.value = ''
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}

// Compute session separators: indices where session_id changes between history messages
const sessionBreaks = computed(() => {
  const breaks = new Set<number>()
  for (let i = 1; i < props.messages.length; i++) {
    const prev = props.messages[i - 1]
    const curr = props.messages[i]
    if (prev.isHistory && curr.isHistory && prev.sessionId && curr.sessionId && prev.sessionId !== curr.sessionId) {
      breaks.add(i)
    }
    // Separator between history and live messages
    if (prev.isHistory && !curr.isHistory) {
      breaks.add(i)
    }
  }
  return breaks
})

watch(
  () => props.historyLoading,
  (loading) => {
    if (loading) {
      isLoadingHistory = true
      // Capture scroll height BEFORE history messages are inserted
      const container = messagesContainer.value
      if (container) {
        scrollHeightBeforeLoad = container.scrollHeight
      }
    }
  }
)

watch(
  () => props.messages.length,
  async (newLen) => {
    const container = messagesContainer.value
    if (!container) return

    if (isLoadingHistory && newLen > prevMessageCount) {
      // History was prepended: preserve scroll position using height captured before load
      await nextTick()
      const newHeight = container.scrollHeight
      container.scrollTop += newHeight - scrollHeightBeforeLoad
      isLoadingHistory = false
      initialLoadDone = true
    } else {
      // New message appended: scroll to bottom
      await nextTick()
      container.scrollTop = container.scrollHeight
    }
    prevMessageCount = newLen
  }
)

onMounted(() => {
  prevMessageCount = props.messages.length
  // Set up IntersectionObserver for infinite scroll up
  if (sentinel.value && messagesContainer.value) {
    observer = new IntersectionObserver(
      (entries) => {
        // Skip triggers before initial history load completes (sentinel is visible in empty container)
        if (!initialLoadDone) return
        if (entries[0]?.isIntersecting && props.historyHasMore && !props.historyLoading) {
          emit('loadMore')
        }
      },
      { root: messagesContainer.value, threshold: 0.1 }
    )
    observer.observe(sentinel.value)
  }
})

onUnmounted(() => {
  observer?.disconnect()
})
</script>

<template>
  <div class="chat-panel">
    <div ref="messagesContainer" class="messages">
      <!-- Sentinel for infinite scroll up -->
      <div ref="sentinel" class="sentinel">
        <div v-if="historyLoading" class="history-loading">
          <span class="loading-dot" /><span class="loading-dot" /><span class="loading-dot" />
        </div>
        <div v-else-if="historyHasMore" class="load-more-hint">
          ↑ 向上滚动加载更多
        </div>
        <div v-else-if="messages.some(m => m.isHistory)" class="history-end">
          — 已加载全部历史 —
        </div>
      </div>

      <template v-for="(msg, i) in messages" :key="msg.id || `msg-${msg.timestamp}-${i}`">
        <!-- Session separator -->
        <div v-if="sessionBreaks.has(i)" class="session-separator">
          <span class="separator-line" />
          <span v-if="msg.isHistory" class="separator-label">上一次对话</span>
          <span v-else class="separator-label">当前对话</span>
          <span class="separator-line" />
        </div>

        <div
          class="message"
          :class="[msg.role, { history: msg.isHistory }]"
        >
          <div class="message-content">{{ msg.content }}</div>
        </div>
      </template>

      <div v-if="currentTranscript" class="message user typing">
        <div class="message-content">{{ currentTranscript }}...</div>
      </div>

      <div v-if="currentLLMResponse" class="message assistant typing">
        <div class="message-content">{{ currentLLMResponse }}</div>
      </div>
    </div>

    <div class="input-bar">
      <input
        v-model="inputText"
        type="text"
        placeholder="Type a message..."
        @keydown="handleKeydown"
      />
      <button class="send-btn" @click="handleSend" :disabled="!inputText.trim()">
        Send
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #1e1e1e;
  overflow: hidden;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sentinel {
  min-height: 1px;
  display: flex;
  justify-content: center;
  padding: 4px 0;
}

.history-loading {
  display: flex;
  gap: 4px;
  align-items: center;
  padding: 8px;
}

.loading-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #666;
  animation: pulse 1.2s ease-in-out infinite;
}
.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1; }
}

.load-more-hint {
  font-size: 12px;
  color: #666;
  padding: 4px 0;
}

.history-end {
  font-size: 12px;
  color: #555;
  padding: 4px 0;
}

.session-separator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 0;
}

.separator-line {
  flex: 1;
  height: 1px;
  background: #333;
}

.separator-label {
  font-size: 11px;
  color: #666;
  white-space: nowrap;
}

.message {
  max-width: 80%;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.4;
}
.message.user {
  align-self: flex-end;
  background: #2563eb;
  color: white;
}
.message.assistant {
  align-self: flex-start;
  background: #333;
  color: #eee;
}
.message.history {
  opacity: 0.75;
}
.message.typing {
  opacity: 0.7;
}

.input-bar {
  display: flex;
  gap: 8px;
  padding: 12px;
  border-top: 1px solid #333;
}

.input-bar input {
  flex: 1;
  padding: 8px 12px;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 8px;
  color: white;
  outline: none;
}
.input-bar input:focus {
  border-color: #2563eb;
}

.send-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  background: #2563eb;
  color: white;
}
.send-btn:disabled {
  opacity: 0.5;
  cursor: default;
}
</style>
