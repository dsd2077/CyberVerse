import { ref, computed } from 'vue'
import { getConversationMessages, sendMessage } from '../services/api'

export interface ChatMessage {
  id?: string  // Optional ID for deduplication
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  isHistory?: boolean
  sessionId?: string
}

export type AvatarStatus = 'idle' | 'speaking' | 'processing'

export function useChat(sessionId: () => string) {
  const ws = ref<WebSocket | null>(null)
  const messages = ref<ChatMessage[]>([])
  const currentTranscript = ref('')

  // New state variables for separate pipeline tracking
  const currentVoiceResponse = ref('')      // For transcript events (voice pipeline)
  const currentTextResponse = ref('')        // For llm_token events (text pipeline)
  const activeResponseId = ref<string>('')   // Track active response to prevent duplicates
  const pipelineMode = ref<'text' | 'voice' | null>(null)  // Track active pipeline

  // Accumulation state for voice responses
  const accumulatedVoiceResponse = ref('')  // Accumulates all transcript texts
  const voiceResponseFinalized = ref(false) // Prevents duplicate finalization
  const latestTurnSeq = ref(0)

  // Computed property to show the appropriate response based on active pipeline
  const currentLLMResponse = computed(() => {
    // Show whichever has content, prioritizing the active pipeline
    if (pipelineMode.value === 'voice') {
      return currentVoiceResponse.value || currentTextResponse.value
    }
    return currentTextResponse.value || currentVoiceResponse.value
  })

  const avatarStatus = ref<AvatarStatus>('idle')
  const idleVideoUrls = ref<string[]>([])
  const idleVideoUrl = computed(() => idleVideoUrls.value.length > 0 ? idleVideoUrls.value[0] : '')
  const isConnected = ref(false)

  // Signaling handler for Direct WebRTC mode
  let signalingHandler: ((data: any) => void) | null = null

  function resetTransientState() {
    pipelineMode.value = null
    activeResponseId.value = ''
    currentVoiceResponse.value = ''
    currentTextResponse.value = ''
    currentTranscript.value = ''
    accumulatedVoiceResponse.value = ''
    voiceResponseFinalized.value = false
  }

  function parseTurnSeq(data: any): number {
    const turnSeq = Number(data?.turn_seq ?? 0)
    return Number.isFinite(turnSeq) && turnSeq > 0 ? turnSeq : 0
  }

  function acceptTurnSeq(turnSeq: number): boolean {
    if (!turnSeq) return true
    if (turnSeq < latestTurnSeq.value) {
      return false
    }
    if (turnSeq > latestTurnSeq.value) {
      latestTurnSeq.value = turnSeq
      resetTransientState()
    }
    return true
  }

  function registerSignalingHandler(fn: (data: any) => void) {
    signalingHandler = fn
  }

  function sendSignaling(msg: any) {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) return
    ws.value.send(JSON.stringify(msg))
  }

  function connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsBase = import.meta.env.VITE_WS_BASE || `ws://${window.location.host}`
      const url = `${wsBase}/ws/chat/${sessionId()}`
      ws.value = new WebSocket(url)

      ws.value.onopen = () => {
        isConnected.value = true
        resolve()
      }

      ws.value.onclose = () => {
        isConnected.value = false
      }

      ws.value.onerror = (e) => {
        reject(e)
      }

    ws.value.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'transcript': {
          const turnSeq = parseTurnSeq(data)
          if (!acceptTurnSeq(turnSeq)) {
            break
          }
          const role: ChatMessage['role'] = data.speaker === 'assistant' ? 'assistant' : 'user'

          if (role === 'assistant') {
            const responseId = turnSeq ? `voice-${turnSeq}` : (activeResponseId.value || `voice-${Date.now()}`)
            if (!activeResponseId.value || pipelineMode.value !== 'voice') {
              activeResponseId.value = responseId
              accumulatedVoiceResponse.value = ''
              voiceResponseFinalized.value = false
              pipelineMode.value = 'voice'
            }
            activeResponseId.value = responseId

            if (data.is_final) {
              const finalText = data.text || accumulatedVoiceResponse.value || currentVoiceResponse.value
              if (finalText) {
                const alreadyExists = messages.value.some(m => m.id === responseId)
                if (!alreadyExists) {
                  messages.value.push({
                    id: responseId,
                    role,
                    content: finalText,
                    timestamp: Date.now(),
                  })
                }
              }

              resetTransientState()
            } else {
              accumulatedVoiceResponse.value += data.text || ''
              currentVoiceResponse.value = accumulatedVoiceResponse.value
            }
          } else {
            currentTranscript.value = data.text
            if (data.is_final) {
              const messageId = turnSeq ? `user-${turnSeq}` : undefined
              const alreadyExists = messageId ? messages.value.some(m => m.id === messageId) : false
              if (!alreadyExists) {
                messages.value.push({
                  id: messageId,
                  role,
                  content: data.text,
                  timestamp: Date.now(),
                })
              }
              currentTranscript.value = ''
            }
          }
          break
        }

        case 'llm_token': {
          const turnSeq = parseTurnSeq(data)
          if (!acceptTurnSeq(turnSeq)) {
            break
          }
          const responseId = turnSeq ? `text-${turnSeq}` : (activeResponseId.value || `text-${Date.now()}`)

          if (!pipelineMode.value) {
            pipelineMode.value = 'text'
            activeResponseId.value = responseId
          }
          if (pipelineMode.value !== 'text') {
            resetTransientState()
            pipelineMode.value = 'text'
          }
          activeResponseId.value = responseId

          currentTextResponse.value = data.accumulated

          if (data.is_final) {
            const alreadyExists = messages.value.some(m => m.id === responseId)

            if (responseId && !alreadyExists) {
              messages.value.push({
                id: responseId,
                role: 'assistant',
                content: data.accumulated,
                timestamp: Date.now(),
              })
            }

            currentTextResponse.value = ''
            if (!currentVoiceResponse.value) {
              pipelineMode.value = null
              activeResponseId.value = ''
            }
          }
          break
        }

        case 'idle_video_ready':
          if (data.urls && data.urls.length > 0) {
            idleVideoUrls.value = data.urls
          } else if (data.url) {
            idleVideoUrls.value = [data.url]
          }
          break

        case 'avatar_warning':
          console.warn('[CyberVerse]', data.message || data)
          break

        case 'avatar_status': {
          const turnSeq = parseTurnSeq(data)
          if (!acceptTurnSeq(turnSeq)) {
            break
          }
          avatarStatus.value = data.status
          if (data.status === 'idle') {
            resetTransientState()
          }
          break
        }

        case 'webrtc_config':
        case 'webrtc_offer':
        case 'ice_candidate':
          if (signalingHandler) {
            signalingHandler(data)
          }
          break

        default:
          console.warn('Unknown message type:', data.type)
      }
    }
    })
  }

  function sendText(text: string) {
    const trimmed = text.trim()
    if (!trimmed) return

    const sid = sessionId()
    const payload = JSON.stringify({ type: 'text_input', text: trimmed })
    let sentViaWS = false

    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      try {
        ws.value.send(payload)
        sentViaWS = true
      } catch (err) {
        console.error('[useChat] WS send failed, fallback to HTTP:', err)
      }
    }

    if (!sentViaWS) {
      if (!sid) {
        console.error('[useChat] Cannot send text: missing session id')
        messages.value.push({
          role: 'system',
          content: '发送失败：会话未初始化，请刷新后重试。',
          timestamp: Date.now(),
        })
        return
      }
      void sendMessage(sid, trimmed).catch((err) => {
        console.error('[useChat] HTTP fallback send failed:', err)
        messages.value.push({
          role: 'system',
          content: '发送失败：网络异常，请稍后重试。',
          timestamp: Date.now(),
        })
      })
    }

    messages.value.push({
      role: 'user',
      content: trimmed,
      timestamp: Date.now(),
    })
  }

  function interrupt() {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) return
    ws.value.send(JSON.stringify({ type: 'interrupt' }))
  }

  // ── History loading ──
  const historyLoading = ref(false)
  const historyHasMore = ref(false)
  const historyNextCursor = ref('')

  async function loadHistory(characterId: string) {
    if (!characterId || historyLoading.value) return
    historyLoading.value = true
    try {
      const resp = await getConversationMessages(
        characterId,
        50,
        historyNextCursor.value || undefined,
      )
      const historyMessages: ChatMessage[] = resp.messages.map((m) => ({
        role: m.role as ChatMessage['role'],
        content: m.content,
        timestamp: new Date(m.timestamp).getTime() || 0,
        isHistory: true,
        sessionId: m.session_id,
      }))
      // Prepend history before current messages
      messages.value = [...historyMessages, ...messages.value]
      historyNextCursor.value = resp.next_cursor
      historyHasMore.value = resp.has_more
    } catch (e) {
      console.error('[useChat] Failed to load history:', e)
    } finally {
      historyLoading.value = false
    }
  }

  function disconnect() {
    ws.value?.close()
    ws.value = null
    isConnected.value = false
    latestTurnSeq.value = 0
    resetTransientState()
  }

  return {
    messages,
    currentTranscript,
    currentLLMResponse,  // Now a computed property
    avatarStatus,
    idleVideoUrl,
    idleVideoUrls,
    isConnected,
    historyLoading,
    historyHasMore,
    connect,
    sendText,
    interrupt,
    disconnect,
    loadHistory,
    registerSignalingHandler,
    sendSignaling,
  }
}
