import { ref, computed } from 'vue'
import {
  getConversationMessages,
  getTaskArtifactUrl,
  getTaskEvents,
  listSessionTasks,
  sendMessage,
  type AgentTask,
  type AgentTaskEvent,
} from '../services/api'
import { translate } from '../i18n'

export type TaskStatus = 'queued' | 'running' | 'waiting_user' | 'completed' | 'failed' | 'cancelled'

export interface ChatTaskTimelineItem {
  seq: number
  eventType: string
  title: string
  description: string
  status: TaskStatus
  progress: number
  createdAt?: string
}

export interface ChatTaskArtifact {
  id: string
  title: string
  type: string
  url: string
  mimeType?: string
}

export interface ChatTaskState {
  id: string
  agentName: string
  title: string
  status: TaskStatus
  progress: number
  eventCount: number
  currentStep: string
  events: ChatTaskTimelineItem[]
  artifacts: ChatTaskArtifact[]
}

export interface ChatMessage {
  id?: string  // Optional ID for deduplication
  kind?: 'text' | 'task'
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  isHistory?: boolean
  sessionId?: string
  artifactUrl?: string
  task?: ChatTaskState
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

  const latestTurnSeq = ref(0)
  const voiceDrafts = new Map<string, string>()
  const lastTaskSeqByTaskId = new Map<string, number>()

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
  }

  function parseTurnSeq(data: any): number {
    const turnSeq = Number(data?.turn_seq ?? 0)
    return Number.isFinite(turnSeq) && turnSeq > 0 ? turnSeq : 0
  }

  function beginTurnSeq(turnSeq: number) {
    if (!turnSeq) return
    if (turnSeq > latestTurnSeq.value) {
      latestTurnSeq.value = turnSeq
      resetTransientState()
    }
  }

  function isOlderTurn(turnSeq: number): boolean {
    return turnSeq > 0 && turnSeq < latestTurnSeq.value
  }

  function upsertMessage(message: ChatMessage, afterId?: string) {
    if (message.id) {
      const existingIndex = messages.value.findIndex(m => m.id === message.id)
      if (existingIndex >= 0) {
        messages.value[existingIndex] = { ...messages.value[existingIndex], ...message }
        return
      }
    }

    if (afterId) {
      const anchorIndex = messages.value.findIndex(m => m.id === afterId)
      if (anchorIndex >= 0) {
        messages.value.splice(anchorIndex + 1, 0, message)
        return
      }
    }

    messages.value.push(message)
  }

  function clearActiveResponse(responseId: string) {
    if (activeResponseId.value === responseId) {
      resetTransientState()
    }
  }

  function registerSignalingHandler(fn: (data: any) => void) {
    signalingHandler = fn
  }

  function asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
  }

  function readString(value: unknown): string {
    return typeof value === 'string' ? value.trim() : ''
  }

  function readTaskStatus(value: unknown, fallback: TaskStatus = 'running'): TaskStatus {
    switch (value) {
      case 'queued':
      case 'running':
      case 'waiting_user':
      case 'completed':
      case 'failed':
      case 'cancelled':
        return value
      default:
        return fallback
    }
  }

  function normalizeProgress(value: unknown, fallback = 0): number {
    const progress = Number(value ?? fallback)
    if (!Number.isFinite(progress)) return fallback
    return Math.min(100, Math.max(0, Math.round(progress)))
  }

  function normalizeSeq(value: unknown): number {
    const seq = Number(value ?? 0)
    if (!Number.isFinite(seq) || seq <= 0) return 0
    return Math.trunc(seq)
  }

  function readTimestamp(value: unknown): number {
    if (typeof value !== 'string' || !value.trim()) return 0
    const timestamp = new Date(value).getTime()
    return Number.isFinite(timestamp) ? timestamp : 0
  }

  function taskText(key: string, named?: Record<string, unknown>): string {
    return translate(`task.${key}`, named)
  }

  function agentNameForKind(kind: string): string {
    if (kind === 'research') return taskText('agent.research')
    if (!kind) return taskText('agent.generic')
    const label = `${kind.charAt(0).toUpperCase()}${kind.slice(1)}`
    return taskText('agent.typed', { kind: label })
  }

  function taskCardTitle(task: Record<string, unknown>, fallbackTitle: string): string {
    return readString(task.user_request) || readString(task.title) || fallbackTitle || taskText('defaultTitle')
  }

  function eventTitle(eventType: string, message: string): string {
    switch (eventType) {
      case 'task.queued':
        return taskText('event.queued')
      case 'task.started':
        return taskText('event.started')
      case 'plan.created':
        return taskText('event.planCreated')
      case 'research.blocked':
        return taskText('event.researchBlocked')
      case 'artifact.created':
        return taskText('event.artifactCreated')
      case 'task.completed':
        return taskText('event.completed')
      case 'task.failed':
        return taskText('event.failed')
      case 'task.cancelled':
        return taskText('event.cancelled')
      default:
        return message || eventType || taskText('event.statusUpdate')
    }
  }

  function artifactTypeLabel(type: string): string {
    const normalized = type.toLowerCase()
    if (normalized.includes('html')) return 'HTML'
    if (normalized.includes('markdown') || normalized === 'md') return 'MD'
    return (type || 'HTML').toUpperCase()
  }

  function eventDescription(eventType: string, message: string, payload: Record<string, unknown>): string {
    if (eventType === 'plan.created') {
      const steps = Array.isArray(payload.steps) ? payload.steps.map(readString).filter(Boolean) : []
      if (steps.length > 0) return taskText('event.planDescription', { steps: steps.join(taskText('listJoiner')) })
    }
    if (eventType === 'artifact.created') {
      return taskText('event.artifactDescription', { type: artifactTypeLabel(readString(payload.type) || 'html') })
    }
    if (eventType === 'task.completed') {
      return message || taskText('event.completedDescription')
    }
    return message || eventTitle(eventType, message)
  }

  function upsertArtifact(
    artifacts: ChatTaskArtifact[],
    taskId: string,
    payload: Record<string, unknown>,
    fallbackTitle: string,
  ): ChatTaskArtifact[] {
    const artifactId = readString(payload.artifact_id)
    if (!taskId || !artifactId) return artifacts

    const existing = artifacts.find((artifact) => artifact.id === artifactId)
    const type = readString(payload.type) || existing?.type || 'html'
    const mimeType = readString(payload.mime_type)
      || existing?.mimeType
      || (type.includes('html') ? 'text/html; charset=utf-8' : 'text/plain; charset=utf-8')
    const title = readString(payload.title) || existing?.title || fallbackTitle || taskText('defaultArtifactTitle')
    const content = readString(payload.content)
    const projectedUrl = readString(payload.url)
    const nextArtifact: ChatTaskArtifact = {
      id: artifactId,
      title,
      type,
      mimeType,
      url: projectedUrl
        || (content
          ? URL.createObjectURL(new Blob([content], { type: mimeType }))
          : existing?.url || getTaskArtifactUrl(taskId, artifactId)),
    }

    if (existing) {
      return artifacts.map((artifact) => artifact.id === artifactId ? { ...artifact, ...nextArtifact } : artifact)
    }
    return [...artifacts, nextArtifact]
  }

  function buildTaskMessage(
    data: any,
    fallbackTask?: Partial<AgentTask>,
    options: { isHistory?: boolean; sessionId?: string } = {},
  ): ChatMessage | null {
    const task = {
      ...asRecord(fallbackTask),
      ...asRecord(data.task),
    }
    const taskId = readString(data.task_id) || readString(task.id)
    if (!taskId) return null

    const previousMessage = messages.value.find(m => m.id === `task-${taskId}`)
    const previous = previousMessage?.task
    const payload = asRecord(data.payload)
    const eventType = readString(data.event_type)
    const message = readString(data.message) || taskText('event.statusUpdated')
    const status = readTaskStatus(data.status || task.status, previous?.status || 'running')
    const progress = normalizeProgress(data.progress ?? task.progress, previous?.progress || 0)
    const kind = readString(task.kind) || 'research'
    const title = taskCardTitle(task, previous?.title || taskText('defaultTitle'))
    const seq = normalizeSeq(data.seq)

    const nextEvent: ChatTaskTimelineItem | null = eventType
      ? {
          seq,
          eventType,
          title: eventTitle(eventType, message),
          description: eventDescription(eventType, message, payload),
          status,
          progress,
          createdAt: readString(data.created_at),
        }
      : null

    let events = previous?.events ? [...previous.events] : []
    if (nextEvent) {
      const existingIndex = nextEvent.seq > 0 ? events.findIndex(event => event.seq === nextEvent.seq) : -1
      if (existingIndex >= 0) {
        events[existingIndex] = nextEvent
      } else {
        events.push(nextEvent)
      }
      events = events.sort((a, b) => a.seq - b.seq)
    }

    let artifacts = previous?.artifacts ? [...previous.artifacts] : []
    if (payload.artifact_id) {
      artifacts = upsertArtifact(artifacts, taskId, payload, readString(task.title) || title)
    }

    const latestEvent = events[events.length - 1]
    const currentStep = status === 'completed' && artifacts.length > 0
      ? taskText('current.completedWithArtifact')
      : status === 'failed'
        ? message || taskText('event.failed')
        : status === 'cancelled'
          ? taskText('event.cancelled')
          : latestEvent?.title || message

    return {
      id: `task-${taskId}`,
      kind: 'task',
      role: 'system',
      content: title,
      timestamp: readTimestamp(data.created_at) || readTimestamp(task.updated_at) || previousMessage?.timestamp || Date.now(),
      isHistory: options.isHistory ?? previousMessage?.isHistory,
      sessionId: options.sessionId || readString(data.session_id) || readString(task.session_id) || previousMessage?.sessionId,
      task: {
        id: taskId,
        agentName: agentNameForKind(kind),
        title,
        status,
        progress,
        eventCount: Math.max(events.length, previous?.eventCount || 0),
        currentStep,
        events,
        artifacts,
      },
    }
  }

  function handleTaskEvent(
    data: AgentTaskEvent | any,
    fallbackTask?: Partial<AgentTask>,
    options: { isHistory?: boolean; orderByTimestamp?: boolean; sessionId?: string } = {},
  ) {
    const task = data.task || fallbackTask || {}
    const taskId = data.task_id || task.id || ''
    const seq = Number(data.seq ?? 0)
    if (taskId && Number.isFinite(seq) && seq > 0) {
      const previousSeq = lastTaskSeqByTaskId.get(taskId) || 0
      if (seq <= previousSeq) return
      lastTaskSeqByTaskId.set(taskId, seq)
    }
    const taskMessage = buildTaskMessage(data, fallbackTask, options)
    if (taskMessage) {
      upsertMessage(taskMessage)
      if (options.orderByTimestamp) {
        messages.value = [...messages.value].sort((a, b) => a.timestamp - b.timestamp)
      }
    }
  }

  async function recoverTaskEventsForSession(
    sid: string,
    options: { isHistory?: boolean; orderByTimestamp?: boolean } = {},
  ) {
    if (!sid) return
    try {
      const resp = await listSessionTasks(sid)
      await Promise.all((resp.tasks || []).map(async (task) => {
        const afterSeq = lastTaskSeqByTaskId.get(task.id) || 0
        const eventsResp = await getTaskEvents(task.id, afterSeq)
        for (const event of eventsResp.events || []) {
          handleTaskEvent(event, task, { ...options, sessionId: task.session_id || sid })
        }
      }))
    } catch (err) {
      console.warn(`[useChat] Failed to recover task events for session ${sid}:`, err)
    }
  }

  async function recoverTaskEvents() {
    await recoverTaskEventsForSession(sessionId())
  }

  function sendSignaling(msg: any) {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) return
    ws.value.send(JSON.stringify(msg))
  }

  function sendWSMessage(msg: any): boolean {
    if (!ws.value || ws.value.readyState !== WebSocket.OPEN) return false
    ws.value.send(JSON.stringify(msg))
    return true
  }

  function connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsBase = import.meta.env.VITE_WS_BASE || `ws://${window.location.host}`
      const url = `${wsBase}/ws/chat/${sessionId()}`
      ws.value = new WebSocket(url)

      ws.value.onopen = () => {
        isConnected.value = true
        void recoverTaskEvents()
        resolve()
      }

      ws.value.onclose = () => {
        isConnected.value = false
      }

      ws.value.onerror = (e) => {
        console.error('[useChat] WebSocket connection failed:', e)
        reject(new Error('WebSocket connection failed'))
      }

    ws.value.onmessage = (event: MessageEvent) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'transcript': {
          const turnSeq = parseTurnSeq(data)
          const olderTurn = isOlderTurn(turnSeq)
          if (!olderTurn) beginTurnSeq(turnSeq)
          const role: ChatMessage['role'] = data.speaker === 'assistant' ? 'assistant' : 'user'

          if (role === 'assistant') {
            const responseId = turnSeq ? `voice-${turnSeq}` : (activeResponseId.value || `voice-${Date.now()}`)

            if (data.is_final) {
              const finalText = data.text || voiceDrafts.get(responseId) || (activeResponseId.value === responseId ? currentVoiceResponse.value : '')
              if (finalText) {
                upsertMessage({
                  id: responseId,
                  role,
                  content: finalText,
                  timestamp: Date.now(),
                }, turnSeq ? `user-${turnSeq}` : undefined)
              }

              voiceDrafts.delete(responseId)
              clearActiveResponse(responseId)
            } else {
              if (olderTurn) break
              const nextText = (voiceDrafts.get(responseId) || '') + (data.text || '')
              voiceDrafts.set(responseId, nextText)
              activeResponseId.value = responseId
              pipelineMode.value = 'voice'
              currentVoiceResponse.value = nextText
            }
          } else {
            if (!olderTurn) {
              currentTranscript.value = data.text
            }
            if (data.is_final) {
              const messageId = turnSeq ? `user-${turnSeq}` : undefined
              upsertMessage({
                id: messageId,
                role,
                content: data.text,
                timestamp: Date.now(),
              })
              if (!olderTurn) {
                currentTranscript.value = ''
              }
            }
          }
          break
        }

        case 'llm_token': {
          const turnSeq = parseTurnSeq(data)
          const olderTurn = isOlderTurn(turnSeq)
          if (!olderTurn) beginTurnSeq(turnSeq)
          const responseId = turnSeq ? `text-${turnSeq}` : (activeResponseId.value || `text-${Date.now()}`)

          if (olderTurn) {
            if (data.is_final && data.accumulated) {
              upsertMessage({
                id: responseId,
                role: 'assistant',
                content: data.accumulated,
                timestamp: Date.now(),
              }, turnSeq ? `user-${turnSeq}` : undefined)
            }
            break
          }

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
            if (responseId) {
              upsertMessage({
                id: responseId,
                role: 'assistant',
                content: data.accumulated,
                timestamp: Date.now(),
              }, turnSeq ? `user-${turnSeq}` : undefined)
            }

            currentTextResponse.value = ''
            clearActiveResponse(responseId)
          }
          break
        }

        case 'assistant_message': {
          const turnSeq = parseTurnSeq(data)
          if (!isOlderTurn(turnSeq)) beginTurnSeq(turnSeq)
          const responseId = turnSeq ? `assistant-message-${turnSeq}` : `assistant-message-${Date.now()}`
          upsertMessage({
            id: responseId,
            role: 'assistant',
            content: data.text || data.message || '',
            timestamp: Date.now(),
          })
          break
        }

        case 'task_event': {
          handleTaskEvent(data)
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

        case 'visual_input_error':
        case 'visual_input_unsupported':
          console.warn('[CyberVerse]', data.message || data)
          break

        case 'avatar_status': {
          const turnSeq = parseTurnSeq(data)
          if (isOlderTurn(turnSeq)) {
            break
          }
          beginTurnSeq(turnSeq)
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
          content: translate('chat.sendFailedNoSession'),
          timestamp: Date.now(),
        })
        return
      }
      void sendMessage(sid, trimmed).catch((err) => {
        console.error('[useChat] HTTP fallback send failed:', err)
        messages.value.push({
          role: 'system',
          content: translate('chat.sendFailedNetwork'),
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
      const historySessionIds = Array.from(new Set(
        historyMessages
          .map(message => message.sessionId || '')
          .filter(Boolean),
      ))
      // Prepend history before current messages
      messages.value = [...historyMessages, ...messages.value]
      historyNextCursor.value = resp.next_cursor
      historyHasMore.value = resp.has_more
      await Promise.all(historySessionIds.map(sid =>
        recoverTaskEventsForSession(sid, { isHistory: true, orderByTimestamp: true }),
      ))
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
    voiceDrafts.clear()
    lastTaskSeqByTaskId.clear()
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
    sendWSMessage,
  }
}
