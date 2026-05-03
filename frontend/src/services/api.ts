import type { AvatarModelInfo, Character, CharacterForm, ComponentsResponse, ImageInfo, Settings, LaunchConfig, LaunchConfigUpdate } from '../types'

const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'

// ── Helpers ──

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) {
    let message = `API error ${res.status}: ${path}`
    try {
      const data = await res.clone().json() as { error?: string }
      if (data?.error) message = data.error
    } catch {
      try {
        const text = await res.text()
        if (text) message = text
      } catch {
        // keep default message
      }
    }
    throw new Error(message)
  }
  if (res.status === 204) return undefined as T
  return res.json()
}

// ── Sessions (existing) ──

export interface CreateSessionResponse {
  session_id: string
  streaming_mode: string  // "direct" or "livekit"
  livekit_url?: string
  livekit_token?: string
  idle_video_url?: string
  idle_video_urls?: string[]
  warnings?: string[]
}

export interface SessionInfo {
  id: string
  state: string
}

export interface HealthResponse {
  status: string
  sessions: number
  inference_connected: boolean
  error?: string
}

export async function createSession(characterId: string, mode: string = 'standard'): Promise<CreateSessionResponse> {
  return request('/sessions', {
    method: 'POST',
    body: JSON.stringify({ character_id: characterId, mode }),
  })
}

export async function getComponents(): Promise<ComponentsResponse> {
  return request('/components')
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}`, { method: 'DELETE' })
  if (!res.ok && res.status !== 404) throw new Error(`Failed to delete session: ${res.status}`)
}

export async function sendMessage(sessionId: string, text: string): Promise<void> {
  return request(`/sessions/${sessionId}/message`, {
    method: 'POST',
    body: JSON.stringify({ text }),
  })
}

export async function listSessions(): Promise<SessionInfo[]> {
  return request('/sessions')
}

export async function getHealth(): Promise<HealthResponse> {
  return request('/health')
}

// ── Conversation History ──

export interface ConversationMessagesResponse {
  messages: { role: string; content: string; timestamp: string; session_id: string }[]
  next_cursor: string
  has_more: boolean
}

export async function getConversationMessages(
  characterId: string,
  limit: number = 50,
  before?: string,
): Promise<ConversationMessagesResponse> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (before) params.set('before', before)
  return request(`/characters/${characterId}/conversations/messages?${params}`)
}

// ── Characters ──

export async function getCharacters(): Promise<Character[]> {
  return request('/characters')
}

export async function getCharacter(id: string): Promise<Character> {
  return request(`/characters/${id}`)
}

export async function createCharacter(data: CharacterForm): Promise<Character> {
  return request('/characters', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export async function updateCharacter(id: string, data: CharacterForm): Promise<Character> {
  return request(`/characters/${id}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  })
}

export async function deleteCharacter(id: string): Promise<void> {
  return request(`/characters/${id}`, { method: 'DELETE' })
}

export async function testCharacterVoice(data: { voice_provider: string; voice_type: string }): Promise<{ status: string }> {
  return request('/characters/test-voice', {
    method: 'POST',
    body: JSON.stringify(data),
  })
}

export async function uploadAvatar(id: string, file: File): Promise<{ path: string }> {
  const formData = new FormData()
  formData.append('avatar', file)
  const res = await fetch(`${API_BASE}/characters/${id}/avatar`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) throw new Error(`Failed to upload avatar: ${res.status}`)
  return res.json()
}

// ── Character Images ──

export async function getCharacterImages(id: string): Promise<ImageInfo[]> {
  return request(`/characters/${id}/images`)
}

export async function deleteCharacterImage(id: string, filename: string): Promise<void> {
  const res = await fetch(`${API_BASE}/characters/${id}/images/${filename}`, { method: 'DELETE' })
  if (!res.ok && res.status !== 404) throw new Error(`Failed to delete image: ${res.status}`)
}

export async function activateCharacterImage(id: string, filename: string): Promise<void> {
  const res = await fetch(`${API_BASE}/characters/${id}/images/${filename}/activate`, { method: 'PUT' })
  if (!res.ok) throw new Error(`Failed to activate image: ${res.status}`)
}

// ── Settings ──

export async function getSettings(): Promise<Settings> {
  return request('/settings')
}

export async function updateSettings(data: Settings): Promise<void> {
  return request('/settings', {
    method: 'PUT',
    body: JSON.stringify(data),
  })
}

export async function testConnection(): Promise<{ status: string }> {
  return request('/settings/test', { method: 'POST' })
}

// ── Launch Config ──

export async function getAvatarModelInfo(): Promise<AvatarModelInfo> {
  return request('/config/avatar-model')
}

export async function getLaunchConfig(model?: string): Promise<LaunchConfig> {
  const qs = model ? `?model=${encodeURIComponent(model)}` : ''
  return request(`/config/launch${qs}`)
}

export async function updateLaunchConfig(data: LaunchConfigUpdate): Promise<{ status: string; requires_restart: boolean }> {
  return request('/config/launch', {
    method: 'PUT',
    body: JSON.stringify(data),
  })
}
