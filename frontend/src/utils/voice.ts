import { OPENAI_VOICE_OPTIONS, QWEN_TTS_VOICE_OPTIONS, VOICE_OPTIONS } from '../types'

export const DEFAULT_OFFICIAL_VOICE = '温柔文雅'
export const DEFAULT_QWEN_TTS_VOICE = 'Momo'

const officialVoiceLabelMap = new Map(
  VOICE_OPTIONS.map(option => [option.value, option.label]),
)
const qwenTTSVoiceLabelMap = new Map(
  QWEN_TTS_VOICE_OPTIONS.map(option => [option.value, option.label]),
)
const openAIVoiceLabelMap = new Map(
  OPENAI_VOICE_OPTIONS.map(option => [option.value, option.label]),
)

export function isOfficialVoiceType(value: string): boolean {
  return officialVoiceLabelMap.has(value.trim())
}

export function isQwenTTSVoiceType(value: string): boolean {
  return qwenTTSVoiceLabelMap.has(value.trim())
}

export function isOpenAIVoiceType(value: string): boolean {
  return openAIVoiceLabelMap.has(value.trim())
}

export function formatVoiceTypeDisplay(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) return '—'
  return qwenTTSVoiceLabelMap.get(trimmed)
    ?? openAIVoiceLabelMap.get(trimmed)
    ?? officialVoiceLabelMap.get(trimmed)
    ?? `克隆音色 · ${trimmed}`
}
