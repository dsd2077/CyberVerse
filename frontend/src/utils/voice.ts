import { VOICE_OPTIONS } from '../types'

export const DEFAULT_OFFICIAL_VOICE = '温柔文雅'

const officialVoiceLabelMap = new Map(
  VOICE_OPTIONS.map(option => [option.value, option.label]),
)

export function isOfficialVoiceType(value: string): boolean {
  return officialVoiceLabelMap.has(value.trim())
}

export function formatVoiceTypeDisplay(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) return '—'
  return officialVoiceLabelMap.get(trimmed) ?? `克隆音色 · ${trimmed}`
}
