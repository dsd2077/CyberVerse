import { OPENAI_VOICE_OPTIONS, QWEN_OMNI_VOICE_OPTIONS, QWEN_TTS_VOICE_OPTIONS, VOICE_OPTIONS } from '../types'
import type { ComposerTranslation } from 'vue-i18n'

export const DEFAULT_OFFICIAL_VOICE = '温柔文雅'
export const DEFAULT_QWEN_TTS_VOICE = 'Momo'
export const DEFAULT_QWEN_OMNI_VOICE = 'Tina'

type VoiceDisplayOption = {
  label: string
  value: string
}

const officialVoiceLabelMap = new Map(
  VOICE_OPTIONS.map(option => [option.value, option.label]),
)
const qwenTTSVoiceLabelMap = new Map(
  QWEN_TTS_VOICE_OPTIONS.map(option => [option.value, option.label]),
)
const qwenOmniVoiceLabelMap = new Map(
  QWEN_OMNI_VOICE_OPTIONS.map(option => [option.value, option.label]),
)
const openAIVoiceLabelMap = new Map(
  OPENAI_VOICE_OPTIONS.map(option => [option.value, option.label]),
)

const officialVoiceEnglishLabelMap = new Map<string, string>([
  ['傲娇女友', 'Tsundere girlfriend'],
  ['冰娇姐姐', 'Cool older sister'],
  ['成熟姐姐', 'Mature older sister'],
  ['可爱女生', 'Cute girl'],
  ['暖心学姐', 'Warm senior student'],
  ['贴心女友', 'Considerate girlfriend'],
  ['温柔文雅', 'Gentle and refined'],
  ['妩媚御姐', 'Charming mature woman'],
  ['性感御姐', 'Sultry mature woman'],
  ['爱气凌人', 'Commanding voice'],
  ['傲娇公子', 'Tsundere young gentleman'],
  ['傲娇精英', 'Tsundere elite'],
  ['傲慢少爷', 'Arrogant young master'],
  ['霸道少爷', 'Dominant young master'],
  ['冰娇白莲', 'Cool and delicate voice'],
  ['不羁青年', 'Free-spirited young man'],
  ['成熟总裁', 'Mature executive'],
  ['磁性男嗓', 'Resonant male voice'],
  ['醋精男友', 'Jealous boyfriend'],
  ['风发少年', 'Energetic young man'],
  ['腹黑公子', 'Cunning young gentleman'],
])

export function isOfficialVoiceType(value: string): boolean {
  return officialVoiceLabelMap.has(value.trim())
}

export function isQwenTTSVoiceType(value: string): boolean {
  return qwenTTSVoiceLabelMap.has(value.trim())
}

export function isQwenOmniVoiceType(value: string): boolean {
  return qwenOmniVoiceLabelMap.has(value.trim())
}

export function isOpenAIVoiceType(value: string): boolean {
  return openAIVoiceLabelMap.has(value.trim())
}

function englishLabelFromCurrentLabel(label: string, value: string): string {
  const officialLabel = officialVoiceEnglishLabelMap.get(value)
  if (officialLabel) return officialLabel
  const match = label.match(/\(([^)]+)\)\s*$/)
  return match?.[1] || value
}

export function localizedVoiceOptions(
  options: VoiceDisplayOption[],
  locale: string,
): VoiceDisplayOption[] {
  const useEnglish = locale.toLowerCase().startsWith('en')
  return options.map((option) => ({
    value: option.value,
    label: useEnglish ? englishLabelFromCurrentLabel(option.label, option.value) : option.label,
  }))
}

export function formatVoiceTypeDisplay(
  value: string,
  t?: ComposerTranslation,
  locale: string = 'zh-CN',
): string {
  const trimmed = value.trim()
  if (!trimmed) return '—'
  const label = qwenTTSVoiceLabelMap.get(trimmed)
    ?? qwenOmniVoiceLabelMap.get(trimmed)
    ?? openAIVoiceLabelMap.get(trimmed)
    ?? officialVoiceLabelMap.get(trimmed)
  if (label) {
    return locale.toLowerCase().startsWith('en') ? englishLabelFromCurrentLabel(label, trimmed) : label
  }
  return t ? t('voices.cloned', { id: trimmed }) : `Cloned voice · ${trimmed}`
}
