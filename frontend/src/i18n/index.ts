import { createI18n } from 'vue-i18n'
import { baseMessages } from './messages'

export const SUPPORTED_LOCALES = ['zh-CN', 'en-US'] as const
export type SupportedLocale = (typeof SUPPORTED_LOCALES)[number]

export const LOCALE_STORAGE_KEY = 'cyberverse.locale'

function isSupportedLocale(value: string | null | undefined): value is SupportedLocale {
  return SUPPORTED_LOCALES.includes(value as SupportedLocale)
}

function normalizeLocale(value: string | null | undefined): SupportedLocale {
  if (!value) return 'en-US'
  return value.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en-US'
}

export function detectInitialLocale(): SupportedLocale {
  if (typeof window === 'undefined') return 'en-US'
  const stored = window.localStorage.getItem(LOCALE_STORAGE_KEY)
  if (isSupportedLocale(stored)) return stored
  return normalizeLocale(window.navigator.language)
}

export const i18n = createI18n({
  legacy: false,
  locale: detectInitialLocale(),
  fallbackLocale: 'en-US',
  messages: baseMessages,
})

export function setLocale(locale: SupportedLocale) {
  i18n.global.locale.value = locale
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(LOCALE_STORAGE_KEY, locale)
  }
  document.documentElement.lang = locale
}

export function translate(key: string, named?: Record<string, unknown>): string {
  return i18n.global.t(key, named ?? {})
}

if (typeof document !== 'undefined') {
  document.documentElement.lang = i18n.global.locale.value
}
