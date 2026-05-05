<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { setLocale, type SupportedLocale } from '../i18n'

const { locale, t } = useI18n({ useScope: 'global' })

const currentLocale = computed(() => locale.value as SupportedLocale)
const open = ref(false)
const containerRef = ref<HTMLElement | null>(null)

const languageOptions = computed(() => [
  { locale: 'en-US' as const, label: t('locale.en'), flag: '🇺🇸' },
  { locale: 'zh-CN' as const, label: t('locale.zh'), flag: '🇨🇳' },
])

const activeOption = computed(() =>
  languageOptions.value.find(option => option.locale === currentLocale.value) ?? languageOptions.value[0],
)

function switchLocale(next: SupportedLocale) {
  if (next !== currentLocale.value) {
    setLocale(next)
  }
  open.value = false
}

function handleOutsideClick(event: MouseEvent) {
  if (!containerRef.value?.contains(event.target as Node)) {
    open.value = false
  }
}

function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape') {
    open.value = false
  }
}

onMounted(() => {
  document.addEventListener('mousedown', handleOutsideClick)
  document.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  document.removeEventListener('mousedown', handleOutsideClick)
  document.removeEventListener('keydown', handleKeydown)
})
</script>

<template>
  <div
    ref="containerRef"
    class="relative inline-block text-xs"
  >
    <button
      type="button"
      class="inline-flex h-8 min-w-[112px] items-center gap-2 rounded-2xl border border-cv-border bg-cv-elevated px-3 text-cv-text shadow-[0_8px_18px_rgba(0,0,0,0.16)] transition-colors hover:bg-cv-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-cv-accent/70 cursor-pointer"
      :aria-label="t('locale.switchLabel')"
      :aria-expanded="open"
      aria-haspopup="listbox"
      @click="open = !open"
    >
      <span class="grid h-5 w-5 place-items-center rounded-full text-[18px] leading-none">
        {{ activeOption.flag }}
      </span>
      <span class="flex-1 text-left text-[13px] font-medium">
        {{ activeOption.label }}
      </span>
      <svg
        class="h-3.5 w-3.5 shrink-0 text-cv-text-secondary transition-transform"
        :class="{ 'rotate-180': open }"
        viewBox="0 0 16 16"
        fill="none"
        stroke="currentColor"
        stroke-width="1.8"
      >
        <path d="m4 6 4 4 4-4" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </button>

    <Transition
      enter-active-class="transition duration-120 ease-out"
      enter-from-class="opacity-0 translate-y-[-4px]"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition duration-100 ease-in"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 translate-y-[-4px]"
    >
      <div
        v-if="open"
        class="absolute right-0 top-[calc(100%+7px)] z-50 w-[196px] overflow-hidden rounded-2xl border border-cv-border bg-cv-surface py-1.5 shadow-[0_18px_42px_rgba(0,0,0,0.36)]"
        role="listbox"
      >
        <button
          v-for="option in languageOptions"
          :key="option.locale"
          type="button"
          role="option"
          class="flex h-10 w-full items-center gap-2.5 px-4 text-left text-[14px] font-semibold transition-colors cursor-pointer hover:bg-cv-hover"
          :class="option.locale === currentLocale ? 'text-cv-accent' : 'text-cv-text'"
          :aria-selected="option.locale === currentLocale"
          @click="switchLocale(option.locale)"
        >
          <span class="grid h-6 w-6 place-items-center rounded-full text-[21px] leading-none">
            {{ option.flag }}
          </span>
          <span class="min-w-0 flex-1 truncate">{{ option.label }}</span>
          <svg
            v-if="option.locale === currentLocale"
            class="h-4 w-4 shrink-0"
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <path d="M3 8.5 6.5 12 13 5" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
      </div>
    </Transition>
  </div>
</template>
