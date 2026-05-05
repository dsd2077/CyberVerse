<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { getHealth } from '../services/api'
import LanguageSwitcher from './LanguageSwitcher.vue'

const router = useRouter()
const { t } = useI18n()
const search = ref('')
const serviceConnected = ref(false)

onMounted(async () => {
  try {
    const h = await getHealth()
    serviceConnected.value = h.inference_connected
  } catch {
    serviceConnected.value = false
  }
})

withDefaults(defineProps<{
  showBack?: boolean
  breadcrumb?: string[]
  logoTo?: string
}>(), {
  logoTo: '/characters',
})
</script>

<template>
  <header class="h-14 bg-cv-surface border-b border-cv-border-subtle flex items-center px-12 shrink-0">
    <!-- Left -->
    <div class="flex items-center gap-3">
      <button v-if="showBack" @click="router.back()"
              class="text-cv-text-secondary hover:text-cv-text text-sm cursor-pointer transition-colors">
        {{ t('common.back') }}
      </button>
      <span v-if="showBack" class="text-cv-border">|</span>
      <span class="text-lg font-bold text-cv-text tracking-[-0.5px] cursor-pointer" @click="router.push(logoTo)">
        CyberVerse
      </span>
      <LanguageSwitcher />
    </div>

    <!-- Center: Search -->
    <div class="flex-1 flex justify-center" v-if="!breadcrumb">
      <div class="relative w-[280px]">
        <input
          v-model="search"
          type="text"
          :placeholder="t('common.searchCharacters')"
          class="w-full h-9 bg-cv-elevated border border-cv-border rounded-cv-md px-4 pr-8 text-sm text-cv-text placeholder:text-cv-text-muted focus:border-cv-accent focus:outline-none focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)] transition-all"
        />
      </div>
    </div>

    <!-- Center: Breadcrumb -->
    <div class="flex-1 flex justify-center" v-if="breadcrumb">
      <div class="flex items-center gap-2 text-sm">
        <template v-for="(item, i) in breadcrumb" :key="i">
          <span v-if="i < breadcrumb.length - 1"
                class="text-cv-accent cursor-pointer hover:text-cv-accent-hover transition-colors"
                @click="router.push(i === 0 ? '/characters' : '')">
            {{ item }}
          </span>
          <span v-if="i < breadcrumb.length - 1" class="text-cv-text-muted">/</span>
          <span v-if="i === breadcrumb.length - 1" class="text-cv-text-secondary">{{ item }}</span>
        </template>
      </div>
    </div>

    <!-- Right: Status + Settings -->
    <div class="flex items-center gap-4">
      <div class="flex items-center gap-2 text-[13px]">
        <span class="w-2 h-2 rounded-full" :class="serviceConnected ? 'bg-cv-success' : 'bg-cv-danger'" />
        <span class="text-cv-text-secondary">{{ serviceConnected ? t('common.serviceConnected') : t('common.serviceDisconnected') }}</span>
      </div>
      <button type="button"
              :aria-label="t('appHeader.statusSettingsLabel')"
              @click="router.push('/settings')"
              class="w-8 h-8 flex items-center justify-center rounded-cv-md text-cv-text-secondary hover:text-cv-text hover:bg-cv-hover transition-all cursor-pointer">
        <svg class="w-[18px] h-[18px] shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z" />
          <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />
          <path d="M12 2v2" />
          <path d="M12 22v-2" />
          <path d="m17 20.66-1-1.73" />
          <path d="M11 10.27 7 3.34" />
          <path d="m20.66 17-1.73-1" />
          <path d="m3.34 7 1.73 1" />
          <path d="M14 12h8" />
          <path d="M2 12h2" />
          <path d="m20.66 7-1.73 1" />
          <path d="m3.34 17 1.73-1" />
          <path d="m17 3.34-1 1.73" />
          <path d="m11 13.73-4 6.93" />
        </svg>
      </button>
    </div>
  </header>
</template>
