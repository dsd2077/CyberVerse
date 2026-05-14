<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import type { Character } from '../types'
import { resolveApiUrl } from '../services/api'
import { formatVoiceTypeDisplay } from '../utils/voice'

const router = useRouter()
const { t, locale } = useI18n()
const props = defineProps<{ character: Character }>()
const emit = defineEmits<{ delete: [id: string] }>()
const coverImage = computed(() =>
  resolveApiUrl(props.character.active_image
    ? `/api/v1/characters/${props.character.id}/images/${encodeURIComponent(props.character.active_image)}`
    : props.character.avatar_image)
)

// Generate a gradient from character name hash
function nameToGradient(name: string): string {
  let hash = 0
  for (const ch of name) hash = ch.charCodeAt(0) + ((hash << 5) - hash)
  const h1 = Math.abs(hash % 360)
  const h2 = (h1 + 40) % 360
  return `linear-gradient(135deg, hsl(${h1}, 55%, 25%), hsl(${h2}, 45%, 18%))`
}

function launch() {
  router.push(`/launch/${props.character.id}`)
}

function edit() {
  router.push(`/characters/${props.character.id}/edit`)
}
</script>

<template>
  <div class="group relative min-h-[340px] flex flex-col justify-end bg-cv-surface border border-cv-border rounded-cv-lg overflow-hidden hover:border-cv-accent hover:shadow-[0_0_20px_rgba(59,130,246,0.15)] hover:-translate-y-0.5 transition-all duration-200 cursor-pointer"
       @click="edit">
    <!-- Avatar image -->
    <div class="absolute inset-0">
      <div class="w-full h-full" :style="{ background: coverImage ? undefined : nameToGradient(character.name) }">
        <img v-if="coverImage" :src="coverImage" :alt="character.name" class="w-full h-full object-cover object-top" />
      </div>
    </div>

    <!-- Hover actions -->
    <div class="absolute top-3 right-3 z-20 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
      <button @click.stop="edit"
              class="w-7 h-7 flex items-center justify-center rounded-cv-sm bg-black/60 text-white/80 hover:bg-black/80 text-xs backdrop-blur-sm cursor-pointer">
        <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M11.5 1.5l3 3L5 14H2v-3L11.5 1.5z" />
        </svg>
      </button>
      <button @click.stop="emit('delete', character.id)"
              class="w-7 h-7 flex items-center justify-center rounded-cv-sm bg-black/60 text-red-400 hover:bg-red-900/60 text-xs backdrop-blur-sm cursor-pointer">
        <svg class="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M2 4h12M5 4V2h6v2M6 7v5M10 7v5M3 4l1 10h8l1-10" />
        </svg>
      </button>
    </div>

    <!-- Content -->
    <div class="relative z-10 px-4 pb-4 pt-0 overflow-hidden">
      <div class="absolute inset-0 bg-gradient-to-t from-cv-surface/70 via-cv-surface/40 to-transparent backdrop-blur-[2px]" />
      <div class="relative z-10">
        <h3 class="text-base font-semibold text-cv-text drop-shadow-[0_1px_2px_rgba(0,0,0,0.75)]">{{ character.name }}</h3>
        <p class="mt-1 text-[13px] text-cv-text/75 leading-5 line-clamp-2 drop-shadow-[0_1px_2px_rgba(0,0,0,0.75)]">
          {{ character.description || t('characterCard.noDescription') }}
        </p>

        <!-- Divider -->
        <div class="my-3 h-px bg-black/35" />

        <!-- Footer -->
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-cv-success" />
            <span
              class="max-w-[165px] truncate text-[11px] text-cv-text/60 drop-shadow-[0_1px_2px_rgba(0,0,0,0.75)]"
              :title="formatVoiceTypeDisplay(character.voice_type, t, locale)"
            >
              {{ formatVoiceTypeDisplay(character.voice_type, t, locale) }}
            </span>
          </div>
          <button @click.stop="launch"
                  class="px-4 py-1.5 bg-cv-accent text-white text-[13px] font-medium rounded-cv-md hover:bg-cv-accent-hover transition-colors cursor-pointer">
            {{ t('characterCard.launch') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
