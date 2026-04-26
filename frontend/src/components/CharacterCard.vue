<script setup lang="ts">
import { useRouter } from 'vue-router'
import type { Character } from '../types'
import { formatVoiceTypeDisplay } from '../utils/voice'

const router = useRouter()
const props = defineProps<{ character: Character }>()
const emit = defineEmits<{ delete: [id: string] }>()

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
  <div class="group bg-cv-surface border border-cv-border rounded-cv-lg overflow-hidden hover:border-cv-accent hover:shadow-[0_0_20px_rgba(59,130,246,0.15)] hover:-translate-y-0.5 transition-all duration-200 cursor-pointer"
       @click="edit">
    <!-- Avatar area -->
    <div class="relative h-[180px] overflow-hidden">
      <div class="w-full h-full" :style="{ background: character.avatar_image ? undefined : nameToGradient(character.name) }">
        <img v-if="character.avatar_image" :src="character.avatar_image" class="w-full h-full object-cover" />
      </div>
      <!-- Hover actions -->
      <div class="absolute top-3 right-3 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
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
      <!-- Bottom gradient overlay -->
      <div class="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-cv-surface to-transparent" />
    </div>

    <!-- Content -->
    <div class="p-4">
      <h3 class="text-base font-semibold text-cv-text">{{ character.name }}</h3>
      <p class="mt-1 text-[13px] text-cv-text-secondary leading-5 line-clamp-2">
        {{ character.description || '暂无描述' }}
      </p>

      <!-- Divider -->
      <div class="my-3 h-px bg-cv-border-subtle" />

      <!-- Footer -->
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-1.5">
          <span class="w-1.5 h-1.5 rounded-full bg-cv-success" />
          <span
            class="max-w-[165px] truncate text-[11px] text-cv-text-muted"
            :title="formatVoiceTypeDisplay(character.voice_type)"
          >
            {{ formatVoiceTypeDisplay(character.voice_type) }}
          </span>
        </div>
        <button @click.stop="launch"
                class="px-4 py-1.5 bg-cv-accent text-white text-[13px] font-medium rounded-cv-md hover:bg-cv-accent-hover transition-colors cursor-pointer">
          启动 →
        </button>
      </div>
    </div>
  </div>
</template>
