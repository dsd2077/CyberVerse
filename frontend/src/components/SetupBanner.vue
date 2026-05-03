<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const dismissed = ref(false)

defineProps<{ show: boolean }>()
</script>

<template>
  <div v-if="show && !dismissed"
       class="bg-cv-accent-muted border border-cv-accent/30 rounded-cv-lg px-6 py-4 mb-6 flex items-start justify-between gap-4">
    <div>
      <div class="flex items-center gap-2 mb-1">
        <svg class="w-4 h-4 text-cv-warning" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M8 1L1 14h14L8 1z" stroke-linejoin="round" />
          <path d="M8 6v3M8 11.5v.5" stroke-linecap="round" />
        </svg>
        <span class="text-cv-accent text-sm font-medium">系统尚未配置</span>
      </div>
      <p class="text-[13px] text-cv-text-secondary">
        请先配置 DashScope 或所选组件的服务凭证，以便正常启动数字人会话。
      </p>
    </div>
    <div class="flex items-center gap-3 shrink-0">
      <button @click="router.push('/settings')"
              class="px-4 py-1.5 bg-cv-accent text-white text-[13px] font-medium rounded-cv-md hover:bg-cv-accent-hover transition-colors cursor-pointer">
        去配置 →
      </button>
      <button @click="dismissed = true"
              class="text-cv-text-muted hover:text-cv-text transition-colors cursor-pointer">
        <svg class="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M4 4l8 8M12 4l-8 8" stroke-linecap="round" />
        </svg>
      </button>
    </div>
  </div>
</template>
