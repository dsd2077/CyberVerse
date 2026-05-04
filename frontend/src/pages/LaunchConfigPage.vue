<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useCharacterStore } from '../stores/characters'
import { createSession, getAvatarModelInfo, getHealth, getLaunchConfig, updateLaunchConfig } from '../services/api'
import CvSelect from '../components/CvSelect.vue'
import type { AvatarModelInfo, ConfigSection, ConfigParam } from '../types'
import { formatVoiceTypeDisplay } from '../utils/voice'

const router = useRouter()
const route = useRoute()
const store = useCharacterStore()
const characterId = computed(() => route.params.id as string)
const connecting = ref(false)
const serviceConnected = ref(false)

// Config state
const configSections = ref<ConfigSection[]>([])
const originalSections = ref<ConfigSection[]>([])
const loading = ref(true)
const saving = ref(false)
const saveMessage = ref('')
const errorMessage = ref('')
const avatarModelInfo = ref<AvatarModelInfo | null>(null)

const activeAvatarModel = computed(() => avatarModelInfo.value?.active_model || '')
const configuredDefaultModel = computed(() => avatarModelInfo.value?.configured_default_model || '')
const runtimeConfigMismatch = computed(() =>
  !!activeAvatarModel.value &&
  !!configuredDefaultModel.value &&
  activeAvatarModel.value !== configuredDefaultModel.value
)

// Input width auto-sizing (in ch units)
const INPUT_MIN_WIDTH_CH = 16
const INPUT_MAX_WIDTH_CH = 48
const INPUT_PADDING_CH = 0

function inputWidth(value: string | number): string {
  const len = String(value).length + INPUT_PADDING_CH
  return Math.min(Math.max(len, INPUT_MIN_WIDTH_CH), INPUT_MAX_WIDTH_CH) + 'ch'
}

// Deep clone helper
function cloneSections(sections: ConfigSection[]): ConfigSection[] {
  return JSON.parse(JSON.stringify(sections))
}

// Check if there are unsaved changes
const hasChanges = computed(() => {
  if (originalSections.value.length === 0) return false
  return JSON.stringify(configSections.value) !== JSON.stringify(originalSections.value)
})

function sectionHasRestartPending(section: ConfigSection): boolean {
  const orig = originalSections.value.find(s => s.title === section.title)
  if (!orig) return false
  for (const param of section.params) {
    if (!param.requires_restart) continue
    const origParam = orig.params.find((p: ConfigParam) => p.path === param.path)
    if (origParam && origParam.value !== param.value) return true
  }
  return false
}

const restartBadgeHint =
  '保存当前配置，然后重启 inference 推理服务，使配置生效'

onMounted(async () => {
  await store.fetchOne(characterId.value).catch(() => {})
  try {
    const health = await getHealth()
    serviceConnected.value = health.inference_connected
  } catch {
    serviceConnected.value = false
  }

  // Fetch real config
  try {
    avatarModelInfo.value = await getAvatarModelInfo()
    const config = await getLaunchConfig()
    configSections.value = config.sections.map(s => ({ ...s, collapsed: false }))
    originalSections.value = cloneSections(configSections.value)
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : '加载配置失败'
    console.error('Failed to load launch config:', e)
  } finally {
    loading.value = false
  }
})

async function saveConfig() {
  saving.value = true
  saveMessage.value = ''
  errorMessage.value = ''

  // Collect changed non-readonly params
  const changedParams: Array<{ path: string; value: string | number }> = []
  for (const section of configSections.value) {
    for (const param of section.params) {
      if (param.readonly) continue
      // Find original value
      const origSection = originalSections.value.find(s => s.title === section.title)
      const origParam = origSection?.params.find(p => p.path === param.path)
      if (origParam && origParam.value !== param.value) {
        changedParams.push({ path: param.path, value: param.value })
      }
    }
  }

  if (changedParams.length === 0) {
    saving.value = false
    return
  }

  try {
    const resp = await updateLaunchConfig({
      model: activeAvatarModel.value,
      params: changedParams,
    })
    originalSections.value = cloneSections(configSections.value)
    if (resp.requires_restart) {
      saveMessage.value = '配置已保存，需要重启推理服务才能生效'
    } else {
      saveMessage.value = '配置已保存'
    }
  } catch (e) {
    errorMessage.value = '保存配置失败'
    console.error('Failed to save launch config:', e)
  } finally {
    saving.value = false
  }
}

async function launch() {
  if (!activeAvatarModel.value) return
  connecting.value = true
  try {
    const launchMode = store.current?.mode || 'standard'
    const resp = await createSession(characterId.value, launchMode)
    resp.warnings?.forEach((warning) => {
      console.warn('[CyberVerse]', warning)
    })
    router.push({
      path: `/session/${resp.session_id}`,
      query: {
        streaming_mode: resp.streaming_mode || 'direct',
        mode: resp.mode || launchMode,
        livekit_url: resp.livekit_url,
        livekit_token: resp.livekit_token,
        idle_video_url: resp.idle_video_url,
        idle_video_urls: resp.idle_video_urls ? JSON.stringify(resp.idle_video_urls) : undefined,
        visual_input: resp.visual_input ? JSON.stringify(resp.visual_input) : undefined,
        character_id: characterId.value,
      },
    })
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : '启动失败'
    console.error('Failed to launch:', e)
  } finally {
    connecting.value = false
  }
}

</script>

<template>
  <div class="min-h-screen bg-cyber-base text-cyber-text">
    <!-- Header -->
    <header class="h-16 bg-[rgba(6,7,10,0.95)] border-b border-white/5 flex items-center justify-between px-8">
      <div class="flex items-center gap-4">
        <button @click="router.back()"
                class="h-[34px] px-3 bg-[#0f1218] border border-[rgba(72,80,92,0.4)] text-[#969eaa] text-[13px] hover:border-cyber-cyan/40 transition-colors cursor-pointer">
          ← 返回
        </button>
        <span class="text-sm font-bold tracking-[1.68px] uppercase text-[#f6efe8]">CyberVerse</span>
      </div>
      <span class="text-[13px] text-[#505864]">角色列表 / 部署配置</span>
      <div class="flex items-center gap-2">
        <span
          class="w-[7px] h-[7px] rounded-full"
          :class="serviceConnected ? 'bg-cyber-cyan shadow-[0_0_6px_rgba(52,230,243,0.5)]' : 'bg-[#ff6b6b] shadow-[0_0_6px_rgba(255,107,107,0.35)]'"
        />
        <span class="text-[13px]" :class="serviceConnected ? 'text-[#8fe8ef]' : 'text-[#ff9b9b]'">
          {{ serviceConnected ? '推理服务已连接' : '推理服务未连接' }}
        </span>
      </div>
    </header>

    <div class="flex">
      <!-- Left sidebar: Character summary -->
      <aside class="w-[380px] shrink-0 p-8 pt-10 ml-16">
        <div v-if="store.current" class="bg-cv-surface border border-cv-border rounded-cv-lg overflow-hidden shadow-[0_4px_16px_-2px_rgba(0,0,0,0.25)]">
          <!-- Avatar image -->
          <div class="h-[220px] bg-gradient-to-b from-[#142659] to-[#2e1a66] rounded-t-cv-lg ">
            <img v-if="store.current.avatar_image" :src="store.current.avatar_image" class="w-full h-full object-cover" :alt="store.current.name" />
          </div>

          <div class="p-5 flex flex-col gap-4">
            <h2 class="text-[22px] font-bold text-cv-text tracking-[-0.2px]">{{ store.current.name }}</h2>
            <p class="text-sm text-[#80808c] leading-[22px]">{{ store.current.description }}</p>

            <div class="h-px bg-cv-border-subtle" />

            <!-- Info rows -->
            <div class="flex justify-between text-xs font-medium">
              <span class="text-[#80808c]">声线</span>
              <span
                class="max-w-[200px] truncate text-right text-cv-text"
                :title="formatVoiceTypeDisplay(store.current.voice_type)"
              >
                {{ formatVoiceTypeDisplay(store.current.voice_type) }}
              </span>
            </div>
            <div class="flex justify-between text-xs font-medium">
              <span class="text-[#80808c]">说话风格</span>
              <span class="text-cv-text">{{ store.current.speaking_style || '—' }}</span>
            </div>
            <div v-if="store.current.personality" class="flex justify-between text-xs">
              <span class="font-medium text-[#80808c]">性格</span>
              <span class="text-cv-text truncate max-w-[200px]">{{ store.current.personality }}</span>
            </div>
            <div v-if="store.current.welcome_message" class="flex justify-between text-xs">
              <span class="font-medium text-[#80808c]">欢迎语</span>
              <span class="text-cv-text truncate max-w-[200px]">{{ store.current.welcome_message }}</span>
            </div>

            <!-- Role prompt -->
            <div class="bg-[#131317] border border-[#24242b] rounded-cv-md px-3 py-2.5">
              <p class="text-[10px] font-semibold text-cv-text-muted tracking-[0.8px] uppercase mb-1.5">Role Prompt</p>
              <p class="text-xs text-cv-text-secondary leading-[18px] line-clamp-4">{{ store.current.system_prompt }}</p>
            </div>

            <button @click="router.push(`/characters/${characterId}/edit`)"
                    class="text-[13px] font-medium text-[#619ef5] hover:text-cv-accent-hover transition-colors cursor-pointer self-start">
              编辑角色 →
            </button>
          </div>
        </div>
      </aside>

      <!-- Right: Config area -->
      <main class="flex-1 pl-4 pr-12 py-10 flex flex-col gap-7">
        <div>
          <h1 class="text-[28px] font-extrabold text-[#fbf6ef]">部署配置</h1>
          <p class="text-sm text-[#6e7682] mt-2">
            根据您的硬件环境调整推理参数。当前运行模型：{{ activeAvatarModel || '未连接' }}
          </p>
        </div>

        <!-- Loading -->
        <div v-if="loading" class="text-[#6e7682] text-sm py-8">加载配置中...</div>

        <!-- Error -->
        <div v-if="errorMessage" class="text-[13px] text-[#ff9b9b] bg-[rgba(255,107,107,0.08)] border border-[rgba(255,107,107,0.2)] px-4 py-2.5">
          {{ errorMessage }}
        </div>

        <div v-if="runtimeConfigMismatch" class="text-[13px] text-[#ffb36b] bg-[rgba(255,147,70,0.08)] border border-[rgba(255,147,70,0.2)] px-4 py-2.5">
          配置文件默认模型是 {{ configuredDefaultModel }}，但当前推理进程实际运行的是 {{ activeAvatarModel }}。部署页以运行时模型为准；修改默认模型后需要重启推理服务。
        </div>

        <!-- Save message -->
        <div v-if="saveMessage" class="text-[13px] text-[#8fe8ef] bg-[rgba(52,230,243,0.08)] border border-[rgba(52,230,243,0.2)] px-4 py-2.5">
          {{ saveMessage }}
        </div>

        <!-- Config sections -->
        <div v-for="section in configSections" :key="section.title"
             class="bg-cyber-surface border border-white/6 overflow-hidden">
          <!-- Section header -->
          <div class="bg-[#0b0e14] border-b border-white/6 px-6 py-4 flex items-center justify-between">
            <div class="flex items-center gap-3">
              <span class="text-[10px] text-[#505a6e]" @click="section.collapsed = !section.collapsed" style="cursor:pointer">
                {{ section.collapsed ? '▶' : '▼' }}
              </span>
              <span class="text-sm font-bold text-[#c8d0dc]">{{ section.title }}</span>
            </div>
            <div
              v-if="sectionHasRestartPending(section)"
              class="group relative flex items-center gap-1"
            >
              <span class="px-2 py-0.5 text-[11px] bg-[rgba(255,147,70,0.12)] border border-[rgba(255,147,70,0.4)] text-[#ff9346]">
                需重启
              </span>
              <button
                type="button"
                class="inline-flex h-[18px] min-w-[18px] shrink-0 cursor-help items-center justify-center rounded-full border border-[rgba(255,147,70,0.45)] bg-[rgba(255,147,70,0.06)] px-1 text-[11px] font-semibold leading-none text-[#ff9346] outline-none hover:bg-[rgba(255,147,70,0.12)] focus-visible:ring-2 focus-visible:ring-[rgba(255,147,70,0.45)]"
                tabindex="0"
                :aria-label="restartBadgeHint"
              >
                ?
              </button>
              <div
                role="tooltip"
                class="pointer-events-none invisible absolute right-0 top-[calc(100%+6px)] z-30 w-[min(288px,calc(100vw-3rem))] rounded-md border border-white/10 bg-[#12161c] px-3 py-2 text-left text-[11px] leading-relaxed text-[#b8c0cc] shadow-xl opacity-0 transition-opacity duration-150 group-hover:visible group-hover:opacity-100 group-focus-within:visible group-focus-within:opacity-100"
              >
                {{ restartBadgeHint }}
              </div>
            </div>
          </div>

          <!-- Params -->
          <div v-if="!section.collapsed">
            <div v-for="param in section.params" :key="param.name"
                 class="px-6 py-4 border-b border-white/4 flex items-center justify-between">
              <div>
                <p class="text-[13px] font-medium text-[#c8d0dc]">{{ param.name }}</p>
                <p class="text-[11px] text-[#505864]">{{ param.path }}</p>
              </div>
              <div class="flex items-center gap-2.5">
                <!-- Readonly: plain text -->
                <span v-if="param.readonly" class="text-[13px] text-[#a0a8b4] text-right max-w-[300px] truncate">{{ param.value }}</span>
                <!-- Select: use CvSelect for params with options -->
                <CvSelect
                  v-else-if="param.options && param.options.length > 0"
                  :modelValue="String(param.value)"
                  @update:modelValue="param.value = $event"
                  :options="param.options"
                  class="w-[200px]"
                />
                <!-- Number input (no spinner) -->
                <input
                  v-else-if="typeof param.value === 'number'"
                  type="text"
                  inputmode="numeric"
                  :value="param.value"
                  @input="param.value = Number(($event.target as HTMLInputElement).value) || 0"
                  :style="{ width: inputWidth(param.value) }"
                  class="bg-[#0f1218] border border-[rgba(72,80,92,0.4)] text-[#a0a8b4] text-[13px] px-2 py-1 text-right focus:border-cyber-cyan/60 outline-none transition-colors"
                />
                <!-- Text input (auto-width based on content) -->
                <input
                  v-else
                  type="text"
                  v-model="param.value"
                  :style="{ width: inputWidth(param.value) }"
                  class="bg-[#0f1218] border border-[rgba(72,80,92,0.4)] text-[#a0a8b4] text-[13px] px-2 py-1 text-right focus:border-cyber-cyan/60 outline-none transition-colors"
                />
                <!-- Lock icon for readonly params -->
                <span v-if="param.readonly && param.requires_restart" class="text-[12px] text-[#505864]">
                  <svg class="w-3 h-3.5 inline" viewBox="0 0 9 11" fill="none">
                    <ellipse cx="4.5" cy="3.5" rx="3" ry="3" stroke="#73737d" stroke-width="1.2" />
                    <rect x="0" y="5" width="9" height="6" rx="1" fill="#73737d" />
                  </svg>
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Action buttons -->
        <div class="flex justify-end gap-4 mt-4">
          <button @click="saveConfig" :disabled="!hasChanges || saving"
                  class="h-12 px-6 bg-[#0f1218] border border-[rgba(72,80,92,0.4)] text-[#969eaa] text-[14px] font-medium hover:border-cyber-cyan/40 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed">
            {{ saving ? '保存中...' : '保存配置' }}
          </button>
          <button @click="launch" :disabled="connecting || !activeAvatarModel"
                  class="h-12 px-8 bg-gradient-to-b from-cyber-cyan to-[#14a0ac] text-cyber-base text-[15px] font-extrabold shadow-[0_0_20px_rgba(52,230,243,0.2)] hover:shadow-[0_0_30px_rgba(52,230,243,0.35)] transition-shadow cursor-pointer disabled:opacity-50">
            {{ connecting ? '连接中...' : '启动数字人' }}
          </button>
        </div>
      </main>
    </div>
  </div>
</template>
