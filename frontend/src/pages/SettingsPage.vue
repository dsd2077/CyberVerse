<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import AppHeader from '../components/AppHeader.vue'
import { useSettingsStore } from '../stores/settings'
import type { Settings } from '../types'

const router = useRouter()
const store = useSettingsStore()
const saving = ref(false)
const testing = ref(false)
const testResult = ref<string | null>(null)

type LegacySettings = Partial<Settings> & {
  llm?: { api_key?: string }
}

function defaultSettings(): Settings {
  return {
    doubao: { access_token: '', app_id: '' },
    livekit: { url: '', api_key: '', api_secret: '' },
    model_providers: {
      dashscope_api_key: '',
      openai_api_key: '',
    },
    inference: { grpc_addr: 'localhost:50051' },
  }
}

function normalizeSettings(data?: LegacySettings): Settings {
  const defaults = defaultSettings()
  return {
    doubao: { ...defaults.doubao, ...data?.doubao },
    livekit: { ...defaults.livekit, ...data?.livekit },
    model_providers: {
      dashscope_api_key: data?.model_providers?.dashscope_api_key || '',
      openai_api_key: data?.model_providers?.openai_api_key || data?.llm?.api_key || '',
    },
    inference: { ...defaults.inference, ...data?.inference },
  }
}

const form = ref<Settings>(defaultSettings())

// Password visibility toggles
const showTokens = ref<Record<string, boolean>>({})

function toggleShow(key: string) {
  showTokens.value[key] = !showTokens.value[key]
}

onMounted(async () => {
  await store.fetch().catch(() => {})
  if (store.settings) {
    form.value = normalizeSettings(JSON.parse(JSON.stringify(store.settings)))
  }
})

async function save() {
  saving.value = true
  try {
    await store.save(form.value)
    router.push('/characters')
  } catch (e) {
    console.error('Save failed:', e)
  } finally {
    saving.value = false
  }
}

async function test() {
  testing.value = true
  testResult.value = null
  try {
    const res = await store.testConnection()
    testResult.value = res.status === 'ok' ? '连接成功' : '连接失败'
  } catch {
    testResult.value = '连接失败'
  } finally {
    testing.value = false
  }
}
</script>

<template>
  <div class="min-h-screen bg-cv-base">
    <AppHeader showBack :breadcrumb="['角色列表', '系统设置']" />

    <main class="max-w-[800px] mx-auto px-8 py-10">
      <h1 class="text-xl font-semibold text-cv-text mb-1">系统设置</h1>
      <p class="text-[13px] text-cv-text-muted mb-8">配置服务凭证，角色可在组件配置中选择已启用的模型组件</p>

      <div class="flex flex-col gap-6">
        <!-- Doubao -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <h3 class="text-sm font-semibold text-cv-text mb-4">豆包语音</h3>
          <label class="block mb-3">
            <span class="text-[13px] text-cv-text-secondary">Access Token</span>
            <div class="relative mt-1.5">
              <input v-model="form.doubao.access_token" :type="showTokens['doubao_token'] ? 'text' : 'password'"
                     class="w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 pr-10 text-sm text-cv-text focus:border-cv-accent focus:outline-none transition-all" />
              <button @click="toggleShow('doubao_token')" class="absolute right-3 top-1/2 -translate-y-1/2 text-cv-text-muted hover:text-cv-text cursor-pointer text-xs">
                {{ showTokens['doubao_token'] ? '隐藏' : '显示' }}
              </button>
            </div>
          </label>
          <label class="block mb-3">
            <span class="text-[13px] text-cv-text-secondary">App ID</span>
            <input v-model="form.doubao.app_id" class="mt-1.5 w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text focus:border-cv-accent focus:outline-none transition-all" />
          </label>
        </section>

        <!-- LiveKit -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <h3 class="text-sm font-semibold text-cv-text mb-4">LiveKit (WebRTC)</h3>
          <label class="block mb-3">
            <span class="text-[13px] text-cv-text-secondary">URL</span>
            <input v-model="form.livekit.url" placeholder="wss://your-livekit-server.com"
                   class="mt-1.5 w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text placeholder:text-cv-text-muted focus:border-cv-accent focus:outline-none transition-all" />
          </label>
          <div class="grid grid-cols-2 gap-4">
            <label class="block">
              <span class="text-[13px] text-cv-text-secondary">API Key</span>
              <input v-model="form.livekit.api_key" class="mt-1.5 w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text focus:border-cv-accent focus:outline-none transition-all" />
            </label>
            <label class="block">
              <span class="text-[13px] text-cv-text-secondary">API Secret</span>
              <div class="relative mt-1.5">
                <input v-model="form.livekit.api_secret" :type="showTokens['lk_secret'] ? 'text' : 'password'"
                       class="w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 pr-10 text-sm text-cv-text focus:border-cv-accent focus:outline-none transition-all" />
                <button @click="toggleShow('lk_secret')" class="absolute right-3 top-1/2 -translate-y-1/2 text-cv-text-muted hover:text-cv-text cursor-pointer text-xs">
                  {{ showTokens['lk_secret'] ? '隐藏' : '显示' }}
                </button>
              </div>
            </label>
          </div>
        </section>

        <!-- Qwen / DashScope -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <h3 class="text-sm font-semibold text-cv-text mb-4">Qwen / DashScope</h3>
          <label class="block">
            <span class="text-[13px] text-cv-text-secondary">API Key</span>
            <div class="relative mt-1.5">
              <input
                v-model="form.model_providers.dashscope_api_key"
                :type="showTokens['dashscope_key'] ? 'text' : 'password'"
                placeholder="sk-..."
                class="w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 pr-10 text-sm text-cv-text placeholder:text-cv-text-muted focus:border-cv-accent focus:outline-none transition-all"
              />
              <button @click="toggleShow('dashscope_key')" class="absolute right-3 top-1/2 -translate-y-1/2 text-cv-text-muted hover:text-cv-text cursor-pointer text-xs">
                {{ showTokens['dashscope_key'] ? '隐藏' : '显示' }}
              </button>
            </div>
          </label>
        </section>

        <!-- OpenAI -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <h3 class="text-sm font-semibold text-cv-text mb-4">OpenAI</h3>
          <label class="block">
            <span class="text-[13px] text-cv-text-secondary">API Key</span>
            <div class="relative mt-1.5">
              <input
                v-model="form.model_providers.openai_api_key"
                :type="showTokens['openai_key'] ? 'text' : 'password'"
                placeholder="sk-..."
                class="w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 pr-10 text-sm text-cv-text placeholder:text-cv-text-muted focus:border-cv-accent focus:outline-none transition-all"
              />
              <button @click="toggleShow('openai_key')" class="absolute right-3 top-1/2 -translate-y-1/2 text-cv-text-muted hover:text-cv-text cursor-pointer text-xs">
                {{ showTokens['openai_key'] ? '隐藏' : '显示' }}
              </button>
            </div>
          </label>
        </section>

        <!-- Inference -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <h3 class="text-sm font-semibold text-cv-text mb-4">推理服务连接</h3>
          <label class="block">
            <span class="text-[13px] text-cv-text-secondary">gRPC 地址</span>
            <input v-model="form.inference.grpc_addr" placeholder="localhost:50051"
                   class="mt-1.5 w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text placeholder:text-cv-text-muted focus:border-cv-accent focus:outline-none transition-all" />
          </label>
        </section>

        <!-- Actions -->
        <div class="flex items-center justify-end gap-3 pb-4">
          <span v-if="testResult" class="text-sm" :class="testResult === '连接成功' ? 'text-cv-success' : 'text-cv-danger'">{{ testResult }}</span>
          <button @click="test" :disabled="testing"
                  class="px-5 py-2.5 border border-cv-border text-cv-text-secondary text-sm rounded-cv-md hover:bg-cv-hover hover:text-cv-text transition-all cursor-pointer disabled:opacity-40">
            {{ testing ? '测试中...' : '测试连接' }}
          </button>
          <button @click="save" :disabled="saving"
                  class="px-6 py-2.5 bg-cv-accent text-white text-sm font-medium rounded-cv-md hover:bg-cv-accent-hover transition-colors cursor-pointer disabled:opacity-40 shadow-[0_2px_8px_rgba(59,130,246,0.3)]">
            {{ saving ? '保存中...' : '保存' }}
          </button>
        </div>
      </div>
    </main>
  </div>
</template>
