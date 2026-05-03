import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { Settings } from '../types'
import * as api from '../services/api'

export const useSettingsStore = defineStore('settings', () => {
  const settings = ref<Settings | null>(null)
  const loading = ref(false)

  const isConfigured = computed(() => {
    if (!settings.value) return false
    const s = settings.value
    return !!(
      s.model_providers?.dashscope_api_key ||
      s.model_providers?.openai_api_key ||
      s.doubao.access_token
    )
  })

  async function fetch() {
    loading.value = true
    try {
      settings.value = await api.getSettings()
    } finally {
      loading.value = false
    }
  }

  async function save(data: Settings) {
    await api.updateSettings(data)
    settings.value = data
  }

  async function testConnection() {
    return api.testConnection()
  }

  return { settings, loading, isConfigured, fetch, save, testConnection }
})
