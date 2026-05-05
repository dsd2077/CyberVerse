<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import AppHeader from '../components/AppHeader.vue'
import CharacterCard from '../components/CharacterCard.vue'
import SetupBanner from '../components/SetupBanner.vue'
import { useCharacterStore } from '../stores/characters'
import { useSettingsStore } from '../stores/settings'

const router = useRouter()
const { t } = useI18n()
const characterStore = useCharacterStore()
const settingsStore = useSettingsStore()
const search = ref('')

onMounted(async () => {
  await Promise.all([
    characterStore.fetchAll().catch(() => {}),
    settingsStore.fetch().catch(() => {}),
  ])
})

const filtered = computed(() => {
  if (!search.value) return characterStore.characters
  const q = search.value.toLowerCase()
  return characterStore.characters.filter(c =>
    c.name.toLowerCase().includes(q) || c.description.toLowerCase().includes(q)
  )
})

async function handleDelete(id: string) {
  if (confirm(t('characterList.deleteConfirm'))) {
    await characterStore.remove(id)
  }
}
</script>

<template>
  <div class="min-h-screen bg-cv-base">
    <AppHeader logo-to="/" />

    <main class="max-w-[1200px] mx-auto px-12 py-12">
      <!-- Setup banner -->
      <SetupBanner :show="!settingsStore.isConfigured" />

      <!-- Title area -->
      <div class="flex items-start justify-between mb-8">
        <div>
          <h1 class="text-[32px] font-semibold text-cv-text tracking-[-0.5px]">{{ t('characterList.title') }}</h1>
          <p class="mt-2 text-sm text-cv-text-secondary">{{ t('characterList.subtitle') }}</p>
        </div>
        <button @click="router.push('/characters/new')"
                class="px-5 py-2.5 bg-cv-accent text-white text-sm font-medium rounded-cv-md hover:bg-cv-accent-hover transition-colors cursor-pointer">
          {{ t('characterList.create') }}
        </button>
      </div>

      <!-- Loading -->
      <div v-if="characterStore.loading" class="text-center py-20 text-cv-text-muted">
        {{ t('common.loading') }}
      </div>

      <!-- Character grid -->
      <div v-else-if="filtered.length > 0"
           class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
        <CharacterCard
          v-for="char in filtered"
          :key="char.id"
          :character="char"
          @delete="handleDelete"
        />
      </div>

      <!-- Empty state -->
      <div v-else class="flex flex-col items-center justify-center py-24">
        <div class="w-16 h-16 rounded-full bg-cv-elevated flex items-center justify-center mb-6">
          <svg class="w-8 h-8 text-cv-text-muted" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <circle cx="12" cy="8" r="4" />
            <path d="M6 20v-2a4 4 0 014-4h4a4 4 0 014 4v2" />
            <path d="M16 4l2 2M18 4l-2 2" stroke-linecap="round" />
          </svg>
        </div>
        <h2 class="text-xl font-semibold text-cv-text-secondary mb-2">{{ t('characterList.emptyTitle') }}</h2>
        <p class="text-sm text-cv-text-muted mb-6 text-center">
          <span v-html="t('characterList.emptyBody')" />
        </p>
        <button @click="router.push('/characters/new')"
                class="px-7 py-3 bg-cv-accent text-white text-sm font-medium rounded-cv-md hover:bg-cv-accent-hover transition-colors cursor-pointer">
          {{ t('characterList.createFirst') }}
        </button>
      </div>
    </main>
  </div>
</template>
