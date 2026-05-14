<script setup lang="ts">
import { ref, computed, nextTick, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import AppHeader from '../components/AppHeader.vue'
import AvatarUpload from '../components/AvatarUpload.vue'
import CvSelect from '../components/CvSelect.vue'
import KnowledgeSourceManager from '../components/KnowledgeSourceManager.vue'
import { useCharacterStore } from '../stores/characters'
import type { CharacterComponents, CharacterForm, ComponentOption, ComponentsResponse, ImageInfo } from '../types'
import { OPENAI_VOICE_OPTIONS, QWEN_OMNI_VOICE_OPTIONS, QWEN_TTS_VOICE_OPTIONS, VOICE_OPTIONS } from '../types'
import { uploadAvatar, getCharacterImages, deleteCharacterImage, activateCharacterImage, testCharacterVoice, getComponents } from '../services/api'
import { DEFAULT_OFFICIAL_VOICE, DEFAULT_QWEN_OMNI_VOICE, DEFAULT_QWEN_TTS_VOICE, isOfficialVoiceType, isOpenAIVoiceType, isQwenOmniVoiceType, isQwenTTSVoiceType, localizedVoiceOptions } from '../utils/voice'

const router = useRouter()
const route = useRoute()
const store = useCharacterStore()
const { t, locale } = useI18n()

const isEdit = computed(() => !!route.params.id)
const characterId = computed(() => route.params.id as string)

const DEFAULT_COMPONENTS: CharacterComponents = { llm: 'qwen', asr: 'qwen', tts: 'qwen' }

const form = ref<CharacterForm>({
  name: '',
  description: '',
  avatar_image: '',
  use_face_crop: false,
  image_mode: 'fixed',
  mode: 'standard',
  voice_provider: 'qwen',
  voice_type: DEFAULT_QWEN_TTS_VOICE,
  components: { ...DEFAULT_COMPONENTS },
  speaking_style: '',
  personality: '',
  welcome_message: '',
  system_prompt: '',
  tags: [],
})

const saving = ref(false)
const pendingFiles = ref<File[]>([])
const images = ref<ImageInfo[]>([])
const deletedImageFilenames = ref<Set<string>>(new Set())
const voiceMode = ref<'official' | 'custom'>('official')
const customVoiceType = ref('')
const voiceError = ref('')
const testingVoice = ref(false)
const voiceTestStatus = ref<'success' | 'error' | null>(null)
const voiceTestMessage = ref('')
const showModeHelp = ref(false)
const hydratingCharacter = ref(false)
const OFFICIAL_VOICE_PREVIEW_URL = 'https://console.volcengine.com/speech/new/experience/call'
const CUSTOM_VOICE_CLONE_URL = 'https://console.volcengine.com/speech/new/experience/clone'
const QWEN_TTS_VOICE_PREVIEW_URL = 'https://help.aliyun.com/zh/model-studio/qwen-tts-realtime'
const QWEN_OMNI_VOICE_LIST_URL = 'https://help.aliyun.com/zh/model-studio/omni-voice-list'
const componentCatalog = ref<ComponentsResponse>({
  llm: [{ id: 'qwen', name: 'Qwen', model: 'qwen3.6-plus', default: true, available: true }],
  asr: [{ id: 'qwen', name: 'Qwen', model: 'qwen3-asr-flash-realtime', default: true, available: true }],
  tts: [{ id: 'qwen', name: 'Qwen', model: 'qwen3-tts-flash-realtime', default: true, available: true }],
})

const visibleImages = computed(() =>
  images.value.filter(img => !deletedImageFilenames.value.has(img.filename))
)

const trimmedCustomVoiceType = computed(() => customVoiceType.value.trim())
const selectedTTS = computed(() => form.value.components?.tts || DEFAULT_COMPONENTS.tts)
const selectedOmniProvider = computed(() => form.value.voice_provider || 'doubao')
const usesDoubaoVoice = computed(() =>
  form.value.mode === 'omni'
    ? selectedOmniProvider.value === 'doubao'
    : selectedTTS.value === 'doubao'
)
const usesQwenOmniVoice = computed(() =>
  form.value.mode === 'omni' && selectedOmniProvider.value === 'qwen_omni'
)
const isOpenAIVoice = computed(() => !usesDoubaoVoice.value && selectedTTS.value === 'openai')
const omniProviderOptions = computed(() => [
  { label: t('settings.doubaoVoice'), value: 'doubao' },
  { label: 'Qwen Omni', value: 'qwen_omni' },
])
const providerSelectOptions = (items: ComponentOption[]) =>
  items.map(item => ({
    label: item.name,
    value: item.id,
  }))
const llmProviderOptions = computed(() => providerSelectOptions(componentCatalog.value.llm))
const asrProviderOptions = computed(() => providerSelectOptions(componentCatalog.value.asr))
const ttsProviderOptions = computed(() => providerSelectOptions(componentCatalog.value.tts))
function selectedComponent(category: 'llm' | 'asr' | 'tts') {
  const selected = form.value.components?.[category] || DEFAULT_COMPONENTS[category]
  return componentCatalog.value[category].find(item => item.id === selected)
}
function modelOptions(category: 'llm' | 'asr' | 'tts') {
  const model = selectedComponent(category)?.model || ''
  return model ? [{ label: model, value: model }] : []
}
const llmModel = computed({
  get: () => selectedComponent('llm')?.model || '',
  set: () => {},
})
const asrModel = computed({
  get: () => selectedComponent('asr')?.model || '',
  set: () => {},
})
const ttsModel = computed({
  get: () => selectedComponent('tts')?.model || '',
  set: () => {},
})
const llmModelOptions = computed(() => modelOptions('llm'))
const asrModelOptions = computed(() => modelOptions('asr'))
const ttsModelOptions = computed(() => modelOptions('tts'))
const qwenTTSVoiceOptions = computed(() => localizedVoiceOptions(QWEN_TTS_VOICE_OPTIONS, locale.value))
const qwenOmniVoiceOptions = computed(() => localizedVoiceOptions(QWEN_OMNI_VOICE_OPTIONS, locale.value))
const officialVoiceOptions = computed(() => localizedVoiceOptions(VOICE_OPTIONS, locale.value))
const openAIVoiceOptions = computed(() => localizedVoiceOptions(OPENAI_VOICE_OPTIONS, locale.value))
const canSave = computed(() =>
  !!form.value.name.trim() && (
    usesDoubaoVoice.value
      ? (voiceMode.value === 'official' || !!trimmedCustomVoiceType.value)
      : !!form.value.voice_type.trim()
  )
)
const canCheckVoice = computed(() =>
  (usesDoubaoVoice.value && (voiceMode.value === 'official' || !!trimmedCustomVoiceType.value))
  || (usesQwenOmniVoice.value && !!form.value.voice_type.trim())
)
const voiceCheckSucceeded = computed(() => voiceTestStatus.value === 'success')

function normalizeComponents(components?: Partial<CharacterComponents>): CharacterComponents {
  return {
    llm: components?.llm || DEFAULT_COMPONENTS.llm,
    asr: components?.asr || DEFAULT_COMPONENTS.asr,
    tts: components?.tts || DEFAULT_COMPONENTS.tts,
  }
}

function defaultVoiceForTTS(tts: string) {
  if (tts === 'openai') return 'nova'
  if (tts === 'doubao') return DEFAULT_OFFICIAL_VOICE
  return DEFAULT_QWEN_TTS_VOICE
}

function defaultVoiceForOmni(provider: string) {
  return provider === 'qwen_omni' ? DEFAULT_QWEN_OMNI_VOICE : DEFAULT_OFFICIAL_VOICE
}

function normalizeOmniProvider(provider: string) {
  return provider === 'qwen_omni' ? 'qwen_omni' : 'doubao'
}

function normalizeMode(mode?: string): CharacterForm['mode'] {
  return mode === 'omni' || mode === 'voice_llm' ? 'omni' : 'standard'
}

function applyTTSVoiceDefault(tts: string, force = false) {
  if (form.value.mode === 'omni') {
    return
  }
  form.value.voice_provider = tts
  const current = form.value.voice_type.trim()
  if (force || !current) {
    form.value.voice_type = defaultVoiceForTTS(tts)
    if (tts === 'doubao') syncVoiceInputs(form.value.voice_type)
    return
  }

  if (tts === 'qwen' && !isQwenTTSVoiceType(current)) {
    form.value.voice_type = DEFAULT_QWEN_TTS_VOICE
  } else if (tts === 'openai' && !isOpenAIVoiceType(current)) {
    form.value.voice_type = 'nova'
  } else if (tts === 'doubao') {
    const looksLikeNonDoubaoVoice = isQwenTTSVoiceType(current) || isOpenAIVoiceType(current) || isQwenOmniVoiceType(current)
    syncVoiceInputs(looksLikeNonDoubaoVoice ? DEFAULT_OFFICIAL_VOICE : current)
  }
}

function applyModeVoiceDefault(force = false) {
  if (form.value.mode !== 'omni') {
    applyTTSVoiceDefault(form.value.components.tts, force)
    return
  }

  form.value.voice_provider = normalizeOmniProvider(form.value.voice_provider)
  const current = form.value.voice_type.trim()
  const provider = form.value.voice_provider

  if (provider === 'qwen_omni') {
    if (force || !current || !isQwenOmniVoiceType(current)) {
      form.value.voice_type = DEFAULT_QWEN_OMNI_VOICE
    }
    voiceMode.value = 'official'
    customVoiceType.value = ''
    return
  }

  const looksLikeNonDoubaoVoice = isQwenTTSVoiceType(current) || isOpenAIVoiceType(current) || isQwenOmniVoiceType(current)
  if (force || !current || looksLikeNonDoubaoVoice) {
    form.value.voice_type = defaultVoiceForOmni(provider)
  }
  syncVoiceInputs(form.value.voice_type)
}

function toggleMode() {
  form.value.mode = form.value.mode === 'standard' ? 'omni' : 'standard'
  showModeHelp.value = false
}

function clearVoiceTestResult() {
  voiceTestStatus.value = null
  voiceTestMessage.value = ''
}

function syncVoiceInputs(voiceType: string) {
  const normalized = voiceType.trim()
  if (normalized && !isOfficialVoiceType(normalized)) {
    voiceMode.value = 'custom'
    customVoiceType.value = normalized
    form.value.voice_type = normalized
    return
  }

  voiceMode.value = 'official'
  customVoiceType.value = ''
  form.value.voice_type = normalized || DEFAULT_OFFICIAL_VOICE
}

function setVoiceMode(mode: 'official' | 'custom') {
  voiceMode.value = mode
  voiceError.value = ''

  if (mode === 'official') {
    if (!isOfficialVoiceType(form.value.voice_type)) {
      form.value.voice_type = DEFAULT_OFFICIAL_VOICE
    }
    return
  }

  if (!isOfficialVoiceType(form.value.voice_type)) {
    customVoiceType.value = form.value.voice_type.trim()
  }
}

function resolveVoiceType() {
  if (usesQwenOmniVoice.value) {
    const voice = form.value.voice_type.trim() || DEFAULT_QWEN_OMNI_VOICE
    form.value.voice_type = voice
    return voice
  }

  if (!usesDoubaoVoice.value) {
    applyTTSVoiceDefault(selectedTTS.value)
    return form.value.voice_type.trim() || defaultVoiceForTTS(selectedTTS.value)
  }

  if (voiceMode.value === 'custom') {
    if (!trimmedCustomVoiceType.value) {
      voiceError.value = t('characterEdit.customSpeakerRequired')
      return null
    }
    return trimmedCustomVoiceType.value
  }

  return form.value.voice_type.trim() || DEFAULT_OFFICIAL_VOICE
}

watch(
  [
    () => form.value.voice_provider,
    () => form.value.voice_type,
    () => form.value.mode,
    () => form.value.components.tts,
    () => voiceMode.value,
    () => customVoiceType.value,
  ],
  () => {
    clearVoiceTestResult()
  }
)

watch(
  () => form.value.components.tts,
  (tts) => {
    if (hydratingCharacter.value) return
    voiceError.value = ''
    applyTTSVoiceDefault(tts, true)
  }
)

watch(
  () => form.value.mode,
  () => {
    if (hydratingCharacter.value) return
    voiceError.value = ''
    applyModeVoiceDefault(true)
  }
)

watch(
  () => form.value.voice_provider,
  () => {
    if (hydratingCharacter.value) return
    if (form.value.mode === 'omni') {
      voiceError.value = ''
      applyModeVoiceDefault(true)
    }
  }
)

onMounted(async () => {
  try {
    componentCatalog.value = await getComponents()
  } catch (e) {
    console.warn('Failed to load components:', e)
  }

  if (isEdit.value) {
    await store.fetchOne(characterId.value)
    if (store.current) {
      const c = store.current
      hydratingCharacter.value = true
      try {
        form.value = {
          name: c.name,
          description: c.description,
          avatar_image: c.avatar_image,
          use_face_crop: c.use_face_crop,
          image_mode: c.image_mode || 'fixed',
          mode: normalizeMode(c.mode),
          voice_provider: c.voice_provider,
          voice_type: c.voice_type,
          components: normalizeComponents(c.components),
          speaking_style: c.speaking_style,
          personality: c.personality,
          welcome_message: c.welcome_message,
          system_prompt: c.system_prompt,
          tags: [...c.tags],
        }
        applyModeVoiceDefault(!form.value.voice_type)
        await nextTick()
      } finally {
        hydratingCharacter.value = false
      }
      await loadImages()
    }
  } else {
    applyModeVoiceDefault(true)
  }
})

async function loadImages() {
  if (!isEdit.value) return
  try {
    images.value = await getCharacterImages(characterId.value)
  } catch {
    images.value = []
  }
}

async function handleFileSelected(file: File, options?: { activate?: boolean }) {
  if (isEdit.value) {
    // Edit mode: upload immediately
    try {
      const existingFilenames = new Set(images.value.map(img => img.filename))
      const uploaded = await uploadAvatar(characterId.value, file)
      await loadImages()
      const uploadedFilename = uploaded.filename || images.value.find(img => !existingFilenames.has(img.filename))?.filename
      if (options?.activate && uploadedFilename) {
        await activateCharacterImage(characterId.value, uploadedFilename)
        await loadImages()
      }
      await store.fetchOne(characterId.value)
      if (store.current) {
        form.value.avatar_image = store.current.avatar_image
      }
    } catch (e) {
      console.error('Upload failed:', e)
    }
  } else {
    // Create mode: queue for upload after save
    pendingFiles.value = [...pendingFiles.value, file]
  }
}

function handleReplacePending(index: number, file: File) {
  pendingFiles.value = pendingFiles.value.map((pendingFile, i) => i === index ? file : pendingFile)
}

function handleDeletePending(index: number) {
  pendingFiles.value = pendingFiles.value.filter((_, i) => i !== index)
}

const activeImage = computed(() => store.current?.active_image)

async function handleActivateImage(filename: string) {
  if (!isEdit.value) return
  try {
    await activateCharacterImage(characterId.value, filename)
    await loadImages()
    await store.fetchOne(characterId.value)
    if (store.current) {
      form.value.avatar_image = store.current.avatar_image
    }
  } catch (e) {
    console.error('Activate image failed:', e)
  }
}

function handleDeleteImage(filename: string) {
  deletedImageFilenames.value = new Set([...deletedImageFilenames.value, filename])
}

async function handleCheckVoice() {
  voiceError.value = ''
  clearVoiceTestResult()

  const voiceType = resolveVoiceType()
  if (!voiceType) return

  testingVoice.value = true
  try {
    await testCharacterVoice({
      voice_provider: form.value.voice_provider.trim(),
      voice_type: voiceType,
    })
    voiceTestStatus.value = 'success'
    voiceTestMessage.value = ''
  } catch (e) {
    voiceTestStatus.value = 'error'
    voiceTestMessage.value = e instanceof Error ? e.message : String(e)
  } finally {
    testingVoice.value = false
  }
}

async function save() {
  if (!form.value.name.trim()) return
  voiceError.value = ''
  saving.value = true
  try {
    const payload = { ...form.value }
    if (payload.avatar_image.startsWith('blob:')) {
      payload.avatar_image = ''
    }

    const voiceType = resolveVoiceType()
    if (!voiceType) {
      return
    }
    payload.voice_type = voiceType
    payload.components = normalizeComponents(payload.components)
    payload.voice_provider = payload.mode === 'omni'
      ? normalizeOmniProvider(payload.voice_provider)
      : payload.components.tts

    let id: string
    if (isEdit.value) {
      await store.update(characterId.value, payload)
      id = characterId.value
    } else {
      const char = await store.create(payload)
      id = char.id
    }

    // Delete images marked for removal
    for (const filename of deletedImageFilenames.value) {
      await deleteCharacterImage(id, filename)
    }

    // Upload all pending files
    for (const file of pendingFiles.value) {
      await uploadAvatar(id, file)
    }

    router.push('/characters')
  } catch (e) {
    console.error('Save failed:', e)
  } finally {
    saving.value = false
  }
}

async function handleDelete() {
  if (!confirm(t('characterEdit.deleteConfirm'))) return
  await store.remove(characterId.value)
  router.push('/characters')
}

const promptLength = computed(() => form.value.system_prompt.length)

const breadcrumb = computed(() =>
  isEdit.value
    ? [t('characterEdit.breadcrumbs.list'), t('characterEdit.breadcrumbs.edit')]
    : [t('characterEdit.breadcrumbs.list'), t('characterEdit.breadcrumbs.create')]
)
</script>

<template>
  <div class="min-h-screen bg-cv-base flex flex-col">
    <AppHeader showBack :breadcrumb="breadcrumb" />

    <!-- Page title -->
    <div class="text-center py-8">
      <h1 class="text-2xl font-bold text-cv-text">{{ isEdit ? t('characterEdit.pageTitleEdit') : t('characterEdit.pageTitleCreate') }}</h1>
    </div>

    <!-- Content -->
    <main class="flex-1 max-w-[1100px] mx-auto w-full px-12 pb-24 flex gap-8">
      <!-- Left column: Avatar -->
      <div class="w-[300px] shrink-0">
        <AvatarUpload
          :use-face-crop="form.use_face_crop"
          :images="visibleImages"
          :character-id="isEdit ? characterId : undefined"
          :pending-files="pendingFiles"
          :active-image="activeImage"
          :image-mode="form.image_mode"
          @update:use-face-crop="v => form.use_face_crop = v"
          @file-selected="handleFileSelected"
          @replace-pending="handleReplacePending"
          @delete-image="handleDeleteImage"
          @delete-pending="handleDeletePending"
          @activate-image="handleActivateImage"
        />

        <!-- Image mode toggle -->
        <div v-if="isEdit && visibleImages.length > 1"
             class="mt-4 bg-cv-surface border border-cv-border rounded-cv-lg p-4">
          <div class="flex items-center justify-between">
            <div>
              <span class="text-[13px] font-medium text-cv-text-secondary">{{ t('characterEdit.randomAvatar') }}</span>
              <p class="text-[11px] text-cv-text-muted mt-1">{{ t('characterEdit.randomAvatarHint') }}</p>
            </div>
            <button @click="form.image_mode = form.image_mode === 'random' ? 'fixed' : 'random'"
                    class="relative w-11 h-6 rounded-full transition-colors cursor-pointer"
                    :class="form.image_mode === 'random' ? 'bg-cv-accent' : 'bg-cv-elevated'">
              <span class="absolute top-0.5 left-0.5 w-5 h-5 rounded-full transition-transform duration-200"
                    :class="form.image_mode === 'random' ? 'translate-x-5 bg-white' : 'translate-x-0 bg-cv-text-muted'" />
            </button>
          </div>
        </div>
      </div>

      <!-- Right column: Form -->
      <div class="flex-1 flex flex-col gap-6">
        <!-- Section 1: Basic info -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <h2 class="text-base font-semibold text-cv-text mb-5">{{ t('characterEdit.basicInfo') }}</h2>

          <label class="block mb-4">
            <span class="text-[13px] font-medium text-cv-text-secondary">{{ t('characterEdit.name') }} <span class="text-cv-danger">*</span></span>
            <input v-model="form.name" type="text" :placeholder="t('characterEdit.namePlaceholder')"
                   class="mt-1.5 w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text placeholder:text-cv-text-muted focus:border-cv-accent focus:outline-none focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)] transition-all" />
          </label>

          <label class="block">
            <span class="text-[13px] font-medium text-cv-text-secondary">{{ t('characterEdit.description') }}</span>
            <textarea v-model="form.description" :placeholder="t('characterEdit.descriptionPlaceholder')"
                      class="mt-1.5 w-full h-20 bg-cv-elevated border border-cv-border rounded-cv-md px-4 py-3 text-sm text-cv-text placeholder:text-cv-text-muted resize-y focus:border-cv-accent focus:outline-none focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)] transition-all" />
          </label>
        </section>

        <!-- Section 2: Component configuration -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <div class="mb-5 flex flex-wrap items-center gap-3">
            <h2 class="text-base font-semibold text-cv-text">{{ t('characterEdit.components') }}</h2>
            <div class="relative flex items-center gap-2">
              <button
                type="button"
                @click="toggleMode"
                class="grid h-9 w-[280px] max-w-[calc(100vw-120px)] grid-cols-2 rounded-cv-md border border-cv-border bg-cv-elevated p-1 text-sm transition-colors cursor-pointer"
                :aria-label="t('characterEdit.modeToggleLabel', { mode: form.mode })"
              >
                <span
                  class="flex items-center justify-center rounded-cv-sm transition-colors"
                  :class="form.mode === 'standard'
                    ? 'bg-cv-accent text-white'
                    : 'text-cv-text-secondary'"
                >
                  standard
                </span>
                <span
                  class="flex items-center justify-center rounded-cv-sm transition-colors"
                  :class="form.mode === 'omni'
                    ? 'bg-cv-accent text-white'
                    : 'text-cv-text-secondary'"
                >
                  {{ t('characterEdit.omniMode') }}
                </span>
              </button>
              <button
                type="button"
                @click="showModeHelp = !showModeHelp"
                class="flex h-7 w-7 items-center justify-center rounded-full border border-cv-border text-sm font-medium text-cv-text-muted transition-colors hover:bg-cv-hover hover:text-cv-text cursor-pointer"
                :aria-expanded="showModeHelp"
                :aria-label="t('characterEdit.modeHelpLabel')"
              >
                ?
              </button>
              <div
                v-if="showModeHelp"
                class="absolute right-0 top-[calc(100%+8px)] z-20 w-[340px] max-w-[calc(100vw-64px)] rounded-cv-md border border-cv-border bg-cv-elevated p-4 shadow-lg"
              >
                <div class="mb-3">
                  <div class="text-[13px] font-semibold text-cv-text">standard</div>
                  <p class="mt-1 text-[12px] leading-5 text-cv-text-secondary">
                    {{ t('characterEdit.standardHelp') }}
                  </p>
                </div>
                <div>
                  <div class="text-[13px] font-semibold text-cv-text">{{ t('characterEdit.omniMode') }}</div>
                  <p class="mt-1 text-[12px] leading-5 text-cv-text-secondary">
                    {{ t('characterEdit.omniHelp') }}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div v-if="form.mode === 'standard'" class="flex flex-col gap-4">
            <div class="grid gap-3 md:grid-cols-[90px_minmax(0,1fr)_minmax(0,1fr)] md:items-end">
              <span class="text-[13px] font-medium text-cv-text-secondary md:pb-3">LLM</span>
              <label class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">Provider</span>
                <CvSelect
                  v-model="form.components.llm"
                  :options="llmProviderOptions"
                  class="mt-1.5"
                />
              </label>
              <label class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">{{ t('common.model') }}</span>
                <CvSelect
                  v-model="llmModel"
                  :options="llmModelOptions"
                  class="mt-1.5"
                />
              </label>
            </div>

            <div class="grid gap-3 md:grid-cols-[90px_minmax(0,1fr)_minmax(0,1fr)] md:items-end">
              <span class="text-[13px] font-medium text-cv-text-secondary md:pb-3">ASR</span>
              <label class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">Provider</span>
                <CvSelect
                  v-model="form.components.asr"
                  :options="asrProviderOptions"
                  class="mt-1.5"
                />
              </label>
              <label class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">{{ t('common.model') }}</span>
                <CvSelect
                  v-model="asrModel"
                  :options="asrModelOptions"
                  class="mt-1.5"
                />
              </label>
            </div>

            <div class="grid gap-3 md:grid-cols-[90px_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)] md:items-start">
              <span class="text-[13px] font-medium text-cv-text-secondary md:pt-[31px]">TTS</span>
              <label class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">Provider</span>
                <CvSelect
                  v-model="form.components.tts"
                  :options="ttsProviderOptions"
                  class="mt-1.5"
                />
              </label>
              <label class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">{{ t('common.model') }}</span>
                <CvSelect
                  v-model="ttsModel"
                  :options="ttsModelOptions"
                  class="mt-1.5"
                />
              </label>
              <template v-if="!usesDoubaoVoice && selectedTTS === 'qwen'">
                <label class="block">
                  <span class="text-[12px] font-medium text-cv-text-muted">{{ t('common.voice') }}</span>
                  <CvSelect
                    v-model="form.voice_type"
                    :options="qwenTTSVoiceOptions"
                    class="mt-1.5"
                  />
                </label>
                <p class="text-[11px] leading-5 text-cv-text-muted md:col-span-3 md:col-start-2 md:-mt-1">
                  {{ t('characterEdit.canPreviewAt') }}
                  <a
                    :href="QWEN_TTS_VOICE_PREVIEW_URL"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="underline underline-offset-2 transition-colors hover:text-cv-text"
                  >
                    {{ t('characterEdit.qwenTTSVoiceList') }}
                  </a>
                  {{ t('characterEdit.previewVoice') }}
                </p>
              </template>
              <label v-else-if="isOpenAIVoice" class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">{{ t('common.voice') }}</span>
                <CvSelect
                  v-model="form.voice_type"
                  :options="openAIVoiceOptions"
                  class="mt-1.5"
                />
              </label>
              <div v-else class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">{{ t('common.voice') }}</span>
                <div class="mt-1.5 grid h-[42px] grid-cols-2 rounded-cv-md border border-cv-border bg-cv-elevated p-1">
                  <button
                    type="button"
                    @click="setVoiceMode('official')"
                    class="rounded-cv-sm px-3 text-sm transition-colors cursor-pointer"
                    :class="voiceMode === 'official'
                      ? 'bg-cv-accent text-white'
                      : 'text-cv-text-secondary hover:bg-cv-hover hover:text-cv-text'"
                  >
                    {{ t('characterEdit.officialVoice') }}
                  </button>
                  <button
                    type="button"
                    @click="setVoiceMode('custom')"
                    class="rounded-cv-sm px-3 text-sm transition-colors cursor-pointer"
                    :class="voiceMode === 'custom'
                      ? 'bg-cv-accent text-white'
                      : 'text-cv-text-secondary hover:bg-cv-hover hover:text-cv-text'"
                  >
                    {{ t('characterEdit.clonedVoice') }}
                  </button>
                </div>
                <div class="mt-3 flex items-start gap-3">
                  <CvSelect
                    v-if="voiceMode === 'official'"
                    v-model="form.voice_type"
                    :options="officialVoiceOptions"
                    :success="voiceCheckSucceeded"
                    class="min-w-0 flex-1"
                  />
                  <div v-else class="relative min-w-0 flex-1">
                    <input
                      v-model="customVoiceType"
                      type="text"
                      :placeholder="t('characterEdit.customSpeakerPlaceholder')"
                      class="h-[42px] w-full bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text placeholder:text-cv-text-muted focus:outline-none transition-all"
                      :class="voiceCheckSucceeded
                        ? 'pr-11 border-cv-success focus:border-cv-success focus:shadow-[0_0_0_2px_rgba(34,197,94,0.15)]'
                        : 'focus:border-cv-accent focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)]'"
                    />
                    <span
                      v-if="voiceCheckSucceeded"
                      class="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-cv-success"
                    >
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <path d="M3.5 8.5l3 3 6-6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
                      </svg>
                    </span>
                  </div>
                  <button
                    type="button"
                    @click="handleCheckVoice"
                    :disabled="testingVoice || !canCheckVoice"
                    :class="{ 'opacity-40 cursor-not-allowed': testingVoice || !canCheckVoice }"
                    class="inline-flex h-[42px] shrink-0 items-center rounded-cv-md border border-cv-border px-4 text-sm text-cv-text-secondary transition-all hover:bg-cv-hover hover:text-cv-text cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {{ t('common.check') }}
                  </button>
                </div>
                <p v-if="voiceError" class="mt-2 text-[11px] text-cv-danger">{{ voiceError }}</p>
                <p
                  v-if="voiceTestStatus === 'error' && voiceTestMessage"
                  class="mt-2 text-[11px] leading-5 text-cv-danger whitespace-pre-wrap break-all"
                >
                  {{ voiceTestMessage }}
                </p>
              </div>
            </div>
          </div>

          <div v-else class="flex flex-col gap-4">
            <div class="grid gap-3 md:grid-cols-[90px_minmax(0,1fr)_minmax(0,1fr)] md:items-end">
              <span class="text-[13px] font-medium text-cv-text-secondary md:pb-3">{{ t('characterEdit.omniModel') }}</span>
              <label class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">Provider</span>
                <CvSelect
                  v-model="form.voice_provider"
                  :options="omniProviderOptions"
                  class="mt-1.5"
                />
              </label>
              <div v-if="usesDoubaoVoice" class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">{{ t('characterEdit.voiceType') }}</span>
                <div class="mt-1.5 grid h-[42px] grid-cols-2 rounded-cv-md border border-cv-border bg-cv-elevated p-1">
                  <button
                    type="button"
                    @click="setVoiceMode('official')"
                    class="rounded-cv-sm px-3 text-sm transition-colors cursor-pointer"
                    :class="voiceMode === 'official'
                      ? 'bg-cv-accent text-white'
                      : 'text-cv-text-secondary hover:bg-cv-hover hover:text-cv-text'"
                  >
                    {{ t('characterEdit.officialVoice') }}
                  </button>
                  <button
                    type="button"
                    @click="setVoiceMode('custom')"
                    class="rounded-cv-sm px-3 text-sm transition-colors cursor-pointer"
                    :class="voiceMode === 'custom'
                      ? 'bg-cv-accent text-white'
                      : 'text-cv-text-secondary hover:bg-cv-hover hover:text-cv-text'"
                  >
                    {{ t('characterEdit.clonedVoice') }}
                  </button>
                </div>
              </div>
              <label v-else class="block">
                <span class="text-[12px] font-medium text-cv-text-muted">{{ t('common.model') }}</span>
                <input
                  type="text"
                  value="qwen3.5-omni-flash-realtime"
                  readonly
                  class="mt-1.5 h-[42px] w-full bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text-secondary focus:outline-none"
                />
              </label>
            </div>

            <div class="grid gap-3 md:grid-cols-[90px_minmax(0,1fr)] md:items-start">
              <span class="text-[13px] font-medium text-cv-text-secondary md:pt-3">{{ t('characterEdit.lineVoice') }}</span>
              <label class="block">
                <div class="flex items-start gap-3">
                  <CvSelect
                    v-if="usesQwenOmniVoice"
                    v-model="form.voice_type"
                    :options="qwenOmniVoiceOptions"
                    :success="voiceCheckSucceeded"
                    class="min-w-0 flex-1"
                  />
                  <CvSelect
                    v-else-if="voiceMode === 'official'"
                    v-model="form.voice_type"
                    :options="officialVoiceOptions"
                    :success="voiceCheckSucceeded"
                    class="min-w-0 flex-1"
                  />
                  <div v-else class="relative min-w-0 flex-1">
                    <input
                      v-model="customVoiceType"
                      type="text"
                      :placeholder="t('characterEdit.registeredSpeakerPlaceholder')"
                      class="h-[42px] w-full bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text placeholder:text-cv-text-muted focus:outline-none transition-all"
                      :class="voiceCheckSucceeded
                        ? 'pr-11 border-cv-success focus:border-cv-success focus:shadow-[0_0_0_2px_rgba(34,197,94,0.15)]'
                        : 'focus:border-cv-accent focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)]'"
                    />
                    <span
                      v-if="voiceCheckSucceeded"
                      class="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-cv-success"
                    >
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <path d="M3.5 8.5l3 3 6-6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
                      </svg>
                    </span>
                  </div>
                  <button
                    type="button"
                    @click="handleCheckVoice"
                    :disabled="testingVoice || !canCheckVoice"
                    :class="{ 'opacity-40 cursor-not-allowed': testingVoice || !canCheckVoice }"
                    class="inline-flex h-[42px] shrink-0 items-center rounded-cv-md border border-cv-border px-4 text-sm text-cv-text-secondary transition-all hover:bg-cv-hover hover:text-cv-text cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {{ t('common.check') }}
                  </button>
                </div>
                <p v-if="usesDoubaoVoice && voiceMode === 'official'" class="mt-2 text-[11px] leading-5 text-cv-text-muted">
                  {{ t('characterEdit.canPreviewAt') }}
                  <a
                    :href="OFFICIAL_VOICE_PREVIEW_URL"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="underline underline-offset-2 transition-colors hover:text-cv-text"
                  >
                    {{ t('characterEdit.doubaoVoiceConsole') }}
                  </a>
                  {{ t('characterEdit.previewVoice') }}
                </p>
                <p v-if="usesQwenOmniVoice" class="mt-2 text-[11px] leading-5 text-cv-text-muted">
                  {{ t('characterEdit.canPreviewAt') }}
                  <a
                    :href="QWEN_OMNI_VOICE_LIST_URL"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="underline underline-offset-2 transition-colors hover:text-cv-text"
                  >
                    {{ t('characterEdit.qwenOmniVoiceList') }}
                  </a>
                  {{ t('characterEdit.previewVoice') }}
                </p>
                <p v-if="usesDoubaoVoice && voiceMode === 'custom'" class="mt-2 text-[11px] leading-5 text-cv-text-muted">
                  {{ t('characterEdit.clonePrerequisitePrefix') }}
                  <a
                    :href="CUSTOM_VOICE_CLONE_URL"
                    target="_blank"
                    rel="noopener noreferrer"
                    class="underline underline-offset-2 transition-colors hover:text-cv-text"
                  >
                    {{ t('characterEdit.doubaoVoiceConsole') }}
                  </a>
                  {{ t('characterEdit.clonePrerequisiteSuffix') }}
                </p>
                <p v-if="voiceError" class="mt-2 text-[11px] text-cv-danger">{{ voiceError }}</p>
                <p
                  v-if="voiceTestStatus === 'error' && voiceTestMessage"
                  class="mt-2 text-[11px] leading-5 text-cv-danger whitespace-pre-wrap break-all"
                >
                  {{ voiceTestMessage }}
                </p>
              </label>
            </div>
          </div>
        </section>

        <!-- Section 3: Persona and style -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <h2 class="text-base font-semibold text-cv-text mb-5">{{ t('characterEdit.personaStyle') }}</h2>

          <label class="block mb-4">
            <span class="text-[13px] font-medium text-cv-text-secondary">{{ t('characterEdit.speakingStyle') }}</span>
            <input v-model="form.speaking_style" type="text" :placeholder="t('characterEdit.speakingStylePlaceholder')"
                   class="mt-1.5 w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text placeholder:text-cv-text-muted focus:border-cv-accent focus:outline-none focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)] transition-all" />
            <p class="text-[11px] text-cv-text-muted mt-1">{{ t('characterEdit.speakingStyleHint') }}</p>
          </label>

          <label class="block mb-4">
            <span class="text-[13px] font-medium text-cv-text-secondary">{{ t('characterEdit.personality') }}</span>
            <textarea v-model="form.personality" :placeholder="t('characterEdit.personalityPlaceholder')"
                      class="mt-1.5 w-full h-20 bg-cv-elevated border border-cv-border rounded-cv-md px-4 py-3 text-sm text-cv-text placeholder:text-cv-text-muted resize-y focus:border-cv-accent focus:outline-none focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)] transition-all" />
            <p class="text-[11px] text-cv-text-muted mt-1">{{ t('characterEdit.personalityHint') }}</p>
          </label>

          <label class="block">
            <span class="text-[13px] font-medium text-cv-text-secondary">{{ t('characterEdit.welcomeMessage') }}</span>
            <textarea v-model="form.welcome_message" :placeholder="t('characterEdit.welcomeMessagePlaceholder')"
                      class="mt-1.5 w-full h-[60px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 py-3 text-sm text-cv-text placeholder:text-cv-text-muted resize-y focus:border-cv-accent focus:outline-none focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)] transition-all" />
            <p class="text-[11px] text-cv-text-muted mt-1">{{ t('characterEdit.welcomeMessageHint') }}</p>
          </label>
        </section>

        <!-- Section 4: Role prompt -->
        <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
          <h2 class="text-base font-semibold text-cv-text mb-5">{{ t('characterEdit.systemPrompt') }}</h2>

          <textarea v-model="form.system_prompt"
                    :placeholder="t('characterEdit.systemPromptPlaceholder')"
                    class="w-full h-40 bg-cv-elevated border border-cv-border rounded-cv-md px-4 py-3 text-[13px] text-cv-text placeholder:text-cv-text-muted resize-y leading-[22px] focus:border-cv-accent focus:outline-none focus:shadow-[0_0_0_2px_rgba(59,130,246,0.15)] transition-all" />
          <p class="text-right text-[11px] text-cv-text-muted mt-1">{{ promptLength }} / 2000</p>
        </section>

        <KnowledgeSourceManager v-if="isEdit" :character-id="characterId" />
      </div>
    </main>

    <!-- Bottom action bar -->
    <div class="fixed bottom-0 left-0 right-0 bg-cv-surface border-t border-cv-border-subtle px-12 py-4 z-20">
      <div class="max-w-[1100px] mx-auto flex items-center justify-between">
        <button v-if="isEdit" @click="handleDelete"
                class="text-cv-danger text-sm hover:bg-cv-danger-muted px-3 py-1.5 rounded-cv-md transition-colors cursor-pointer">
          {{ t('characterEdit.deleteCharacter') }}
        </button>
        <div v-else />
        <div class="flex items-center gap-3">
          <button @click="router.back()"
                  class="px-5 py-2.5 border border-cv-border text-cv-text-secondary text-sm rounded-cv-md hover:bg-cv-hover hover:text-cv-text transition-all cursor-pointer">
            {{ t('common.cancel') }}
          </button>
          <button @click="save" :disabled="saving || !canSave"
                  :class="{ 'opacity-40 cursor-not-allowed': saving || !canSave }"
                  class="px-6 py-2.5 bg-cv-accent text-white text-sm font-medium rounded-cv-md hover:bg-cv-accent-hover transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed">
            {{ saving ? t('common.saving') : t('characterEdit.saveCharacter') }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>
