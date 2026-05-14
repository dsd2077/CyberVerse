<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import type { KnowledgeSource, KnowledgeUploadSkippedFile } from '../types'
import {
  deleteKnowledgeSource,
  getKnowledgeSources,
  reindexKnowledgeSource,
  uploadKnowledgeFiles,
} from '../services/api'

type BrowserFileSystemEntry = {
  isFile: boolean
  isDirectory: boolean
  name: string
  fullPath?: string
}

type BrowserFileEntry = BrowserFileSystemEntry & {
  file: (success: (file: File) => void, failure?: (error: DOMException) => void) => void
}

type BrowserDirectoryEntry = BrowserFileSystemEntry & {
  createReader: () => {
    readEntries: (
      success: (entries: BrowserFileSystemEntry[]) => void,
      failure?: (error: DOMException) => void,
    ) => void
  }
}

type MaterialFileEntry = {
  kind: 'file'
  key: string
  name: string
  depth: number
  source: KnowledgeSource
}

type MaterialFolderEntry = {
  kind: 'folder'
  key: string
  name: string
  path: string
  depth: number
  fileCount: number
  expanded: boolean
}

type MaterialTreeEntry = MaterialFileEntry | MaterialFolderEntry

type MaterialFolderNode = {
  name: string
  path: string
  depth: number
  folders: Map<string, MaterialFolderNode>
  files: MaterialFileEntry[]
}

const props = defineProps<{
  characterId: string
}>()

const { t } = useI18n()

const sources = ref<KnowledgeSource[]>([])
const loading = ref(false)
const saving = ref(false)
const dragOver = ref(false)
const error = ref('')
const skippedFiles = ref<KnowledgeUploadSkippedFile[]>([])
const fileInput = ref<HTMLInputElement | null>(null)
const folderInput = ref<HTMLInputElement | null>(null)
const expandedFolders = ref<Set<string>>(new Set())
let pollTimer: number | undefined

const hasIndexing = computed(() => sources.value.some(item => item.status === 'indexing'))
const materialTreeEntries = computed(() => buildMaterialTreeEntries(sources.value, expandedFolders.value))

function statusLabel(status: KnowledgeSource['status']) {
  if (status === 'ready') return 'ready'
  if (status === 'failed') return 'failed'
  return 'indexing'
}

function statusClass(status: KnowledgeSource['status']) {
  if (status === 'ready') return 'border-cv-success/40 text-cv-success bg-cv-success/10'
  if (status === 'failed') return 'border-cv-danger/40 text-cv-danger bg-cv-danger-muted'
  return 'border-cv-accent/40 text-cv-accent bg-cv-accent/10'
}

function materialPath(source: KnowledgeSource) {
  return (source.relative_path || source.filename || source.title || source.id).replace(/\\/g, '/').replace(/^\/+/, '')
}

function materialPathParts(source: KnowledgeSource) {
  return materialPath(source).split('/').map(part => part.trim()).filter(Boolean)
}

function sourceFileName(source: KnowledgeSource) {
  const parts = materialPathParts(source)
  return parts[parts.length - 1] || source.filename || source.title || source.id
}

function compareName(left: string, right: string) {
  return left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' })
}

function createFolderNode(name: string, path: string, depth: number): MaterialFolderNode {
  return { name, path, depth, folders: new Map(), files: [] }
}

function countFolderFiles(folder: MaterialFolderNode): number {
  let count = folder.files.length
  for (const child of folder.folders.values()) {
    count += countFolderFiles(child)
  }
  return count
}

function flattenFolder(folder: MaterialFolderNode, entries: MaterialTreeEntry[], expanded: Set<string>) {
  const isExpanded = expanded.has(folder.path)
  entries.push({
    kind: 'folder',
    key: `folder:${folder.path}`,
    name: folder.name,
    path: folder.path,
    depth: folder.depth,
    fileCount: countFolderFiles(folder),
    expanded: isExpanded,
  })
  if (!isExpanded) return

  const childFolders = Array.from(folder.folders.values()).sort((a, b) => compareName(a.name, b.name))
  for (const child of childFolders) {
    flattenFolder(child, entries, expanded)
  }
  const files = [...folder.files].sort((a, b) => compareName(a.name, b.name))
  entries.push(...files)
}

function buildMaterialTreeEntries(items: KnowledgeSource[], expanded: Set<string>): MaterialTreeEntry[] {
  const root = createFolderNode('', '', -1)
  const sorted = [...items].sort((a, b) => compareName(materialPath(a), materialPath(b)))

  for (const source of sorted) {
    const parts = materialPathParts(source)
    const fileName = parts.pop() || sourceFileName(source)
    let folder = root
    let currentPath = ''
    for (const part of parts) {
      currentPath = currentPath ? `${currentPath}/${part}` : part
      let next = folder.folders.get(part)
      if (!next) {
        next = createFolderNode(part, currentPath, folder.depth + 1)
        folder.folders.set(part, next)
      }
      folder = next
    }
    folder.files.push({
      kind: 'file',
      key: source.id,
      name: fileName,
      depth: folder.depth + 1,
      source,
    })
  }

  const entries: MaterialTreeEntry[] = []
  const folders = Array.from(root.folders.values()).sort((a, b) => compareName(a.name, b.name))
  for (const folder of folders) {
    flattenFolder(folder, entries, expanded)
  }
  entries.push(...[...root.files].sort((a, b) => compareName(a.name, b.name)))
  return entries
}

function entryPadding(depth: number) {
  return `${16 + Math.max(0, depth) * 22}px`
}

function toggleFolder(path: string) {
  const next = new Set(expandedFolders.value)
  if (next.has(path)) {
    next.delete(path)
  } else {
    next.add(path)
  }
  expandedFolders.value = next
}

function fileLabel(source: KnowledgeSource) {
  return sourceFileName(source)
}

function skippedSummary() {
  if (skippedFiles.value.length === 0) return ''
  const names = skippedFiles.value.slice(0, 4).map(item => item.filename).join(', ')
  const suffix = skippedFiles.value.length > 4 ? ', ...' : ''
  return `${names}${suffix}`
}

async function loadSources() {
  loading.value = true
  error.value = ''
  try {
    sources.value = await getKnowledgeSources(props.characterId)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

function startPolling() {
  stopPolling()
  pollTimer = window.setInterval(async () => {
    if (!hasIndexing.value) {
      stopPolling()
      return
    }
    await loadSources()
  }, 2500)
}

function stopPolling() {
  if (pollTimer !== undefined) {
    window.clearInterval(pollTimer)
    pollTimer = undefined
  }
}

function resetInputs() {
  if (fileInput.value) fileInput.value.value = ''
  if (folderInput.value) folderInput.value.value = ''
}

async function submitFiles(files: File[]) {
  if (saving.value) return
  error.value = ''
  skippedFiles.value = []
  if (files.length === 0) {
    error.value = t('characterEdit.noSupportedMaterials') as string
    resetInputs()
    return
  }

  saving.value = true
  try {
    const result = await uploadKnowledgeFiles(props.characterId, files)
    skippedFiles.value = result.skipped || []
    await loadSources()
    startPolling()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  } finally {
    saving.value = false
    resetInputs()
  }
}

function chooseFiles() {
  if (!saving.value) fileInput.value?.click()
}

function chooseFolder() {
  if (!saving.value) folderInput.value?.click()
}

function onDropzoneClick(event: MouseEvent) {
  const target = event.target
  if (target === fileInput.value || target === folderInput.value) return
  chooseFiles()
}

function onFileInput(event: Event) {
  const input = event.target as HTMLInputElement
  void submitFiles(Array.from(input.files || []))
}

function fileFromEntry(entry: BrowserFileEntry): Promise<File> {
  return new Promise((resolve, reject) => {
    entry.file(file => {
      const relativePath = (entry.fullPath || entry.name).replace(/^\/+/, '')
      ;(file as File & { relativePath?: string }).relativePath = relativePath
      try {
        Object.defineProperty(file, 'webkitRelativePath', { value: relativePath })
      } catch {
        // Browsers that expose a readonly webkitRelativePath still upload the file name.
      }
      resolve(file)
    }, reject)
  })
}

async function readDirectoryEntry(entry: BrowserDirectoryEntry): Promise<File[]> {
  const reader = entry.createReader()
  const entries: BrowserFileSystemEntry[] = []
  while (true) {
    const batch = await new Promise<BrowserFileSystemEntry[]>((resolve, reject) => {
      reader.readEntries(resolve, reject)
    })
    if (batch.length === 0) break
    entries.push(...batch)
  }
  const nested = await Promise.all(entries.map(readEntryFiles))
  return nested.flat()
}

async function readEntryFiles(entry: BrowserFileSystemEntry): Promise<File[]> {
  if (entry.isFile) return [await fileFromEntry(entry as BrowserFileEntry)]
  if (entry.isDirectory) return readDirectoryEntry(entry as BrowserDirectoryEntry)
  return []
}

async function filesFromDrop(event: DragEvent): Promise<File[]> {
  const items = Array.from(event.dataTransfer?.items || [])
  const entries: BrowserFileSystemEntry[] = []
  for (const item of items) {
    const entry = (item as DataTransferItem & { webkitGetAsEntry?: () => BrowserFileSystemEntry | null })
      .webkitGetAsEntry?.()
    if (entry) entries.push(entry)
  }
  const entryPromises = entries.map(readEntryFiles)
  if (entryPromises.length > 0) {
    return (await Promise.all(entryPromises)).flat()
  }
  return Array.from(event.dataTransfer?.files || [])
}

async function onDrop(event: DragEvent) {
  dragOver.value = false
  const files = await filesFromDrop(event)
  await submitFiles(files)
}

async function removeSource(source: KnowledgeSource) {
  if (!confirm(`删除素材「${fileLabel(source)}」？`)) return
  error.value = ''
  try {
    await deleteKnowledgeSource(props.characterId, source.id)
    sources.value = sources.value.filter(item => item.id !== source.id)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

async function rebuildSource(source: KnowledgeSource) {
  error.value = ''
  try {
    await reindexKnowledgeSource(props.characterId, source.id)
    await loadSources()
    startPolling()
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
  }
}

onMounted(async () => {
  await loadSources()
  if (hasIndexing.value) startPolling()
})

onUnmounted(stopPolling)
</script>

<template>
  <section class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
    <div class="mb-5 flex items-center justify-between gap-4">
      <div>
        <h2 class="text-base font-semibold text-cv-text">{{ t('characterEdit.knowledgeBase') }}</h2>
        <p class="mt-1 text-[12px] leading-5 text-cv-text-muted">{{ t('characterEdit.knowledgeBaseHint') }}</p>
      </div>
      <button
        type="button"
        @click="loadSources"
        class="h-8 shrink-0 rounded-cv-md border border-cv-border px-3 text-xs text-cv-text-secondary transition-colors hover:bg-cv-hover hover:text-cv-text"
      >
        {{ loading ? t('common.loading') : t('common.refresh') }}
      </button>
    </div>

    <div
      class="flex min-h-[220px] flex-col items-center justify-center rounded-cv-lg border border-dashed px-6 py-8 text-center transition-colors"
      :class="dragOver ? 'border-cv-accent bg-cv-accent/10' : 'border-cv-border bg-cv-elevated'"
      @click="onDropzoneClick"
      @dragenter.prevent="dragOver = true"
      @dragover.prevent="dragOver = true"
      @dragleave.prevent="dragOver = false"
      @drop.prevent="onDrop"
    >
      <input
        ref="fileInput"
        type="file"
        multiple
        class="hidden"
        @click.stop
        @change="onFileInput"
      />
      <input
        ref="folderInput"
        type="file"
        multiple
        webkitdirectory
        directory
        class="hidden"
        @click.stop
        @change="onFileInput"
      />
      <div class="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-cv-hover text-cv-text-muted">
        <svg viewBox="0 0 24 24" class="h-9 w-9" aria-hidden="true">
          <path fill="currentColor" d="M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 6 20h13a4.996 4.996 0 0 0 .35-9.96ZM13 13v4h-2v-4H8l4-4 4 4h-3Z" />
        </svg>
      </div>
      <p class="text-sm font-medium text-cv-text">
        {{ saving ? t('characterEdit.uploadingMaterials') : t('characterEdit.uploadDropTitle') }}
      </p>
      <p class="mt-2 text-[12px] leading-5 text-cv-text-muted">{{ t('characterEdit.uploadDropHint') }}</p>
      <div class="mt-5 flex flex-wrap justify-center gap-3">
        <button
          type="button"
          :disabled="saving"
          class="h-[38px] rounded-cv-md bg-cv-accent px-4 text-sm font-medium text-white transition-colors hover:bg-cv-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
          @click.stop="chooseFiles"
        >
          {{ t('characterEdit.chooseMaterialFiles') }}
        </button>
        <button
          type="button"
          :disabled="saving"
          class="h-[38px] rounded-cv-md border border-cv-border px-4 text-sm text-cv-text-secondary transition-colors hover:bg-cv-hover hover:text-cv-text disabled:cursor-not-allowed disabled:opacity-40"
          @click.stop="chooseFolder"
        >
          {{ t('characterEdit.chooseMaterialFolder') }}
        </button>
      </div>
      <p class="mt-4 text-[11px] text-cv-text-muted">{{ t('characterEdit.supportedMaterialFiles') }}</p>
    </div>

    <p v-if="error" class="mt-3 whitespace-pre-wrap break-all text-[12px] leading-5 text-cv-danger">{{ error }}</p>
    <p v-if="skippedFiles.length" class="mt-3 text-[12px] leading-5 text-cv-warning">
      {{ t('characterEdit.skippedMaterialFiles', { count: skippedFiles.length, files: skippedSummary() }) }}
    </p>

    <div class="mt-6 overflow-hidden rounded-cv-md border border-cv-border">
      <div v-if="sources.length === 0" class="bg-cv-elevated px-4 py-5 text-sm text-cv-text-muted">
        {{ t('characterEdit.noMaterials') }}
      </div>
      <div v-else class="bg-cv-elevated">
        <div
          v-for="(entry, entryIndex) in materialTreeEntries"
          :key="entry.key"
          :class="entryIndex < materialTreeEntries.length - 1 ? 'border-b border-cv-border' : ''"
        >
          <div
            v-if="entry.kind === 'folder'"
            class="flex min-w-0 cursor-pointer items-center gap-2 bg-cv-surface/40 py-2 pr-4 text-xs text-cv-text-secondary transition-colors hover:bg-cv-hover"
            :style="{ paddingLeft: entryPadding(entry.depth) }"
            role="button"
            tabindex="0"
            :aria-expanded="entry.expanded"
            @click="toggleFolder(entry.path)"
            @keydown.enter.prevent="toggleFolder(entry.path)"
            @keydown.space.prevent="toggleFolder(entry.path)"
          >
            <svg
              viewBox="0 0 24 24"
              class="h-3.5 w-3.5 shrink-0 text-cv-text-muted transition-transform"
              :class="entry.expanded ? 'rotate-90' : ''"
              aria-hidden="true"
            >
              <path fill="currentColor" d="M9 6l6 6-6 6V6Z" />
            </svg>
            <svg viewBox="0 0 24 24" class="h-4 w-4 shrink-0 text-cv-text-muted" aria-hidden="true">
              <path fill="currentColor" d="M10 4l2 2h8a2 2 0 0 1 2 2v1H2V6a2 2 0 0 1 2-2h6Zm12 7v7a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-7h20Z" />
            </svg>
            <span class="truncate font-medium text-cv-text">{{ entry.name }}</span>
            <span class="shrink-0 text-[11px] text-cv-text-muted">{{ t('characterEdit.materialFileCount', { count: entry.fileCount }) }}</span>
          </div>
          <div
            v-else
            class="grid gap-3 py-3 pr-4 md:grid-cols-[minmax(0,1fr)_auto] md:items-center"
            :style="{ paddingLeft: entryPadding(entry.depth) }"
          >
          <div class="min-w-0">
            <div class="flex flex-wrap items-center gap-2">
              <span class="truncate text-sm font-medium text-cv-text" :title="materialPath(entry.source)">{{ entry.name }}</span>
              <span class="rounded-full border px-2 py-0.5 text-[11px]" :class="statusClass(entry.source.status)">
                {{ statusLabel(entry.source.status) }}
              </span>
            </div>
            <div class="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-cv-text-muted">
              <span>{{ entry.source.indexable ? t('characterEdit.materialIndexed') : t('characterEdit.materialStoredOnly') }}</span>
              <span>{{ entry.source.chunk_count }} chunks</span>
              <span v-if="entry.source.indexed_at">{{ entry.source.indexed_at }}</span>
            </div>
            <p v-if="entry.source.error" class="mt-1 break-all text-[11px] leading-5 text-cv-danger">{{ entry.source.error }}</p>
          </div>
          <div class="flex items-center gap-2">
            <button
              v-if="entry.source.indexable"
              type="button"
              @click="rebuildSource(entry.source)"
              class="h-8 rounded-cv-md border border-cv-border px-3 text-xs text-cv-text-secondary transition-colors hover:bg-cv-hover hover:text-cv-text"
            >
              {{ t('characterEdit.reindexMaterial') }}
            </button>
            <button
              type="button"
              @click="removeSource(entry.source)"
              class="h-8 rounded-cv-md px-3 text-xs text-cv-danger transition-colors hover:bg-cv-danger-muted"
            >
              {{ t('common.delete') }}
            </button>
          </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>
