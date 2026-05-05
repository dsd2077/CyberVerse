<script setup lang="ts">
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue'
import type { ImageInfo } from '../types'

interface DisplayImage {
  key: string        // unique key for v-for
  src: string        // display URL
  filename?: string  // server filename (undefined for pending uploads)
  pending?: boolean  // true if not yet uploaded
  pendingIndex?: number
}

interface CropBox {
  x: number
  y: number
  width: number
  height: number
}

interface CropPoint {
  x: number
  y: number
}

type CropHandle = 'nw' | 'ne' | 'sw' | 'se'
type CropDimension = 'width' | 'height'
type CropDragMode = 'draw' | 'move' | 'resize'

const props = defineProps<{
  useFaceCrop: boolean
  images: ImageInfo[]
  characterId?: string
  pendingFiles?: File[]
  activeImage?: string
  imageMode?: string
}>()

const emit = defineEmits<{
  'update:useFaceCrop': [value: boolean]
  fileSelected: [file: File, options?: { activate?: boolean }]
  replacePending: [index: number, file: File]
  deleteImage: [filename: string]
  deletePending: [index: number]
  activateImage: [filename: string]
}>()

const currentIndex = ref(0)
const dragOver = ref(false)
const showLightbox = ref(false)
const isEditingCrop = ref(false)
const applyingCrop = ref(false)
const cropError = ref('')
const cropImageEl = ref<HTMLImageElement | null>(null)
const imageNaturalWidth = ref(0)
const imageNaturalHeight = ref(0)
const imageRenderedWidth = ref(0)
const imageRenderedHeight = ref(0)
const cropBox = ref<CropBox | null>(null)
const isDraggingCrop = ref(false)
const cropDragStart = ref<CropPoint>({ x: 0, y: 0 })
const cropDragStartBox = ref<CropBox | null>(null)
const activeCropHandle = ref<CropHandle | null>(null)
const activeCropDragMode = ref<CropDragMode | null>(null)
const hasCropDragMovement = ref(false)
const activeDimensionInput = ref<CropDimension | null>(null)
const cropWidthDraft = ref('')
const cropHeightDraft = ref('')
const pendingObjectUrls = ref<string[]>([])
const MIN_CROP_SIZE = 24
const MIN_DRAG_DISTANCE = 3

function cleanupPendingUrls(urls = pendingObjectUrls.value) {
  for (const url of urls) {
    URL.revokeObjectURL(url)
  }
}

watch(
  () => props.pendingFiles,
  (files) => {
    const oldUrls = pendingObjectUrls.value
    pendingObjectUrls.value = (files || []).map(file => URL.createObjectURL(file))
    cleanupPendingUrls(oldUrls)
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  cleanupPendingUrls()
  stopCropDrag()
  window.removeEventListener('resize', handleViewportResize)
})

// Merge server images + pending local files into a unified display list
const displayImages = computed<DisplayImage[]>(() => {
  const list: DisplayImage[] = []

  // Server images
  if (props.images) {
    const orderedImages = [...props.images].sort((a, b) => {
      if (!props.activeImage) return 0
      if (a.filename === props.activeImage) return -1
      if (b.filename === props.activeImage) return 1
      return 0
    })

    for (const img of orderedImages) {
      list.push({
        key: 'srv-' + img.filename,
        src: img.url || (props.characterId
          ? `/api/v1/characters/${props.characterId}/images/${img.filename}`
          : ''),
        filename: img.filename,
      })
    }
  }

  // Pending local files
  if (props.pendingFiles) {
    for (let i = 0; i < props.pendingFiles.length; i++) {
      const src = pendingObjectUrls.value[i]
      if (!src) continue
      list.push({
        key: 'pending-' + i,
        src,
        pending: true,
        pendingIndex: i,
      })
    }
  }

  return list
})

const totalCount = computed(() => displayImages.value.length)
const hasImages = computed(() => totalCount.value > 0)
const currentImage = computed(() => displayImages.value[currentIndex.value] || null)
const displayImageOrderKey = computed(() => displayImages.value.map(img => img.key).join('|'))
const cropBoxStyle = computed(() => {
  const box = cropBox.value
  if (!box) return {}
  return {
    left: `${box.x}px`,
    top: `${box.y}px`,
    width: `${box.width}px`,
    height: `${box.height}px`,
  }
})
const originalResolutionLabel = computed(() => {
  if (!imageNaturalWidth.value || !imageNaturalHeight.value) return '读取中'
  return `${imageNaturalWidth.value} × ${imageNaturalHeight.value} px`
})
const cropResolution = computed(() => {
  const box = cropBox.value
  if (!box || !imageRenderedWidth.value || !imageRenderedHeight.value) return null
  return {
    width: Math.max(1, Math.round((box.width / imageRenderedWidth.value) * imageNaturalWidth.value)),
    height: Math.max(1, Math.round((box.height / imageRenderedHeight.value) * imageNaturalHeight.value)),
  }
})
const minCropNaturalWidth = computed(() => {
  if (!imageRenderedWidth.value || !imageNaturalWidth.value) return 1
  return Math.max(1, Math.ceil((MIN_CROP_SIZE / imageRenderedWidth.value) * imageNaturalWidth.value))
})
const minCropNaturalHeight = computed(() => {
  if (!imageRenderedHeight.value || !imageNaturalHeight.value) return 1
  return Math.max(1, Math.ceil((MIN_CROP_SIZE / imageRenderedHeight.value) * imageNaturalHeight.value))
})
const canApplyCrop = computed(() =>
  !!currentImage.value
  && !!cropBox.value
  && !!imageNaturalWidth.value
  && !!imageNaturalHeight.value
  && cropBox.value.width >= MIN_CROP_SIZE
  && cropBox.value.height >= MIN_CROP_SIZE
  && !applyingCrop.value
)

// Clamp index when list shrinks
watch(totalCount, (n) => {
  if (currentIndex.value >= n && n > 0) {
    currentIndex.value = n - 1
  }
})

watch(
  [() => props.activeImage, displayImageOrderKey],
  ([filename]) => {
    if (!filename) return
    const index = displayImages.value.findIndex(img => img.filename === filename)
    if (index >= 0) currentIndex.value = index
  },
  { immediate: true },
)

watch(cropResolution, () => {
  syncCropDimensionDrafts(activeDimensionInput.value)
})

function prev() {
  if (currentIndex.value > 0) currentIndex.value--
}

function next() {
  if (currentIndex.value < totalCount.value - 1) currentIndex.value++
}

function handleFile(file: File) {
  if (!file.type.startsWith('image/')) return
  emit('fileSelected', file)
  // Jump to the new image (will be appended at end)
  // Use nextTick-like delay so the list updates first
  setTimeout(() => {
    currentIndex.value = displayImages.value.length - 1
  }, 50)
}

function onDrop(e: DragEvent) {
  dragOver.value = false
  const file = e.dataTransfer?.files[0]
  if (file) handleFile(file)
}

function onFileInput(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) {
    for (const file of Array.from(input.files)) {
      handleFile(file)
    }
  }
  // Reset so same file can be re-selected
  input.value = ''
}

function handleDelete() {
  const img = currentImage.value
  if (!img) return

  if (img.pending) {
    if (typeof img.pendingIndex === 'number') {
      emit('deletePending', img.pendingIndex)
    }
  } else if (img.filename) {
    emit('deleteImage', img.filename)
  }
}

function triggerUpload() {
  ;(document.getElementById('avatar-file-input') as HTMLInputElement)?.click()
}

function openLightbox() {
  if (!currentImage.value) return
  showLightbox.value = true
  resetCropState()
  void nextTick(measureLightboxImage)
}

function closeLightbox() {
  showLightbox.value = false
  isEditingCrop.value = false
  resetCropState()
  stopCropDrag()
  window.removeEventListener('resize', handleViewportResize)
}

function startCropEditing() {
  cropError.value = ''
  isEditingCrop.value = true
  window.addEventListener('resize', handleViewportResize)
  void nextTick(() => {
    measureLightboxImage()
    createDefaultCropBox()
  })
}

function cancelCropEditing() {
  isEditingCrop.value = false
  resetCropState()
  stopCropDrag()
  window.removeEventListener('resize', handleViewportResize)
}

function resetCropState() {
  cropError.value = ''
  cropBox.value = null
  activeDimensionInput.value = null
  cropWidthDraft.value = ''
  cropHeightDraft.value = ''
  imageNaturalWidth.value = 0
  imageNaturalHeight.value = 0
  imageRenderedWidth.value = 0
  imageRenderedHeight.value = 0
}

function handleLightboxImageLoad() {
  measureLightboxImage()
  if (isEditingCrop.value && !cropBox.value) {
    createDefaultCropBox()
  }
}

function handleViewportResize() {
  const previousWidth = imageRenderedWidth.value
  const previousHeight = imageRenderedHeight.value
  const previousBox = cropBox.value ? { ...cropBox.value } : null
  measureLightboxImage()

  if (!previousBox || !previousWidth || !previousHeight || !imageRenderedWidth.value || !imageRenderedHeight.value) {
    return
  }

  cropBox.value = {
    x: (previousBox.x / previousWidth) * imageRenderedWidth.value,
    y: (previousBox.y / previousHeight) * imageRenderedHeight.value,
    width: (previousBox.width / previousWidth) * imageRenderedWidth.value,
    height: (previousBox.height / previousHeight) * imageRenderedHeight.value,
  }
  cropBox.value = normalizeCropBox(cropBox.value)
}

function measureLightboxImage() {
  const image = cropImageEl.value
  if (!image) return

  if (image.naturalWidth && image.naturalHeight) {
    imageNaturalWidth.value = image.naturalWidth
    imageNaturalHeight.value = image.naturalHeight
  }

  const rect = image.getBoundingClientRect()
  imageRenderedWidth.value = rect.width
  imageRenderedHeight.value = rect.height
}

function createDefaultCropBox() {
  if (!imageRenderedWidth.value || !imageRenderedHeight.value) return
  const size = Math.min(imageRenderedWidth.value, imageRenderedHeight.value) * 0.72
  cropBox.value = normalizeCropBox({
    x: (imageRenderedWidth.value - size) / 2,
    y: (imageRenderedHeight.value - size) / 2,
    width: size,
    height: size,
  })
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function getCropPoint(e: PointerEvent): CropPoint | null {
  const image = cropImageEl.value
  if (!image) return null

  const rect = image.getBoundingClientRect()
  return {
    x: clamp(e.clientX - rect.left, 0, rect.width),
    y: clamp(e.clientY - rect.top, 0, rect.height),
  }
}

function normalizeCropBox(box: CropBox): CropBox {
  const width = clamp(box.width, MIN_CROP_SIZE, imageRenderedWidth.value || box.width)
  const height = clamp(box.height, MIN_CROP_SIZE, imageRenderedHeight.value || box.height)
  return {
    x: clamp(box.x, 0, Math.max(0, imageRenderedWidth.value - width)),
    y: clamp(box.y, 0, Math.max(0, imageRenderedHeight.value - height)),
    width,
    height,
  }
}

function startCropDrag(e: PointerEvent) {
  if (!isEditingCrop.value) return
  const point = getCropPoint(e)
  if (!point) return

  e.preventDefault()
  cropError.value = ''
  measureLightboxImage()
  cropDragStart.value = point
  cropDragStartBox.value = cropBox.value ? { ...cropBox.value } : null
  activeCropHandle.value = null
  activeCropDragMode.value = 'draw'
  hasCropDragMovement.value = false
  isDraggingCrop.value = true
  window.addEventListener('pointermove', updateCropDrag)
  window.addEventListener('pointerup', finishCropDrag)
  window.addEventListener('pointercancel', finishCropDrag)
}

function startCropMove(e: PointerEvent) {
  if (!isEditingCrop.value || !cropBox.value) return
  const point = getCropPoint(e)
  if (!point) return

  e.preventDefault()
  cropError.value = ''
  measureLightboxImage()
  cropDragStart.value = point
  cropDragStartBox.value = { ...cropBox.value }
  activeCropHandle.value = null
  activeCropDragMode.value = 'move'
  hasCropDragMovement.value = false
  isDraggingCrop.value = true
  window.addEventListener('pointermove', updateCropDrag)
  window.addEventListener('pointerup', finishCropDrag)
  window.addEventListener('pointercancel', finishCropDrag)
}

function startCropResize(handle: CropHandle, e: PointerEvent) {
  if (!isEditingCrop.value || !cropBox.value) return
  const point = getCropPoint(e)
  if (!point) return

  e.preventDefault()
  cropError.value = ''
  measureLightboxImage()
  cropDragStart.value = point
  cropDragStartBox.value = { ...cropBox.value }
  activeCropHandle.value = handle
  activeCropDragMode.value = 'resize'
  hasCropDragMovement.value = false
  isDraggingCrop.value = true
  window.addEventListener('pointermove', updateCropDrag)
  window.addEventListener('pointerup', finishCropDrag)
  window.addEventListener('pointercancel', finishCropDrag)
}

function updateCropDrag(e: PointerEvent) {
  if (!isDraggingCrop.value) return
  const point = getCropPoint(e)
  if (!point) return

  const start = cropDragStart.value
  const movedDistance = Math.hypot(point.x - start.x, point.y - start.y)
  if (movedDistance >= MIN_DRAG_DISTANCE) {
    hasCropDragMovement.value = true
  }

  if (activeCropDragMode.value === 'resize' && activeCropHandle.value && cropDragStartBox.value) {
    updateCropResize(point)
    return
  }

  if (activeCropDragMode.value === 'move' && cropDragStartBox.value) {
    updateCropMove(point)
    return
  }

  if (activeCropDragMode.value === 'draw' && !hasCropDragMovement.value) {
    return
  }

  cropBox.value = {
    x: Math.min(start.x, point.x),
    y: Math.min(start.y, point.y),
    width: Math.abs(point.x - start.x),
    height: Math.abs(point.y - start.y),
  }
}

function updateCropMove(point: CropPoint) {
  const startBox = cropDragStartBox.value
  if (!startBox) return

  cropBox.value = normalizeCropBox({
    ...startBox,
    x: startBox.x + point.x - cropDragStart.value.x,
    y: startBox.y + point.y - cropDragStart.value.y,
  })
}

function updateCropResize(point: CropPoint) {
  const startBox = cropDragStartBox.value
  const handle = activeCropHandle.value
  if (!startBox || !handle) return

  let left = startBox.x
  let top = startBox.y
  let right = startBox.x + startBox.width
  let bottom = startBox.y + startBox.height

  if (handle.includes('w')) {
    left = clamp(point.x, 0, right - MIN_CROP_SIZE)
  }
  if (handle.includes('e')) {
    right = clamp(point.x, left + MIN_CROP_SIZE, imageRenderedWidth.value)
  }
  if (handle.includes('n')) {
    top = clamp(point.y, 0, bottom - MIN_CROP_SIZE)
  }
  if (handle.includes('s')) {
    bottom = clamp(point.y, top + MIN_CROP_SIZE, imageRenderedHeight.value)
  }

  cropBox.value = normalizeCropBox({
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
  })
}

function finishCropDrag(e?: PointerEvent) {
  const mode = activeCropDragMode.value
  const previousBox = cropDragStartBox.value ? { ...cropDragStartBox.value } : null
  if (e) updateCropDrag(e)
  const didMove = hasCropDragMovement.value
  isDraggingCrop.value = false
  stopCropDrag()

  const box = cropBox.value
  if (mode === 'draw' && (!didMove || !box || box.width < MIN_CROP_SIZE || box.height < MIN_CROP_SIZE)) {
    cropBox.value = previousBox
  }
}

function stopCropDrag() {
  window.removeEventListener('pointermove', updateCropDrag)
  window.removeEventListener('pointerup', finishCropDrag)
  window.removeEventListener('pointercancel', finishCropDrag)
  activeCropHandle.value = null
  cropDragStartBox.value = null
  activeCropDragMode.value = null
  hasCropDragMovement.value = false
}

function syncCropDimensionDrafts(preserve: CropDimension | null = null) {
  const resolution = cropResolution.value
  if (!resolution) {
    if (preserve !== 'width') cropWidthDraft.value = ''
    if (preserve !== 'height') cropHeightDraft.value = ''
    return
  }

  if (preserve !== 'width') cropWidthDraft.value = String(resolution.width)
  if (preserve !== 'height') cropHeightDraft.value = String(resolution.height)
}

function handleCropDimensionFocus(dimension: CropDimension) {
  activeDimensionInput.value = dimension
}

function handleCropDimensionInput(dimension: CropDimension, e: Event) {
  const input = e.target as HTMLInputElement
  const rawValue = input.value
  activeDimensionInput.value = dimension
  if (dimension === 'width') {
    cropWidthDraft.value = rawValue
  } else {
    cropHeightDraft.value = rawValue
  }

  if (!rawValue.trim()) return

  const value = Math.round(Number(rawValue))
  if (!Number.isFinite(value)) return
  setCropDimension(dimension, value)
}

function commitCropDimensionInput(dimension: CropDimension) {
  const draft = dimension === 'width' ? cropWidthDraft.value : cropHeightDraft.value
  const value = Math.round(Number(draft))
  if (draft.trim() && Number.isFinite(value)) {
    setCropDimension(dimension, value)
  }

  activeDimensionInput.value = null
  syncCropDimensionDrafts()
}

function setCropDimension(dimension: CropDimension, naturalValue: number) {
  if (!cropBox.value || !imageNaturalWidth.value || !imageNaturalHeight.value) return
  measureLightboxImage()
  if (!imageRenderedWidth.value || !imageRenderedHeight.value) return

  const current = cropBox.value
  const centerX = current.x + current.width / 2
  const centerY = current.y + current.height / 2
  const nextNaturalWidth = dimension === 'width'
    ? clamp(naturalValue, minCropNaturalWidth.value, imageNaturalWidth.value)
    : cropResolution.value?.width || minCropNaturalWidth.value
  const nextNaturalHeight = dimension === 'height'
    ? clamp(naturalValue, minCropNaturalHeight.value, imageNaturalHeight.value)
    : cropResolution.value?.height || minCropNaturalHeight.value
  const nextWidth = (nextNaturalWidth / imageNaturalWidth.value) * imageRenderedWidth.value
  const nextHeight = (nextNaturalHeight / imageNaturalHeight.value) * imageRenderedHeight.value

  cropBox.value = normalizeCropBox({
    x: centerX - nextWidth / 2,
    y: centerY - nextHeight / 2,
    width: nextWidth,
    height: nextHeight,
  })
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality?: number) {
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob)
      } else {
        reject(new Error('裁剪图片失败'))
      }
    }, type, quality)
  })
}

async function applyCrop() {
  const image = cropImageEl.value
  const source = currentImage.value
  const box = cropBox.value
  if (!image || !source || !box || !canApplyCrop.value) return

  applyingCrop.value = true
  cropError.value = ''
  try {
    measureLightboxImage()
    const scaleX = imageNaturalWidth.value / imageRenderedWidth.value
    const scaleY = imageNaturalHeight.value / imageRenderedHeight.value
    const sx = Math.round(box.x * scaleX)
    const sy = Math.round(box.y * scaleY)
    const sw = Math.max(1, Math.round(box.width * scaleX))
    const sh = Math.max(1, Math.round(box.height * scaleY))

    const canvas = document.createElement('canvas')
    canvas.width = sw
    canvas.height = sh
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('当前浏览器不支持图片裁剪')

    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh)
    const blob = await canvasToBlob(canvas, 'image/png')
    const file = new File([blob], `avatar-crop-${Date.now()}.png`, { type: blob.type || 'image/png' })

    if (source.pending && typeof source.pendingIndex === 'number') {
      emit('replacePending', source.pendingIndex, file)
    } else {
      emit('fileSelected', file, { activate: true })
    }

    closeLightbox()
  } catch (e) {
    cropError.value = e instanceof Error ? e.message : '裁剪图片失败'
  } finally {
    applyingCrop.value = false
  }
}
</script>

<template>
  <div class="bg-cv-surface border border-cv-border rounded-cv-lg p-6">
    <!-- Carousel or empty upload -->
    <div v-if="hasImages"
         class="relative w-full aspect-square rounded-cv-lg overflow-hidden group">
      <!-- Current image -->
      <img :src="currentImage?.src" class="w-full h-full object-cover transition-opacity duration-200 cursor-pointer" @click="openLightbox" />

      <!-- Pending badge -->
      <div v-if="currentImage?.pending"
           class="absolute bottom-3 left-3 px-2.5 py-1 bg-cv-accent/80 text-white text-[11px] font-medium rounded-full backdrop-blur-sm">
        待上传
      </div>

      <!-- Active image badge -->
      <div v-else-if="currentImage?.filename && currentImage.filename === activeImage && imageMode !== 'random'"
           class="absolute bottom-3 left-3 px-2.5 py-1 bg-emerald-500/80 text-white text-[11px] font-medium rounded-full backdrop-blur-sm flex items-center gap-1">
        <svg class="w-3 h-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M3 8.5l3.5 3.5L13 5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        当前头像
      </div>

      <!-- Set as active button -->
      <button v-else-if="currentImage?.filename && imageMode !== 'random'"
              @click.stop="emit('activateImage', currentImage!.filename!)"
              class="absolute bottom-3 left-3 px-2.5 py-1 bg-black/60 text-white/80 text-[11px] font-medium rounded-full opacity-0 group-hover:opacity-100 transition-all backdrop-blur-sm flex items-center gap-1 cursor-pointer hover:bg-cv-accent/80">
        <svg class="w-3 h-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M3 8.5l3.5 3.5L13 5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
        设为头像
      </button>

      <!-- Delete button (top-right) -->
      <button @click.stop="handleDelete"
              class="absolute top-3 right-3 w-8 h-8 flex items-center justify-center rounded-full bg-black/60 text-white/80 hover:bg-red-600/80 transition-all opacity-0 group-hover:opacity-100 cursor-pointer backdrop-blur-sm">
        <svg class="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M4 4l8 8M12 4l-8 8" stroke-linecap="round" />
        </svg>
      </button>

      <!-- Add more button (top-left, on hover) -->
      <button @click.stop="triggerUpload"
              class="absolute top-3 left-3 px-2.5 py-1 bg-black/60 text-white/80 text-[11px] rounded-full opacity-0 group-hover:opacity-100 transition-all backdrop-blur-sm flex items-center gap-1 cursor-pointer hover:bg-black/80">
        <svg class="w-3 h-3" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M8 3v10M3 8h10" stroke-linecap="round" />
        </svg>
        添加图片
      </button>

      <!-- Left arrow -->
      <button v-if="currentIndex > 0"
              @click.stop="prev"
              class="absolute left-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center rounded-full bg-black/50 text-white hover:bg-black/70 transition-all cursor-pointer backdrop-blur-sm">
        <svg class="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M10 3L5 8l5 5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>

      <!-- Right arrow -->
      <button v-if="currentIndex < totalCount - 1"
              @click.stop="next"
              class="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center rounded-full bg-black/50 text-white hover:bg-black/70 transition-all cursor-pointer backdrop-blur-sm">
        <svg class="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M6 3l5 5-5 5" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </button>

      <!-- Dots indicator -->
      <div v-if="totalCount > 1"
           class="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1.5">
        <button v-for="(_, i) in displayImages" :key="i"
                @click.stop="currentIndex = i"
                class="w-2 h-2 rounded-full transition-all cursor-pointer"
                :class="i === currentIndex ? 'bg-white w-4' : 'bg-white/50 hover:bg-white/70'" />
      </div>
    </div>

    <!-- Empty state: upload placeholder -->
    <div v-else
         class="relative w-full aspect-square rounded-cv-lg bg-cv-elevated border-2 border-dashed border-cv-border flex flex-col items-center justify-center cursor-pointer hover:border-cv-accent hover:bg-cv-accent/5 transition-all"
         :class="{ 'border-cv-accent bg-cv-accent/5': dragOver }"
         @dragover.prevent="dragOver = true"
         @dragleave="dragOver = false"
         @drop.prevent="onDrop"
         @click="triggerUpload">
      <div class="w-12 h-12 rounded-full bg-cv-hover flex items-center justify-center mb-3">
        <svg class="w-5 h-5 text-cv-text-secondary" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M10 4v12M4 10h12" stroke-linecap="round" />
        </svg>
      </div>
      <p class="text-sm font-medium text-cv-text-secondary">上传角色头像</p>
      <p class="text-xs text-cv-text-muted mt-1">支持 PNG、JPG，建议 512x512</p>
    </div>

    <!-- Counter -->
    <div v-if="hasImages" class="mt-3 text-center">
      <span class="text-[12px] text-cv-text-muted">{{ currentIndex + 1 }} / {{ totalCount }}</span>
    </div>

    <!-- Hidden file input -->
    <input id="avatar-file-input" type="file" accept="image/*" multiple class="hidden" @change="onFileInput" />

    <!-- Face crop toggle -->
    <div class="mt-4 pt-4 border-t border-cv-border-subtle">
      <div class="flex items-center justify-between">
        <span class="text-[13px] text-cv-text-secondary">是否裁剪人脸</span>
        <button @click="emit('update:useFaceCrop', !useFaceCrop)"
                class="relative w-11 h-6 rounded-full transition-colors cursor-pointer"
                :class="useFaceCrop ? 'bg-cv-accent' : 'bg-cv-elevated'">
          <span class="absolute top-0.5 left-0.5 w-5 h-5 rounded-full transition-transform duration-200"
                :class="useFaceCrop ? 'translate-x-5 bg-white' : 'translate-x-0 bg-cv-text-muted'" />
        </button>
      </div>
      <p class="text-[11px] text-cv-text-muted mt-2 leading-4">开启后将自动检测并裁剪图片中的人脸区域</p>
    </div>
  </div>

  <!-- Lightbox modal -->
  <Teleport to="body">
    <Transition name="lightbox">
      <div v-if="showLightbox" class="fixed inset-0 z-50 bg-black/85 backdrop-blur-sm" @click="closeLightbox">
        <div class="absolute right-4 top-4 z-20 flex items-center gap-2">
          <button v-if="!isEditingCrop"
                  class="h-10 px-4 rounded-full bg-white/10 text-sm font-medium text-white hover:bg-white/20 transition-colors cursor-pointer"
                  @click.stop="startCropEditing">
            编辑
          </button>
          <button v-else
                  class="h-10 px-4 rounded-full bg-white/10 text-sm font-medium text-white hover:bg-white/20 transition-colors cursor-pointer"
                  @click.stop="cancelCropEditing">
            退出编辑
          </button>
          <button class="w-10 h-10 flex items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20 transition-colors cursor-pointer"
                  @click.stop="closeLightbox"
                  aria-label="关闭预览">
            <svg class="w-5 h-5" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M4 4l8 8M12 4l-8 8" stroke-linecap="round" />
            </svg>
          </button>
        </div>

        <div class="flex h-full w-full items-center justify-center px-5 py-16" @click.stop>
          <div class="flex h-full w-full max-w-[1280px] flex-col items-center justify-center gap-4 lg:flex-row">
            <div class="flex min-h-0 flex-1 items-center justify-center">
              <div class="relative max-h-full max-w-full">
                <img
                  ref="cropImageEl"
                  :src="currentImage?.src"
                  class="block max-h-[82vh] max-w-[90vw] select-none rounded-lg object-contain shadow-2xl"
                  :class="isEditingCrop ? 'lg:max-w-[calc(100vw-390px)]' : ''"
                  draggable="false"
                  @load="handleLightboxImageLoad"
                  @click.stop
                />
                <div
                  v-if="isEditingCrop"
                  class="absolute inset-0 cursor-crosshair overflow-hidden rounded-lg"
                  @pointerdown.stop="startCropDrag"
                >
                  <div class="absolute inset-0 bg-black/20" />
                  <div
                    v-if="cropBox"
                    class="absolute cursor-move border-2 border-white shadow-[0_0_0_9999px_rgba(0,0,0,0.42)]"
                    :style="cropBoxStyle"
                    @pointerdown.stop="startCropMove"
                  >
                    <div class="absolute inset-0 bg-cv-accent/15" />
                    <button
                      type="button"
                      class="crop-handle crop-handle-nw"
                      aria-label="拖拽左上角调整裁剪区域"
                      @pointerdown.stop="startCropResize('nw', $event)"
                    />
                    <button
                      type="button"
                      class="crop-handle crop-handle-ne"
                      aria-label="拖拽右上角调整裁剪区域"
                      @pointerdown.stop="startCropResize('ne', $event)"
                    />
                    <button
                      type="button"
                      class="crop-handle crop-handle-sw"
                      aria-label="拖拽左下角调整裁剪区域"
                      @pointerdown.stop="startCropResize('sw', $event)"
                    />
                    <button
                      type="button"
                      class="crop-handle crop-handle-se"
                      aria-label="拖拽右下角调整裁剪区域"
                      @pointerdown.stop="startCropResize('se', $event)"
                    />
                  </div>
                </div>
              </div>
            </div>

            <aside
              v-if="isEditingCrop"
              class="w-full shrink-0 rounded-cv-lg border border-white/10 bg-cv-surface/95 p-5 text-cv-text shadow-2xl lg:w-[300px]"
              @click.stop
            >
              <h2 class="text-base font-semibold">头像裁剪</h2>
              <p class="mt-2 text-[12px] leading-5 text-cv-text-muted">拖拽图片框选要保留的头像区域，应用后会生成新的裁剪头像。</p>

              <div class="mt-5 space-y-3">
                <div class="rounded-cv-md bg-cv-elevated p-3">
                  <div class="text-[11px] text-cv-text-muted">原图分辨率</div>
                  <div class="mt-1 text-sm font-medium text-cv-text">{{ originalResolutionLabel }}</div>
                </div>
                <div class="rounded-cv-md bg-cv-elevated p-3">
                  <div class="text-[11px] text-cv-text-muted">裁剪区域</div>
                  <div class="mt-2 grid grid-cols-2 gap-2">
                    <label class="block">
                      <span class="text-[11px] text-cv-text-muted">宽</span>
                      <span class="mt-1 flex h-9 items-center rounded-cv-sm border border-cv-border bg-cv-surface px-2 focus-within:border-cv-accent">
                        <input
                          type="number"
                          inputmode="numeric"
                          class="min-w-0 flex-1 bg-transparent text-sm font-medium text-cv-text outline-none"
                          :min="minCropNaturalWidth"
                          :max="imageNaturalWidth || 1"
                          :value="cropWidthDraft"
                          @focus="handleCropDimensionFocus('width')"
                          @input="handleCropDimensionInput('width', $event)"
                          @blur="commitCropDimensionInput('width')"
                          @keydown.enter.prevent="commitCropDimensionInput('width')"
                        />
                        <span class="ml-1 text-[11px] text-cv-text-muted">px</span>
                      </span>
                    </label>
                    <label class="block">
                      <span class="text-[11px] text-cv-text-muted">高</span>
                      <span class="mt-1 flex h-9 items-center rounded-cv-sm border border-cv-border bg-cv-surface px-2 focus-within:border-cv-accent">
                        <input
                          type="number"
                          inputmode="numeric"
                          class="min-w-0 flex-1 bg-transparent text-sm font-medium text-cv-text outline-none"
                          :min="minCropNaturalHeight"
                          :max="imageNaturalHeight || 1"
                          :value="cropHeightDraft"
                          @focus="handleCropDimensionFocus('height')"
                          @input="handleCropDimensionInput('height', $event)"
                          @blur="commitCropDimensionInput('height')"
                          @keydown.enter.prevent="commitCropDimensionInput('height')"
                        />
                        <span class="ml-1 text-[11px] text-cv-text-muted">px</span>
                      </span>
                    </label>
                  </div>
                </div>
              </div>

              <p v-if="cropError" class="mt-3 text-[12px] leading-5 text-cv-danger">{{ cropError }}</p>

              <div class="mt-5 flex gap-2">
                <button
                  class="h-10 flex-1 rounded-cv-md bg-cv-accent px-3 text-sm font-medium text-white transition-colors enabled:hover:bg-cv-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                  :disabled="!canApplyCrop"
                  @click="applyCrop"
                >
                  {{ applyingCrop ? '处理中...' : '应用裁剪' }}
                </button>
                <button
                  class="h-10 rounded-cv-md border border-cv-border px-3 text-sm font-medium text-cv-text-secondary transition-colors hover:bg-cv-hover"
                  @click="createDefaultCropBox"
                >
                  重置
                </button>
              </div>
            </aside>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.lightbox-enter-active,
.lightbox-leave-active {
  transition: opacity 0.2s ease;
}
.lightbox-enter-from,
.lightbox-leave-to {
  opacity: 0;
}

.crop-handle {
  position: absolute;
  width: 18px;
  height: 18px;
  appearance: none;
  background: transparent;
  border: 0 solid white;
  padding: 0;
  touch-action: none;
}

.crop-handle-nw {
  left: -8px;
  top: -8px;
  border-left-width: 3px;
  border-top-width: 3px;
  cursor: nwse-resize;
}

.crop-handle-ne {
  right: -8px;
  top: -8px;
  border-right-width: 3px;
  border-top-width: 3px;
  cursor: nesw-resize;
}

.crop-handle-sw {
  left: -8px;
  bottom: -8px;
  border-left-width: 3px;
  border-bottom-width: 3px;
  cursor: nesw-resize;
}

.crop-handle-se {
  right: -8px;
  bottom: -8px;
  border-right-width: 3px;
  border-bottom-width: 3px;
  cursor: nwse-resize;
}
</style>
