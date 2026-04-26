<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'

const props = defineProps<{
  displayMode?: 'webrtc' | 'standby' | 'placeholder'
  standbySrc?: string
  standbySources?: string[]
}>()

const videoRef = ref<HTMLVideoElement | null>(null)
const standbyVideoRef = ref<HTMLVideoElement | null>(null)
const containerRef = ref<HTMLDivElement | null>(null)
const currentIndex = ref(0)
const currentDisplayMode = computed(() => props.displayMode || 'placeholder')
const activeAspectRatio = ref<number | null>(null)
const containerWidth = ref(0)
const containerHeight = ref(0)
let containerResizeObserver: ResizeObserver | null = null

// Merge standbySources and standbySrc into one effective list
const effectiveSources = computed(() => {
  if (props.standbySources?.length) return props.standbySources
  if (props.standbySrc) return [props.standbySrc]
  return []
})

const currentStandbySrc = computed(() => {
  const srcs = effectiveSources.value
  if (srcs.length === 0) return ''
  return srcs[currentIndex.value % srcs.length]
})

const shouldLoop = computed(() => effectiveSources.value.length <= 1)

const stageStyle = computed(() => {
  const width = containerWidth.value
  const height = containerHeight.value
  const aspectRatio = activeAspectRatio.value
  if (!width || !height || !aspectRatio) {
    return undefined
  }

  const containerAspectRatio = width / height
  if (containerAspectRatio > aspectRatio) {
    return {
      width: `${Math.round(height * aspectRatio)}px`,
      height: `${height}px`,
    }
  }

  return {
    width: `${width}px`,
    height: `${Math.round(width / aspectRatio)}px`,
  }
})

function onStandbyEnded() {
  const srcs = effectiveSources.value
  if (srcs.length <= 1) return // single video uses native loop
  currentIndex.value = (currentIndex.value + 1) % srcs.length
}

function syncContainerSize() {
  const el = containerRef.value
  containerWidth.value = el?.clientWidth ?? 0
  containerHeight.value = el?.clientHeight ?? 0
}

function getIntrinsicAspectRatio(videoEl: HTMLVideoElement | null): number | null {
  if (!videoEl) return null
  const { videoWidth, videoHeight } = videoEl
  if (!videoWidth || !videoHeight) return null
  return videoWidth / videoHeight
}

function syncActiveAspectRatio() {
  const primary =
    currentDisplayMode.value === 'standby'
      ? standbyVideoRef.value
      : videoRef.value

  const fallback =
    currentDisplayMode.value === 'standby'
      ? videoRef.value
      : standbyVideoRef.value

  const aspectRatio =
    getIntrinsicAspectRatio(primary) ??
    getIntrinsicAspectRatio(fallback)

  if (aspectRatio) {
    activeAspectRatio.value = aspectRatio
  }
}

function handleVideoIntrinsicSizeChange() {
  syncActiveAspectRatio()
}

function bindVideoSizeListeners(nextEl: HTMLVideoElement | null, prevEl: HTMLVideoElement | null) {
  if (prevEl) {
    prevEl.removeEventListener('loadedmetadata', handleVideoIntrinsicSizeChange)
    prevEl.removeEventListener('resize', handleVideoIntrinsicSizeChange)
  }
  if (nextEl) {
    nextEl.addEventListener('loadedmetadata', handleVideoIntrinsicSizeChange)
    nextEl.addEventListener('resize', handleVideoIntrinsicSizeChange)
  }
  syncActiveAspectRatio()
}

// Reset index when sources change
watch(effectiveSources, () => {
  currentIndex.value = 0
})

watch(videoRef, (nextEl, prevEl) => {
  bindVideoSizeListeners(nextEl, prevEl)
})

watch(standbyVideoRef, (nextEl, prevEl) => {
  bindVideoSizeListeners(nextEl, prevEl)
})

// Play when switching to standby mode or when source changes
watch(
  () => [currentDisplayMode.value, currentStandbySrc.value] as const,
  async ([mode, src]) => {
    syncActiveAspectRatio()
    if (mode !== 'standby' || !src) return
    await nextTick()
    const el = standbyVideoRef.value
    if (!el) return
    try {
      await el.play()
    } catch {
      /* ignore autoplay / visibility restrictions */
    }
  },
)

onMounted(async () => {
  await nextTick()
  syncContainerSize()
  syncActiveAspectRatio()

  if (!containerRef.value || typeof ResizeObserver === 'undefined') {
    return
  }

  containerResizeObserver = new ResizeObserver(() => {
    syncContainerSize()
  })
  containerResizeObserver.observe(containerRef.value)
})

onUnmounted(() => {
  bindVideoSizeListeners(null, videoRef.value)
  bindVideoSizeListeners(null, standbyVideoRef.value)
  containerResizeObserver?.disconnect()
})

defineExpose({ videoRef })
</script>

<template>
  <div ref="containerRef" class="video-container">
    <div class="video-stage" :style="stageStyle">
      <video
        ref="videoRef"
        autoplay
        playsinline
        class="video-element"
        :class="{ 'video-hidden': currentDisplayMode === 'standby' }"
      />
      <video
        v-if="currentStandbySrc"
        ref="standbyVideoRef"
        :src="currentStandbySrc"
        muted
        autoplay
        :loop="shouldLoop"
        playsinline
        preload="auto"
        class="video-element"
        :class="{ 'video-hidden': currentDisplayMode !== 'standby' }"
        @ended="onStandbyEnded"
      />
    </div>
  </div>
</template>

<style scoped>
.video-container {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #1a1a1a;
  border-radius: 8px;
  overflow: hidden;
}

.video-stage {
  position: relative;
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  border-radius: inherit;
  overflow: hidden;
  background: #000;
}

.video-element {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center;
  transition: opacity 180ms ease;
}

.video-hidden {
  opacity: 0;
  pointer-events: none;
}
</style>
