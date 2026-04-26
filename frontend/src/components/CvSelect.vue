<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'

interface Option {
  label: string
  value: string
}

const props = defineProps<{
  modelValue: string
  options: (Option | string)[]
  placeholder?: string
  success?: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const open = ref(false)
const containerRef = ref<HTMLElement>()

const normalizedOptions = computed<Option[]>(() =>
  props.options.map(o => typeof o === 'string' ? { label: o, value: o } : o)
)

const selectedLabel = computed(
  () => normalizedOptions.value.find(o => o.value === props.modelValue)?.label ?? props.modelValue
)

function select(value: string) {
  emit('update:modelValue', value)
  open.value = false
}

function handleOutsideClick(e: MouseEvent) {
  if (containerRef.value && !containerRef.value.contains(e.target as Node)) {
    open.value = false
  }
}

onMounted(() => document.addEventListener('mousedown', handleOutsideClick))
onUnmounted(() => document.removeEventListener('mousedown', handleOutsideClick))
</script>

<template>
  <div ref="containerRef" class="relative">
    <button
      type="button"
      @click="open = !open"
      class="w-full h-[42px] bg-cv-elevated border border-cv-border rounded-cv-md px-4 text-sm text-cv-text text-left cursor-pointer transition-all focus:outline-none"
      :class="[
        props.success ? 'pr-16 border-cv-success shadow-[0_0_0_2px_rgba(34,197,94,0.15)]' : 'pr-9',
        open
          ? props.success
            ? 'border-cv-success shadow-[0_0_0_2px_rgba(34,197,94,0.15)]'
            : 'border-cv-accent shadow-[0_0_0_2px_rgba(59,130,246,0.15)]'
          : props.success
            ? 'border-cv-success hover:border-cv-success'
            : 'hover:border-cv-text-muted',
      ]"
    >
      <span :class="selectedLabel ? 'text-cv-text' : 'text-cv-text-muted'">
        {{ selectedLabel || placeholder || '' }}
      </span>
    </button>

    <span
      v-if="props.success"
      class="pointer-events-none absolute right-8 top-1/2 -translate-y-1/2 text-cv-success"
    >
      <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
        <path d="M3.5 8.5l3 3 6-6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
    </span>

    <!-- Chevron -->
    <span
      class="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 transition-transform duration-150"
      :class="open ? 'rotate-180' : ''"
    >
      <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
        <path d="M1 1l4 4 4-4" stroke="#8b8b9e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </span>

    <!-- Dropdown panel -->
    <Transition
      enter-active-class="transition-all duration-150 ease-out"
      enter-from-class="opacity-0 translate-y-[-4px]"
      enter-to-class="opacity-100 translate-y-0"
      leave-active-class="transition-all duration-100 ease-in"
      leave-from-class="opacity-100 translate-y-0"
      leave-to-class="opacity-0 translate-y-[-4px]"
    >
      <div
        v-if="open"
        class="absolute z-50 top-full left-0 right-0 mt-1 bg-cv-elevated border border-cv-border rounded-cv-md overflow-y-auto shadow-[0_8px_24px_rgba(0,0,0,0.5)]"
        style="max-height: 216px;"
      >
        <button
          v-for="opt in normalizedOptions"
          :key="opt.value"
          type="button"
          @click="select(opt.value)"
          class="w-full px-4 py-2.5 text-sm text-left cursor-pointer transition-colors"
          :class="opt.value === modelValue
            ? 'bg-cv-accent-muted text-cv-accent font-medium'
            : 'text-cv-text hover:bg-cv-hover'"
        >
          {{ opt.label }}
        </button>
      </div>
    </Transition>
  </div>
</template>
