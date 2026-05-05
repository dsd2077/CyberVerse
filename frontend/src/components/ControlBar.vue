<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import type { ConnectionState } from '../composables/useWebRTC'

defineProps<{
  connectionState: ConnectionState
  isMuted: boolean
}>()

const emit = defineEmits<{
  connect: []
  disconnect: []
  toggleMute: []
}>()

const { t } = useI18n()
</script>

<template>
  <div class="control-bar">
    <button
      v-if="connectionState === 'disconnected' || connectionState === 'error'"
      class="btn btn-primary"
      @click="$emit('connect')"
    >
      {{ t('controlBar.connect') }}
    </button>
    <button
      v-else-if="connectionState === 'connected'"
      class="btn btn-danger"
      @click="$emit('disconnect')"
    >
      {{ t('controlBar.disconnect') }}
    </button>
    <button v-else class="btn" disabled>
      {{ t('controlBar.connecting') }}
    </button>

    <button
      class="btn"
      :class="{ 'btn-muted': isMuted }"
      @click="$emit('toggleMute')"
      :disabled="connectionState !== 'connected'"
    >
      {{ isMuted ? t('controlBar.unmute') : t('controlBar.mute') }}
    </button>

    <span class="connection-status" :class="connectionState">
      {{ connectionState }}
    </span>
  </div>
</template>

<style scoped>
.control-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: #1e1e1e;
  border-radius: 12px;
}

.btn {
  padding: 8px 20px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  background: #333;
  color: white;
}
.btn:disabled {
  opacity: 0.5;
  cursor: default;
}
.btn-primary { background: #2563eb; }
.btn-danger { background: #dc2626; }
.btn-muted { background: #ca8a04; }

.connection-status {
  margin-left: auto;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 10px;
}
.connection-status.connected { background: #166534; color: #86efac; }
.connection-status.connecting { background: #854d0e; color: #fde047; }
.connection-status.disconnected { background: #333; color: #999; }
.connection-status.error { background: #7f1d1d; color: #fca5a5; }
</style>
