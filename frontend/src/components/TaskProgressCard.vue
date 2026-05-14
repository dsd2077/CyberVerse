<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import type { ChatTaskArtifact, ChatTaskTimelineItem, ChatTaskState, TaskStatus } from '../composables/useChat'

const props = defineProps<{
  task: ChatTaskState
}>()

const { t } = useI18n()
const expanded = ref(false)

const progressStyle = computed(() => ({
  width: `${Math.min(100, Math.max(0, props.task.progress))}%`,
}))

const statusLabel = computed(() => {
  const labels: Record<TaskStatus, string> = {
    queued: t('task.status.queued'),
    running: t('task.status.running'),
    waiting_user: t('task.status.waitingUser'),
    completed: t('task.status.completed'),
    failed: t('task.status.failed'),
    cancelled: t('task.status.cancelled'),
  }
  return labels[props.task.status] || t('task.status.running')
})

const taskMeta = computed(() => {
  const shortId = props.task.id.length > 14 ? `${props.task.id.slice(0, 8)}...` : props.task.id
  return t('task.meta', { id: shortId, count: props.task.eventCount })
})

function toggleExpanded() {
  expanded.value = !expanded.value
}

function handleToggleKeydown(event: KeyboardEvent) {
  if (event.key !== 'Enter' && event.key !== ' ') return
  event.preventDefault()
  toggleExpanded()
}

function artifactTypeLabel(artifact: ChatTaskArtifact): string {
  const type = artifact.type.toLowerCase()
  if (type.includes('html')) return 'HTML'
  if (type.includes('markdown') || type === 'md') return 'MD'
  return (artifact.type || 'HTML').toUpperCase()
}

function artifactHint(artifact: ChatTaskArtifact): string {
  return artifactTypeLabel(artifact) === 'HTML' ? t('task.artifactPreviewHint') : t('task.artifactOpenHint')
}

function isDoneEvent(event: ChatTaskTimelineItem): boolean {
  return ['completed', 'failed', 'cancelled'].includes(props.task.status) || event.progress < props.task.progress
}
</script>

<template>
  <div class="task-card-stack">
    <article
      class="task-card"
      :class="[`status-${task.status}`, { expanded }]"
      role="button"
      tabindex="0"
      :aria-expanded="expanded"
      @click="toggleExpanded"
      @keydown="handleToggleKeydown"
    >
      <div class="task-header">
        <div class="task-heading">
          <div class="agent-name">{{ task.agentName }}</div>
          <div class="task-title">{{ task.title }}</div>
        </div>
        <div class="task-status">{{ statusLabel }}</div>
      </div>

      <div class="task-meta-row">
        <span class="task-meta">{{ taskMeta }}</span>
        <span class="task-chevron">{{ expanded ? '⌃' : '⌄' }}</span>
      </div>

      <div class="task-progress-row">
        <div class="progress-track">
          <div class="progress-bar" :style="progressStyle" />
        </div>
        <span class="progress-value">{{ task.progress }}%</span>
      </div>

      <div class="current-step">{{ task.currentStep }}</div>

      <div v-if="expanded" class="task-timeline">
        <div class="timeline-title">{{ t('task.timelineTitle') }}</div>
        <div v-if="task.events.length" class="timeline-list">
          <div
            v-for="event in task.events"
            :key="`${event.seq}-${event.eventType}`"
            class="timeline-item"
            :class="{ done: isDoneEvent(event), current: event.seq === task.events[task.events.length - 1]?.seq }"
          >
            <span class="timeline-dot" />
            <div class="timeline-copy">
              <span class="timeline-event-title">{{ event.title }}</span>
              <span class="timeline-event-desc">{{ event.description }}</span>
            </div>
          </div>
        </div>
        <div v-else class="timeline-empty">{{ t('task.timelineEmpty') }}</div>
      </div>
    </article>

    <article
      v-for="artifact in task.artifacts"
      :key="artifact.id"
      class="artifact-card"
    >
      <div class="artifact-main">
        <div class="artifact-label">{{ t('task.artifactLabel') }}</div>
        <div class="artifact-title">{{ artifact.title }}</div>
        <div class="artifact-meta">
          <span class="artifact-type">{{ artifactTypeLabel(artifact) }}</span>
          <span>{{ artifactHint(artifact) }}</span>
        </div>
      </div>
      <div class="artifact-actions">
        <a class="artifact-btn secondary" :href="artifact.url" target="_blank" rel="noreferrer">{{ t('task.preview') }}</a>
        <a class="artifact-btn primary" :href="artifact.url" target="_blank" rel="noreferrer">{{ t('task.open') }}</a>
      </div>
    </article>
  </div>
</template>

<style scoped>
.task-card-stack {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.task-card {
  box-sizing: border-box;
  width: 100%;
  padding: 14px 16px 12px;
  background: #19191c;
  border: 1px solid #24436e;
  border-radius: 10px;
  cursor: pointer;
  color: #f0f0f5;
  box-shadow: 0 6px 18px -8px rgba(0, 0, 0, 0.12);
  transition: border-color 160ms ease, background 160ms ease;
}

.task-card:hover,
.task-card:focus-visible {
  border-color: #335d9e;
  outline: none;
}

.task-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.task-heading {
  min-width: 0;
}

.agent-name {
  color: #9abcfb;
  font-size: 12px;
  font-weight: 500;
  line-height: 16px;
}

.task-title {
  margin-top: 5px;
  color: #f0f0f5;
  font-size: 15px;
  font-weight: 600;
  line-height: 21px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-status {
  flex: 0 0 auto;
  min-width: 58px;
  padding: 3px 10px;
  border: 1px solid #1f5e3a;
  border-radius: 999px;
  background: #11251a;
  color: #22c55e;
  font-size: 12px;
  font-weight: 500;
  line-height: 16px;
  text-align: center;
}

.status-failed .task-status {
  border-color: rgba(239, 68, 68, 0.45);
  background: rgba(239, 68, 68, 0.12);
  color: #fca5a5;
}

.status-cancelled .task-status {
  border-color: #343440;
  background: #151519;
  color: #8b8b9e;
}

.task-meta-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 9px;
}

.task-meta {
  min-width: 0;
  color: #55556a;
  font-size: 11px;
  line-height: 15px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-chevron {
  flex: 0 0 auto;
  width: 18px;
  color: #8b8b9e;
  font-size: 16px;
  line-height: 18px;
  text-align: center;
}

.task-progress-row {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-top: 8px;
}

.progress-track {
  flex: 1;
  height: 5px;
  overflow: hidden;
  border-radius: 999px;
  background: #252631;
}

.progress-bar {
  height: 100%;
  border-radius: inherit;
  background: #3b82f5;
  transition: width 180ms ease, background 180ms ease;
}

.status-completed .progress-bar {
  background: #22c55e;
}

.status-failed .progress-bar {
  background: #ef4444;
}

.progress-value {
  flex: 0 0 42px;
  color: #8b8b9e;
  font-size: 12px;
  font-weight: 500;
  line-height: 17px;
}

.current-step {
  margin-top: 8px;
  color: #9abcfb;
  font-size: 12px;
  line-height: 17px;
}

.status-completed .current-step {
  color: #22c55e;
}

.status-failed .current-step {
  color: #fca5a5;
}

.task-timeline {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid #292930;
}

.timeline-title {
  color: #f0f0f5;
  font-size: 13px;
  font-weight: 600;
  line-height: 18px;
}

.timeline-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 10px;
}

.timeline-item {
  position: relative;
  display: grid;
  grid-template-columns: 10px minmax(0, 1fr);
  gap: 10px;
  min-height: 20px;
}

.timeline-item:not(:last-child)::after {
  content: '';
  position: absolute;
  left: 4px;
  top: 14px;
  width: 1px;
  height: calc(100% + 2px);
  background: #292930;
}

.timeline-dot {
  width: 6px;
  height: 6px;
  margin-top: 5px;
  border-radius: 50%;
  background: #363645;
}

.timeline-item.done .timeline-dot {
  background: #22c55e;
}

.timeline-item.current .timeline-dot {
  background: #3b82f5;
}

.status-completed .timeline-item.current .timeline-dot {
  background: #22c55e;
}

.timeline-copy {
  display: grid;
  grid-template-columns: minmax(7rem, 0.7fr) minmax(0, 1fr);
  gap: 12px;
}

.timeline-event-title {
  color: #8b8b9e;
  font-size: 12px;
  font-weight: 500;
  line-height: 16px;
}

.timeline-event-desc {
  color: #55556a;
  font-size: 11px;
  line-height: 15px;
}

.timeline-empty {
  margin-top: 10px;
  color: #55556a;
  font-size: 12px;
  line-height: 17px;
}

.artifact-card {
  box-sizing: border-box;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 13px 16px;
  background: #19191c;
  border: 1px solid #292930;
  border-radius: 10px;
}

.artifact-main {
  min-width: 0;
}

.artifact-label {
  color: #55556a;
  font-size: 12px;
  font-weight: 500;
  line-height: 16px;
}

.artifact-title {
  margin-top: 5px;
  color: #f0f0f5;
  font-size: 15px;
  font-weight: 600;
  line-height: 21px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.artifact-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 5px;
  color: #55556a;
  font-size: 12px;
  line-height: 16px;
}

.artifact-type {
  display: inline-flex;
  align-items: center;
  height: 22px;
  padding: 0 10px;
  border: 1px solid #24436e;
  border-radius: 999px;
  background: #14233d;
  color: #9abcfb;
  font-size: 11px;
  font-weight: 500;
}

.artifact-actions {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.artifact-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 44px;
  height: 28px;
  padding: 0 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 500;
  line-height: 17px;
  text-decoration: none;
}

.artifact-btn.secondary {
  border: 1px solid #343440;
  background: #151519;
  color: #8b8b9e;
}

.artifact-btn.primary {
  border: 1px solid #3b82f5;
  background: #3b82f5;
  color: #fff;
}
</style>
