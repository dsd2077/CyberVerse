<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { getHealth } from '../services/api'
import LanguageSwitcher from '../components/LanguageSwitcher.vue'

const router = useRouter()
const { t } = useI18n()
const voiceOnline = ref(false)
const personaCount = ref(0)

onMounted(async () => {
  try {
    const health = await getHealth()
    voiceOnline.value = health.inference_connected
  } catch {
    voiceOnline.value = false
  }
})

function enter() {
  router.push('/characters')
}
</script>

<template>
  <div class="relative min-h-screen bg-cyber-base overflow-hidden">
    <!-- Background glows -->
    <div class="absolute -top-[150px] right-[calc(50%-120px)] w-[600px] h-[400px] rounded-full bg-cyber-cyan/8 blur-[120px] pointer-events-none" />
    <div class="absolute -top-[140px] -left-[80px] w-[500px] h-[380px] rounded-full bg-cyber-rose/6 blur-[120px] pointer-events-none" />

    <!-- Diagonal accent lines -->
    <div class="absolute top-[130px] right-[200px] w-[200px] h-px bg-gradient-to-r from-cyber-cyan/30 to-transparent rotate-[18deg]" />
    <div class="absolute bottom-[160px] left-[32px] w-[240px] h-px bg-gradient-to-r from-transparent to-cyber-cyan/20 rotate-[10deg]" />

    <!-- Nav -->
    <nav class="relative z-10 flex items-center justify-between px-12 h-16">
      <div class="flex items-center gap-4">
        <span class="text-[13px] font-bold tracking-[2.34px] uppercase text-[#f6efe8]">CyberVerse</span>
        <div class="w-12 h-px bg-cyber-cyan/40" />
        <span class="text-xs tracking-[0.96px] uppercase text-cyber-label">{{ t('landing.protocol') }}</span>
        <LanguageSwitcher />
      </div>
      <div class="flex items-center gap-3">
        <div class="flex items-center h-8 px-3 border text-xs tracking-[0.72px] uppercase"
             :class="voiceOnline
               ? 'bg-cyber-cyan/8 border-cyber-cyan/28 text-[#9ff4fb]'
               : 'bg-red-500/8 border-red-500/28 text-red-400'">
          {{ voiceOnline ? t('landing.voiceOnline') : t('landing.voiceOffline') }}
        </div>
        <div class="flex items-center h-8 px-3 bg-[rgba(12,16,20,0.95)] border border-[rgba(72,80,92,0.72)] text-[#d7dde4] text-xs cursor-pointer hover:border-cyber-cyan/40 transition-colors">
          {{ t('landing.search') }}
        </div>
      </div>
    </nav>

    <!-- Hero -->
    <div class="relative z-10 mx-12 mt-5">
      <div class="relative flex min-h-[calc(100vh-104px)] flex-col border border-cyber-border bg-gradient-to-b from-[rgba(13,15,20,0.88)] to-[rgba(7,8,12,0.94)]">
        <!-- Accent bar top-left -->
        <div class="absolute top-[17px] left-[17px] w-[78px] h-1.5 bg-cyber-cyan shadow-[0_0_18px_rgba(52,230,243,0.42)]" />

        <!-- District label -->
        <p class="text-center text-xs tracking-[1.44px] uppercase text-[#8fe8ef] pt-[60px]">
          {{ t('landing.district') }}
        </p>

        <div class="flex flex-1 flex-col px-6 pb-10">
          <div class="flex flex-1 flex-col justify-center">
            <!-- Main title -->
            <div class="relative mx-auto w-fit -translate-x-[17.5%] select-none">
              <!-- CYBER text -->
              <div class="inline-block relative">
                <span class="text-[clamp(80px,15vw,220px)] font-black tracking-[-0.08em] leading-none text-[#fbf6ef]">CYBER</span>
                <!-- Cyan line through CYBER -->
                <div class="absolute left-[15%] top-[8px] w-[45%] h-px bg-cyber-cyan/60" />
              </div>
              <!-- VERSE text overlapping -->
              <div class="relative -mt-[clamp(30px,6vw,80px)] ml-[35%]">
                <span class="inline-block text-[clamp(80px,15vw,220px)] font-black -skew-x-6 tracking-[-0.08em] leading-none text-cyber-rose drop-shadow-[0_0_30px_rgba(255,95,137,0.34)]">VERSE</span>
              </div>
            </div>

            <!-- Subtitle -->
            <p class="text-center text-lg text-[#b7bfca] mt-16 max-w-[660px] mx-auto leading-[1.82]">
              {{ t('landing.subtitle') }}
            </p>

            <!-- Enter button with lines -->
            <div class="flex items-center justify-center gap-0 mt-12">
              <div class="w-40 h-px bg-gradient-to-r from-transparent to-cyber-cyan/40" />
              <button
                @click="enter"
                class="relative px-8 h-[52px] bg-gradient-to-b from-[#ff4f79] to-[#a71333] text-[#fff6f8] text-[15px] font-extrabold tracking-[0.6px] shadow-[0_0_24px_rgba(255,79,121,0.22)] hover:shadow-[0_0_36px_rgba(255,79,121,0.35)] transition-shadow cursor-pointer"
              >
                {{ t('landing.enter') }}
              </button>
              <div class="w-40 h-px bg-gradient-to-l from-transparent to-cyber-cyan/40" />
            </div>
          </div>

          <!-- HUD footer -->
          <p class="pt-8 text-center text-[11px] tracking-[0.88px] uppercase text-cyber-muted">
            {{ t('landing.footer', { count: personaCount || 4 }) }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>
