<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { completeZhihuCallback } from '../services/api'
import { getZhihuRedirectUri, ZHIHU_AFTER_AUTH_KEY, ZHIHU_OAUTH_STATE_KEY } from '../utils/zhihuAuth'

const router = useRouter()
const callbackError = ref('')

onMounted(async () => {
  const params = new URLSearchParams(window.location.search)
  const code = params.get('code') || params.get('authorization_code') || params.get('auth_code') || ''
  const returnedState = params.get('state') || ''
  const errorCode = params.get('error') || params.get('error_code') || ''
  const errorDescription = params.get('error_description') || params.get('error_msg') || params.get('message') || ''
  const savedState = window.sessionStorage.getItem(ZHIHU_OAUTH_STATE_KEY)
  const state = returnedState || savedState || ''

  try {
    if (errorCode || errorDescription) {
      throw new Error(`知乎授权失败：${errorDescription || errorCode}`)
    }
    if (!code) {
      const query = window.location.search || '无查询参数'
      throw new Error(`知乎授权回调缺少 code。当前回调参数：${query}`)
    }
    if (!state || (returnedState && returnedState !== savedState)) {
      throw new Error('知乎授权状态校验失败，请重新登录。')
    }

    await completeZhihuCallback({
      code,
      state,
      redirect_uri: getZhihuRedirectUri(),
    })

    const afterAuth = window.sessionStorage.getItem(ZHIHU_AFTER_AUTH_KEY)
    window.sessionStorage.removeItem(ZHIHU_OAUTH_STATE_KEY)
    window.sessionStorage.removeItem(ZHIHU_AFTER_AUTH_KEY)

    if (afterAuth === 'voice') {
      router.replace({ path: '/kanshan', query: { start: 'voice' } })
      return
    }
    router.replace('/kanshan')
  } catch (error) {
    window.sessionStorage.removeItem(ZHIHU_OAUTH_STATE_KEY)
    window.sessionStorage.removeItem(ZHIHU_AFTER_AUTH_KEY)
    callbackError.value = error instanceof Error ? error.message : '知乎登录失败，请重新尝试。'
  }
})
</script>

<template>
  <main class="callback-page">
    <section class="callback-panel" role="status" aria-live="polite">
      <p class="callback-kicker">Zhihu OAuth</p>
      <h1>{{ callbackError ? '登录未完成' : '正在完成知乎登录' }}</h1>
      <p>{{ callbackError || '请稍候，页面会自动返回刘看山语音入口。' }}</p>
      <button v-if="callbackError" type="button" @click="router.replace('/kanshan')">返回刘看山页面</button>
    </section>
  </main>
</template>

<style scoped>
.callback-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: #eef3fa;
  color: #1a1a1a;
  padding: 32px;
}

.callback-panel {
  width: min(520px, 100%);
  padding: 34px;
  border: 1px solid rgba(255, 255, 255, 0.72);
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.72);
  box-shadow: 0 18px 32px rgba(0, 132, 255, 0.12);
}

.callback-kicker {
  margin: 0 0 12px;
  color: #0084ff;
  font-size: 13px;
  font-weight: 800;
}

.callback-panel h1 {
  margin: 0;
  font-size: 34px;
  line-height: 1.2;
}

.callback-panel p:not(.callback-kicker) {
  margin: 18px 0 0;
  color: #4a4a4a;
  font-size: 15px;
  line-height: 25px;
}

.callback-panel button {
  height: 44px;
  margin-top: 24px;
  padding: 0 18px;
  border: 0;
  border-radius: 14px;
  background: #0084ff;
  color: white;
  cursor: pointer;
  font: inherit;
  font-weight: 700;
}
</style>
