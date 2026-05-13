<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { createSession, getCharacter } from '../services/api'
import type { PipelineMode } from '../types'
import { buildSessionLaunchState, saveSessionLaunchState } from '../utils/sessionLaunchState'

const router = useRouter()

const KANSHAN_CHARACTER_ID = 'b3bf8345-21c9-463a-bf0c-dac7a034476c'
const KANSHAN_RETURN_PATH = '/kanshan'
const GITHUB_URL = 'https://github.com/dsd2077/CyberVerse'
const DEMO_VIDEO_URL = ''
const idleVideoSrc = '/liukanshan/idle-preview.mp4'

const connecting = ref(false)
const launchError = ref('')
const incomingCallVisible = ref(true)

const features = [
  {
    label: 'VOICE FIRST',
    title: '实时语音聊天',
    text: '用户以自然语音发起问题；系统支持实时响应、会话打断和语音/文本混合输入。语音是陪伴式交互的主入口，不是文字聊天的附属按钮。',
  },
  {
    label: 'MULTI AGENT',
    title: '数字人 Agent 系统',
    text: 'PersonaAgent 始终负责前台对话：看山快速接住用户。调研、搜索、整理和 HTML 报告生成由后台 SubAgent 执行，更适合语音场景。',
  },
  {
    label: 'ZHIHU TOOLS',
    title: '知乎生态工具链',
    text: 'SubAgent 已接入知乎搜索、全网搜索、知乎直答、知乎热榜，并保留继续扩展知乎接口的空间。',
  },
  {
    label: 'MEMORY + RAG',
    title: '跨会话记忆与 RAG 联动',
    text: '会话历史会按角色持久化，新的语音会话会加载最近对话上下文；角色素材库检索结果会被注入回答，让人物背景和对话连续性更稳定。',
  },
  {
    label: 'KNOWLEDGE',
    title: '素材导入与角色知识库',
    text: '支持单文件、多文件和文件夹导入；文本、Markdown、JSON、PDF、Word 可索引，用于知识、文档和人物生平类素材问答。',
  },
]

async function enterKanshanVoiceCall() {
  if (connecting.value) return

  connecting.value = true
  launchError.value = ''

  try {
    let launchMode: PipelineMode = 'omni'
    try {
      const character = await getCharacter(KANSHAN_CHARACTER_ID)
      launchMode = character.mode || launchMode
    } catch {
      // Keep the landing CTA usable if the character detail endpoint is temporarily unavailable.
    }

    const response = await createSession(KANSHAN_CHARACTER_ID, launchMode)
    response.warnings?.forEach((warning) => {
      console.warn('[CyberVerse]', warning)
    })
    saveSessionLaunchState(buildSessionLaunchState(response, KANSHAN_CHARACTER_ID, launchMode, KANSHAN_RETURN_PATH))
    router.push(`/session/${response.session_id}`)
  } catch (error) {
    launchError.value = error instanceof Error ? error.message : '语音通话启动失败，请检查服务状态。'
  } finally {
    connecting.value = false
  }
}

function watchDemo() {
  if (!DEMO_VIDEO_URL) return
  window.open(DEMO_VIDEO_URL, '_blank', 'noopener')
}

function rejectIncomingCall() {
  incomingCallVisible.value = false
}
</script>

<template>
  <main class="liu-page">
    <div class="glow glow-right" />
    <div class="glow glow-left" />
    <div class="glow glow-mid" />

    <section class="hero section-shell" aria-labelledby="hero-title">
      <div class="hero-copy">
        <div class="eyebrow">
          <span>刘看山专属部署</span>
          <i aria-hidden="true" />
        </div>

        <h1 id="hero-title">
          <span>和刘看山</span>
          <strong>实时语音聊天</strong>
        </h1>

        <p class="hero-lead">
          语音天然比文字更适合陪伴与协作。<br>
          用户开口，看山立即响应；复杂检索、整理和报告交给后台 SubAgent 完成。
        </p>

        <div class="hero-actions">
          <button class="primary-action" type="button" :disabled="connecting" @click="enterKanshanVoiceCall">
            <span>{{ connecting ? '正在进入...' : '开始语音通话' }}</span>
            <i aria-hidden="true">›</i>
          </button>

          <button class="secondary-action" type="button" @click="watchDemo">
            <span>观看演示</span>
            <i aria-hidden="true">▶</i>
          </button>

          <a class="github-action" :href="GITHUB_URL" target="_blank" rel="noreferrer" aria-label="GitHub 开源地址">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path
                fill="currentColor"
                d="M12 2C6.48 2 2 6.59 2 12.25c0 4.52 2.87 8.35 6.84 9.71.5.09.68-.22.68-.49v-1.9c-2.78.62-3.37-1.21-3.37-1.21-.45-1.18-1.11-1.5-1.11-1.5-.91-.64.07-.63.07-.63 1 .07 1.53 1.06 1.53 1.06.9 1.57 2.36 1.12 2.93.85.09-.67.35-1.12.63-1.38-2.22-.26-4.56-1.14-4.56-5.06 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05A9.35 9.35 0 0 1 12 6.95c.85 0 1.7.12 2.5.34 1.9-1.33 2.74-1.05 2.74-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.93-2.34 4.8-4.57 5.05.36.32.68.94.68 1.9v2.82c0 .27.18.59.69.49A10.17 10.17 0 0 0 22 12.25C22 6.59 17.52 2 12 2Z"
              />
            </svg>
          </a>
        </div>

        <p v-if="launchError" class="launch-error" role="alert">{{ launchError }}</p>
      </div>

      <aside class="video-panel" aria-label="刘看山实时视频区">
        <div v-if="incomingCallVisible" class="incoming-call-card" aria-live="polite">
          <div class="caller-avatar">
            <img src="/liukanshan/idle-poster.png" alt="">
          </div>
          <div class="incoming-call-copy">
            <strong>看山邀请你语音通话</strong>
            <span>语音通话邀请</span>
          </div>
          <div class="ringing-bars" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
        </div>

        <div class="video-stage">
          <video
            class="idle-video"
            :src="idleVideoSrc"
            poster="/liukanshan/idle-poster.png"
            autoplay
            muted
            loop
            playsinline
          />
          <div v-if="incomingCallVisible" class="call-controls" aria-label="语音通话操作">
            <button class="call-control decline" type="button" @click="rejectIncomingCall">
              <span aria-hidden="true">
                <svg viewBox="0 0 24 24">
                  <path d="M6.6 10.8c3.5-2.3 7.3-2.3 10.8 0l1.7 1.1c.6.4.8 1.1.5 1.8l-.8 1.9c-.3.7-1.1 1-1.8.8l-2.4-.8c-.5-.2-.9-.6-1-1.1l-.2-1.1a8.6 8.6 0 0 0-2.8 0l-.2 1.1c-.1.5-.5.9-1 1.1l-2.4.8c-.7.2-1.5-.1-1.8-.8l-.8-1.9c-.3-.7-.1-1.4.5-1.8l1.7-1.1Z" />
                </svg>
              </span>
              <em>拒绝</em>
            </button>
            <button class="call-control accept" type="button" :disabled="connecting" @click="enterKanshanVoiceCall">
              <span aria-hidden="true">
                <svg viewBox="0 0 24 24">
                  <path d="M6.6 10.8c3.5-2.3 7.3-2.3 10.8 0l1.7 1.1c.6.4.8 1.1.5 1.8l-.8 1.9c-.3.7-1.1 1-1.8.8l-2.4-.8c-.5-.2-.9-.6-1-1.1l-.2-1.1a8.6 8.6 0 0 0-2.8 0l-.2 1.1c-.1.5-.5.9-1 1.1l-2.4.8c-.7.2-1.5-.1-1.8-.8l-.8-1.9c-.3-.7-.1-1.4.5-1.8l1.7-1.1Z" />
                </svg>
              </span>
              <em>{{ connecting ? '接听中' : '接听' }}</em>
            </button>
          </div>
        </div>
      </aside>
    </section>

    <section class="asset-band section-shell" aria-labelledby="asset-title">
      <div class="asset-image">
        <img src="/liukanshan/poses.png" alt="刘看山多姿态角色素材" loading="lazy">
      </div>
      <div class="asset-copy">
        <p class="section-kicker">角色形象</p>
        <h2 id="asset-title">使用专属角色素材，保持品牌识别</h2>
        <p>
          首页围绕真实角色素材、待机视频、会话截图与能力说明展开。视觉上保留清爽留白、知乎蓝和玻璃质感，让用户第一眼知道这是刘看山专属部署。
        </p>
      </div>
    </section>

    <section class="capabilities section-shell" aria-labelledby="capabilities-title">
      <p class="section-kicker">CORE CAPABILITIES</p>
      <h2 id="capabilities-title">这是数字人 Agent，不只是静态吉祥物</h2>
      <p class="section-desc">
        页面主要负责解释产品价值；真正能力来自当前 CyberVerse 的实时语音会话、PersonaAgent、SubAgent、知乎工具链、跨会话历史和角色 RAG。
      </p>

      <div class="feature-grid">
        <article v-for="(feature, index) in features" :key="feature.title" class="feature-card" :class="{ wide: index > 2 }">
          <div class="feature-head">
            <span>{{ index + 1 }}</span>
            <div>
              <p>{{ feature.label }}</p>
              <h3>{{ feature.title }}</h3>
            </div>
          </div>
          <p>{{ feature.text }}</p>
        </article>
      </div>
    </section>

    <section class="live-section section-shell" aria-labelledby="live-title">
      <div class="section-heading">
        <div>
          <p class="section-kicker">LIVE SESSION</p>
          <h2 id="live-title">看山一直在线，后台任务安静推进</h2>
          <p class="section-desc">
            语音界面保留即时陪伴感；当用户提出搜索、调研或整理任务时，SubAgent 进度和产物会在对话侧呈现，不打断当前语音体验。
          </p>
        </div>
      </div>

      <div class="session-gallery">
        <figure class="screenshot-card large">
          <img src="/liukanshan/session-a.png" alt="刘看山语音会话与后台任务截图" loading="lazy">
        </figure>
        <figure class="screenshot-card large">
          <img src="/liukanshan/session-b.png" alt="语音会话中的 SubAgent 任务进度截图" loading="lazy">
        </figure>
      </div>
    </section>

    <section class="architecture section-shell" aria-labelledby="architecture-title">
      <div class="architecture-copy">
        <p class="section-kicker">AGENT ARCHITECTURE</p>
        <h2 id="architecture-title">看山负责回应，SubAgent 负责干活</h2>
        <p>
          数字人主Agent采用实时语音模型，判断用户意图：闲聊直接回答；搜索、热点、调研、报告等任务则下发给SubAgent,任务执行完成再汇报给主Agent,再由主Agent整理信息之后播报给用户。
        </p>
      </div>

      <div class="flow-map" aria-label="语音 Agent 工作流">
        <svg class="flow-lines" viewBox="0 0 600 320" preserveAspectRatio="none" aria-hidden="true">
          <path d="M108 99 H150" />
          <path d="M320 94 H372" />
          <path d="M457 130 V190" />
          <path d="M69 190 H531" />
          <path d="M69 190 V234" />
          <path d="M223 190 V234" />
          <path d="M377 190 V234" />
          <path d="M531 190 V234" />
        </svg>
        <div class="flow-node user">用户语音</div>
        <div class="flow-node persona">PersonaAgent<br>刘看山前台响应</div>
        <div class="flow-node subagent">SubAgent<br>后台执行</div>
        <div class="tool-row">
          <span>知乎搜索</span>
          <span>全网搜索</span>
          <span>知乎直答</span>
          <span>知乎热榜</span>
        </div>
      </div>
    </section>

    <section class="knowledge-section section-shell" aria-labelledby="knowledge-title">
      <p class="section-kicker">PERSONA &amp; KNOWLEDGE</p>
      <h2 id="knowledge-title">角色风格、长期上下文和素材库一起工作</h2>
      <p class="section-desc">
        人设配置决定“看山怎么说”；会话历史提供跨会话连续性；角色知识库和 RAG 负责把导入的文档、人物背景、知识资料变成可检索上下文。
      </p>

      <div class="knowledge-gallery">
        <figure class="screenshot-card">
          <img src="/liukanshan/persona.png" alt="刘看山人设与提示词配置截图">
        </figure>
        <figure class="screenshot-card">
          <img src="/liukanshan/knowledge.png" alt="刘看山素材库与 RAG 配置截图">
        </figure>
      </div>
    </section>

    <section class="bottom-cta section-shell" aria-labelledby="cta-title">
      <div>
        <h2 id="cta-title">让刘看山成为能聊、能查、能记住的网页端数字人</h2>
        <p>静态页面负责建立信任和解释能力；开始语音通话按钮负责进入现有实时会话。</p>
      </div>
      <button class="primary-action" type="button" :disabled="connecting" @click="enterKanshanVoiceCall">
        <span>{{ connecting ? '正在进入...' : '开始语音通话' }}</span>
        <i aria-hidden="true">›</i>
      </button>
    </section>

    <p class="legal-note">AI 生成内容仅供体验参考 · 通话、历史和素材能力以当前部署配置为准</p>
  </main>
</template>

<style scoped>
.liu-page {
  --blue: #0084ff;
  --ink: #1a1a1a;
  --muted: #4a4a4a;
  --soft: rgba(255, 255, 255, 0.64);
  --stroke: rgba(255, 255, 255, 0.72);
  position: relative;
  min-height: 100vh;
  overflow: hidden;
  background:
    radial-gradient(circle at 88% 0%, rgba(0, 132, 255, 0.12), transparent 30%),
    radial-gradient(circle at 8% 26%, rgba(255, 255, 255, 0.9), transparent 25%),
    #eef3fa;
  color: var(--ink);
  padding: 42px 0 56px;
}

.glow {
  position: absolute;
  border-radius: 999px;
  pointer-events: none;
  filter: blur(78px);
}

.glow-right {
  top: -250px;
  right: 0;
  width: 900px;
  height: 760px;
  background: rgba(0, 132, 255, 0.16);
}

.glow-left {
  top: 520px;
  left: -260px;
  width: 760px;
  height: 620px;
  background: rgba(255, 255, 255, 0.86);
}

.glow-mid {
  top: 1920px;
  right: 0;
  width: 720px;
  height: 620px;
  background: rgba(0, 132, 255, 0.08);
}

.section-shell {
  position: relative;
  z-index: 1;
  width: min(1296px, calc(100% - 96px));
  margin: 0 auto;
}

.hero {
  min-height: 700px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 520px;
  align-items: center;
  gap: 62px;
}

.hero-copy {
  padding-top: 18px;
}

.eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 12px;
  height: 34px;
  padding: 0 16px 0 18px;
  border: 1px solid rgba(255, 255, 255, 0.68);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.72);
  box-shadow: 0 6px 14px rgba(0, 132, 255, 0.07);
  color: var(--ink);
  font-size: 14px;
  font-weight: 500;
}

.eyebrow i {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--blue);
}

.hero h1 {
  margin: 34px 0 0;
  font-size: clamp(58px, 5.4vw, 78px);
  line-height: 1.13;
  font-weight: 800;
}

.hero h1 span,
.hero h1 strong {
  display: block;
}

.hero h1 strong {
  color: var(--blue);
  font-weight: 800;
}

.hero-lead {
  margin: 30px 0 0 4px;
  max-width: 600px;
  color: var(--muted);
  font-size: 18px;
  line-height: 31px;
}

.hero-actions {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-top: 74px;
}

.primary-action,
.secondary-action,
.github-action {
  border: 0;
  cursor: pointer;
  font: inherit;
}

.primary-action,
.secondary-action {
  height: 58px;
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  border-radius: 18px;
  font-size: 16px;
  font-weight: 700;
  white-space: nowrap;
  transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease;
}

.primary-action {
  min-width: 218px;
  padding: 0 14px 0 26px;
  background: var(--blue);
  color: white;
  box-shadow: 0 10px 24px rgba(0, 132, 255, 0.24);
}

.primary-action i,
.secondary-action i {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 13px;
  font-style: normal;
  flex: 0 0 auto;
}

.primary-action i {
  background: white;
  color: var(--blue);
  font-size: 24px;
  line-height: 1;
}

.secondary-action {
  min-width: 160px;
  padding: 0 14px 0 24px;
  border: 1px solid rgba(26, 26, 26, 0.1);
  background: rgba(255, 255, 255, 0.82);
  color: var(--ink);
}

.secondary-action i {
  background: rgba(0, 132, 255, 0.1);
  color: var(--blue);
  font-size: 14px;
}

.github-action {
  width: 58px;
  height: 58px;
  display: grid;
  place-items: center;
  border-radius: 18px;
  background: rgba(17, 19, 22, 0.96);
  color: white;
  box-shadow: 0 10px 12px rgba(26, 26, 26, 0.18);
  text-decoration: none;
  transition: transform 160ms ease, box-shadow 160ms ease;
}

.github-action svg {
  width: 25px;
  height: 25px;
}

.primary-action:hover,
.secondary-action:hover,
.github-action:hover {
  transform: translateY(-2px);
}

.primary-action:disabled {
  cursor: wait;
  opacity: 0.76;
}

.launch-error {
  width: fit-content;
  max-width: 560px;
  margin: 18px 0 0;
  padding: 10px 14px;
  border: 1px solid rgba(255, 77, 79, 0.22);
  border-radius: 12px;
  background: rgba(255, 77, 79, 0.08);
  color: #b42318;
  font-size: 13px;
}

.video-panel {
  min-height: 650px;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 26px 34px 32px;
  border: 1px solid var(--stroke);
  border-radius: 32px;
  background: rgba(255, 255, 255, 0.68);
  box-shadow: 0 18px 23px rgba(0, 132, 255, 0.12);
}

.incoming-call-card {
  width: 100%;
  max-width: 318px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 11px;
  border: 1px solid rgba(255, 255, 255, 0.78);
  border-radius: 19px;
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 14px 34px rgba(26, 26, 26, 0.11);
  backdrop-filter: blur(18px);
}

.caller-avatar {
  width: 38px;
  height: 38px;
  flex: 0 0 auto;
  overflow: hidden;
  border-radius: 12px;
  background: #dfe9f7;
}

.caller-avatar img {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
  object-position: center 34%;
}

.incoming-call-copy {
  min-width: 0;
  flex: 1;
}

.incoming-call-copy strong,
.incoming-call-copy span {
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.incoming-call-copy strong {
  color: var(--ink);
  font-size: 14px;
  line-height: 18px;
  font-weight: 800;
}

.incoming-call-copy span {
  margin-top: 2px;
  color: var(--muted);
  font-size: 11px;
  line-height: 14px;
}

.ringing-bars {
  flex: 0 0 auto;
  display: flex;
  align-items: flex-end;
  gap: 3px;
  width: 20px;
  height: 18px;
}

.ringing-bars span {
  width: 4px;
  border-radius: 999px;
  background: #1bc760;
  animation: ring-bar 900ms ease-in-out infinite;
}

.ringing-bars span:nth-child(1) {
  height: 8px;
}

.ringing-bars span:nth-child(2) {
  height: 14px;
  animation-delay: 120ms;
}

.ringing-bars span:nth-child(3) {
  height: 10px;
  animation-delay: 240ms;
}

.video-stage {
  position: relative;
  width: 354px;
  height: 552px;
  margin: 18px auto 0;
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.35);
  border-radius: 24px;
  background: #dfe9f7;
  box-shadow: 0 18px 38px rgba(26, 26, 26, 0.18);
}

.idle-video {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
}

.idle-pill {
  position: absolute;
  top: 16px;
  left: 26px;
  padding: 8px 18px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.84);
  color: var(--blue);
  font-size: 11px;
  font-weight: 800;
}

.call-controls {
  position: absolute;
  left: 50%;
  bottom: 16px;
  display: flex;
  gap: 48px;
  align-items: center;
  justify-content: center;
  width: 220px;
  transform: translateX(-50%);
}

.call-control {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 7px;
  padding: 0;
  border: 0;
  background: transparent;
  color: white;
  cursor: pointer;
  font: inherit;
}

.call-control span {
  display: grid;
  place-items: center;
  width: 54px;
  height: 54px;
  border-radius: 999px;
  box-shadow: 0 10px 22px rgba(26, 26, 26, 0.3);
  transition: transform 160ms ease, box-shadow 160ms ease, opacity 160ms ease;
}

.call-control svg {
  width: 29px;
  height: 29px;
  fill: currentColor;
}

.call-control em {
  font-style: normal;
  font-size: 13px;
  font-weight: 800;
  line-height: 18px;
  text-shadow: 0 1px 6px rgba(0, 0, 0, 0.45);
}

.call-control:hover span {
  transform: translateY(-2px);
}

.call-control:disabled {
  cursor: wait;
  opacity: 0.78;
}

.call-control.decline span {
  background: #ff4d4f;
}

.call-control.decline svg {
  transform: rotate(135deg);
}

.call-control.accept span {
  background: #20c768;
}

.call-control.accept svg {
  transform: rotate(-45deg);
}

@keyframes ring-bar {
  0%,
  100% {
    transform: scaleY(0.62);
    opacity: 0.58;
  }
  50% {
    transform: scaleY(1);
    opacity: 1;
  }
}

.asset-band,
.architecture,
.bottom-cta {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(360px, 0.85fr);
  align-items: center;
  gap: 50px;
  margin-top: 54px;
  padding: 36px 52px;
  border: 1px solid var(--stroke);
  border-radius: 34px;
  background: rgba(255, 255, 255, 0.58);
  box-shadow: 0 18px 21px rgba(26, 26, 26, 0.08);
}

.asset-image {
  overflow: hidden;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.86);
}

.asset-image img {
  width: 100%;
  height: 295px;
  display: block;
  object-fit: cover;
}

.section-kicker {
  margin: 0 0 14px;
  color: var(--blue);
  font-size: 13px;
  font-weight: 800;
}

.asset-copy h2,
.capabilities h2,
.live-section h2,
.architecture h2,
.knowledge-section h2,
.bottom-cta h2 {
  margin: 0;
  color: var(--ink);
  font-size: clamp(34px, 3.3vw, 46px);
  line-height: 1.22;
  font-weight: 800;
}

.asset-copy p:not(.section-kicker),
.architecture-copy p:not(.section-kicker),
.bottom-cta p {
  margin: 24px 0 0;
  color: var(--muted);
  font-size: 16px;
  line-height: 27px;
}

.capabilities,
.live-section,
.knowledge-section {
  margin-top: 60px;
}

.section-desc {
  max-width: 900px;
  margin: 22px 0 0;
  color: var(--muted);
  font-size: 16px;
  line-height: 26px;
}

.feature-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 32px;
  margin-top: 50px;
}

.feature-card {
  min-height: 170px;
  padding: 23px;
  border: 1px solid var(--stroke);
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.62);
  box-shadow: 0 12px 14px rgba(0, 132, 255, 0.06);
}

.feature-card.wide {
  grid-column: span 1;
}

.feature-card:nth-child(4) {
  grid-column: span 2;
}

.feature-card:nth-child(5) {
  grid-column: span 1;
}

.feature-head {
  display: flex;
  gap: 16px;
  align-items: center;
}

.feature-head > span {
  display: grid;
  place-items: center;
  width: 46px;
  height: 46px;
  flex: 0 0 auto;
  border-radius: 16px;
  background: var(--blue);
  color: white;
  font-size: 19px;
  font-weight: 800;
}

.feature-head p {
  margin: 0 0 6px;
  color: var(--blue);
  font-size: 12px;
  font-weight: 800;
}

.feature-head h3 {
  margin: 0;
  color: var(--ink);
  font-size: 23px;
  line-height: 29px;
}

.feature-card > p {
  margin: 20px 0 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 24px;
}

.session-gallery {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 72px;
  margin-top: 50px;
  align-items: start;
}

.screenshot-card {
  margin: 0;
  overflow: hidden;
  border-radius: 28px;
  background: #111316;
  box-shadow: 0 18px 42px rgba(26, 26, 26, 0.18);
}

.screenshot-card img {
  width: 100%;
  display: block;
  object-fit: cover;
}

.session-gallery .large img {
  aspect-ratio: 612 / 382;
}

.architecture {
  grid-template-columns: 0.86fr 1.14fr;
  min-height: 460px;
  margin-top: 70px;
}

.flow-map {
  position: relative;
  width: 600px;
  max-width: 100%;
  min-height: 320px;
  margin: 0 auto;
}

.flow-lines {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
  pointer-events: none;
  z-index: 0;
}

.flow-lines path {
  fill: none;
  stroke: rgba(0, 132, 255, 0.24);
  stroke-width: 2.4;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.flow-node {
  position: absolute;
  z-index: 1;
  display: grid;
  place-items: center;
  border: 1px solid var(--stroke);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 8px 20px rgba(0, 132, 255, 0.07);
  color: var(--ink);
  text-align: center;
  font-size: 14px;
  font-weight: 700;
  line-height: 19px;
}

.flow-node.user {
  top: 72px;
  left: 0;
  width: 108px;
  height: 54px;
}

.flow-node.persona {
  top: 58px;
  left: 150px;
  width: 170px;
  height: 72px;
  border-color: rgba(0, 132, 255, 0.72);
  background: var(--blue);
  color: white;
}

.flow-node.subagent {
  top: 58px;
  left: 372px;
  width: 170px;
  height: 72px;
}

.tool-row {
  position: absolute;
  z-index: 1;
  left: 0;
  right: 0;
  bottom: 32px;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.tool-row span {
  display: grid;
  place-items: center;
  height: 54px;
  border: 1px solid var(--stroke);
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 8px 20px rgba(0, 132, 255, 0.07);
  color: var(--ink);
  font-size: 14px;
  font-weight: 700;
}

.knowledge-gallery {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 72px;
  margin-top: 50px;
}

.knowledge-gallery .screenshot-card img {
  aspect-ratio: 612 / 456;
}

.bottom-cta {
  grid-template-columns: 1fr auto;
  margin-top: 70px;
}

.bottom-cta h2 {
  font-size: 34px;
  line-height: 42px;
}

.legal-note {
  position: relative;
  z-index: 1;
  width: min(560px, calc(100% - 48px));
  margin: 34px auto 0;
  color: var(--muted);
  text-align: center;
  font-size: 12px;
  line-height: 16px;
}

@media (max-width: 1180px) {
  .section-shell {
    width: min(1040px, calc(100% - 48px));
  }

  .hero {
    grid-template-columns: 1fr;
    gap: 44px;
  }

  .video-panel {
    max-width: 456px;
  }

  .feature-grid,
  .session-gallery,
  .knowledge-gallery,
  .asset-band,
  .architecture,
  .bottom-cta {
    grid-template-columns: 1fr;
  }

  .feature-card:nth-child(4),
  .feature-card:nth-child(5) {
    grid-column: span 1;
  }

  .flow-map {
    max-width: 680px;
  }
}

@media (max-width: 760px) {
  .liu-page {
    padding-top: 28px;
  }

  .hero-actions {
    flex-wrap: wrap;
    margin-top: 42px;
  }

  .primary-action,
  .secondary-action {
    width: 100%;
  }

  .github-action {
    width: 58px;
  }

  .asset-band,
  .architecture,
  .bottom-cta {
    padding: 24px;
    border-radius: 24px;
  }

  .video-panel {
    padding: 22px;
    border-radius: 24px;
  }

  .video-stage {
    width: min(300px, 100%);
  }

  .feature-grid {
    gap: 18px;
  }

  .session-gallery,
  .knowledge-gallery {
    gap: 24px;
  }

  .tool-row {
    position: static;
    padding-top: 190px;
    grid-template-columns: 1fr 1fr;
  }

  .flow-map::after {
    display: none;
  }
}
</style>
