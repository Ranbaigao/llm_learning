<script setup lang="ts">
/**
 * HTML 弹窗查看器：正文中的 a.html-modal-link 点击后在此弹窗内以 iframe 打开交互页面。
 * 移植自旧站 content/.assets/javascripts/html_modal.js，视觉适配暗色主题。
 */
const props = defineProps<{
  open: boolean
  href: string
  title: string
}>()

const emit = defineEmits<{ close: [] }>()

function close() {
  emit('close')
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') close()
}

watch(
  () => props.open,
  (v) => {
    if (import.meta.client) document.body.classList.toggle('html-modal-active', v)
  },
)

onMounted(() => document.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown)
  document.body.classList.remove('html-modal-active')
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="html-modal-overlay"
      role="dialog"
      aria-modal="true"
      @click.self="close"
    >
      <div class="html-modal-dialog">
        <div class="html-modal-bar">
          <span class="html-modal-title">{{ title || '交互演示' }}</span>
          <span class="html-modal-actions">
            <a
              class="html-modal-open-external"
              :href="href"
              target="_blank"
              rel="noopener"
              title="在新标签页打开"
              >↗</a
            >
            <button class="html-modal-close" type="button" title="关闭 (Esc)" @click="close">
              ✕
            </button>
          </span>
        </div>
        <!-- v-if 销毁即释放 iframe，等价于旧实现关闭时置 about:blank -->
        <iframe class="html-modal-frame" :src="href" title="HTML 弹窗内容" loading="lazy"></iframe>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.html-modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 3vh 3vw;
  background: rgba(2, 6, 23, 0.72);
  backdrop-filter: blur(3px);
  animation: html-modal-fade 0.18s ease;
}

@keyframes html-modal-fade {
  from {
    opacity: 0;
  }
}

.html-modal-dialog {
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 1200px;
  height: 100%;
  max-height: 92vh;
  background: #0f172a;
  border: 1px solid rgba(56, 189, 248, 0.25);
  border-radius: 10px;
  overflow: hidden;
  box-shadow:
    0 20px 60px rgba(0, 0, 0, 0.5),
    0 0 40px rgba(56, 189, 248, 0.08);
  animation: html-modal-pop 0.18s ease;
}

@keyframes html-modal-pop {
  from {
    transform: translateY(8px) scale(0.98);
  }
}

.html-modal-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.45rem 0.9rem;
  background: rgba(30, 41, 59, 0.85);
  border-bottom: 1px solid rgba(56, 189, 248, 0.18);
  flex: 0 0 auto;
}

.html-modal-title {
  font-size: 0.85rem;
  font-weight: 600;
  color: #e2e8f0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.html-modal-actions {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex: 0 0 auto;
}

.html-modal-open-external,
.html-modal-close {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.7rem;
  height: 1.7rem;
  border: none;
  border-radius: 0.35rem;
  background: transparent;
  color: #94a3b8;
  font-size: 0.9rem;
  text-decoration: none;
  cursor: pointer;
}

.html-modal-open-external:hover,
.html-modal-close:hover {
  background: rgba(56, 189, 248, 0.15);
  color: #38bdf8;
}

.html-modal-frame {
  flex: 1 1 auto;
  width: 100%;
  border: none;
  background: #f8fafc;
}

/* 窄屏：弹窗几乎占满全屏 */
@media (max-width: 720px) {
  .html-modal-overlay {
    padding: 0;
  }
  .html-modal-dialog {
    max-width: none;
    max-height: none;
    height: 100%;
    border-radius: 0;
  }
}
</style>
