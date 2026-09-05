<script setup lang="ts">
const props = defineProps<{
  article: ArticleDetail
}>()

const contentRef = ref<HTMLElement | null>(null)
const router = useRouter()

// ---- 文章头信息 ----
const displayDate = computed(() => {
  const meta = props.article.meta || {}
  const raw = meta.date || props.article.content_updated_at
  if (!raw) return ''
  return String(raw).slice(0, 10)
})

const metaCategories = computed<string[]>(() => {
  const cats = props.article.meta?.categories
  if (Array.isArray(cats)) return cats.map(String)
  if (typeof cats === 'string') return [cats]
  return []
})

// 预计阅读时长：剥离 HTML 标签后按中文约 400 字/分钟估算
const readingMinutes = computed(() => {
  const text = props.article.html.replace(/<[^>]+>/g, '')
  return Math.max(1, Math.round(text.length / 400))
})

// ---- MathJax ----
// 后端 dollarmath 输出的 span.math.inline / div.math.block 内是「无定界符」的纯 TeX，
// MathJax 按定界符扫描，因此 typeset 前先包上 \( \) / \[ \]
function wrapMathDelimiters(el: HTMLElement) {
  el.querySelectorAll('span.math.inline').forEach((node) => {
    const t = node.textContent || ''
    if (t.trim() && !t.trimStart().startsWith('\\(')) {
      node.textContent = `\\(${t}\\)`
    }
  })
  el.querySelectorAll('div.math.block').forEach((node) => {
    const t = node.textContent || ''
    if (t.trim() && !t.trimStart().startsWith('\\[')) {
      node.textContent = `\\[${t}\\]`
    }
  })
}

function mathjaxReady(): Promise<any> {
  return new Promise((resolve) => {
    const check = () => {
      const mj = (window as any).MathJax
      if (mj?.typesetPromise) return resolve(mj)
      if (mj?.startup?.promise) {
        mj.startup.promise.then(() => resolve((window as any).MathJax))
        return
      }
      setTimeout(check, 100)
    }
    check()
  })
}

let typesetSeq = 0
async function typesetMath() {
  if (!import.meta.client) return
  const seq = ++typesetSeq
  await nextTick()
  const el = contentRef.value
  if (!el || !el.querySelector('.math')) return
  const mj = await mathjaxReady()
  if (seq !== typesetSeq || !contentRef.value) return // 期间文章已切换
  try {
    // 切换文章后旧容器已销毁，清掉 MathJax 内部的陈旧状态
    mj.startup?.document?.clear?.()
    wrapMathDelimiters(contentRef.value)
    await mj.typesetPromise([contentRef.value])
  } catch (e) {
    console.warn('MathJax typeset 失败', e)
  }
}

// ---- v-html 内链接行为 ----
// a.html-modal-link 走 HtmlModal 弹窗；站内 /notes/ 链接走前端路由；其余 /api/assets/*.html 新标签页
const modalOpen = ref(false)
const modalHref = ref('')
const modalTitle = ref('')

function openHtmlModal(href: string, title: string) {
  modalHref.value = href
  modalTitle.value = title
  modalOpen.value = true
}

function onContentClick(e: MouseEvent) {
  const anchor = (e.target as HTMLElement).closest('a')
  if (!anchor) return
  if (anchor.classList.contains('html-modal-link')) {
    e.preventDefault()
    openHtmlModal(
      anchor.getAttribute('href') || '',
      anchor.getAttribute('data-title') || anchor.textContent?.trim() || '',
    )
    return
  }
  const href = anchor.getAttribute('href') || ''
  if (href.startsWith('/api/assets/') && href.split('?')[0]?.endsWith('.html')) {
    e.preventDefault()
    window.open(href, '_blank', 'noopener')
    return
  }
  if (href.startsWith('/notes/') || href.startsWith('/blog/')) {
    e.preventDefault()
    router.push(href)
  }
}

onMounted(() => {
  typesetMath()
})

watch(
  () => props.article.html,
  () => typesetMath(),
)
</script>

<template>
  <article class="article-view">
    <header class="article-head card">
      <h1 class="article-title">{{ article.title }}</h1>
      <div class="article-meta">
        <span class="meta-chip">{{ article.category }}</span>
        <span v-for="c in metaCategories" :key="c" class="meta-chip tag">{{ c }}</span>
        <span v-if="displayDate" class="meta-item">📅 {{ displayDate }}</span>
        <span class="meta-item">👁 {{ article.views }} 阅读</span>
        <span class="meta-item">♥ {{ article.like_count }} 点赞</span>
        <span class="meta-item">⏱ 约 {{ readingMinutes }} 分钟</span>
        <span v-if="article.format === 'jupyter'" class="meta-chip tag">Jupyter</span>
      </div>
    </header>

    <!-- 后端渲染的可信 HTML（本站内容） -->
    <!-- eslint-disable-next-line vue/no-v-html -->
    <div
      ref="contentRef"
      class="article-content card"
      @click="onContentClick"
      v-html="article.html"
    ></div>

    <HtmlModal
      :open="modalOpen"
      :href="modalHref"
      :title="modalTitle"
      @close="modalOpen = false"
    />
  </article>
</template>

<style scoped>
.article-view {
  min-width: 0;
}
.article-head {
  padding: 28px 32px 20px;
  margin-bottom: 20px;
}
.article-title {
  margin: 0 0 14px;
  font-size: 28px;
  text-shadow: 0 0 20px rgba(56, 189, 248, 0.25);
}
.article-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 16px;
  font-size: 13px;
  color: var(--text-secondary);
}
.meta-chip {
  padding: 2px 10px;
  border-radius: 10px;
  background: rgba(56, 189, 248, 0.12);
  color: var(--accent);
  font-size: 12px;
}
.meta-chip.tag {
  background: rgba(37, 99, 235, 0.18);
}
.article-content {
  padding: 28px 32px;
}

@media (max-width: 768px) {
  .article-head {
    padding: 20px 18px 14px;
  }
  .article-title {
    font-size: 22px;
  }
  .article-content {
    padding: 18px;
  }
}
</style>
