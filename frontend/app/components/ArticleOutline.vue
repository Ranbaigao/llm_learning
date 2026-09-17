<script setup lang="ts">
const props = defineProps<{
  content: HTMLElement | null
  html: string
}>()

interface OutlineEntry {
  id: string
  title: string
  level: number
  element: HTMLElement
}

const entries = shallowRef<OutlineEntry[]>([])
const activeId = ref('')
const drawerOpen = ref(false)
const navRef = ref<HTMLElement | null>(null)
const toggleRef = ref<HTMLButtonElement | null>(null)
const outlineId = useId()
const minLevel = computed(() => Math.min(...entries.value.map(entry => entry.level)))

let resizeObserver: ResizeObserver | null = null
let scrollFrame = 0
let hashFrame = 0

function updateActiveHeading() {
  scrollFrame = 0
  if (!entries.value.length) return
  const headerBottom = document.querySelector('.site-header')?.getBoundingClientRect().bottom ?? 60
  let current = entries.value[0]!
  for (const entry of entries.value) {
    if (entry.element.getBoundingClientRect().top > headerBottom + 24) break
    current = entry
  }
  activeId.value = current.id
}

function scheduleActiveHeading() {
  if (!scrollFrame) scrollFrame = requestAnimationFrame(updateActiveHeading)
}

function rebuildOutline() {
  resizeObserver?.disconnect()
  cancelAnimationFrame(hashFrame)
  drawerOpen.value = false
  activeId.value = ''
  const content = props.content
  if (!content) {
    entries.value = []
    return
  }

  // 保留后端已有锚点；为 h3–h6、原生 HTML 和 notebook 中缺失/重复的锚点补齐 ID。
  const usedIds = new Set(Array.from(document.querySelectorAll('[id]'), el => el.id))
  const next: OutlineEntry[] = []
  content.querySelectorAll<HTMLElement>('h1, h2, h3, h4, h5, h6').forEach((heading, index) => {
    const label = heading.cloneNode(true) as HTMLElement
    label.querySelectorAll('.anchor-link, .headerlink, .header-anchor, mjx-assistive-mml')
      .forEach(el => el.remove())
    const title = (label.textContent || '').replace(/\s+/g, ' ').trim()
    if (!title) return

    if (!heading.id || document.getElementById(heading.id) !== heading) {
      const base = `article-section-${index + 1}`
      let id = base
      let suffix = 2
      while (usedIds.has(id)) id = `${base}-${suffix++}`
      heading.id = id
      usedIds.add(id)
    }
    next.push({ id: heading.id, title, level: Number(heading.tagName[1]), element: heading })
  })
  entries.value = next
  resizeObserver?.observe(content)

  // 目录渲染完成后恢复深链接；初次加载时浏览器可能还找不到补生成的锚点。
  hashFrame = requestAnimationFrame(() => {
    hashFrame = 0
    let hash = ''
    try { hash = decodeURIComponent(window.location.hash.slice(1)) } catch { /* 忽略无效转义 */ }
    const target = next.find(entry => entry.id === hash)
    target?.element.scrollIntoView({ block: 'start' })
    scheduleActiveHeading()
  })
}

function closeDrawer() {
  drawerOpen.value = false
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape' && drawerOpen.value) {
    closeDrawer()
    toggleRef.value?.focus()
  }
}

// 只滚动目录自身，避免 scrollIntoView 同时带动正文。
watch(activeId, async () => {
  await nextTick()
  const nav = navRef.value
  const link = nav?.querySelector<HTMLElement>('[aria-current="location"]')
  if (!nav || !link) return
  const navRect = nav.getBoundingClientRect()
  const linkRect = link.getBoundingClientRect()
  if (linkRect.top < navRect.top) nav.scrollTop += linkRect.top - navRect.top - 8
  else if (linkRect.bottom > navRect.bottom) nav.scrollTop += linkRect.bottom - navRect.bottom + 8
})

watch([() => props.content, () => props.html], rebuildOutline, { flush: 'post' })

onMounted(() => {
  // 图片加载、公式排版会改变标题位置，也需要刷新当前章节。
  resizeObserver = new ResizeObserver(scheduleActiveHeading)
  rebuildOutline()
  window.addEventListener('scroll', scheduleActiveHeading, { passive: true })
  window.addEventListener('resize', scheduleActiveHeading)
  window.addEventListener('hashchange', scheduleActiveHeading)
  document.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  cancelAnimationFrame(scrollFrame)
  cancelAnimationFrame(hashFrame)
  window.removeEventListener('scroll', scheduleActiveHeading)
  window.removeEventListener('resize', scheduleActiveHeading)
  window.removeEventListener('hashchange', scheduleActiveHeading)
  document.removeEventListener('keydown', onKeydown)
})
</script>

<template>
  <template v-if="entries.length">
    <button
      ref="toggleRef"
      class="outline-toggle"
      type="button"
      aria-label="打开笔记大纲"
      :aria-expanded="drawerOpen"
      :aria-controls="outlineId"
      @click="drawerOpen = !drawerOpen"
    >☷ 大纲</button>
    <button
      v-if="drawerOpen"
      class="outline-mask"
      type="button"
      aria-label="关闭笔记大纲"
      @click="closeDrawer"
    ></button>
    <aside :id="outlineId" class="article-outline" :class="{ 'is-open': drawerOpen }">
      <div class="outline-head">
        <span>笔记大纲</span>
        <button class="outline-close" type="button" aria-label="关闭大纲" @click="closeDrawer">×</button>
      </div>
      <nav ref="navRef" class="outline-body" aria-label="笔记大纲">
        <ol>
          <li v-for="entry in entries" :key="entry.id">
            <a
              :href="`#${encodeURIComponent(entry.id)}`"
              :style="{ paddingLeft: `${10 + Math.min(entry.level - minLevel, 3) * 12}px` }"
              :aria-current="activeId === entry.id ? 'location' : undefined"
              @click="closeDrawer"
            >{{ entry.title }}</a>
          </li>
        </ol>
      </nav>
    </aside>
  </template>
</template>

<style scoped>
.article-outline {
  position: sticky;
  top: calc(var(--header-height) + 20px);
  flex: 0 0 224px;
  width: 224px;
  max-height: calc(100dvh - var(--header-height) - 40px);
  display: flex;
  flex-direction: column;
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-card);
  padding: 14px;
}
.outline-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  padding-bottom: 10px;
  margin-bottom: 8px;
  border-bottom: 1px solid rgba(56, 189, 248, 0.1);
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 600;
}
.outline-body {
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
}
.outline-body ol {
  list-style: none;
  margin: 0;
  padding: 0;
}
.outline-body a {
  display: block;
  padding: 6px 8px;
  border-left: 2px solid transparent;
  border-radius: 0 6px 6px 0;
  color: var(--text-secondary);
  font-size: 13px;
  line-height: 1.6;
  overflow-wrap: anywhere;
}
.outline-body a:hover,
.outline-body a:focus-visible {
  color: var(--accent);
  background: rgba(56, 189, 248, 0.06);
}
.outline-body a[aria-current="location"] {
  color: var(--accent);
  border-left-color: var(--accent);
  background: rgba(56, 189, 248, 0.1);
}
.outline-toggle,
.outline-close,
.outline-mask {
  display: none;
}

@media (max-width: 1200px) {
  .article-outline {
    position: fixed;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(320px, 85vw);
    max-height: 100dvh;
    z-index: 230;
    border-radius: 0;
    background: #0b1220;
    visibility: hidden;
    transform: translateX(105%);
    transition: transform 0.25s ease, visibility 0.25s;
  }
  .article-outline.is-open {
    visibility: visible;
    transform: translateX(0);
    box-shadow: 0 0 40px rgba(0, 0, 0, 0.6);
  }
  .outline-toggle {
    display: block;
    position: fixed;
    right: 12px;
    bottom: 20px;
    z-index: 200;
    padding: 10px 16px;
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 24px;
    background: rgba(11, 18, 32, 0.95);
    color: var(--accent);
    font-size: 14px;
    cursor: pointer;
    box-shadow: var(--glow-accent);
  }
  .outline-close {
    display: block;
    border: 0;
    background: none;
    color: var(--text-secondary);
    font-size: 22px;
    cursor: pointer;
  }
  .outline-mask {
    display: block;
    position: fixed;
    inset: 0;
    z-index: 225;
    border: 0;
    background: rgba(0, 0, 0, 0.55);
    touch-action: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .article-outline { transition: none; }
}
</style>
