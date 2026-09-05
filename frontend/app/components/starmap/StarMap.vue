<script setup lang="ts">
/**
 * 知识星图首页主组件：画布 + 全部 DOM 面板 + 数据获取（GET /api/graph）。
 * 渲染/物理/输入交互全在 engine.ts 的 StarMapEngine 中；本组件负责
 * 面板状态（目录树/榜单/搜索/工具栏）与左侧栏动态布局编排。
 */
import StarMapQuote from './StarMapQuote.vue'
import StarMapSidebar from './StarMapSidebar.vue'
import StarMapPanels from './StarMapPanels.vue'
import {
  StarMapEngine,
  buildSidebarTree,
  latestNotes,
  hotNotes,
  totalPageviews,
  TYPE_LABEL,
  clamp,
} from './engine'
import type {
  EngineNode,
  GraphDataInput,
  GraphStats,
  SidebarTreeNode,
  WidgetNote,
} from './engine'

const rootEl = ref<HTMLElement | null>(null)
const canvasEl = ref<HTMLCanvasElement | null>(null)
const labelLayerEl = ref<HTMLElement | null>(null)
const tooltipEl = ref<HTMLElement | null>(null)
const searchResultsEl = ref<HTMLElement | null>(null)

let engine: StarMapEngine | null = null

const loading = ref(true)
const errorMsg = ref('')

const tree = ref<SidebarTreeNode[]>([])
const noteCount = ref(0)
const stats = ref<GraphStats>({})
const sitePv = ref<number | null>(null)
const siteUv = ref<number | null>(null)
const latest = ref<WidgetNote[]>([])
const hot = ref<WidgetNote[]>([])

// 星图固定（pin）的节点 id：侧栏据此高亮（等价旧版 syncSidebarActive）
const activeNodeId = ref<string | null>(null)

// 侧栏与信息栏的动态位置/高度（layoutSideWidgets 计算下发）
const sidebarCollapsed = ref(false)
const sidebarTop = ref('')
const sidebarMaxHeight = ref('')
const latestTop = ref('')
const latestMaxHeight = ref('')
const hotTop = ref('')
const hotMaxHeight = ref('')

const paused = ref(false)

// 搜索状态：query 双向绑定输入框；引擎内部另存一份匹配集（用于节点高亮/标签）
const searchInput = ref('')
const searchResults = ref<EngineNode[]>([])
const searchResultsOpen = ref(false)
const activeSearchIndex = ref(-1)

// 重名结果计数（渲染「重名 xN」徽章）
const searchNameCounts = computed(() => {
  const counts = new Map<string, number>()
  for (const node of searchResults.value) {
    const key = String(node.name || '').trim().toLowerCase()
    counts.set(key, (counts.get(key) || 0) + 1)
  }
  return counts
})

const visibleSearchResults = computed(() => searchResults.value.slice(0, 18))

function searchTypeLabel(node: EngineNode): string {
  return node.format === 'jupyter'
    ? 'Jupyter'
    : (TYPE_LABEL[node.type] || node.type || '').toUpperCase()
}

// ======= 数据加载 =======

async function loadGraph() {
  loading.value = true
  errorMsg.value = ''
  try {
    const data = await $fetch<GraphDataInput>('/api/graph', {
      query: { t: Date.now() },
    })
    tree.value = buildSidebarTree(data)
    noteCount.value = data.nodes.filter(node => node.type === 'note').length
    stats.value = data.stats || {}
    // 仅作兜底：/api/stats/site 返回的权威 PV 优先（两处异步完成顺序不定，防止互相覆盖）
    if (sitePv.value == null) sitePv.value = totalPageviews(data)
    latest.value = latestNotes(data)
    hot.value = hotNotes(data)
    await nextTick()
    engine?.setData(data)
    await nextTick()
    layoutSideWidgets()
  } catch (error) {
    errorMsg.value = `知识星图数据加载失败：${(error as Error).message}。请确认后端服务已启动（FastAPI 8000 端口）。`
  } finally {
    loading.value = false
  }
}

// 站点级 PV/UV（后端 /api/stats/site，比图谱节点 pv 求和更权威）；失败则保留图谱求和的兜底值
async function loadSiteStats() {
  try {
    const data = await $fetch<{ site_pv: number; site_uv: number }>('/api/stats/site')
    sitePv.value = data.site_pv
    siteUv.value = data.site_uv
  } catch {
    // 静默失败：统计卡片保持图谱求和 PV 与 '-' UV
  }
}

// ======= 引擎回调 =======

function openNode(node: { type: string; url?: string }) {
  // 只有笔记节点可跳转；目录节点在星图/侧边栏中定位
  if (node.type !== 'note' || !node.url) return
  navigateTo(node.url)
}

// ======= 搜索 =======

function clearSearchResults() {
  searchResults.value = []
  activeSearchIndex.value = -1
  searchResultsOpen.value = false
  layoutSidebarForSearch()
}

function setActiveSearchResult(index: number, syncFocus = true) {
  if (!searchResults.value.length) return
  const nextIndex = clamp(index, 0, searchResults.value.length - 1)
  activeSearchIndex.value = nextIndex
  if (syncFocus) engine?.focusSearchNode(searchResults.value[nextIndex])
}

function selectSearchResult(index: number, openAfterSelect = false) {
  if (!searchResults.value.length) return
  setActiveSearchResult(index, true)
  if (openAfterSelect) openNode(searchResults.value[activeSearchIndex.value])
}

async function onSearchInput(event: Event) {
  // 不用 v-model：原生 input 上 v-model 的赋值监听器注册晚于用户 @input，
  // 同事件内读 ref 会拿到旧值；这里直接读事件目标并写回 ref
  const value = (event.target as HTMLInputElement).value
  searchInput.value = value
  const ranked = engine?.setSearchQuery(value) ?? []
  if (!value.trim()) {
    clearSearchResults()
    return
  }
  if (ranked.length) {
    searchResults.value = ranked
    activeSearchIndex.value = 0 // setActiveSearchResult(0, false)：不重复移动焦点
    searchResultsOpen.value = true
    // 等候选列表完成布局后测量其实际高度，驱动侧边栏下移
    await nextTick()
    layoutSidebarForSearch()
  } else {
    clearSearchResults()
  }
}

function onSearchKeydown(event: KeyboardEvent) {
  // 阻止冒泡：按键只服务于本搜索框，不被页面级快捷键抢走
  event.stopPropagation()
  if (!searchResultsOpen.value || !searchResults.value.length) return
  if (event.key === 'ArrowDown') {
    event.preventDefault()
    setActiveSearchResult(activeSearchIndex.value + 1, true)
    return
  }
  if (event.key === 'ArrowUp') {
    event.preventDefault()
    setActiveSearchResult(activeSearchIndex.value - 1, true)
    return
  }
  if (event.key === 'Enter') {
    event.preventDefault()
    // 回车直接跳转当前候选
    selectSearchResult(activeSearchIndex.value >= 0 ? activeSearchIndex.value : 0, true)
    return
  }
  if (event.key === 'Escape') {
    event.preventDefault()
    clearSearchResults()
  }
}

// 点击搜索框与候选列表之外的区域时收起候选
function onDocumentPointerdown(event: PointerEvent) {
  const target = event.target
  if (!(target instanceof Element)) return
  if (target.closest('.search-panel') || target.closest('.search-results')) return
  clearSearchResults()
}

// 引擎内部清空搜索状态（点空白 / 侧栏选中）时同步清空输入框
function clearSearchUI() {
  searchInput.value = ''
  clearSearchResults()
}

// ======= 左侧列动态布局（移植旧版 layoutSideWidgets / layoutSidebarForSearch）=======

function widgetNaturalHeight(panel: HTMLElement): number {
  if (panel.classList.contains('is-collapsed')) return 44
  const list = panel.querySelector('.widget-list')
  return Math.min(260, Math.round(list?.scrollHeight || 0) + 46) // 46 = 44px 标题栏 + 上下边框
}

// 左侧列整体布局：笔记目录 → 最新更新 → 浏览热度 依次堆叠，
// 下边界是底部工具栏（移动端为底部统计卡片）。信息栏按内容取自然高度，
// 侧边栏让位：max-height 收紧到剩余空间，目录展开过多时树形列表内部滚动，
// 保证目录无论展开多大，信息栏都不会遮挡底部工具栏。
async function layoutSideWidgets() {
  await nextTick()
  const root = rootEl.value
  if (!root) return
  const sidebar = root.querySelector<HTMLElement>('.sidebar')
  const latestPanel = root.querySelector<HTMLElement>('.side-widget--latest')
  const hotPanel = root.querySelector<HTMLElement>('.side-widget--hot')
  const toolbar = root.querySelector<HTMLElement>('.toolbar')
  const headerPanel = root.querySelector<HTMLElement>('.header-panel')
  if (!sidebar || !latestPanel || !hotPanel || !toolbar || !headerPanel) return

  const gap = 10
  const isMobile = window.innerWidth <= 760
  const rootRect = root.getBoundingClientRect()
  // 左列下边界：桌面端是左下工具栏，移动端是底部通栏统计卡片（容器相对坐标）
  const bottomLimit = (isMobile ? headerPanel : toolbar).getBoundingClientRect().top - rootRect.top - gap

  const sbRect = sidebar.getBoundingClientRect()
  const sbTop = sbRect.top - rootRect.top
  // 桌面端为两个信息栏预留空间（收起按 44px 标题栏计）
  const reserved = isMobile ? 0 : [latestPanel, hotPanel]
    .reduce((sum, panel) => sum + widgetNaturalHeight(panel) + gap, 0)
  const sidebarMax = Math.max(140, bottomLimit - sbTop - reserved)
  sidebarMaxHeight.value = `${Math.round(sidebarMax)}px`

  if (isMobile) return // 移动端信息栏隐藏，只需收紧侧边栏

  const sbVisibleHeight = sidebarCollapsed.value ? 44 : Math.min(sbRect.height, sidebarMax)
  let top = Math.round(sbTop + sbVisibleHeight + gap)
  for (const panel of [
    { el: latestPanel, key: 'latest' },
    { el: hotPanel, key: 'hot' },
  ] as const) {
    const natural = widgetNaturalHeight(panel.el)
    // 窗口过矮时信息栏让位：压缩 max-height，列表内部滚动
    const visible = Math.max(64, Math.min(natural, bottomLimit - top))
    if (panel.key === 'latest') {
      latestTop.value = `${top}px`
      latestMaxHeight.value = `${Math.round(visible)}px`
    } else {
      hotTop.value = `${top}px`
      hotMaxHeight.value = `${Math.round(visible)}px`
    }
    top += visible + gap
  }
}

// 搜索候选列表弹出/收起时，动态调整笔记目录的顶边位置：
// 候选列表展开 → 侧边栏平滑下移到列表下方，避免被遮挡；收起 → 回到原位
async function layoutSidebarForSearch() {
  await nextTick()
  const root = rootEl.value
  if (!root) return
  if (searchResultsOpen.value && searchResults.value.length && searchResultsEl.value) {
    const resultsBottom = searchResultsEl.value.getBoundingClientRect().bottom
    const rootTop = root.getBoundingClientRect().top
    sidebarTop.value = `${Math.round(resultsBottom - rootTop + 10)}px`
  } else {
    sidebarTop.value = ''
  }
  // 侧边栏的 max-height 收紧与下方信息栏对齐统一交给 layoutSideWidgets
  layoutSideWidgets()
}

// ======= 工具栏动作 =======

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
  layoutSideWidgets()
}

function togglePause() {
  paused.value = !paused.value
  engine?.setPaused(paused.value)
}

function onWindowResize() {
  engine?.resize()
  layoutSidebarForSearch()
  layoutSideWidgets()
}

// ======= 生命周期 =======

onMounted(() => {
  engine = new StarMapEngine({
    canvas: canvasEl.value!,
    labelLayer: labelLayerEl.value!,
    tooltip: tooltipEl.value!,
    badgeUrl: '/api/assets/root_badge.png',
    callbacks: {
      onOpenNote: openNode,
      onPinChange: id => { activeNodeId.value = id },
      onClearSearch: clearSearchUI,
    },
  })
  sidebarCollapsed.value = window.innerWidth < 760 // 小屏默认收起侧边栏
  loadGraph()
  loadSiteStats()
  window.addEventListener('resize', onWindowResize)
  document.addEventListener('pointerdown', onDocumentPointerdown)

  // dev 环境暴露调试句柄（验证脚本用）
  if (import.meta.dev) {
    (window as any).__starmap = {
      get engine() { return engine },
      reload: loadGraph,
    }
  }
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onWindowResize)
  document.removeEventListener('pointerdown', onDocumentPointerdown)
  engine?.dispose()
  engine = null
  if (import.meta.dev && (window as any).__starmap) {
    delete (window as any).__starmap
  }
})
</script>

<template>
  <div ref="rootEl" class="starmap">
    <canvas ref="canvasEl" class="scene" aria-label="notebooks 3D 知识星图"></canvas>
    <div ref="labelLayerEl" class="labels" aria-hidden="true"></div>

    <StarMapQuote />

    <label class="panel search-panel" for="starmap-search-input" aria-label="搜索节点">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="7"/><path d="m20 20-4.2-4.2"/></svg>
      <input
        id="starmap-search-input"
        :value="searchInput"
        type="search"
        placeholder="搜索笔记或目录"
        autocomplete="off"
        :aria-expanded="String(searchResultsOpen)"
        @input="onSearchInput"
        @keydown="onSearchKeydown"
      />
    </label>
    <section
      v-show="searchResultsOpen"
      ref="searchResultsEl"
      class="panel search-results"
      aria-label="搜索结果"
    >
      <button
        v-for="(node, index) in visibleSearchResults"
        :key="node.id"
        type="button"
        class="search-result-item"
        :class="{ 'is-active': index === activeSearchIndex }"
        :aria-label="`${node.name}，${node.id}`"
        @mouseenter="setActiveSearchResult(index, true)"
        @click="selectSearchResult(index, true)"
        @dblclick="selectSearchResult(index, true)"
      >
        <div class="search-result-head">
          <div class="search-result-name">{{ node.name }}</div>
          <span
            v-if="(searchNameCounts.get(String(node.name || '').trim().toLowerCase()) || 0) > 1"
            class="search-result-badge"
          >重名 x{{ searchNameCounts.get(String(node.name || '').trim().toLowerCase()) }}</span>
        </div>
        <div class="search-result-meta">/{{ node.id }}</div>
        <div class="search-result-type">{{ searchTypeLabel(node) }}</div>
      </button>
    </section>

    <StarMapSidebar
      :tree="tree"
      :note-count="noteCount"
      :active-id="activeNodeId"
      v-model:collapsed="sidebarCollapsed"
      :top="sidebarTop"
      :max-height="sidebarMaxHeight"
      @select="node => engine?.pinNodeById(node.id)"
      @open="openNode"
      @layout-change="layoutSideWidgets"
      @transitionend="layoutSideWidgets"
    />

    <StarMapPanels
      :stats="stats"
      :site-pv="sitePv"
      :site-uv="siteUv"
      :latest="latest"
      :hot="hot"
      :sidebar-collapsed="sidebarCollapsed"
      :paused="paused"
      :latest-top="latestTop"
      :latest-max-height="latestMaxHeight"
      :hot-top="hotTop"
      :hot-max-height="hotMaxHeight"
      @toggle-sidebar="toggleSidebar"
      @fit="engine?.resetView()"
      @relayout="engine?.relayout()"
      @toggle-pause="togglePause"
      @refresh="loadGraph"
      @select="node => engine?.pinNodeById(node.id, { syncSidebar: false })"
      @open="openNode"
      @layout-change="layoutSideWidgets"
    />

    <div ref="tooltipEl" class="starmap-tooltip" role="status" aria-live="polite"></div>
    <div v-if="loading" class="starmap-loading"><span class="spinner"></span><span>加载知识星图</span></div>
    <div v-if="errorMsg" class="starmap-error">{{ errorMsg }}</div>
  </div>
</template>

<style>
/* ===== 非 scoped：@font-face、引擎创建的 DOM（标签/提示框）与各子组件共享的 .panel ===== */

/* 字体由后端静态服务托管（content/.assets/fonts/）：
   - Share Tech：一级目录题名（官方 latin 切片，~15KB）
   - Long Cang：顶部名言行楷（按名言字符集子集化，~118KB；
     若修改 SIGNATURE_QUOTES 引入新字，需重新子集化：见 .cache/subset_fonts.py） */
@font-face {
  font-family: 'Share Tech';
  src: url('/api/assets/fonts/share-tech-latin.woff2') format('woff2');
  font-weight: normal;
  font-style: normal;
  font-display: swap;
}
@font-face {
  font-family: 'Long Cang';
  src: url('/api/assets/fonts/long-cang-quote.woff2') format('woff2');
  font-weight: normal;
  font-style: normal;
  font-display: swap;
}

.starmap .panel {
  position: absolute;
  z-index: 4;
  background: var(--panel-bg);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  box-shadow: var(--panel-shadow);
  backdrop-filter: blur(10px) saturate(130%);
  -webkit-backdrop-filter: blur(10px) saturate(130%);
}

.starmap .graph-label {
  position: absolute;
  left: 0;
  top: 0;
  max-width: 220px;
  color: var(--label-color, #cbd5e1);
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, system-ui, sans-serif;
  line-height: 1.16;
  letter-spacing: 0.05em;
  text-align: center;
  white-space: nowrap;
  text-shadow:
    0 1px 0 rgba(2, 6, 23, 0.92),
    0 -1px 0 rgba(2, 6, 23, 0.86),
    1px 0 0 rgba(2, 6, 23, 0.86),
    -1px 0 0 rgba(2, 6, 23, 0.86),
    0 0 12px rgba(2, 6, 23, 0.60);
  transform: translate3d(-999px, -999px, 0) translateX(-50%);
  transform-origin: 50% 0;
  will-change: transform, opacity;
  transition: opacity 240ms ease;
}

.starmap .graph-label--note { font-weight: 500; }

/* 节点提示浮层：旧版 position:fixed + 画布即视口；现改为 absolute，
   定位坐标与画布坐标系一致（引擎按画布尺寸做避让计算） */
.starmap .starmap-tooltip {
  position: absolute;
  z-index: 8;
  min-width: 190px;
  max-width: min(330px, calc(100vw - 28px));
  padding: 10px 12px 11px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 8px;
  background: rgba(10, 16, 32, 0.85);
  backdrop-filter: blur(12px) saturate(140%);
  -webkit-backdrop-filter: blur(12px) saturate(140%);
  box-shadow: 0 20px 45px rgba(0, 0, 0, 0.55);
  color: var(--text-color);
  pointer-events: none;
  opacity: 0;
  transition: opacity 140ms ease;
}

.starmap .starmap-tooltip.visible { opacity: 1; }

.starmap .tip-type {
  display: inline-flex;
  align-items: center;
  height: 20px;
  padding: 0 7px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.starmap .tip-title {
  margin-top: 7px;
  color: var(--title-color);
  font-size: 14px;
  font-weight: 750;
  line-height: 1.35;
  overflow-wrap: anywhere;
}

.starmap .tip-path {
  margin-top: 5px;
  color: var(--muted);
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 10.5px;
  line-height: 1.45;
  overflow-wrap: anywhere;
}

.starmap .tip-hint {
  margin-top: 6px;
  color: var(--muted);
  font-size: 10.5px;
}
</style>

<style scoped>
.starmap {
  /* 星图面板的私有设计变量（旧版定义在 :root） */
  --bg-color: #030712;
  --text-color: #cbd5e1;
  --title-color: #f1f5f9;
  --panel-bg: rgba(10, 16, 32, 0.58);
  --border-color: rgba(148, 163, 184, 0.16);
  --panel-shadow: 0 8px 28px rgba(0, 0, 0, 0.45);
  --accent-primary: #60a5fa;
  --accent-text: #93c5fd;
  --accent-light: rgba(59, 130, 246, 0.20);
  --success: #4ade80;
  --warning: #fbbf24;
  --muted: #7d8ca3;
  --root-color: #f1f5f9;
  --llm-color: #60a5fa;
  --cv-color: #2dd4bf;
  --performance-color: #fbbf24;
  --app-color: #a78bfa;
  --practice-color: #22d3ee;
  --reference-color: #a78bfa;

  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background:
    radial-gradient(circle at 24% 16%, rgba(28, 50, 105, 0.14) 0, transparent 46%),
    radial-gradient(circle at 80% 22%, rgba(48, 24, 88, 0.10) 0, transparent 48%),
    radial-gradient(circle at 56% 88%, rgba(7, 30, 52, 0.16) 0, transparent 55%),
    radial-gradient(circle at 8% 78%, rgba(40, 20, 66, 0.10) 0, transparent 42%),
    linear-gradient(160deg, #01020a 0%, #020410 48%, #03061a 100%);
  color: var(--text-color);
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}

.starmap::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    repeating-linear-gradient(0deg, rgba(148, 163, 184, 0.028) 0 1px, transparent 1px 40px),
    repeating-linear-gradient(90deg, rgba(148, 163, 184, 0.024) 0 1px, transparent 1px 40px);
  mask-image: linear-gradient(180deg, transparent 0%, black 16%, black 84%, transparent 100%);
  -webkit-mask-image: linear-gradient(180deg, transparent 0%, black 16%, black 84%, transparent 100%);
}

.scene {
  position: absolute;
  inset: 0;
  z-index: 1;
  width: 100%;
  height: 100%;
  display: block;
  cursor: grab;
  touch-action: none;
}

.scene.is-dragging { cursor: grabbing; }

.labels {
  position: absolute;
  inset: 0;
  z-index: 3;
  overflow: hidden;
  pointer-events: none;
  contain: layout style paint;
}

/* 搜索面板 - 左上方 */
.search-panel {
  top: 18px;
  left: 18px;
  width: min(310px, calc(100vw - 112px));
  height: 48px;
  padding: 0 13px;
  display: flex;
  align-items: center;
  gap: 9px;
}

.search-panel svg {
  width: 17px;
  height: 17px;
  color: var(--muted);
  flex: 0 0 auto;
}

.search-panel input {
  width: 100%;
  min-width: 0;
  border: 0;
  outline: 0;
  background: transparent;
  color: var(--title-color);
  font: inherit;
  font-size: 13px;
}

.search-panel input::placeholder { color: #5b6b84; }

/* 搜索结果弹窗优先级提高(z-index: 10)，位置匹配左上方 */
.search-results {
  position: absolute;
  z-index: 10;
  top: 72px;
  left: 18px;
  width: min(340px, calc(100vw - 34px));
  max-height: min(380px, calc(100vh - 190px));
  overflow: auto;
  padding: 8px;
  display: grid;
  gap: 6px;
}

.search-result-item {
  width: 100%;
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 10px;
  background: rgba(148, 163, 184, 0.05);
  color: var(--title-color);
  padding: 8px 10px;
  text-align: left;
  cursor: pointer;
  transition: border-color 150ms ease, background 150ms ease, transform 150ms ease;
}

.search-result-item:hover,
.search-result-item.is-active {
  border-color: rgba(96, 165, 250, 0.45);
  background: rgba(59, 130, 246, 0.16);
  transform: translateY(-1px);
}

.search-result-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.search-result-name {
  font-size: 12.5px;
  font-weight: 700;
  line-height: 1.25;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.search-result-badge {
  flex: 0 0 auto;
  height: 18px;
  border-radius: 999px;
  padding: 0 7px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  font-weight: 700;
  color: #bfdbfe;
  background: rgba(59, 130, 246, 0.25);
}

.search-result-meta {
  margin-top: 4px;
  color: var(--muted);
  font-size: 10.5px;
  line-height: 1.4;
  overflow-wrap: anywhere;
}

.search-result-type {
  margin-top: 5px;
  font-size: 10px;
  color: #5eead4;
  font-weight: 700;
  letter-spacing: 0.02em;
  text-transform: uppercase;
}

.starmap-loading,
.starmap-error {
  position: absolute;
  z-index: 6;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
}

.starmap-loading {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--muted);
  font-size: 13px;
  background: rgba(10, 16, 32, 0.75);
  border: 1px solid rgba(148, 163, 184, 0.2);
  border-radius: 8px;
  padding: 12px 15px;
  box-shadow: var(--panel-shadow);
}

.spinner {
  width: 22px;
  height: 22px;
  border: 2px solid rgba(148, 163, 184, 0.25);
  border-top-color: var(--accent-primary);
  border-radius: 999px;
  animation: spin 760ms linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.starmap-error {
  width: min(460px, calc(100vw - 32px));
  padding: 16px 18px;
  border: 1px solid rgba(248, 113, 113, 0.3);
  border-radius: 8px;
  background: rgba(69, 10, 10, 0.5);
  color: #fca5a5;
  box-shadow: var(--panel-shadow);
  font-size: 13px;
  line-height: 1.55;
}

@media (max-width: 760px) {
  .search-panel {
    top: 10px;
    left: 10px;
    width: min(240px, calc(100vw - 66px));
  }

  .search-results {
    top: 64px;
    left: 10px;
    width: min(280px, calc(100vw - 20px));
    max-height: min(330px, calc(100vh - 170px));
  }
}
</style>
