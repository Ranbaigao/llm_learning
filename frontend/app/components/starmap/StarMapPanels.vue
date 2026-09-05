<script setup lang="ts">
/**
 * 星图信息面板群：右下统计卡片、右上图例、左下工具栏、左侧「最新更新 / 浏览热度」榜单。
 * 榜单与目录同一套视觉与折叠动画；top/max-height 由父组件按左列堆叠布局动态下发。
 */
import { formatCount, formatRelativeTime } from './engine'
import type { GraphStats, WidgetNote } from './engine'

defineProps<{
  stats: GraphStats
  sitePv: number | null
  siteUv: number | null
  latest: WidgetNote[]
  hot: WidgetNote[]
  sidebarCollapsed: boolean
  paused: boolean
  latestTop: string
  latestMaxHeight: string
  hotTop: string
  hotMaxHeight: string
}>()

const emit = defineEmits<{
  toggleSidebar: []
  fit: []
  relayout: []
  togglePause: []
  refresh: []
  /** 榜单行单击：星图中固定定位（不联动笔记目录） */
  select: [node: WidgetNote]
  /** 榜单行双击：打开笔记 */
  open: [node: WidgetNote]
  /** 榜单折叠状态变化，父组件据此重排左列 */
  layoutChange: []
}>()

const latestCollapsed = ref(false)
const hotCollapsed = ref(false)

function toggleLatest() {
  latestCollapsed.value = !latestCollapsed.value
  emit('layoutChange')
}

function toggleHot() {
  hotCollapsed.value = !hotCollapsed.value
  emit('layoutChange')
}
</script>

<template>
  <section class="panel header-panel" aria-label="图谱统计">
    <p class="eyebrow">Knowledge Graph</p>
    <h1>我的 LLM 笔记星图</h1>
    <p class="site-view-stats">总浏览量 <strong>{{ sitePv == null ? '-' : formatCount(sitePv) }}</strong> 次 · 访客 <strong>{{ siteUv == null ? '-' : formatCount(siteUv) }}</strong> 人</p>
    <div class="metric-row">
      <div class="metric"><strong>{{ stats.nodes ?? 0 }}</strong><span>节点</span></div>
      <div class="metric"><strong>{{ stats.notes ?? 0 }}</strong><span>笔记</span></div>
      <div class="metric"><strong>{{ stats.contains ?? 0 }}</strong><span>结构</span></div>
      <div class="metric"><strong>{{ stats.references ?? 0 }}</strong><span>引用</span></div>
    </div>
  </section>

  <aside class="panel legend-panel" aria-label="图例">
    <div class="legend-title">Constellation</div>
    <div class="legend-item"><span class="legend-dot" style="background: var(--root-color); color: var(--root-color);"></span><span>根星</span></div>
    <div class="legend-item"><span class="legend-dot" style="background: var(--llm-color); color: var(--llm-color);"></span><span>NLP 星座</span></div>
    <div class="legend-item"><span class="legend-dot" style="background: var(--cv-color); color: var(--cv-color);"></span><span>CV 星座</span></div>
    <div class="legend-item"><span class="legend-dot" style="background: var(--performance-color); color: var(--performance-color);"></span><span>性能优化</span></div>
    <div class="legend-item"><span class="legend-dot" style="background: var(--practice-color); color: var(--practice-color);"></span><span>代码实践</span></div>
  </aside>

  <div class="panel toolbar" aria-label="图谱工具">
    <button
      class="icon-button"
      :class="{ 'is-active': !sidebarCollapsed }"
      type="button"
      title="展开/收起笔记目录"
      aria-label="展开或收起笔记目录"
      @click="emit('toggleSidebar')"
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h10"/></svg>
    </button>
    <button class="icon-button" type="button" title="重置视角" aria-label="重置视角" @click="emit('fit')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 3H5a2 2 0 0 0-2 2v3"/><path d="M16 3h3a2 2 0 0 1 2 2v3"/><path d="M8 21H5a2 2 0 0 1-2-2v-3"/><path d="M16 21h3a2 2 0 0 0 2-2v-3"/></svg>
    </button>
    <button class="icon-button" type="button" title="重新排布星图" aria-label="重新排布星图" @click="emit('relayout')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 12a9 9 0 1 1-2.64-6.36"/><path d="M21 3v6h-6"/></svg>
    </button>
    <button
      class="icon-button"
      type="button"
      :title="paused ? '继续动画' : '暂停动画'"
      :aria-label="paused ? '继续动画' : '暂停动画'"
      @click="emit('togglePause')"
    >
      <svg v-if="paused" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 5v14l11-7z"/></svg>
      <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10 4H6v16h4z"/><path d="M18 4h-4v16h4z"/></svg>
    </button>
    <button class="icon-button" type="button" title="重新读取数据" aria-label="重新读取数据" @click="emit('refresh')">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 11a8.1 8.1 0 0 0-15.5-2"/><path d="M4 4v5h5"/><path d="M4 13a8.1 8.1 0 0 0 15.5 2"/><path d="M20 20v-5h-5"/></svg>
    </button>
  </div>

  <aside
    class="panel side-widget side-widget--latest"
    :class="{ 'is-collapsed': latestCollapsed }"
    :style="{ top: latestTop, maxHeight: latestMaxHeight }"
    aria-label="最新更新的笔记"
  >
    <div
      class="sidebar-header widget-header"
      role="button"
      :aria-expanded="String(!latestCollapsed)"
      title="收起/展开最新更新"
      @click="toggleLatest"
    >
      <span class="sidebar-title">最新更新</span>
      <span class="sidebar-count">{{ latest.length ? `TOP ${latest.length}` : '暂无数据' }}</span>
      <button
        class="sidebar-collapse"
        type="button"
        :title="latestCollapsed ? '展开最新更新' : '收起最新更新'"
        :aria-label="latestCollapsed ? '展开最新更新' : '收起最新更新'"
      >
        <svg v-if="latestCollapsed" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m6 9 6 6 6-6"/></svg>
        <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m18 15-6-6-6 6"/></svg>
      </button>
    </div>
    <div class="sidebar-tree widget-list">
      <button
        v-for="node in latest"
        :key="node.id"
        type="button"
        class="tree-row is-note"
        title="单击在星图中定位 · 双击打开笔记"
        @click="emit('select', node)"
        @dblclick="emit('open', node)"
      >
        <span class="tree-dot" :style="{ background: node.color, color: node.color }"></span>
        <span class="tree-name">{{ node.name }}</span>
        <span class="widget-meta" title="最近更新时间">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg><span>{{ formatRelativeTime(node.mtime!) }}</span>
        </span>
      </button>
    </div>
  </aside>

  <aside
    class="panel side-widget side-widget--hot"
    :class="{ 'is-collapsed': hotCollapsed }"
    :style="{ top: hotTop, maxHeight: hotMaxHeight }"
    aria-label="浏览热度最高的笔记"
  >
    <div
      class="sidebar-header widget-header"
      role="button"
      :aria-expanded="String(!hotCollapsed)"
      title="收起/展开浏览热度"
      @click="toggleHot"
    >
      <span class="sidebar-title">浏览热度</span>
      <span class="sidebar-count">{{ hot.length ? `TOP ${hot.length}` : '暂无数据' }}</span>
      <button
        class="sidebar-collapse"
        type="button"
        :title="hotCollapsed ? '展开浏览热度' : '收起浏览热度'"
        :aria-label="hotCollapsed ? '展开浏览热度' : '收起浏览热度'"
      >
        <svg v-if="hotCollapsed" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m6 9 6 6 6-6"/></svg>
        <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m18 15-6-6-6 6"/></svg>
      </button>
    </div>
    <div class="sidebar-tree widget-list">
      <button
        v-for="node in hot"
        :key="node.id"
        type="button"
        class="tree-row is-note"
        title="单击在星图中定位 · 双击打开笔记"
        @click="emit('select', node)"
        @dblclick="emit('open', node)"
      >
        <span class="tree-dot" :style="{ background: node.color, color: node.color }"></span>
        <span class="tree-name">{{ node.name }}</span>
        <span class="widget-meta" title="阅读量">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg><span>{{ formatCount(node.pv) }} 次</span>
        </span>
      </button>
    </div>
  </aside>
</template>

<style scoped>
/* 统计卡片 - 右下方 */
.header-panel {
  bottom: 18px;
  right: 18px;
  width: min(430px, calc(100vw - 36px));
  padding: 14px 16px;
}

.eyebrow {
  margin: 0 0 4px;
  color: var(--accent-text);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.header-panel h1 {
  margin: 0;
  color: var(--title-color);
  font-size: 19px;
  line-height: 1.2;
  font-weight: 750;
}

.metric-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 7px;
  margin-top: 12px;
}

.metric {
  min-width: 0;
  padding: 7px 8px;
  background: rgba(148, 163, 184, 0.07);
  border: 1px solid rgba(148, 163, 184, 0.12);
  border-radius: 8px;
}

.metric strong {
  display: block;
  color: var(--title-color);
  font-size: 16px;
  line-height: 1;
}

.metric span {
  display: block;
  margin-top: 5px;
  color: var(--muted);
  font-size: 10px;
  white-space: nowrap;
}

/* 站点总浏览量：标题下方的弱化一行 */
.site-view-stats {
  margin: 7px 0 0;
  color: var(--muted);
  font-size: 11px;
  letter-spacing: 0.02em;
}

.site-view-stats strong {
  color: var(--accent-text);
  font-weight: 750;
  font-variant-numeric: tabular-nums;
}

.legend-panel {
  top: 18px;
  right: 18px;
  min-width: 166px;
  padding: 11px 12px;
  display: grid;
  gap: 8px;
}

.legend-title {
  color: var(--muted);
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.10em;
  text-transform: uppercase;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #cbd5e1;
  font-size: 12px;
  line-height: 1.1;
}

.legend-dot {
  width: 11px;
  height: 11px;
  border-radius: 999px;
  box-shadow: 0 0 0 4px color-mix(in srgb, currentColor 14%, transparent);
  flex: 0 0 auto;
}

.toolbar {
  left: 18px;
  bottom: 18px;
  padding: 7px;
  display: flex;
  gap: 6px;
}

.icon-button {
  width: 36px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 8px;
  background: rgba(148, 163, 184, 0.06);
  color: #cbd5e1;
  cursor: pointer;
  transition: transform 160ms ease, background 160ms ease, color 160ms ease, border-color 160ms ease;
}

.icon-button:hover {
  transform: translateY(-1px);
  background: var(--accent-light);
  color: var(--accent-text);
  border-color: rgba(96, 165, 250, 0.42);
}

.icon-button:active { transform: translateY(0) scale(0.97); }

.icon-button svg {
  width: 17px;
  height: 17px;
  stroke-width: 2.1;
}

.icon-button.is-active {
  background: var(--accent-light);
  color: var(--accent-text);
  border-color: rgba(96, 165, 250, 0.42);
}

.sidebar-header {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 44px;
  padding: 0 10px 0 14px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  flex: 0 0 auto;
  cursor: pointer;
  user-select: none;
}

.sidebar-title {
  color: var(--title-color);
  font-size: 12.5px;
  font-weight: 750;
  letter-spacing: 0.04em;
}

.sidebar-count {
  margin-left: auto;
  color: var(--muted);
  font-size: 10.5px;
  white-space: nowrap;
}

.sidebar-collapse {
  flex: 0 0 auto;
  width: 22px;
  height: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: var(--muted);
  cursor: pointer;
  transition: background 140ms ease, color 140ms ease;
}

.sidebar-collapse:hover {
  background: rgba(148, 163, 184, 0.16);
  color: var(--title-color);
}

.sidebar-collapse svg { width: 14px; height: 14px; stroke-width: 2.4; }

/* 笔记目录下方的信息栏（最新更新 / 浏览热度）：外观与折叠动画完全复用侧边栏风格，
   top 由父组件动态计算（跟随侧边栏实际底边），过渡时长与侧边栏收起动画保持一致 */
.side-widget {
  left: 18px;
  /* 初始 top 仅是数据加载完成前的兜底，布局函数会修正 */
  top: 260px;
  width: min(310px, calc(100vw - 112px));
  max-height: 260px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  clip-path: inset(0 0 0 0 round 12px);
  transition:
    clip-path 320ms cubic-bezier(0.4, 0, 0.2, 1),
    top 260ms cubic-bezier(0.4, 0, 0.2, 1),
    max-height 260ms cubic-bezier(0.4, 0, 0.2, 1);
}

.side-widget.is-collapsed {
  /* 同侧边栏：只保留 44px 标题栏，其余卷起隐藏 */
  clip-path: inset(0 0 calc(100% - 44px) 0 round 12px);
  pointer-events: none;
}

.side-widget.is-collapsed .widget-header {
  pointer-events: auto;
}

.widget-list {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 6px;
  scrollbar-width: thin;
  scrollbar-color: rgba(148, 163, 184, 0.35) transparent;
}

.widget-list::-webkit-scrollbar { width: 6px; }
.widget-list::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.28); border-radius: 999px; }

.tree-row {
  display: flex;
  align-items: center;
  gap: 5px;
  width: 100%;
  border: 0;
  border-radius: 7px;
  background: transparent;
  padding: 4px 7px 4px 4px;
  font: inherit;
  font-size: 12px;
  text-align: left;
  cursor: pointer;
  color: var(--text-color);
  transition: background 140ms ease, color 140ms ease;
}

.tree-row:hover { background: rgba(148, 163, 184, 0.10); }

.tree-dot {
  flex: 0 0 auto;
  width: 7px;
  height: 7px;
  border-radius: 999px;
  margin: 0 4.5px;
  box-shadow: 0 0 6px currentColor;
}

.tree-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 行尾元信息（相对时间 / 阅读量）：复用 tree-pv 的小图标 + 数字风格 */
.widget-meta {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  color: var(--muted);
  font-size: 10px;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.widget-meta svg { width: 11px; height: 11px; stroke-width: 2; opacity: 0.75; }

@media (max-width: 760px) {
  /* 统计卡片：底部通栏，避免与工具栏在水平方向互相挤压重叠 */
  .header-panel {
    left: 10px;
    right: 10px;
    bottom: 10px;
    width: auto;
  }

  .legend-panel { display: none; }

  /* 移动端屏幕有限：最新更新 / 浏览热度两个信息栏不显示 */
  .side-widget { display: none; }

  /* 工具栏：右侧竖排居中，避开左上搜索/目录与底部统计卡片 */
  .toolbar {
    left: auto;
    right: 10px;
    top: 50%;
    bottom: auto;
    transform: translateY(-50%);
    flex-direction: column;
  }

  /* 通栏卡片宽度足够，指标恢复 4 列让卡片更矮 */
  .metric-row { grid-template-columns: repeat(4, minmax(0, 1fr)); }
}
</style>
