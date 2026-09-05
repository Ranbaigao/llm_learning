<script setup lang="ts">
/**
 * 笔记目录侧边栏：树形折叠目录，与星图节点一一对应。
 * 折叠时树形列表从下往上卷起，顶部标题栏原样保留并充当展开把手；
 * top/max-height 由父组件按搜索结果弹层与下方信息栏位置动态下发。
 */
import StarMapTreeItem from './StarMapTreeItem.vue'
import type { SidebarTreeNode } from './engine'

const props = defineProps<{
  tree: SidebarTreeNode[]
  noteCount: number
  activeId: string | null
  collapsed: boolean
  top: string
  maxHeight: string
}>()

const emit = defineEmits<{
  'update:collapsed': [value: boolean]
  select: [node: SidebarTreeNode]
  open: [node: SidebarTreeNode]
  /** 目录展开/收起改变侧边栏高度，父组件据此重排信息栏 */
  layoutChange: []
}>()

// 各目录节点的展开状态（一级目录默认折叠，只展示顶层目录名）
const expanded = ref(new Set<string>())
const rootEl = ref<HTMLElement | null>(null)

// 子节点 id → 父节点 id，用于星图选中时展开祖先目录
const parentMap = computed(() => {
  const map = new Map<string, string>()
  const walk = (nodes: SidebarTreeNode[], parentId: string | null) => {
    for (const node of nodes) {
      if (parentId) map.set(node.id, parentId)
      walk(node.children, node.id)
    }
  }
  walk(props.tree, null)
  return map
})

function toggle(id: string) {
  const next = new Set(expanded.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  expanded.value = next
  emit('layoutChange')
}

// 星图选中状态同步到侧边栏：高亮对应行、展开祖先目录并滚动到可见位置
watch(
  () => props.activeId,
  async id => {
    if (!id) return
    const next = new Set(expanded.value)
    let parent = parentMap.value.get(id)
    while (parent) {
      next.add(parent)
      parent = parentMap.value.get(parent)
    }
    expanded.value = next
    emit('layoutChange')
    await nextTick()
    rootEl.value
      ?.querySelector(`[data-id="${CSS.escape(id)}"]`)
      ?.scrollIntoView({ block: 'nearest' })
  },
)
</script>

<template>
  <aside
    ref="rootEl"
    class="panel sidebar"
    :class="{ 'is-collapsed': collapsed }"
    :style="{ top, maxHeight }"
    aria-label="笔记目录"
  >
    <div
      class="sidebar-header"
      role="button"
      :aria-expanded="String(!collapsed)"
      title="收起/展开笔记目录"
      @click="emit('update:collapsed', !collapsed)"
    >
      <span class="sidebar-title">笔记目录</span>
      <span class="sidebar-count">{{ noteCount }} 篇笔记</span>
      <button
        class="sidebar-collapse"
        type="button"
        :title="collapsed ? '展开笔记目录' : '收起笔记目录'"
        :aria-label="collapsed ? '展开笔记目录' : '收起笔记目录'"
      >
        <!-- 展开状态显示上箭头（收起），折叠状态显示下箭头（展开） -->
        <svg v-if="collapsed" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m6 9 6 6 6-6"/></svg>
        <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m18 15-6-6-6 6"/></svg>
      </button>
    </div>
    <div class="sidebar-tree" role="tree">
      <StarMapTreeItem
        v-for="node in tree"
        :key="node.id"
        :node="node"
        :active-id="activeId"
        :expanded="expanded"
        @toggle="toggle"
        @select="emit('select', $event)"
        @open="emit('open', $event)"
      />
    </div>
  </aside>
</template>

<style scoped>
/* 笔记目录侧边栏：折叠时树形列表从下往上卷起，顶部标题栏原样保留
   并充当展开把手；宽度与搜索框对齐；
   top 过渡用于搜索候选列表弹出时动态下移让位 */
.sidebar {
  top: 78px;
  left: 18px;
  /* 高度随目录行数自适应：不设 bottom，顶边紧贴搜索栏下方；
     max-height 仅是兜底——运行时由父组件按下方信息栏与工具栏的位置
     动态收紧，目录展开过多时树形列表内部滚动 */
  max-height: calc(100vh - 152px);
  width: min(310px, calc(100vw - 112px));
  display: flex;
  flex-direction: column;
  overflow: hidden;
  clip-path: inset(0 0 0 0 round 12px);
  transition:
    clip-path 320ms cubic-bezier(0.4, 0, 0.2, 1),
    top 260ms cubic-bezier(0.4, 0, 0.2, 1),
    max-height 260ms cubic-bezier(0.4, 0, 0.2, 1);
}

.sidebar.is-collapsed {
  /* 只保留 44px 标题栏，其余部分卷起隐藏 */
  clip-path: inset(0 0 calc(100% - 44px) 0 round 12px);
  pointer-events: none;
}

.sidebar.is-collapsed .sidebar-header {
  pointer-events: auto;
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

/* 侧边栏头部的收起按钮 */
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

.sidebar-tree {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 6px;
  scrollbar-width: thin;
  scrollbar-color: rgba(148, 163, 184, 0.35) transparent;
}

.sidebar-tree::-webkit-scrollbar { width: 6px; }
.sidebar-tree::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.28); border-radius: 999px; }

@media (max-width: 760px) {
  .sidebar {
    top: 64px;
    left: 10px;
    max-height: calc(100vh - 124px);
    width: min(240px, calc(100vw - 66px));
  }
}
</style>
