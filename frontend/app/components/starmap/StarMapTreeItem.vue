<script setup lang="ts">
/**
 * 笔记目录树的单个节点（递归组件）：目录行带折叠箭头，笔记行带色点与阅读量。
 * 单击目录行：展开/收起 + 星图定位；单击箭头：仅展开/收起；双击笔记：打开笔记页。
 */
import { formatCount } from './engine'
import type { SidebarTreeNode } from './engine'

const props = defineProps<{
  node: SidebarTreeNode
  activeId: string | null
  expanded: Set<string>
}>()

const emit = defineEmits<{
  toggle: [id: string]
  select: [node: SidebarTreeNode]
  open: [node: SidebarTreeNode]
}>()

const isDir = computed(() => props.node.children.length > 0)

function onRowClick() {
  if (isDir.value) emit('toggle', props.node.id)
  emit('select', props.node)
}

function onDblClick() {
  if (props.node.type === 'note') emit('open', props.node)
}
</script>

<template>
  <div class="tree-item" :class="{ 'is-expanded': expanded.has(node.id) }">
    <button
      type="button"
      class="tree-row"
      :class="{ 'is-dir': isDir, 'is-note': !isDir, 'is-active': activeId === node.id }"
      :data-id="node.id"
      :title="node.type === 'note' ? '单击在星图中定位 · 双击打开笔记' : '单击定位并展开/收起'"
      @click="onRowClick"
      @dblclick="onDblClick"
    >
      <span v-if="isDir" class="tree-chevron" @click.stop="emit('toggle', node.id)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m9 5 7 7-7 7"/></svg>
      </span>
      <span v-else class="tree-dot" :style="{ background: node.color, color: node.color }"></span>
      <span class="tree-name">{{ node.name }}</span>
      <span v-if="isDir" class="tree-count">{{ node.noteCount || '' }}</span>
      <span v-else-if="node.type === 'note'" class="tree-pv" title="阅读量" :hidden="node.pv == null">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/><circle cx="12" cy="12" r="3"/></svg><span class="tree-pv-num">{{ formatCount(node.pv) }}</span>
      </span>
    </button>
    <div v-if="isDir" class="tree-children">
      <StarMapTreeItem
        v-for="child in node.children"
        :key="child.id"
        :node="child"
        :active-id="activeId"
        :expanded="expanded"
        @toggle="emit('toggle', $event)"
        @select="emit('select', $event)"
        @open="emit('open', $event)"
      />
    </div>
  </div>
</template>

<style scoped>
/* 侧边栏笔记行的阅读量（小眼睛图标 + 数字） */
.tree-pv {
  flex: 0 0 auto;
  display: inline-flex;
  align-items: center;
  gap: 3px;
  color: var(--muted);
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}

.tree-pv[hidden] { display: none; }

.tree-pv svg { width: 11px; height: 11px; stroke-width: 2; opacity: 0.75; }

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

.tree-row.is-active { background: rgba(59, 130, 246, 0.18); }

.tree-chevron {
  flex: 0 0 auto;
  width: 16px;
  height: 16px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  border-radius: 4px;
  transition: transform 180ms ease, background 140ms ease;
}

.tree-chevron:hover { background: rgba(148, 163, 184, 0.16); color: var(--title-color); }

.tree-chevron svg { width: 11px; height: 11px; stroke-width: 2.4; }

.tree-item.is-expanded > .tree-row .tree-chevron { transform: rotate(90deg); }

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

.tree-row.is-dir .tree-name { font-weight: 650; color: var(--title-color); }

.tree-row.is-active .tree-name { color: #dbeafe; }

.tree-count {
  flex: 0 0 auto;
  color: var(--muted);
  font-size: 10px;
}

.tree-children {
  margin-left: 11px;
  padding-left: 7px;
  border-left: 1px solid rgba(148, 163, 184, 0.12);
  display: none;
}

.tree-item.is-expanded > .tree-children { display: block; }
</style>
