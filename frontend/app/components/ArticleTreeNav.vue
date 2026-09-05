<script setup lang="ts">
const props = defineProps<{
  currentSlug?: string
}>()

const { apiFetch, encodeSlug } = useApi()

const { data: tree } = useAsyncData<TreeNode>(
  'article-tree',
  () => apiFetch<TreeNode>('/api/articles/tree'),
  { lazy: false },
)

// 折叠状态确定性计算（SSR/客户端一致，避免 hydration mismatch）：
// 默认规则——目录处于「当前文章路径上」则展开，否则收起；toggled 记录用户手动翻转
const toggled = ref<Set<string>>(new Set())

function isOpen(node: TreeNode): boolean {
  const def = props.currentSlug ? props.currentSlug.startsWith(node.slug + '/') : true
  return toggled.value.has(node.slug) ? !def : def
}

function toggle(slug: string) {
  const next = new Set(toggled.value)
  if (next.has(slug)) next.delete(slug)
  else next.add(slug)
  toggled.value = next
}

// 移动端抽屉
const drawerOpen = ref(false)
function closeDrawer() {
  drawerOpen.value = false
}
</script>

<template>
  <!-- 移动端悬浮展开按钮 -->
  <button class="drawer-toggle" aria-label="打开目录" @click="drawerOpen = true">☰ 目录</button>
  <div v-if="drawerOpen" class="drawer-mask" @click="closeDrawer"></div>

  <aside class="tree-nav" :class="{ open: drawerOpen }">
    <div class="tree-head">
      <span class="tree-title">知识库目录</span>
      <button class="drawer-close" aria-label="关闭目录" @click="closeDrawer">×</button>
    </div>
    <nav v-if="tree" class="tree-body">
      <TreeBranch
        v-for="node in tree.children"
        :key="node.slug"
        :node="node"
        :current-slug="currentSlug || ''"
        :toggled-set="toggled"
        :encode-slug="encodeSlug"
        @toggle="toggle"
        @navigate="closeDrawer"
      />
    </nav>
  </aside>
</template>

<style scoped>
.tree-nav {
  width: 260px;
  flex-shrink: 0;
  align-self: flex-start;
  position: sticky;
  top: calc(var(--header-height) + 20px);
  max-height: calc(100vh - var(--header-height) - 40px);
  overflow-y: auto;
  background: var(--bg-card);
  border: 1px solid var(--border-card);
  border-radius: var(--radius-card);
  padding: 14px;
}
.tree-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(56, 189, 248, 0.1);
}
.tree-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}
.drawer-toggle,
.drawer-close,
.drawer-mask {
  display: none;
}
.tree-body {
  font-size: 14px;
}

@media (max-width: 900px) {
  .tree-nav {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    z-index: 210;
    width: 300px;
    max-width: 85vw;
    max-height: none;
    border-radius: 0;
    /* 抽屉形态用不透明背景，避免透出正文 */
    background: #0b1220;
    transform: translateX(-105%);
    transition: transform 0.25s ease;
  }
  .tree-nav.open {
    transform: translateX(0);
    box-shadow: 0 0 40px rgba(0, 0, 0, 0.6);
  }
  .drawer-toggle {
    display: block;
    position: fixed;
    left: 12px;
    bottom: 20px;
    z-index: 200;
    padding: 10px 16px;
    border-radius: 24px;
    border: 1px solid rgba(56, 189, 248, 0.4);
    background: rgba(11, 18, 32, 0.95);
    color: var(--accent);
    font-size: 14px;
    cursor: pointer;
    box-shadow: var(--glow-accent);
  }
  .drawer-close {
    display: block;
    border: none;
    background: none;
    color: var(--text-faint);
    font-size: 20px;
    cursor: pointer;
  }
  .drawer-mask {
    display: block;
    position: fixed;
    inset: 0;
    z-index: 205;
    background: rgba(0, 0, 0, 0.55);
  }
}
</style>
