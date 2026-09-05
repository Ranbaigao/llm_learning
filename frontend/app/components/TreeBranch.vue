<script setup lang="ts">
defineOptions({ name: 'TreeBranch' })

const props = defineProps<{
  node: TreeNode
  currentSlug: string
  toggledSet: Set<string>
  encodeSlug: (slug: string) => string
}>()

const emit = defineEmits<{
  toggle: [slug: string]
  navigate: []
}>()

const isDir = computed(() => props.node.type !== 'note')
// 与 ArticleTreeNav 一致的确定性规则：当前文章路径上的目录默认展开，toggled 为手动翻转
const isOpen = computed(() => {
  const def = props.currentSlug ? props.currentSlug.startsWith(props.node.slug + '/') : true
  return props.toggledSet.has(props.node.slug) ? !def : def
})
const isActive = computed(() => props.node.slug === props.currentSlug)
</script>

<template>
  <div class="branch">
    <div v-if="isDir" class="dir-row" @click="emit('toggle', node.slug)">
      <span class="arrow" :class="{ open: isOpen }">▸</span>
      <span class="dir-name">{{ node.name }}</span>
      <span class="count">{{ node.note_count }}</span>
    </div>
    <NuxtLink
      v-else
      :to="'/notes/' + encodeSlug(node.slug)"
      class="note-link"
      :class="{ active: isActive }"
      @click="emit('navigate')"
    >
      {{ node.name }}
    </NuxtLink>

    <div v-if="isDir && isOpen && node.children.length" class="children">
      <TreeBranch
        v-for="child in node.children"
        :key="child.slug"
        :node="child"
        :current-slug="currentSlug"
        :toggled-set="toggledSet"
        :encode-slug="encodeSlug"
        @toggle="(s) => emit('toggle', s)"
        @navigate="emit('navigate')"
      />
    </div>
  </div>
</template>

<style scoped>
.branch {
  margin: 2px 0;
}
.dir-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  border-radius: 6px;
  cursor: pointer;
  color: var(--text-secondary);
  user-select: none;
}
.dir-row:hover {
  background: rgba(37, 99, 235, 0.12);
  color: var(--text-primary);
}
.arrow {
  font-size: 11px;
  color: var(--text-faint);
  transition: transform 0.15s ease;
  width: 12px;
  flex-shrink: 0;
}
.arrow.open {
  transform: rotate(90deg);
}
.dir-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.count {
  margin-left: auto;
  font-size: 11px;
  color: var(--text-faint);
  background: rgba(56, 189, 248, 0.08);
  padding: 0 6px;
  border-radius: 8px;
  flex-shrink: 0;
}
.children {
  margin-left: 14px;
  border-left: 1px solid rgba(56, 189, 248, 0.1);
  padding-left: 6px;
}
.note-link {
  display: block;
  padding: 5px 8px 5px 26px;
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.note-link:hover {
  background: rgba(37, 99, 235, 0.12);
  color: var(--text-primary);
  text-shadow: none;
}
.note-link.active {
  color: var(--accent);
  background: rgba(37, 99, 235, 0.2);
  box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.3);
}
</style>
