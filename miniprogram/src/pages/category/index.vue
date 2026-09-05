<template>
  <view class="page">
    <view v-if="loading" class="hint">加载中…</view>
    <view v-else-if="flatNodes.length === 0" class="hint">暂无目录</view>
    <view v-else class="tree">
      <view
        v-for="item in flatNodes"
        :key="item.node.slug"
        class="row"
        :style="{ paddingLeft: 24 + item.depth * 40 + 'rpx' }"
        @tap="onTap(item.node)"
      >
        <text v-if="item.node.type !== 'note'" class="arrow">
          {{ isExpanded(item.node.slug) ? "▾" : "▸" }}
        </text>
        <text v-else class="dot">📄</text>
        <text
          class="name"
          :class="{ note: item.node.type === 'note' }"
          >{{ item.node.name }}</text
        >
        <text v-if="item.node.type !== 'note'" class="count"
          >{{ item.node.note_count }}</text
        >
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { onShow } from "@dcloudio/uni-app";
import { request } from "@/common/request";

interface TreeNode {
  name: string;
  slug: string;
  type: string; // category | subcategory | note
  note_count: number;
  children: TreeNode[];
}

const tree = ref<TreeNode[]>([]);
const loading = ref(false);
/** 展开的目录 slug 集合（一级目录默认展开） */
const expanded = ref<Record<string, boolean>>({});

function isExpanded(slug: string): boolean {
  return !!expanded.value[slug];
}

interface FlatItem {
  node: TreeNode;
  depth: number;
}

/** 按展开状态把树扁平化为可见行 */
const flatNodes = computed<FlatItem[]>(() => {
  const out: FlatItem[] = [];
  const walk = (nodes: TreeNode[], depth: number) => {
    for (const n of nodes) {
      out.push({ node: n, depth });
      if (
        n.type !== "note" &&
        n.children.length > 0 &&
        isExpanded(n.slug)
      ) {
        walk(n.children, depth + 1);
      }
    }
  };
  walk(tree.value, 0);
  return out;
});

async function loadTree() {
  loading.value = true;
  try {
    const root = await request<TreeNode>({ url: "/articles/tree" });
    tree.value = root.children || [];
    // 默认展开一级目录
    const map: Record<string, boolean> = {};
    for (const c of tree.value) map[c.slug] = true;
    expanded.value = map;
  } catch (e) {
    // request 已 toast
  } finally {
    loading.value = false;
  }
}

onShow(loadTree);

function onTap(node: TreeNode) {
  if (node.type === "note") {
    uni.navigateTo({
      url: `/pages/article/index?slug=${encodeURIComponent(node.slug)}`,
    });
    return;
  }
  if (node.children.length === 0) return;
  expanded.value[node.slug] = !expanded.value[node.slug];
}
</script>

<style>
.tree {
  padding: 12rpx 0 60rpx;
}
.row {
  display: flex;
  align-items: center;
  gap: 16rpx;
  padding-top: 20rpx;
  padding-bottom: 20rpx;
  padding-right: 24rpx;
  border-bottom: 1rpx solid #1e2a4a;
}
.arrow {
  color: #38bdf8;
  width: 32rpx;
}
.dot {
  font-size: 24rpx;
}
.name {
  color: #e2e8f0;
  font-size: 28rpx;
  flex: 1;
}
.name.note {
  color: #cbd5e1;
}
.count {
  color: #94a3b8;
  font-size: 22rpx;
  background: #111c33;
  border-radius: 999rpx;
  padding: 2rpx 16rpx;
}
.hint {
  text-align: center;
  color: #94a3b8;
  padding: 80rpx 0;
}
</style>
