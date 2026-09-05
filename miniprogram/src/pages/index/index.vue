<template>
  <view class="page">
    <!-- 顶部分类入口：tree 一级目录 + 博客 -->
    <scroll-view scroll-x class="cats" enhanced :show-scrollbar="false">
      <view class="cats-row">
        <view class="cat-chip blog-chip" @tap="goBlog">📝 博客</view>
        <view
          v-for="c in categories"
          :key="c.slug"
          class="cat-chip"
          @tap="goCategory"
        >
          {{ c.name }}
          <text class="cat-count">{{ c.note_count }}</text>
        </view>
      </view>
    </scroll-view>

    <!-- 榜单切换：最新更新 / 浏览热度 -->
    <view class="tabs">
      <view
        class="tab"
        :class="{ active: tab === 'latest' }"
        @tap="tab = 'latest'"
        >最新更新</view
      >
      <view
        class="tab"
        :class="{ active: tab === 'hot' }"
        @tap="tab = 'hot'"
        >浏览热度</view
      >
    </view>

    <view v-if="loading" class="hint">加载中…</view>
    <view v-else-if="currentList.length === 0" class="hint">暂无内容</view>
    <view v-else class="list">
      <view
        v-for="a in currentList"
        :key="a.id"
        class="card"
        @tap="goArticle(a.slug)"
      >
        <view class="card-title">{{ a.title }}</view>
        <view class="card-meta">
          <text class="badge">{{ a.category }}</text>
          <text>👁 {{ a.views }}</text>
          <text>👍 {{ a.like_count }}</text>
          <text>💬 {{ a.comment_count }}</text>
          <text class="date">{{ formatDate(a.content_updated_at) }}</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { onShow } from "@dcloudio/uni-app";
import { request } from "@/common/request";

interface ArticleListItem {
  id: number;
  slug: string;
  title: string;
  category: string;
  views: number;
  like_count: number;
  comment_count: number;
  content_updated_at: string | null;
}

interface TreeNode {
  name: string;
  slug: string;
  type: string;
  note_count: number;
  children: TreeNode[];
}

const categories = ref<TreeNode[]>([]);
const latest = ref<ArticleListItem[]>([]);
const hot = ref<ArticleListItem[]>([]);
const tab = ref<"latest" | "hot">("latest");
const loading = ref(false);

const currentList = computed(() =>
  tab.value === "latest" ? latest.value : hot.value
);

function formatDate(s: string | null): string {
  return s ? s.slice(0, 10) : "";
}

async function loadData() {
  loading.value = true;
  try {
    const [tree, latestList, hotList] = await Promise.all([
      request<TreeNode>({ url: "/articles/tree" }),
      request<ArticleListItem[]>({ url: "/articles/latest?n=10" }),
      request<ArticleListItem[]>({ url: "/articles/hot?n=10" }),
    ]);
    categories.value = tree.children || [];
    latest.value = latestList;
    hot.value = hotList;
  } catch (e) {
    // request 已 toast
  } finally {
    loading.value = false;
  }
}

onShow(loadData);

function goCategory() {
  uni.navigateTo({ url: "/pages/category/index" });
}

function goBlog() {
  uni.navigateTo({ url: "/pages/blog/index" });
}

function goArticle(slug: string) {
  uni.navigateTo({
    url: `/pages/article/index?slug=${encodeURIComponent(slug)}`,
  });
}
</script>

<style>
.cats {
  white-space: nowrap;
  padding: 20rpx 24rpx 4rpx;
}
.cats-row {
  display: inline-flex;
  gap: 16rpx;
}
.cat-chip {
  display: inline-flex;
  align-items: center;
  padding: 10rpx 24rpx;
  border-radius: 999rpx;
  background: #111c33;
  border: 1rpx solid #1e2a4a;
  color: #e2e8f0;
  font-size: 26rpx;
}
.blog-chip {
  color: #38bdf8;
  border-color: #38bdf8;
}
.cat-count {
  margin-left: 8rpx;
  font-size: 22rpx;
  color: #94a3b8;
}
.tabs {
  display: flex;
  gap: 32rpx;
  padding: 24rpx 24rpx 8rpx;
}
.tab {
  font-size: 30rpx;
  font-weight: 600;
  color: #94a3b8;
  padding-bottom: 8rpx;
  border-bottom: 4rpx solid transparent;
}
.tab.active {
  color: #38bdf8;
  border-bottom-color: #38bdf8;
}
.hint {
  text-align: center;
  color: #94a3b8;
  padding: 80rpx 0;
}
.list {
  padding: 16rpx 24rpx 40rpx;
}
.card {
  background: #111c33;
  border: 1rpx solid #1e2a4a;
  border-radius: 16rpx;
  padding: 24rpx;
  margin-bottom: 20rpx;
}
.card-title {
  font-size: 30rpx;
  font-weight: 600;
  color: #f1f5f9;
  margin-bottom: 12rpx;
}
.card-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 20rpx;
  font-size: 24rpx;
  color: #94a3b8;
  align-items: center;
}
.badge {
  color: #38bdf8;
  background: rgba(56, 189, 248, 0.12);
  padding: 2rpx 14rpx;
  border-radius: 8rpx;
}
.date {
  margin-left: auto;
}
</style>
