<template>
  <view class="page">
    <view v-if="loading" class="hint">加载中…</view>
    <view v-else-if="posts.length === 0" class="hint">暂无博客</view>
    <view v-else class="list">
      <view
        v-for="a in posts"
        :key="a.id"
        class="card"
        @tap="goArticle(a.slug)"
      >
        <view class="card-title">{{ a.title }}</view>
        <view class="card-meta">
          <text>📅 {{ formatDate(a.content_updated_at) }}</text>
          <text>👁 {{ a.views }}</text>
          <text>👍 {{ a.like_count }}</text>
          <text>💬 {{ a.comment_count }}</text>
        </view>
      </view>
    </view>
  </view>
</template>

<script setup lang="ts">
import { ref } from "vue";
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

const posts = ref<ArticleListItem[]>([]);
const loading = ref(false);

function formatDate(s: string | null): string {
  return s ? s.slice(0, 10) : "";
}

async function load() {
  loading.value = true;
  try {
    posts.value = await request<ArticleListItem[]>({
      url: "/articles?category=blog&n=50",
    });
  } catch (e) {
    // request 已 toast
  } finally {
    loading.value = false;
  }
}

onShow(load);

function goArticle(slug: string) {
  uni.navigateTo({
    url: `/pages/article/index?slug=${encodeURIComponent(slug)}`,
  });
}
</script>

<style>
.list {
  padding: 24rpx;
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
}
.hint {
  text-align: center;
  color: #94a3b8;
  padding: 80rpx 0;
}
</style>
