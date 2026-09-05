<template>
  <view class="page">
    <view v-if="loading" class="hint">加载中…</view>
    <view v-else-if="!article" class="hint">文章不存在或加载失败</view>
    <view v-else>
      <!-- 头部信息 -->
      <view class="header">
        <view class="title">{{ article.title }}</view>
        <view class="meta">
          <text class="badge">{{ article.category }}</text>
          <text v-if="displayDate">📅 {{ displayDate }}</text>
          <text>👁 {{ article.views }}</text>
          <text>💬 {{ article.comment_count }}</text>
        </view>
      </view>

      <!-- 正文：rich-text 渲染后端 HTML（数学公式仅显示 TeX 源码，见 README） -->
      <view class="body">
        <rich-text :nodes="processedHtml"></rich-text>
      </view>

      <!-- 点赞 -->
      <view class="like-bar">
        <view
          class="like-btn"
          :class="{ liked: liked }"
          @tap="toggleLike"
        >
          {{ liked ? "❤️ 已点赞" : "🤍 点赞" }} {{ article.like_count }}
        </view>
      </view>

      <!-- 评论区 -->
      <CommentList :article-id="article.id" @commented="onCommented" />
    </view>
  </view>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { onLoad } from "@dcloudio/uni-app";
import { request } from "@/common/request";
import { SERVER_ORIGIN } from "@/common/config";
import { getVisitorId } from "@/common/auth";
import CommentList from "@/components/CommentList.vue";

interface ArticleDetail {
  id: number;
  slug: string;
  title: string;
  category: string;
  format: string;
  html: string;
  views: number;
  like_count: number;
  comment_count: number;
  content_updated_at: string | null;
  meta: Record<string, any>;
}

const article = ref<ArticleDetail | null>(null);
const loading = ref(true);
const liked = ref(false);

const LIKED_KEY = "liked_ids";

function getLikedMap(): Record<string, boolean> {
  return uni.getStorageSync(LIKED_KEY) || {};
}

const displayDate = computed(() => {
  const a = article.value;
  if (!a) return "";
  const d = a.meta?.date || a.content_updated_at || "";
  return String(d).slice(0, 10);
});

/**
 * html 预处理：
 * 1. 后端图片等资源是 /api/assets/... 的相对路径，rich-text 的 img 需要完整 URL；
 * 2. 给 img 补 max-width，避免大图撑破屏幕；
 * 3. 给 table 补基础样式。
 */
const processedHtml = computed(() => {
  if (!article.value) return "";
  return article.value.html
    .replace(/(src|href)="\/api\//g, `$1="${SERVER_ORIGIN}/api/`)
    .replace(/<img(?![^>]*style=)/g, '<img style="max-width:100%;height:auto;"')
    .replace(/<table(?![^>]*style=)/g, '<table style="width:100%;"');
});

onLoad(async (options: any) => {
  const slug = decodeURIComponent(options?.slug || "");
  if (!slug) {
    loading.value = false;
    return;
  }
  try {
    const data = await request<ArticleDetail>({
      url: `/articles/${encodeURIComponent(slug)}`,
    });
    article.value = data;
    uni.setNavigationBarTitle({ title: data.title.slice(0, 20) });
    liked.value = !!getLikedMap()[String(data.id)];
    recordView(data.id);
  } catch (e) {
    // request 已 toast
  } finally {
    loading.value = false;
  }
});

async function recordView(id: number) {
  try {
    const res = await request<{ counted: boolean; views: number }>({
      url: `/articles/${id}/view`,
      method: "POST",
      data: { visitor_id: getVisitorId() },
      silent: true,
    });
    if (article.value) article.value.views = res.views;
  } catch (e) {
    // 浏览量上报失败无感
  }
}

/** 点赞：乐观更新，失败回滚 */
async function toggleLike() {
  const a = article.value;
  if (!a) return;
  const prevLiked = liked.value;
  const prevCount = a.like_count;
  liked.value = !prevLiked;
  a.like_count = prevCount + (liked.value ? 1 : -1);

  const map = getLikedMap();
  if (liked.value) map[String(a.id)] = true;
  else delete map[String(a.id)];
  uni.setStorageSync(LIKED_KEY, map);

  try {
    const res = await request<{ liked: boolean; like_count: number }>({
      url: `/articles/${a.id}/like`,
      method: prevLiked ? "DELETE" : "POST",
      data: { visitor_id: getVisitorId() },
    });
    liked.value = res.liked;
    a.like_count = res.like_count;
  } catch (e) {
    liked.value = prevLiked;
    a.like_count = prevCount;
    if (prevLiked) map[String(a.id)] = true;
    else delete map[String(a.id)];
    uni.setStorageSync(LIKED_KEY, map);
  }
}

function onCommented(count: number) {
  if (article.value) article.value.comment_count = count;
}
</script>

<style>
.header {
  padding: 32rpx 24rpx 24rpx;
  border-bottom: 1rpx solid #1e2a4a;
}
.title {
  font-size: 38rpx;
  font-weight: 700;
  color: #f1f5f9;
  margin-bottom: 16rpx;
  line-height: 1.4;
}
.meta {
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
.body {
  padding: 24rpx;
  color: #e2e8f0;
  font-size: 28rpx;
  word-break: break-word;
  overflow-wrap: break-word;
}
.like-bar {
  display: flex;
  justify-content: center;
  padding: 32rpx 0;
}
.like-btn {
  padding: 16rpx 48rpx;
  border-radius: 999rpx;
  border: 1rpx solid #1e2a4a;
  background: #111c33;
  color: #e2e8f0;
  font-size: 28rpx;
}
.like-btn.liked {
  border-color: #f472b6;
  color: #f472b6;
}
.hint {
  text-align: center;
  color: #94a3b8;
  padding: 80rpx 0;
}
</style>
