<script setup lang="ts">
const route = useRoute()
const { apiFetch, encodeSlug } = useApi()
const { ensure } = useVisitor()

// 多级 slug：路由参数已解码，拼回后端使用的 posix 路径
const slug = computed(() => {
  const p = route.params.slug
  const arr = Array.isArray(p) ? p : [p]
  return arr.filter(Boolean).join('/')
})

const { data: article, error } = await useAsyncData<ArticleDetail>(
  // key 含 slug：同路由跳转（/notes/A → /notes/B）会重建页面组件，
  // 静态 key 会命中旧文章的缓存且不重新请求（真实踩过的坑）
  `note-article:${slug.value}`,
  () => apiFetch<ArticleDetail>(`/api/articles/${encodeSlug(slug.value)}`),
  // deep: 需要深层响应式，以便浏览量/点赞数的客户端变更刷新 UI
  { watch: [slug], deep: true },
)

if (error.value || !article.value) {
  // /notes 无 slug 时不报错，走下方目录页分支
  if (slug.value) {
    throw createError({ statusCode: 404, statusMessage: '文章不存在' })
  }
}

// 路由切换文章后滚回顶部
watch(slug, () => {
  if (import.meta.client) window.scrollTo({ top: 0 })
})

useHead(() => ({
  title: article.value
    ? `${article.value.title} · 星尘知识库`
    : '知识库目录 · 星尘知识库',
  meta: [
    {
      name: 'description',
      content: article.value
        ? `${article.value.category} · ${article.value.title}`
        : 'LLM 学习知识库笔记目录',
    },
  ],
}))

// 浏览量上报（客户端，后端按访客去重）；切换文章时对新文章再次上报
async function reportView() {
  if (!import.meta.client || !article.value) return
  try {
    const res = await apiFetch<ViewOut>(`/api/articles/${article.value.id}/view`, {
      method: 'POST',
      body: { visitor_id: ensure() },
    })
    if (article.value) article.value.views = res.views
  } catch {
    /* 浏览量上报失败不影响阅读 */
  }
}
onMounted(reportView)
watch(() => article.value?.id, reportView)

function onLikeUpdate(count: number) {
  if (article.value) article.value.like_count = count
}
</script>

<template>
  <div class="notes-layout">
    <ArticleTreeNav :current-slug="slug" />

    <div v-if="article" class="article-col">
      <ArticleView :article="article" />
      <div class="article-actions">
        <LikeButton
          :article-id="article.id"
          :like-count="article.like_count"
          @update="onLikeUpdate"
        />
      </div>
      <CommentSection :article-id="article.id" />
    </div>

    <div v-else class="article-col">
      <div class="card notes-index">
        <h1>📚 知识库笔记</h1>
        <p>从左侧目录选择一篇笔记开始阅读，或使用顶部搜索框检索内容。</p>
      </div>
    </div>
  </div>
</template>

<style scoped>
.notes-layout {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}
.article-col {
  flex: 1;
  min-width: 0;
}
.article-actions {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}
.notes-index {
  padding: 48px 32px;
  text-align: center;
  color: var(--text-secondary);
}
.notes-index h1 {
  margin-top: 0;
}
</style>
