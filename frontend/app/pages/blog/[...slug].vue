<script setup lang="ts">
// 博客文章页：slug 前缀 blog/，复用文章页组件
const route = useRoute()
const { apiFetch, encodeSlug } = useApi()
const { ensure } = useVisitor()

const slug = computed(() => {
  const p = route.params.slug
  const arr = Array.isArray(p) ? p : [p]
  return 'blog/' + arr.filter(Boolean).join('/')
})

const { data: article, error } = await useAsyncData<ArticleDetail>(
  // key 含 slug：防止同路由跳转时命中旧文章缓存（与 notes 页同理）
  `blog-article:${slug.value}`,
  () => apiFetch<ArticleDetail>(`/api/articles/${encodeSlug(slug.value)}`),
  // deep: 需要深层响应式，以便浏览量/点赞数的客户端变更刷新 UI
  { watch: [slug], deep: true },
)

if (error.value || !article.value) {
  throw createError({ statusCode: 404, statusMessage: '博客文章不存在' })
}

useHead(() => ({
  title: article.value ? `${article.value.title} · 博客` : '博客',
  meta: [
    {
      name: 'description',
      content: article.value ? `博客 · ${article.value.title}` : '博客文章',
    },
  ],
}))

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
  <div v-if="article" class="blog-article">
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
</template>

<style scoped>
.blog-article {
  max-width: 860px;
  margin: 0 auto;
}
.article-actions {
  margin-top: 20px;
  display: flex;
  justify-content: center;
}
</style>
