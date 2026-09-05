<script setup lang="ts">
const { apiFetch, encodeSlug } = useApi()

useHead({
  title: '博客 · 星尘知识库',
  meta: [{ name: 'description', content: '更即时的记录：随笔、想法与阶段性总结' }],
})

const { data: posts } = await useAsyncData<ArticleListItem[]>('blog-list', () =>
  apiFetch<ArticleListItem[]>('/api/articles', { params: { category: 'blog', n: 50 } }),
)

// 列表项没有 frontmatter，逐篇取详情拿 meta.date / categories（博客数量少）
const { data: details } = await useAsyncData<Record<number, ArticleDetail>>(
  'blog-details',
  async () => {
    const map: Record<number, ArticleDetail> = {}
    await Promise.all(
      (posts.value || []).map(async (p) => {
        map[p.id] = await apiFetch<ArticleDetail>(`/api/articles/${encodeSlug(p.slug)}`)
      }),
    )
    return map
  },
  { watch: [posts] },
)

function dateOf(p: ArticleListItem): string {
  const meta = details.value?.[p.id]?.meta || {}
  return String(meta.date || p.content_updated_at || '').slice(0, 10)
}

function tagsOf(p: ArticleListItem): string[] {
  const cats = details.value?.[p.id]?.meta?.categories
  if (Array.isArray(cats)) return cats.map(String)
  if (typeof cats === 'string') return [cats]
  return []
}

function linkOf(p: ArticleListItem): string {
  return '/blog/' + encodeSlug(p.slug.replace(/^blog\//, ''))
}
</script>

<template>
  <div class="blog-page">
    <header class="blog-head">
      <h1 class="blog-title">📝 博客</h1>
      <p class="blog-sub">更即时的记录：随笔、想法与阶段性总结</p>
    </header>

    <div v-if="!posts?.length" class="card empty-tip">暂无博客文章</div>

    <ul v-else class="post-list">
      <li v-for="p in posts" :key="p.id" class="card post-card">
        <NuxtLink :to="linkOf(p)" class="post-link">
          <h2 class="post-title">{{ p.title }}</h2>
          <div class="post-meta">
            <span>📅 {{ dateOf(p) }}</span>
            <span v-for="t in tagsOf(p)" :key="t" class="post-tag">{{ t }}</span>
            <span class="post-stats">👁 {{ p.views }} · ♥ {{ p.like_count }} · 💬 {{ p.comment_count }}</span>
          </div>
        </NuxtLink>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.blog-page {
  max-width: 800px;
  margin: 0 auto;
}
.blog-head {
  margin-bottom: 24px;
}
.blog-title {
  margin: 0 0 6px;
  font-size: 30px;
  text-shadow: 0 0 24px rgba(56, 189, 248, 0.35);
}
.blog-sub {
  margin: 0;
  color: var(--text-faint);
  font-size: 14px;
}
.empty-tip {
  padding: 40px;
  text-align: center;
  color: var(--text-faint);
}
.post-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.post-card {
  transition: all 0.2s ease;
}
.post-card:hover {
  border-color: rgba(56, 189, 248, 0.4);
  box-shadow: var(--glow-accent);
}
.post-link {
  display: block;
  padding: 20px 24px;
  color: inherit;
}
.post-link:hover {
  text-shadow: none;
}
.post-title {
  margin: 0 0 10px;
  font-size: 20px;
}
.post-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  font-size: 13px;
  color: var(--text-faint);
}
.post-tag {
  padding: 1px 10px;
  border-radius: 10px;
  background: rgba(56, 189, 248, 0.12);
  color: var(--accent);
  font-size: 12px;
}
.post-stats {
  margin-left: auto;
}
</style>
