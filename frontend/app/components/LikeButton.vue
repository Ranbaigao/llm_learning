<script setup lang="ts">
const props = defineProps<{
  articleId: number
  likeCount: number
}>()

const emit = defineEmits<{
  update: [likeCount: number]
}>()

const { apiFetch } = useApi()
const { ensure } = useVisitor()

const LIKED_KEY = 'kb-liked-articles'
const liked = ref(false)
const pending = ref(false)

function readLiked(): number[] {
  if (!import.meta.client) return []
  try {
    return JSON.parse(localStorage.getItem(LIKED_KEY) || '[]')
  } catch {
    return []
  }
}

function writeLiked(ids: number[]) {
  localStorage.setItem(LIKED_KEY, JSON.stringify(ids))
}

onMounted(() => {
  liked.value = readLiked().includes(props.articleId)
})

// 同路由切换文章（组件复用场景）时按新文章恢复点赞状态
watch(
  () => props.articleId,
  (id) => {
    liked.value = readLiked().includes(id)
  },
)

async function toggle() {
  if (pending.value) return
  const visitor_id = ensure()
  if (!visitor_id) return
  pending.value = true

  const prevLiked = liked.value
  const prevCount = props.likeCount
  // 乐观更新
  liked.value = !prevLiked
  emit('update', prevLiked ? prevCount - 1 : prevCount + 1)

  try {
    const res = await apiFetch<LikeOut>(`/api/articles/${props.articleId}/like`, {
      method: prevLiked ? 'DELETE' : 'POST',
      body: { visitor_id },
    })
    liked.value = res.liked
    emit('update', res.like_count)
    const ids = readLiked().filter((id) => id !== props.articleId)
    if (res.liked) ids.push(props.articleId)
    writeLiked(ids)
  } catch {
    // 回滚
    liked.value = prevLiked
    emit('update', prevCount)
  } finally {
    pending.value = false
  }
}
</script>

<template>
  <button
    class="like-btn"
    :class="{ liked }"
    :disabled="pending"
    :title="liked ? '取消点赞' : '点赞'"
    @click="toggle"
  >
    <span class="heart">{{ liked ? '❤️' : '🤍' }}</span>
    <span class="count">{{ likeCount }}</span>
    <span class="label">{{ liked ? '已赞' : '点赞' }}</span>
  </button>
</template>

<style scoped>
.like-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 18px;
  border-radius: 20px;
  border: 1px solid rgba(56, 189, 248, 0.35);
  background: rgba(15, 23, 42, 0.6);
  color: var(--text-secondary);
  font-size: 14px;
  cursor: pointer;
  transition: all 0.25s ease;
}
.like-btn:hover:not(:disabled) {
  border-color: var(--accent);
  box-shadow: var(--glow-accent);
}
.like-btn.liked {
  border-color: rgba(244, 114, 182, 0.6);
  background: rgba(244, 114, 182, 0.12);
  color: #f9a8d4;
  box-shadow: 0 0 14px rgba(244, 114, 182, 0.25);
}
.like-btn:disabled {
  cursor: wait;
}
.count {
  font-weight: 600;
}
</style>
