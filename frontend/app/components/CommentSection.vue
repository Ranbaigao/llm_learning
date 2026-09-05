<script setup lang="ts">
const props = defineProps<{
  articleId: number
}>()

const { apiFetch } = useApi()
const { ensure } = useVisitor()

const NICK_KEY = 'kb-comment-nickname'

interface FlatComment extends CommentOut {
  depth: number
}

const comments = ref<CommentOut[]>([])
const loading = ref(true)
const loadError = ref('')

const nickname = ref('')
const content = ref('')
const replyTo = ref<CommentOut | null>(null)
const submitting = ref(false)
const errorMsg = ref('')
const successMsg = ref('')

// 树 → 扁平列表（带缩进层级）
const flatComments = computed<FlatComment[]>(() => {
  const out: FlatComment[] = []
  const walk = (nodes: CommentOut[], depth: number) => {
    for (const node of nodes) {
      out.push({ ...node, depth })
      if (node.children?.length) walk(node.children, depth + 1)
    }
  }
  walk(comments.value, 0)
  return out
})

function formatTime(iso: string): string {
  return iso ? iso.replace('T', ' ').slice(0, 16) : ''
}

async function loadComments() {
  loading.value = true
  loadError.value = ''
  try {
    comments.value = await apiFetch<CommentOut[]>(`/api/articles/${props.articleId}/comments`)
  } catch {
    loadError.value = '评论加载失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

function startReply(c: CommentOut) {
  replyTo.value = c
  errorMsg.value = ''
  document.getElementById('comment-form')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
}

function cancelReply() {
  replyTo.value = null
}

async function submit() {
  errorMsg.value = ''
  successMsg.value = ''
  const visitor_id = ensure()
  if (!visitor_id) return
  submitting.value = true
  try {
    await apiFetch(`/api/articles/${props.articleId}/comments`, {
      method: 'POST',
      body: {
        visitor_id,
        nickname: nickname.value.trim(),
        content: content.value.trim(),
        parent_id: replyTo.value?.id ?? null,
      },
    })
    localStorage.setItem(NICK_KEY, nickname.value.trim())
    content.value = ''
    replyTo.value = null
    successMsg.value = '发表成功'
    await loadComments()
  } catch (e: any) {
    const status = e?.response?.status
    if (status === 429) {
      errorMsg.value = '评论太频繁了，休息一下再发（同 IP 每分钟限 3 条）'
    } else if (status === 422) {
      errorMsg.value = '请检查输入：昵称 1–20 字，内容 1–2000 字，均不能为空'
    } else {
      errorMsg.value = '发表失败，请稍后重试'
    }
  } finally {
    submitting.value = false
  }
}

onMounted(() => {
  nickname.value = localStorage.getItem(NICK_KEY) || ''
  loadComments()
})

// 同路由切换文章（组件复用场景）时跟着加载新文章的评论
watch(() => props.articleId, () => loadComments())
</script>

<template>
  <section class="comment-section card">
    <h2 class="section-title">评论（{{ flatComments.length }}）</h2>

    <div v-if="loading" class="comment-tip">评论加载中…</div>
    <div v-else-if="loadError" class="comment-tip error">{{ loadError }}</div>
    <div v-else-if="!flatComments.length" class="comment-tip">还没有评论，来抢沙发～</div>

    <ul v-else class="comment-list">
      <li
        v-for="c in flatComments"
        :key="c.id"
        class="comment-item"
        :style="{ marginLeft: Math.min(c.depth, 4) * 28 + 'px' }"
      >
        <div class="comment-body">
          <div class="comment-head">
            <span class="comment-nick">{{ c.nickname }}</span>
            <span class="comment-time">{{ formatTime(c.created_at) }}</span>
          </div>
          <p class="comment-text">{{ c.content }}</p>
          <button class="reply-btn" @click="startReply(c)">回复</button>
        </div>
      </li>
    </ul>

    <form id="comment-form" class="comment-form" @submit.prevent="submit">
      <div v-if="replyTo" class="reply-indicator">
        回复 <b>{{ replyTo.nickname }}</b>
        <button type="button" class="cancel-reply" @click="cancelReply">×</button>
      </div>
      <div class="form-row">
        <input
          v-model="nickname"
          type="text"
          class="nick-input"
          placeholder="昵称（必填，20 字内）"
          maxlength="20"
        />
      </div>
      <textarea
        v-model="content"
        class="content-input"
        rows="4"
        placeholder="写下你的想法…（支持楼中楼回复）"
        maxlength="2000"
      ></textarea>
      <div class="form-footer">
        <span v-if="errorMsg" class="msg error">{{ errorMsg }}</span>
        <span v-else-if="successMsg" class="msg success">{{ successMsg }}</span>
        <button type="submit" class="btn submit-btn" :disabled="submitting">
          {{ submitting ? '发表中…' : '发表评论' }}
        </button>
      </div>
    </form>
  </section>
</template>

<style scoped>
.comment-section {
  margin-top: 24px;
  padding: 24px 28px;
}
.section-title {
  margin: 0 0 16px;
  font-size: 20px;
}
.comment-tip {
  color: var(--text-faint);
  font-size: 14px;
  padding: 12px 0;
}
.comment-tip.error {
  color: #f87171;
}
.comment-list {
  list-style: none;
  margin: 0 0 24px;
  padding: 0;
}
.comment-item {
  margin: 10px 0;
}
.comment-body {
  padding: 12px 16px;
  border-radius: 10px;
  background: rgba(11, 18, 32, 0.6);
  border: 1px solid rgba(56, 189, 248, 0.1);
}
.comment-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
}
.comment-nick {
  color: var(--accent);
  font-weight: 600;
  font-size: 14px;
}
.comment-time {
  color: var(--text-faint);
  font-size: 12px;
}
.comment-text {
  margin: 6px 0;
  font-size: 14px;
  white-space: pre-wrap;
}
.reply-btn {
  border: none;
  background: none;
  color: var(--text-faint);
  font-size: 12px;
  cursor: pointer;
  padding: 0;
}
.reply-btn:hover {
  color: var(--accent);
}
.comment-form {
  border-top: 1px solid rgba(56, 189, 248, 0.1);
  padding-top: 16px;
}
.reply-indicator {
  margin-bottom: 10px;
  font-size: 13px;
  color: var(--text-secondary);
}
.cancel-reply {
  border: none;
  background: none;
  color: var(--text-faint);
  cursor: pointer;
  font-size: 14px;
  margin-left: 6px;
}
.cancel-reply:hover {
  color: #f87171;
}
.form-row {
  margin-bottom: 10px;
}
.nick-input,
.content-input {
  width: 100%;
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid rgba(56, 189, 248, 0.2);
  background: rgba(3, 7, 18, 0.6);
  color: var(--text-primary);
  font-size: 14px;
  font-family: inherit;
  outline: none;
  transition: border-color 0.2s ease;
}
.nick-input {
  max-width: 280px;
}
.nick-input:focus,
.content-input:focus {
  border-color: var(--accent);
  box-shadow: var(--glow-accent);
}
.content-input {
  resize: vertical;
}
.form-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-top: 10px;
}
.msg {
  font-size: 13px;
}
.msg.error {
  color: #f87171;
}
.msg.success {
  color: #34d399;
}
.submit-btn {
  margin-left: auto;
}

@media (max-width: 768px) {
  .comment-section {
    padding: 18px 16px;
  }
}
</style>
