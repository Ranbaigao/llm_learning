/** 访客标识：localStorage 持久化 UUID，用于浏览量/点赞/评论去重 */
export function useVisitor() {
  const KEY = 'kb-visitor-id'
  const visitorId = useState<string>('visitor-id', () => '')

  /** 确保拿到 visitor_id（仅客户端有效） */
  function ensure(): string {
    if (!import.meta.client) return ''
    if (visitorId.value) return visitorId.value
    let id = localStorage.getItem(KEY)
    if (!id) {
      id = crypto.randomUUID()
      localStorage.setItem(KEY, id)
    }
    visitorId.value = id
    return id
  }

  onMounted(() => ensure())

  return { visitorId, ensure }
}
