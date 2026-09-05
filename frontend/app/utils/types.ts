/** 后端 API 响应类型（与 backend/app/schemas 对齐） */

export interface ArticleListItem {
  id: number
  slug: string
  title: string
  category: string
  format: string
  views: number
  like_count: number
  comment_count: number
  content_updated_at: string | null
}

export interface ArticleDetail {
  id: number
  slug: string
  title: string
  category: string
  format: string
  html: string
  views: number
  like_count: number
  comment_count: number
  content_updated_at: string | null
  meta: Record<string, any>
}

export interface CommentOut {
  id: number
  article_id: number
  nickname: string
  content: string
  parent_id: number | null
  created_at: string
  children: CommentOut[]
}

export interface SearchResult {
  slug: string
  title: string
  category: string
  snippet: string
}

export interface TreeNode {
  name: string
  slug: string
  type: 'root' | 'category' | 'subcategory' | 'note'
  note_count: number
  children: TreeNode[]
}

export interface LikeOut {
  liked: boolean
  like_count: number
}

export interface ViewOut {
  counted: boolean
  views: number
}
