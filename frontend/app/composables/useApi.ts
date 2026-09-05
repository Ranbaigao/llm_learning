/** API 封装：SSR 期间直连后端（runtimeConfig.apiServer），浏览器端走 /api 相对路径 */
export function useApi() {
  const config = useRuntimeConfig()
  const baseURL = import.meta.server ? String(config.apiServer || '') : ''
  const apiFetch = $fetch.create({ baseURL })

  /** slug 每段单独编码，保留路径分隔符 */
  function encodeSlug(slug: string): string {
    return slug.split('/').map(encodeURIComponent).join('/')
  }

  return { apiFetch, encodeSlug }
}
