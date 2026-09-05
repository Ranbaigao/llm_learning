<script setup lang="ts">
const appConfig = useAppConfig()
const { apiFetch, encodeSlug } = useApi()
const route = useRoute()

const query = ref('')
const results = ref<SearchResult[]>([])
const searching = ref(false)
const showDropdown = ref(false)
const searchBox = ref<HTMLElement | null>(null)

let timer: ReturnType<typeof setTimeout> | null = null

function onInput() {
  if (timer) clearTimeout(timer)
  const q = query.value.trim()
  if (!q) {
    results.value = []
    showDropdown.value = false
    return
  }
  timer = setTimeout(doSearch, 300)
}

async function doSearch() {
  const q = query.value.trim()
  if (!q) return
  searching.value = true
  try {
    results.value = await apiFetch<SearchResult[]>('/api/search', {
      params: { q },
    })
    showDropdown.value = true
  } catch {
    results.value = []
  } finally {
    searching.value = false
  }
}

function resultLink(item: SearchResult): string {
  // 搜索结果里博客文章跳 /blog/，其余跳 /notes/
  if (item.slug.startsWith('blog/')) {
    return '/blog/' + encodeSlug(item.slug.slice('blog/'.length))
  }
  return '/notes/' + encodeSlug(item.slug)
}

function closeDropdown() {
  showDropdown.value = false
}

function onBlur() {
  // 延迟收起，让点击结果先生效
  setTimeout(() => (showDropdown.value = false), 200)
}

// 路由变化后清空搜索
watch(
  () => route.fullPath,
  () => {
    query.value = ''
    results.value = []
    showDropdown.value = false
  },
)
</script>

<template>
  <header class="site-header">
    <div class="header-inner">
      <NuxtLink to="/" class="brand">
        <span class="brand-star">✦</span>
        <span class="brand-name">{{ appConfig.siteName }}</span>
      </NuxtLink>

      <nav class="main-nav">
        <NuxtLink to="/" class="nav-link" :class="{ active: route.path === '/' }">首页</NuxtLink>
        <NuxtLink to="/notes" class="nav-link" :class="{ active: route.path.startsWith('/notes') }">笔记</NuxtLink>
        <NuxtLink to="/blog" class="nav-link" :class="{ active: route.path.startsWith('/blog') }">博客</NuxtLink>
      </nav>

      <div ref="searchBox" class="search-box">
        <input
          v-model="query"
          type="search"
          class="search-input"
          placeholder="搜索笔记…"
          @input="onInput"
          @focus="results.length && (showDropdown = true)"
          @blur="onBlur"
          @keyup.enter="doSearch"
        />
        <span v-if="searching" class="search-hint">搜索中…</span>
        <Transition name="fade">
          <ul v-if="showDropdown && results.length" class="search-dropdown">
            <li v-for="item in results" :key="item.slug">
              <NuxtLink :to="resultLink(item)" class="search-item" @click="closeDropdown">
                <span class="search-item-title">{{ item.title }}</span>
                <span class="search-item-cat">{{ item.category }}</span>
                <span class="search-item-snippet">{{ item.snippet }}</span>
              </NuxtLink>
            </li>
          </ul>
          <div v-else-if="showDropdown && query.trim() && !searching" class="search-dropdown empty">
            没有找到相关内容
          </div>
        </Transition>
      </div>
    </div>
  </header>
</template>

<style scoped>
.site-header {
  position: sticky;
  top: 0;
  z-index: 100;
  height: var(--header-height);
  background: rgba(3, 7, 18, 0.85);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(56, 189, 248, 0.12);
}
.header-inner {
  max-width: 1200px;
  height: 100%;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  align-items: center;
  gap: 24px;
}
.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.brand-star {
  color: var(--accent);
  font-size: 20px;
  text-shadow: 0 0 12px rgba(56, 189, 248, 0.8);
}
.brand-name {
  font-size: 18px;
  font-weight: 700;
  color: #f1f5f9;
  letter-spacing: 1px;
}
.main-nav {
  display: flex;
  gap: 4px;
}
.nav-link {
  padding: 6px 14px;
  border-radius: 8px;
  color: var(--text-secondary);
  font-size: 15px;
  transition: all 0.2s ease;
}
.nav-link:hover {
  color: var(--accent);
}
.nav-link.active {
  color: var(--accent);
  background: rgba(37, 99, 235, 0.18);
  box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.25);
}
.search-box {
  position: relative;
  margin-left: auto;
  width: 280px;
  flex-shrink: 1;
}
.search-input {
  width: 100%;
  padding: 8px 14px;
  border-radius: 20px;
  border: 1px solid rgba(56, 189, 248, 0.25);
  background: rgba(15, 23, 42, 0.7);
  color: var(--text-primary);
  font-size: 14px;
  outline: none;
  transition: all 0.2s ease;
}
.search-input:focus {
  border-color: var(--accent);
  box-shadow: var(--glow-accent);
}
.search-input::placeholder {
  color: var(--text-faint);
}
.search-hint {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 12px;
  color: var(--text-faint);
  pointer-events: none;
}
.search-dropdown {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: 380px;
  max-width: 90vw;
  max-height: 420px;
  overflow-y: auto;
  margin: 0;
  padding: 6px;
  list-style: none;
  background: rgba(11, 18, 32, 0.97);
  border: 1px solid var(--border-card);
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), var(--glow-accent);
}
.search-dropdown.empty {
  padding: 16px;
  color: var(--text-faint);
  font-size: 14px;
  text-align: center;
}
.search-item {
  display: block;
  padding: 10px 12px;
  border-radius: 8px;
}
.search-item:hover {
  background: rgba(37, 99, 235, 0.15);
  text-shadow: none;
}
.search-item-title {
  display: inline;
  color: #f1f5f9;
  font-size: 14px;
  font-weight: 600;
}
.search-item-cat {
  display: inline-block;
  margin-left: 8px;
  padding: 1px 8px;
  border-radius: 10px;
  font-size: 11px;
  color: var(--accent);
  background: rgba(56, 189, 248, 0.12);
}
.search-item-snippet {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-faint);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 768px) {
  .header-inner {
    gap: 10px;
    padding: 0 12px;
  }
  .brand-name {
    display: none;
  }
  .nav-link {
    padding: 6px 8px;
    font-size: 14px;
  }
  .search-box {
    width: auto;
    flex: 1;
  }
}
</style>
