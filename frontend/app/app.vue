<script setup lang="ts">
const appConfig = useAppConfig()
const route = useRoute()
// 首页为全屏星图：主区去宽度限制与内边距，且不显示页脚
const isHome = computed(() => route.path === '/')
</script>

<template>
  <div class="site-shell">
    <SiteHeader />
    <main class="site-main" :class="{ 'site-main--home': isHome }">
      <NuxtPage />
    </main>
    <footer v-if="!isHome" class="site-footer">
      <span>{{ appConfig.siteName }} · 以笔记为星，以思考为轨</span>
    </footer>
  </div>
</template>
