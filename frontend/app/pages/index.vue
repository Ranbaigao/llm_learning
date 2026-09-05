<script setup lang="ts">
import StarMap from '~/components/starmap/StarMap.vue'

const appConfig = useAppConfig()

useHead({
  title: `${appConfig.siteName} · 首页`,
  meta: [{ name: 'description', content: appConfig.siteSlogan as string }],
  // 星图首页整页无滚动（旧站首页即星图满屏）；离开本页后 Nuxt 自动移除该属性
  bodyAttrs: { style: 'overflow: hidden' },
})
</script>

<template>
  <div class="home-starmap">
    <StarMap />
  </div>
</template>

<style scoped>
.home-starmap {
  width: 100%;
  /* 高度 100dvh 减去站点头部（app.vue 中 SiteHeader 高 60px） */
  height: calc(100dvh - var(--header-height));
}
</style>
