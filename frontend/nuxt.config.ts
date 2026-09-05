// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: { enabled: false },
  ssr: true,

  css: ['~/assets/css/main.css', '~/assets/css/pygments.css'],

  runtimeConfig: {
    // 仅服务端可见：SSR 渲染期间直连后端，浏览器端走 /api 相对路径（dev 代理由下方 devProxy 处理）
    apiServer: process.env.NUXT_API_SERVER || 'http://127.0.0.1:8000',
  },

  nitro: {
    devProxy: {
      '/api': {
        target: 'http://127.0.0.1:8000/api',
        changeOrigin: true,
      },
    },
  },

  app: {
    head: {
      htmlAttrs: { lang: 'zh-CN' },
      title: '星尘知识库',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        {
          name: 'description',
          content: '个人 LLM 学习知识库：模型架构、推理优化、工程实践与博客随笔',
        },
      ],
      script: [
        {
          key: 'mathjax-loader',
          // 本地内置 MathJax（public/mathjax），避免 CDN 不稳定；typeset:false 关闭自动排版，
          // 公式统一由 ArticleView 在 v-html 挂载/更新后手动 typeset，杜绝与 Vue hydration 竞争
          innerHTML:
            'window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]]},chtml:{fontURL:"/mathjax/fonts/woff-v2"},options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]},startup:{typeset:false}};(function(){var s=document.createElement("script");s.src="/mathjax/tex-mml-chtml.js";s.async=true;document.head.appendChild(s);})();',
        },
      ],
    },
  },
})
