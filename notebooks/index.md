---
hide:
  - navigation
  - toc
comments: false
---

<style>
  .md-main__inner {
    max-width: none;
    margin: 0;
  }

  .md-content {
    max-width: none;
  }

  .md-content__inner {
    max-width: none;
    margin: 0;
    padding: 0;
  }

  .md-content__inner::before {
    display: none;
  }

  /* 首页只有星图：隐藏页脚与 Material 自动注入的 "Home" 占位标题，保证满屏无滚动 */
  .md-footer {
    display: none;
  }

  .md-content__inner > h1:first-of-type {
    display: none;
  }

  /* iframe 会被 Markdown 包进 <p>，去掉段落自带的上下外边距，保证星图满屏无滚动 */
  .md-content__inner > p {
    margin: 0;
  }

  .knowledge-graph-frame {
    width: 100%;
    height: calc(100vh - 48px);
    /* 移动端浏览器地址栏会压缩可视高度，dvh 动态跟随，避免星图底部组件被裁掉 */
    height: calc(100dvh - 48px);
    min-height: 480px;
    border: none;
    border-radius: 0;
    box-shadow: none;
    margin: 0;
    display: block;
    background: #030712;
  }

  /* 隐藏 MkDocs 右上角自带搜索 */
  .md-search, .md-header__source {
      display: none !important;
  }
</style>

<iframe
  class="knowledge-graph-frame"
  src=".assets/knowledge_graph.html"
  loading="lazy"
  title="LLM 笔记知识图谱"
></iframe>
