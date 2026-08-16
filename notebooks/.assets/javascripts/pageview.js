/* 浏览量统计：在文章标题下显示本文阅读量，在页脚显示全站总浏览量与访客数。
 * 必须在统计服务脚本（如 vercount）之前同步执行，确保 span 元素已存在。
 * 使用不蒜子兼容标记，切换统计服务时无需改动本文件。 */
(function () {
  // --- 文章级浏览量：插入到正文 h1 标题下方 ---
  // 首页为满屏知识星图 iframe，无文章标题，跳过注入
  var inner = document.querySelector(".md-content__inner");
  if (inner && !inner.querySelector(".knowledge-graph-frame")) {
    var page = document.createElement("div");
    page.className = "pageview-info";
    page.innerHTML =
      '<span id="busuanzi_container_page_pv">本文阅读量：<span id="busuanzi_value_page_pv">-</span> 次</span>';
    var h1 = inner.querySelector("h1");
    if (h1) {
      h1.parentNode.insertBefore(page, h1.nextSibling);
    } else {
      inner.insertBefore(page, inner.firstChild);
    }
  }

  // --- 全站总浏览量：追加到页脚版权行 ---
  var copyright = document.querySelector(".md-copyright");
  if (copyright) {
    var site = document.createElement("div");
    site.className = "pageview-site";
    site.innerHTML =
      '<span id="busuanzi_container_site_pv">总浏览量：<span id="busuanzi_value_site_pv">-</span> 次</span>' +
      '<span class="pageview-sep">·</span>' +
      '<span id="busuanzi_container_site_uv">访客数：<span id="busuanzi_value_site_uv">-</span> 人</span>';
    copyright.appendChild(site);
  }
})();
