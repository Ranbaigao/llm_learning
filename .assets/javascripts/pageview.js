/* 浏览量统计：页脚版权行追加「本文阅读量 · 总浏览量 · 访客数」。
 * 必须在统计服务脚本（vercount）之前同步执行，确保 span 元素已存在。
 * 使用不蒜子兼容标记，切换统计服务时无需改动本文件。 */
(function () {
  var copyright = document.querySelector(".md-copyright");
  if (!copyright) return;
  var site = document.createElement("div");
  site.className = "pageview-site";
  site.innerHTML =
    '<span id="busuanzi_container_page_pv">本文阅读量：<span id="busuanzi_value_page_pv">-</span> 次</span>' +
    '<span class="pageview-sep">·</span>' +
    '<span id="busuanzi_container_site_pv">总浏览量：<span id="busuanzi_value_site_pv">-</span> 次</span>' +
    '<span class="pageview-sep">·</span>' +
    '<span id="busuanzi_container_site_uv">访客数：<span id="busuanzi_value_site_uv">-</span> 人</span>';
  copyright.appendChild(site);
})();
