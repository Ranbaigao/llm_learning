/* HTML 弹窗查看器
 *
 * 用法：在 Markdown 中写入
 *   <a class="html-modal-link" href="相对路径/xxx.html">弹窗查看：xxx</a>
 * 点击后在弹窗中以 iframe 懒加载该 HTML；
 * 若 JS 未加载/失效，则退化为普通链接，在新标签页打开。
 */
(function () {
  var overlay = null;
  var frame = null;
  var lastTrigger = null;

  function buildOverlay() {
    overlay = document.createElement("div");
    overlay.className = "html-modal-overlay";
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.innerHTML =
      '<div class="html-modal-dialog">' +
      '  <div class="html-modal-bar">' +
      '    <span class="html-modal-title"></span>' +
      '    <span class="html-modal-actions">' +
      '      <a class="html-modal-open-external" href="#" target="_blank" rel="noopener" title="在新标签页打开">↗</a>' +
      '      <button class="html-modal-close" type="button" title="关闭 (Esc)">✕</button>' +
      "    </span>" +
      "  </div>" +
      '  <iframe class="html-modal-frame" title="HTML 弹窗内容" loading="lazy"></iframe>' +
      "</div>";
    document.body.appendChild(overlay);

    frame = overlay.querySelector(".html-modal-frame");

    overlay.addEventListener("click", function (e) {
      if (e.target === overlay) closeModal();
    });
    overlay.querySelector(".html-modal-close").addEventListener("click", closeModal);
  }

  function openModal(href, title, trigger) {
    if (!overlay) buildOverlay();
    lastTrigger = trigger || null;
    overlay.querySelector(".html-modal-title").textContent = title || "交互演示";
    overlay.querySelector(".html-modal-open-external").href = href;
    // 弹窗是临时的：每次打开重新设置 src，关闭时清空以释放资源
    frame.src = href;
    overlay.classList.add("is-open");
    document.body.classList.add("html-modal-active");
  }

  function closeModal() {
    if (!overlay) return;
    overlay.classList.remove("is-open");
    document.body.classList.remove("html-modal-active");
    frame.src = "about:blank";
    if (lastTrigger && lastTrigger.focus) lastTrigger.focus();
  }

  // 事件委托：兼容 Material 主题的 instant navigation
  document.addEventListener("click", function (e) {
    var link = e.target.closest && e.target.closest("a.html-modal-link");
    if (!link) return;
    e.preventDefault();
    openModal(link.getAttribute("href"), link.getAttribute("data-title") || link.textContent.trim(), link);
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeModal();
  });
})();
