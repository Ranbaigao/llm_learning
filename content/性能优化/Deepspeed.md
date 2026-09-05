<!-- <iframe src="../../.assets/Deepspeed_zero2.html" width="100%" height="1200px" style="border:none;"></iframe> -->
# Deepspeed

--8<-- ".assets/Deepspeed_zero2.html"


## 校对
[1] 有关ZeRO的描述有误

## 参考文献

[1] 知乎专栏. 《从啥也不会到DeepSpeed————一篇大模型分布式训练的学习过程总结》. 见于 2026年6月19日. https://zhuanlan.zhihu.com/p/688873027.

[2] 知乎专栏. 《手把手推导Ring All-reduce的数学性质》. 见于 2026年6月19日. https://zhuanlan.zhihu.com/p/504957661.

[3] Rajbhandari, Samyam, Jeff Rasley, Olatunji Ruwase和Yuxiong He. 《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》. arXiv:1910.02054. 预印本, arXiv, 2020年5月13日. https://doi.org/10.48550/arXiv.1910.02054.



<script>
  function setupAutoHeightIframes() {
    document.querySelectorAll(".auto-height-frame").forEach((iframe) => {
      if (iframe.dataset.autoHeightReady === "true") {
        return;
      }

      let observer;
      iframe.dataset.autoHeightReady = "true";

      function iframeDocument() {
        try {
          return iframe.contentDocument || iframe.contentWindow.document;
        } catch (error) {
          return null;
        }
      }

      function resizeIframe() {
        const doc = iframeDocument();
        if (!doc) {
          return;
        }

        iframe.style.height =
          Math.max(
            doc.body ? doc.body.scrollHeight : 0,
            doc.documentElement ? doc.documentElement.scrollHeight : 0
          ) + "px";
      }

      function observeIframe() {
        const doc = iframeDocument();
        if (!doc) {
          return;
        }

        resizeIframe();

        if (observer || !doc.documentElement) {
          return;
        }

        observer = new ResizeObserver(resizeIframe);
        observer.observe(doc.documentElement);

        if (doc.body) {
          observer.observe(doc.body);
        }
      }

      iframe.addEventListener("load", observeIframe);
      window.addEventListener("resize", resizeIframe);
      observeIframe();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupAutoHeightIframes);
  } else {
    setupAutoHeightIframes();
  }
</script>