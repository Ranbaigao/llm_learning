window.MathJax = {
  loader: { 
    load: ['[tex]/cancel'] 
  },
  tex: {
    packages: { '[+]': ['cancel'] }, 
    // 让 MathJax 内部只认 \( \) 和 \[ \]，把 $ $ 留给 MkDocs 的 Arithmatex 去处理
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // 忽略除 arithmatex 以外的所有 class
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex"
  }
};

document$.subscribe(() => { 
  MathJax.startup.output.clearCache()
  MathJax.typesetClear()
  MathJax.texReset()
  MathJax.typesetPromise()
})