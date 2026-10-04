// Markdown pages: pymdownx.arithmatex wraps math in .arithmatex elements as \( \) or \[ \].
// Tutorial pages: mkdocs-jupyter leaves $ and $$ math inside .jp-RenderedMarkdown cells.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    // Skip every element that has a class unless it is one of the two containers above.
    // Elements without a class inherit their parent's setting, so the paragraphs and list
    // items inside a notebook cell are typeset, while dollar signs elsewhere are left alone.
    ignoreHtmlClass: ".+",
    processHtmlClass: "arithmatex|jp-RenderedMarkdown"
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
