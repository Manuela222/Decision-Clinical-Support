// Fullscreen "focus mode" for a single method's card, with prev/next
// navigation across the other cards currently on the page -- no page
// navigation, no reload, just an overlay driven by a static item list.
const FocusModal = (() => {
  let items = [];
  let index = 0;
  let overlay, titleEl, bodyEl, prevBtn, nextBtn, closeBtn;

  function init() {
    overlay = document.getElementById("focus-overlay");
    if (!overlay) return;
    titleEl = document.getElementById("focus-title");
    bodyEl = document.getElementById("focus-body");
    prevBtn = document.getElementById("focus-prev");
    nextBtn = document.getElementById("focus-next");
    closeBtn = document.getElementById("focus-close");

    prevBtn.addEventListener("click", () => show(index - 1));
    nextBtn.addEventListener("click", () => show(index + 1));
    closeBtn.addEventListener("click", close);
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) close();
    });
    document.addEventListener("keydown", (e) => {
      if (overlay.classList.contains("hidden")) return;
      if (e.key === "Escape") close();
      if (e.key === "ArrowLeft") show(index - 1);
      if (e.key === "ArrowRight") show(index + 1);
    });
  }

  function show(i) {
    if (!items.length) return;
    index = (i + items.length) % items.length;
    titleEl.textContent = items[index].title;
    bodyEl.innerHTML = items[index].bodyHtml;
    const multi = items.length > 1;
    prevBtn.style.visibility = multi ? "visible" : "hidden";
    nextBtn.style.visibility = multi ? "visible" : "hidden";
  }

  function open(newItems, startIndex) {
    if (!overlay || !newItems.length) return;
    items = newItems;
    overlay.classList.remove("hidden");
    document.body.style.overflow = "hidden";
    show(startIndex || 0);
  }

  function close() {
    if (!overlay) return;
    overlay.classList.add("hidden");
    document.body.style.overflow = "";
  }

  document.addEventListener("DOMContentLoaded", init);
  return { open, close };
})();
