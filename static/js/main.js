const LOG_POLL_INTERVAL_MS = 500;
const VIDEO_VIEWPORT_GAP_PX = 12;

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  const mainTab = document.getElementById("video-tab");
  const omniTab = document.getElementById("omni-tab");
  if (mainTab) mainTab.hidden = name !== "main";
  if (omniTab) omniTab.hidden = name !== "omni";
  updateVideoPanelHeightLimit();
}

function updateVideoPanelHeightLimit() {
  if (document.fullscreenElement) return;
  const header = document.getElementById("header");
  const headerH = header ? header.offsetHeight : 86;
  const availableHeight = Math.max(160, window.innerHeight - headerH - 24);
  document.documentElement.style.setProperty("--video-panel-max-height", `${availableHeight}px`);
}

function toggleFullscreen() {
  const container = document.querySelector(".video-container");
  if (!container) return;

  if (!document.fullscreenElement) {
    container
      .requestFullscreen()
      .then(() => document.body.classList.add("fullscreen-mode"))
      .catch((error) => console.warn(`fullscreen failed: ${error.message}`));
    return;
  }

  document
    .exitFullscreen()
    .then(() => document.body.classList.remove("fullscreen-mode"))
    .catch((error) => console.warn(`exit fullscreen failed: ${error.message}`));
}

function bindControls() {
  document.getElementById("fullscreenBtn")?.addEventListener("click", toggleFullscreen);
  document.getElementById("multiLineChart")?.addEventListener("change", updateCharts);
  document.getElementById("chartSelectControls")?.addEventListener("change", (event) => {
    if (event.target.matches("input[type='checkbox']")) updateCharts();
  });
  document.getElementById("applyMainRange")?.addEventListener("click", updateMainRange);

  document.querySelectorAll(".fullscreen-btn[data-omni]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = btn.dataset.omni;
      const container = document.getElementById(`omni-container-${idx}`);
      if (!container) return;
      if (!document.fullscreenElement) {
        container.requestFullscreen()
          .then(() => document.body.classList.add("fullscreen-mode"))
          .catch((e) => console.warn(`omni fullscreen failed: ${e.message}`));
      } else {
        document.exitFullscreen()
          .then(() => document.body.classList.remove("fullscreen-mode"))
          .catch((e) => console.warn(`exit fullscreen failed: ${e.message}`));
      }
    });
  });

  document.addEventListener("fullscreenchange", () => {
    document.body.classList.toggle("fullscreen-mode", Boolean(document.fullscreenElement));
    if (!document.fullscreenElement) updateVideoPanelHeightLimit();
  });

  window.addEventListener("resize", updateVideoPanelHeightLimit);
  window.addEventListener("orientationchange", updateVideoPanelHeightLimit);
}

document.addEventListener("DOMContentLoaded", () => {
  bindControls();
  updateVideoPanelHeightLimit();
  initCharts();
  startDataLoop();
  fetchAndDisplayJsonWithTree("json-log", "/log");
  setInterval(() => fetchAndDisplayJsonWithTree("json-log", "/log"), LOG_POLL_INTERVAL_MS);

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
});
