/* Nados Preview Enhancer — in-app preview for every code artifact.
   1) Auto-enables the built-in code preview (html/svg/css blocks) on appear.
   2) Adds an in-app preview button on any artifact-looking code block.
   3) Full-screen in-app modal (sandboxed iframe) instead of external windows. */
(function () {
  var PREVIEWABLE = new Set(["html", "htm", "svg", "css"]);

  var CSS = ""
    + ".nados-inapp-preview{background:none;border:none;cursor:pointer;font-size:14px;line-height:1;padding:4px;border-radius:6px;}"
    + ".nados-inapp-preview:hover{background:rgba(0,0,0,.08)}"
    + "#nados-pv-overlay{position:fixed;inset:0;z-index:99999;background:rgba(20,22,18,.62);display:flex;align-items:center;justify-content:center;backdrop-filter:blur(3px)}"
    + "#nados-pv-panel{width:min(1100px,94vw);height:min(760px,90vh);background:#f4f5f2;border-radius:14px;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 24px 70px rgba(0,0,0,.35);border:1px solid #e3e5de}"
    + "#nados-pv-head{display:flex;align-items:center;gap:10px;padding:10px 14px;background:#fbfbf9;border-bottom:1px solid #e3e5de;font-family:Cairo,system-ui,sans-serif}"
    + "#nados-pv-title{font-weight:700;color:#2a2c26;font-size:14px;flex:1}"
    + "#nados-pv-head button{font-family:Cairo,inherit;border:1px solid #dcded6;background:#fff;border-radius:8px;padding:6px 12px;cursor:pointer;font-size:13px;color:#2a2c26}"
    + "#nados-pv-head button:hover{background:#efefe9}"
    + "#nados-pv-frame{flex:1;border:none;width:100%;background:#fff}";

  function buildDoc(code, lang) {
    code = String(code || "");
    if (lang === "css") {
      return '<!doctype html><html dir="rtl"><head><meta charset="utf-8"><style>' + code + "</style></head><body><main><h1>Nados Preview</h1><p>معاينة مباشرة لتنسيق CSS</p><button>زر تجريبي</button></main></body></html>";
    }
    if (/<html[\s>]/i.test(code)) return code;
    return '<!doctype html><html dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>' + code + "</body></html>";
  }

  var overlay = null;
  function closePreview() {
    if (overlay) { overlay.remove(); overlay = null; }
  }
  function openPreview(code, lang) {
    closePreview();
    overlay = document.createElement("div");
    overlay.id = "nados-pv-overlay";
    var doc = buildDoc(code, lang);
    overlay.innerHTML = ""
      + '<div id="nados-pv-panel">'
      + '<div id="nados-pv-head"><span id="nados-pv-title">معاينة داخل التطبيق · ' + (lang || "artifact") + "</span>"
      + '<button data-act="tab">فتح في تبويب</button>'
      + '<button data-act="close">إغلاق</button></div>'
      + '<iframe id="nados-pv-frame" sandbox="allow-scripts allow-same-origin allow-popups"></iframe></div>';
    document.body.appendChild(overlay);
    var frame = overlay.querySelector("#nados-pv-frame");
    frame.srcdoc = doc;
    overlay.querySelector('[data-act="close"]').onclick = closePreview;
    overlay.querySelector('[data-act="tab"]').onclick = function () {
      var blob = new Blob([doc], { type: "text/html;charset=utf-8" });
      var url = URL.createObjectURL(blob);
      var w = window.open(url, "_blank", "noopener,noreferrer");
      if (w) w.opener = null;
      setTimeout(function () { URL.revokeObjectURL(url); }, 30000);
    };
    overlay.addEventListener("click", function (e) { if (e.target === overlay) closePreview(); });
    document.addEventListener("keydown", function esc(e) {
      if (e.key === "Escape") { closePreview(); document.removeEventListener("keydown", esc); }
    });
  }

  function enhance(block) {
    if (block.dataset.nadosEnh) return;
    block.dataset.nadosEnh = "1";
    var langEl = block.querySelector(".code-block__lang");
    var lang = (langEl ? langEl.textContent : "").trim().toLowerCase();
    var codeEl = block.querySelector("pre code");
    var code = codeEl ? codeEl.textContent : "";
    var looksArtifact = /<!doctype html|<html[\s>]|<div[\s>]|<svg[\s>]|<section[\s>]|<canvas[\s>]|<style[\s>]/i.test(code);

    // 1) auto-enable the built-in preview (React toggle) once per block
    if (PREVIEWABLE.has(lang) && !block.querySelector("iframe.code-preview")) {
      var toggle = Array.prototype.find.call(block.querySelectorAll("button"), function (b) {
        var l = b.getAttribute("aria-label") || "";
        return l.indexOf("معاينة") !== -1;
      });
      if (toggle) { try { toggle.click(); } catch (e) {} }
    }

    // 2) in-app preview button for artifact-looking code
    if ((looksArtifact || PREVIEWABLE.has(lang)) && code.trim()) {
      var tools = block.querySelector(".code-block__tools");
      if (tools && !tools.querySelector(".nados-inapp-preview")) {
        var btn = document.createElement("button");
        btn.className = "nados-inapp-preview";
        btn.type = "button";
        btn.title = "معاينة داخل التطبيق";
        btn.setAttribute("aria-label", "معاينة داخل التطبيق");
        btn.textContent = "👁";
        btn.addEventListener("click", function (e) {
          e.preventDefault();
          e.stopPropagation();
          openPreview(code, lang);
        });
        tools.insertBefore(btn, tools.firstChild);
      }
    }
  }

  function scan() {
    var blocks = document.querySelectorAll(".code-block");
    for (var i = 0; i < blocks.length; i++) enhance(blocks[i]);
  }

  var style = document.createElement("style");
  style.textContent = CSS;
  document.head.appendChild(style);

  var obs = new MutationObserver(function () { scan(); });
  function start() {
    scan();
    obs.observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
