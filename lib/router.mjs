// HTTP router: static SPA + JSON API. Applies auth, rate limiting, security headers.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { config, PUBLIC_DIR } from "./config.mjs";
import { log } from "./log.mjs";
import { isAuthorised, sessionCookie, rateLimit, applySecurityHeaders, publicError, newRequestId, acquireSlot, releaseSlot } from "./security.mjs";
import { handleChatStream, initEngine, engine } from "./chat.mjs";
import { BACKENDS, CF_PRIMARY, callTeacher, isQuotaError } from "./providers.mjs";
import { budgetSnapshot, neuronsUsed, cacheSize } from "./store.mjs";
import { webSearch } from "../web-search.mjs";
import { runTrainingCycle, trainingSnapshot } from "./training.mjs";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".json": "application/json; charset=utf-8",
  ".webmanifest": "application/manifest+json",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
};

function sendJson(res, status, obj, extraHeaders = {}) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    ...extraHeaders,
  });
  res.end(body);
}

/** Resolves a request path inside PUBLIC safely (no prefix-match bypass). */
function resolveStatic(pathname) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return null;
  }
  if (decoded.includes("\0")) return null;
  const target = path.resolve(PUBLIC_DIR, "." + path.posix.normalize(decoded));
  const rel = path.relative(PUBLIC_DIR, target);
  if (rel.startsWith("..") || path.isAbsolute(rel)) return null;
  return target;
}

function serveStatic(req, res, pathname, requestId) {
  let file = resolveStatic(pathname);
  if (!file) {
    sendJson(res, 403, { error: "ممنوع", requestId }, { "X-Request-Id": requestId });
    return;
  }
  if (pathname === "/" || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    file = path.join(PUBLIC_DIR, "index.html");
  }
  let stat;
  try {
    stat = fs.statSync(file);
  } catch {
    sendJson(res, 404, { error: "غير موجود", requestId }, { "X-Request-Id": requestId });
    return;
  }
  const ext = path.extname(file).toLowerCase();
  const isHtml = ext === ".html";
  const etag = `W/"${stat.size}-${Math.floor(stat.mtimeMs)}"`;
  if (req.headers["if-none-match"] === etag) {
    res.writeHead(304, { ETag: etag });
    return res.end();
  }
  const headers = {
    "Content-Type": MIME[ext] || "application/octet-stream",
    "Cache-Control": isHtml ? "no-cache" : "public, max-age=3600",
    ETag: etag,
  };
  const accept = String(req.headers["accept-encoding"] || "");
  const compressible = ["html", "js", "css", "json", "svg", "webmanifest"].includes(ext.slice(1));
  if (compressible && /\bgzip\b/.test(accept) && stat.size > 1024) {
    headers["Content-Encoding"] = "gzip";
    headers["Vary"] = "Accept-Encoding";
    res.writeHead(200, headers);
    fs.createReadStream(file).pipe(zlib.createGzip({ level: 6 })).pipe(res);
  } else {
    res.writeHead(200, headers);
    fs.createReadStream(file).pipe(res);
  }
}

async function readBody(req, limitBytes = 2 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const c of req) {
    size += c.length;
    if (size > limitBytes) throw Object.assign(new Error("payload too large"), { code: "VALIDATION_ERROR" });
    chunks.push(c);
  }
  return Buffer.concat(chunks);
}

async function parseForm(req) {
  const req2 = new Request("http://local/api/chat/stream", {
    method: "POST",
    headers: req.headers,
    body: req,
    duplex: "half",
  });
  return req2.formData();
}

function requireAuth(req, res, requestId) {
  if (isAuthorised(req)) return true;
  applySecurityHeaders(res, { html: false });
  sendJson(res, 401, { error: "مطلوب مفتاح وصول صالح (x-nados-key) أو جلسة متصفح", requestId }, { "X-Request-Id": requestId });
  return false;
}

function applyRateLimit(req, res, requestId, opts) {
  const r = rateLimit(req, opts);
  if (!r.allowed) {
    sendJson(res, 429, { error: "عدد الطلبات تجاوز الحد المسموح، حاول بعد قليل", requestId }, { "Retry-After": String(r.retryAfter), "X-Request-Id": requestId });
    return false;
  }
  return true;
}

export function createServer() {
  const server = http.createServer(async (req, res) => {
    const requestId = newRequestId();
    res.setHeader("X-Request-Id", requestId);
    let url;
    try {
      url = new URL(req.url, `http://${req.headers.host || "localhost"}`);
    } catch {
      sendJson(res, 400, { error: "طلب غير صالح", requestId });
      return;
    }
    const p = url.pathname;
    const isApi = p.startsWith("/api/");

    try {
      // headers first so every response (including 401/429/404) carries them
      applySecurityHeaders(res, { html: !isApi });

      // ---------- health (public: the supervisor/monitoring must reach it) ----------
      if (p === "/api/health") {
        const accounts = budgetSnapshot(config.cfAccounts);
        return sendJson(res, 200, {
          ok: true,
          configured: Boolean(config.cfToken && config.cfAccounts.length),
          provider: "cloudflare",
          model: "Cloudflare Workers AI",
          orchestration: {
            name: "cloudflare-workers-ai",
            mode: "sequential-failover",
            chain: [...new Set(BACKENDS.map((t) => t.label))],
            accounts: config.authEnabled ? accounts : accounts.map((a) => ({ ...a, account: a.account.slice(0, 4) + "…" })),
            neuronsRemaining: accounts.reduce((s, a) => s + a.remaining, 0),
            externalProviders: BACKENDS.filter((b) => b.kind === "ext").map((b) => b.label),
          },
          features: {
            chat: true, webSearch: true, files: false, vision: false, video: false,
            transcription: false, speech: false, computer: false,
          },
        });
      }

      // ---------- API: auth + rate limit ----------
      if (isApi) {
        if (!requireAuth(req, res, requestId)) return;
        if (!applyRateLimit(req, res, requestId)) return;
      }

      if (p === "/api/diagnostics" && req.method === "GET") {
        if (!applyRateLimit(req, res, requestId, { limit: 6, burst: 3 })) return;
        const checks = [];
        const time = async (name, fn) => {
          const t0 = Date.now();
          try {
            checks.push({ name, ok: true, detail: await fn(), ms: Date.now() - t0, kind: "ok" });
          } catch (e) {
            const quota = isQuotaError(e);
            checks.push({ name, ok: false, detail: String(e.message || e).slice(0, 120), ms: Date.now() - t0, kind: quota ? "quota" : "error" });
          }
        };
        await time("server", async () => `up (pid ${process.pid})`);
        await time("trained-model", async () => {
          if (!engine()) throw new Error("not loaded");
          return `Nados v1.1 loaded — ${engine().info().params} params`;
        });
        if (url.searchParams.get("deep") === "1") {
          for (const b of BACKENDS) {
            await time(`backend:${b.label}${b.accountId ? `@${b.accountId.slice(0, 8)}` : ""}`, async () => {
              const r = await callTeacher(b, [{ role: "user", content: "ping" }], 0.1, false, { max_tokens: 4 });
              if (!r.ok) throw new Error(`HTTP ${r.status}`);
              return "reachable";
            });
          }
        }
        await time("web-search", async () => {
          const s = await webSearch("الذكاء الاصطناعي", { lang: "ar", limit: 2 });
          if (!s.results.length) throw new Error(`no results (${s.errors.join("; ")})`);
          return `${s.results.length} results`;
        });
        await time("cache", async () => `${cacheSize()} cached answers`);
        await time("preview-asset", async () => {
          const f = path.join(PUBLIC_DIR, "preview-enhance.js");
          if (!fs.existsSync(f)) throw new Error("missing");
          return `preview-enhance.js ${fs.statSync(f).size} bytes`;
        });
        await time("training-engine", async () => trainingSnapshot());
        const failed = checks.filter((c) => !c.ok && c.kind !== "quota");
        const quotaBlocked = checks.filter((c) => c.kind === "quota");
        return sendJson(res, 200, {
          ok: failed.length === 0,
          at: new Date().toISOString(),
          summary: { total: checks.length, ok: checks.length - failed.length - quotaBlocked.length, quotaBlocked: quotaBlocked.length, failed: failed.length },
          neurons: budgetSnapshot(config.cfAccounts),
          checks,
        });
      }

      if (p === "/api/web/search" && req.method === "GET") {
        const q = String(url.searchParams.get("q") || "").trim().slice(0, 300);
        if (!q) {
          const e = new Error("q مطلوب");
          e.code = "VALIDATION_ERROR";
          throw e;
        }
        return sendJson(res, 200, await webSearch(q, { lang: url.searchParams.get("lang") || "ar", limit: 5 }));
      }

      if (p === "/api/models") {
        return sendJson(res, 200, {
          chatDefault: "nados-v1.1",
          models: [
            {
              id: "nados-v1.1",
              label: `Nados v1.1 · ${CF_PRIMARY.label} + مُرتّب مدرّب (${engine()?.info().params || 245} معامل)`,
              providerId: "cloudflare", provider: "Cloudflare Workers AI",
              model: [...new Set(BACKENDS.map((t) => t.label))].join(" → "),
              webSearch: true, vision: false, files: false, recommended: true, chatDefault: true,
            },
            {
              id: "nados-local",
              label: `Nados v1.1 · محلي حصري (${engine()?.info().params || 245} معامل)`,
              providerId: "local", provider: "Nados Local",
              model: "NadosRanker v1.1 (no cloud)",
              webSearch: false, vision: false, files: false, recommended: false, chatDefault: false,
            },
          ],
        });
      }

      if (p === "/api/providers" && req.method === "GET") {
        return sendJson(res, 200, { providers: BACKENDS.map((b) => ({ id: b.key, label: b.label, kind: b.kind, model: b.id || b.model, account: b.accountId ? b.accountId.slice(0, 8) : null })) });
      }

      if (p === "/api/providers/stats" && req.method === "GET") {
        return sendJson(res, 200, { stats: {} });
      }

      if (p === "/api/agent/board" && req.method === "GET") {
        return sendJson(res, 200, { board: [], running: 0 });
      }

      if (p === "/api/agent/task" && req.method === "POST") {
        await readBody(req);
        return sendJson(res, 200, { taskId: crypto.randomUUID(), status: "queued", note: "محرك الوكلاء يعمل بوضع محدود محلياً" });
      }

      if (p === "/api/training/registry" && req.method === "GET") {
        return sendJson(res, 200, {
          providers: BACKENDS.filter((b) => b.kind === "cf").map((b) => ({ id: b.label, name: b.label, model: b.id, capabilities: { contextWindow: 128000 } })),
          trainingProvider: { state: "ready", name: "Nados V1.0 Local Trainer" },
        });
      }

      if (p === "/api/training/stats" && req.method === "GET") {
        return sendJson(res, 200, trainingSnapshot(true));
      }

      if (p === "/api/training/kaggle/status" && req.method === "GET") {
        return sendJson(res, 200, { enabled: false, state: "غير مهيأ", reason: "تكامل Kaggle غير مطلوب محلياً" });
      }

      if (p === "/api/training/generate" && req.method === "POST") {
        await readBody(req);
        return sendJson(res, 200, { ok: true, generated: 0, note: "توليد بيانات التدريب متاح عبر دورة التدريب" });
      }

      if (p === "/api/training/cycle" && req.method === "POST") {
        if (!applyRateLimit(req, res, requestId, { limit: 4, burst: 2 })) return;
        await readBody(req);
        const result = await runTrainingCycle();
        return sendJson(res, 200, result);
      }

      if (p === "/api/chat/stream" && req.method === "POST") {
        if (!acquireSlot()) {
          const e = new Error("too many concurrent requests");
          e.code = "NO_CAPACITY";
          throw e;
        }
        try {
          const form = await parseForm(req);
          await handleChatStream(req, res, form, { requestId });
        } finally {
          releaseSlot();
        }
        return;
      }

      if (p === "/api/audio/transcribe" && req.method === "POST") {
        await readBody(req);
        return sendJson(res, 501, { error: "خدمة تحويل الصوت إلى نص غير مفعّلة", requestId });
      }

      if (p === "/api/audio/speech" && req.method === "POST") {
        await readBody(req);
        return sendJson(res, 501, { error: "خدمة تحويل النص إلى صوت غير مفعّلة", requestId });
      }

      if (isApi) {
        const e = new Error("not found");
        e.code = "NOT_FOUND";
        throw e;
      }

      // ---------- static SPA ----------
      if (p === "/" && isAuthorised(req) === false && config.authEnabled) {
        // issue the browser session cookie with the page
        res.setHeader("Set-Cookie", sessionCookie());
      } else if (p === "/" && config.authEnabled) {
        res.setHeader("Set-Cookie", sessionCookie());
      }
      return serveStatic(req, res, p, requestId);
    } catch (err) {
      log.error("request.failed", { requestId, path: p, reason: String(err.message || err), code: err.code || null });
      if (res.headersSent) {
        try { res.end(); } catch {}
        return;
      }
      const status = err.code === "VALIDATION_ERROR" ? 400 : err.code === "NOT_FOUND" ? 404 : err.code === "AUTH_REQUIRED" ? 401 : 500;
      sendJson(res, status, publicError(err, requestId), { "X-Request-Id": requestId });
    }
  });

  server.on("clientError", (err, socket) => {
    log.warn("http.client_error", { reason: err.message });
    if (socket.writable) socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
  });

  return server;
}
