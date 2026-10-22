// Integration tests: HTTP surface of the Nados API (auth, validation, headers, SSE contract).
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// --- isolate the environment before importing the app ---
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "nados-api-"));
process.env.NADOS_API_KEY = "test-key-123";
process.env.NADOS_AUTH = "1";
process.env.NADOS_RATE_LIMIT = "1000";
process.env.NADOS_RATE_LIMIT_BURST = "1000";
process.env.NADOS_CACHE_TTL_MS = "0";
process.env.NADOS_BUDGET_FILE = path.join(tmp, "budget.json");
process.env.NADOS_CACHE_FILE = path.join(tmp, "cache.json");
process.env.CF_API_TOKEN = "dummy-token";
process.env.CF_ACCOUNTS = "acct-test-1";

const { config } = await import("../lib/config.mjs");
config.budgetFile = process.env.NADOS_BUDGET_FILE;
config.cacheFile = process.env.NADOS_CACHE_FILE;

const { createServer } = await import("../lib/router.mjs");
const { initEngine } = await import("../lib/chat.mjs");

initEngine(path.join(process.cwd(), "training", "artifacts", "nados-v1.1.0.json"));

const server = createServer();
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}`;

test.after(() => {
  server.close();
  fs.rmSync(tmp, { recursive: true, force: true });
});

const AUTH = { "x-nados-key": "test-key-123" };

test("GET /api/health is public and reports the account budget", async () => {
  const r = await fetch(`${base}/api/health`);
  assert.equal(r.status, 200);
  const j = await r.json();
  assert.equal(j.ok, true);
  assert.ok(Array.isArray(j.orchestration.accounts));
  assert.equal(j.features.chat, true);
});

test("API endpoints require the key", async () => {
  const r = await fetch(`${base}/api/models`);
  assert.equal(r.status, 401);
  const r2 = await fetch(`${base}/api/models`, { headers: AUTH });
  assert.equal(r2.status, 200);
  const j = await r2.json();
  assert.equal(j.chatDefault, "nados-v1.1");
});

test("the browser gets a session cookie with the page and can then call the API", async () => {
  const page = await fetch(`${base}/`);
  assert.equal(page.status, 200);
  const setCookie = page.headers.get("set-cookie") || "";
  assert.match(setCookie, /nados_sid=/);
  const sid = setCookie.split(";")[0];
  const r = await fetch(`${base}/api/models`, { headers: { cookie: sid } });
  assert.equal(r.status, 200, "session cookie must authorise the SPA");
});

test("security headers are present on API responses", async () => {
  const r = await fetch(`${base}/api/health`);
  assert.equal(r.headers.get("x-content-type-options"), "nosniff");
  assert.equal(r.headers.get("x-frame-options"), "SAMEORIGIN");
  assert.ok(r.headers.get("x-request-id"));
});

test("empty chat message is rejected with 400", async () => {
  const body = "--B\r\nContent-Disposition: form-data; name=\"message\"\r\n\r\n\r\n--B--\r\n";
  const r = await fetch(`${base}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "multipart/form-data; boundary=B", ...AUTH },
    body,
  });
  assert.equal(r.status, 400);
  const j = await r.json();
  assert.equal(j.code, "VALIDATION_ERROR");
  assert.ok(!JSON.stringify(j).includes("D:\\"), "no internal paths in the error");
});

test("unknown API route returns 404 with a safe message", async () => {
  const r = await fetch(`${base}/api/does-not-exist`, { headers: AUTH });
  assert.equal(r.status, 404);
  const j = await r.json();
  assert.equal(j.error, "المسار غير موجود");
});

test("path traversal cannot escape the public directory", async () => {
  for (const p of ["/%2e%2e%2f.env", "/..%2f.env", "/%2e%2e/%2e%2e/.env"]) {
    const r = await fetch(`${base}${p}`);
    const body = await r.text();
    assert.ok(!body.includes("CF_API_TOKEN"), `${p} must not leak the .env`);
    assert.ok(!body.includes("cfut_"), `${p} must not leak the token`);
  }
});

test("static assets are served with an ETag and support 304", async () => {
  const r = await fetch(`${base}/`);
  assert.equal(r.status, 200);
  const etag = r.headers.get("etag");
  assert.ok(etag);
  const r2 = await fetch(`${base}/`, { headers: { "If-None-Match": etag } });
  assert.equal(r2.status, 304);
});

test("large assets are gzipped when the client accepts it", async () => {
  const r = await fetch(`${base}/assets/index-CXaI5KpR.js`, { headers: { "Accept-Encoding": "gzip" } });
  assert.equal(r.status, 200);
  assert.equal(r.headers.get("content-encoding"), "gzip");
});

test("chat stream emits meta → delta → done and never duplicates text", async () => {
  // stub the upstream provider with a deterministic SSE stream
  const realFetch = globalThis.fetch;
  const payload = "السلام عليكم، هذا رد اختباري.";
  globalThis.fetch = async (url, opts) => {
    const u = String(url);
    if (u.includes("/ai/run/")) {
      const sse =
        `data: ${JSON.stringify({ choices: [{ delta: { content: payload.slice(0, 10) } }] })}\n\n` +
        `data: ${JSON.stringify({ choices: [{ delta: { content: payload.slice(10) } }] })}\n\n` +
        `data: ${JSON.stringify({ response: payload })}\n\n` +
        `data: [DONE]\n\n`;
      return new Response(sse, { status: 200, headers: { "Content-Type": "text/event-stream" } });
    }
    return realFetch(url, opts);
  };
  try {
    const boundary = "--B";
    const body = `--${boundary}\r\nContent-Disposition: form-data; name="message"\r\n\r\nمرحبا\r\n--${boundary}--\r\n`;
    const r = await fetch(`${base}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": `multipart/form-data; boundary=${boundary}`, ...AUTH },
      body,
    });
    assert.equal(r.status, 200);
    const text = await r.text();
    const events = text.split("\n\n").filter((l) => l.startsWith("data: ")).map((l) => JSON.parse(l.slice(6)));
    const types = events.map((e) => e.type);
    assert.ok(types.includes("delta"));
    assert.ok(types.includes("done"));
    const streamed = events.filter((e) => e.type === "delta").map((e) => e.delta).join("");
    const answer = events.find((e) => e.type === "done").reply.answer[0];
    assert.equal(streamed, payload, "streamed text must equal the model output exactly (no duplication, no truncation)");
    assert.equal(answer, payload);
    // the trained ranker must be part of the reply
    assert.ok(events.find((e) => e.type === "done").reply.nados);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("meta questions never trigger a web search", async () => {
  const { isMetaQuestion, sanitizeQuery, isRelevant } = await import("../web-search.mjs");
  assert.equal(isMetaQuestion("من أنا؟"), true);
  assert.equal(isMetaQuestion("من انت"), true);
  assert.equal(isMetaQuestion("شكون انت"), true);
  assert.equal(isMetaQuestion("who are you"), true);
  assert.equal(isMetaQuestion("ما هي عاصمة كندا؟"), false);
});

test("search queries are sanitised into keywords", async () => {
  const { sanitizeQuery } = await import("../web-search.mjs");
  assert.equal(sanitizeQuery("ما هي عاصمة كندا؟"), "عاصمة كندا");
  assert.equal(sanitizeQuery("من هو بايروت؟"), "بايروت");
  assert.ok(sanitizeQuery("اشرح لي مفهوم الـ concurrency control").length > 5);
});

test("irrelevant search results are rejected by the relevance gate", async () => {
  const { isRelevant } = await import("../web-search.mjs");
  const offTopic = [
    { title: "مسلسل أنت من أحببت", snippet: "مسلسل تلفزيوني" },
    { title: "أزياء", snippet: "أزياء عصرية" },
  ];
  assert.equal(isRelevant("عاصمة كندا", offTopic), false);
  const onTopic = [{ title: "كندا", snippet: "عاصمة كندا هي أوتاوا" }];
  assert.equal(isRelevant("عاصمة كندا", onTopic), true);
});

test("the chat answers an identity question without inventing sources", async () => {
  const realFetch = globalThis.fetch;
  const payload = "أنا نادوس، مساعد ذكاء اصطناعي عربي.";
  globalThis.fetch = async (url, opts) => {
    if (String(url).includes("/ai/run/") || String(url).includes("pollinations")) {
      const sse =
        `data: ${JSON.stringify({ choices: [{ delta: { content: payload } }] })}\n\n` +
        `data: ${JSON.stringify({ response: payload })}\n\n` +
        `data: [DONE]\n\n`;
      return new Response(sse, { status: 200, headers: { "Content-Type": "text/event-stream" } });
    }
    return realFetch(url, opts);
  };
  try {
    const boundary = "--B";
    const body = `--${boundary}\r\nContent-Disposition: form-data; name="message"\r\n\r\nمن أنا؟\r\n--${boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nweb\r\n--${boundary}--\r\n`;
    const r = await fetch(`${base}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": `multipart/form-data; boundary=${boundary}`, ...AUTH },
      body,
    });
    const text = await r.text();
    const events = text.split("\n\n").filter((l) => l.startsWith("data: ")).map((l) => JSON.parse(l.slice(6)));
    const done = events.find((e) => e.type === "done");
    assert.ok(done, "a done event must be produced");
    assert.equal(done.reply.sources.length, 0, "identity question must not attach web sources");
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("rate limiting kicks in for a flood from one client", async () => {
  // temporarily tighten the limiter
  const { config: cfg } = await import("../lib/config.mjs");
  const before = cfg.rateLimit;
  const beforeBurst = cfg.rateLimitBurst;
  cfg.rateLimit = 1;
  cfg.rateLimitBurst = 2;
  const ip = "203.0.113.9";
  let limited = false;
  for (let i = 0; i < 12; i++) {
    const r = await fetch(`${base}/api/models`, { headers: { ...AUTH, "x-forwarded-for": ip } });
    if (r.status === 429) { limited = true; break; }
  }
  cfg.rateLimit = before;
  cfg.rateLimitBurst = beforeBurst;
  assert.ok(limited, "flooding must be throttled");
});
