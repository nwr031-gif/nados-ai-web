// Unit tests for the security layer: auth, rate limiting, headers, error sanitisation.
import test from "node:test";
import assert from "node:assert/strict";
import { rateLimit, publicError, sessionCookie, isAuthorised, applySecurityHeaders, acquireSlot, releaseSlot } from "../lib/security.mjs";

function mockReq({ ip = "1.2.3.4", cookie = "", key = "" } = {}) {
  const headers = {};
  if (cookie) headers.cookie = cookie;
  if (key) headers["x-nados-key"] = key;
  return { headers, socket: { remoteAddress: ip } };
}

test("rate limiter allows the burst then throttles", () => {
  const req = mockReq({ ip: "10.0.0.1" });
  const opts = { limit: 1, burst: 3 };
  assert.equal(rateLimit(req, opts).allowed, true);
  assert.equal(rateLimit(req, opts).allowed, true);
  assert.equal(rateLimit(req, opts).allowed, true);
  const blocked = rateLimit(req, opts);
  assert.equal(blocked.allowed, false);
  assert.ok(blocked.retryAfter >= 1);
});

test("rate limiter isolates clients", () => {
  const a = mockReq({ ip: "10.0.0.2" });
  const b = mockReq({ ip: "10.0.0.3" });
  for (let i = 0; i < 3; i++) rateLimit(a, { limit: 1, burst: 3 });
  assert.equal(rateLimit(a, { limit: 1, burst: 3 }).allowed, false);
  assert.equal(rateLimit(b, { limit: 1, burst: 3 }).allowed, true);
});

test("session cookie is httpOnly + sameSite", () => {
  const c = sessionCookie();
  assert.match(c, /HttpOnly/i);
  assert.match(c, /SameSite=Strict/i);
  assert.match(c, /Path=\//);
});

test("internal error details never reach the client", () => {
  const e = publicError(new Error("ENOENT: D:\\secret\\path\\file.json"), "abc123");
  assert.equal(e.error, "حدث خطأ داخلي");
  assert.ok(!JSON.stringify(e).includes("secret"));
  assert.equal(e.requestId, "abc123");
});

test("known error codes map to Arabic messages", () => {
  assert.equal(publicError({ code: "VALIDATION_ERROR" }, "r").error, "طلب غير صالح");
  assert.equal(publicError({ code: "RATE_LIMITED" }, "r").error, "عدد الطلبات تجاوز الحد المسموح");
  assert.equal(publicError({ code: "NOT_FOUND" }, "r").error, "المسار غير موجود");
});

test("security headers are applied", () => {
  const headers = {};
  const res = { setHeader: (k, v) => { headers[k] = v; } };
  applySecurityHeaders(res, { html: true });
  assert.equal(headers["X-Content-Type-Options"], "nosniff");
  assert.equal(headers["X-Frame-Options"], "SAMEORIGIN");
  assert.ok(headers["Content-Security-Policy"].includes("default-src 'self'"));
  assert.ok(headers["Content-Security-Policy"].includes("object-src 'none'"));
});

test("concurrency slots are bounded", async () => {
  const { config } = await import("../lib/config.mjs");
  let taken = 0;
  for (let i = 0; i < config.maxConcurrent; i++) {
    assert.equal(acquireSlot(), true);
    taken++;
  }
  assert.equal(acquireSlot(), false, "extra slot must be refused");
  for (let i = 0; i < taken; i++) releaseSlot();
  assert.equal(acquireSlot(), true, "slot released again");
  releaseSlot();
});
