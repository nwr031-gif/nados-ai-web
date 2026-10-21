// Security layer: authentication, rate limiting, security headers, safe errors.
import crypto from "node:crypto";
import { config } from "./config.mjs";

// ---------------- authentication ----------------
// The SPA is a pre-built bundle that cannot send custom headers, so the browser is
// authorised through an httpOnly session cookie issued with the page. CLI tools use
// the `x-nados-key` header with NADOS_API_KEY.
let sessionToken = config.apiKey || crypto.randomBytes(32).toString("hex");
export const apiKey = () => sessionToken;

export function setSessionToken(t) {
  if (t) sessionToken = t;
}
export const hasApiKeyConfigured = () => Boolean(config.apiKey);

function parseCookies(header) {
  const out = {};
  for (const part of String(header || "").split(";")) {
    const i = part.indexOf("=");
    if (i < 0) continue;
    out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

export function isAuthorised(req) {
  if (!config.authEnabled) return true;
  const header = req.headers["x-nados-key"];
  if (header && crypto.timingSafeEqual(Buffer.from(String(header)), Buffer.from(sessionToken))) return true;
  const cookies = parseCookies(req.headers.cookie);
  if (cookies.nados_sid && crypto.timingSafeEqual(Buffer.from(cookies.nados_sid), Buffer.from(sessionToken))) return true;
  return false;
}

export function sessionCookie() {
  return `nados_sid=${encodeURIComponent(sessionToken)}; Path=/; HttpOnly; SameSite=Strict; Max-Age=86400`;
}

// ---------------- rate limiting (token bucket per client) ----------------
const buckets = new Map();

export function rateLimit(req, { limit = config.rateLimit, burst = config.rateLimitBurst } = {}) {
  const ip = req.socket?.remoteAddress || "unknown";
  const now = Date.now();
  let b = buckets.get(ip);
  if (!b) {
    b = { tokens: burst, last: now };
    buckets.set(ip, b);
  }
  const refill = ((now - b.last) / 60000) * limit;
  b.tokens = Math.min(burst, b.tokens + refill);
  b.last = now;
  if (b.tokens < 1) {
    const wait = Math.ceil(((1 - b.tokens) / limit) * 60000);
    return { allowed: false, retryAfter: Math.max(1, wait) };
  }
  b.tokens -= 1;
  return { allowed: true, remaining: Math.floor(b.tokens) };
}

// housekeeping so the map cannot grow unbounded
setInterval(() => {
  const now = Date.now();
  for (const [ip, b] of buckets) if (now - b.last > 10 * 60 * 1000) buckets.delete(ip);
}, 5 * 60 * 1000).unref?.();

// ---------------- concurrency guard ----------------
let inFlight = 0;
export function acquireSlot() {
  if (inFlight >= config.maxConcurrent) return false;
  inFlight++;
  return true;
}
export function releaseSlot() {
  if (inFlight > 0) inFlight--;
}
export const inFlightCount = () => inFlight;

// ---------------- headers ----------------
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
  "font-src 'self' https://fonts.gstatic.com data:",
  "img-src 'self' data: blob: https:",
  "connect-src 'self'",
  "frame-src 'self' blob:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'self'",
].join("; ");

export function applySecurityHeaders(res, { html = false } = {}) {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("X-Frame-Options", "SAMEORIGIN");
  res.setHeader("Referrer-Policy", "strict-origin-when-cross-origin");
  res.setHeader("Permissions-Policy", "geolocation=(), microphone=(), camera=()");
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  if (html) res.setHeader("Content-Security-Policy", CSP);
}

// ---------------- safe errors ----------------
export function publicError(err, requestId) {
  // Never leak internals (paths, provider ids, upstream payloads) to the client.
  const code = err?.code || "INTERNAL_ERROR";
  const map = {
    VALIDATION_ERROR: "طلب غير صالح",
    AUTH_REQUIRED: "مطلوب مفتاح وصول صالح",
    RATE_LIMITED: "عدد الطلبات تجاوز الحد المسموح",
    NO_CAPACITY: "لا تتوفر حالياً سعة توليد كافية، حاول بعد قليل",
    NOT_FOUND: "المسار غير موجود",
    FORBIDDEN: "ممنوع",
  };
  return { error: map[code] || "حدث خطأ داخلي", requestId, code };
}

export function newRequestId() {
  return crypto.randomBytes(8).toString("hex");
}
