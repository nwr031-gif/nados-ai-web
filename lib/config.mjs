// Centralised configuration — every value comes from the environment, never hard-coded.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.join(__dirname, "..");
export const PUBLIC_DIR = path.join(ROOT, "public");
export const TRAIN_DIR = path.join(ROOT, "training");

/** Loads KEY=VALUE pairs from .env without overriding real environment variables. */
export function loadEnv(file = path.join(ROOT, ".env")) {
  try {
    const raw = fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "");
    for (const line of raw.split(/\r?\n/)) {
      const m = line.match(/^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$/);
      if (!m) continue;
      const value = m[2].replace(/^"(.*)"$/, "$1").replace(/^'(.*)'$/, "$1");
      if (process.env[m[1]] === undefined) process.env[m[1]] = value;
    }
  } catch {
    // .env is optional — production may inject real environment variables
  }
}

const num = (v, d) => (v === undefined || v === "" || !isFinite(Number(v)) ? d : Number(v));
const bool = (v, d) => (v === undefined || v === "" ? d : /^(1|true|yes|on)$/i.test(String(v)));

// The environment MUST be loaded before the config object is built.
loadEnv();

export const config = {
  port: num(process.env.PORT, 4000),
  host: process.env.HOST || "127.0.0.1",
  env: process.env.NODE_ENV || "production",
  debugRaw: bool(process.env.NADOS_DEBUG_RAW, false),
  authEnabled: bool(process.env.NADOS_AUTH, true),
  apiKey: process.env.NADOS_API_KEY || "",
  rateLimit: num(process.env.NADOS_RATE_LIMIT, 30), // requests / minute / client
  rateLimitBurst: num(process.env.NADOS_RATE_LIMIT_BURST, 15),
  providerTimeout: num(process.env.NADOS_PROVIDER_TIMEOUT, 90000),
  cacheTtlMs: num(process.env.NADOS_CACHE_TTL_MS, 30 * 60 * 1000),
  maxConcurrent: num(process.env.NADOS_MAX_CONCURRENT, 8),
  dailyNeurons: num(process.env.CF_DAILY_NEURONS, 10000),
  cfToken: process.env.CF_API_TOKEN || "",
  cfAccountId: process.env.CF_ACCOUNT_ID || "",
  cfModel: process.env.CF_AI_MODEL || "@cf/meta/llama-3.1-8b-instruct",
  cfAccounts: (process.env.CF_ACCOUNTS || "")
    .split(",").map((s) => s.trim()).filter(Boolean),
  budgetFile: path.join(ROOT, ".nados-budget.json"),
  cacheFile: path.join(ROOT, ".nados-cache.json"),
};

export const isProduction = config.env === "production";
