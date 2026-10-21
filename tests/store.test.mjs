// Unit tests for the persistence layer: neuron budget + answer cache.
import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

// isolate the store on a temporary file so tests never touch the real state
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "nados-store-"));
process.env.NADOS_BUDGET_FILE = path.join(tmp, "budget.json");
process.env.NADOS_CACHE_FILE = path.join(tmp, "cache.json");
process.env.NADOS_CACHE_TTL_MS = "1000";

const { config } = await import("../lib/config.mjs");
config.budgetFile = process.env.NADOS_BUDGET_FILE;
config.cacheFile = process.env.NADOS_CACHE_FILE;

const store = await import("../lib/store.mjs");

test.after(() => fs.rmSync(tmp, { recursive: true, force: true }));

test("budget starts empty and reports remaining allowance", () => {
  assert.equal(store.neuronsUsed("acct-a"), 0);
  assert.equal(store.neuronsRemaining("acct-a"), config.dailyNeurons);
  assert.equal(store.isExhausted("acct-a"), false);
});

test("neuron usage accumulates and is clamped to the daily limit", () => {
  store.addNeurons("acct-b", 100);
  store.addNeurons("acct-b", 50);
  assert.equal(store.neuronsUsed("acct-b"), 150);
  store.addNeurons("acct-b", config.dailyNeurons * 10);
  assert.equal(store.neuronsUsed("acct-b"), config.dailyNeurons, "never exceeds the daily allowance");
  assert.equal(store.isExhausted("acct-b"), true);
  assert.equal(store.neuronsRemaining("acct-b"), 0);
});

test("negative / invalid neuron values are ignored", () => {
  const before = store.neuronsUsed("acct-c");
  store.addNeurons("acct-c", -5);
  store.addNeurons("acct-c", NaN);
  store.addNeurons("acct-c", 0);
  assert.equal(store.neuronsUsed("acct-c"), before);
});

test("budget survives a restart (persisted atomically)", () => {
  store.addNeurons("acct-d", 42);
  const raw = fs.readFileSync(config.budgetFile, "utf8");
  const parsed = JSON.parse(raw); // must be valid JSON (atomic write)
  assert.equal(parsed.accounts["acct-d"].used, 42);
  assert.ok(!fs.existsSync(`${config.budgetFile}.${process.pid}.tmp`), "no temp file left behind");
});

test("cache stores and returns the answer with its sources", () => {
  const key = store.cacheKey("chat|ar", [{ role: "user", content: "مرحبا" }]);
  store.cacheSet(key, { text: "أهلاً", teacher: "nemotron", sources: [{ title: "s", url: "u" }] });
  const hit = store.cacheGet(key);
  assert.ok(hit);
  assert.equal(hit.text, "أهلاً");
  assert.equal(hit.teacher, "nemotron");
  assert.equal(hit.sources.length, 1);
});

test("cache entries expire after the TTL", async () => {
  const key = store.cacheKey("chat|ar", [{ role: "user", content: "قديم" }]);
  store.cacheSet(key, { text: "رد قديم", teacher: "x", sources: [] });
  assert.ok(store.cacheGet(key), "present before expiry");
  config.cacheTtlMs = 30;
  await new Promise((r) => setTimeout(r, 60));
  assert.equal(store.cacheGet(key), null, "expired entry must not be served");
  config.cacheTtlMs = 1000;
});

test("cache key differs per tag and payload", () => {
  const a = store.cacheKey("chat", [{ role: "user", content: "x" }]);
  const b = store.cacheKey("web", [{ role: "user", content: "x" }]);
  const c = store.cacheKey("chat", [{ role: "user", content: "y" }]);
  assert.notEqual(a, b);
  assert.notEqual(a, c);
});

test("neuron estimation follows the calibrated Cloudflare formula", () => {
  // 120B model, 1000 tokens -> (1000 * 120) / 1500 = 80 neurons
  assert.equal(Math.round(store.estimateNeurons(1000, 120)), 80);
  // 12B active model is ~10x cheaper
  assert.ok(store.estimateNeurons(1000, 12) < store.estimateNeurons(1000, 120));
});
