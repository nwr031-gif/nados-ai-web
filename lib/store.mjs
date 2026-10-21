// Persistent state: Cloudflare neuron budget + answer cache.
// Writes are atomic (temp file + rename) so a crash can never corrupt the state.
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { config } from "./config.mjs";

function writeAtomic(file, data) {
  const tmp = `${file}.${process.pid}.tmp`;
  try {
    fs.writeFileSync(tmp, data, "utf8");
    fs.renameSync(tmp, file);
  } catch {
    try { fs.unlinkSync(tmp); } catch {}
  }
}

const today = () => new Date().toISOString().slice(0, 10);

// ---------------- neuron budget ----------------
let budgetState = readBudget();

function readBudget() {
  try {
    const b = JSON.parse(fs.readFileSync(config.budgetFile, "utf8"));
    if (b.date === today() && b.accounts && typeof b.accounts === "object") return b.accounts;
  } catch {}
  return {};
}

function saveBudget() {
  writeAtomic(config.budgetFile, JSON.stringify({ date: today(), accounts: budgetState }));
}

const accState = (id) =>
  budgetState[id] && typeof budgetState[id] === "object" ? budgetState[id] : { used: 0, exhausted: false };

export function setAccount(id, patch) {
  budgetState[id] = { ...accState(id), ...patch };
  saveBudget();
}

export const neuronsUsed = (id) => accState(id).used;
export const isExhausted = (id) => accState(id).exhausted === true;
export const neuronsRemaining = (id) => (isExhausted(id) ? 0 : Math.max(0, config.dailyNeurons - neuronsUsed(id)));

export function addNeurons(accountId, n) {
  if (!n || !isFinite(n) || n <= 0) return;
  const st = accState(accountId);
  const used = Math.min(config.dailyNeurons, st.used + n);
  setAccount(accountId, { used, exhausted: st.exhausted || used >= config.dailyNeurons });
}

/** Cloudflare bills neurons as (active params × tokens) / ~1.5e9 — calibrated against real reports. */
export const estimateNeurons = (tokens, activeParams) => (tokens * activeParams) / 1500;

export function markExhaustedAll(accounts) {
  for (const a of accounts) setAccount(a, { exhausted: true });
}

export function budgetSnapshot(accounts) {
  return accounts.map((a) => ({
    account: a.slice(0, 12),
    used: Math.round(neuronsUsed(a)),
    remaining: Math.round(neuronsRemaining(a)),
    daily: config.dailyNeurons,
    exhausted: isExhausted(a),
  }));
}

// ---------------- answer cache ----------------
let cacheState = readCache();

function readCache() {
  try {
    const c = JSON.parse(fs.readFileSync(config.cacheFile, "utf8"));
    if (c.date === today() && c.items && typeof c.items === "object") return c.items;
  } catch {}
  return {};
}

function saveCache() {
  writeAtomic(config.cacheFile, JSON.stringify({ date: today(), items: cacheState }));
}

export function cacheKey(tag, payload) {
  return crypto.createHash("sha1").update(tag + "|" + JSON.stringify(payload)).digest("hex");
}

export function cacheGet(key) {
  const e = cacheState[key];
  if (!e) return null;
  if (config.cacheTtlMs > 0 && Date.now() - (e.at || 0) > config.cacheTtlMs) {
    delete cacheState[key];
    saveCache();
    return null;
  }
  const text = e.text ?? e.value;
  if (!text) return null;
  return { text, teacher: e.teacher || "cloud", sources: e.sources || [] };
}

export function cacheSet(key, value) {
  cacheState[key] = { text: value.text, teacher: value.teacher, sources: value.sources || [], at: Date.now() };
  const keys = Object.keys(cacheState);
  if (keys.length > 120) for (const old of keys.slice(0, keys.length - 120)) delete cacheState[old];
  saveCache();
}

export const cacheSize = () => Object.keys(cacheState).length;
export const cacheClear = () => { cacheState = {}; saveCache(); };
