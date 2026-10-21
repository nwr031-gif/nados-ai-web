// Provider backends: Cloudflare Workers AI (multi-account) + OpenAI-compatible providers.
import { config } from "./config.mjs";
import { log } from "./log.mjs";
import { estimateNeurons, neuronsRemaining, isExhausted, setAccount, addNeurons } from "./store.mjs";

export const CF_MODELS = [
  { id: "@cf/openai/gpt-oss-120b", label: "gpt-oss-120B", style: "openai", active: 120 },
  { id: "@cf/nvidia/nemotron-3-120b-a12b", label: "nemotron-120B(A12B)", style: "openai", active: 12 },
  { id: "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", label: "deepseek-r1-32B", style: "llama", active: 32 },
  { id: "@cf/qwen/qwen3.8-27b", label: "qwen3-27B", style: "llama", active: 27 },
  { id: "@cf/openai/gpt-oss-20b", label: "gpt-oss-20B", style: "openai", active: 20 },
  { id: "@cf/meta/llama-3.1-8b-instruct", label: "llama-3.1-8B", style: "llama", active: 8 },
];

export function extProviders() {
  const list = [];
  if (process.env.GROQ_API_KEY) list.push({
    kind: "ext", label: "groq", style: "openai", cost: 0,
    baseUrl: "https://api.groq.com/openai/v1/chat/completions",
    apiKey: process.env.GROQ_API_KEY, model: process.env.GROQ_MODEL || "llama-3.3-70b-versatile",
  });
  if (process.env.NVIDIA_NIM_API_KEY) list.push({
    kind: "ext", label: "nvidia-nim", style: "openai", cost: 0,
    baseUrl: process.env.NVIDIA_NIM_BASE_URL || "https://integrate.api.nvidia.com/v1/chat/completions",
    apiKey: process.env.NVIDIA_NIM_API_KEY, model: process.env.NVIDIA_NIM_MODEL || "meta/llama-3.1-70b-instruct",
  });
  if (process.env.OPENROUTER_API_KEY) list.push({
    kind: "ext", label: "openrouter", style: "openai", cost: 0,
    baseUrl: "https://openrouter.ai/api/v1/chat/completions",
    apiKey: process.env.OPENROUTER_API_KEY, model: process.env.OPENROUTER_MODEL || "meta-llama/llama-3.3-70b-instruct:free",
  });
  // keyless public fallback — real LLM, no quota, no signup
  list.push({
    kind: "ext", label: "pollinations", style: "openai", cost: 0, keyless: true,
    baseUrl: "https://text.pollinations.ai/openai",
    apiKey: "public", model: process.env.POLLINATIONS_MODEL || "gpt-oss-20b",
  });
  return list;
}

export function buildBackends() {
  const list = [];
  for (const acc of config.cfAccounts) {
    for (const m of CF_MODELS) list.push({ ...m, kind: "cf", accountId: acc, key: `cf:${acc}:${m.label}` });
  }
  for (const p of extProviders()) list.push({ ...p, key: `ext:${p.label}` });
  return list;
}

export const BACKENDS = buildBackends();
export const CF_PRIMARY = BACKENDS.find((b) => b.kind === "cf" && b.label === "gpt-oss-120B") || BACKENDS[0];

const PREFERRED = {
  math: ["nemotron-120B(A12B)", "gpt-oss-120B", "deepseek-r1-32B", "qwen3-27B", "gpt-oss-20B", "llama-3.1-8B"],
  code: ["nemotron-120B(A12B)", "gpt-oss-120B", "qwen3-27B", "gpt-oss-20B", "llama-3.1-8B"],
  creative: ["nemotron-120B(A12B)", "qwen3-27B", "gpt-oss-20B", "llama-3.1-8B"],
  deep: ["nemotron-120B(A12B)", "deepseek-r1-32B", "gpt-oss-120B", "qwen3-27B", "llama-3.1-8B"],
  chat: ["nemotron-120B(A12B)", "gpt-oss-120B", "gpt-oss-20B", "llama-3.1-8B"],
};

/**
 * Chooses the best backend for a task within the remaining daily neuron budget.
 * Keyed external providers win (no quota), then Cloudflare when affordable,
 * then the keyless fallback, so the app never hard-fails.
 */
export function pickTeacher(tag, expectedTokens = 700) {
  const ext = BACKENDS.filter((b) => b.kind === "ext");
  const keyed = ext.filter((b) => !b.keyless);
  const keyless = ext.filter((b) => b.keyless);
  if (keyed.length) return keyed[0];

  const affordable = (b) => {
    if (b.kind === "ext") return true;
    if (isExhausted(b.accountId)) return false;
    return neuronsRemaining(b.accountId) > estimateNeurons(expectedTokens, b.active) * 1.3;
  };
  for (const label of PREFERRED[tag] || PREFERRED.chat) {
    const b = BACKENDS.find((x) => x.label === label && affordable(x));
    if (b) return b;
  }
  if (keyless.length) return keyless[0];
  return BACKENDS.filter(affordable)[0] || BACKENDS[BACKENDS.length - 1];
}

export const isQuotaError = (e) =>
  /daily free allocation|neurons|quota|insufficient|rate limit/i.test(String(e?.message || e));

export async function callTeacher(backend, messages, temperature, stream, extra = {}) {
  const body = { messages, max_tokens: extra.max_tokens || 3072 };
  if (typeof temperature === "number" && temperature >= 0 && temperature <= 2) body.temperature = temperature;
  if (stream) body.stream = true;
  // `reasoning` is a Cloudflare Workers AI parameter — sending it to external
  // OpenAI-compatible endpoints makes them return an empty completion.
  if (backend.kind === "cf" && backend.style === "openai" && extra.reasoning) body.reasoning = extra.reasoning;

  const url = backend.kind === "ext"
    ? backend.baseUrl
    : `https://api.cloudflare.com/client/v4/accounts/${backend.accountId}/ai/run/${backend.id}`;
  const headers = backend.kind === "ext"
    ? { Authorization: `Bearer ${backend.apiKey}`, "Content-Type": "application/json" }
    : { Authorization: `Bearer ${config.cfToken}`, "Content-Type": "application/json" };

  if (config.debugRaw) log.debug("provider.request", { backend: backend.label, max_tokens: body.max_tokens, temperature: body.temperature, reasoning: body.reasoning ?? null });

  const r = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(config.providerTimeout),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      msg = j.errors?.[0]?.message || j.error?.message || msg;
    } catch {}
    const err = new Error(`${backend.label}: ${msg}`);
    err.status = r.status;
    throw err;
  }
  if (!stream && backend.kind === "cf") {
    try {
      r.clone().json().then((j) => addNeurons(backend.accountId, j?.result?.usage?.neurons)).catch(() => {});
    } catch {}
  }
  return r;
}

/** Tries every backend in order; marks exhausted Cloudflare accounts so we stop hitting them. */
export async function runTeacher(messages, opts = {}, backends = BACKENDS) {
  let lastErr;
  for (const b of backends) {
    if (b.kind === "cf" && (isExhausted(b.accountId) || neuronsRemaining(b.accountId) <= 0)) continue;
    try {
      const res = await callTeacher(b, messages, opts.temperature, opts.stream || false, opts);
      return { teacher: b, res };
    } catch (e) {
      lastErr = e;
      if (isQuotaError(e) && b.kind === "cf") setAccount(b.accountId, { exhausted: true });
      log.warn("provider.failed", { backend: b.label, reason: String(e.message || e).slice(0, 200) });
    }
  }
  const err = lastErr || new Error("all teachers failed");
  err.noCapacity = true;
  throw err;
}

/** Normalises the two response shapes: llama-style {response} and OpenAI-style {choices}. */
export function parseTeacherJSON(result) {
  if (typeof result?.response === "string" && result.response) return { content: result.response };
  const msg = result?.choices?.[0]?.message;
  if (msg) return { content: msg.content || "", reasoning: msg.reasoning || msg.reasoning_content || "" };
  return { content: "" };
}

/**
 * Normalised SSE reader.
 * - Cloudflare gpt-oss sends delta chunks AND a final aggregated `response` field.
 * - The aggregate may contain the full text while the delta stream was cut short,
 *   so the caller appends only the missing tail (never duplicating text).
 */
export async function* iterateSSE(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const emit = function* (obj) {
    const d = obj.choices?.[0]?.delta;
    if (d) {
      const c = d.content;
      const r = d.reasoning || d.reasoning_content;
      if (typeof c === "string" && c) yield { content: c };
      if (typeof r === "string" && r) yield { reasoning: r };
      return;
    }
    if (typeof obj.response === "string" && obj.response) yield { aggregate: obj.response };
  };
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n");
    buf = parts.pop() || "";
    for (const line of parts) {
      const t = line.trim();
      if (!t.startsWith("data:")) continue;
      const data = t.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      try { yield* emit(JSON.parse(data)); } catch {}
    }
  }
  if (buf.trim().startsWith("data:")) {
    const data = buf.trim().slice(5).trim();
    if (data && data !== "[DONE]") {
      try { yield* emit(JSON.parse(data)); } catch {}
    }
  }
}
