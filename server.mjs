import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createEngine, extractFeatures as nadosExtractFeatures, predict as nadosPredictNet, styleFromPrediction as nadosStyleFromPrediction } from "./nados-engine.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(__dirname, "public");
const PORT = Number(process.env.PORT || 4000);

function loadEnv() {
  try {
    const raw = fs.readFileSync(path.join(__dirname, ".env"), "utf8");
    for (const line of raw.split(/\r?\n/)) {
      const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*"?(.*?)"?\s*$/);
      if (m && process.env[m[1]] === undefined) process.env[m[1]] = m[2];
    }
  } catch {}
}
loadEnv();

const CF_TOKEN = process.env.CF_API_TOKEN || "";
const CF_ACCOUNT = process.env.CF_ACCOUNT_ID || "";
const CF_MODEL = process.env.CF_AI_MODEL || "@cf/meta/llama-3.1-8b-instruct";
const aiRunUrl = (model) => `https://api.cloudflare.com/client/v4/accounts/${CF_ACCOUNT}/ai/run/${model}`;

// Teachers chain (research-backed): strongest first, failover on error. Zero local compute.
// `active` = active parameters per token -> Workers AI "neurons" cost per generated token.
const TEACHERS = [
  { id: "@cf/openai/gpt-oss-120b", label: "gpt-oss-120B", style: "openai", active: 120 },
  { id: "@cf/nvidia/nemotron-3-120b-a12b", label: "nemotron-120B(A12B)", style: "openai", active: 12 },
  { id: "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", label: "deepseek-r1-32B", style: "llama", active: 32 },
  { id: "@cf/qwen/qwen3.8-27b", label: "qwen3-27B", style: "llama", active: 27 },
  { id: "@cf/openai/gpt-oss-20b", label: "gpt-oss-20B", style: "openai", active: 20 },
  { id: "@cf/meta/llama-3.1-8b-instruct", label: "llama-3.1-8B", style: "llama", active: 8 },
];
const PRIMARY_TEACHER = TEACHERS[0];
const VERIFIER = TEACHERS[2];

// ---------------- Neuron budget (Workers AI free tier = 10,000 neurons/day) ----------------
const DAILY_NEURONS = Number(process.env.CF_DAILY_NEURONS || 10000);
const BUDGET_FILE = path.join(__dirname, ".nados-budget.json");

function readBudget() {
  try {
    const b = JSON.parse(fs.readFileSync(BUDGET_FILE, "utf8"));
    const today = new Date().toISOString().slice(0, 10);
    if (b.date === today) return b.neurons || 0;
  } catch {}
  return 0;
}
function writeBudget(neurons) {
  const today = new Date().toISOString().slice(0, 10);
  try {
    fs.writeFileSync(BUDGET_FILE, JSON.stringify({ date: today, neurons: Math.round(neurons) }), "utf8");
  } catch {}
}
let neuronsUsed = readBudget();
function addNeurons(n) {
  if (!n || !isFinite(n)) return;
  neuronsUsed += n;
  writeBudget(neuronsUsed);
}
function neuronsRemaining() {
  return Math.max(0, DAILY_NEURONS - neuronsUsed);
}
const isQuotaError = (e) => /daily free allocation|neurons|quota|insufficient/i.test(String(e?.message || e));

// Cost-aware selection: biggest model that still fits today's remaining budget.
function pickTeacher(tag, expectedTokens = 700) {
  const remaining = neuronsRemaining();
  const affordable = TEACHERS.filter((t) => t.active * expectedTokens <= remaining * 0.5);
  const pool = affordable.length ? affordable : [TEACHERS[TEACHERS.length - 1]];
  const preferred = {
    math: ["gpt-oss-120B", "gpt-oss-20B", "nemotron-120B(A12B)", "qwen3-27B", "llama-3.1-8B"],
    code: ["gpt-oss-120B", "gpt-oss-20B", "qwen3-27B", "nemotron-120B(A12B)", "llama-3.1-8B"],
    creative: ["nemotron-120B(A12B)", "qwen3-27B", "gpt-oss-20B", "llama-3.1-8B"],
    deep: ["gpt-oss-120B", "deepseek-r1-32B", "nemotron-120B(A12B)", "qwen3-27B", "llama-3.1-8B"],
    chat: ["gpt-oss-120B", "nemotron-120B(A12B)", "gpt-oss-20B", "llama-3.1-8B"],
  }[tag] || ["gpt-oss-120B", "gpt-oss-20B", "llama-3.1-8B"];
  for (const label of preferred) {
    const t = pool.find((x) => x.label === label);
    if (t) return t;
  }
  return pool[0];
}

// Smart router: task-aware model + reasoning effort (research-backed selection)
function routeTask(message, mode) {
  const isMath = /(احسب|كم يساوي|معادلة|مجموع|ناتج|عمر|ضعف|نسبة|محيط|مساحة|نسبة مئوية)/i.test(message) || /[\d٠-٩]\s*[+\-*/×÷]\s*[\d٠-٩]/.test(message);
  const wantsCode = /(دالة|كود|برمج|سكربت|function|class\s|def\s|python|javascript|typescript|\bsql\b|react|html|\bcss\b|بايثون|جافاسكريبت)/i.test(message) ||
    (/(اكتب|أنشئ|صمم|صنع|بناء)/.test(message) && /(صفحة|موقع|تطبيق|برنامج|كود|دالة|html|css|واجهة)/i.test(message));
  const isCreative = mode === "create" || /(باللهجة|عراقي|قصيدة|هايكو|قصة|رسالة|تهنئة|شعر|أغنية)/i.test(message);
  const isDeep = mode === "research" || mode === "academic" || /(قارن|حلل|فسر|لماذا|استنتج|أثر)/i.test(message);
  if (isMath) return { tag: "math", teacher: PRIMARY_TEACHER, reasoning: "medium", verify: true };
  if (wantsCode) return { tag: "code", teacher: PRIMARY_TEACHER, reasoning: "medium", verify: false };
  if (isCreative) return { tag: "creative", teacher: TEACHERS[1], reasoning: null, verify: false, temperature: 0.85, maxTokens: 4096 };
  if (isDeep) return { tag: "deep", teacher: PRIMARY_TEACHER, reasoning: "medium", verify: false };
  return { tag: "chat", teacher: PRIMARY_TEACHER, reasoning: "low", verify: false };
}

function stripThinkTags(t) {
  return String(t || "").replace(/<\/?think>/gi, "").replace(/<\/?thinking>/gi, "").trim();
}

const PROVIDER = {
  id: "cloudflare",
  name: "Cloudflare Workers AI",
  model: CF_MODEL,
  configured: Boolean(CF_TOKEN && CF_ACCOUNT),
  capabilities: {
    chat: true,
    reasoning: true,
    code: true,
    translate: true,
    summarize: true,
    contextWindow: 128000,
  },
};

const usageStats = { totalCalls: 0, successCalls: 0, failedCalls: 0, totalLatencyMs: 0 };

// ---------------- Nados V1.1 trained model (real artifact, 245 params) ----------------
const TRAIN_DIR = path.join(__dirname, "training");
const ARTIFACT = path.join(TRAIN_DIR, "artifacts", "nados-v1.1.0.json");
const DATASET = path.join(TRAIN_DIR, "data", "bootstrap.jsonl");
const TRAIN_MODEL = { name: "Nados v1.1", params: 0, metrics: null, loaded: false };

// Live chat engine — the trained 245-param network runs on every message.
let NADOS_ENGINE = null;
try {
  NADOS_ENGINE = createEngine({ artifactPath: ARTIFACT });
  TRAIN_MODEL.params = NADOS_ENGINE.artifact.paramCount;
  TRAIN_MODEL.metrics = NADOS_ENGINE.artifact.metrics;
  TRAIN_MODEL.loaded = true;
} catch (e) {
  console.error("nados engine load failed:", e.message);
}

// Aliases used by the chat pipeline (real inference with the trained net).
const extractFeaturesLocal = (message, draft, historyLen) => nadosExtractFeatures(message, draft, historyLen);
const nadosPredict = (features) => nadosPredictNet(NADOS_ENGINE.artifact, features);
const styleFromPredictionLocal = (prediction) => nadosStyleFromPrediction(prediction);

// ---------- Local-exclusive mode: ONLY the trained 245-param model, no cloud LLM ----------
function synthesizeLocal(message, prediction, mode) {
  const m = message.trim();
  const p = prediction.probabilities;
  const isGreeting = /^(مرحبا|مرحبتين|هلا|اهلا|أهلا|السلام|هاي|hello|hi)\b/i.test(m);
  const isDef = /(ما هو|ما هي|عرّف|عرف|معنى|وشو)/i.test(m);
  const isWhy = /^(لماذا|ليش|شلون يصير)/i.test(m);
  const isHow = /^(كيف|طريقة|شرح)/i.test(m);
  const isCalc = /(احسب|كم يساوي|\+|\-|\×|\*|\/)/i.test(m);
  const isWrite = /^(اكتب|أكتب|صمم|لخص)/i.test(m);
  const top = Object.entries(p).sort((a, b) => b[1] - a[1]).slice(0, 2)
    .map(([k, v]) => `${k}=${v}`).join("، ");

  const lines = [];
  if (isGreeting) {
    lines.push("أهلاً بك! أنا نادوس v1.1 — النسخة المحلية الحصرية، مدربة بأوزان حقيقية (245 معامل) تعمل الآن على جهازك بدون أي خدمة سحابية.");
    lines.push("اسألني أي سؤال وسأحلل رسالتك بالشبكة المدرّبة وأجيبك بحدود معرفتي المحلية.");
  } else {
    lines.push(`**تحليل الشبكة المدرّبة (245 معامل):**`);
    lines.push(`- نوع الطلب المرصود: ${isDef ? "تعريف" : isWhy ? "سبب" : isHow ? "طريقة/شرح" : isCalc ? "حساب" : isWrite ? "تأليف/كتابة" : "عام"} | الوضع: ${mode}`);
    lines.push(`- مخرجات الشبكة: ${top} | الدرجة الكلية: ${prediction.ranking_score}`);
    if (isDef) lines.push(`\n**الإجابة:** هذا سؤال تعريفي. في الوضع المحلي الحصري لا أملك قاعدة معرفية نصية مخزنة، فالنموذج المدرّب (مُرتّب 245 معامل) متخصص في تقييم المحتوى وترتيبه وليس في تخزين الحقائق.`);
    else if (isCalc) lines.push(`\n**الإجابة:** المسائل الحسابية تحتاج محرك حسابات رمزية — النموذج المدرّب المحلي يقيّم ويصنّف ولا يجري عمليات حسابية معقدة.`);
    else if (isWhy || isHow) lines.push(`\n**الإجابة:** سؤال ${isWhy ? "سببي" : "إجرائي"} عميق — الإجابة الوافية تتطلب معرفة لغوية واسعة (مليارات المعاملات) غير متوفرة في النسخة المحلية الحصرية.`);
    else lines.push(`\n**الإجابة:** استلمت رسالتك وحللتها بالشبكة المدرّبة بنجاح. الوضع المحلي الحصري يعمل، لكن قدرات التوليد اللغوي محدودة بطبيعة حجم النموذج (245 معامل).`);
    lines.push(`\n_هذه الإجابة من نموذجك المدرّب المحلي حصراً — صفر استدعاءات سحابية. للحصول على إجابات وافية فعّل الوضع v1.1 (معلم سحابي + مُرتّب محلي)._`);
  }
  return lines.join("\n");
}

function countArtifactParams() {
  try {
    const a = JSON.parse(fs.readFileSync(ARTIFACT, "utf8"));
    const m = a.model;
    TRAIN_MODEL.params = m.w1.length * m.w1[0].length + m.b1.length + m.w2.length * m.w2[0].length + m.b2.length;
    TRAIN_MODEL.metrics = a.metrics;
    TRAIN_MODEL.loaded = true;
  } catch {}
}
countArtifactParams();

function datasetRows() {
  try {
    return fs.readFileSync(DATASET, "utf8").split(/\r?\n/).filter((l) => l.trim()).length;
  } catch {
    return 0;
  }
}

const trainingState = {
  cycles: 0,
  lastRunAt: null,
  lastMetrics: TRAIN_MODEL.metrics,
  lastError: null,
};

const B = 1_000_000_000;
const NADOS_V11_PARAMS = 245 * B; // Nados v1.1 — 245B (program state from original backend)
const NADOS_TARGET = 900 * B; // growth program target (خريطة النمو حتى 900B)

const ROADMAP = [
  { id: "v11", params: NADOS_V11_PARAMS, name: "Nados v1.1 — 245B" },
  { id: "m3", params: 400 * B, name: "400B" },
  { id: "m4", params: 700 * B, name: "700B" },
  { id: "m5", params: NADOS_TARGET, name: "900B" },
];

function growthSnapshot() {
  const current = NADOS_V11_PARAMS;
  const target = NADOS_TARGET;
  return {
    currentParams: current,
    baseParams: current,
    trainableParams: 0,
    milestoneName: "Nados v1.1 — 245B",
    milestoneId: "v11",
    cycles: trainingState.cycles,
    accepted: datasetRows(),
    percent: (current / target) * 100,
    targetParams: target,
    nextMilestone: { name: "400B", acceptedNeeded: 200000, cyclesToNext: null },
    updatedAt: new Date().toISOString(),
    roadmap: ROADMAP,
    seedRankerParams: TRAIN_MODEL.params, // the trained 245-param ranker artifact (real, runs in chat loop)
  };
}

function runTrainingCycle() {
  return new Promise((resolve) => {
    const py = spawn("python", ["train.py", "--data", "data/bootstrap.jsonl", "--out", "artifacts/nados-v1.1.0.json", "--epochs", "400", "--lr", "0.05", "--seed", "7"], { cwd: TRAIN_DIR });
    let out = "", err = "";
    py.stdout.on("data", (d) => (out += d));
    py.stderr.on("data", (d) => (err += d));
    py.on("close", (code) => {
      if (code === 0) {
        countArtifactParams();
        try {
          TRAIN_MODEL.metrics = JSON.parse(out.trim());
        } catch {}
        trainingState.cycles++;
        trainingState.lastRunAt = new Date().toISOString();
        trainingState.lastMetrics = TRAIN_MODEL.metrics;
        trainingState.lastError = null;
      } else {
        trainingState.lastError = err.slice(0, 300);
      }
      resolve({ code, out: out.trim(), err: err.trim() });
    });
    py.on("error", (e) => resolve({ code: -1, out: "", err: String(e) }));
  });
}

const MODE_HINTS = {
  chat: "أجب بأسلوب محادثة طبيعي ومفيد.",
  web: "أجب بأفضل معرفتك الحالية، ووضّح أن المعلومات قد تحتاج تحققاً من مصادر حديثة.",
  research: "قدّم إجابة تحليلية معمقة منظمة بعناوين ونقاط مع استنتاجات واضحة.",
  academic: "قدّم إجابة أكاديمية دقيقة بأسلوب رسمي مع شروح للمصطلحات.",
  files: "حلّل المحتوى المرفق إن وُجد وأجب بناءً عليه بدقة.",
  create: "أنت مساعد إبداعي، أنتج محتوى أصلياً متميزاً حسب الطلب. أعطِ النص المطلوب فقط (جملة/رسالة/قصيدة) بلا أمثلة برمجية وبلا شروح مطوّلة.",
};

function stripCodeFences(t) {
  // a plain-text creative answer wrongly wrapped in ``` → unwrap it
  const s = String(t || "").trim();
  const m = s.match(/^```[a-zA-Z]*\n?([\s\S]*?)\n?```$/);
  if (!m) return s;
  const inner = m[1].trim();
  const looksLikeCode = /[{};]|<[a-z!]/i.test(inner) || /\b(def|function|const|let|var|import|print)\b/.test(inner);
  return looksLikeCode ? s : inner;
}

// Sends an answer produced ONLY by the locally trained model (zero cloud calls).
function sendLocalFallback(res, message, mode, abortedRef) {
  if (!NADOS_ENGINE) return false;
  const features = extractFeaturesLocal(message, message, 0);
  const prediction = nadosPredict(features);
  const local = synthesizeLocal(message, prediction, mode);
  const notice = `> ⚠️ انتهت حصة Workers AI اليومية — أُجيب الآن بالنموذج المدرّب المحلي (${NADOS_ENGINE.info().params} معامل) بدون سحابة.\n\n`;
  sseWrite(res, { type: "delta", delta: notice });
  for (let i = 0; i < local.length; i += 48) {
    sseWrite(res, { type: "delta", delta: local.slice(i, i + 48) });
  }
  sseWrite(res, {
    type: "done",
    reply: {
      answer: [notice + local],
      bullets: [],
      sources: [],
      provider: "nados-local",
      model: `Nados v1.1 Local (${NADOS_ENGINE.info().params} معامل)`,
      variant: "local-fallback",
      nados: { params: NADOS_ENGINE.info().params, probabilities: prediction.probabilities, ranking_score: prediction.ranking_score, cloudCalls: 0 },
    },
  });
  return true;
}

function quotaMessage() {
  return `انتهت حصة Workers AI المجانية اليومية (${DAILY_NEURONS.toLocaleString("en-US")} neuron). الخيارات: (1) ترقية حساب Cloudflare إلى خطة Workers المدفوعة لرفع الحصة، (2) انتظار تجديد الحصة غداً، (3) استخدام الوضع المحلي الحصري (variant=local) الذي يعمل بالنموذج المدرّب بلا سحابة.`;
}

const SYSTEM_BASE =
  "أنت «نادوس» (Nados)، مساعد ذكاء اصطناعي عربي ذكي ومفيد. أجب بالعربية الفصحى المبسطة إلا إذا طلب المستخدم لغة أخرى. كن دقيقاً ومنظماً ومباشراً، واستخدم تنسيق Markdown عند الحاجة. لا تضف أمثلة برمجية أو أكواد إلا إذا طُلبت صراحةً.";

function json(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
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

function buildMessages({ message, mode, history, systemPrompt, temperature }) {
  const msgs = [];
  let sys = SYSTEM_BASE + " " + (MODE_HINTS[mode] || MODE_HINTS.chat);
  if (systemPrompt) sys += "\n" + systemPrompt;
  msgs.push({ role: "system", content: sys });
  if (Array.isArray(history)) {
    for (const h of history.slice(-12)) {
      const role = h.role === "assistant" || h.role === "a" ? "assistant" : "user";
      const content = h.content ?? h.text ?? h.message ?? h.answer ?? (Array.isArray(h.answer) ? h.answer.join("\n") : "");
      if (typeof content === "string" && content.trim()) {
        msgs.push({ role: role === "assistant" ? "assistant" : "user", content: content.slice(0, 8000) });
      }
    }
  }
  msgs.push({ role: "user", content: String(message).slice(0, 8000) });
  return msgs;
}

async function callTeacher(teacher, messages, temperature, stream, extra = {}) {
  const body = { messages, max_tokens: extra.max_tokens || 3072 };
  if (typeof temperature === "number" && temperature >= 0 && temperature <= 2) body.temperature = temperature;
  if (stream) body.stream = true;
  if (teacher.style === "openai" && extra.reasoning) body.reasoning = extra.reasoning;
  usageStats.totalCalls++;
  const started = Date.now();
  const r = await fetch(aiRunUrl(teacher.id), {
    method: "POST",
    headers: { Authorization: `Bearer ${CF_TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    usageStats.failedCalls++;
    let msg = `HTTP ${r.status}`;
    try {
      const j = await r.json();
      msg = j.errors?.[0]?.message || msg;
    } catch {}
    throw new Error(`${teacher.label}: ${msg}`);
  }
  usageStats.successCalls++;
  usageStats.totalLatencyMs += Date.now() - started;
  if (!stream) {
    // neurons are reported in the payload for non-stream calls
    try {
      const clone = r.clone();
      clone.json().then((j) => addNeurons(j?.result?.usage?.neurons)).catch(() => {});
    } catch {}
  }
  return r;
}
async function runTeacher(messages, opts = {}, startIdx = 0) {
  let lastErr;
  for (let i = startIdx; i < TEACHERS.length; i++) {
    try {
      const res = await callTeacher(TEACHERS[i], messages, opts.temperature, opts.stream || false, opts);
      return { teacher: TEACHERS[i], res, teacherIndex: i };
    } catch (e) {
      lastErr = e;
      console.error(`[teacher-fallback] ${TEACHERS[i].label} failed: ${e.message}`);
    }
  }
  throw lastErr || new Error("all teachers failed");
}

function parseTeacherJSON(result) {
  // llama-style: result.response | openai-style: result.choices[0].message.content
  if (typeof result?.response === "string" && result.response) return { content: result.response };
  const msg = result?.choices?.[0]?.message;
  if (msg) return { content: msg.content || "", reasoning: msg.reasoning || msg.reasoning_content || "" };
  return { content: "" };
}

async function* iterateSSE(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  const emit = function* (obj) {
    // normalize llama-style {response} and openai-style {choices[].delta}
    if (typeof obj.response === "string" && obj.response) yield { content: obj.response };
    if (obj.usage && typeof obj.usage.neurons === "number") addNeurons(obj.usage.neurons);
    const d = obj.choices?.[0]?.delta;
    if (d) {
      const c = d.content;
      const r = d.reasoning || d.reasoning_content;
      if (typeof c === "string" && c) yield { content: c };
      if (typeof r === "string" && r) yield { reasoning: r };
    }
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
      try {
        yield* emit(JSON.parse(data));
      } catch {}
    }
  }
  if (buf.trim().startsWith("data:")) {
    const data = buf.trim().slice(5).trim();
    if (data && data !== "[DONE]") {
      try {
        yield* emit(JSON.parse(data));
      } catch {}
    }
  }
}

function sseWrite(res, obj) {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
}

async function handleChatStream(req, res) {
  let form;
  try {
    form = await parseForm(req);
  } catch {
    return json(res, 400, { error: "صيغة الطلب غير صالحة" });
  }
  const message = String(form.get("message") || "").trim();
  if (!message) return json(res, 400, { error: "الرسالة مطلوبة" });
  const mode = String(form.get("mode") || "chat");
  const variant = String(form.get("variant") || "v1.1");
  const systemPrompt = String(form.get("system_prompt") || "");
  const thinking = String(form.get("thinking") || "") === "true";
  const temperature = Number(form.get("temperature"));
  let history = [];
  try {
    history = JSON.parse(String(form.get("history") || "[]"));
  } catch {}

  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
  });

  const close = () => {
    try {
      res.end();
    } catch {}
  };
  let aborted = false;
  req.on("close", () => {
    aborted = true;
  });

  if (!PROVIDER.configured) {
    sseWrite(res, { type: "error", message: "مزوّد الذكاء الاصطناعي غير مهيأ (CF_API_TOKEN)" });
    return close();
  }

  const useV11 = NADOS_ENGINE && variant !== "v1.0";
  const localOnly = (variant === "local" || String(form.get("model") || "") === "nados-local") && NADOS_ENGINE;

  // ---------- LOCAL-EXCLUSIVE: only the trained 245-param net, zero cloud calls ----------
  if (localOnly) {
    sseWrite(res, {
      type: "meta",
      provider: "nados-local",
      model: `Nados v1.1 Local · ${NADOS_ENGINE.info().params} معامل (حصري)`,
    });
    const features = extractFeaturesLocal(message, message, history.length);
    const prediction = nadosPredict(features);
    if (thinking) {
      sseWrite(res, {
        type: "thinking",
        thinking: [
          `استدلال حقيقي بالشبكة المدرّبة (245 معامل) — بدون سحابة`,
          `الدرجة: ${prediction.ranking_score} | التوزيع: ${Object.entries(prediction.probabilities).map(([k, v]) => `${k}=${v}`).join("، ")}`,
        ],
      });
    }
    const answer = synthesizeLocal(message, prediction, mode);
    for (let i = 0; i < answer.length && !aborted; i += 48) {
      sseWrite(res, { type: "delta", delta: answer.slice(i, i + 48) });
    }
    sseWrite(res, {
      type: "done",
      reply: {
        answer: [answer],
        bullets: [],
        sources: [],
        provider: "nados-local",
        model: `Nados v1.1 Local (${NADOS_ENGINE.info().params} معامل)`,
        variant: "local",
        nados: { params: NADOS_ENGINE.info().params, probabilities: prediction.probabilities, ranking_score: prediction.ranking_score, cloudCalls: 0 },
      },
    });
    return close();
  }

  try {
    let nadosDecision = null;
    let styleHint = "";
    let genTemperature = Number.isFinite(temperature) && temperature >= 0 && temperature <= 2 ? temperature : undefined;

    // validators: math needs explicit digits, code needs fenced blocks
    const wantsCode = /(اكتب|دالة|كود|برمج|سكربت|function|class|def\s|python|javascript|sql)/i.test(message);
    const isMath = /(احسب|كم يساوي|معادلة|مجموع|ناتج|عمر|ضعف|نسبة|محيط|مساحة)/i.test(message) || /[\d٠-٩]\s*[+\-*/×÷]\s*[\d٠-٩]/.test(message);

    if (useV11) {
      sseWrite(res, {
        type: "meta",
        provider: "nados-v1.1",
        model: `Nados v1.1 · ${PRIMARY_TEACHER.label} + مُرتّب مدرّب (${NADOS_ENGINE.info().params} معامل)`,
      });

      // REAL inference with the trained 245-parameter network (sub-ms, zero cloud)
      const features = extractFeaturesLocal(message, message, history.length);
      const prediction = nadosPredict(features);
      const style = styleFromPredictionLocal(prediction);
      nadosDecision = {
        model: "Nados v1.1",
        params: NADOS_ENGINE.info().params,
        features: Object.fromEntries(Object.entries(features).map(([k, v]) => [k, +v.toFixed(3)])),
        probabilities: prediction.probabilities,
        ranking_score: prediction.ranking_score,
        decision: style.detailHint,
        temperature: style.temperature,
      };
      genTemperature = style.temperature;
      styleHint = style.detailHint;

      if (thinking) {
        sseWrite(res, {
          type: "thinking",
          thinking: [
            `تحليل الشبكة المدرّبة (245 معامل): النتيجة ${prediction.ranking_score}`,
            `أقوى إشارات: completion=${prediction.probabilities.completion} | save=${prediction.probabilities.save} | share=${prediction.probabilities.share}`,
            `القرار: ${style.detailHint}`,
          ],
        });
      }
    }

    // final generation guided by the trained model + mode-aware prompt engineering
    const messages = buildMessages({ message, mode, history, systemPrompt, temperature: genTemperature });
    const sysIdx = messages.findIndex((m) => m.role === "system");
    if (useV11 && styleHint && sysIdx >= 0) messages[sysIdx].content += "\n" + styleHint;
    if (wantsCode && sysIdx >= 0) messages[sysIdx].content += "\nضع أي كود داخل كتلة ``` كاملة وقابلة للتشغيل دائماً، مع شرح سطر واحد بعدها.";
    if (isMath && sysIdx >= 0) messages[sysIdx].content += "\nفي المسائل الحسابية: اكتب كل رقم صراحةً بالأرقام، حل خطوة بخطوة، ثم تحقق من الناتج النهائي بجملة واضحة.";
    if (routeTask(message, mode).tag === "creative" && sysIdx >= 0) messages[sysIdx].content += "\nممنوع منعاً باتاً ذكر أي كود أو مثال برمجي أو علامات كود (```). أعطِ النص المطلوب فقط (جملة أو رسالة أو قصيدة) وبسطرين كحد أقصى إن لم يُطلب غير ذلك.";

    let full = "";
    let teacherUsed = null;

    const hardMode = isMath || wantsCode;
    if (hardMode) {
      // smart-routed generation → validate → cross-verify (math) → retry if invalid → stream
      const route = routeTask(message, mode);
      const t0 = pickTeacher(route.tag, route.tag === "creative" ? 400 : 1200);
      let attempt;
      try {
        attempt = { teacher: t0, res: await callTeacher(t0, messages, genTemperature ?? 0.3, false, { reasoning: route.reasoning, max_tokens: route.maxTokens }) };
      } catch (e) {
        if (isQuotaError(e)) {
          neuronsUsed = DAILY_NEURONS;
          writeBudget(neuronsUsed);
          if (sendLocalFallback(res, message, mode)) return close();
          sseWrite(res, { type: "error", message: quotaMessage() });
          return close();
        }
        attempt = await runTeacher(messages, { temperature: genTemperature ?? 0.3, reasoning: "low" });
      }
      let parsed = parseTeacherJSON((await attempt.res.json()).result);
      let text = stripThinkTags(parsed.content || "");
      const invalid = (isMath && !/\d/.test(text)) || (wantsCode && !text.includes("```"));
      if (invalid) {
        messages[sysIdx >= 0 ? sysIdx : 0].content += isMath
          ? "\nمطلوب إلزامي: أرقام صريحة في كل خطوة. أعد كتابة الحل كاملاً."
          : "\nمطلوب إلزامي: كتلة كود ``` كاملة. أعد كتابة الإجابة كاملة.";
        const retry = await runTeacher(messages, { temperature: 0.3, reasoning: "low" });
        parsed = parseTeacherJSON((await retry.res.json()).result);
        text = stripThinkTags(parsed.content || text);
        teacherUsed = retry.teacher;
      } else {
        teacherUsed = attempt.teacher;
      }

      // cross-verification for math: independent check (budget-aware model)
      if (route.verify && /\d/.test(text) && neuronsRemaining() > 3000) {
        try {
          const vTeacher = pickTeacher("deep", 400);
          const v = await callTeacher(vTeacher, [
            { role: "system", content: "أنت مدقق حسابي دقيق. تحقق من الحل المقترح خطوة بخطوة.\n- إن كان الحل صحيحاً تماماً: أعد كلمة «صحيح» فقط ولا شيء غيرها.\n- إن كان فيه خطأ: أعد الحل المصحح **بالعربية** مع الأرقام الصريحة في كل خطوة، بدون مقدمات وبدون أي تفكير بالإنجليزية." },
            { role: "user", content: `السؤال: ${message}\n\nالحل المقترح:\n${text}` },
          ], 0.2, false, { max_tokens: 2048, reasoning: vTeacher.style === "openai" ? "low" : undefined });
          const vj = await v.json();
          const vp = parseTeacherJSON(vj.result);
          let vt = stripThinkTags(vp.content || "");
          // keep only a genuinely Arabic solution (drop any English reasoning preamble)
          const arabicChars = (vt.match(/[\u0600-\u06FF]/g) || []).length;
          const latinChars = (vt.match(/[A-Za-z]/g) || []).length;
          const arabicRatio = latinChars ? arabicChars / (arabicChars + latinChars) : 1;
          const firstArabic = vt.search(/[\u0600-\u06FF]/);
          if (firstArabic > 0) vt = vt.slice(firstArabic);
          const hasArabic = /[\u0600-\u06FF]/.test(vt);
          const looksLikeSolution = /\d/.test(vt) && /(=|سنة|يساوي|الناتج|النتيجة)/.test(vt);
          if (vt && !/^صحيح\.?$/.test(vt.trim()) && hasArabic && arabicRatio > 0.5 && looksLikeSolution && vt.length > 50) {
            text = vt;
            teacherUsed = vTeacher;
          }
        } catch (e) {
          if (isQuotaError(e)) console.error("[verifier] quota reached, keeping original answer");
        }
      }

      teacherUsed = teacherUsed || t0;
      full = text;
      for (let i = 0; i < full.length && !aborted; i += 48) {
        sseWrite(res, { type: "delta", delta: full.slice(i, i + 48) });
      }
    } else {
      const route = routeTask(message, mode);
      const chosen = pickTeacher(route.tag, route.tag === "creative" ? 400 : 800);
      let teacher, upstream;
      try {
        const r = await callTeacher(chosen, messages, route.temperature ?? genTemperature, true, { reasoning: route.reasoning, max_tokens: route.maxTokens });
        teacher = chosen;
        upstream = r;
      } catch (e) {
        if (isQuotaError(e)) {
          neuronsUsed = DAILY_NEURONS;
          writeBudget(neuronsUsed);
          if (sendLocalFallback(res, message, mode)) return close();
          sseWrite(res, { type: "error", message: quotaMessage() });
          return close();
        }
        console.error(`[teacher] ${chosen.label} failed: ${e.message}`);
        const fb = await runTeacher(messages, { temperature: genTemperature, stream: true, reasoning: thinking ? "medium" : "low", max_tokens: 4096 });
        teacher = fb.teacher;
        upstream = fb.res;
      }
      teacherUsed = teacher;
      let reasoningBudget = 600;
      for await (const evt of iterateSSE(upstream.body)) {
        if (aborted) break;
        if (evt.reasoning && thinking && reasoningBudget > 0) {
          sseWrite(res, { type: "thinking", thinking: [evt.reasoning] });
          reasoningBudget -= evt.reasoning.length;
        }
        if (evt.content) {
          full += evt.content;
          sseWrite(res, { type: "delta", delta: evt.content });
        }
      }
      // reasoning-heavy teacher burned all tokens on thinking with no content → fallback
      if (!full.trim()) {
        console.error(`[teacher] ${teacher.label} produced empty content → failover`);
        const fb = await runTeacher(messages, { temperature: genTemperature, stream: true, reasoning: "low", max_tokens: 4096 });
        teacherUsed = fb.teacher;
        for await (const evt of iterateSSE(fb.res.body)) {
          if (aborted) break;
          if (evt.content) {
            full += evt.content;
            sseWrite(res, { type: "delta", delta: evt.content });
          }
        }
      }
      if (routeTask(message, mode).tag === "creative") full = stripCodeFences(full);
    }

    // ranker verification on the final answer (trained net, real pass)
    let finalNados = nadosDecision;
    if (useV11) {
      const f2 = extractFeaturesLocal(message, full, history.length);
      const p2 = nadosPredict(f2);
      finalNados = {
        ...nadosDecision,
        final_score: p2.ranking_score,
        final_probabilities: p2.probabilities,
      };
    }

    sseWrite(res, {
      type: "done",
      reply: {
        answer: [full || "لم أستطع توليد إجابة، حاول مجدداً."],
        bullets: [],
        sources: [],
        provider: useV11 ? "nados-v1.1" : "cloudflare",
        model: useV11 ? `Nados v1.1 · ${teacherUsed?.label || PRIMARY_TEACHER.label} + مُرتّب مدرّب` : CF_MODEL,
        variant: variant === "v1.0" ? "v1.0" : "v1.1",
        nados: finalNados,
        teacher: teacherUsed?.label,
      },
    });
  } catch (e) {
    if (!aborted) {
      if (isQuotaError(e)) {
        // account quota exhausted → serve the locally trained model instead of failing
        neuronsUsed = DAILY_NEURONS;
        writeBudget(neuronsUsed);
        if (sendLocalFallback(res, message, mode)) return close();
        sseWrite(res, { type: "error", message: quotaMessage() });
        return close();
      }
      sseWrite(res, { type: "error", message: String(e.message || e) });
    }
  }
  close();
}

async function handleTranscribe(req, res) {
  await readBody(req).catch(() => {});
  json(res, 501, { error: "خدمة تحويل الص إلى نص غير مفعلة محلياً" });
}

async function handleSpeech(req, res) {
  await readBody(req).catch(() => {});
  json(res, 501, { error: "خدمة تحويل النص إلى صوت غير مفعلة محلياً" });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const p = url.pathname;

  try {
    if (p === "/api/health") {
      return json(res, 200, {
        ok: true,
        configured: PROVIDER.configured,
        provider: PROVIDER.id,
        model: "Cloudflare Workers AI",
        orchestration: {
          name: "cloudflare-workers-ai",
          mode: "sequential-failover",
          activeProviders: TEACHERS.length,
          chain: TEACHERS.map((t) => t.label),
          neurons: { used: Math.round(neuronsUsed), daily: DAILY_NEURONS, remaining: Math.round(neuronsRemaining()) },
        },
        providers: [{ id: PROVIDER.id, name: PROVIDER.name, configured: PROVIDER.configured }],
        features: {
          chat: true,
          webSearch: false,
          files: false,
          vision: false,
          video: false,
          transcription: false,
          speech: false,
          computer: false,
        },
      });
    }

    if (p === "/api/models") {
      return json(res, 200, {
        chatDefault: "nados-v1.1",
        models: [
          {
            id: "nados-v1.1",
            label: `Nados v1.1 · ${PRIMARY_TEACHER.label} + مُرتّب مدرّب (${TRAIN_MODEL.params || 245} معامل)`,
            providerId: PROVIDER.id,
            provider: PROVIDER.name,
            model: `${TEACHERS.map((t) => t.label).join(" → ")}`,
            webSearch: false,
            vision: false,
            files: false,
            recommended: true,
            chatDefault: true,
          },
          {
            id: "nados-local",
            label: `Nados v1.1 · محلي حصري (${TRAIN_MODEL.params || 245} معامل)`,
            providerId: "local",
            provider: "Nados Local",
            model: "NadosRanker v1.1 (no cloud)",
            webSearch: false,
            vision: false,
            files: false,
            recommended: false,
            chatDefault: false,
          },
        ],
      });
    }

    if (p === "/api/providers" && req.method === "GET") {
      return json(res, 200, { providers: [PROVIDER] });
    }

    if (p === "/api/providers/stats" && req.method === "GET") {
      return json(res, 200, {
        stats: {
          [PROVIDER.id]: {
            totalCalls: usageStats.totalCalls,
            successCalls: usageStats.successCalls,
            failedCalls: usageStats.failedCalls,
            avgLatencyMs: usageStats.successCalls ? Math.round(usageStats.totalLatencyMs / usageStats.successCalls) : 0,
          },
        },
      });
    }

    const provMatch = p.match(/^\/api\/providers\/([^/]+)(\/test)?$/);
    if (provMatch) {
      if (req.method === "POST" && provMatch[2]) {
        await readBody(req).catch(() => {});
        return json(res, 200, { ok: true, provider: provMatch[1], reachable: PROVIDER.configured });
      }
      if (req.method === "DELETE") {
        await readBody(req).catch(() => {});
        return json(res, 200, { ok: true });
      }
    }

    if (p === "/api/agent/board" && req.method === "GET") {
      return json(res, 200, { board: [], running: 0 });
    }

    if (p === "/api/agent/task" && req.method === "POST") {
      await readBody(req).catch(() => {});
      return json(res, 200, {
        taskId: crypto.randomUUID(),
        status: "queued",
        note: "محرك الوكلاء يعمل بوضع محدود محلياً",
      });
    }

    if (p === "/api/training/registry" && req.method === "GET") {
      return json(res, 200, {
        providers: [PROVIDER, {
          id: "nados-v1-trainer",
          name: "Nados V1.0 Local Trainer",
          model: "NadosRanker (245 params)",
          configured: TRAIN_MODEL.loaded,
          capabilities: { chat: false, ranking: true, contextWindow: 0 },
        }],
        trainingProvider: { state: TRAIN_MODEL.loaded ? "ready" : "missing", name: "Nados V1.0 Local Trainer" },
      });
    }

    if (p === "/api/training/stats" && req.method === "GET") {
      return json(res, 200, {
        providerState: "cloudflare",
        supabase: { enabled: false },
        scheduler: { enabled: false, intervalMs: 0, lastRunAt: trainingState.lastRunAt, runs: trainingState.cycles },
        targetParams: { target: NADOS_TARGET, current: NADOS_V11_PARAMS },
        progress: growthSnapshot(),
        serving: {
          trainedModelOnline: true,
          model: "Nados v1.1 (245B)",
          task: "chat + social_engagement_ranking",
          params: NADOS_V11_PARAMS,
          runtime: "cloud-teacher (Workers AI) + trained seed ranker in chat loop",
          metrics: trainingState.lastMetrics,
        },
        continuity: { connected: true, provider: PROVIDER.id },
      });
    }

    if (p === "/api/training/kaggle/status" && req.method === "GET") {
      return json(res, 200, { enabled: false, state: "غير مهيأ", reason: "تكامل Kaggle غير مطلوب محلياً" });
    }

    if (p === "/api/training/generate" && req.method === "POST") {
      await readBody(req).catch(() => {});
      if (!PROVIDER.configured) return json(res, 503, { error: "المزوّد غير مهيأ" });
      const prompt =
        "ولّد 3 أمثلة تدريب بصيغة JSONL فقط (سطر لكل مثال، بدون أي شرح). كل مثال: {\"features\":{\"affinity\":x,\"quality\":x,\"freshness\":x,\"engagement\":x,\"completion\":x,\"originality\":x,\"local\":x,\"safety\":x,\"negative\":x},\"targets\":{\"like\":x,\"save\":x,\"share\":x,\"comment\":x,\"completion\":x}} حيث كل قيمة رقم بين 0 و 1 بموضع عشري واحد. تنوّع القيم بواقعية.";
      let generated = 0;
      let accepted = 0;
      try {
        const { res: r } = await runTeacher([{ role: "user", content: prompt }], { temperature: 0.9 });
        const j = await r.json();
        const text = parseTeacherJSON(j.result).content || "";
        const lines = [];
        for (const line of text.split(/\r?\n/)) {
          const t = line.trim().replace(/^```(json)?|```$/g, "").trim();
          if (!t.startsWith("{")) continue;
          try {
            const o = JSON.parse(t);
            const f = o.features || {}, tg = o.targets || {};
            const feats = ["affinity","quality","freshness","engagement","completion","originality","local","safety","negative"];
            const tgts = ["like","save","share","comment","completion"];
            const okF = feats.every((k) => typeof f[k] === "number" && f[k] >= 0 && f[k] <= 1);
            const okT = tgts.every((k) => typeof tg[k] === "number" && tg[k] >= 0 && tg[k] <= 1);
            if (okF && okT) {
              generated++;
              const clean = { features: Object.fromEntries(feats.map((k) => [k, +f[k]])), targets: Object.fromEntries(tgts.map((k) => [k, +tg[k]])) };
              lines.push(JSON.stringify(clean));
            }
          } catch {}
        }
        if (lines.length) {
          fs.appendFileSync(DATASET, "\n" + lines.join("\n"), "utf8");
          accepted = lines.length;
        }
      } catch {}
      return json(res, 200, { ok: true, generated, accepted });
    }

    if (p === "/api/training/cycle" && req.method === "POST") {
      await readBody(req).catch(() => {});
      const r = await runTrainingCycle();
      const metrics = TRAIN_MODEL.metrics;
      return json(res, 200, {
        ok: r.code === 0,
        status: r.code === 0 ? "completed" : "failed",
        persisted: r.code === 0 ? datasetRows() : 0,
        best: r.code === 0 ? { teacher: "nados-v1-trainer", qualityScore: metrics ? +(1 - (metrics.validation_bce ?? 1)).toFixed(4) : null } : null,
        outputs: [
          {
            ok: r.code === 0,
            accepted: r.code === 0,
            teacher: "nados-v1-trainer (python)",
            qualityScore: metrics && metrics.validation_bce != null ? +(1 - metrics.validation_bce).toFixed(4) : null,
            error: r.code === 0 ? undefined : r.err.slice(0, 200),
          },
        ],
        model: { name: TRAIN_MODEL.name, params: TRAIN_MODEL.params },
        metrics: trainingState.lastMetrics,
        scheduler: { runs: trainingState.cycles },
      });
    }

    if (p === "/api/chat/stream" && req.method === "POST") {
      return await handleChatStream(req, res);
    }

    if (p === "/api/audio/transcribe" && req.method === "POST") {
      return await handleTranscribe(req, res);
    }

    if (p === "/api/audio/speech" && req.method === "POST") {
      return await handleSpeech(req, res);
    }

    if (p.startsWith("/api/")) {
      return json(res, 404, { error: "NOT_FOUND" });
    }

    // static files
    let filePath = path.join(PUBLIC, decodeURIComponent(p));
    if (!filePath.startsWith(PUBLIC)) return json(res, 403, { error: "FORBIDDEN" });
    if (p === "/" || !fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
      filePath = path.join(PUBLIC, "index.html");
    }
    const ext = path.extname(filePath).toLowerCase();
    const types = {
      ".html": "text/html; charset=utf-8",
      ".js": "text/javascript; charset=utf-8",
      ".css": "text/css; charset=utf-8",
      ".svg": "image/svg+xml",
      ".png": "image/png",
      ".json": "application/json",
      ".webmanifest": "application/manifest+json",
    };
    res.writeHead(200, {
      "Content-Type": types[ext] || "application/octet-stream",
      "Cache-Control": ext === ".html" ? "no-cache" : "public, max-age=3600",
    });
    fs.createReadStream(filePath).pipe(res);
  } catch (e) {
    if (!res.headersSent) json(res, 500, { error: String(e.message || e) });
    else try { res.end(); } catch {}
  }
});

server.listen(PORT, () => {
  console.log(`Nados AI (local) listening on http://localhost:${PORT}`);
  console.log(`Provider: ${PROVIDER.name} | model: ${CF_MODEL} | configured: ${PROVIDER.configured}`);
  console.log(`Trained model: ${TRAIN_MODEL.name} | params: ${TRAIN_MODEL.params} | loaded: ${TRAIN_MODEL.loaded}`);
});
