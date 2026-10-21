// Nados Benchmark Runner — 100 expert questions, budget-aware, resumable, auto-scored.
// Usage: node benchmark.mjs [--once] [--limit N] [--judge]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const BASE = process.env.NADOS_BASE || "http://localhost:4000";
const OUT = path.join(__dirname, "results.jsonl");
const REPORT = path.join(__dirname, "report.md");
const texts = JSON.parse(fs.readFileSync(path.join(__dirname, "texts.json"), "utf8"));
const meta = JSON.parse(fs.readFileSync(path.join(__dirname, "questions.json"), "utf8"));
const kwById = Object.fromEntries(meta.questions.map((q) => [q.id, q.kw]));

const args = process.argv.slice(2);
const ONCE = args.includes("--once");
const LIMIT = Number((args.find((a) => a.startsWith("--limit=")) || "").split("=")[1] || 0);
// One answer (~900 tokens, 12B active) costs ~7-50 neurons — the daily free tier is 10,000.
// Keep a small safety margin so we never start a request we cannot finish.
const MIN_BUDGET = Number(process.env.NADOS_MIN_BUDGET || 250);

// Concise answering style keeps each benchmark answer cheap on the daily neuron quota
// while still covering the key technical terms we score against.
const BENCH_SYSTEM_PROMPT =
  "أجب بأسلوب منظم ومباشر: عناوين قصيرة ونقاط، من 250 إلى 400 كلمة، مع مثال أو قاعدة عملية عند الحاجة. " +
  "اذكر المصطلحات التقنية الإنجليزية الصحيحة لكل مفهوم. لا تكرر السؤال ولا تضف مقدمات.";

function doneIds() {
  const ids = new Set();
  try {
    for (const line of fs.readFileSync(OUT, "utf8").split("\n")) {
      if (!line.trim()) continue;
      try {
        const o = JSON.parse(line);
        if (o.ok) ids.add(o.id);
      } catch {}
    }
  } catch {}
  return ids;
}

async function health() {
  try {
    const r = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(8000) });
    return await r.json();
  } catch {
    return null;
  }
}

function budgetAvailable(h) {
  if (!h) return 0;
  const accs = h.orchestration?.accounts || [];
  return accs.reduce((sum, a) => sum + (a.remaining || 0), 0);
}

async function ask(question) {
  const boundary = "----nadosbench";
  const body =
    `--${boundary}\r\nContent-Disposition: form-data; name="message"\r\n\r\n${question}\r\n` +
    `--${boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\nresearch\r\n` +
    `--${boundary}\r\nContent-Disposition: form-data; name="system_prompt"\r\n\r\n${BENCH_SYSTEM_PROMPT}\r\n` +
    `--${boundary}\r\nContent-Disposition: form-data; name="max_tokens"\r\n\r\n${process.env.NADOS_BENCH_TOKENS || 900}\r\n` +
    `--${boundary}--\r\n`;
  const res = await fetch(`${BASE}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
    body,
    signal: AbortSignal.timeout(240000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", text = "", done = null, error = null;
  while (true) {
    const { value, done: fin } = await reader.read();
    if (fin) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() || "";
    for (const p of parts) {
      const line = p.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        const o = JSON.parse(line.slice(6));
        if (o.type === "delta") text += o.delta;
        if (o.type === "done") done = o.reply;
        if (o.type === "error") error = o.message;
      } catch {}
    }
  }
  return { text, done, error };
}

// ---- heuristic scoring (no extra cloud cost) ----
function tokenPrefixMatch(textTokens, keyword) {
  const words = keyword.toLowerCase().split(/[^a-z0-9+]+/i).filter((w) => w.length >= 3);
  if (!words.length) return false;
  return words.every((w) => {
    const p = w.slice(0, Math.min(5, w.length));
    return textTokens.some((t) => t.startsWith(p) || (w.length > 5 && t.startsWith(w.slice(0, 4))));
  });
}

function score(id, text, done) {
  const kw = kwById[id] || [];
  const lower = text.toLowerCase();
  const textTokens = lower.match(/[a-z0-9+#.]+/g) || [];
  const hits = kw.filter((k) => tokenPrefixMatch(textTokens, k));
  const coverage = kw.length ? hits.length / kw.length : 0;
  const len = text.length;
  const arabic = (text.match(/[\u0600-\u06FF]/g) || []).length;
  const latin = (text.match(/[A-Za-z]/g) || []).length;
  const arabicRatio = arabic + latin ? arabic / (arabic + latin) : 0;
  const hasStructure = /^#{1,4}\s|^\s*[-*•]\s|^\s*\d+[.)]\s/m.test(text);
  const hasCode = /```/.test(text);
  const latinTokens = [...new Set(textTokens.filter((t) => t.length >= 4 && !/^\d+$/.test(t) && !["with","from","that","this","which","when","then","than","them","have","been","into","your","will","about","using","used","more","also","does","make","made","only","such","each","some","most","many","here","there","what","where","while","would","could","should","because","before","after","where","these","those","their","there","which","while","other","being","其"].includes(t)))];
  const latinDensity = Math.min(1, latinTokens.length / 30);
  const truncated = !text.trim().endsWith(".") && !/[.!؟?)\]}`"']$/.test(text.trim());
  const refused = /لا أستطيع|作为一个AI|I cannot|عذرا/.test(text.slice(0, 300));
  const empty = len < 120;
  const localOnly = done?.provider === "nados-local";

  let s = 0;
  s += coverage * 45;                       // exact technical-term coverage
  s += latinDensity * 15;                    // breadth of technical vocabulary
  s += len >= 600 ? 18 : len >= 300 ? 13 : len >= 150 ? 7 : 2;  // depth
  s += hasStructure ? 12 : 4;               // organization
  s += hasCode ? 5 : 0;
  s += arabicRatio > 0.25 ? 8 : 0;          // Arabic answer quality
  s += truncated ? -12 : 0;                 // truncation penalty
  s += refused ? -20 : 0;
  s += empty ? -30 : 0;
  s += localOnly ? -25 : 0;                 // local fallback = not a real answer
  s = Math.max(0, Math.min(100, Math.round(s)));
  return { score: s, coverage: +coverage.toFixed(2), hits, len, hasStructure, hasCode, truncated, refused, localOnly, arabicRatio: +arabicRatio.toFixed(2), latinTerms: latinTokens.length };
}

function writeReport() {
  const rows = [];
  for (const line of fs.readFileSync(OUT, "utf8").split("\n")) {
    if (!line.trim()) continue;
    try {
      const o = JSON.parse(line);
      if (o.ok) rows.push(o);
    } catch {}
  }
  rows.sort((a, b) => a.id - b.id);
  const avg = rows.length ? rows.reduce((a, b) => a + b.score, 0) / rows.length : 0;
  const bySection = {};
  for (const r of rows) {
    const sec = Object.entries(meta.sections).find(([range]) => {
      const [a, b] = range.split("-").map(Number);
      return r.id >= a && r.id <= b;
    });
    const name = sec ? sec[1] : "أخرى";
    (bySection[name] ||= []).push(r.score);
  }
  let md = `# تقرير اختبار Nados v1.1 — 100 سؤال\n\n`;
  md += `- مجموع الأسئلة المجتاز: **${rows.length}/100**\n- المتوسط العام: **${avg.toFixed(1)}/100**\n- آخر تحديث: ${new Date().toISOString()}\n\n`;
  md += `## النتائج حسب المحور\n\n| المحور | عدد | المتوسط |\n|---|---|---|\n`;
  for (const [name, arr] of Object.entries(bySection)) {
    md += `| ${name} | ${arr.length} | ${(arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1)} |\n`;
  }
  md += `\n## تفصيل الأسئلة\n\n| # | المحور | النتيجة | التغطية | المعلم | أول 90 حرفًا |\n|---|---|---|---|---|---|\n`;
  for (const r of rows) {
    const sec = Object.entries(meta.sections).find(([range]) => {
      const [a, b] = range.split("-").map(Number);
      return r.id >= a && r.id <= b;
    });
    const snippet = (r.text || "").replace(/\s+/g, " ").slice(0, 90);
    md += `| ${r.id} | ${sec ? sec[1] : ""} | ${r.score} | ${(r.coverage * 100).toFixed(0)}% | ${r.teacher || "-"} | ${snippet} |\n`;
  }
  fs.writeFileSync(REPORT, md, "utf8");
  return { count: rows.length, avg };
}

async function main() {
  if (args.includes("--rescore")) {
    const lines = fs.readFileSync(OUT, "utf8").split("\n").filter((l) => l.trim());
    const out = [];
    for (const l of lines) {
      const o = JSON.parse(l);
      if (o.ok) Object.assign(o, score(o.id, o.text || "", { provider: o.teacher }));
      out.push(JSON.stringify(o));
    }
    fs.writeFileSync(OUT, out.join("\n") + "\n", "utf8");
    const rep = writeReport();
    console.log(`[bench] rescored: ${rep.count}/100 avg=${rep.avg.toFixed(1)}`);
    return;
  }
  const finished = doneIds();
  const pending = Object.keys(texts).map(Number).filter((id) => !finished.has(id));
  console.log(`[bench] done=${finished.size} pending=${pending.length}`);

  let processed = 0;
  while (pending.length) {
    const id = pending.shift();
    const h = await health();
    const budget = budgetAvailable(h);
    if (budget < MIN_BUDGET) {
      console.log(`[bench] budget low (${budget} neurons) → pausing. Will resume when quota refills.`);
      if (ONCE) break;
      // wait 10 minutes then re-check (quota resets daily at 00:00 UTC)
      await new Promise((r) => setTimeout(r, 10 * 60 * 1000));
      continue;
    }
    const started = Date.now();
    try {
      const r = await ask(texts[id]);
      const sc = score(id, r.text, r.done);
      const rec = {
        id,
        ok: !r.error && r.text.length > 0,
        question: texts[id],
        text: r.text,
        teacher: r.done?.teacher || r.done?.provider || null,
        error: r.error || null,
        secs: +((Date.now() - started) / 1000).toFixed(1),
        at: new Date().toISOString(),
        ...sc,
      };
      fs.appendFileSync(OUT, JSON.stringify(rec) + "\n", "utf8");
      const rep = writeReport();
      console.log(`[bench] Q${id} score=${rec.score} cov=${rec.coverage} ${rec.secs}s | avg=${rep.avg.toFixed(1)} (${rep.count}/100)`);
      processed++;
      if (LIMIT && processed >= LIMIT) break;
    } catch (e) {
      console.error(`[bench] Q${id} failed: ${e.message}`);
      fs.appendFileSync(OUT, JSON.stringify({ id, ok: false, error: String(e.message), at: new Date().toISOString() }) + "\n", "utf8");
    }
  }
  const rep = writeReport();
  console.log(`[bench] finished: ${rep.count}/100, avg=${rep.avg.toFixed(1)} → ${REPORT}`);
}

main();
