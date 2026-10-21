// Deep diagnostics CLI — runs the full check suite against the running Nados instance.
// Usage: node diagnose.mjs [--verbose] [--deep]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.env.NADOS_BASE || "http://localhost:4000";
const verbose = process.argv.includes("--verbose");
const deep = process.argv.includes("--deep");

// read the local API key from .env (CLI tools authenticate with x-nados-key)
function apiKey() {
  if (process.env.NADOS_API_KEY) return process.env.NADOS_API_KEY;
  try {
    const raw = fs.readFileSync(path.join(__dirname, ".env"), "utf8");
    const m = raw.match(/^\s*NADOS_API_KEY\s*=\s*(.+)$/m);
    return m ? m[1].trim().replace(/^"|"$/g, "") : "";
  } catch {
    return "";
  }
}
const KEY = apiKey();
const headers = KEY ? { "x-nados-key": KEY } : {};

const C = { g: "\x1b[32m", r: "\x1b[31m", y: "\x1b[33m", c: "\x1b[36m", d: "\x1b[2m", x: "\x1b[0m" };

async function main() {
  let report;
  try {
    const res = await fetch(`${BASE}/api/diagnostics${deep ? "?deep=1" : ""}`, { headers, signal: AbortSignal.timeout(180000) });
    if (!res.ok) {
      const body = await res.text();
      console.log(`${C.r}✗ الخادم رد ${res.status}: ${body.slice(0, 200)}${C.x}`);
      process.exit(1);
    }
    report = await res.json();
  } catch (e) {
    console.log(`${C.r}✗ لا يمكن الوصول للخادم على ${BASE}: ${e.message}${C.x}`);
    console.log(`${C.d}تأكد من التشغيل: run-24x7.cmd${C.x}`);
    process.exit(1);
  }
  console.log(`\n${C.c}━━━ فحص Nados الشامل ━━━${C.x}  ${report.at}`);
  const s = report.summary;
  console.log(`${C.d}النتيجة: ${s.ok} ناجح · ${s.quotaBlocked || 0} محجوب بالحصة · ${s.failed} خطأ حقيقي${C.x}\n`);
  for (const c of report.checks) {
    const tag = c.ok ? `${C.g}✓${C.x}` : c.kind === "quota" ? `${C.y}◐${C.x}` : `${C.r}✗${C.x}`;
    console.log(`${tag} ${c.name.padEnd(46)} ${C.d}${String(c.ms).padStart(5)}ms${C.x}  ${c.detail}`);
  }
  console.log(`\n${C.c}ميزانية العصبونات:${C.x}`);
  for (const n of report.neurons) {
    const daily = n.daily || 10000;
    const bar = "█".repeat(Math.min(10, Math.round((n.used / daily) * 10)));
    const flag = n.exhausted ? `${C.r}مستنفد${C.x}` : `${C.g}متاح${C.x}`;
    console.log(`  ${n.account}  ${String(n.used).padStart(6)} / ${daily}  ${C.y}${bar.padEnd(10)}${C.x} ${flag}`);
  }
  if (verbose) {
    const h = await (await fetch(`${BASE}/api/health`, { headers })).json();
    console.log(`\n${C.c}السلسلة:${C.x} ${h.orchestration.chain.join(" → ")}`);
    console.log(`${C.c}مزوّدون خارجيون:${C.x} ${h.orchestration.externalProviders.join(", ") || "—"}`);
    console.log(`${C.c}المزايا:${C.x} ${JSON.stringify(h.features)}`);
  }
  console.log(report.ok ? `\n${C.g}كل الفحوص سليمة ✓${C.x}\n` : `\n${C.r}توجد فشل — راجع أعلاه${C.x}\n`);
  process.exit(report.ok ? 0 : 1);
}
main();
