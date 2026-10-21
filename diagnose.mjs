// Deep diagnostics CLI — runs the full check suite against the running Nados instance.
// Usage: node diagnose.mjs [--verbose]
const BASE = process.env.NADOS_BASE || "http://localhost:4000";
const verbose = process.argv.includes("--verbose");

const C = { g: "\x1b[32m", r: "\x1b[31m", y: "\x1b[33m", c: "\x1b[36m", d: "\x1b[2m", x: "\x1b[0m" };

async function main() {
  let report;
  try {
    const res = await fetch(`${BASE}/api/diagnostics`, { signal: AbortSignal.timeout(120000) });
    report = await res.json();
  } catch (e) {
    console.log(`${C.r}✗ لا يمكن الوصول للخادم على ${BASE}: ${e.message}${C.x}`);
    console.log(`${C.d}تأكد من تشغيل: run-24x7.cmd${C.x}`);
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
    const h = await (await fetch(`${BASE}/api/health`)).json();
    console.log(`\n${C.c}السلسلة:${C.x} ${h.orchestration.chain.join(" → ")}`);
    console.log(`${C.c}مزوّدون خارجيون:${C.x} ${h.orchestration.externalProviders.join(", ") || "—"}`);
    console.log(`${C.c}المزايا:${C.x} ${JSON.stringify(h.features)}`);
  }
  console.log(report.ok ? `\n${C.g}كل الفحوص سليمة ✓${C.x}\n` : `\n${C.r}توجد فشل — راجع أعلاه${C.x}\n`);
  process.exit(report.ok ? 0 : 1);
}
main();
