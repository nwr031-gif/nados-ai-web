// quick manual probe helper: node probe.mjs <mode> <message...>
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const mode = process.argv[2] || "chat";
const message = process.argv.slice(3).join(" ") || "مرحبا";
const boundary = "----nadosprobe";
const body =
  `--${boundary}\r\nContent-Disposition: form-data; name="message"\r\n\r\n${message}\r\n` +
  `--${boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\n${mode}\r\n` +
  `--${boundary}--\r\n`;

// CLI auth: read the key from .env
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
const res = await fetch("http://localhost:4000/api/chat/stream", {
  method: "POST",
  headers: { "Content-Type": `multipart/form-data; boundary=${boundary}`, ...(KEY ? { "x-nados-key": KEY } : {}) },
  body,
  signal: AbortSignal.timeout(150000),
});
const reader = res.body.getReader();
const dec = new TextDecoder();
let buf = "", text = "", done = null;
const t0 = Date.now();
const watchdog = setTimeout(() => {
  console.error(`✗ timeout: stream did not end after 150s (received ${text.length} chars)`);
  process.exit(2);
}, 150000);
while (true) {
  const { value, done: fin } = await reader.read();
  if (fin) break;
  buf += dec.decode(value, { stream: true });
  const parts = buf.split("\n\n");
  buf = parts.pop() || "";
  for (const p of parts) {
    const line = p.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    let o;
    try { o = JSON.parse(line.slice(6)); } catch { continue; }
    if (o.type === "delta") {
      text += o.delta;
    }
    if (o.type === "done") done = o.reply;
    if (o.type === "error") console.error("ERROR:", o.message);
  }
}
clearTimeout(watchdog);
console.log(`\n[${((Date.now() - t0) / 1000).toFixed(1)}s]`);
console.log("mode:", mode, "| teacher:", done?.teacher, "| provider:", done?.provider, "| sources:", done?.sources?.length || 0);
for (const s of done?.sources || []) console.log("  -", s.title, "|", s.source);
console.log("---");
console.log(text.slice(0, 900));
// AbortSignal.timeout keeps a pending timer alive — exit explicitly so the process ends.
process.exit(0);
