// quick manual probe helper: node probe.mjs <mode> <message...>
const mode = process.argv[2] || "chat";
const message = process.argv.slice(3).join(" ") || "مرحبا";
const boundary = "----nadosprobe";
const body =
  `--${boundary}\r\nContent-Disposition: form-data; name="message"\r\n\r\n${message}\r\n` +
  `--${boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\n${mode}\r\n` +
  `--${boundary}--\r\n`;
const res = await fetch("http://localhost:4000/api/chat/stream", {
  method: "POST",
  headers: { "Content-Type": `multipart/form-data; boundary=${boundary}` },
  body,
  signal: AbortSignal.timeout(150000),
});
const reader = res.body.getReader();
const dec = new TextDecoder();
let buf = "", text = "", done = null;
const t0 = Date.now();
while (true) {
  const { value, fin } = await reader.read();
  if (fin) break;
  buf += dec.decode(value, { stream: true });
  const parts = buf.split("\n\n");
  buf = parts.pop() || "";
  for (const p of parts) {
    const line = p.split("\n").find((l) => l.startsWith("data: "));
    if (!line) continue;
    const o = JSON.parse(line.slice(6));
    if (o.type === "delta") {
      text += o.delta;
      process.stdout.write(".");
    }
    if (o.type === "done") done = o.reply;
    if (o.type === "error") console.log("ERROR:", o.message);
  }
}
console.log(`\n[${((Date.now() - t0) / 1000).toFixed(1)}s]`);
console.log("mode:", mode, "| teacher:", done?.teacher, "| provider:", done?.provider, "| sources:", done?.sources?.length || 0);
for (const s of done?.sources || []) console.log("  -", s.title, "|", s.source);
console.log("---");
console.log(text.slice(0, 900));
