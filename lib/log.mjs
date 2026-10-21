// Structured logging — machine readable (JSON lines) so monitoring/alerting can parse it.
import fs from "node:fs";
import path from "node:path";
import { ROOT } from "./config.mjs";

const LOG_DIR = path.join(ROOT, "logs");
let stream = null;

function ensureStream() {
  if (stream) return stream;
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    const file = path.join(LOG_DIR, "app.log");
    if (fs.existsSync(file) && fs.statSync(file).size > 8 * 1024 * 1024) {
      fs.renameSync(file, path.join(LOG_DIR, "app.log.1"));
    }
    stream = fs.createWriteStream(file, { flags: "a" });
  } catch {
    stream = null;
  }
  return stream;
}

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };
const minLevel = LEVELS[(process.env.NADOS_LOG_LEVEL || "info").toLowerCase()] ?? 20;

function emit(level, msg, fields = {}) {
  if (LEVELS[level] < minLevel) return;
  const rec = { ts: new Date().toISOString(), level, msg, ...fields };
  const line = JSON.stringify(rec);
  const s = ensureStream();
  if (s) s.write(line + "\n");
  if (level === "error" || level === "warn" || process.env.NADOS_LOG_STDOUT === "1") {
    console.log(line);
  }
}

export const log = {
  debug: (m, f) => emit("debug", m, f),
  info: (m, f) => emit("info", m, f),
  warn: (m, f) => emit("warn", m, f),
  error: (m, f) => emit("error", m, f),
};
