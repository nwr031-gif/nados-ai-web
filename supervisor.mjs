// Nados 24/7 supervisor — keeps the server alive and drives the benchmark runner.
// Start:  node supervisor.mjs        (or via the scheduled task "Nados24x7")
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname; // supervisor.mjs lives in the project root
const BENCH = path.join(__dirname, "benchmark", "benchmark.mjs");
const BASE = "http://localhost:4000";
const LOG = path.join(__dirname, "logs");
fs.mkdirSync(LOG, { recursive: true });

const stamp = () => new Date().toISOString().slice(0, 19).replace("T", " ");
const log = (m) => {
  const line = `[${stamp()}] ${m}`;
  console.log(line);
  try {
    fs.appendFileSync(path.join(LOG, "supervisor.log"), line + "\n", "utf8");
  } catch {}
};

let server = null;
let bench = null;
let restarts = 0;

function startServer() {
  if (server) return;
  const out = fs.openSync(path.join(LOG, "server.log"), "a");
  server = spawn(process.execPath, [path.join(ROOT, "server.mjs")], {
    cwd: ROOT,
    stdio: ["ignore", out, out],
    env: { ...process.env },
  });
  server.on("exit", (code) => {
    log(`server exited code=${code} → restarting in 5s`);
    server = null;
    restarts++;
    setTimeout(startServer, 5000);
  });
  log("server started");
}

function startBench() {
  if (bench) return;
  const out = fs.openSync(path.join(LOG, "benchmark.log"), "a");
  bench = spawn(process.execPath, [BENCH], {
    cwd: ROOT,
    stdio: ["ignore", out, out],
    env: { ...process.env },
  });
  bench.on("exit", (code) => {
    log(`benchmark runner exited code=${code}`);
    bench = null;
    // runner sleeps/pauses on low quota internally; restart it every 30 min to re-check
    setTimeout(startBench, 30 * 60 * 1000);
  });
  log("benchmark runner started");
}

async function healthy() {
  try {
    const r = await fetch(`${BASE}/api/health`, { signal: AbortSignal.timeout(6000) });
    const j = await r.json();
    return j && j.ok === true;
  } catch {
    return false;
  }
}

// periodic health check: restart the server if it became unresponsive
setInterval(async () => {
  const ok = await healthy();
  if (!ok) {
    log("health check FAILED → killing server for restart");
    try {
      server?.kill("SIGKILL");
    } catch {}
  } else {
    const accs = (await (await fetch(`${BASE}/api/health`)).json()).orchestration?.accounts || [];
    log(`health OK | accounts remaining: ${accs.map((a) => a.remaining).join("/")} | restarts=${restarts}`);
  }
}, 5 * 60 * 1000);

process.on("SIGTERM", () => {
  server?.kill();
  bench?.kill();
  process.exit(0);
});
process.on("SIGINT", () => {
  server?.kill();
  bench?.kill();
  process.exit(0);
});

startServer();
setTimeout(startBench, 8000);
log("Nados 24/7 supervisor active");
