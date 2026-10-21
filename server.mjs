#!/usr/bin/env node
// Nados AI — entry point.
// Starts the HTTP server, probes the Cloudflare quotas and shuts down gracefully.
import path from "node:path";
import { config, loadEnv, ROOT, TRAIN_DIR } from "./lib/config.mjs";
import { log } from "./lib/log.mjs";
import { createServer } from "./lib/router.mjs";
import { initEngine } from "./lib/chat.mjs";
import { setAccount, isExhausted } from "./lib/store.mjs";
import { CF_PRIMARY } from "./lib/providers.mjs";
import { apiKey, hasApiKeyConfigured } from "./lib/security.mjs";

loadEnv();

if (!config.cfToken || !config.cfAccounts.length) {
  log.error("config.missing", { hint: "انسخ .env.example إلى .env واملأ CF_API_TOKEN و CF_ACCOUNTS" });
}

// load the trained model (Nados v1.1 — 245 parameters)
initEngine(path.join(TRAIN_DIR, "artifacts", "nados-v1.1.0.json"));

const server = createServer();

server.on("error", (err) => {
  if (err.code === "EADDRINUSE") {
    log.error("server.port_in_use", { port: config.port, hint: "أغلق العملية الأخرى أو غيّر PORT" });
    process.exit(1);
  }
  log.error("server.error", { reason: err.message });
  process.exit(1);
});

server.listen(config.port, config.host, () => {
  log.info("server.listening", {
    url: `http://${config.host}:${config.port}`,
    env: config.env,
    auth: config.authEnabled,
    api_key_source: hasApiKeyConfigured() ? "env" : "generated",
  });
  if (!hasApiKeyConfigured()) {
    log.info("api_key.generated", { hint: "أضفه إلى .env لاستخدامه من سطر الأوامر (diagnose/benchmark)" });
  }
  probeQuotas();
});

/** A 1-token probe tells us whether an account still has daily quota. */
async function probeQuotas() {
  for (const acc of config.cfAccounts) {
    try {
      const r = await fetch(`https://api.cloudflare.com/client/v4/accounts/${acc}/ai/run/@cf/meta/llama-3.1-8b-instruct`, {
        method: "POST",
        headers: { Authorization: `Bearer ${config.cfToken}`, "Content-Type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: "hi" }], max_tokens: 1 }),
        signal: AbortSignal.timeout(20000),
      });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        const m = j.errors?.[0]?.message || "";
        if (/daily free allocation|neurons|quota/i.test(m)) {
          setAccount(acc, { exhausted: true });
          log.info("quota.exhausted", { account: acc.slice(0, 8) });
        }
      } else {
        setAccount(acc, { exhausted: false });
        log.info("quota.available", { account: acc.slice(0, 8) });
      }
    } catch (e) {
      log.warn("quota.probe_failed", { account: acc.slice(0, 8), reason: e.message });
    }
  }
}

let shuttingDown = false;
function shutdown(signal) {
  if (shuttingDown) return;
  shuttingDown = true;
  log.info("server.shutdown", { signal });
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 5000).unref?.();
}
process.on("SIGTERM", () => shutdown("SIGTERM"));
process.on("SIGINT", () => shutdown("SIGINT"));
process.on("unhandledRejection", (reason) => log.error("unhandled_rejection", { reason: String(reason) }));
process.on("uncaughtException", (err) => log.error("uncaught_exception", { reason: err.message, stack: err.stack?.split("\n")[1] || null }));
