// Chat pipeline: trained Nados ranker -> cloud teacher -> streamed answer.
import { config } from "./config.mjs";
import { log } from "./log.mjs";
import { webSearch, buildContext } from "../web-search.mjs";
import { createEngine, extractFeatures, predict, styleFromPrediction } from "../nados-engine.mjs";
import { callTeacher, runTeacher, pickTeacher, parseTeacherJSON, iterateSSE, isQuotaError, CF_PRIMARY } from "./providers.mjs";
import { cacheKey, cacheGet, cacheSet, addNeurons, estimateNeurons, markExhaustedAll } from "./store.mjs";

// ---------------- trained model (Nados v1.1 — 245 parameters) ----------------
let ENGINE = null;
export function initEngine(artifactPath) {
  try {
    ENGINE = createEngine({ artifactPath });
    log.info("engine.loaded", { params: ENGINE.info().params, version: ENGINE.info().name });
    return ENGINE;
  } catch (e) {
    log.error("engine.load_failed", { reason: e.message });
    return null;
  }
}
export const engine = () => ENGINE;

const VALID_MODES = new Set(["chat", "web", "research", "academic", "files", "create"]);
const MAX_MESSAGE = 8000;

const MODE_HINTS = {
  chat: "أجب بأسلوب محادثة طبيعي ومفيد.",
  web: "أجب بأفضل معرفتك الحالية، ووضّح أن المعلومات قد تحتاج تحققاً من مصادر حديثة.",
  research: "قدّم إجابة تحليلية معمقة منظمة بعناوين ونقاط مع استنتاجات واضحة.",
  academic: "قدّم إجابة أكاديمية دقيقة بأسلوب رسمي مع شروح للمصطلحات.",
  files: "حلّل المحتوى المرفق إن وُجد وأجب بناءً عليه بدقة.",
  create: "أنت مساعد إبداعي، أنتج محتوى أصلياً متميزاً حسب الطلب. أعطِ النص المطلوب فقط (جملة/رسالة/قصيدة) بلا أمثلة برمجية وبلا شروح مطوّلة.",
};

const SYSTEM_BASE =
  "أنت «نادوس» (Nados)، مساعد ذكاء اصطناعي عربي ذكي ومفيد. أجب بالعربية الفصحى المبسطة إلا إذا طلب المستخدم لغة أخرى. كن دقيقاً ومنظماً ومباشراً، واستخدم تنسيق Markdown عند الحاجة. لا تضف أمثلة برمجية أو أكواد إلا إذا طُلبت صراحةً.";

export function routeTask(message, mode) {
  const isMath = /(احسب|كم يساوي|معادلة|مجموع|ناتج|عمر|ضعف|نسبة|محيط|مساحة|نسبة مئوية)/i.test(message) || /[\d٠-٩]\s*[+\-*/×÷]\s*[\d٠-٩]/.test(message);
  const wantsCode = /(دالة|كود|برمج|سكربت|function|class\s|def\s|python|javascript|typescript|\bsql\b|react|html|\bcss\b|بايثون|جافاسكريبت)/i.test(message) ||
    (/(اكتب|أنشئ|صمم|صنع|بناء)/.test(message) && /(صفحة|موقع|تطبيق|برنامج|كود|دالة|html|css|واجهة)/i.test(message));
  const isCreative = mode === "create" || /(باللهجة|عراقي|قصيدة|هايكو|قصة|رسالة|تهنئة|شعر|أغنية)/i.test(message);
  const isDeep = mode === "research" || mode === "academic" || /(قارن|حلل|فسر|لماذا|استنتج|أثر)/i.test(message);
  if (isMath) return { tag: "math", reasoning: "low", verify: true, maxTokens: 3000 };
  if (wantsCode) return { tag: "code", reasoning: "low", verify: false, maxTokens: 3000 };
  if (isCreative) return { tag: "creative", reasoning: null, verify: false, temperature: 0.85, maxTokens: 1500 };
  if (isDeep) return { tag: "deep", reasoning: "medium", verify: false, maxTokens: 3000 };
  return { tag: "chat", reasoning: "low", verify: false, maxTokens: 2000 };
}

function buildMessages({ message, mode, history, systemPrompt, temperature }) {
  const msgs = [];
  let sys = SYSTEM_BASE + " " + (MODE_HINTS[mode] || MODE_HINTS.chat);
  if (systemPrompt) sys += "\n" + String(systemPrompt).slice(0, 2000);
  msgs.push({ role: "system", content: sys });
  if (Array.isArray(history)) {
    for (const h of history.slice(-12)) {
      const role = h.role === "assistant" || h.role === "a" ? "assistant" : "user";
      const content = h.content ?? h.text ?? h.message ?? h.answer ?? (Array.isArray(h.answer) ? h.answer.join("\n") : "");
      if (typeof content === "string" && content.trim()) {
        msgs.push({ role, content: content.slice(0, 8000) });
      }
    }
  }
  msgs.push({ role: "user", content: String(message).slice(0, MAX_MESSAGE) });
  return msgs;
}

const stripThinkTags = (t) => String(t || "").replace(/<\/?think>/gi, "").replace(/<\/?thinking>/gi, "").trim();

function stripCodeFences(t) {
  const s = String(t || "").trim();
  const m = s.match(/^```[a-zA-Z]*\n?([\s\S]*?)\n?```$/);
  if (!m) return s;
  const inner = m[1].trim();
  const looksLikeCode = /[{};]|<[a-z!]/i.test(inner) || /\b(def|function|const|let|var|import|print)\b/.test(inner);
  return looksLikeCode ? s : inner;
}

/** Local-only answer produced by the trained 245-parameter model (no cloud calls). */
export function synthesizeLocal(message, prediction, mode) {
  const m = message.trim();
  const p = prediction.probabilities;
  const isGreeting = /^(مرحبا|مرحبتين|هلا|اهلا|أهلا|السلام|هاي|hello|hi)\b/i.test(m);
  const isDef = /(ما هو|ما هي|عرّف|عرف|معنى|وشو)/i.test(m);
  const isWhy = /^(لماذا|ليش|شلون يصير)/i.test(m);
  const isHow = /^(كيف|طريقة|شرح)/i.test(m);
  const isCalc = /(احسب|كم يساوي|\+|\-|\×|\*|\/)/i.test(m);
  const isWrite = /^(اكتب|أكتب|صمم|لخص)/i.test(m);
  const top = Object.entries(p).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k, v]) => `${k}=${v}`).join("، ");
  const lines = [];
  if (isGreeting) {
    lines.push("أهلاً بك! أنا نادوس v1.1 — النسخة المحلية الحصرية، مدربة بأوزان حقيقية (245 معامل) تعمل الآن على جهازك بدون أي خدمة سحابية.");
    lines.push("اسألني أي سؤال وسأحلل رسالتك بالشبكة المدرّبة وأجيبك بحدود معرفتي المحلية.");
  } else {
    lines.push("**تحليل الشبكة المدرّبة (245 معامل):**");
    lines.push(`- نوع الطلب المرصود: ${isDef ? "تعريف" : isWhy ? "سبب" : isHow ? "طريقة/شرح" : isCalc ? "حساب" : isWrite ? "تأليف/كتابة" : "عام"} | الوضع: ${mode}`);
    lines.push(`- مخرجات الشبكة: ${top} | الدرجة الكلية: ${prediction.ranking_score}`);
    if (isDef) lines.push("\n**الإجابة:** هذا سؤال تعريفي. في الوضع المحلي الحصري لا أملك قاعدة معرفية نصية مخزنة، فالنموذج المدرّب (مُرتّب 245 معامل) متخصص في تقييم المحتوى وترتيبه وليس في تخزين الحقائق.");
    else if (isCalc) lines.push("\n**الإجابة:** المسائل الحسابية تحتاج محرك حسابات رمزية — النموذج المدرّب المحلي يقيّم ويصنّف ولا يجري عمليات حسابية معقدة.");
    else if (isWhy || isHow) lines.push(`\n**الإجابة:** سؤال ${isWhy ? "سببي" : "إجرائي"} عميق — الإجابة الوافية تتطلب معرفة لغوية واسعة (مليارات المعاملات) غير متوفرة في النسخة المحلية الحصرية.`);
    else lines.push("\n**الإجابة:** استلمت رسالتك وحللتها بالشبكة المدرّبة بنجاح. الوضع المحلي الحصري يعمل، لكن قدرات التوليد اللغوي محدودة بطبيعة حجم النموذج (245 معامل).");
    lines.push("\n_هذه الإجابة من نموذجك المدرّب المحلي حصراً — صفر استدعاءات سحابية. للحصول على إجابات وافية فعّل الوضع v1.1 (معلم سحابي + مُرتّب محلي)._");
  }
  return lines.join("\n");
}

function isDegenerate(t) {
  if (!t) return true;
  if (t.length > 200 && t.indexOf(t.slice(0, 100), 100) !== -1) return true;
  const head = t.slice(0, 400).replace(/\s/g, "");
  if (head.length > 60) {
    const counts = {};
    for (const ch of head) counts[ch] = (counts[ch] || 0) + 1;
    if (Math.max(...Object.values(counts)) / head.length > 0.4) return true;
  }
  return false;
}

export const sseWrite = (res, obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`);

/**
 * Main chat handler. Streams SSE: meta → delta* → done (or error).
 */
export async function handleChatStream(req, res, form, { requestId } = {}) {
  const message = String(form.get("message") || "").trim();
  if (!message) {
    const e = new Error("الرسالة مطلوبة");
    e.code = "VALIDATION_ERROR";
    throw e;
  }
  const rawMode = String(form.get("mode") || "chat");
  const mode = VALID_MODES.has(rawMode) ? rawMode : "chat";
  const variant = String(form.get("variant") || "v1.1");
  const systemPrompt = String(form.get("system_prompt") || "");
  const thinking = String(form.get("thinking") || "") === "true";
  const temperature = Number(form.get("temperature"));
  const rawMax = form.get("max_tokens");
  const requestedMax = rawMax == null || rawMax === "" ? NaN : Number(rawMax);
  const maxTokensCap = Number.isFinite(requestedMax) ? Math.max(256, Math.min(4096, requestedMax)) : null;
  let history = [];
  try { history = JSON.parse(String(form.get("history") || "[]")); } catch {}

  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no",
    "X-Request-Id": requestId,
  });

  const close = () => { try { res.end(); } catch {} };
  let aborted = false;
  const startedAt = Date.now();
  req.on("close", () => { aborted = true; });

  const useV11 = ENGINE && variant !== "v1.0";
  const localOnly = (variant === "local" || String(form.get("model") || "") === "nados-local") && ENGINE;

  // ---------- local exclusive (trained model only, zero cloud calls) ----------
  if (localOnly) {
    sseWrite(res, { type: "meta", provider: "nados-local", model: `Nados v1.1 Local · ${ENGINE.info().params} معامل (حصري)` });
    const features = extractFeatures(message, message, history.length);
    const prediction = predict(ENGINE.artifact, features);
    if (thinking) {
      sseWrite(res, {
        type: "thinking",
        thinking: [
          "استدلال حقيقي بالشبكة المدرّبة (245 معامل) — بدون سحابة",
          `الدرجة: ${prediction.ranking_score} | التوزيع: ${Object.entries(prediction.probabilities).map(([k, v]) => `${k}=${v}`).join("، ")}`,
        ],
      });
    }
    const answer = synthesizeLocal(message, prediction, mode);
    for (let i = 0; i < answer.length && !aborted; i += 48) sseWrite(res, { type: "delta", delta: answer.slice(i, i + 48) });
    sseWrite(res, {
      type: "done",
      reply: {
        answer: [answer], bullets: [], sources: [],
        provider: "nados-local",
        model: `Nados v1.1 Local (${ENGINE.info().params} معامل)`,
        variant: "local",
        nados: { params: ENGINE.info().params, probabilities: prediction.probabilities, ranking_score: prediction.ranking_score, cloudCalls: 0 },
      },
    });
    return close();
  }

  let nadosDecision = null;
  let styleHint = "";
  let genTemperature = Number.isFinite(temperature) && temperature >= 0 && temperature <= 2 ? temperature : undefined;

  try {
    if (useV11) {
      sseWrite(res, {
        type: "meta",
        provider: "nados-v1.1",
        model: `Nados v1.1 · ${CF_PRIMARY.label} + مُرتّب مدرّب (${ENGINE.info().params} معامل)`,
      });
      const features = extractFeatures(message, message, history.length);
      const prediction = predict(ENGINE.artifact, features);
      const style = styleFromPrediction(prediction);
      nadosDecision = {
        model: "Nados v1.1",
        params: ENGINE.info().params,
        features: Object.fromEntries(Object.entries(features).map(([k, v]) => [k, +v.toFixed(3)])),
        probabilities: prediction.probabilities,
        ranking_score: prediction.ranking_score,
        decision: style.detailHint,
        temperature: style.temperature,
        experimental: true, // trained on a tiny bootstrap set — indicative, not a quality score
      };
      genTemperature = Math.min(style.temperature, 0.3);
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

    const messages = buildMessages({ message, mode, history, systemPrompt, temperature: genTemperature });
    const sysIdx = messages.findIndex((m) => m.role === "system");
    const route = routeTask(message, mode);
    if (maxTokensCap) route.maxTokens = Math.min(route.maxTokens, maxTokensCap);
    if (useV11 && styleHint && sysIdx >= 0) messages[sysIdx].content += "\n" + styleHint;
    const wantsCode = route.tag === "code";
    const isMath = route.tag === "math";
    if (wantsCode && sysIdx >= 0) messages[sysIdx].content += "\nضع أي كود داخل كتلة ``` كاملة وقابلة للتشغيل دائماً، مع شرح سطر واحد بعدها.";
    if (isMath && sysIdx >= 0) messages[sysIdx].content += "\nفي المسائل الحسابية: اكتب كل رقم صراحةً بالأرقام، حل خطوة بخطوة، ثم تحقق من الناتج النهائي بجملة واضحة.";
    if (route.tag === "creative" && sysIdx >= 0) messages[sysIdx].content += "\nممنوع منعاً باتاً ذكر أي كود أو مثال برمجي أو علامات كود (```). أعطِ النص المطلوب فقط (جملة أو رسالة أو قصيدة) وبسطرين كحد أقصى إن لم يُطلب غير ذلك.";

    // live web search (free, keyless)
    let searchResults = null;
    if (mode === "web") {
      try {
        searchResults = await webSearch(message, { lang: "ar", limit: 4 });
        if (searchResults.results.length) {
          messages[messages.length - 1].content += buildContext(searchResults, message);
          if (sysIdx >= 0) messages[sysIdx].content += "\nاستخدم نتائج البحث المرفقة في الإجابة، واذكر المصادر بأرقامها [1] [2].";
        }
      } catch (e) {
        log.warn("web_search.failed", { reason: e.message });
      }
    }

    const ck = cacheKey(route.tag + "|" + mode, messages);
    const cached = cacheGet(ck);
    if (cached) {
      sseWrite(res, { type: "meta", provider: "nados-v1.1", model: `Nados v1.1 · ${cached.teacher} (من الكاش)` });
      for (let i = 0; i < cached.text.length && !aborted; i += 48) sseWrite(res, { type: "delta", delta: cached.text.slice(i, i + 48) });
      sseWrite(res, {
        type: "done",
        reply: {
          answer: [cached.text], bullets: [], sources: cached.sources || [],
          provider: "nados-v1.1", model: `Nados v1.1 · ${cached.teacher} (كاش)`, variant: "v1.1",
          nados: nadosDecision, teacher: cached.teacher, cached: true,
        },
      });
      return close();
    }

    let full = "";
    let teacherUsed = null;
    const hardMode = isMath || wantsCode;

    if (hardMode) {
      const t0 = pickTeacher(route.tag, route.tag === "creative" ? 400 : 900);
      let teacher = t0;
      let upstream;
      try {
        upstream = await callTeacher(t0, messages, genTemperature ?? 0.3, true, { reasoning: route.reasoning, max_tokens: route.maxTokens });
      } catch (e) {
        if (isQuotaError(e) && t0.kind === "cf") { /* marked inside runTeacher */ }
        const fb = await runTeacher(messages, { temperature: genTemperature ?? 0.3, stream: true, reasoning: "low", max_tokens: route.maxTokens });
        teacher = fb.teacher;
        upstream = fb.res;
      }
      teacherUsed = teacher;
      let reasoningBudget = 500;
      let firstContentAt = null;
      for await (const evt of iterateSSE(upstream.body)) {
        if (aborted) break;
        if (evt.reasoning && thinking && reasoningBudget > 0) {
          sseWrite(res, { type: "thinking", thinking: [evt.reasoning] });
          reasoningBudget -= evt.reasoning.length;
        }
        if (evt.content) {
          if (!firstContentAt) firstContentAt = Date.now();
          full += evt.content;
          sseWrite(res, { type: "delta", delta: evt.content });
        }
        if (evt.aggregate && evt.aggregate.length > full.length) {
          const tail = evt.aggregate.slice(full.length);
          full += tail;
          sseWrite(res, { type: "delta", delta: tail });
        }
      }
      log.info("chat.stream", { tag: route.tag, teacher: teacherUsed.label, chars: full.length, ms: Date.now() - startedAt, first_ms: firstContentAt ? firstContentAt - startedAt : null });

      if (isDegenerate(full) && !aborted) {
        log.warn("chat.degenerate", { teacher: teacherUsed.label, chars: full.length });
        full = "";
        try {
          const fb = await runTeacher(messages, { temperature: 0.2, stream: true, reasoning: "low", max_tokens: route.maxTokens });
          teacherUsed = fb.teacher;
          for await (const evt of iterateSSE(fb.res.body)) {
            if (aborted) break;
            if (evt.content) { full += evt.content; sseWrite(res, { type: "delta", delta: evt.content }); }
            if (evt.aggregate && evt.aggregate.length > full.length) {
              const tail = evt.aggregate.slice(full.length);
              full += tail;
              sseWrite(res, { type: "delta", delta: tail });
            }
          }
        } catch (e) {
          log.error("chat.failover_failed", { reason: e.message });
        }
      }

      // validation → corrective pass
      const invalid = (isMath && !/\d/.test(full)) || (wantsCode && !full.includes("```"));
      if (invalid && !aborted) {
        try {
          messages[sysIdx >= 0 ? sysIdx : 0].content += isMath
            ? "\nمطلوب إلزامي: أرقام صريحة في كل خطوة. أعد كتابة الحل كاملاً."
            : "\nمطلوب إلزامي: كتلة كود ``` كاملة. أعد كتابة الإجابة كاملة.";
          const retry = await runTeacher(messages, { temperature: 0.3, reasoning: "low", max_tokens: route.maxTokens });
          const parsed = parseTeacherJSON((await retry.res.json()).result);
          const fixed = stripThinkTags(parsed.content || "");
          if (fixed && fixed.length > 40) {
            const block = `\n\n**تصحيح/إكمال:**\n${fixed}`;
            sseWrite(res, { type: "delta", delta: block });
            full += block;
            teacherUsed = retry.teacher;
          }
        } catch (e) {
          log.warn("chat.corrective_failed", { reason: e.message });
        }
      }

      // independent cross-verification for maths
      if (route.verify && /\d/.test(full)) {
        try {
          const vTeacher = pickTeacher("deep", 400);
          const v = await callTeacher(vTeacher, [
            { role: "system", content: "أنت مدقق حسابي دقيق. تحقق من الحل المقترح خطوة بخطوة.\n- إن كان الحل صحيحاً تماماً: أعد كلمة «صحيح» فقط ولا شيء غيرها.\n- إن كان فيه خطأ: أعد الحل المصحح **بالعربية** مع الأرقام الصريحة في كل خطوة، بدون مقدمات وبدون أي تفكير بالإنجليزية." },
            { role: "user", content: `السؤال: ${message}\n\nالحل المقترح:\n${full}` },
          ], 0.2, false, { max_tokens: 2048, reasoning: vTeacher.style === "openai" ? "low" : undefined });
          const vj = await v.json();
          const vt = stripThinkTags(parseTeacherJSON(vj.result).content || "");
          const arabicChars = (vt.match(/[\u0600-\u06FF]/g) || []).length;
          const latinChars = (vt.match(/[A-Za-z]/g) || []).length;
          const arabicRatio = latinChars ? arabicChars / (arabicChars + latinChars) : 1;
          const firstArabic = vt.search(/[\u0600-\u06FF]/);
          if (firstArabic > 0) vt = vt.slice(firstArabic);
          const looksLikeSolution = /\d/.test(vt) && /(=|سنة|يساوي|الناتج|النتيجة)/.test(vt);
          if (vt && !/^صحيح\.?$/.test(vt.trim()) && /[\u0600-\u06FF]/.test(vt) && arabicRatio > 0.5 && looksLikeSolution && vt.length > 50) {
            const block = `\n\n**تدقيق مستقل (${vTeacher.label}):**\n${vt}`;
            sseWrite(res, { type: "delta", delta: block });
            full += block;
            teacherUsed = vTeacher;
          }
        } catch (e) {
          if (isQuotaError(e)) log.warn("chat.verifier_quota");
        }
      }
    } else {
      const chosen = pickTeacher(route.tag, route.tag === "creative" ? 400 : 800);
      let teacher, upstream;
      try {
        const r = await callTeacher(chosen, messages, route.temperature ?? genTemperature, true, { reasoning: route.reasoning, max_tokens: route.maxTokens });
        teacher = chosen;
        upstream = r;
      } catch (e) {
        log.warn("provider.failed", { backend: chosen.label, reason: e.message });
        const fb = await runTeacher(messages, { temperature: genTemperature, stream: true, reasoning: thinking ? "medium" : "low", max_tokens: route.maxTokens });
        teacher = fb.teacher;
        upstream = fb.res;
      }
      teacherUsed = teacher;
      let reasoningBudget = 600;
      let firstContentAt = null;
      for await (const evt of iterateSSE(upstream.body)) {
        if (aborted) break;
        if (evt.reasoning && thinking && reasoningBudget > 0) {
          sseWrite(res, { type: "thinking", thinking: [evt.reasoning] });
          reasoningBudget -= evt.reasoning.length;
        }
        if (evt.content) {
          if (!firstContentAt) firstContentAt = Date.now();
          full += evt.content;
          sseWrite(res, { type: "delta", delta: evt.content });
        }
        if (evt.aggregate && evt.aggregate.length > full.length) {
          const tail = evt.aggregate.slice(full.length);
          full += tail;
          sseWrite(res, { type: "delta", delta: tail });
        }
      }
      log.info("chat.stream", { tag: route.tag, teacher: teacherUsed.label, chars: full.length, ms: Date.now() - startedAt, first_ms: firstContentAt ? firstContentAt - startedAt : null });

      if (!full.trim() && !aborted) {
        log.warn("chat.empty_content", { teacher: teacherUsed.label });
        try {
          const fb = await runTeacher(messages, { temperature: 0.2, stream: true, reasoning: "low", max_tokens: route.maxTokens });
          teacherUsed = fb.teacher;
          for await (const evt of iterateSSE(fb.res.body)) {
            if (aborted) break;
            if (evt.content) { full += evt.content; sseWrite(res, { type: "delta", delta: evt.content }); }
            if (evt.aggregate && evt.aggregate.length > full.length) {
              const tail = evt.aggregate.slice(full.length);
              full += tail;
              sseWrite(res, { type: "delta", delta: tail });
            }
          }
        } catch (e) {
          log.error("chat.failover_failed", { reason: e.message });
        }
      }
      if (route.tag === "creative") full = stripCodeFences(full);
    }

    let finalNados = nadosDecision;
    if (useV11) {
      const f2 = extractFeatures(message, full, history.length);
      const p2 = predict(ENGINE.artifact, f2);
      finalNados = { ...nadosDecision, final_score: p2.ranking_score, final_probabilities: p2.probabilities };
    }

    const sources = (searchResults?.results || []).map((r) => ({ title: r.title, url: r.url, snippet: r.snippet, source: r.source }));
    sseWrite(res, {
      type: "done",
      reply: {
        answer: [full || "لم أستطع توليد إجابة، حاول مجدداً."],
        bullets: [],
        sources,
        provider: useV11 ? "nados-v1.1" : "cloudflare",
        model: useV11 ? `Nados v1.1 · ${teacherUsed?.label || CF_PRIMARY.label} + مُرتّب مدرّب` : config.cfModel,
        variant: variant === "v1.0" ? "v1.0" : "v1.1",
        nados: finalNados,
        teacher: teacherUsed?.label,
      },
    });
    log.debug("chat.done_sent", { requestId, chars: full.length });

    if (full && !aborted) {
      if (teacherUsed?.kind === "cf" && teacherUsed.active) {
        addNeurons(teacherUsed.accountId, estimateNeurons(full.length / 3.5, teacherUsed.active));
      }
      cacheSet(ck, { text: full, teacher: teacherUsed?.label || "cloud", sources });
    }
  } catch (e) {
    const noCapacity = e?.noCapacity || isQuotaError(e);
    if (!aborted) {
      if (noCapacity) {
        markExhaustedAll(config.cfAccounts);
        if (ENGINE) {
          const features = extractFeatures(message, message, history.length);
          const prediction = predict(ENGINE.artifact, features);
          const local = synthesizeLocal(message, prediction, mode);
          const notice = `> ⚠️ لا تتوفر حالياً سعة توليد سحابية (الحصة اليومية منتهية) — أُجيب الآن بالنموذج المدرّب المحلي (${ENGINE.info().params} معامل) بدون سحابة.\n\n`;
          sseWrite(res, { type: "delta", delta: notice });
          for (let i = 0; i < local.length && !aborted; i += 48) sseWrite(res, { type: "delta", delta: local.slice(i, i + 48) });
          sseWrite(res, {
            type: "done",
            reply: {
              answer: [notice + local], bullets: [], sources: [],
              provider: "nados-local", model: `Nados v1.1 Local (${ENGINE.info().params} معامل)`, variant: "local-fallback",
              nados: { params: ENGINE.info().params, probabilities: prediction.probabilities, ranking_score: prediction.ranking_score, cloudCalls: 0 },
            },
          });
          return close();
        }
      }
      log.error("chat.failed", { requestId, reason: String(e.message || e), code: e.code || null });
      sseWrite(res, { type: "error", message: "تعذّر إكمال الطلب حالياً", requestId });
    }
  }
  log.debug("chat.closing", { requestId, aborted });
  close();
}
