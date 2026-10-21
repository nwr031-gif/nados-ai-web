// Free, keyless web search for Nados: Wikipedia REST API (ar+en) + DuckDuckGo Instant Answer.
// No quota, no API keys, no local compute.
const WIKI = (lang, q, limit = 3) =>
  `https://${lang}.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(
    q
  )}&format=json&srlimit=${limit}&origin=*`;

const DDG = (q) =>
  `https://api.duckduckgo.com/?q=${encodeURIComponent(q)}&format=json&no_html=1&skip_disambig=1`;

// short-lived cache so repeated questions do not hammer the public APIs
const CACHE_TTL_MS = 5 * 60 * 1000;
const cache = new Map();

function cacheGet(key) {
  const hit = cache.get(key);
  if (!hit) return null;
  if (Date.now() - hit.at > CACHE_TTL_MS) { cache.delete(key); return null; }
  return hit.value;
}
function cacheSet(key, value) {
  cache.set(key, { value, at: Date.now() });
  if (cache.size > 50) cache.delete(cache.keys().next().value);
}

function stripHtml(s) {
  return String(s || "")
    .replace(/<[^>]*>/g, "")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/\s+/g, " ")
    .trim();
}

async function jget(url, timeoutMs = 12000) {
  const res = await fetch(url, { headers: { "User-Agent": "NadosAI/1.0 (local)" }, signal: AbortSignal.timeout(timeoutMs) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function webSearch(query, { lang = "ar", limit = 4 } = {}) {
  const cacheKey = `${lang}|${limit}|${query}`;
  const cached = cacheGet(cacheKey);
  if (cached) return cached;

  const results = [];
  const errors = [];

  // 1) Wikipedia (Arabic first, then English)
  for (const l of [lang, "en"]) {
    try {
      const d = await jget(WIKI(l, query, 3));
      for (const r of d?.query?.search || []) {
        results.push({
          title: r.title,
          snippet: stripHtml(r.snippet),
          url: `https://${l}.wikipedia.org/wiki/${encodeURIComponent(r.title.replace(/ /g, "_"))}`,
          source: `wikipedia:${l}`,
        });
      }
    } catch (e) {
      errors.push(`wiki-${l}: ${e.message}`);
    }
    if (results.length >= limit) break;
  }

  // 2) DuckDuckGo instant answer (facts, not just links)
  try {
    const d = await jget(DDG(query));
    if (d?.AbstractText) {
      results.unshift({
        title: d.Heading || query,
        snippet: stripHtml(d.AbstractText),
        url: d.AbstractURL || `https://duckduckgo.com/?q=${encodeURIComponent(query)}`,
        source: "duckduckgo",
      });
    }
    for (const t of d?.RelatedTopics || []) {
      if (results.length >= limit) break;
      if (t.Text) {
        results.push({
          title: stripHtml(t.Text).split(" - ")[0].slice(0, 90),
          snippet: stripHtml(t.Text),
          url: t.FirstURL || `https://duckduckgo.com/?q=${encodeURIComponent(query)}`,
          source: "duckduckgo",
        });
      }
    }
  } catch (e) {
    errors.push(`ddg: ${e.message}`);
  }

  const unique = [];
  const seen = new Set();
  for (const r of results) {
    const key = (r.title || "").toLowerCase();
    if (!key || seen.has(key)) continue;
    seen.add(key);
    unique.push(r);
    if (unique.length >= limit) break;
  }

  const out = { query, results: unique, errors, at: new Date().toISOString() };
  cacheSet(cacheKey, out);
  return out;
}

export function buildContext(search, query) {
  if (!search?.results?.length) return "";
  const lines = search.results.map((r, i) => `[${i + 1}] ${r.title}: ${r.snippet}`);
  return `\n\nنتائج البحث الحي عن «${query}» (استخدمها للإجابة مع ذكر المصدر):\n${lines.join("\n")}`;
}
