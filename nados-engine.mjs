// Nados v1.1 chat engine — runs the TRAINED 245-parameter model in the chat loop.
// Pipeline: message features -> NadosRanker (245 params, real inference)
//           -> adapt style -> generate candidates -> rank candidates with the trained net -> stream winner.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const FEATURES = ["affinity","quality","freshness","engagement","completion","originality","local","safety","negative"];
const TARGETS = ["like","save","share","comment","completion"];
const clamp = (n, min = 0, max = 1) => Math.max(min, Math.min(max, n));
const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));

export function loadArtifact(p) {
  const artifact = JSON.parse(fs.readFileSync(p, "utf8"));
  const s = artifact.model || artifact.state;
  if (!s?.w1?.length) throw new Error("INVALID_ARTIFACT");
  return {
    name: artifact.name,
    version: artifact.version || "1.1.0",
    state: s,
    metrics: artifact.metrics || null,
    paramCount:
      s.w1.length * s.w1[0].length + s.b1.length + s.w2.length * s.w2[0].length + s.b2.length,
  };
}

// Real 245-parameter forward pass.
export function predict(artifact, input = {}) {
  const state = artifact.state;
  const x = FEATURES.map((k) => clamp(Number(input[k] ?? 0)));
  const hidden = state.w1.map((row, i) =>
    Math.tanh(row.reduce((sum, w, j) => sum + Number(w) * x[j], 0) + Number(state.b1[i] || 0))
  );
  const probs = state.w2.map((row, i) =>
    sigmoid(row.reduce((sum, w, j) => sum + Number(w) * hidden[j], 0) + Number(state.b2[i] || 0))
  );
  const probabilities = Object.fromEntries(TARGETS.map((k, i) => [k, +probs[i].toFixed(4)]));
  const ranking_score = +clamp(
    probabilities.like * 0.24 + probabilities.save * 0.2 + probabilities.share * 0.2 +
      probabilities.comment * 0.16 + probabilities.completion * 0.2
  ).toFixed(4);
  return { model: artifact.name, version: artifact.version, probabilities, ranking_score };
}

const AR_RE = /[\u0600-\u06FF]/g;
const STOP = new Set(["في","من","على","عن","إلى","الى","ما","هو","هي","هذا","هذه","the","a","an","of","and","to","is","are","مع","او","أو","كل","عند","بين","ثم","كما","قد","لا","أن","ان","إن","يا","هل","كيف","ماذا","لماذا","شنو","شلون"]);

function tokenize(t) {
  return String(t).toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((w) => w.length > 1 && !STOP.has(w));
}

// Heuristic 0..1 features for a (message, candidate) pair — input to the trained net.
export function extractFeatures(message, candidate, historyLen = 0) {
  const c = String(candidate);
  const m = String(message);
  const len = c.length;
  const qTokens = [...new Set(tokenize(m))];
  const cTokens = tokenize(c);
  const cSet = new Set(cTokens);
  const overlap = qTokens.length ? qTokens.filter((t) => cSet.has(t)).length / qTokens.length : 0.4;
  const hasStructure = /(^|\n)\s*([-*•]|\d+[.)])/.test(c) || /\n#{1,3}\s|\*\*/.test(c);
  const sentences = m.split(/[.؟?!،,]+/).filter((s) => s.trim().length > 2).length || 1;
  const coveredParts = qTokens.filter((t) => cSet.has(t)).length;
  const arChars = (c.match(AR_RE) || []).length;
  const uniqueRatio = cTokens.length ? new Set(cTokens).size / cTokens.length : 0.5;
  const repeats = cTokens.length - new Set(cTokens).size;
  return {
    affinity: clamp(overlap * 1.4),
    quality: clamp((hasStructure ? 0.55 : 0.25) + clamp((len - 100) / 900) * 0.45),
    freshness: clamp(0.6 + (len > 250 ? 0.2 : 0) - (repeats > 12 ? 0.3 : 0)),
    engagement: clamp(0.35 + (/[?؟]\s*$/.test(c.trim()) ? 0.25 : 0) + (/مثال|لنقل|تخيل/.test(c) ? 0.2 : 0) + clamp(historyLen / 20) * 0.2),
    completion: clamp(0.3 + (coveredParts / Math.max(1, qTokens.length)) * 0.7),
    originality: clamp(uniqueRatio * (len > 150 ? 1.1 : 0.8)),
    local: clamp(arChars / Math.max(1, len) * 1.3),
    safety: clamp(1 - ((c.match(/لا أستطيع|عذرا، لا/g) || []).length > 1 ? 0.4 : 0)),
    negative: clamp((len < 60 ? 0.6 : 0) + (repeats > 20 ? 0.4 : 0)),
  };
}

export function styleFromPrediction(prediction) {
  const p = prediction.probabilities;
  const detailed = p.completion > 0.45 || p.save > 0.45;
  return {
    temperature: +(clamp(0.45 + p.share * 0.5, 0.3, 1.0).toFixed(2)),
    detailHint: detailed
      ? "أجب بإجابة وافية منظمة بعناوين أو نقاط مع مثال عملي قصير."
      : "أجب بإجابة موجزة مباشرة في فقرات قصيرة.",
  };
}

export function createEngine({ artifactPath }) {
  const artifact = loadArtifact(artifactPath);
  return {
    artifact,
    info() {
      return {
        name: `${artifact.name} v${artifact.version}`.replace(" v1.1.0", " v1.1"),
        params: artifact.paramCount,
        metrics: artifact.metrics,
      };
    },
    rank(message, candidates, historyLen = 0) {
      return candidates.map((text, i) => {
        const features = extractFeatures(message, text, historyLen);
        const prediction = predict(artifact, features);
        return { i, text, features, prediction };
      });
    },
  };
}
