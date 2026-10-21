// Unit tests for the trained Nados engine (245 parameters).
import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createEngine, predict, extractFeatures, styleFromPrediction, loadArtifact } from "../nados-engine.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ARTIFACT = path.join(__dirname, "..", "training", "artifacts", "nados-v1.1.0.json");

test("artifact loads and exposes the trained weights", () => {
  const a = loadArtifact(ARTIFACT);
  assert.equal(a.state.w1.length, 16, "hidden layer = 16");
  assert.equal(a.state.w1[0].length, 9, "9 input features");
  assert.equal(a.state.w2.length, 5, "5 output targets");
  assert.equal(a.state.w2[0].length, 16);
  assert.ok(a.metrics, "metrics present");
});

test("the trained model has exactly 245 parameters", () => {
  const engine = createEngine({ artifactPath: ARTIFACT });
  assert.equal(engine.info().params, 245);
  // manual count must match
  const s = engine.artifact.state;
  const manual = s.w1.length * s.w1[0].length + s.b1.length + s.w2.length * s.w2[0].length + s.b2.length;
  assert.equal(manual, 245);
});

test("prediction is deterministic for the same input", () => {
  const engine = createEngine({ artifactPath: ARTIFACT });
  const features = { affinity: 0.5, quality: 0.8, freshness: 0.4, engagement: 0.6, completion: 0.9, originality: 0.7, local: 1, safety: 1, negative: 0 };
  const a = predict(engine.artifact, features);
  const b = predict(engine.artifact, features);
  assert.deepEqual(a.probabilities, b.probabilities);
  assert.equal(a.ranking_score, b.ranking_score);
});

test("probabilities stay inside [0,1] and the score is a valid weighted sum", () => {
  const engine = createEngine({ artifactPath: ARTIFACT });
  const p = predict(engine.artifact, { affinity: 1, quality: 1, freshness: 1, engagement: 1, completion: 1, originality: 1, local: 1, safety: 1, negative: 1 });
  for (const v of Object.values(p.probabilities)) {
    assert.ok(v >= 0 && v <= 1, `probability ${v} in range`);
  }
  const expected = p.probabilities.like * 0.24 + p.probabilities.save * 0.2 + p.probabilities.share * 0.2 +
    p.probabilities.comment * 0.16 + p.probabilities.completion * 0.2;
  assert.ok(Math.abs(p.ranking_score - expected) < 1e-3);
});

test("features are clamped to [0,1] and always have the 9 dimensions", () => {
  const f = extractFeatures("سؤال", "إجابة", 0);
  const keys = ["affinity", "quality", "freshness", "engagement", "completion", "originality", "local", "safety", "negative"];
  assert.equal(Object.keys(f).length, 9);
  for (const k of keys) {
    assert.ok(f[k] >= 0 && f[k] <= 1, `${k} in range`);
  }
});

test("an irrelevant candidate scores lower than a relevant one", () => {
  const q = "اشرح كيف تعمل الذاكرة الافتراضية في أنظمة التشغيل";
  const good = extractFeatures(q, "الذاكرة الافتراضية تخزن عناوين افتراضية وتترجمها إلى صفحات فعلية عند الحاجة.", 0);
  const bad = extractFeatures(q, "القمر جرم سماوي يدور حول الأرض.", 0);
  const engine = createEngine({ artifactPath: ARTIFACT });
  const scoreGood = predict(engine.artifact, good).ranking_score;
  const scoreBad = predict(engine.artifact, bad).ranking_score;
  assert.ok(scoreGood > scoreBad, `relevant (${scoreGood}) should beat irrelevant (${scoreBad})`);
});

test("style mapping returns a sane temperature and a hint", () => {
  const engine = createEngine({ artifactPath: ARTIFACT });
  const p = predict(engine.artifact, { quality: 0.9, completion: 0.9, share: 0.9 });
  const style = styleFromPrediction(p);
  assert.ok(style.temperature >= 0.3 && style.temperature <= 1, `temperature ${style.temperature}`);
  assert.ok(typeof style.detailHint === "string" && style.detailHint.length > 10);
});

test("invalid artifact paths are rejected instead of crashing", () => {
  assert.throws(() => loadArtifact(path.join(__dirname, "does-not-exist.json")));
});
