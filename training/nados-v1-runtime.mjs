// Nados V1.0 portable inference runtime.
// Loads the JSON artifact produced by ai/nados-v1/train.py and keeps inference dependency-free.
import fs from 'node:fs';

const FEATURES = ['affinity','quality','freshness','engagement','completion','originality','local','safety','negative'];
const TARGETS = ['like','save','share','comment','completion'];
const clamp = (n, min=0, max=1) => Math.max(min, Math.min(max, n));
const sigmoid = x => 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x))));

export function loadNadosV1(path) {
  const artifact = JSON.parse(fs.readFileSync(path, 'utf8'));
  if (artifact?.version !== '1.0.0' || artifact?.state?.hidden == null) throw new Error('INVALID_NADOS_V1_ARTIFACT');
  return artifact;
}

export function predictNadosV1(artifact, input={}) {
  const state = artifact.state;
  const x = FEATURES.map(k => clamp(Number(input[k] ?? 0)));
  const hidden = state.w1.map((row, i) => Math.tanh(row.reduce((sum, weight, j) => sum + Number(weight) * x[j], 0) + Number(state.b1[i] || 0)));
  const probabilities = state.w2.map((row, i) => sigmoid(row.reduce((sum, weight, j) => sum + Number(weight) * hidden[j], 0) + Number(state.b2[i] || 0)));
  const targets = Object.fromEntries(TARGETS.map((k, i) => [k, probabilities[i]]));
  const score = clamp(targets.like*.24 + targets.save*.20 + targets.share*.20 + targets.comment*.16 + targets.completion*.20);
  return { model: artifact.name, version: artifact.version, probabilities: targets, ranking_score: score };
}

export function nadosV1Score(artifact, input={}) {
  return predictNadosV1(artifact, input).ranking_score;
}

export const NADOS_V1_FEATURES = Object.freeze(FEATURES);
export const NADOS_V1_TARGETS = Object.freeze(TARGETS);
