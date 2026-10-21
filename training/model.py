import math
import random

FEATURES = ["affinity","quality","freshness","engagement","completion","originality","local","safety","negative"]
TARGETS = ["like","save","share","comment","completion"]


def sigmoid(x):
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def vectorize(features):
    return [max(0.0, min(1.0, float(features.get(k, 0.0)))) for k in FEATURES]


class NadosRanker:
    """Small dependency-free multi-target neural ranker for Nados V1.0."""
    def __init__(self, hidden=16, seed=42):
        rng = random.Random(seed)
        self.hidden = hidden
        self.w1 = [[rng.uniform(-0.15, 0.15) for _ in FEATURES] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.uniform(-0.15, 0.15) for _ in range(hidden)] for _ in TARGETS]
        self.b2 = [0.0] * len(TARGETS)

    def forward(self, x):
        h = [math.tanh(sum(w*v for w,v in zip(row,x)) + b) for row,b in zip(self.w1,self.b1)]
        y = [sigmoid(sum(w*v for w,v in zip(row,h)) + b) for row,b in zip(self.w2,self.b2)]
        return h, y

    def predict(self, features):
        _, y = self.forward(vectorize(features))
        return dict(zip(TARGETS, y))

    def score(self, features):
        p = self.predict(features)
        return 0.24*p["like"] + 0.20*p["save"] + 0.20*p["share"] + 0.16*p["comment"] + 0.20*p["completion"]

    def fit(self, rows, epochs=250, lr=0.03):
        for _ in range(epochs):
            for row in rows:
                x = vectorize(row["features"])
                target = [float(row["targets"].get(k, 0.0)) for k in TARGETS]
                h, y = self.forward(x)
                dz2 = [(p-t) for p,t in zip(y,target)]
                dz1 = [sum(self.w2[j][i]*dz2[j] for j in range(len(TARGETS))) * (1-h[i]*h[i]) for i in range(self.hidden)]
                for j in range(len(TARGETS)):
                    for i in range(self.hidden): self.w2[j][i] -= lr*dz2[j]*h[i]
                    self.b2[j] -= lr*dz2[j]
                for i in range(self.hidden):
                    for k in range(len(FEATURES)): self.w1[i][k] -= lr*dz1[i]*x[k]
                    self.b1[i] -= lr*dz1[i]
        return self

    def state_dict(self):
        return {"features":FEATURES,"targets":TARGETS,"hidden":self.hidden,"w1":self.w1,"b1":self.b1,"w2":self.w2,"b2":self.b2}

    @classmethod
    def from_state(cls, state):
        m=cls(hidden=int(state["hidden"]))
        m.w1=state["w1"];m.b1=state["b1"];m.w2=state["w2"];m.b2=state["b2"]
        return m
