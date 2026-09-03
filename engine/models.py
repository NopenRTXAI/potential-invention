"""CPU-friendly model architectures (pure NumPy).

All models implement the same minimal interface:
    forward(X)            -> logits
    backward(grad)        -> parameter grads (dict name -> ndarray)
    params / grads        -> dict of ndarrays (shared keys)
    state_dict() / load_state_dict()

Architectures: MLP, CNN (1 conv layer), GRU, TinyTransformer.
The training procedure (not the architecture) is the research subject,
so architectures are deliberately small and standard.
"""

import numpy as np


# ---------------------------------------------------------------- primitives

class Dense:
    def __init__(self, in_dim, out_dim, seed=None):
        rng = np.random.RandomState(seed)
        bound = np.sqrt(6.0 / (in_dim + out_dim))  # Glorot uniform
        self.W = rng.uniform(-bound, bound, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = None

    def forward(self, X):
        self._cache = X
        return X @ self.W + self.b

    def backward(self, dlogits):
        X = self._cache
        self.dW = X.T @ dlogits
        self.db = dlogits.sum(axis=0)
        return dlogits @ self.W.T

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    def set_param(self, name, value):
        setattr(self, name, value)


class ReLU:
    def forward(self, X):
        self._mask = X > 0
        return X * self._mask

    def backward(self, d):
        return d * self._mask


class Tanh:
    def forward(self, X):
        self._out = np.tanh(X)
        return self._out

    def backward(self, d):
        return d * (1.0 - self._out ** 2)


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """Returns (mean loss, dlogits). Numerically stable."""
    probs = softmax(logits)
    p = probs[np.arange(len(y)), y]
    loss = -np.mean(np.log(np.clip(p, 1e-12, None)))
    d = probs.copy()
    d[np.arange(len(y)), y] -= 1.0
    return float(loss), d / len(y)


# ---------------------------------------------------------------- base model

class BaseModel:
    def params(self):
        raise NotImplementedError

    def state_dict(self):
        out = {}
        for k, v in self.params().items():
            out[k] = v.copy()
        return out

    def load_state_dict(self, sd):
        for k, v in sd.items():
            if k in self.params():
                self.set_param(k, v)

    def predict_proba(self, X):
        return softmax(self.forward(X))

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def param_count(self):
        return int(sum(v.size for v in self.params().values()))


class MLP(BaseModel):
    """2-hidden-layer MLP. hidden dims scale with input size (CPU-friendly)."""

    def __init__(self, in_dim, n_classes, hidden=(128, 64), seed=0):
        rng = np.random.RandomState(seed)
        h1, h2 = hidden
        self.l1 = Dense(in_dim, h1, seed=rng.randint(1 << 30))
        self.r1 = ReLU()
        self.l2 = Dense(h1, h2, seed=rng.randint(1 << 30))
        self.r2 = ReLU()
        self.l3 = Dense(h2, n_classes, seed=rng.randint(1 << 30))

    def forward(self, X):
        self._a1 = self.r1.forward(self.l1.forward(X))
        self._a2 = self.r2.forward(self.l2.forward(self._a1))
        return self.l3.forward(self._a2)

    def backward(self, dlogits):
        d = self.l3.backward(dlogits)
        d = self.r2.backward(d)
        d = self.l2.backward(d)
        d = self.r1.backward(d)
        d = self.l1.backward(d)
        return d

    def params(self):
        return {f"l{i}.{k}": v for i, l in enumerate((self.l1, self.l2, self.l3), 1)
                for k, v in l.params().items()}

    def grads(self):
        return {f"l{i}.{k}": v for i, l in enumerate((self.l1, self.l2, self.l3), 1)
                for k, v in l.grads().items()}

    def set_param(self, name, value):
        layer_name, param_name = name.split(".", 1)
        getattr(self, layer_name).set_param(param_name, value)


class GRU(BaseModel):
    """Single-layer GRU over feature sequences.

    For fixed-size feature vectors we treat the vector as a short sequence
    of chunks — a CPU-friendly way to get recurrent dynamics on tabular data.
    """

    def __init__(self, in_dim, n_classes, hidden=64, chunks=4, seed=0):
        rng = np.random.RandomState(seed)
        self.hidden = hidden
        self.chunks = chunks
        self.step = in_dim // chunks
        self.pad = self.step * chunks - in_dim
        if self.pad < 0:
            self.step = in_dim
            self.chunks = 1
            self.pad = 0
        b = np.sqrt(2.0 / (self.step + hidden))
        self.Wz = rng.uniform(-b, b, (self.step, hidden)).astype(np.float32)
        self.Uz = rng.uniform(-b, b, (hidden, hidden)).astype(np.float32)
        self.Wr = rng.uniform(-b, b, (self.step, hidden)).astype(np.float32)
        self.Ur = rng.uniform(-b, b, (hidden, hidden)).astype(np.float32)
        self.Wh = rng.uniform(-b, b, (self.step, hidden)).astype(np.float32)
        self.Uh = rng.uniform(-b, b, (hidden, hidden)).astype(np.float32)
        self.Wy = rng.uniform(-b, b, (hidden, n_classes)).astype(np.float32)
        self.by = np.zeros(n_classes, dtype=np.float32)
        self._grads = {k: np.zeros_like(v) for k, v in self._param_dict().items()}

    def _param_dict(self):
        return {"Wz": self.Wz, "Uz": self.Uz, "Wr": self.Wr, "Ur": self.Ur,
                "Wh": self.Wh, "Uh": self.Uh, "Wy": self.Wy, "by": self.by}

    def forward(self, X):
        if self.pad:
            X = np.pad(X, ((0, 0), (0, -self.pad)))
        seq = X.reshape(len(X), self.chunks, self.step)
        h = np.zeros((len(X), self.hidden), dtype=np.float32)
        self._cache = []
        for t in range(self.chunks):
            xt = seq[:, t, :]
            z = self._sig(xt @ self.Wz + h @ self.Uz)
            r = self._sig(xt @ self.Wr + h @ self.Ur)
            cand = np.tanh(xt @ self.Wh + (r * h) @ self.Uh)
            h_new = (1 - z) * cand + z * h
            self._cache.append((xt, h, z, r, cand))
            h = h_new
        return h @ self.Wy + self.by

    def backward(self, dlogits):
        """Backpropagation through time over the chunk sequence."""
        g = {k: np.zeros_like(v) for k, v in self._param_dict().items()}
        dWy = self._cache[-1][1].T @ dlogits
        dby = dlogits.sum(axis=0)
        g["Wy"], g["by"] = dWy, dby
        dh = dlogits @ self.Wy.T  # (B, H) grad wrt final h
        for t in reversed(range(self.chunks)):
            xt, h_prev, z, r, cand = self._cache[t]
            # h_new = (1-z)*cand + z*h_prev
            dcand = dh * (1 - z)
            dh_prev = dh * z  # direct path to previous h
            dz = dh * (h_prev - cand)
            dpre_z = dz * z * (1 - z)  # sigmoid grad
            g["Wz"] += xt.T @ dpre_z
            g["Uz"] += h_prev.T @ dpre_z
            # cand = tanh(xt@Wh + (r*h_prev)@Uh)
            dtanh = dcand * (1 - cand ** 2)
            g["Wh"] += xt.T @ dtanh
            drh = dtanh @ self.Uh.T  # grad wrt (r*h_prev)
            dr = drh * h_prev
            dh_prev += drh * r
            dpre_r = dr * r * (1 - r)
            g["Wr"] += xt.T @ dpre_r
            g["Ur"] += h_prev.T @ dpre_r
            dh = dh_prev
        self._grads = g
        return None

    def grads(self):
        return self._grads

    def params(self):
        return self._param_dict()

    def set_param(self, name, value):
        setattr(self, name, value)

    @staticmethod
    def _sig(x):
        return 1.0 / (1.0 + np.exp(-x))


class TinyTransformer(BaseModel):
    """1-layer, single-head transformer over feature chunks (CPU-tiny)."""

    def __init__(self, in_dim, n_classes, d_model=64, chunks=8, seed=0):
        rng = np.random.RandomState(seed)
        self.d = d_model
        self.chunks = chunks
        self.step = max(1, in_dim // chunks)
        self.real_in = self.step * chunks
        self.embed_scale = np.sqrt(1.0 / self.step)
        b = np.sqrt(2.0 / (self.step + d_model))
        self.Wemb = rng.uniform(-b, b, (self.step, d_model)).astype(np.float32)
        self.pos = rng.normal(0, 0.02, (chunks, d_model)).astype(np.float32)
        self.Wq = rng.uniform(-b, b, (d_model, d_model)).astype(np.float32)
        self.Wk = rng.uniform(-b, b, (d_model, d_model)).astype(np.float32)
        self.Wv = rng.uniform(-b, b, (d_model, d_model)).astype(np.float32)
        self.Wo = rng.uniform(-b, b, (d_model, d_model)).astype(np.float32)
        self.W1 = rng.uniform(-b, b, (d_model, 2 * d_model)).astype(np.float32)
        self.b1 = np.zeros(2 * d_model, dtype=np.float32)
        self.W2 = rng.uniform(-b, b, (2 * d_model, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)
        self.Wout = rng.uniform(-b, b, (d_model, n_classes)).astype(np.float32)
        self.bout = np.zeros(n_classes, dtype=np.float32)
        self._grads = {}

    def params(self):
        return {"Wemb": self.Wemb, "pos": self.pos, "Wq": self.Wq, "Wk": self.Wk,
                "Wv": self.Wv, "Wo": self.Wo, "W1": self.W1, "b1": self.b1,
                "W2": self.W2, "b2": self.b2,
                "Wout": self.Wout, "bout": self.bout}

    def grads(self):
        return self._grads

    def set_param(self, name, value):
        setattr(self, name, value)

    def forward(self, X):
        B = len(X)
        pad = self.real_in - X.shape[1]
        if pad > 0:
            X = np.pad(X, ((0, 0), (0, pad)))
        elif pad < 0:
            X = X[:, :self.real_in]
        seq = X.reshape(B, self.chunks, self.step)  # (B, T, S)
        E = seq @ self.Wemb * self.embed_scale + self.pos  # (B, T, D)
        Q = E @ self.Wq
        K = E @ self.Wk
        V = E @ self.Wv
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d)  # (B, T, T)
        scores = scores - scores.max(axis=-1, keepdims=True)
        A = np.exp(scores)
        A = A / A.sum(axis=-1, keepdims=True)
        C = A @ V  # (B, T, D)
        H = C @ self.Wo
        Hres = H + E
        Z1 = Hres @ self.W1 + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = A1 @ self.W2 + self.b2
        M = Z2 + Hres
        pooled = M.mean(axis=1)  # mean over positions
        out = pooled @ self.Wout + self.bout
        self._cache = (seq, E, Q, K, V, A, C, H, Hres, Z1, A1, Z2, M, pooled)
        return out

    def backward(self, dlogits):
        (seq, E, Q, K, V, A, C, H, Hres, Z1, A1, Z2, M, pooled) = self._cache
        B, T, _ = E.shape
        g = {k: np.zeros_like(v) for k, v in self.params().items()}

        g["Wout"] = pooled.T @ dlogits
        g["bout"] = dlogits.sum(axis=0)
        dpooled = dlogits @ self.Wout.T  # (B, D)
        dM = np.repeat(dpooled[:, None, :], T, axis=1) / T  # (B, T, D)

        dZ2 = dM
        g["W2"] += (A1.reshape(B * T, -1).T @ dZ2.reshape(B * T, -1)).reshape(g["W2"].shape)
        g["b2"] += dZ2.sum(axis=(0, 1))
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        g["W1"] += (Hres.reshape(B * T, -1).T @ dZ1.reshape(B * T, -1)).reshape(g["W1"].shape)
        g["b1"] += dZ1.sum(axis=(0, 1))
        dHres = dZ1 @ self.W1.T + dM

        dH = dHres.copy()
        dC = dH @ self.Wo.T
        g["Wo"] += (C.reshape(B * T, -1).T @ dH.reshape(B * T, -1)).reshape(g["Wo"].shape)
        dV = A.transpose(0, 2, 1) @ dC
        dA = dC @ V.transpose(0, 2, 1)
        dSc = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        dQ = dSc @ K / np.sqrt(self.d)
        dK = (dSc.transpose(0, 2, 1) @ Q) / np.sqrt(self.d)
        g["Wq"] += (E.reshape(B * T, -1).T @ dQ.reshape(B * T, -1)).reshape(g["Wq"].shape)
        g["Wk"] += (E.reshape(B * T, -1).T @ dK.reshape(B * T, -1)).reshape(g["Wk"].shape)
        g["Wv"] += (E.reshape(B * T, -1).T @ dV.reshape(B * T, -1)).reshape(g["Wv"].shape)
        dE = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T + dHres
        g["pos"] += dE.sum(axis=0)
        dSeq = (dE @ self.Wemb.T * self.embed_scale)  # (B, T, S)
        g["Wemb"] += (seq.reshape(B * T, -1).T @ dE.reshape(B * T, -1)).reshape(g["Wemb"].shape)
        self._grads = g
        return dSeq.reshape(B, -1)


ARCHITECTURES = {
    "mlp": MLP,
    "gru": GRU,
    "tiny_transformer": TinyTransformer,
}


def build_model(arch, in_dim, n_classes, seed=0, **kwargs):
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{arch}'")
    return ARCHITECTURES[arch](in_dim, n_classes, seed=seed, **kwargs)
