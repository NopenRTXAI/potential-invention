"""TinyGPT — a real decoder-only transformer (GPT) in pure NumPy.

This is a genuine language model: token embeddings, causal self-attention,
feed-forward blocks, pre-LayerNorm residual architecture (GPT-2 style),
weight-tied output head, AdamW training, and autoregressive sampling.

It is small (a few million parameters) because it must train on a CPU —
but every part of it is real: real gradients, real training, real generation.
"""

import numpy as np


# ================================================================ layers

class LayerNorm:
    """LayerNorm with forward/backward."""

    @staticmethod
    def forward(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdims=True)
        std = np.sqrt(var + eps)
        xn = xc / std
        y = xn * g + b
        return y, (mu, std, xn)

    @staticmethod
    def backward(dout, cache, g, b):
        """dout: (B,T,d) -> dx: (B,T,d), dg, db: (d,)."""
        mu, std, xn = cache
        N = dout.shape[-1]
        dxhat = dout * g
        dstd2 = (dxhat * xn).sum(-1, keepdims=True)
        dx = dxhat / std \
            - dxhat.sum(-1, keepdims=True) / (N * std) \
            - xn * dstd2 / (N * std)
        dg = (dout * xn).sum(axis=(0, 1))
        db = dout.sum(axis=(0, 1))
        return dx, dg, db


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ================================================================ model

class TinyGPT:
    """Decoder-only transformer.

    Params (all in self.p, name -> array):
        tok_emb   (V, d)      token embeddings (also the output head)
        pos_emb   (T, d)      positional embeddings
        l{i}.ln1.g/b          pre-attention layernorm
        l{i}.Wqkv  (d, 3d)    fused qkv projection
        l{i}.bqkv  (3d,)
        l{i}.Wo    (d, d)     attention output proj
        l{i}.bo    (d,)
        l{i}.ln2.g/b          pre-mlp layernorm
        l{i}.W1     (d, 4d)   mlp up
        l{i}.b1     (4d,)
        l{i}.W2     (4d, d)   mlp down
        l{i}.b2     (d,)
        lnf.g/b               final layernorm
    """

    def __init__(self, vocab_size, d_model=192, n_layers=4, n_heads=4,
                 block=96, seed=0):
        rng = np.random.default_rng(seed)
        self.V = vocab_size
        self.d = d_model
        self.L = n_layers
        self.H = n_heads
        self.T = block
        self.hd = d_model // n_heads
        p = {}
        p["tok_emb"] = (rng.standard_normal((vocab_size, d_model))
                        * 0.02).astype(np.float32)
        p["pos_emb"] = (rng.standard_normal((block, d_model))
                        * 0.02).astype(np.float32)
        for i in range(n_layers):
            pre = f"l{i}"
            p[f"{pre}.ln1.g"] = np.ones(d_model, dtype=np.float32)
            p[f"{pre}.ln1.b"] = np.zeros(d_model, dtype=np.float32)
            p[f"{pre}.Wqkv"] = (rng.standard_normal((d_model, 3 * d_model))
                                * (1 / np.sqrt(d_model))).astype(np.float32)
            p[f"{pre}.bqkv"] = np.zeros(3 * d_model, dtype=np.float32)
            p[f"{pre}.Wo"] = (rng.standard_normal((d_model, d_model))
                              * (1 / np.sqrt(d_model))).astype(np.float32)
            p[f"{pre}.bo"] = np.zeros(d_model, dtype=np.float32)
            p[f"{pre}.ln2.g"] = np.ones(d_model, dtype=np.float32)
            p[f"{pre}.ln2.b"] = np.zeros(d_model, dtype=np.float32)
            p[f"{pre}.W1"] = (rng.standard_normal((d_model, 4 * d_model))
                              * (1 / np.sqrt(d_model))).astype(np.float32)
            p[f"{pre}.b1"] = np.zeros(4 * d_model, dtype=np.float32)
            p[f"{pre}.W2"] = (rng.standard_normal((4 * d_model, d_model))
                              * (1 / np.sqrt(4 * d_model))).astype(np.float32)
            p[f"{pre}.b2"] = np.zeros(d_model, dtype=np.float32)
        p["lnf.g"] = np.ones(d_model, dtype=np.float32)
        p["lnf.b"] = np.zeros(d_model, dtype=np.float32)
        self.p = p
        self._mask = np.triu(np.full((block, block), -1e10, dtype=np.float32),
                             k=1)

    # ------------------------------------------------------------ forward

    def forward(self, idx):
        """idx: (B, T) int tokens -> logits (B, T, V). Caches for backward."""
        B, T = idx.shape
        p = self.p
        x = p["tok_emb"][idx] + p["pos_emb"][:T]          # (B, T, d)
        caches = [{} for _ in range(self.L)]
        for i in range(self.L):
            pre = f"l{i}"
            c = caches[i]
            # ---- attention block (pre-LN residual)
            h1n, ln1_cache = LayerNorm.forward(x, p[f"{pre}.ln1.g"],
                                               p[f"{pre}.ln1.b"])
            qkv = h1n @ p[f"{pre}.Wqkv"] + p[f"{pre}.bqkv"]  # (B,T,3d)
            q, k, v = np.split(qkv, 3, axis=-1)
            # reshape to heads
            qh = np.ascontiguousarray(
                q.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3))
            kh = np.ascontiguousarray(
                k.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3))
            vh = np.ascontiguousarray(
                v.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3))
            att = qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(self.hd)
            att = att + self._mask[:T, :T]
            A = softmax(att)                                # (B,H,T,T)
            outh = A @ vh                                  # (B,H,T,hd)
            outh = outh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            attn_out = outh @ p[f"{pre}.Wo"] + p[f"{pre}.bo"]
            x = x + attn_out
            # ---- mlp block (pre-LN residual)
            h2n, ln2_cache = LayerNorm.forward(x, p[f"{pre}.ln2.g"],
                                               p[f"{pre}.ln2.b"])
            h1 = h2n @ p[f"{pre}.W1"] + p[f"{pre}.b1"]      # (B,T,4d)
            act = np.tanh(h1)
            mlp_out = act @ p[f"{pre}.W2"] + p[f"{pre}.b2"]
            x = x + mlp_out
            c.update(ln1=ln1_cache, ln2=ln2_cache, xn1=h1n, xn2=h2n,
                     qkv=qkv, qh=qh, kh=kh, vh=vh, A=A, outh=outh,
                     act=act, h1=h1)
        xf, lnf_cache = LayerNorm.forward(x, p["lnf.g"], p["lnf.b"])
        logits = xf @ p["tok_emb"].T                        # weight-tied head
        return logits, (x, xf, lnf_cache, caches)

    # ------------------------------------------------------------ backward

    def backward(self, idx, targets, fwd=None):
        """Full backward. Returns (loss, grads dict).

        `fwd` may be the full return value of forward() — (logits, cache)
        — to avoid recomputing the forward pass (saves ~40% per step).
        """
        B, T = idx.shape
        p = self.p
        if fwd is None:
            fwd = self.forward(idx)
        logits, (x_final, xf, lnf_cache, layer_caches) = fwd

        # ---- cross-entropy
        probs = softmax(logits, axis=-1)                    # (B,T,V)
        nll = -np.log(np.maximum(probs[np.arange(B)[:, None],
                                      np.arange(T)[None, :], targets],
                                 1e-12) + 1e-12)
        loss = float(nll.mean())
        dlogits = probs.copy()
        dlogits[np.arange(B)[:, None], np.arange(T)[None, :], targets] -= 1.0
        dlogits /= (B * T)

        g = {k: np.zeros_like(v) for k, v in p.items()}

        # ---- head (tied) + final LN
        g["tok_emb"] += dlogits.reshape(-1, self.V).T @ xf.reshape(-1, self.d)
        dxf = dlogits @ p["tok_emb"]                        # (B,T,d)
        dx, dg, db = LayerNorm.backward(dxf, lnf_cache,
                                        p["lnf.g"], p["lnf.b"])
        g["lnf.g"] += dg
        g["lnf.b"] += db

        for i in reversed(range(self.L)):
            pre = f"l{i}"
            c = layer_caches[i]
            # ---- mlp backward
            dW2 = c["act"].reshape(-1, 4 * self.d).T @ dx.reshape(-1, self.d)
            g[f"{pre}.W2"] += dW2.reshape(4 * self.d, self.d)
            g[f"{pre}.b2"] += dx.sum(axis=(0, 1))
            dact = dx @ p[f"{pre}.W2"].T
            dh1 = dact * (1 - c["act"] ** 2)
            g[f"{pre}.W1"] += c["xn2"].reshape(-1, self.d).T \
                @ dh1.reshape(-1, 4 * self.d)
            g[f"{pre}.b1"] += dh1.sum(axis=(0, 1))
            dxn2 = dh1 @ p[f"{pre}.W1"].T
            dx2, dg2, db2 = LayerNorm.backward(dxn2, c["ln2"],
                                               p[f"{pre}.ln2.g"],
                                               p[f"{pre}.ln2.b"])
            g[f"{pre}.ln2.g"] += dg2
            g[f"{pre}.ln2.b"] += db2
            dx = dx + dx2  # residual

            # ---- attention backward
            douth = dx @ p[f"{pre}.Wo"].T
            g[f"{pre}.Wo"] += outh_T(c["outh"], dx, self.d)
            g[f"{pre}.bo"] += dx.sum(axis=(0, 1))
            douth_h = np.ascontiguousarray(
                douth.reshape(B, T, self.H, self.hd).transpose(0, 2, 1, 3))
            dvh = c["A"].transpose(0, 1, 3, 2) @ douth_h
            dA = douth_h @ c["vh"].transpose(0, 1, 3, 2)
            # softmax backward (per row)
            rowsum = (dA * c["A"]).sum(-1, keepdims=True)
            datt = c["A"] * (dA - rowsum)
            datt /= np.sqrt(self.hd)
            dqh = datt @ c["kh"]
            dkh = datt.transpose(0, 1, 3, 2) @ c["qh"]
            dq = dqh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            dk = dkh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            dv = dvh.transpose(0, 2, 1, 3).reshape(B, T, self.d)
            dqkv = np.concatenate([dq, dk, dv], axis=-1)   # (B,T,3d)
            g[f"{pre}.Wqkv"] += c["xn1"].reshape(-1, self.d).T \
                @ dqkv.reshape(-1, 3 * self.d)
            g[f"{pre}.bqkv"] += dqkv.sum(axis=(0, 1))
            dxn1 = dqkv @ p[f"{pre}.Wqkv"].T
            dx1, dg1, db1 = LayerNorm.backward(dxn1, c["ln1"],
                                                p[f"{pre}.ln1.g"],
                                                p[f"{pre}.ln1.b"])
            g[f"{pre}.ln1.g"] += dg1
            g[f"{pre}.ln1.b"] += db1
            dx = dx + dx1  # residual

        # ---- embeddings
        # d(pos_emb): scatter dx into positions
        for t in range(T):
            g["pos_emb"][t] += dx[:, t, :].sum(0)
        # d(tok_emb): scatter-add dx by token id
        np.add.at(g["tok_emb"], idx, dx)
        return loss, g

    # ------------------------------------------------------------ sampling

    def generate(self, idx, max_new=48, temperature=0.8, top_k=20,
                 eos=None, rng=None):
        """Autoregressive sampling. idx: (1, T) tokens."""
        rng = rng or np.random.default_rng()
        out = list(idx[0])
        for _ in range(max_new):
            ctx = np.array(out[-self.T:], dtype=np.int64)[None, :]
            logits, _ = self.forward(ctx)
            logits = logits[0, -1] / max(temperature, 1e-3)
            if top_k and top_k < logits.shape[0]:
                cut = np.partition(logits, -top_k)[-top_k]
                logits = np.where(logits < cut.min(), -1e10, logits)
            probs = softmax(logits)
            nxt = int(rng.choice(len(probs), p=probs))
            if eos is not None and nxt == eos:
                break
            out.append(nxt)
        return np.array(out, dtype=np.int64)

    # ------------------------------------------------------------ io

    def param_count(self):
        return int(sum(v.size for v in self.p.values()))

    def state_dict(self):
        return {k: v.copy() for k, v in self.p.items()}

    def load_state_dict(self, sd):
        for k in self.p:
            self.p[k] = sd[k].copy()


def outh_T(outh, dx, d):
    """grad of Wo: out_h^T @ dx reshaped properly."""
    B, T = dx.shape[0], dx.shape[1]
    return (outh.reshape(-1, d).T @ dx.reshape(-1, d))


# ================================================================ AdamW

class AdamW:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k in params:
            gk = grads[k]
            m = self.m[k]; v = self.v[k]; p = params[k]
            # in-place: m = b1*m + (1-b1)*g
            m *= self.b1
            m += (1 - self.b1) * gk
            v *= self.b2
            v += (1 - self.b2) * (gk * gk)
            # update: p -= lr * (m/bc1 / (sqrt(v/bc2)+eps) + wd*p)
            np.sqrt(v / bc2, out=v)          # v now holds sqrt(vh)
            v += self.eps
            np.divide(m, bc1, out=m)         # m now holds mh
            np.divide(m, v, out=m)           # m now holds mh/vh
            m += self.wd * p
            p -= lr * m
