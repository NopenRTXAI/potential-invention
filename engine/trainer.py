"""Genome interpreter: executes any genome as a REAL training procedure.

This is where novel training methods actually run. The trainer implements
every gene mechanism:

  gradient modes: full, topk, sign, stale_reuse, momentum_predict,
                  rand_proj, block_sparse
  update rules:   sgd, momentum, adaptive (diagonal), sign,
                  normalized, trust_ratio, hybrid_adaptive_momentum
  selection:      all, importance_topk, layer_cyclic, block_cyclic, loss_gated
  schedules:      constant, cosine, warmup_cosine, one_cycle, adaptive_plateau,
                  loss_signal, warm_restarts
  memory:         fp16/int8 optimizer state, periodic state reset
  progressive:    freeze schedules, curriculum ordering
  CPU:            micro-batching, gradient accumulation, deterministic seeds

Everything is measured: time, CPU, RAM, samples/sec, divergence, stability.
"""

import time

import numpy as np

from .metrics import classification_report, log_loss, confusion_matrix
from .models import softmax_cross_entropy, softmax


class ResourceTracker:
    """Peak RSS tracking via /proc/self/status (Linux) — zero dependencies."""

    def __init__(self):
        self.peak_rss_kb = 0
        self._start = None

    def start(self):
        self._start = time.time()
        self.peak_rss_kb = self._read_rss()

    def sample(self):
        self.peak_rss_kb = max(self.peak_rss_kb, self._read_rss())

    def elapsed(self):
        return time.time() - self._start

    def _read_rss(self):
        vals = {}
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith(("VmRSS:", "VmHWM:")):
                        k, v = line.split(":", 1)
                        vals[k] = int(v.strip().split()[0])
        except OSError:
            return 0
        return vals.get("VmHWM", vals.get("VmRSS", 0))

    @property
    def peak_rss_mb(self):
        return self.peak_rss_kb / 1024.0


class Trainer:
    """Executes one genome on one dataset for a fixed budget."""

    def __init__(self, genome, model, dataset, budget, seed=0):
        self.genome = genome
        self.g = genome.genes
        self.model = model
        self.ds = dataset
        self.budget = budget  # {"epochs": int, "max_seconds": float}
        self.seed = seed
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [],
                        "lr": [], "updated_frac": [], "epoch_seconds": []}
        self.final = None

    # ------------------------------------------------------------ schedules

    def _lr_at(self, step, total_steps, base_lr):
        g = self.g
        sch = g["lr_schedule"]
        lr_min = base_lr * g["lr_min_frac"]
        warm = max(1, int(total_steps * g["warmup_frac"]))
        p = step / max(1, total_steps)
        if sch == "constant":
            return base_lr
        if sch == "warmup_cosine":
            if step < warm:
                return base_lr * (step + 1) / warm
            q = (step - warm) / max(1, total_steps - warm)
            return lr_min + 0.5 * (base_lr - lr_min) * (1 + np.cos(np.pi * q))
        if sch == "cosine":
            return lr_min + 0.5 * (base_lr - lr_min) * (1 + np.cos(np.pi * p))
        if sch == "one_cycle":
            if p < 0.3:
                return lr_min + (base_lr - lr_min) * (p / 0.3)
            q = (p - 0.3) / 0.7
            return lr_min + (base_lr - lr_min) * 0.5 * (1 + np.cos(np.pi * q))
        if sch == "warm_restarts":
            seg = (p * 3.0) % 1.0
            return lr_min + 0.5 * (base_lr - lr_min) * (1 + np.cos(np.pi * seg))
        if sch == "loss_signal":
            h = self.history["train_loss"]
            if len(h) < 3:
                return base_lr
            slope = h[-1] - h[-3]
            if slope < -1e-4:
                return min(base_lr * 1.05, base_lr * 2.0)
            if slope > 1e-4:
                return max(base_lr * 0.7, lr_min)
            return base_lr
        if sch == "adaptive_plateau":
            return getattr(self, "_plateau_lr", base_lr)
        return base_lr

    # ------------------------------------------------------------ state

    def _init_state(self):
        """Optimizer state per parameter, with genome-controlled precision."""
        prec = self.g["state_precision"]
        state = {}
        for k, v in self.model.params().items():
            if prec == "fp32":
                state[k] = {"m": np.zeros_like(v), "v": np.zeros_like(v)}
            elif prec == "fp16":
                state[k] = {"m": np.zeros(v.shape, dtype=np.float16),
                            "v": np.zeros(v.shape, dtype=np.float16)}
            else:  # int8 quantized state
                state[k] = {"m": np.zeros(v.shape, dtype=np.int8),
                            "v": np.zeros(v.shape, dtype=np.int8),
                            "m_scale": 1e-6, "v_scale": 1e-6}
        return state

    def _state_read(self, state, k, field):
        """Read state entry, dequantizing if needed.

        The second moment ('v') is stored sqrt-compressed in low-precision
        modes: sqrt(v) utilizes the tiny dynamic range of fp16/int8 far
        better than v itself (v ~ 1e-8 underflows; sqrt(v) ~ 1e-4 does not).
        """
        e = state[k]
        if e[field].dtype == np.int8:
            out = e[field].astype(np.float32) * e[f"{field}_scale"]
        else:
            out = e[field].astype(np.float32)
        if field == "v" and e[field].dtype != np.float32:
            out = out * out  # un-compress sqrt(v) -> v
        return out

    def _state_write(self, state, k, field, value):
        e = state[k]
        if field == "v" and e[field].dtype != np.float32:
            value = np.sqrt(np.maximum(value, 0.0))  # compress v -> sqrt(v)
        if e[field].dtype == np.int8:
            scale = max(float(np.abs(value).max()), 1e-8) / 127.0
            e[f"{field}_scale"] = scale
            with np.errstate(over="ignore", invalid="ignore"):
                q = np.clip(np.round(value / scale), -127, 127)
            e[field] = np.nan_to_num(q, nan=0.0).astype(np.int8)
        elif e[field].dtype == np.float16:
            e[field] = np.nan_to_num(value, nan=0.0,
                                     posinf=65504.0, neginf=-65504.0
                                     ).astype(np.float16)
        else:
            e[field][:] = value

    # ------------------------------------------------------------ gradients

    def _transform_grads(self, raw, stale, momentum_buf, step):
        """Apply the genome's gradient-computation strategy.

        Returns (grads, used_exact).
        """
        g = self.g
        mode = g["grad_mode"]

        if mode == "full":
            return raw, True

        if mode == "topk":
            out = {}
            frac = g["grad_topk_frac"]
            for k, v in raw.items():
                flat = np.abs(v).ravel()
                kk = max(1, int(flat.size * frac))
                thresh = np.partition(flat, -kk)[-kk]
                out[k] = v * (np.abs(v) >= thresh)
            return out, True

        if mode == "sign":
            # compressed gradient: sign * fourth-root magnitude (keeps some scale info)
            return {k: np.sign(v) * np.sqrt(np.sqrt(np.abs(v) + 1e-12))
                    for k, v in raw.items()}, True

        if mode == "stale_reuse":
            if step % g["stale_steps"] == 0 or stale is None:
                return raw, True
            return stale, False

        if mode == "momentum_predict":
            # exact every `stale_steps`; between exact steps, extrapolate the
            # gradient using the running momentum of past exact gradients.
            period = max(1, g["stale_steps"])
            if step % period == 0 or stale is None:
                for k, v in raw.items():
                    momentum_buf[k] = 0.8 * momentum_buf[k] + 0.2 * v
                return raw, True
            out = {}
            for k in raw:
                out[k] = stale[k] + g["momentum"] * momentum_buf[k]
            return out, False

        if mode == "rand_proj":
            # random-direction sketch: keep only the component of the gradient
            # along a random sign vector (cheap 1-D gradient estimate)
            out = {}
            for k, v in raw.items():
                rs = np.random.RandomState((step * 2654435761 + hash(k)) % (2 ** 31))
                signs = rs.choice([-1.0, 1.0], size=v.shape).astype(np.float32)
                proj = float((v * signs).sum())
                norm = float(np.sqrt((v * v).sum())) + 1e-12
                out[k] = signs * (proj / norm)
            return out, True

        if mode == "block_sparse":
            # exact gradient for a rotating subset of rows; stale elsewhere
            out = {}
            for k, v in raw.items():
                if stale is not None and k in stale:
                    base = stale[k].copy()
                else:
                    base = np.zeros_like(v)
                if v.ndim == 2:
                    rows = max(1, v.shape[0] // 2)
                    rs = np.random.RandomState((step * 40503 + hash(k)) % (2 ** 31))
                    ridx = rs.choice(v.shape[0], rows, replace=False)
                    base[ridx] = v[ridx]
                else:
                    base = v
                out[k] = base
            return out, True

        return raw, True

    # ------------------------------------------------------------ selection

    def _select_params(self, step, grads, importance, age, frozen, layer_names, layer_cycle):
        """Which parameters may update this step. Returns dict of bool masks."""
        g = self.g
        mode = g["update_selection"]
        masks = {}
        params = self.model.params()

        if mode == "all":
            for k, v in params.items():
                m = np.ones(v.shape, dtype=bool)
                if frozen.get(k):
                    m[:] = False
                masks[k] = m
            return masks

        if mode == "importance_topk":
            src = g["importance_source"]
            for k, v in params.items():
                if frozen.get(k):
                    masks[k] = np.zeros(v.shape, dtype=bool)
                    continue
                if src == "grad_mag":
                    imp = np.abs(grads[k])
                elif src in ("grad_history", "adam_diag"):
                    imp = importance[k]
                else:  # age
                    imp = -age[k].astype(np.float32)
                flat = imp.ravel()
                kk = max(1, int(flat.size * 0.5))
                thresh = np.partition(flat, -kk)[-kk]
                masks[k] = imp >= thresh
            return masks

        if mode == "layer_cyclic":
            groups = {}
            for k in params:
                groups.setdefault(k.split(".")[0], []).append(k)
            names = list(groups)
            if not names:
                return {k: np.ones(v.shape, dtype=bool) for k, v in params.items()}
            active = names[layer_cycle % len(names)]
            for k, v in params.items():
                m = np.zeros(v.shape, dtype=bool)
                if k.split(".")[0] == active and not frozen.get(k):
                    m[:] = True
                masks[k] = m
            return masks

        if mode == "block_cyclic":
            for k, v in params.items():
                m = np.zeros(v.shape, dtype=bool)
                if not frozen.get(k):
                    flat = m.ravel()
                    half = max(1, flat.size // 2)
                    off = (step * half) % flat.size
                    idx = (np.arange(half) + off) % flat.size
                    flat[idx] = True
                    m = flat.reshape(v.shape)
                masks[k] = m
            return masks

        if mode == "loss_gated":
            h = self.history["train_loss"]
            improving = True
            if len(h) >= 2:
                improving = h[-1] <= h[-2] * 1.02
            for k, v in params.items():
                m = np.zeros(v.shape, dtype=bool)
                if improving and not frozen.get(k):
                    m[:] = True
                masks[k] = m
            return masks

        for k, v in params.items():
            masks[k] = np.ones(v.shape, dtype=bool)
        return masks

    # ------------------------------------------------------------ updates

    def _apply_update(self, grads, masks, lr, state, momentum_buf, importance, age):
        g = self.g
        rule = g["update_rule"]
        mom = g["momentum"]
        beta2, eps, wd = g["beta2"], g["eps"], g["weight_decay"]
        clip = g["update_clip"]
        norm_mode = g["update_norm_mode"]
        params = self.model.params()
        self._t = getattr(self, "_t", 0) + 1

        for k, p in params.items():
            mask = masks[k]
            if not mask.any():
                age[k] += 1
                continue
            grad = grads[k].astype(np.float32)
            m = self._state_read(state, k, "m")
            v = self._state_read(state, k, "v")

            if rule == "sgd":
                delta = grad
            elif rule == "momentum":
                m = mom * m + grad
                delta = m
            elif rule == "adaptive":
                m = 0.9 * m + 0.1 * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                mhat = m / (1 - 0.9 ** self._t)
                vhat = v / (1 - beta2 ** self._t)
                delta = mhat / (np.sqrt(np.maximum(vhat, 0)) + eps)
            elif rule == "sign":
                m = mom * m + grad
                delta = np.sign(m)
            elif rule == "normalized":
                rms = np.sqrt(np.mean(grad * grad)) + eps
                delta = grad / rms
            elif rule == "trust_ratio":
                m = mom * m + grad
                u = m / (np.sqrt(np.mean(m * m)) + eps)
                pn = np.sqrt(np.mean(p * p)) + eps
                un = np.sqrt(np.mean(u * u)) + eps
                delta = u * (pn / un)
            elif rule == "hybrid_adaptive_momentum":
                # novel combination: adaptive second moment + heavy momentum
                # + gradient-magnitude gating (small grads amplified less)
                m = mom * m + (1 - mom) * grad
                v = beta2 * v + (1 - beta2) * grad * grad
                gate = np.abs(grad) / (np.abs(grad).mean() + eps)
                gate = np.clip(gate, 0.25, 4.0)
                delta = (m / (np.sqrt(v) + eps)) * gate
            else:
                delta = grad

            if wd > 0:
                delta = delta + wd * p

            if norm_mode == "l2_per_param":
                nrm = np.sqrt(np.sum(delta * delta)) + eps
                delta = delta / max(nrm, 1.0)
            elif norm_mode == "rms_clip":
                rms = np.sqrt(np.mean(delta * delta)) + eps
                if rms > 1.0:
                    delta = delta / rms
            elif norm_mode == "trust_coef":
                pn = np.sqrt(np.mean(p * p)) + eps
                un = np.sqrt(np.mean(delta * delta)) + eps
                delta = delta * min(1.0, pn / un)
            if clip > 0:
                delta = np.clip(delta, -clip, clip)

            step_vec = (lr * delta).astype(p.dtype)
            p -= np.where(mask, step_vec, np.zeros_like(step_vec)).astype(p.dtype)

            self._state_write(state, k, "m", m)
            self._state_write(state, k, "v", v)
            importance[k] = 0.99 * importance[k] + 0.01 * np.abs(grad)
            age[k] = np.where(mask, 0, age[k] + 1)

    # ------------------------------------------------------------ progressive

    def _curriculum_order(self, rng, Xtr, ytr):
        mode = self.g["curriculum"]
        n = len(ytr)
        if not self.g["shuffle"]:
            base = np.arange(n)
        else:
            base = rng.permutation(n)
        if mode == "none" or not self.history["train_loss"]:
            return base
        # curriculum only after the first epoch
        try:
            logits = self.model.forward(Xtr)
            probs = softmax(logits)
            conf = probs[np.arange(n), ytr]
            if mode == "easy_first":
                score = conf
            elif mode == "loss_sorted":
                score = -np.log(np.clip(conf, 1e-9, 1))
                score = -score  # lowest loss first
            else:
                return base
            order = np.argsort(-score)
            blocks = 8
            out = []
            for b in range(blocks):
                blk = order[b * n // blocks:(b + 1) * n // blocks]
                out.extend(rng.permutation(blk))
            rest = order[blocks * n // blocks:]
            out.extend(rest)
            return np.array(out, dtype=int)
        except Exception:
            return base

    def _apply_freeze(self, epoch, total_epochs, frozen, layer_names):
        sch = self.g["freeze_schedule"]
        if sch == "none":
            return
        groups = {}
        for k in layer_names:
            groups.setdefault(k.split(".")[0], []).append(k)
        names = list(groups)
        if not names:
            return
        if sch == "freeze_first_then_unfreeze":
            frac = epoch / max(1, total_epochs)
            n_unfrozen = max(1, int(len(names) * frac))
            for i, nm in enumerate(names):
                for k in groups[nm]:
                    frozen[k] = i >= n_unfrozen
        elif sch == "freeze_last_then_unfreeze":
            frac = epoch / max(1, total_epochs)
            n_unfrozen = max(1, int(len(names) * frac))
            for i, nm in enumerate(reversed(names)):
                for k in groups[nm]:
                    frozen[k] = i >= n_unfrozen
        elif sch == "alternate_freeze":
            half = max(1, len(names) // 2)
            for i, nm in enumerate(names):
                active = (i // half) == (epoch % 2) if len(names) > 1 else True
                for k in groups[nm]:
                    frozen[k] = not active

    # ------------------------------------------------------------ main loop

    def train(self):
        (Xtr, ytr), (Xva, yva), (Xte, yte) = self.ds.split(seed=self.seed)
        g = self.g
        rng = np.random.RandomState(self.seed)
        res = ResourceTracker()
        res.start()

        n = len(ytr)
        bs = g["micro_batch"]
        accum = g["grad_accum_steps"]
        steps_per_epoch = max(1, int(np.ceil(n / (bs * accum))))
        total_steps = self.budget["epochs"] * steps_per_epoch
        base_lr = g["lr"]
        self._plateau_lr = base_lr
        self._t = 0

        state = self._init_state()
        momentum_buf = {k: np.zeros(v.shape, dtype=np.float32)
                        for k, v in self.model.params().items()}
        importance = {k: np.zeros(v.shape, dtype=np.float32)
                      for k, v in self.model.params().items()}
        age = {k: np.zeros(v.shape, dtype=np.int32)
               for k, v in self.model.params().items()}
        frozen = {k: False for k in self.model.params()}
        layer_names = list(self.model.params().keys())
        layer_cycle = 0
        step_count = 0
        diverged = False
        plateau_count = 0
        stale_grads = None
        updated_frac_hist = []
        epoch_train_loss = []

        for epoch in range(self.budget["epochs"]):
            t0 = time.time()
            order = self._curriculum_order(rng, Xtr, ytr)
            self._apply_freeze(epoch, self.budget["epochs"], frozen, layer_names)

            for start in range(0, n, bs * accum):
                if res.elapsed() > self.budget["max_seconds"]:
                    break

                lr = self._lr_at(step_count, total_steps, base_lr)
                if g["lr_schedule"] == "adaptive_plateau" and len(self.history["val_loss"]) >= 2:
                    if self.history["val_loss"][-1] > self.history["val_loss"][-2] + 1e-4:
                        plateau_count += 1
                        if plateau_count >= 2:
                            self._plateau_lr *= 0.5
                            plateau_count = 0
                    else:
                        plateau_count = 0
                    lr = self._plateau_lr

                # ---- gradient accumulation over micro-batches
                acc_grads = None
                micro_losses = []
                for micro in range(accum):
                    lo = start + micro * bs
                    hi = min(start + (micro + 1) * bs, n)
                    if lo >= hi:
                        continue
                    idx = order[lo:hi]
                    Xb, yb = Xtr[idx], ytr[idx]
                    logits = self.model.forward(Xb)
                    loss, dlogits = softmax_cross_entropy(logits, yb)
                    self.model.backward(dlogits)
                    raw = self.model.grads()
                    if acc_grads is None:
                        acc_grads = {k: v.copy() for k, v in raw.items()}
                    else:
                        for k in acc_grads:
                            acc_grads[k] += raw[k]
                    micro_losses.append(loss)
                if acc_grads is None:
                    continue
                epoch_train_loss.append(float(np.mean(micro_losses)))

                # ---- gradient transformation (novel core)
                grads, used_exact = self._transform_grads(
                    acc_grads, stale_grads, momentum_buf, step_count)
                if used_exact:
                    stale_grads = {k: v.copy() for k, v in grads.items()}

                # ---- parameter selection
                masks = self._select_params(
                    step_count, grads, importance, age, frozen, layer_names, layer_cycle)
                layer_cycle += 1

                # ---- apply update
                self._apply_update(grads, masks, lr, state, momentum_buf,
                                   importance, age)

                # ---- state memory management
                if g["state_reset_every"] and step_count and \
                        step_count % g["state_reset_every"] == 0:
                    for k in state:
                        self._state_write(state, k, "m",
                                          np.zeros_like(state[k]["m"], dtype=np.float32))
                        self._state_write(state, k, "v",
                                          np.zeros_like(state[k]["v"], dtype=np.float32))

                step_count += 1
                updated_frac_hist.append(
                    float(np.mean([m.mean() for m in masks.values()])))
                res.sample()
                if epoch_train_loss and not np.isfinite(epoch_train_loss[-1]):
                    diverged = True
                    break
                if epoch_train_loss and epoch_train_loss[-1] > 1e4:
                    diverged = True
                    break

            # ---- epoch end: validation
            val_logits = self.model.forward(Xva)
            vloss, _ = softmax_cross_entropy(val_logits, yva)
            vpred = val_logits.argmax(1)
            vacc = float(np.mean(vpred == yva))
            self.history["train_loss"].append(
                float(np.mean(epoch_train_loss)) if epoch_train_loss else float("nan"))
            self.history["val_loss"].append(vloss)
            self.history["val_acc"].append(vacc)
            self.history["lr"].append(lr)
            self.history["updated_frac"].append(
                float(np.mean(updated_frac_hist)) if updated_frac_hist else 0.0)
            self.history["epoch_seconds"].append(time.time() - t0)
            updated_frac_hist = []
            if diverged or res.elapsed() > self.budget["max_seconds"]:
                break

        # ---- final metrics on the held-out test split
        test_logits = self.model.forward(Xte)
        test_pred = test_logits.argmax(1)
        report = classification_report(yte, test_pred, self.ds.n_classes)
        probs = softmax(test_logits)
        self.final = {
            "test_accuracy": report["accuracy"],
            "test_f1": report["f1"],
            "test_precision": report["precision"],
            "test_recall": report["recall"],
            "test_loss": log_loss(probs, yte),
            "val_accuracy": max(self.history["val_acc"]) if self.history["val_acc"] else 0.0,
            "val_accuracy_final": self.history["val_acc"][-1] if self.history["val_acc"] else 0.0,
            "train_seconds": res.elapsed(),
            "peak_rss_mb": res.peak_rss_mb,
            "samples_per_sec": (n * len(self.history["val_loss"])) / max(0.1, res.elapsed()),
            "param_count": self.model.param_count(),
            "epochs_completed": len(self.history["val_loss"]),
            "diverged": diverged,
            "steps": step_count,
            "confusion": confusion_matrix(yte, test_pred, self.ds.n_classes),
            "history": self.history,
        }
        return self.final
