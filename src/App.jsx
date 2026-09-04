import { useEffect, useState, useCallback, useRef, Fragment } from 'react';
import { api, fmt } from './api.js';

// ================================================================ helpers

const usePoll = (fn, ms, deps) => {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const load = useCallback(async () => {
    try {
      setData(await fn());
      setError(null);
    } catch (e) {
      setError(String(e.message || e));
    }
  }, deps || []);
  useEffect(() => {
    load();
    if (!ms) return;
    const t = setInterval(load, ms);
    return () => clearInterval(t);
  }, [load, ms]);
  return [data, error, load];
};

const kindBadge = (kind) =>
  kind ? <span className={`badge ${kind}`}>{kind}</span> : null;

const resultBadge = (r) =>
  r ? <span className={`badge ${r}`}>{r}</span> : null;

// ================================================================ Research

function ResearchSection({ status }) {
  const [form, setForm] = useState({
    dataset: 'spam',
    seconds: 3600,
    architecture: 'mlp',
    epochs: 8,
    max_train_seconds: 120,
    population_size: 10,
    priority: 'quality',
    seed: 0,
  });
  const [datasets, setDatasets] = useState({ builtins: [] });
  const [msg, setMsg] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.datasets().then(setDatasets).catch(() => {});
  }, []);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  const start = async () => {
    setErr(null);
    setMsg(null);
    try {
      await api.startResearch({
        ...form,
        seconds: Number(form.seconds),
        epochs: Number(form.epochs),
        max_train_seconds: Number(form.max_train_seconds),
        population_size: Number(form.population_size),
        seed: Number(form.seed),
      });
      setMsg('Research started. The engine is now autonomously inventing and testing training algorithms.');
    } catch (e) {
      setErr(String(e.message || e));
    }
  };

  const stop = async () => {
    try {
      await api.stopResearch();
      setMsg('Stop requested — the agent will halt after the current experiment.');
    } catch (e) {
      setErr(String(e.message || e));
    }
  };

  const running = status?.status === 'researching';
  const cfg = status?.config || {};
  const elapsed = cfg.research_seconds ? cfg.research_seconds - (status?.time_left ?? cfg.research_seconds) : 0;

  return (
    <div>
      <h1 className="page-title">Research Control</h1>
      <p className="page-sub">
        Launch an autonomous research session. The engine invents novel CPU-first
        training algorithms, runs controlled experiments against baselines, and
        evolves the most promising mechanisms — all on real data, no simulation.
      </p>

      {err && <div className="error-banner">{err}</div>}
      {msg && (
        <div className="panel" style={{ borderColor: 'var(--green)' }}>{msg}</div>
      )}

      {running && (
        <div className="panel">
          <h3>Session in progress</h3>
          <div className="progress-bar" style={{ marginBottom: 10 }}>
            <div
              className="fill"
              style={{
                width: `${Math.min(100, (elapsed / (cfg.research_seconds || 1)) * 100)}%`,
              }}
            />
          </div>
          <div className="kv">
            <span className="k">Dataset</span><span className="v">{cfg.dataset}</span>
            <span className="k">Budget</span><span className="v">{fmt.secs(cfg.research_seconds)}</span>
            <span className="k">Elapsed</span><span className="v">{fmt.secs(elapsed)}</span>
            <span className="k">Generation</span><span className="v">{status.generation ?? '—'}</span>
            <span className="k">Experiments</span><span className="v">{status.experiments ?? '—'}</span>
          </div>
          <button className="btn danger" style={{ marginTop: 14 }} onClick={stop}>
            Request Stop
          </button>
        </div>
      )}

      <div className="panel">
        <h3>New research session</h3>
        <div className="grid-3">
          <div className="field">
            <label>Dataset</label>
            <select value={form.dataset} onChange={set('dataset')} disabled={running}>
              {(datasets.builtins || ['spam', 'digits', 'circles', 'spirals']).map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
            <div className="hint">Built-in datasets load automatically (offline fallback included)</div>
          </div>
          <div className="field">
            <label>Architecture</label>
            <select value={form.architecture} onChange={set('architecture')} disabled={running}>
              <option value="mlp">MLP</option>
              <option value="gru">GRU</option>
              <option value="tiny_transformer">TinyTransformer</option>
            </select>
            <div className="hint">Pure NumPy models, CPU-native</div>
          </div>
          <div className="field">
            <label>Research budget</label>
            <select value={form.seconds} onChange={set('seconds')} disabled={running}>
              <option value={600}>10 minutes</option>
              <option value={3600}>1 hour</option>
              <option value={21600}>6 hours</option>
              <option value={43200}>12 hours</option>
              <option value={86400}>24 hours</option>
              <option value={172800}>48 hours</option>
              <option value={604800}>7 days</option>
            </select>
          </div>
          <div className="field">
            <label>Epochs per experiment</label>
            <input type="number" min="1" value={form.epochs} onChange={set('epochs')} disabled={running} />
            <div className="hint">Identical for every algorithm — compute-fair comparison</div>
          </div>
          <div className="field">
            <label>Max train seconds / experiment</label>
            <input
              type="number" min="5" value={form.max_train_seconds}
              onChange={set('max_train_seconds')} disabled={running}
            />
          </div>
          <div className="field">
            <label>Population size</label>
            <input
              type="number" min="4" max="40" value={form.population_size}
              onChange={set('population_size')} disabled={running}
            />
            <div className="hint">Algorithms invented per generation</div>
          </div>
          <div className="field">
            <label>Priority</label>
            <select value={form.priority} onChange={set('priority')} disabled={running}>
              <option value="quality">Quality first (training time is not the primary objective)</option>
              <option value="balanced">Balanced</option>
              <option value="efficiency">Efficiency</option>
            </select>
          </div>
          <div className="field">
            <label>Random seed</label>
            <input type="number" value={form.seed} onChange={set('seed')} disabled={running} />
            <div className="hint">Fixed seed = reproducible experiments</div>
          </div>
        </div>
        <button className="btn" onClick={start} disabled={running}>
          {running ? 'Research running…' : '▶ Start Autonomous Research'}
        </button>
      </div>
    </div>
  );
}

// ================================================================ Algorithms (lineage graph)

function LineageGraph({ algs, selected, onSelect }) {
  if (!algs || !Object.keys(algs).length) {
    return <div className="empty"><div className="big">🧬</div>No algorithms yet — start a research session.</div>;
  }

  // Build generation layers
  const byGen = {};
  for (const a of Object.values(algs)) {
    const g = a.generation || 0;
    (byGen[g] = byGen[g] || []).push(a);
  }
  const gens = Object.keys(byGen).map(Number).sort((x, y) => x - y);

  // Cap the graph at the most recent 12 generations for readability
  const shown = gens.slice(-12);
  const shownSet = new Set(shown);
  const W = 240;
  const nodeW = 150;
  const nodeH = 30;
  const rowH = 64;
  const colW = 200;

  const nodes = [];
  const pos = {};
  shown.forEach((g) => {
    const row = byGen[g]
      .sort((a, b) => (b.score ?? -1) - (a.score ?? -1))
      .slice(0, 10); // top 10 per generation
    row.forEach((a, i) => {
      pos[a.name] = { x: 20 + g * colW, y: 20 + i * rowH };
      nodes.push(a);
    });
  });

  const edges = [];
  for (const a of nodes) {
    if (a.parent && pos[a.parent] && pos[a.name]) {
      edges.push({ from: pos[a.parent], to: pos[a.name], xo: false });
    }
    if (a.parent2 && pos[a.parent2] && pos[a.name]) {
      edges.push({ from: pos[a.parent2], to: pos[a.name], xo: true });
    }
  }

  const height = 20 + Math.max(...shown.map((g) => Math.min(10, byGen[g].length))) * rowH + 20;
  const width = 20 + (shown.length ? shown[shown.length - 1] : 0) * colW + nodeW + 20;

  return (
    <div className="graph-wrap">
      <svg width={width} height={height} style={{ minWidth: '100%' }}>
        {edges.map((e, i) => (
          <path
            key={i}
            className={`graph-edge ${e.xo ? 'xo' : ''}`}
            d={`M ${e.from.x + nodeW} ${e.from.y + nodeH / 2} C ${e.from.x + nodeW + 40} ${e.from.y + nodeH / 2}, ${e.to.x - 40} ${e.to.y + nodeH / 2}, ${e.to.x} ${e.to.y + nodeH / 2}`}
          />
        ))}
        {nodes.map((a) => (
          <g
            key={a.name}
            className={`graph-node ${selected === a.name ? 'selected' : ''} ${a.generation === 0 ? 'baseline' : ''}`}
            onClick={() => onSelect(a.name)}
            style={{ cursor: 'pointer' }}
          >
            <rect x={pos[a.name].x} y={pos[a.name].y} width={nodeW} height={nodeH} rx={7} />
            <text x={pos[a.name].x + 10} y={pos[a.name].y + 19}>
              {a.name.length > 17 ? a.name.slice(0, 16) + '…' : a.name}
            </text>
            <text
              x={pos[a.name].x + nodeW - 10}
              y={pos[a.name].y + 19}
              textAnchor="end"
              style={{ fill: a.score >= 0.9 ? 'var(--green)' : 'var(--text-faint)' }}
            >
              {a.score != null ? (a.score * 100).toFixed(1) : ''}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function AlgorithmsSection() {
  const [lineage, , reload] = usePoll(api.lineage, 5000, []);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    if (!selected) return;
    const a = lineage?.[selected];
    if (a) {
      setDetail(a);
    } else if (lineage) {
      api.experiment(a?.experiment_id).catch(() => setDetail(null));
    }
  }, [selected, lineage]);

  const algs = lineage || {};
  const sel = selected ? algs[selected] : null;

  return (
    <div>
      <h1 className="page-title">Algorithms & Lineage</h1>
      <p className="page-sub">
        Every algorithm the engine has invented, its ancestry (mutations and
        crossovers), and what changed. Click a node to inspect its genome.
      </p>

      <div className="panel">
        <h3>Evolution graph — click to inspect</h3>
        <LineageGraph algs={algs} selected={selected} onSelect={setSelected} />
      </div>

      {sel && (
        <div className="panel">
          <h3>{sel.name}</h3>
          <div className="grid-2">
            <div>
              <div className="kv">
                <span className="k">Score (test acc)</span>
                <span className="v">{fmt.pct(sel.score)}</span>
                <span className="k">Generation</span>
                <span className="v">{sel.generation}</span>
                <span className="k">Parent</span>
                <span className="v mono">{sel.parent || '— (root)'}</span>
                <span className="k">Second parent</span>
                <span className="v mono">{sel.parent2 || '—'}</span>
                <span className="k">Mutation</span>
                <span className="v mono" style={{ whiteSpace: 'pre-wrap' }}>
                  {sel.mutation || '—'}
                </span>
                <span className="k">UID</span>
                <span className="v mono">{sel.uid}</span>
              </div>
            </div>
            <div>
              <h3 style={{ marginTop: 0 }}>Hypothesis</h3>
              <p style={{ color: 'var(--text-dim)', marginBottom: 12 }}>{sel.hypothesis}</p>
              <h3>Genome (the algorithm)</h3>
              <div className="kv mono" style={{ gridTemplateColumns: '150px 1fr' }}>
                {Object.entries(sel.genes || {}).map(([k, v]) => (
                  <Fragment key={k}>
                    <span className="k">{k}</span>
                    <span className="v">{String(v)}</span>
                  </Fragment>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ================================================================ Experiments

function ExperimentsSection() {
  const [exps, , reload] = usePoll(api.experiments, 6000, []);
  const [selId, setSelId] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    if (selId == null) return;
    api.experiment(selId).then(setDetail).catch(() => setDetail(null));
  }, [selId]);

  const rows = exps || [];

  return (
    <div>
      <h1 className="page-title">Experiments</h1>
      <p className="page-sub">
        Every controlled experiment ever run — including failures. Nothing is
        thrown away; failed algorithms are research data too.
      </p>

      <div className="panel">
        <h3>{rows.length} experiments</h3>
        <div className="scroll-y">
          <table className="data">
            <thead>
              <tr>
                <th>#</th><th>Name</th><th>Kind</th><th>Result</th>
                <th>Score</th><th>RAM</th><th>Time</th><th>Epochs</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice().reverse().map((e) => (
                <tr key={e.id} className="clickable" onClick={() => setSelId(e.id)}>
                  <td className="mono">{e.id}</td>
                  <td className="mono">{e.name}</td>
                  <td>{kindBadge(e.kind)}</td>
                  <td>{resultBadge(e.result)}</td>
                  <td>{fmt.pct(e.score)}</td>
                  <td>{fmt.mb(e.metrics?.peak_rss_mb)}</td>
                  <td>{fmt.secs(e.metrics?.train_seconds)}</td>
                  <td>{e.metrics?.epochs_completed ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {detail && (
        <div className="panel">
          <h3>Experiment #{detail.id} — {detail.name}</h3>
          <div className="grid-2">
            <div>
              <div className="kv">
                <span className="k">Hypothesis</span>
                <span className="v">{detail.hypothesis}</span>
                <span className="k">Dataset</span><span className="v">{detail.dataset}</span>
                <span className="k">Architecture</span><span className="v">{detail.architecture}</span>
                <span className="k">Seed</span><span className="v mono">{detail.seed}</span>
                <span className="k">Parent</span><span className="v mono">{detail.parent || '—'}</span>
                <span className="k">Mutation</span>
                <span className="v mono">{detail.mutation || '—'}</span>
                <span className="k">Test accuracy</span><span className="v">{fmt.pct(detail.metrics?.test_accuracy)}</span>
                <span className="k">Test F1</span><span className="v">{fmt.num(detail.metrics?.test_f1, 4)}</span>
                <span className="k">Val accuracy</span><span className="v">{fmt.pct(detail.metrics?.val_accuracy)}</span>
                <span className="k">Peak RAM</span><span className="v">{fmt.mb(detail.metrics?.peak_rss_mb)}</span>
                <span className="k">Train time</span><span className="v">{fmt.secs(detail.metrics?.train_seconds)}</span>
                <span className="k">Samples/sec</span><span className="v">{fmt.num(detail.metrics?.samples_per_sec, 0)}</span>
                <span className="k">Params</span><span className="v">{detail.metrics?.param_count ?? '—'}</span>
                <span className="k">Diverged</span><span className="v">{String(detail.metrics?.diverged)}</span>
              </div>
              {detail.error && (
                <div className="error-banner" style={{ marginTop: 12 }}>{detail.error}</div>
              )}
            </div>
            <div>
              <h3 style={{ marginTop: 0 }}>Genome</h3>
              <div className="kv mono" style={{ gridTemplateColumns: '150px 1fr' }}>
                {Object.entries(detail.genes || {}).map(([k, v]) => (
                  <Fragment key={k}>
                    <span className="k">{k}</span>
                    <span className="v">{String(v)}</span>
                  </Fragment>
                ))}
              </div>
              {detail.history && (
                <>
                  <h3 style={{ marginTop: 16 }}>Learning curves</h3>
                  <LossChart history={detail.history} />
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function LossChart({ history }) {
  const tl = history?.train_loss || [];
  const vl = history?.val_loss || [];
  if (!tl.length) return null;
  const W = 460, H = 160, pad = 30;
  const all = [...tl, ...vl].filter(Number.isFinite);
  const max = Math.max(...all, 0.1);
  const min = Math.min(...all, 0);
  const x = (i) => pad + (i / Math.max(1, tl.length - 1)) * (W - pad * 2);
  const y = (v) => H - pad - ((v - min) / Math.max(1e-9, max - min)) * (H - pad * 2);
  const path = (arr) => arr.map((v, i) => `${i ? 'L' : 'M'} ${x(i)} ${y(v)}`).join(' ');
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ background: 'var(--bg-2)', borderRadius: 8 }}>
      <path d={path(tl)} stroke="var(--accent)" fill="none" strokeWidth="2" />
      {vl.length > 0 && <path d={path(vl)} stroke="var(--yellow)" fill="none" strokeWidth="2" strokeDasharray="4 3" />}
      <text x={pad} y={16} fontSize="10" fill="var(--text-faint)">loss (epoch) — train solid, val dashed</text>
      <text x={pad} y={y(max) + 4} fontSize="9" fill="var(--text-faint)">{max.toFixed(2)}</text>
      <text x={pad} y={y(min) + 4} fontSize="9" fill="var(--text-faint)">{min.toFixed(2)}</text>
    </svg>
  );
}

// ================================================================ Models

function ModelsSection({ status }) {
  const [model, setModel] = useState(null);
  const [err, setErr] = useState(null);

  const load = useCallback(async () => {
    try {
      setModel(await api.model());
      setErr(null);
    } catch (e) {
      setErr(String(e.message || e));
      setModel(null);
    }
  }, []);
  useEffect(() => {
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, [load]);

  if (!model) {
    return (
      <div>
        <h1 className="page-title">Models</h1>
        <p className="page-sub">The final trained model produced by the winning algorithm.</p>
        <div className="empty">
          <div className="big">📦</div>
          {err?.includes('no trained model')
            ? 'No final model yet — complete a research session first.'
            : 'Loading…'}
        </div>
      </div>
    );
  }

  return (
    <div>
      <h1 className="page-title">Models</h1>
      <p className="page-sub">Real model artifacts trained by the discovered algorithm — weights, metadata, and a standalone predictor.</p>

      <div className="panel">
        <h3>Best model — {model.name}</h3>
        <div className="grid-3">
          <div className="stat-card">
            <div className="label">Test accuracy</div>
            <div className="value green">{fmt.pct(model.metrics?.test_accuracy)}</div>
          </div>
          <div className="stat-card">
            <div className="label">Test F1</div>
            <div className="value">{fmt.num(model.metrics?.test_f1, 4)}</div>
          </div>
          <div className="stat-card">
            <div className="label">Parameters</div>
            <div className="value accent">{model.param_count?.toLocaleString?.() ?? '—'}</div>
          </div>
        </div>
        <div style={{ marginTop: 16 }} className="kv">
          <span className="k">Architecture</span><span className="v">{model.architecture}</span>
          <span className="k">Dataset</span><span className="v">{model.dataset}</span>
          <span className="k">Input dim</span><span className="v mono">{model.in_dim}</span>
          <span className="k">Classes</span><span className="v">{(model.label_names || []).join(', ') || model.n_classes}</span>
          <span className="k">Trained with</span><span className="v mono">{model.genome?.name || model.name}</span>
          <span className="k">Artifacts</span>
          <span className="v mono">models/best_model/ — model.npz, metadata.json, predictor.py</span>
        </div>
      </div>

      {model.genome?.genes && (
        <div className="panel">
          <h3>Winning training algorithm (genome)</h3>
          <div className="kv mono" style={{ gridTemplateColumns: '150px 1fr' }}>
            {Object.entries(model.genome.genes).map(([k, v]) => (
              <Fragment key={k}>
                <span className="k">{k}</span>
                <span className="v">{String(v)}</span>
              </Fragment>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ================================================================ Datasets

function DatasetsSection() {
  const [ds, setDs] = useState(null);
  useEffect(() => {
    api.datasets().then(setDs).catch(() => {});
  }, []);
  const info = {
    spam: 'UCI SMS Spam Collection — binary text classification (spam / ham). Falls back to a synthetic replica when offline.',
    digits: 'Sklearn-style 8×4 downsampled digits, 10 classes, 64 features. Embedded synthetic templates when offline.',
    circles: 'Two concentric circles — 1200 points, 2 features, binary. Tests non-linear decision boundaries.',
    spirals: 'Three interleaved spirals — 1500 points, 2 features, 3 classes. Deliberately hard: the research target.',
  };
  return (
    <div>
      <h1 className="page-title">Datasets</h1>
      <p className="page-sub">
        Built-in datasets with fixed 70/15/15 train/val/test splits and seeded
        shuffling — every algorithm sees identical data.
      </p>
      <div className="grid-2">
        {Object.entries(info).map(([name, desc]) => (
          <div className="panel" key={name} style={{ marginBottom: 0 }}>
            <h3>{name}</h3>
            <p style={{ color: 'var(--text-dim)' }}>{desc}</p>
          </div>
        ))}
      </div>
      {ds?.note && (
        <div className="panel" style={{ marginTop: 16 }}>
          <h3>Custom data</h3>
          <p style={{ color: 'var(--text-dim)' }}>{ds.note}</p>
        </div>
      )}
    </div>
  );
}

// ================================================================ Hardware

function HardwareSection({ hardware }) {
  if (!hardware) return <div className="loading">Loading hardware…</div>;
  const caches = hardware.caches || {};
  return (
    <div>
      <h1 className="page-title">Hardware</h1>
      <p className="page-sub">
        CPU-first design: the engine adapts its strategies to the actual silicon —
        cache sizes, SIMD width, core count, and RAM.
      </p>
      <div className="grid-4">
        <div className="stat-card">
          <div className="label">Cores</div>
          <div className="value accent">{hardware.cores}</div>
        </div>
        <div className="stat-card">
          <div className="label">SIMD</div>
          <div className="value" style={{ textTransform: 'uppercase' }}>{hardware.simd}</div>
          <div className="sub">vector instruction set</div>
        </div>
        <div className="stat-card">
          <div className="label">RAM</div>
          <div className="value">{hardware.ram_gb?.toFixed(1)} GB</div>
        </div>
        <div className="stat-card">
          <div className="label">Weak hardware mode</div>
          <div className={`value ${hardware.weak_hardware ? 'yellow' : 'green'}`}>
            {hardware.weak_hardware ? 'ON' : 'OFF'}
          </div>
          <div className="sub">{hardware.weak_hardware ? 'memory-saving mechanisms prioritized' : 'full mechanism space enabled'}</div>
        </div>
      </div>
      <div className="panel">
        <h3>Cache hierarchy</h3>
        <div className="kv">
          <span className="k">L1 cache</span><span className="v">{(caches.L1 / 1024).toFixed(0)} KB</span>
          <span className="k">L2 cache</span><span className="v">{(caches.L2 / 1024).toFixed(0)} KB</span>
          <span className="k">L3 cache</span><span className="v">{(caches.L3 / (1024 * 1024)).toFixed(1)} MB</span>
          <span className="k">Platform</span><span className="v mono">{hardware.platform}</span>
          <span className="k">Python</span><span className="v mono">{hardware.python}</span>
          <span className="k">NumPy</span><span className="v mono">{hardware.numpy}</span>
        </div>
      </div>
    </div>
  );
}

// ================================================================ Leaderboard

function LeaderboardSection() {
  const [lb, , reload] = usePoll(api.leaderboard, 8000, []);
  const rows = lb?.rows || [];
  return (
    <div>
      <h1 className="page-title">Leaderboard</h1>
      <p className="page-sub">
        Best test accuracy per algorithm under identical compute budgets. Only
        multi-seed-validated candidates are promoted as discoveries.
      </p>
      <div className="panel">
        {rows.length ? (
          <table className="data">
            <thead>
              <tr>
                <th>Rank</th><th>Method</th><th>Kind</th><th>Score</th>
                <th>RAM</th><th>Time</th><th>Epochs</th><th>Gen</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={r.name}>
                  <td className="mono">{i + 1}</td>
                  <td className="mono">{r.name}</td>
                  <td>{kindBadge(r.kind)}</td>
                  <td style={{ fontWeight: 700, color: i === 0 ? 'var(--green)' : undefined }}>
                    {fmt.pct(r.score)}
                  </td>
                  <td>{fmt.mb(r.ram_mb)}</td>
                  <td>{fmt.secs(r.seconds)}</td>
                  <td>{r.epochs ?? '—'}</td>
                  <td>{r.generation ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="empty"><div className="big">🏆</div>No results yet — run a research session.</div>
        )}
      </div>
    </div>
  );
}

// ================================================================ Journal

function JournalSection() {
  const [entries, , reload] = usePoll(() => api.journal(200), 5000, []);
  const rows = entries || [];
  const render = (e) => {
    switch (e.type) {
      case 'experiment':
        return (
          <span>
            <b className="mono">{e.name}</b> — {e.result}
            {e.score != null && <> · score <b>{fmt.pct(e.score)}</b></>}
          </span>
        );
      case 'generation':
        return <>— generation {e.generation} begins ({e.population} algorithms) —</>;
      case 'baselines_complete':
        return (
          <>Baselines established: {Object.entries(e.scores || {}).map(([k, v]) => (
            <span key={k} className="mono" style={{ marginRight: 10 }}>{k}={fmt.pct(v)}</span>
          ))}</>
        );
      case 'promising':
        return <span style={{ color: 'var(--green)' }}>★ PROMISING: <b className="mono">{e.name}</b> — {fmt.pct(e.score)} {e.note || ''}</span>;
      case 'validation':
        return (
          <span style={{ color: e.confirmed ? 'var(--green)' : 'var(--yellow)' }}>
            Validation: <b className="mono">{e.name}</b> mean {fmt.pct(e.mean)} over {e.seeds} seeds —{' '}
            {e.confirmed ? 'confirmed ✓' : 'not confirmed'}
          </span>
        );
      case 'ablation':
        return <span>Ablation study on <b className="mono">{e.name}</b></span>;
      case 'failure_analysis':
        return (
          <span style={{ color: 'var(--red)' }}>
            Failure: <b className="mono">{e.name}</b> — {e.analysis?.cause}: {e.analysis?.explanation}
          </span>
        );
      case 'phase':
        return <span style={{ color: 'var(--accent)' }}>— {e.phase} —</span>;
      case 'research_start':
        return <span style={{ color: 'var(--accent)' }}>▶ Research started ({e.run_id})</span>;
      case 'research_end':
        return <span style={{ color: 'var(--accent)' }}>■ Research ended — {e.experiments} experiments, {e.generations} generations</span>;
      case 'checkpoint_resumed':
        return <span>Checkpoint resumed</span>;
      case 'fatal_error':
        return <span style={{ color: 'var(--red)' }}>FATAL: {e.error}</span>;
      default:
        return <span>{e.type}</span>;
    }
  };
  return (
    <div>
      <h1 className="page-title">Research Journal</h1>
      <p className="page-sub">
        The complete, append-only log of the research process — every decision,
        every experiment, every failure analysis.
      </p>
      <div className="panel">
        {rows.length ? (
          <div className="scroll-y" style={{ maxHeight: 600 }}>
            {rows.slice().reverse().map((e, i) => (
              <div className="journal-entry" key={i}>
                <span className="time">{fmt.time(e.ts)}</span>
                <span className="body">{render(e)}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty"><div className="big">📓</div>Journal is empty — run a research session.</div>
        )}
      </div>
    </div>
  );
}

// ================================================================ Playground

function ChatSection() {
  const [messages, setMessages] = useState([]); // {role: 'u'|'a', text}
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [chatStatus, reloadChat] = usePoll(api.chatStatus, 4000, []);
  const [training, setTraining] = useState(false);
  const [minutes, setMinutes] = useState(30);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const trained = chatStatus?.trained;
  const isTraining = chatStatus?.training;
  const prog = chatStatus?.progress || {};

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput('');
    setMessages((m) => [...m, { role: 'u', text }]);
    setBusy(true);
    try {
      const history = messages.slice(-6).map((m) => ({
        speaker: m.role,
        text: m.text,
      }));
      const r = await api.chat(text, history);
      setMessages((m) => [...m, { role: 'a', text: r.reply }]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: 'a', text: `⚠️ ${e.message}` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function startTraining() {
    setTraining(true);
    try {
      await api.chatTrain({ minutes: Number(minutes) });
      reloadChat();
    } catch (e) {
      alert(`Training failed to start: ${e.message}`);
    } finally {
      setTraining(false);
    }
  }

  return (
    <section className="section">
      <h2>💬 TinyGPT Chat</h2>
      <p className="muted">
        A real decoder-only transformer (pure NumPy) trained on{' '}
        <b>11,118 real human conversations</b> (DailyDialog). It generates
        replies token-by-token with next-token prediction — no retrieval, no
        templates. Small model, honest quality: replies are simple English.
      </p>

      <div className="chat-status-bar">
        {trained ? (
          <span className="ok-chip">
            ✓ Model ready — {chatStatus.meta?.params?.toLocaleString()} params,
            vocab {chatStatus.meta?.vocab}, loss{' '}
            {chatStatus.meta?.final_loss?.toFixed(3)}
          </span>
        ) : isTraining ? (
          <span className="warn-chip">
            ⏳ Training — {prog.stage}, step {prog.steps || 0}, loss{' '}
            {prog.loss ?? '—'}, {prog.elapsed_min ?? 0} min
          </span>
        ) : (
          <span className="muted">No chat model yet — train it below.</span>
        )}
        {chatStatus?.error && (
          <div className="error-banner">{chatStatus.error}</div>
        )}
      </div>

      <div className="chat-box">
        {messages.length === 0 && (
          <div className="muted chat-hint">
            Say something in English, e.g. <i>“how was your day?”</i>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <div className="bubble">{m.text}</div>
          </div>
        ))}
        {busy && <div className="chat-msg a"><div className="bubble">…</div></div>}
        <div ref={endRef} />
      </div>

      <div className="chat-input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder={trained ? 'Type a message…' : 'Train the model first'}
          disabled={!trained || busy}
        />
        <button onClick={send} disabled={!trained || busy || !input.trim()}>
          Send
        </button>
      </div>

      {!trained && !isTraining && (
        <div className="chat-train-row">
          <label>
            Training minutes:{' '}
            <input
              type="number"
              min="5"
              max="240"
              value={minutes}
              onChange={(e) => setMinutes(e.target.value)}
              style={{ width: 70 }}
            />
          </label>
          <button
            className="primary"
            onClick={startTraining}
            disabled={training}
          >
            {training ? 'Starting…' : `🚂 Train TinyGPT (${minutes} min)`}
          </button>
        </div>
      )}
    </section>
  );
}

function PlaygroundSection() {
  const [text, setText] = useState('Congratulations! You won a $5000 prize, click to claim now!');
  const [result, setResult] = useState(null);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState(null);

  useEffect(() => {
    api.model().then(setModel).catch(() => {});
  }, []);

  // Speak the prediction out loud using the browser's speech synthesis
  const speak = (r) => {
    if (!r || !('speechSynthesis' in window)) return;
    const conf = Math.round((r.confidence || 0) * 100);
    const utter = new SpeechSynthesisUtterance(
      `The message is ${r.label}. Confidence ${conf} percent.`
    );
    utter.lang = 'en-US';
    utter.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utter);
  };

  const run = async () => {
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const r = await api.playground(text.trim() ? { text: text } : {});
      setResult(r);
      speak(r); // announce the prediction out loud
    } catch (e) {
      setErr(String(e.message || e));
    }
    setBusy(false);
  };

  const isSpamDataset = model?.dataset === 'spam';

  return (
    <div>
      <h1 className="page-title">Playground</h1>
      <p className="page-sub">
        Real predictions from the final trained model — no mocks, no fake
        numbers. Every probability below is computed by the discovered algorithm's
        trained weights.
      </p>

      <div className="panel">
        <h3>Input</h3>
        {isSpamDataset ? (
          <div className="field">
            <label>Message text</label>
            <textarea
              rows="3"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type an SMS message to classify…"
            />
            <div className="hint">
              The text is hashed into uni+bigram features with the same vectorizer
              used during training.
            </div>
          </div>
        ) : (
          <div className="field">
            <label>Feature vector ({model?.in_dim ?? '?'} dims, comma-separated)</label>
            <textarea
              rows="3"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="0.5, -0.2, 1.0, …"
            />
            <div className="hint">
              This model expects a raw numeric feature vector of dimension {model?.in_dim ?? '?'}.
            </div>
          </div>
        )}
        <button className="btn" onClick={run} disabled={busy}>
          {busy ? 'Predicting…' : 'Predict'}
        </button>
        <span style={{ marginLeft: 10, fontSize: 12, color: 'var(--text-faint)' }}>
          The result is also spoken out loud automatically.
        </span>
      </div>

      {err && <div className="error-banner">{err}</div>}

      {result && (
        <div className="pred-result">
          <div className="label-big">{result.label}</div>
          <div className="conf">confidence {fmt.pct(result.confidence)}</div>
          <button
            className="btn secondary"
            style={{ marginTop: 12 }}
            onClick={() => speak(result)}
            title="Speak the prediction out loud"
          >
            🔊 Speak
          </button>
          <div style={{ marginTop: 16 }}>
            {Object.entries(result.probabilities || {})
              .sort((a, b) => b[1] - a[1])
              .map(([label, p]) => (
                <div className="prob-row" key={label}>
                  <span className="name">{label}</span>
                  <div className="bar">
                    <div className="fill" style={{ width: `${Math.min(100, p * 100)}%` }} />
                  </div>
                  <span className="pct">{(p * 100).toFixed(2)}%</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ================================================================ Report

function ReportSection() {
  const [report, setReport] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    api.report().then((r) => setReport(r.markdown)).catch((e) => setErr(String(e.message || e)));
  }, []);

  // minimal markdown → HTML (headings, bold, code, tables, lists)
  const md2html = (md) => {
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const lines = esc(md).split('\n');
    let html = '';
    let inList = false;
    let inTable = false;
    const inline = (s) =>
      s
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
        .replace(/\*([^*]+)\*/g, '<i>$1</i>');
    for (const line of lines) {
      const l = line.trim();
      if (l.startsWith('|')) {
        if (!inTable) { html += '<table>'; inTable = true; }
        const cells = l.split('|').slice(1, -1);
        if (cells.every((c) => /^[-: ]+$/.test(c.trim()))) continue;
        html += '<tr>' + cells.map((c) => `<td>${inline(c.trim())}</td>`).join('') + '</tr>';
        continue;
      }
      if (inTable) { html += '</table>'; inTable = false; }
      if (/^#{1,3} /.test(l)) {
        const n = l.match(/^#+/)[0].length;
        html += `<h${n}>${inline(l.replace(/^#+ /, ''))}</h${n}>`;
      } else if (/^[-*] /.test(l)) {
        if (!inList) { html += '<ul>'; inList = true; }
        html += `<li>${inline(l.slice(2))}</li>`;
      } else if (/^> /.test(l)) {
        html += `<blockquote>${inline(l.slice(2))}</blockquote>`;
      } else if (l === '') {
        if (inList) { html += '</ul>'; inList = false; }
      } else {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<p>${inline(l)}</p>`;
      }
    }
    if (inList) html += '</ul>';
    if (inTable) html += '</table>';
    return html;
  };

  return (
    <div>
      <h1 className="page-title">Final Research Report</h1>
      <p className="page-sub">
        The serious, honest report generated after research completes — including
        limitations and statistical caveats. No fake novelty.
      </p>
      {err && (
        <div className="empty">
          <div className="big">📄</div>
          {err.includes('no report') ? 'No report yet — research must complete first.' : err}
        </div>
      )}
      {report && (
        <div className="report-body" dangerouslySetInnerHTML={{ __html: md2html(report) }} />
      )}
    </div>
  );
}

// ================================================================ Settings

function SettingsSection({ hardware }) {
  return (
    <div>
      <h1 className="page-title">Settings</h1>
      <p className="page-sub">Engine configuration and environment.</p>
      <div className="panel">
        <h3>Engine</h3>
        <div className="kv">
          <span className="k">Version</span><span className="v mono">1.0.0</span>
          <span className="k">Backend</span><span className="v">Pure NumPy — no GPU, no ML frameworks</span>
          <span className="k">Server</span><span className="v mono">FastAPI on :8000</span>
          <span className="k">Research database</span><span className="v mono">research/ (experiments, lineage, journal, checkpoints)</span>
          <span className="k">Model artifacts</span><span className="v mono">models/best_model/</span>
        </div>
      </div>
      <div className="panel">
        <h3>Weak-hardware mode</h3>
        <p style={{ color: 'var(--text-dim)' }}>
          Automatically enabled on machines with ≤2 cores or &lt;5 GB RAM. The
          engine then prioritizes memory-saving mechanisms (int8/fp16 optimizer
          state, sparse gradients, partial updates) so training stays practical on
          low-end CPUs. Current status:{' '}
          <b style={{ color: hardware?.weak_hardware ? 'var(--yellow)' : 'var(--green)' }}>
            {hardware?.weak_hardware ? 'ENABLED' : 'disabled'}
          </b>
          .
        </p>
      </div>
      <div className="panel">
        <h3>Reproducibility</h3>
        <p style={{ color: 'var(--text-dim)' }}>
          All experiments use fixed seeds, identical 70/15/15 data splits, and
          identical epoch/time budgets per algorithm. Every experiment —
          successful or failed — is permanently stored in the research database.
        </p>
      </div>
    </div>
  );
}

// ================================================================ App

const SECTIONS = [
  { id: 'research', label: 'Research', ico: '🔬' },
  { id: 'algorithms', label: 'Algorithms', ico: '🧬' },
  { id: 'experiments', label: 'Experiments', ico: '⚗️' },
  { id: 'models', label: 'Models', ico: '📦' },
  { id: 'datasets', label: 'Datasets', ico: '🗂' },
  { id: 'hardware', label: 'Hardware', ico: '🖥' },
  { id: 'leaderboard', label: 'Leaderboard', ico: '🏆' },
  { id: 'journal', label: 'Research Journal', ico: '📓' },
  { id: 'chat', label: 'Chat', ico: '💬' },
  { id: 'playground', label: 'Playground', ico: '🎮' },
  { id: 'report', label: 'Report', ico: '📄' },
  { id: 'settings', label: 'Settings', ico: '⚙️' },
];

export default function App() {
  const [section, setSection] = useState('research');
  const [status, , reloadStatus] = usePoll(api.status, 3000, []);

  const hw = status?.hardware;
  const st = status?.status || 'idle';

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <span className="chip">⚡</span>
          <span>CPU Research Engine</span>
        </div>
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            className={`nav-btn ${section === s.id ? 'active' : ''}`}
            onClick={() => setSection(s.id)}
          >
            <span className="ico">{s.ico}</span>
            <span className="txt">{s.label}</span>
          </button>
        ))}
        <div className="spacer" />
        <div className="status-pill">
          <span className={`status-dot ${st}`} />
          <span>
            {st === 'researching' ? 'Researching…' : st === 'complete' ? 'Complete' : 'Idle'}
          </span>
        </div>
      </aside>

      <main className="main">
        {status?.last_error && <div className="error-banner">{status.last_error}</div>}
        {section === 'research' && <ResearchSection status={status} />}
        {section === 'algorithms' && <AlgorithmsSection />}
        {section === 'experiments' && <ExperimentsSection />}
        {section === 'models' && <ModelsSection status={status} />}
        {section === 'datasets' && <DatasetsSection />}
        {section === 'hardware' && <HardwareSection hardware={hw} />}
        {section === 'leaderboard' && <LeaderboardSection />}
        {section === 'journal' && <JournalSection />}
        {section === 'chat' && <ChatSection />}
        {section === 'playground' && <PlaygroundSection />}
        {section === 'report' && <ReportSection />}
        {section === 'settings' && <SettingsSection hardware={hw} />}
      </main>
    </div>
  );
}
