// Thin API client for the research engine backend (FastAPI :8000)

async function req(path, opts) {
  const r = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = await r.json();
      detail = body.detail || JSON.stringify(body);
    } catch { /* ignore */ }
    throw new Error(detail);
  }
  return r.json();
}

export const api = {
  status: () => req('/api/status'),
  dashboard: () => req('/api/dashboard'),
  experiments: () => req('/api/experiments'),
  experiment: (id) => req(`/api/experiments/${id}`),
  lineage: () => req('/api/lineage'),
  journal: (limit = 100) => req(`/api/journal?limit=${limit}`),
  leaderboard: () => req('/api/leaderboard'),
  hardware: () => req('/api/hardware'),
  datasets: () => req('/api/datasets'),
  model: () => req('/api/model'),
  report: () => req('/api/report'),
  startResearch: (cfg) =>
    req('/api/research/start', { method: 'POST', body: JSON.stringify(cfg) }),
  stopResearch: () => req('/api/research/stop', { method: 'POST' }),
  playground: (payload) =>
    req('/api/playground', { method: 'POST', body: JSON.stringify(payload) }),
  chatStatus: () => req('/api/chat/status'),
  chat: (message, history) =>
    req('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message, history }),
    }),
  chatTrain: (cfg = {}) =>
    req('/api/chat/train', { method: 'POST', body: JSON.stringify(cfg) }),
};

export const fmt = {
  pct: (x) => (x == null ? '—' : `${(x * 100).toFixed(2)}%`),
  pct1: (x) => (x == null ? '—' : `${(x * 100).toFixed(1)}%`),
  num: (x, d = 2) => (x == null ? '—' : Number(x).toFixed(d)),
  secs: (x) => {
    if (x == null) return '—';
    if (x < 60) return `${x.toFixed(1)}s`;
    if (x < 3600) return `${Math.floor(x / 60)}m ${Math.round(x % 60)}s`;
    return `${Math.floor(x / 3600)}h ${Math.round((x % 3600) / 60)}m`;
  },
  mb: (x) => (x == null ? '—' : `${x.toFixed(1)} MB`),
  time: (ts) => {
    try {
      return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return ts;
    }
  },
};
