import { useState, useEffect, useRef } from 'react'
import {
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import {
  MdDashboard, MdSearch, MdHistory, MdNotifications, MdSettings,
  MdShield, MdCircle, MdTrendingUp, MdWarning, MdCheckCircle,
  MdError, MdPlayArrow, MdStop, MdLightMode, MdDarkMode,
  MdAnalytics, MdFilterList, MdClose, MdChevronRight
} from 'react-icons/md'
import './App.css'

// ─── Seed chart data ─────────────────────────────────────────────────────────
const HOURS = ['00h','02h','04h','06h','08h','10h','12h','14h','16h','18h','20h','22h']
const seedBarData = HOURS.map(h => ({ hour: h, anomalies: Math.floor(Math.random() * 8) }))

// Set VITE_API_BASE_URL in frontend/.env to point at your running backend
// (see backend/app.py — a Kaggle/Colab-hosted server exposed via ngrok, or
// any host running `uvicorn backend.app:app`).
const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const API_URL =`${API_BASE}/analyze`;
async function fetchLiveLog() {
  const response = await fetch(
    `${API_BASE}/live-log`,
    {
      headers: {
        "ngrok-skip-browser-warning": "true"
      }
    }
  );

  return await response.json();
}

// ─── Stat Card ───────────────────────────────────────────────────────────────
function StatCard({ icon, value, label, accent, delta }) {
  return (
    <div className={`stat-card stat-card--${accent}`}>
      <div className="stat-card__icon">{icon}</div>
      <div className="stat-card__body">
        <span className="stat-card__value">{value}</span>
        <span className="stat-card__label">{label}</span>
        {delta !== undefined && (
          <span className={`stat-card__delta ${delta >= 0 ? 'pos' : 'neg'}`}>
            {delta >= 0 ? '+' : ''}{delta} this session
          </span>
        )}
      </div>
    </div>
  )
}

// ─── Alert Badge ─────────────────────────────────────────────────────────────
function AlertBadge({ item, onDismiss }) {
  return (
    <div className="alert-badge">
      <MdError className="alert-badge__icon" />
      <div className="alert-badge__body">
        <div className="alert-badge__meta">
          <span className="alert-badge__tag">CRITICAL</span>
          <span className="alert-badge__block">{item.blockId}</span>
          <span className="alert-badge__time">{item.timestamp}</span>
        </div>
        <p className="alert-badge__text">{item.explanation?.slice(0, 140)}…</p>
      </div>
      <button className="alert-badge__close" onClick={() => onDismiss(item.id)}>
        <MdClose />
      </button>
    </div>
  )
}

// ─── History Row ─────────────────────────────────────────────────────────────
function HistoryRow({ entry, idx }) {
  return (
    <tr className={idx % 2 === 0 ? 'row-even' : 'row-odd'}>
      <td>{entry.timestamp}</td>
      <td className="mono">{entry.blockId}</td>
      <td>
        <span className={`badge badge--${entry.status === 'Normal' ? 'ok' : 'err'}`}>
          {entry.status}
        </span>
      </td>
      <td className="mono">{entry.latency}s</td>
    </tr>
  )
}

const PIE_COLORS = ['#10b981', '#ef4444']
const getConfidence = (status, latency, events) => {
  let score = status === "Anomalous" ? 82 : 95;

  if (latency > 15) score -= 8;
  if (events.includes("E22")) score -= 5;
  if (events.includes("E11")) score -= 3;

  return Math.max(60, Math.min(score, 99));
};

const getSeverity = (latency, status) => {
  if (status === "Normal") return "Low";

  if (latency > 20) return "Critical";
  if (latency > 15) return "High";
  if (latency > 10) return "Medium";

  return "Low";
};

const getRecommendations = (events, status) => {
  if (status === "Normal") {
    return [
      "Continue monitoring",
      "No immediate action required"
    ];
  }

  let actions = [];

  if (events.includes("E5"))
    actions.push("Check DataNode availability");

  if (events.includes("E11"))
    actions.push("Verify PacketResponder process");

  if (events.includes("E22"))
    actions.push("Inspect replication pipeline");

  actions.push("Review recent cluster logs");

  return actions;
};
// ─── Main App ────────────────────────────────────────────────────────────────
export default function App() {
  const [dark, setDark] = useState(() => localStorage.getItem('ls-theme') === 'dark')
  const [activeNav, setActiveNav] = useState('analyze')

  // Form state
  const [blockId,      setBlockId]      = useState('blk_987654321')
  const [latency,      setLatency]      = useState('14.2')
  const [semanticText, setSemanticText] = useState(
    'Receiving block blk_987654321, Exception in receiveBlock for block blk_987654321, PacketResponder 1 for block blk_987654321 terminating'
  )
  const [eventIds, setEventIds] = useState('E5, E11, E22')

  // API state
  const [loading,   setLoading]   = useState(false)
  const [result,    setResult]    = useState(null)
  const [error,     setError]     = useState(null)
  const [connected, setConnected] = useState(null)

  // Stats
  const [stats, setStats] = useState({ total: 0, anomalies: 0, healthy: 0, avgLatency: 0 })

  // History + alerts
  const [history,      setHistory]      = useState([])
  const [historySearch,setHistorySearch] = useState('')
  const [historySort,  setHistorySort]   = useState('desc')
  const [alerts,       setAlerts]        = useState([])

  // Charts
  const [barData] = useState(seedBarData)

  // Live monitor
  const [liveOn,   setLiveOn]   = useState(false)
  const [liveLogs, setLiveLogs] = useState([])
  const liveRef  = useRef(null)
  const timerRef = useRef(null)

  const populateFromLiveLogs = () => {

  if (liveLogs.length === 0) return;

  // Last 5 logs
  const recentLogs = liveLogs
    .slice(0, 5)
    .map(log => log.text);

  // Fill Semantic Text box
  setSemanticText(recentLogs.join(", "));

  // Extract Block ID from newest log
  const latestText = recentLogs[0];

  const match = latestText.match(/blk_[-\d]+/);

 if (match) {
  setBlockId(match[0]);
} else {
  setBlockId(`blk_${Date.now()}`);
}

// Generate Event IDs from log content
let events = [];

if (latestText.includes("PacketResponder"))
  events.push("E11");

if (latestText.includes("Received block"))
  events.push("E22");

if (latestText.includes("terminating"))
  events.push("E5");

if (events.length === 0)
  events.push("E1");

setEventIds(events.join(", "));

setLatency(
  (5 + Math.random() * 15).toFixed(1)
);
  setActiveNav("analyze");
};
  
const copyReport = () => {
  if (!result) return;

  navigator.clipboard.writeText(`
Status: ${result.status}
Confidence: ${result.confidence}%
Severity: ${result.severity}
Explanation:
${result.explanation}
`);
};

const exportPDF = () => {
  window.print();
};
  // Theme
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light')
    localStorage.setItem('ls-theme', dark ? 'dark' : 'light')
  }, [dark])

  // Live monitor ticker
  useEffect(() => {
  if (liveOn) {

    timerRef.current = setInterval(async () => {

      try {

        const log = await fetchLiveLog();

        setLiveLogs(prev =>
          [log, ...prev].slice(0, 80)
        );

      } catch (err) {
        console.error(err);
      }

    }, 1000);

  } else {
    clearInterval(timerRef.current);
  }

  return () => clearInterval(timerRef.current);

}, [liveOn]);

  // Backend ping
  useEffect(() => {
    const ping = async () => {
      try {
        const r = await fetch(API_URL.replace('/analyze', '/'), {
          headers: { 'ngrok-skip-browser-warning': 'true' },
          signal: AbortSignal.timeout(4000)
        })
        setConnected(r.ok || r.status < 500)
      } catch {
        setConnected(false)
      }
    }
    ping()
    const id = setInterval(ping, 30000)
    return () => clearInterval(id)
  }, [])

  // Analyze handler
  const handleAnalyze = async (e) => {
    e.preventDefault()
    setLoading(true); setResult(null); setError(null)

    const payload = {
      block_id: blockId,
      semantic_text_sequence: semanticText.split(',').map(s => s.trim()).filter(Boolean),
      event_id_sequence: eventIds.split(',').map(s => s.trim()),
      latency: parseFloat(latency)
    }

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
        body: JSON.stringify(payload)
      })
      if (!response.ok) throw new Error(`API Error: ${response.status}`)
      const data = await response.json()
    const confidence = getConfidence(
  data.status,
  parseFloat(latency),
  eventIds
);

const severity = getSeverity(
  parseFloat(latency),
  data.status
);

data.confidence = confidence;
data.severity = severity;
      setResult(data)
      setConnected(true)

      const entry = {
        id: Date.now(),
        timestamp: new Date().toLocaleTimeString(),
        blockId,
        status: data.status,
        latency: parseFloat(latency),
        explanation: data.explanation
      }
      setHistory(prev => [entry, ...prev])

      setStats(prev => {
        const total      = prev.total + 1
        const anomalies  = prev.anomalies + (data.status === 'Anomalous' ? 1 : 0)
        const healthy    = prev.healthy   + (data.status === 'Normal'    ? 1 : 0)
        const avgLatency = ((prev.avgLatency * prev.total) + parseFloat(latency)) / total
        return { total, anomalies, healthy, avgLatency }
      })

      if (data.status === 'Anomalous') {
        setAlerts(prev => [{
          id: Date.now(),
          blockId,
          timestamp: new Date().toLocaleTimeString(),
          explanation: data.explanation
        }, ...prev])
      }
    } catch (err) {
      setError(err.message)
      setConnected(false)
    } finally {
      setLoading(false)
    }
  }

  // Derived data
  const pieData = [
    { name: 'Healthy',   value: Math.max(stats.healthy,   stats.total === 0 ? 1 : 0) },
    { name: 'Anomalous', value: Math.max(stats.anomalies, 0) }
  ]

  const filteredHistory = history
    .filter(h =>
      h.blockId.toLowerCase().includes(historySearch.toLowerCase()) ||
      h.status.toLowerCase().includes(historySearch.toLowerCase())
    )
    .sort((a, b) => historySort === 'desc' ? b.id - a.id : a.id - b.id)

  const navItems = [
    { id: 'dashboard', icon: <MdDashboard />,      label: 'Dashboard'    },
    { id: 'analyze',   icon: <MdAnalytics />,      label: 'Analyze Logs' },
    { id: 'history',   icon: <MdHistory   />,      label: 'History'      },
    { id: 'alerts',    icon: <MdNotifications />,  label: 'Alerts',       badge: alerts.length || null },
    { id: 'settings',  icon: <MdSettings  />,      label: 'Settings'     },
  ]

  return (
    <div className="ls-root">

      {/* ─── Sidebar ─────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar__brand">
          <MdShield className="brand-icon" />
          <span className="brand-name">LogSense</span>
        </div>

        <nav className="sidebar__nav">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`nav-item ${activeNav === item.id ? 'nav-item--active' : ''}`}
              onClick={() => setActiveNav(item.id)}
            >
              <span className="nav-item__icon">{item.icon}</span>
              <span className="nav-item__label">{item.label}</span>
              {item.badge && <span className="nav-item__badge">{item.badge}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar__footer">
          <MdCircle className={`conn-dot conn-dot--${connected === null ? 'unk' : connected ? 'ok' : 'err'}`} />
          <span className="conn-label">
            {connected === null ? 'Checking…' : connected ? 'API Online' : 'API Offline'}
          </span>
        </div>
      </aside>

      {/* ─── Main wrapper ────────────────────────────────────────── */}
      <div className="main-wrapper">

        {/* Topbar */}
        <header className="topbar">
          <div className="topbar__left">
            <span className="topbar__page">{navItems.find(n => n.id === activeNav)?.label}</span>
            <span className="topbar__sub">LogSense AI · Real-Time Dual-GPU Anomaly Analyzer</span>
          </div>
          <div className="topbar__right">
            <button
              className={`live-toggle ${liveOn ? 'live-toggle--on' : ''}`}
              onClick={() => {

  if (liveOn) {
    populateFromLiveLogs();
  }

  setLiveOn(v => !v);

}}
            >
              {liveOn ? <><MdStop /> Stop Monitor</> : <><MdPlayArrow /> Live Monitor</>}
            </button>
            <button className="theme-btn" onClick={() => setDark(d => !d)}>
              {dark ? <MdLightMode /> : <MdDarkMode />}
            </button>
          </div>
        </header>

        {/* Live terminal */}
        {liveOn && (
          <div className="terminal-bar" ref={liveRef}>
            {liveLogs.length === 0
              ? <span className="terminal-idle">Waiting for log events…</span>
              : liveLogs.map((l, i) => (
                <div key={i} className={`t-line t-line--${l.lvl.toLowerCase()}`}>
                  <span className="t-ts">{l.ts}</span>
                  <span className="t-text">{l.text}</span>
                </div>
              ))
            }
          </div>
        )}

        {/* Page content */}
        <main className="content">

          {/* ═══ DASHBOARD ══════════════════════════════════════════ */}
          {activeNav === 'dashboard' && <>
            <div className="stat-grid">
              <StatCard icon={<MdTrendingUp />}  value={stats.total}     label="Total Processed"  accent="blue"  delta={stats.total} />
              <StatCard icon={<MdWarning />}      value={stats.anomalies} label="Anomalies"        accent="red"   delta={stats.anomalies} />
              <StatCard icon={<MdCheckCircle />}  value={stats.healthy}   label="Healthy Logs"     accent="green" delta={stats.healthy} />
              <StatCard icon={<MdAnalytics />}    value={stats.total > 0 ? stats.avgLatency.toFixed(2)+'s' : '—'} label="Avg Latency" accent="amber" />
            </div>

            <div className="charts-grid">
              <div className="chart-card">
                <h3 className="chart-card__title">Health Distribution</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" cx="50%" cy="50%" outerRadius={90}
                      label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`}
                      labelLine={false}>
                      {pieData.map((_, i) => <Cell key={i} fill={PIE_COLORS[i]} />)}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-card">
                <h3 className="chart-card__title">Anomalies by Hour <span className="chart-sample">(sample data)</span></h3>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={barData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="hour" tick={{ fontSize: 10 }} stroke="var(--text-muted)" />
                    <YAxis tick={{ fontSize: 10 }} stroke="var(--text-muted)" />
                    <Tooltip />
                    <Bar dataKey="anomalies" fill="var(--accent)" radius={[4,4,0,0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>}

          {/* ═══ ANALYZE ════════════════════════════════════════════ */}
          {activeNav === 'analyze' && (
            <div className="analyze-layout">
              {/* Left: form */}
              <div className="panel">
                <h2 className="panel__title"><MdAnalytics /> Incoming Log Block</h2>
                <form onSubmit={handleAnalyze} className="log-form" noValidate>
                  <div className="field">
                    <label className="field__label">Block ID</label>
                    <input className="field__input" type="text" value={blockId}
                      onChange={e => setBlockId(e.target.value)} placeholder="blk_987654321" required />
                  </div>

                  <div className="field">
                    <label className="field__label">Latency <em className="field__unit">seconds</em></label>
                    <input className="field__input" type="number" step="0.01" min="0" value={latency}
                      onChange={e => setLatency(e.target.value)} required />
                  </div>

                  <div className="field">
                    <label className="field__label">Semantic Text Sequence <em className="field__unit">comma separated</em></label>
                    <textarea className="field__input field__textarea" rows={4} value={semanticText}
                      onChange={e => setSemanticText(e.target.value)} required />
                  </div>

                  <div className="field">
                    <label className="field__label">Event ID Sequence <em className="field__unit">comma separated</em></label>
                    <input className="field__input" type="text" value={eventIds}
                      onChange={e => setEventIds(e.target.value)} placeholder="E5, E11, E22" required />
                  </div>

                  <button type="submit" className="analyze-btn" disabled={loading}>
                    {loading
                      ? <><span className="spinner" /> Running LogLLM…</>
                      : <><MdChevronRight /> Analyze Logs</>}
                  </button>
                </form>
              </div>

              {/* Right: result */}
              <div className="panel">
                <h2 className="panel__title"><MdShield /> Diagnostic Report</h2>

                {!result && !error && !loading && (
                  <div className="result-placeholder">
                    <MdAnalytics className="rph-icon" />
                    <p>Submit a log block to run the hybrid<br />LogLLM + Llama 3B pipeline.</p>
                  </div>
                )}

                {loading && (
                  <div className="result-loading">
                    <div className="pulse-ring" />
                    <p className="rl-title">Running LogLLM Classification…</p>
                    <p className="rl-sub">If anomalous, Llama 3B will generate a root-cause report.</p>
                  </div>
                )}

                {error && (
                  <div className="result-card result-card--error">
                    <div className="rc-head"><MdError /> Connection Refused</div>
                    <p className="rc-body">{error}</p>
                    <p className="rc-hint">Ensure your Kaggle notebook is active and the Ngrok URL is correct.</p>
                  </div>
                )}

                {result && (
<div
className={`result-card ${
result.status === "Normal"
? "result-card--ok"
: "result-card--warn"
}`}
>

<div className="rc-head">
  {result.status === "Normal"
    ? <MdCheckCircle />
    : <MdWarning />
  }

  {result.status}
</div>

<div className="meta-chips">
  <span className="chip">
    Confidence: {result.confidence}%
  </span>

  <span className="chip">
    Severity: {result.severity}
  </span>

  <span className="chip">
    Latency: {latency}s
  </span>
</div>

<div className="rc-section">
  <span className="rc-section__label">
    Explainability Report
  </span>

  <p className="rc-body">
    <b>Block ID:</b> {blockId}
    <br />

    <b>Prediction:</b> {result.status}
    <br />

    <b>Confidence:</b> {result.confidence}%
    <br />

    <b>Severity:</b> {result.severity}
    <br />

    <b>Trigger Events:</b> {eventIds}
    <br />

    <b>Latency:</b> {latency}s
    <br /><br />

    <b>Root Cause:</b>
    <br />

    {result.explanation}
  </p>
</div>

<div className="rc-section">
  <span className="rc-section__label">
    Recommended Actions
  </span>

  <ul className="recommend-list">
    {getRecommendations(
      eventIds,
      result.status
    ).map((item, idx) => (
      <li key={idx}>{item}</li>
    ))}
  </ul>
</div>

<div
style={{
display: "flex",
gap: "10px",
marginTop: "20px"
}}
>
<button
className="analyze-btn"
type="button"
onClick={copyReport}
>
Copy Analysis
</button>

<button
className="analyze-btn"
type="button"
onClick={exportPDF}
>
Export PDF
</button>
</div>

</div>
)}

              </div>
            </div>
          )}

          {/* ═══ HISTORY ════════════════════════════════════════════ */}
          {activeNav === 'history' && (
            <div className="panel panel--full">
              <div className="panel-titlebar">
                <h2 className="panel__title"><MdHistory /> Analysis History</h2>
                <div className="history-controls">
                  <div className="search-box">
                    <MdSearch className="sb-icon" />
                    <input className="sb-input" placeholder="Search by Block ID or status…"
                      value={historySearch} onChange={e => setHistorySearch(e.target.value)} />
                  </div>
                  <button className="sort-btn" onClick={() => setHistorySort(s => s === 'desc' ? 'asc' : 'desc')}>
                    <MdFilterList /> {historySort === 'desc' ? 'Newest first' : 'Oldest first'}
                  </button>
                </div>
              </div>

              {filteredHistory.length === 0
                ? (
                  <div className="result-placeholder">
                    <MdHistory className="rph-icon" />
                    <p>No analyses yet. Run an analysis to see results here.</p>
                  </div>
                ) : (
                  <div className="table-wrapper">
                    <table className="ls-table">
                      <thead><tr>
                        <th>Timestamp</th><th>Block ID</th><th>Status</th><th>Latency</th>
                      </tr></thead>
                      <tbody>
                        {filteredHistory.map((entry, i) =>
                          <HistoryRow key={entry.id} entry={entry} idx={i} />
                        )}
                      </tbody>
                    </table>
                  </div>
                )
              }
            </div>
          )}

          {/* ═══ ALERTS ═════════════════════════════════════════════ */}
          {activeNav === 'alerts' && (
            <div className="panel panel--full">
              <div className="panel-titlebar">
                <h2 className="panel__title"><MdNotifications /> Active Alerts</h2>
                {alerts.length > 0 &&
                  <button className="clear-btn" onClick={() => setAlerts([])}>Clear all</button>}
              </div>
              {alerts.length === 0
                ? (
                  <div className="result-placeholder">
                    <MdCheckCircle className="rph-icon rph-icon--ok" />
                    <p>No active alerts. System looks healthy.</p>
                  </div>
                ) : (
                  <div className="alerts-list">
                    {alerts.map(a => (
                      <AlertBadge key={a.id} item={a}
                        onDismiss={id => setAlerts(prev => prev.filter(x => x.id !== id))} />
                    ))}
                  </div>
                )
              }
            </div>
          )}

          {/* ═══ SETTINGS ═══════════════════════════════════════════ */}
          {activeNav === 'settings' && (
            <div className="panel panel--full">
              <h2 className="panel__title"><MdSettings /> Settings</h2>
              <div className="settings-grid">
                <div className="setting-row">
                  <div>
                    <div className="sr-label">API Endpoint</div>
                    <div className="sr-sub">FastAPI backend (Kaggle + Ngrok)</div>
                  </div>
                  <span className="mono sr-val">{API_URL}</span>
                </div>
                <div className="setting-row">
                  <div>
                    <div className="sr-label">Theme</div>
                    <div className="sr-sub">Toggle dark / light mode</div>
                  </div>
                  <button className="theme-btn theme-btn--lg" onClick={() => setDark(d => !d)}>
                    {dark ? <><MdLightMode /> Light</> : <><MdDarkMode /> Dark</>}
                  </button>
                </div>
                <div className="setting-row">
                  <div>
                    <div className="sr-label">Live Monitor Interval</div>
                    <div className="sr-sub">New simulated log line every 1.4 seconds</div>
                  </div>
                  <span className="badge badge--ok">1.4s</span>
                </div>
                <div className="setting-row">
                  <div>
                    <div className="sr-label">Model Pipeline</div>
                    <div className="sr-sub">Classification + Root Cause Analysis</div>
                  </div>
                  <span className="badge badge--info">LogLLM + Llama 3B</span>
                </div>
              </div>
            </div>
          )}

        </main>
      </div>
    </div>
  )
}