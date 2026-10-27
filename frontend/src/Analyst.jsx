import { useEffect, useState } from 'react'
import { ResponsiveContainer, ComposedChart, LineChart, BarChart, Bar, Line, Area, XAxis, YAxis, Tooltip, CartesianGrid, Legend, Cell, ReferenceLine } from 'recharts'
import { fmt, short } from './ui.jsx'

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
const Card = ({ t, s, children }) => <section className="card"><h3>{t}</h3>{s && <p className="sub">{s}</p>}{children}</section>
const Chart = ({ children, h = 250 }) => <ResponsiveContainer width="100%" height={h}>{children}</ResponsiveContainer>
const sign = n => (n > 0 ? '+' : '') + n

function Overview({ d }) {
  const lb = d.leaderboard
  const baselineMape = lb.length ? lb[0].avg_naive : 0
  const selectedAvg = d.skus.length ? d.skus.reduce((a, s) => a + s.comparison.find(c => c.selected).mape, 0) / d.skus.length : 0
  const baselineAvg = d.skus.length ? d.skus.reduce((a, s) => a + s.comparison.find(c => c.baseline).mape, 0) / d.skus.length : 0
  return (
    <>
      <div className="grid2">
        <Card t="Does model selection beat the simple forecast?" s="Average per-SKU walk-forward MAPE; negative difference means the selected models reduce error. Also review MAE/RMSE for unit-level impact.">
          <div className="stat"><div><b>{selectedAvg.toFixed(2)}%</b><span>selected models</span></div><div><b>{baselineAvg.toFixed(2)}%</b><span>same-day-last-week baseline</span></div><div><b>{(selectedAvg - baselineAvg > 0 ? '+' : '') + (selectedAvg - baselineAvg).toFixed(2)}</b><span>MAPE point difference</span></div></div>
        </Card>
        <Card t="Mean validation error by algorithm (MAPE and RMSE, lower is better)" s={`Walk-forward across ${d.selection.n_folds} rolling-origin folds, averaged over all ${d.dataset.skus} SKUs. Grey bar = naive "same day last week" baseline. MAPE is percentage error; RMSE is typical error in units/day and is more stable for low-volume products.`}>
          <Chart><BarChart data={lb.map(r => ({ name: r.name, model: r.avg_mape, baseline: r.avg_naive }))} margin={{ left: -10 }}>
            <CartesianGrid vertical={false} stroke="#e6ebe3" /><XAxis dataKey="name" fontSize={12} /><YAxis unit="%" fontSize={12} /><Tooltip /><Legend />
            <Bar dataKey="baseline" fill="#c9d1c6" name="Naive baseline" /><Bar dataKey="model" name="ML model">{lb.map(r => <Cell key={r.name} fill={r.color} />)}</Bar>
          </BarChart></Chart>
        </Card>
        <Card t="Dataset" s={`${fmt(d.dataset.rows)} daily records · ${d.dataset.skus} products${d.dataset.categories?.length ? ` · ${d.dataset.categories.length} categories` : ''} · ${d.dataset.start} to ${d.dataset.end}`}>
          <Chart h={200}><BarChart data={d.dataset.monthly} margin={{ left: -10 }}><CartesianGrid vertical={false} stroke="#e6ebe3" /><XAxis dataKey="month" fontSize={11} minTickGap={25} /><YAxis fontSize={11} /><Tooltip /><Bar dataKey="units" fill="#2c7a4b" name="Units sold" /></BarChart></Chart>
          <p className="sub" style={{ marginTop: 8 }}>Inputs: {d.dataset.features.join(', ')}. Only optional drivers that are present and vary are used; missing or constant columns (e.g. an unavailable price or temperature) are dropped rather than fed as zeros.</p>
          {d.dataset.categories?.length > 0 && <p className="sub"><b>By category:</b> {d.dataset.categories.map(c => `${c.category} (${fmt(c.units)})`).join(' · ')}</p>}
          <p className="sub">{d.data_source === 'synthetic' ? 'Synthetic demo data models weekly cycles, annual seasons, holidays, promotions with price elasticity, weather and 5% yearly growth; it is not real store data.' : d.data_source === 'favorita' ? 'Favorita subset. onpromotion is used as a binary promotion signal. Unit price, weather and stock are not in train.csv; neutral placeholders are used for unavailable model inputs.' : 'Uploaded historical data. The next 28 days assume 0% discount, last observed price and temperature, and calendar-derived holiday lift; no future promotion/weather schedule is provided.'}</p>
        </Card>
      </div>

      <Card t="Selected model per SKU" s={`Each SKU is evaluated against all four algorithms; the one with the lowest mean validation ${d.selection.metric.toUpperCase()} wins and is refit on all history. A negative improvement means the selected model reduced error versus the simple baseline; positive means the baseline did better. Legacy = old assignment (reference only).`}>
        <div className="scroll"><table className="tbl">
          <thead><tr><th>SKU</th><th>Category</th><th>Selected model</th><th>Legacy</th><th className="num">Val {d.selection.metric.toUpperCase()}</th><th className="num">MAE</th><th className="num">RMSE</th><th className="num">Naive</th><th className="num">Improvement</th></tr></thead>
          <tbody>{d.skus.map(s => (
            <tr key={s.id}>
              <td><b>{s.name}</b></td><td>{s.category}</td>
              <td><span className="chip" style={{ background: s.selected_color }}>{s.selected_model}</span></td>
              <td><small style={{ color: '#5f6f66' }}>{s.legacy_model}{s.legacy_model !== s.selected_model ? '' : ' ✓'}</small></td>
              <td className="num">{s.selection_score}</td>
              <td className="num">{s.comparison.find(c => c.selected)?.mae}</td><td className="num">{s.comparison.find(c => c.selected)?.rmse}</td>
              <td className="num">{s.baseline_score}</td>
              <td className="num" style={{ color: s.improvement > 0 ? '#2c7a4b' : '#d43f2b' }}>{sign(s.improvement)}</td>
            </tr>))}
          </tbody></table></div>
      </Card>

      <Card t="Algorithm leaderboard" s="Average across all SKUs. Wins = number of SKUs where the algorithm was selected.">
        <div className="scroll"><table className="tbl">
          <thead><tr><th>Algorithm</th><th>Strengths</th><th className="num">Avg validation MAPE</th><th className="num">Naive baseline</th><th className="num">Avg RMSE</th><th className="num">Wins</th></tr></thead>
          <tbody>{lb.map(r => (
            <tr key={r.name}>
              <td><span className="chip" style={{ background: r.color }}>{r.name}</span></td>
              <td>{r.aspect}</td>
              <td className="num">{r.avg_mape}%</td><td className="num">{r.avg_naive}%</td>
              <td className="num">{r.avg_rmse}</td><td className="num"><b>{r.wins}</b></td>
            </tr>))}
          </tbody></table></div>
        <p className="sub" style={{ marginTop: 10 }}>Naive baseline across all SKUs: {baselineMape}% MAPE. Model selection uses the validation metric only and is not based on assumptions about which algorithm suits which demand pattern.</p>
      </Card>

      <Card t="Sample of the training data"><div className="scroll"><table className="tbl"><thead><tr>{Object.keys(d.dataset.sample[0]).map(k => <th key={k}>{k}</th>)}</tr></thead>
        <tbody>{d.dataset.sample.map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{v}</td>)}</tr>)}</tbody></table></div></Card>
    </>
  )
}

function Importance({ data, color }) {
  return <Chart h={280}><BarChart data={data} layout="vertical" margin={{ left: 40 }}><CartesianGrid horizontal={false} stroke="#e6ebe3" /><XAxis type="number" unit="%" fontSize={12} /><YAxis type="category" dataKey="feature" fontSize={12} width={90} /><Tooltip /><Bar dataKey="value" fill={color} name="Importance" /></BarChart></Chart>
}

function ComparisonTable({ s }) {
  const base = s.comparison.find(c => c.baseline)
  return (
    <div className="scroll"><table className="tbl">
      <thead><tr><th>Model</th><th className="num">MAPE</th><th className="num">RMSE</th><th className="num">MAE</th><th className="num">R²</th><th className="num">Δ vs naive MAPE</th></tr></thead>
      <tbody>{s.comparison.map(c => {
        const delta = c.baseline ? null : Math.round((c.mape - base.mape) * 100) / 100
        return (
          <tr key={c.model} style={c.selected ? { background: '#eef3ea' } : null}>
            <td><span className="chip" style={{ background: c.color }}>{c.model}</span>{c.selected && <b> ← selected</b>}</td>
            <td className="num">{c.mape}%</td><td className="num">{c.rmse}</td><td className="num">{c.mae}</td><td className="num">{c.r2}</td>
            <td className="num" style={{ color: delta === null ? undefined : delta < 0 ? '#2c7a4b' : '#d43f2b' }}>{delta === null ? '—' : sign(delta)}</td>
          </tr>)
      })}
      </tbody></table></div>
  )
}

function ModelDiagnostics({ name, x, color, residuals }) {
  if (name === 'ARIMA') return <>
    <Card t="Model specification" s="SARIMAX with discount as an external input.">
      <div className="kv"><span>Order (p,d,q)</span><b>{x.order}</b><span>Seasonal (P,D,Q)[s]</span><b>{x.seasonal_order}</b><span>AIC</span><b>{x.aic}</b><span>BIC</span><b>{x.bic}</b><span>Residual std</span><b>{x.resid_std}</b>
        {Object.entries(x.params).map(([k, v]) => [<span key={k}>{k}</span>, <b key={k + 'v'}>{v}</b>])}</div>
    </Card>
    <Card t="Autocorrelation (ACF)" s="Differenced sales vs residuals. Residual bars near zero mean the model captured the pattern; the spike at lag 7 in raw data is the weekly cycle.">
      <Chart><BarChart data={x.acf.map((v, i) => ({ lag: i, sales: v, residuals: x.resid_acf[i] }))} margin={{ left: -10 }}><CartesianGrid vertical={false} stroke="#e6ebe3" /><XAxis dataKey="lag" fontSize={12} /><YAxis fontSize={12} /><Tooltip /><Legend /><Bar dataKey="sales" fill="#aab6f0" name="differenced sales" /><Bar dataKey="residuals" fill={color} /><ReferenceLine y={0} stroke="#999" /></BarChart></Chart>
    </Card></>
  if (name === 'Prophet') return <>
    <Card t="Trend component" s={`${x.changepoints} potential changepoints, ${x.mode} seasonality.`}>
      <Chart><LineChart data={x.trend} margin={{ left: -10 }}><CartesianGrid vertical={false} stroke="#e6ebe3" /><XAxis dataKey="date" tickFormatter={short} fontSize={12} minTickGap={30} /><YAxis fontSize={12} domain={['auto', 'auto']} /><Tooltip /><Line dataKey="v" stroke={color} strokeWidth={2.5} dot={false} name="trend (units/day)" /></LineChart></Chart>
    </Card>
    <Card t="Seasonal effects (% of trend)" s={`Weekly (left) and yearly (right). Holidays add up to ±${x.holiday_effect}%.`}>
      <div className="grid2" style={{ gap: 8 }}>
        <Chart h={220}><BarChart data={x.weekly.map((v, i) => ({ n: DAYS[i], v }))}><XAxis dataKey="n" fontSize={11} /><YAxis fontSize={11} /><Tooltip /><Bar dataKey="v" fill={color} name="%" /></BarChart></Chart>
        <Chart h={220}><BarChart data={x.yearly.map((v, i) => ({ n: MONTHS[i], v }))}><XAxis dataKey="n" fontSize={10} interval={1} /><YAxis fontSize={11} /><Tooltip /><Bar dataKey="v" fill="#f4a261" name="%" /></BarChart></Chart>
      </div>
    </Card></>
  return <>
    <Card t="Feature importance" s="Which inputs drive the prediction."><Importance data={x.importance} color={color} /></Card>
    <Card t="Hyperparameters & residuals" s={name === 'Random Forest' ? `Average disagreement between trees: ±${x.tree_spread} units/day.` : 'Boosted trees fitted sequentially, each correcting the previous errors.'}>
      <div className="kv">{Object.entries(x.params).map(([k, v]) => [<span key={k}>{k}</span>, <b key={k + 'v'}>{v}</b>])}</div>
      <Chart h={170}><BarChart data={residuals.map((v, i) => ({ i, v }))} margin={{ left: -10 }}><XAxis dataKey="i" hide /><YAxis fontSize={11} /><Tooltip /><ReferenceLine y={0} stroke="#999" /><Bar dataKey="v" fill={color} name="actual − predicted" /></BarChart></Chart>
    </Card></>
}

function SkuView({ s, d }) {
  const r = s.result, met = r.metrics, x = r.extras, color = s.selected_color, u = r.uncertainty
  const line = [...r.history.slice(-45).map(h => ({ date: h.date, actual: h.actual })),
    ...r.forecast.map(f => ({ date: f.date, pred: f.pred, band: [f.lo, f.hi] }))]
  const bt = r.backtest
  const beatsBaseline = s.improvement > 0
  const coverage = r.interval_coverage
  return (
    <>
      <Card t={`${s.name} — selected model: ${s.selected_model}`}
        s={`Chosen by lowest mean validation ${s.metric_used.toUpperCase()} over ${d.selection.n_folds} rolling-origin folds. Legacy assignment was ${s.legacy_model} (reference only).`}>
        <div className="why"><b>{s.selected_model} selected:</b> {s.reason}{beatsBaseline
          ? ` It beats the naive baseline by ${s.improvement} ${s.metric_used.toUpperCase()} points.`
          : ` It does not beat the naive baseline on ${s.metric_used.toUpperCase()} — treat this forecast with caution.`}</div>
        <div className="stat">
          <div><b>{met.mape}%</b><span>28-day holdout MAPE (baseline {met.naive_mape}%)</span></div><div><b>{met.mae}</b><span>MAE (units/day)</span></div>
          <div><b>{met.rmse}</b><span>RMSE</span></div><div><b>{met.r2}</b><span>R²</span></div><div><b>{met.accuracy}%</b><span>accuracy</span></div>
        </div>
        <h3 style={{ marginTop: 14, marginBottom: 4 }}>All four models vs the naive baseline</h3>
        <ComparisonTable s={s} />
      </Card>

      <div className="grid2">
        <Card t="Walk-forward validation error by model" s={`Mean ${s.metric_used.toUpperCase()} across ${s.folds[s.selected_model].length} rolling-origin folds per model. Lower is better.`}>
          <Chart><BarChart data={s.comparison.map(c => ({ name: c.model, value: c.mape, selected: c.selected }))} margin={{ left: -10 }}>
            <CartesianGrid vertical={false} stroke="#e6ebe3" /><XAxis dataKey="name" fontSize={11} interval={0} angle={-15} textAnchor="end" height={50} /><YAxis unit="%" fontSize={12} /><Tooltip /><Bar dataKey="value" name="MAPE %">{s.comparison.map(c => <Cell key={c.model} fill={c.color} />)}</Bar>
          </BarChart></Chart>
        </Card>
        <Card t="Validation folds" s="Each fold trains on the past and tests on the next 28 days. Model and naive baseline use identical windows.">
          <div className="scroll"><table className="tbl">
            <thead><tr><th>Model</th>{s.folds[s.selected_model].map(f => <th key={f.fold} className="num">Fold {f.fold}</th>)}</tr></thead>
            <tbody>{s.comparison.filter(c => !c.baseline).map(c => (
              <tr key={c.model} style={c.selected ? { background: '#eef3ea' } : null}><td><span className="chip" style={{ background: c.color }}>{c.model}</span></td>
                {s.folds[c.model].map(f => <td key={f.fold} className="num">{f.mape}%</td>)}
              </tr>))}
              <tr><td><small>test windows</small></td>{s.folds[s.selected_model].map(f => <td key={f.fold} className="num"><small>{short(f.test_start)}→{short(f.test_end)}</small></td>)}</tr>
            </tbody></table></div>
        </Card>
      </div>

      <div className="grid2">
          <Card t="Backtest: predicted vs actual" s={`Selected model (${s.selected_model}) trained without the last ${bt.length} days (${bt[0]?.date} to ${bt[bt.length - 1]?.date}), then asked to predict them. The table above also reports MAE/RMSE in units/day.`}>
          <Chart><LineChart data={bt} margin={{ left: -10 }}><CartesianGrid vertical={false} stroke="#e6ebe3" /><XAxis dataKey="date" tickFormatter={short} fontSize={12} minTickGap={25} /><YAxis fontSize={12} domain={['auto', 'auto']} /><Tooltip labelFormatter={short} /><Legend />
            <Line dataKey="actual" stroke="#16261f" strokeWidth={2} dot={false} /><Line dataKey="pred" name="predicted" stroke={color} strokeWidth={2.5} dot={false} strokeDasharray="5 3" /></LineChart></Chart>
        </Card>
        <Card t="Forecast reliability and uncertainty" s={`Refit on all history. Shaded band is an ${Math.round(u.level * 100)}% prediction interval estimated from backtest errors; the historical check below measures actual coverage on the latest holdout window.`}>
          <Chart><ComposedChart data={line} margin={{ left: -10 }}><CartesianGrid vertical={false} stroke="#e6ebe3" /><XAxis dataKey="date" tickFormatter={short} fontSize={12} minTickGap={25} /><YAxis fontSize={12} domain={['auto', 'auto']} /><Tooltip labelFormatter={short} />
            <Area dataKey="band" stroke="none" fill={color} fillOpacity={.18} name={`${Math.round(u.level * 100)}% range`} /><Line dataKey="actual" stroke="#16261f" strokeWidth={2} dot={false} /><Line dataKey="pred" name="forecast" stroke={color} strokeWidth={2.5} dot={false} /></ComposedChart></Chart>
          <div className="stat">
            <div><b>{coverage ? `${coverage.pct}%` : '—'}</b><span>actual coverage of {Math.round(u.level * 100)}% range ({coverage?.covered ?? 0}/{coverage?.total ?? 0} days)</span></div>
            <div><b>±{Math.round(u.z * u.error_scale[0])}</b><span>day 1 range</span></div>
            <div><b>±{Math.round(u.z * u.error_scale[6])}</b><span>day 7 range</span></div>
            <div><b>±{Math.round(u.z * u.error_scale[13])}</b><span>day 14 range</span></div>
            <div><b>±{Math.round(u.z * u.error_scale[27])}</b><span>day 28 range</span></div>
          </div>
          <p className="sub" style={{ marginTop: 4 }}>Validation period: {bt[0]?.date} to {bt[bt.length - 1]?.date} ({bt.length} days). Coverage is the share of those days where actual sales fell inside a range reconstructed from training-only error estimates. It is a diagnostic, not a guarantee of future coverage.</p>
        </Card>
      </div>

      <div className="grid2"><ModelDiagnostics name={s.selected_model} x={x} color={color} residuals={r.residuals} /></div>
    </>
  )
}

function RunsView({ d, run }) {
  const [runs, setRuns] = useState(null)
  const [mon, setMon] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    let alive = true
    Promise.all([fetch('/api/runs').then(r => r.json()), fetch('/api/monitoring').then(r => r.json())])
      .then(([r, m]) => { if (alive) { setRuns(r); setMon(m) } })
      .catch(() => { if (alive) setErr('Could not load run history.') })
    return () => { alive = false }
  }, [])

  if (err) return <Card t="Model runs"><p className="sub">{err}</p></Card>
  if (!runs || !mon) return <Card t="Model runs"><p className="sub">Loading run history…</p></Card>
  if (!runs.runs || runs.runs.length === 0) return <Card t="Model runs"><div className="empty">No training runs found yet. Use “Retrain” to create one.</div></Card>

  const list = runs.runs || []
  const latest = list[0]
  const chrono = [...list].reverse()
  const flags = mon.deteriorated || []
  return (
    <>
      <Card t="Current run" s="Training is separated from serving: the pipeline produces and saves a run to SQLite; the API just loads it.">
        <div className="kv">
          <span>Run ID</span><b>{run?.run_id || latest?.run_id || '—'}</b>
          <span>Trained at</span><b>{run?.trained_at ? new Date(run.trained_at).toLocaleString() : '—'}</b>
          <span>Duration</span><b>{latest?.duration_s != null ? latest.duration_s + 's' : '—'}</b>
          <span>Dataset</span><b>{fmt(d.dataset.rows)} daily records · {d.dataset.skus} SKUs · {d.dataset.start} → {d.dataset.end}</b>
          <span>Forecast horizon</span><b>{d.horizon} days</b>
          <span>Validation</span><b>{d.selection.n_folds} rolling-origin folds</b>
        </div>
      </Card>

      <div className="grid2">
        <Card t="Validation vs naive baseline" s="Averaged over the selected model per SKU. Lower MAPE is better.">
          {mon.overall ? (
            <div className="stat">
              <div><b>{mon.overall.avg_mape}%</b><span>mean validation MAPE</span></div>
              <div><b>{mon.overall.avg_naive}%</b><span>naive baseline MAPE</span></div>
              <div><b>{sign(mon.overall.improvement)}</b><span>points better than naive</span></div>
            </div>
          ) : <p className="sub">No monitoring data.</p>}
        </Card>
        <Card t="Drift check" s={`Compares the latest run with the previous one. Flags a SKU if validation MAPE rises by ≥${Math.round(mon.thresholds.rel * 100)}% or ≥${mon.thresholds.abs_points} points.`}>
          {mon.has_history ? (
            flags.length
              ? <p className="sub" style={{ color: '#a92917' }}><b>{flags.length} SKU(s) deteriorated:</b> {flags.join(', ')}. Review them in the per-SKU tabs.</p>
              : <p className="sub" style={{ color: '#1d6238' }}>No significant deterioration versus the previous run.</p>
          ) : <p className="sub">Only one run so far — nothing to compare yet. Retrain later to build history.</p>}
        </Card>
      </div>

      <Card t="Per-SKU monitoring" s="Latest validation error, previous run, and the change.">
        <div className="scroll"><table className="tbl">
          <thead><tr><th>SKU</th><th>Model</th><th className="num">MAPE</th><th className="num">RMSE</th><th className="num">MAE</th><th className="num">Naive</th><th className="num">Previous</th><th className="num">Δ MAPE</th><th>Status</th></tr></thead>
          <tbody>{mon.skus.map(s => (
            <tr key={s.sku_id}>
              <td><b>{s.sku_id}</b></td><td>{s.model}</td>
              <td className="num">{s.mape}%</td><td className="num">{s.rmse}</td><td className="num">{s.mae}</td>
              <td className="num">{s.naive_mape}%</td><td className="num">{s.prev_mape != null ? s.prev_mape + '%' : '—'}</td>
              <td className="num" style={{ color: s.delta_mape == null ? undefined : s.delta_mape > 0 ? '#a92917' : '#1d6238' }}>{s.delta_mape == null ? '—' : sign(s.delta_mape)}</td>
              <td>{s.deteriorated ? <span className="pill Critical">Deteriorated</span> : s.model_changed ? <span className="pill Reorder">Model changed</span> : <span className="pill Healthy">Stable</span>}</td>
            </tr>))}
          </tbody></table></div>
      </Card>

      <Card t="Training history" s={`${list.length} saved run(s). Each retrain appends a new run.`}>
        {chrono.length >= 2 && (
          <Chart h={220}><LineChart data={chrono.map(r => ({ t: new Date(r.created_at).toLocaleDateString(), mape: r.avg_mape, naive: r.avg_naive }))} margin={{ left: -10 }}>
            <CartesianGrid vertical={false} stroke="#e6ebe3" /><XAxis dataKey="t" fontSize={11} /><YAxis unit="%" fontSize={11} /><Tooltip /><Legend />
            <Line dataKey="naive" name="Naive" stroke="#c9d1c6" strokeWidth={2} dot /><Line dataKey="mape" name="Selected models" stroke="#2c7a4b" strokeWidth={2.5} dot />
          </LineChart></Chart>
        )}
        <div className="scroll" style={{ marginTop: 8 }}><table className="tbl">
          <thead><tr><th>Run ID</th><th>When</th><th className="num">Duration</th><th className="num">Avg MAPE</th><th className="num">Naive</th><th className="num">SKUs</th><th className="num">Horizon</th></tr></thead>
          <tbody>{list.map(r => (
            <tr key={r.run_id}>
              <td><small>{r.run_id}</small></td><td>{new Date(r.created_at).toLocaleString()}</td>
              <td className="num">{r.duration_s != null ? r.duration_s + 's' : '—'}</td>
              <td className="num">{r.avg_mape}%</td><td className="num">{r.avg_naive}%</td>
              <td className="num">{r.n_skus}</td><td className="num">{r.horizon}</td>
            </tr>))}
          </tbody></table></div>
      </Card>
    </>
  )
}

export default function Analyst({ d, run }) {
  const [sel, setSel] = useState('overview')
  return (
    <>
      <div className="head"><div><h1>Model workbench</h1><p>How each algorithm is trained, validated, selected and monitored per SKU.</p></div></div>
      <div className="side">
        <div className="nav">
          <button className={'navb ' + (sel === 'overview' ? 'on' : '')} style={{ '--c': '#16261f' }} onClick={() => setSel('overview')}>Overview<small>Compare all models</small></button>
          <button className={'navb ' + (sel === 'runs' ? 'on' : '')} style={{ '--c': '#7048c8' }} onClick={() => setSel('runs')}>Model runs<small>Training history &amp; monitoring</small></button>
          {d.skus.map(s => <button key={s.id} className={'navb ' + (sel === s.id ? 'on' : '')} style={{ '--c': s.selected_color }} onClick={() => setSel(s.id)}>{s.name}<small>{s.selected_model} · {s.selection_score} {s.metric_used.toUpperCase()}</small></button>)}
        </div>
        <div>{sel === 'overview' ? <Overview d={d} /> : sel === 'runs' ? <RunsView d={d} run={run} /> : <SkuView key={sel} s={d.skus.find(x => x.id === sel)} d={d} />}</div>
      </div>
    </>
  )
}
