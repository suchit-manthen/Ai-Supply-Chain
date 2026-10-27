import { useEffect, useMemo, useState } from 'react'
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

const money = value => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value || 0)
const number = value => new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 }).format(value || 0)
const Panel = ({ title, note, children }) => <section className="sim-panel"><div className="sim-panel-head"><div><h3>{title}</h3>{note && <p>{note}</p>}</div></div>{children}</section>

async function readApiJson(response, action) {
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.toLowerCase().includes('application/json')) {
    if (response.status === 404) {
      throw new Error(`The Flask server returned 404 for ${action}. It is likely running an older app. Stop that server, open a terminal in this project's backend folder, and run .\\venv\\Scripts\\python.exe app.py (or activate that venv and run python app.py). Reload after Flask starts.`)
    }
    throw new Error(`The backend returned an unexpected ${response.status} response for ${action}. Check that Flask is running on port 5000.`)
  }
  return response.json()
}

function Metric({ title, baseline, scenario, format = number, goodWhen = 'none' }) {
  const delta = scenario - baseline
  const cls = goodWhen === 'higher' ? (delta > 0 ? 'sim-better' : delta < 0 ? 'sim-worse' : '') : goodWhen === 'lower' ? (delta < 0 ? 'sim-better' : delta > 0 ? 'sim-worse' : '') : ''
  return <div className="sim-metric"><span>{title}</span><div><small>Baseline</small><b>{format(baseline)}</b></div><div><small>Scenario</small><b className={cls}>{format(scenario)}</b></div></div>
}

const ChartTooltip = ({ active, payload, label, moneyValues = false }) => active && payload?.length
  ? <div className="sim-tooltip"><b>Day {label}</b>{payload.map(row => <div key={row.dataKey} style={{ color: row.color }}>{row.name}: {moneyValues ? money(row.value) : number(row.value)}</div>)}</div>
  : null

export default function Simulator() {
  const [catalog, setCatalog] = useState(null)
  const [sku, setSku] = useState('')
  const [controls, setControls] = useState({ horizon: 30, demand_multiplier: 1, spike_pct: 0, spike_day: 15, supplier_delay: 0, initial_stock_multiplier: 1, unit_price: 1, unit_cost: .7, annual_holding_rate: .2, order_cost: 50 })
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    fetch('/api/simulator/catalog').then(async response => {
      const payload = await readApiJson(response, 'the simulator catalog API')
      if (!response.ok || !payload.ok) throw new Error(payload.error || 'Could not load the saved forecast catalog.')
      if (!active) return
      setCatalog(payload)
      setSku(payload.items[0]?.id || '')
      setControls(current => ({ ...current, horizon: payload.horizons[0] || 30, spike_day: Math.min(15, payload.horizons[0] || 30), annual_holding_rate: payload.default_costs.annual_holding_rate, order_cost: payload.default_costs.order_cost, unit_price: payload.items[0]?.price ?? 1, unit_cost: payload.items[0]?.cost ?? .7 }))
    }).catch(e => { if (active) setError(e.message) })
    return () => { active = false }
  }, [])

  const selected = useMemo(() => catalog?.items.find(item => item.id === sku), [catalog, sku])
  function update(key, value) {
    setControls(current => ({ ...current, [key]: value }))
    setResult(null)
  }
  function chooseSku(value) {
    const item = catalog?.items.find(entry => entry.id === value)
    setSku(value)
    setControls(current => ({ ...current, unit_price: item?.price ?? 1, unit_cost: item?.cost ?? .7 }))
    setResult(null)
  }
  function chooseHorizon(value) {
    setControls(current => ({ ...current, horizon: value, spike_day: Math.min(current.spike_day, value) }))
    setResult(null)
  }
  async function runSimulation(event) {
    event.preventDefault()
    setBusy(true); setError('')
    try {
      const response = await fetch('/api/simulator/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sku, ...controls }) })
      const payload = await readApiJson(response, 'the simulator run API')
      if (!response.ok || !payload.ok) throw new Error(payload.error || 'Simulation could not be completed.')
      setResult(payload)
    } catch (e) { setError(e.message) }
    setBusy(false)
  }

  return <div className="simulator">
    <div className="sim-hero"><div><div className="sim-eyebrow">SUPPLY CHAIN LAB · WHAT-IF ANALYSIS</div><h1>Supply chain simulator</h1><p>Stress-test a saved SKU forecast against demand, stock and supplier changes. All results are simulated estimates.</p></div><div className="sim-live"><i /> Saved forecast · no model retraining</div></div>

    <form className="sim-controls" onSubmit={runSimulation}>
      <div className="sim-section-title"><div><span>01 / SCENARIO INPUTS</span><h2>Configure a scenario</h2></div><button className="sim-run" type="submit" disabled={busy || !selected}>{busy ? 'Running simulation…' : 'Run comparison'} <span>↗</span></button></div>
      {error && <div className="sim-error" role="alert">{error}</div>}
      {!catalog && !error && <div className="sim-loading">Loading saved SKU forecasts…</div>}
      {catalog && !catalog.items.length && <div className="sim-error">The saved run has no SKU forecasts available for simulation.</div>}
      {catalog && catalog.items.length > 0 && <>
        <div className="sim-control-grid">
          <label className="sim-field sim-wide">SKU
            <select value={sku} onChange={event => chooseSku(event.target.value)}>{catalog.items.map(item => <option key={item.id} value={item.id}>{item.name} · {item.id}</option>)}</select>
            {selected && <small>{selected.category} · baseline stock {number(selected.stock)} · {selected.lead}-day supplier lead</small>}
          </label>
          <label className="sim-field">Forecast horizon
            <select value={controls.horizon} onChange={event => chooseHorizon(Number(event.target.value))}>{catalog.horizons.map(days => <option key={days} value={days}>{days} days</option>)}</select>
          </label>
          <label className="sim-field">Demand multiplier <b>{Number(controls.demand_multiplier).toFixed(1)}×</b>
            <input type="range" min="0.5" max="3" step="0.1" value={controls.demand_multiplier} onChange={event => update('demand_multiplier', Number(event.target.value))} />
            <small>Scales expected daily demand</small>
          </label>
          <label className="sim-field">One-time demand spike <b>{controls.spike_pct}%</b>
            <input type="range" min="0" max="200" step="5" value={controls.spike_pct} onChange={event => update('spike_pct', Number(event.target.value))} />
            <small>Applied once on the selected day</small>
          </label>
          <label className="sim-field">Spike simulation day
            <input type="number" min="1" max={controls.horizon} value={controls.spike_day} onChange={event => update('spike_day', Number(event.target.value))} />
          </label>
          <label className="sim-field">Extra supplier delay <b>{controls.supplier_delay} days</b>
            <input type="range" min="0" max="30" step="1" value={controls.supplier_delay} onChange={event => update('supplier_delay', Number(event.target.value))} />
            <small>Added to this SKU’s baseline lead time</small>
          </label>
          <label className="sim-field">Initial stock <b>{Number(controls.initial_stock_multiplier).toFixed(2)}×</b>
            <input type="range" min="0.25" max="2" step="0.05" value={controls.initial_stock_multiplier} onChange={event => update('initial_stock_multiplier', Number(event.target.value))} />
            <small>Relative to the saved SKU stock level</small>
          </label>
        </div>
        <details className="sim-costs" open><summary>Cost assumptions <small>editable estimates used in both paths</small></summary>
          <div className="sim-cost-grid">
            <label>Unit selling price ($)<input type="number" min="0.01" step="0.01" value={controls.unit_price} onChange={event => update('unit_price', Number(event.target.value))} /></label>
            <label>Unit purchase cost ($)<input type="number" min="0" step="0.01" value={controls.unit_cost} onChange={event => update('unit_cost', Number(event.target.value))} /></label>
            <label>Annual holding rate (%)<input type="number" min="0" max="100" step="1" value={Number(controls.annual_holding_rate) * 100} onChange={event => update('annual_holding_rate', Number(event.target.value) / 100)} /></label>
            <label>Cost per purchase order ($)<input type="number" min="0" step="1" value={controls.order_cost} onChange={event => update('order_cost', Number(event.target.value))} /></label>
          </div>
          <p className="sim-assumption">Monetary results are estimates, not actual business results. {selected?.price_assumed ? 'This saved dataset does not provide a reliable selling price; the prefilled value is a neutral placeholder. ' : 'Selling price is prefilled from the saved SKU where available. '}{selected?.cost_assumed ? 'Purchase cost is prefilled from a configurable fallback. ' : 'Purchase cost is prefilled from the saved SKU where available. '}Review or edit the inputs above.</p>
        </details>
      </>}
    </form>

    {result && <>
      <div className="sim-section-title sim-results-title"><div><span>02 / SIMULATION RESULTS</span><h2>{result.sku.name} <small>· {result.horizon} days</small></h2></div><span className="sim-badge">SIMULATED · {result.source || 'saved forecast'}</span></div>
      <div className="sim-metrics">
        <Metric title="Demand · units" baseline={result.baseline.summary.total_demand} scenario={result.scenario.summary.total_demand} />
        <Metric title="Fulfilled · units" baseline={result.baseline.summary.fulfilled_units} scenario={result.scenario.summary.fulfilled_units} goodWhen="higher" />
        <Metric title="Lost units" baseline={result.baseline.summary.lost_units} scenario={result.scenario.summary.lost_units} goodWhen="lower" />
        <Metric title="Lost revenue" baseline={result.baseline.summary.lost_revenue} scenario={result.scenario.summary.lost_revenue} format={money} goodWhen="lower" />
        <Metric title="Stockout days" baseline={result.baseline.summary.stockout_days} scenario={result.scenario.summary.stockout_days} goodWhen="lower" />
        <Metric title="Service level" baseline={result.baseline.summary.service_level} scenario={result.scenario.summary.service_level} format={value => `${number(value)}%`} goodWhen="higher" />
        <Metric title="Ending inventory" baseline={result.baseline.summary.ending_inventory} scenario={result.scenario.summary.ending_inventory} />
        <Metric title="Average inventory" baseline={result.baseline.summary.average_inventory} scenario={result.scenario.summary.average_inventory} />
        <Metric title="Holding cost" baseline={result.baseline.summary.holding_cost} scenario={result.scenario.summary.holding_cost} format={money} goodWhen="lower" />
        <Metric title="Overstock exposure · unit-days" baseline={result.baseline.summary.overstock_unit_days} scenario={result.scenario.summary.overstock_unit_days} goodWhen="lower" />
        <Metric title="Total estimated cost" baseline={result.baseline.summary.total_estimated_cost} scenario={result.scenario.summary.total_estimated_cost} format={money} goodWhen="lower" />
      </div>
      <div className="sim-chart-grid">
        <Panel title="Inventory position" note="End-of-day on-hand stock with reorder-point thresholds · units">
          <ResponsiveContainer width="100%" height={290}><LineChart data={result.daily} margin={{ top: 10, right: 12, left: 0, bottom: 4 }}><CartesianGrid stroke="#29343d" vertical={false} /><XAxis dataKey="day" stroke="#84919a" tickLine={false} label={{ value: 'Simulation day', position: 'insideBottom', offset: -2, fill: '#84919a' }} /><YAxis stroke="#84919a" tickLine={false} width={52} /><Tooltip content={<ChartTooltip />} /><Legend /><ReferenceLine y={0} stroke="#d9685a" strokeDasharray="3 4" label={{ value: 'Stockout', position: 'insideBottomRight', fill: '#d98275', fontSize: 10 }} /><Line type="monotone" dataKey="baseline_inventory" name="Baseline stock" stroke="#68b88a" strokeWidth={2} dot={false} /><Line type="monotone" dataKey="scenario_inventory" name="Scenario stock" stroke="#f0a05a" strokeWidth={2} dot={false} /><Line type="monotone" dataKey="baseline_reorder_point" name="Baseline reorder point" stroke="#68b88a" strokeDasharray="5 5" strokeOpacity={.6} dot={false} /><Line type="monotone" dataKey="scenario_reorder_point" name="Scenario reorder point" stroke="#f0a05a" strokeDasharray="5 5" strokeOpacity={.6} dot={false} /></LineChart></ResponsiveContainer>
        </Panel>
        <Panel title="Demand vs fulfilled sales" note="Baseline and scenario · units per day">
          <ResponsiveContainer width="100%" height={290}><ComposedChart data={result.daily} margin={{ top: 10, right: 12, left: 0, bottom: 4 }}><CartesianGrid stroke="#29343d" vertical={false} /><XAxis dataKey="day" stroke="#84919a" tickLine={false} label={{ value: 'Simulation day', position: 'insideBottom', offset: -2, fill: '#84919a' }} /><YAxis stroke="#84919a" tickLine={false} width={52} /><Tooltip content={<ChartTooltip />} /><Legend /><Line type="monotone" dataKey="baseline_demand" name="Baseline demand" stroke="#68b88a" strokeDasharray="5 4" dot={false} /><Line type="monotone" dataKey="baseline_fulfilled" name="Baseline fulfilled" stroke="#68b88a" strokeWidth={2} dot={false} /><Line type="monotone" dataKey="scenario_demand" name="Scenario demand" stroke="#f0a05a" strokeDasharray="5 4" dot={false} /><Line type="monotone" dataKey="scenario_fulfilled" name="Scenario fulfilled" stroke="#f0a05a" strokeWidth={2} dot={false} /></ComposedChart></ResponsiveContainer>
        </Panel>
        <Panel title="Cumulative cost exposure" note="Lost revenue and estimated total cost · USD">
          <ResponsiveContainer width="100%" height={290}><LineChart data={result.daily} margin={{ top: 10, right: 12, left: 4, bottom: 4 }}><CartesianGrid stroke="#29343d" vertical={false} /><XAxis dataKey="day" stroke="#84919a" tickLine={false} label={{ value: 'Simulation day', position: 'insideBottom', offset: -2, fill: '#84919a' }} /><YAxis stroke="#84919a" tickLine={false} tickFormatter={value => `$${number(value)}`} width={64} /><Tooltip content={<ChartTooltip moneyValues />} /><Legend /><Area type="monotone" dataKey="scenario_lost_revenue_cumulative" name="Scenario lost revenue" stroke="#e0705f" fill="#e0705f" fillOpacity={.10} /><Line type="monotone" dataKey="baseline_lost_revenue_cumulative" name="Baseline lost revenue" stroke="#68b88a" dot={false} /><Line type="monotone" dataKey="baseline_total_cost_cumulative" name="Baseline total cost" stroke="#68b88a" strokeDasharray="5 5" dot={false} /><Line type="monotone" dataKey="scenario_total_cost_cumulative" name="Scenario total cost" stroke="#f0a05a" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer>
        </Panel>
      </div>
      <div className="sim-footnote"><b>How to read this:</b> Baseline uses the saved forecast, original stock and lead time, with no demand shock. Scenario applies your selected controls. Costs include lost sales at the entered selling price, estimated daily holding cost, placed replenishment purchases and order fees. {result.assumptions.forecast_note} No simulated inventory or orders are written to operational records.</div>
    </>}
  </div>
}
