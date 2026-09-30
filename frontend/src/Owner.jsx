import { useMemo, useState } from 'react'
import { ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts'
import { Kpi, Pill, RiskPill, fmt, usd, short, STATUS_COLOR } from './ui.jsx'

function ForecastChart({ series, height = 260 }) {
  const firstPred = series.find(s => s.pred !== undefined)?.date
  const data = series.map(s => ({ ...s, band: s.lo !== undefined ? [s.lo, s.hi] : undefined }))
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ left: -10, right: 8 }}>
        <CartesianGrid stroke="#e6ebe3" vertical={false} />
        <XAxis dataKey="date" tickFormatter={short} minTickGap={30} fontSize={12} />
        <YAxis fontSize={12} />
        <Tooltip labelFormatter={short} />
        <Area dataKey="band" stroke="none" fill="#2c7a4b" fillOpacity={.15} name="80% range" />
        <Line dataKey="actual" stroke="#16261f" dot={false} strokeWidth={2} name="Actual sales" />
        <Line dataKey="pred" stroke="#2c7a4b" dot={false} strokeWidth={2.5} strokeDasharray="5 3" name="Forecast" />
        <ReferenceLine x={firstPred} stroke="#999" label={{ value: 'today', fontSize: 11, position: 'top' }} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export default function Owner({ d }) {
  const [sel, setSel] = useState(d.items[0].id)
  const [overrides, setOverrides] = useState({})
  const baseItem = d.items.find(i => i.id === sel)
  const item = useMemo(() => {
    const o = overrides[baseItem.id] || {}
    const stock = Math.max(0, Number(o.stock ?? baseItem.stock) || 0)
    const onOrder = Math.max(0, Number(o.on_order ?? baseItem.on_order) || 0)
    const lead = Math.max(1, Number(o.lead ?? baseItem.lead) || 1)
    const service = Number(o.service_level ?? baseItem.service_level ?? d.service_level) || d.service_level
    const unitCost = Math.max(0, Number(o.cost ?? baseItem.cost) || 0)
    const forecastRows = baseItem.series.filter(x => x.pred !== undefined)
    const z = service >= .99 ? 2.326 : service >= .95 ? 1.645 : 1.282
    const safety = Math.ceil(z * Math.sqrt(baseItem.safety ** 2 / 1.645 ** 2 * (lead / baseItem.lead)))
    const demand = forecastRows.slice(0, lead).reduce((a, x) => a + x.pred, 0)
    const upperDemand = forecastRows.slice(0, lead).reduce((a, x) => a + (x.hi ?? x.pred), 0)
    const qty = Math.max(0, Math.ceil((forecastRows.slice(0, lead + 7).reduce((a, x) => a + x.pred, 0) + safety - stock - onOrder) / 10) * 10)
    const coverDays = (() => { let remaining = stock; for (let i = 0; i < forecastRows.length; i++) { if (remaining < forecastRows[i].pred) return i; remaining -= forecastRows[i].pred } return baseItem.cover_days })()
    const status = stock < demand ? 'Critical' : stock + onOrder < demand + safety ? 'Reorder' : coverDays > 14 ? 'Overstock' : 'Healthy'
    const risk = stock < demand ? 'High' : stock < upperDemand ? 'Elevated' : 'Low'
    return { ...baseItem, stock, on_order: onOrder, lead, service_level: service, cost: unitCost, safety, demand_lead: Math.round(demand), demand_lead_upper: Math.round(upperDemand), rop: Math.round(demand + safety), order_qty: qty, order_cost: Math.round(qty * unitCost * 100) / 100,
      cover_days: coverDays, status, stockout_risk: risk,
      inventory_position: stock + onOrder, message: `Scenario: ${stock} on hand + ${onOrder} on order; ${lead}-day lead time; ${Math.round(service * 100)}% service level.` }
  }, [baseItem, overrides, d.service_level])
  const setOverride = (key, value) => setOverrides(v => ({ ...v, [item.id]: { ...v[item.id], [key]: value } }))
  const k = d.kpis
  const alerts = d.items.filter(i => i.status !== 'Healthy' || i.stockout_risk !== 'Low')
  return (
    <>
      <div className="head">
        <div><h1>Store inventory outlook</h1><p>Forecast for the next {d.horizon} days, based on sales up to {short(d.as_of)}.</p></div>
      </div>
      <div className="kpis">
        <Kpi v={k.critical} l="Products at risk of running out" c="#d43f2b" />
        <Kpi v={k.reorder} l="Products to reorder today" c="#d99a12" />
        <Kpi v={k.overstock} l="Overstocked products" c="#3b5bdb" />
        <Kpi v={k.healthy} l="Healthy products" c="#2c7a4b" />
        <Kpi v={usd(k.order_cost)} l={`Purchase needed today (${fmt(k.order_units)} units)`} c="#7048c8" />
        <Kpi v={usd(k.revenue_7)} l="Expected sales, next 7 days" c="#16261f" />
      </div>

      <div className="grid2" style={{ gridTemplateColumns: '1.1fr 1fr' }}>
        <section className="card">
          <h3>What to do today</h3>
          <p className="sub">Sorted by urgency. Click a product for its forecast.</p>
          {alerts.map(a => (
            <div key={a.id} className={'alert ' + a.status} onClick={() => setSel(a.id)} style={{ cursor: 'pointer' }}>
              <div style={{ flex: 1 }}>
                <b>{a.name} <Pill s={a.status} /></b>
                <small>{a.message}</small>
              </div>
              {a.order_qty > 0 && <div style={{ textAlign: 'right' }}><b>{fmt(a.order_qty)}</b><small>units to order</small></div>}
            </div>
          ))}
          {!alerts.length && <div className="empty">Everything is well stocked — no products need action today.</div>}
        </section>
        <section className="card">
          <h3>Total units sold per day</h3>
          <p className="sub">All products combined: last 5 weeks and next 4 weeks.</p>
          <ForecastChart series={d.total} height={280} />
        </section>
      </div>

      <section className="card">
        <h3>Stock plan by product</h3>
        <p className="sub">Reorder point = demand during delivery time + safety stock. Order quantity covers delivery time plus one week.</p>
        <div className="scroll"><table className="tbl">
          <thead><tr><th>Product</th><th>Status</th><th>Stockout risk</th><th className="num">In stock</th><th>Days of cover</th><th className="num">Sales next 7d</th><th className="num">Expected before delivery</th><th className="num">If demand is high</th><th className="num">Delivery (days)</th><th className="num">Safety stock</th><th className="num">Reorder point</th><th className="num">Order qty</th></tr></thead>
          <tbody>{d.items.map(i => (
            <tr key={i.id} className="row" onClick={() => setSel(i.id)} style={i.id === sel ? { background: '#eef3ea' } : null}>
              <td><b>{i.name}</b><br /><small style={{ color: '#5f6f66' }}>{i.category}</small></td>
              <td><Pill s={i.status} /></td>
              <td><RiskPill r={i.stockout_risk} /></td>
              <td className="num">{fmt(i.stock)}</td>
              <td><div className="bar"><i style={{ width: Math.min(100, i.cover_days / 20 * 100) + '%', background: STATUS_COLOR[i.status] }} /></div><small>{i.cover_days} days</small></td>
              <td className="num">{fmt(i.forecast_7)}</td>
              <td className="num">{fmt(i.demand_lead)}</td>
              <td className="num">{fmt(i.demand_lead_upper)}</td>
              <td className="num">{i.lead}</td>
              <td className="num">{fmt(i.safety)}</td><td className="num">{fmt(i.rop)}</td>
              <td className="num"><b>{i.order_qty ? fmt(i.order_qty) : '–'}</b></td>
            </tr>))}</tbody>
        </table></div>
      </section>

      <section className="card">
        <h3>{item.name}: demand forecast</h3>
        <p className="sub">Solid line is past sales, dashed line is the forecast, shaded area is the likely range (it widens further ahead). Forecast made with {item.model}{d.data_source === 'synthetic' ? ' using synthetic demo data.' : '.'}</p>
        <ForecastChart series={item.series} />
        <h3>Inventory scenario planner</h3>
        <p className="sub">Adjust these assumptions to see how the order suggestion changes. Changes are a local planning scenario and are not saved to your inventory system.</p>
        <div className="stat scenario-inputs">
          <label>Current stock<input className="numin" type="number" min="0" value={item.stock} onChange={e => setOverride('stock', e.target.value)} /></label>
          <label>On order<input className="numin" type="number" min="0" value={item.on_order} onChange={e => setOverride('on_order', e.target.value)} /></label>
          <label>Lead time days<input className="numin" type="number" min="1" value={item.lead} onChange={e => setOverride('lead', e.target.value)} /></label>
          <label>Unit cost<input className="numin" type="number" min="0" step="0.01" value={item.cost} onChange={e => setOverride('cost', e.target.value)} /></label>
          <label>Service level<select className="numin" value={item.service_level} onChange={e => setOverride('service_level', e.target.value)}><option value={.90}>90%</option><option value={.95}>95%</option><option value={.99}>99%</option></select></label>
        </div>
        <div className="stat">
          <div><b>{fmt(item.stock)}</b><span>units in stock</span></div>
          <div><b>{fmt(item.inventory_position)}</b><span>inventory position (stock + on order)</span></div>
          <div><b>{fmt(item.demand_lead)}</b><span>expected sales before delivery</span></div>
          <div><b>{fmt(item.demand_lead_upper)}</b><span>if demand is high</span></div>
          <div><b>{fmt(item.forecast_28)}</b><span>expected sales, 28 days</span></div>
          <div><b>{fmt(item.eoq)}</b><span>reference EOQ (units)</span></div>
          <div><b>{usd(item.order_cost)}</b><span>cost of suggested order</span></div>
        </div>
        <p className="sub" style={{ marginTop: 4 }}><b>Scenario status:</b> <Pill s={item.status} /> <b>Suggested order:</b> {item.order_qty ? `${fmt(item.order_qty)} units (${usd(item.order_cost)})` : 'none'} · Reorder point {fmt(item.rop)} units. {item.message}</p>
        <p className="sub" style={{ marginTop: 2 }}>Stockout risk: <b>{item.stockout_risk}</b> — {fmt(item.stock)} units in stock vs {fmt(item.demand_lead)} expected before the next delivery, or {fmt(item.demand_lead_upper)} if demand runs high.</p>
      </section>
    </>
  )
}
