import { useState } from 'react'
import { fmt } from './ui.jsx'

const Card = ({ t, s, children }) => <section className="card"><h3>{t}</h3>{s && <p className="sub">{s}</p>}{children}</section>

const REQUIRED = [
  { role: 'date', label: 'Date' },
  { role: 'product', label: 'Product / entity' },
  { role: 'sales', label: 'Sales / demand' },
]
const OPTIONAL = [
  { role: 'store', label: 'Store / location' },
  { role: 'category', label: 'Category' },
  { role: 'price', label: 'Price' },
  { role: 'discount', label: 'Discount' },
  { role: 'promo', label: 'Promotion flag' },
  { role: 'temp', label: 'Temperature' },
  { role: 'holiday_lift', label: 'Holiday' },
  { role: 'stockout', label: 'Stock-out flag' },
]

const confColor = c => (c >= 0.75 ? '#1d6238' : c >= 0.5 ? '#8a6100' : '#a92917')

export default function GenericUpload({ onStarted, onCancel, onBack, initialUpload = null }) {
  const [info, setInfo] = useState(initialUpload)
  const profile = info?.profile
  const layout = profile?.layout || 'long'
  const isWide = layout === 'wide'
  const [mapping, setMapping] = useState({ ...(initialUpload?.profile?.mapping || {}) })
  const [valueColumns, setValueColumns] = useState(
    initialUpload?.profile?.mapping?.value_columns || initialUpload?.profile?.wide_columns || [])
  const [filters, setFilters] = useState({
    store: '',
    start: initialUpload?.profile?.date_min || '',
    end: initialUpload?.profile?.date_max || '',
  })
  const [prepared, setPrepared] = useState(null)
  const [inv, setInv] = useState({})
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)

  async function upload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true); setMsg(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const r = await (await fetch('/api/upload', { method: 'POST', body: fd })).json()
      if (r.adapter !== 'generic') throw new Error(r.adapter
        ? `This looks like a ${r.adapter} file — use that tab instead.`
        : (r.errors?.[0] || 'Could not profile this CSV.'))
      setInfo(r); setMapping({ ...(r.profile?.mapping || {}) })
      setValueColumns(r.profile?.mapping?.value_columns || r.profile?.wide_columns || [])
      setFilters({ store: '', start: r.profile?.date_min || '', end: r.profile?.date_max || '' })
    } catch (error) { setMsg(error.message) }
    setBusy(false)
  }

  if (!profile) {
    return <>
      <div className="head"><div><h1>Generic CSV</h1><p>Upload any sales CSV. Its columns are profiled and mapped to the model schema automatically; only date, product/entity and sales are required.</p></div></div>
      <div className="srcbtns" style={{ marginBottom: 16 }}><button className="srcbtn" onClick={onBack}>Use another format</button><button className="srcbtn" onClick={onCancel}>Cancel</button></div>
      <Card t="Choose a CSV file" s="Headers are matched case-insensitively. Long and wide layouts are both supported. Favorita and store+family files are detected automatically and offered their own flow.">
        <input type="file" accept=".csv,text/csv" onChange={upload} disabled={busy} />
        {busy && <p className="sub" style={{ marginTop: 8 }}>Profiling…</p>}
        {msg && <p className="sub" style={{ color: '#a92917', marginTop: 8 }}>{msg}</p>}
      </Card>
    </>
  }

  const columns = profile.columns.map(c => c.name)
  const missingRequired = isWide
    ? [...(!mapping.date ? ['Date'] : []), ...(valueColumns.length ? [] : ['Value columns'])]
    : REQUIRED.filter(r => !mapping[r.role]).map(r => r.label)
  const setRole = (role, value) => setMapping(m => ({ ...m, [role]: value || undefined }))
  const toggleValue = col => setValueColumns(cols => cols.includes(col) ? cols.filter(c => c !== col) : [...cols, col])

  async function prepare() {
    setBusy(true); setMsg(null)
    try {
      const sendMapping = isWide
        ? { layout: 'wide', date: mapping.date, store: mapping.store, value_columns: valueColumns, id_columns: profile.mapping?.id_columns || [] }
        : mapping
      const body = {
        mapping: sendMapping,
        store: filters.store || undefined,
        start_date: filters.start || undefined,
        end_date: filters.end || undefined,
        wide_missing: filters.wideMissing || 'zero',
      }
      const result = await (await fetch(`/api/generic/datasets/${info.dataset_id}/prepare`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })).json()
      if (!result.ok) throw new Error(result.error || 'Could not prepare this mapping.')
      setPrepared(result)
      const next = {}
      result.detected.products.forEach(p => { next[p.id] = { stock: p.suggested_stock, lead: 3, service_level: 0.95, on_order: 0, unit_cost: p.price ? (p.price * 0.7).toFixed(2) : '' } })
      setInv(next)
    } catch (error) { setMsg(error.message) }
    setBusy(false)
  }

  async function start() {
    setBusy(true); setMsg(null)
    try {
      const result = await (await fetch('/api/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'uploaded', dataset_id: info.dataset_id, inventory: inv }),
      })).json()
      if (!result.ok) throw new Error(result.error || 'Could not start forecasting.')
      onStarted()
    } catch (error) { setMsg(error.message) }
    setBusy(false)
  }

  const setRow = (id, key, value) => setInv(v => ({ ...v, [id]: { ...v[id], [key]: value } }))
  const roleRow = ({ role, label }) => {
    const conf = profile.confidence?.[role] ?? 0
    const required = REQUIRED.some(r => r.role === role)
    return (
      <tr key={role}>
        <td><b>{label}</b>{required ? <span style={{ color: '#a92917' }}> *</span> : null}</td>
        <td>
          <select className="numin" value={mapping[role] || ''} onChange={e => setRole(role, e.target.value)}>
            <option value="">{required ? '— choose a column —' : '— not present —'}</option>
            {columns.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </td>
        <td className="num">
          {mapping[role] ? <span style={{ color: confColor(conf), fontWeight: 600 }}>{Math.round(conf * 100)}%</span> : <span className="sub">—</span>}
        </td>
        <td className="sub">{profile.roles?.[role]?.[0]?.reasons?.join('; ') || ''}</td>
      </tr>
    )
  }

  const idChoices = columns.filter(c => c !== mapping.date && !valueColumns.includes(c))

  return <>
    <div className="head"><div><h1>Generic CSV</h1><p>Columns were detected automatically. Confirm or fix the mapping; only date, product/entity and sales are required — every other field is optional.</p></div></div>
    <div className="srcbtns" style={{ marginBottom: 16 }}><button className="srcbtn" onClick={onBack}>Use another format</button><button className="srcbtn" onClick={onCancel}>Cancel</button></div>
    <div className="stat" style={{ marginBottom: 12 }}>
      <div><b>{fmt(profile.rows_sampled)}</b><span>rows profiled</span></div>
      <div><b>{layout === 'wide' ? 'Wide (matrix)' : 'Long (one row per obs.)'}</b><span>detected layout</span></div>
      <div><b>{profile.date_format || '—'}</b><span>date format</span></div>
    </div>

    {isWide
      ? <Card t="1. Wide layout" s="Each value column is treated as a separate product/series and melted into long rows. Choose the date column, optional id column and the value columns to keep.">
          {missingRequired.length > 0 && <ul className="issues warn">{missingRequired.map(x => <li key={x}>Required: {x}</li>)}</ul>}
          <div className="scenario-inputs">
            <label>Date column<select className="numin" value={mapping.date || ''} onChange={e => setRole('date', e.target.value)}><option value="">— choose a column —</option>{columns.map(c => <option key={c} value={c}>{c}</option>)}</select></label>
            <label>Id column (optional)<select className="numin" value={mapping.store || ''} onChange={e => setRole('store', e.target.value)}><option value="">— none —</option>{idChoices.map(c => <option key={c} value={c}>{c}</option>)}</select></label>
          </div>
          <p className="sub" style={{ marginTop: 12 }}>Value columns ({valueColumns.length} selected):</p>
          <div className="item-picker">
            {profile.wide_columns.map(c => <label key={c}><input type="checkbox" checked={valueColumns.includes(c)} onChange={() => toggleValue(c)} /> {c}</label>)}
          </div>
        </Card>
      : <Card t="1. Column mapping" s="Confidence is how strongly a column matched a role from its name and values. Adjust any dropdown, then prepare.">
          {missingRequired.length > 0 && <ul className="issues warn">{missingRequired.map(x => <li key={x}>Required role not set: {x}</li>)}</ul>}
          <div className="scroll"><table className="tbl">
            <thead><tr><th>Role</th><th>Column</th><th className="num">Confidence</th><th>Why</th></tr></thead>
            <tbody>{REQUIRED.map(roleRow)}{OPTIONAL.map(roleRow)}</tbody>
          </table></div>
        </Card>}
    {profile.warnings?.length > 0 && <ul className="issues warn" style={{ margin: '0 0 14px 0' }}>{profile.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}

    <Card t="2. Optional filters" s="Optionally restrict rows. If an id/store column is mapped and not filtered, series are keyed as store|entity.">
      <div className="scenario-inputs">
        {profile.stores?.length > 0 && <label>Store<select className="numin" value={filters.store} onChange={e => setFilters(f => ({ ...f, store: e.target.value }))}><option value="">All stores</option>{profile.stores.map(s => <option key={s} value={s}>{s}</option>)}</select></label>}
        <label>Start date<input className="numin" type="date" min={profile.date_min || undefined} max={profile.date_max || undefined} value={filters.start} onChange={e => setFilters(f => ({ ...f, start: e.target.value }))} /></label>
        <label>End date<input className="numin" type="date" min={profile.date_min || undefined} max={profile.date_max || undefined} value={filters.end} onChange={e => setFilters(f => ({ ...f, end: e.target.value }))} /></label>
        {isWide && <label>Missing cells<select className="numin" value={filters.wideMissing || 'zero'} onChange={e => setFilters(f => ({ ...f, wideMissing: e.target.value }))}><option value="zero">Treat as zero</option><option value="dropped">Drop the row</option></select></label>}
      </div>
      {msg && <p className="sub" style={{ color: '#a92917', marginTop: 10 }}>{msg}</p>}
      <div style={{ marginTop: 14 }}><button className="act" onClick={prepare} disabled={busy || missingRequired.length > 0}>Prepare dataset</button></div>
    </Card>

    {prepared && <>
      <Card t="3. Prepared dataset" s="Normalized to date, product, sales, discount, price, temp, holiday_lift and stockout. Duplicates are aggregated per product+date; missing optional fields use neutral defaults.">
        <div className="stat"><div><b>{fmt(prepared.detected.rows)}</b><span>daily rows</span></div><div><b>{prepared.detected.skus}</b><span>SKUs</span></div><div><b>{prepared.detected.date_min} → {prepared.detected.date_max}</b><span>date range</span></div></div>
        {prepared.detected.report && <p className="sub"><b>Normalization:</b> layout {prepared.detected.layout}; date format {prepared.detected.report.date_format || '—'}; duplicates aggregated {prepared.detected.report.duplicates_aggregated ?? 0}; rows dropped for missing required {prepared.detected.report.dropped_missing_required ?? 0}{prepared.detected.report.wide_cells_filled_zero ? `; wide cells filled with 0: ${prepared.detected.report.wide_cells_filled_zero}` : ''}.</p>}
        {prepared.detected.warnings?.length > 0 && <ul className="issues warn">{prepared.detected.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
        <div className="scroll" style={{ marginTop: 10 }}><table className="tbl">
          <thead><tr><th>SKU</th><th>Category</th><th className="num">Rows</th><th className="num">Suggested stock</th></tr></thead>
          <tbody>{prepared.detected.products.map(p => <tr key={p.id}><td><b>{p.id}</b></td><td>{p.category}</td><td className="num">{p.rows}</td><td className="num">{p.suggested_stock}</td></tr>)}</tbody>
        </table></div>
      </Card>

      <Card t="4. Inventory assumptions" s="The CSV may not include stock or lead time; set them here. Suggested stock is a demo default from recent sales.">
        <div className="scroll"><table className="tbl">
          <thead><tr><th>SKU</th><th className="num">Current stock</th><th className="num">On order</th><th className="num">Lead time (days)</th><th className="num">Unit cost</th><th className="num">Service level</th></tr></thead>
          <tbody>{prepared.detected.products.map(p => (
            <tr key={p.id}>
              <td><b>{p.id}</b></td>
              <td className="num"><input className="numin" type="number" min="0" value={inv[p.id]?.stock ?? 0} onChange={e => setRow(p.id, 'stock', e.target.value)} /></td>
              <td className="num"><input className="numin" type="number" min="0" value={inv[p.id]?.on_order ?? 0} onChange={e => setRow(p.id, 'on_order', e.target.value)} /></td>
              <td className="num"><input className="numin" type="number" min="1" value={inv[p.id]?.lead ?? 3} onChange={e => setRow(p.id, 'lead', e.target.value)} /></td>
              <td className="num"><input className="numin" type="number" min="0" step="0.01" value={inv[p.id]?.unit_cost ?? ''} onChange={e => setRow(p.id, 'unit_cost', e.target.value)} /></td>
              <td className="num"><select className="numin" value={inv[p.id]?.service_level ?? .95} onChange={e => setRow(p.id, 'service_level', e.target.value)}><option value={.9}>90%</option><option value={.95}>95%</option><option value={.99}>99%</option></select></td>
            </tr>))}
          </tbody></table></div>
        <button className="act" style={{ marginTop: 12 }} onClick={start} disabled={busy}>Start forecasting</button>
      </Card>
    </>}
  </>
}
