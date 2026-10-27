import { useState } from 'react'
import { fmt } from './ui.jsx'
import { Stepper, WizardNav } from './Wizard.jsx'

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

export default function GenericUpload({ onStarted, onCancel, initialUpload = null, onAdapter }) {
  const [info, setInfo] = useState(initialUpload)
  const profile = info?.profile
  const layout = profile?.layout || 'long'
  const isWide = layout === 'wide'
  const [step, setStep] = useState(initialUpload ? 1 : 0)
  const [mapping, setMapping] = useState({ ...(initialUpload?.profile?.mapping || {}) })
  const [valueColumns, setValueColumns] = useState(
    initialUpload?.profile?.mapping?.value_columns || initialUpload?.profile?.wide_columns || [])
  const [filters, setFilters] = useState({ store: '', products: [], start: '', end: '', wideMissing: 'zero' })
  const [prepared, setPrepared] = useState(null)
  const [inv, setInv] = useState({})
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)

  function adopt(r) {
    setInfo(r)
    setMapping({ ...(r.profile?.mapping || {}) })
    setValueColumns(r.profile?.mapping?.value_columns || r.profile?.wide_columns || [])
    setFilters({ store: '', products: [], start: r.profile?.date_min || '', end: r.profile?.date_max || '', wideMissing: 'zero' })
    setPrepared(null); setInv({}); setStep(1); setMsg(null)
  }

  async function upload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true); setMsg(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const r = await (await fetch('/api/upload', { method: 'POST', body: fd })).json()
      if (r.adapter === 'favorita' || r.adapter === 'store_family') {
        if (onAdapter) { onAdapter(r); return }
      }
      if (r.adapter !== 'generic') throw new Error(r.errors?.[0] || 'Could not profile this CSV.')
      adopt(r)
    } catch (error) { setMsg(error.message) }
    setBusy(false)
  }

  async function prepare() {
    setBusy(true); setMsg(null)
    try {
      const sendMapping = isWide
        ? { layout: 'wide', date: mapping.date, store: mapping.store, value_columns: valueColumns, id_columns: profile.mapping?.id_columns || [] }
        : mapping
      const body = {
        mapping: sendMapping,
        store: filters.store || undefined,
        products: !isWide && filters.products.length ? filters.products : undefined,
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
      setStep(5)
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

  const setRole = (role, value) => setMapping(m => ({ ...m, [role]: value || undefined }))
  const toggleValue = col => setValueColumns(cols => cols.includes(col) ? cols.filter(c => c !== col) : [...cols, col])
  const toggleProduct = p => setFilters(f => ({ ...f, products: f.products.includes(p) ? f.products.filter(x => x !== p) : [...f.products, p] }))
  const setRow = (id, key, value) => setInv(v => ({ ...v, [id]: { ...v[id], [key]: value } }))

  if (!profile) {
    return <>
      <div className="head"><div><h1>Upload a CSV</h1><p>A guided path: upload, understand, map, check quality, select, set inventory, forecast.</p></div></div>
      <Stepper current={0} />
      <Card t="1. Upload CSV" s="Any sales export. Headers are matched case-insensitively; long and wide layouts are both supported.">
        <input type="file" accept=".csv,text/csv" onChange={upload} disabled={busy} />
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 12 }}>
          <a className="srcbtn" href="/api/sample-csv" download>Download sample CSV</a>
          <button className="srcbtn" onClick={onCancel} disabled={busy}>Cancel</button>
        </div>
        {busy && <p className="sub" style={{ marginTop: 8 }}>Uploading and profiling…</p>}
        {msg && <p className="sub" style={{ color: '#a92917', marginTop: 8 }}>{msg}</p>}
      </Card>
    </>
  }

  const columns = profile.columns.map(c => c.name)
  const presentOptional = (profile.optional?.present || []).map(r => (OPTIONAL.find(o => o.role === r) || {}).label || r)
  const absentOptional = OPTIONAL.filter(o => !(profile.optional?.present || []).includes(o.role)).map(o => o.label)
  const missingRequired = isWide
    ? [...(!mapping.date ? ['Date'] : []), ...(valueColumns.length ? [] : ['Value columns'])]
    : REQUIRED.filter(r => !mapping[r.role]).map(r => r.label)
  const idChoices = columns.filter(c => c !== mapping.date && !valueColumns.includes(c))

  const stepTitle = (idx, title, sub) => <div className="steptitle"><span className="idx">STEP {idx + 1}</span><h2>{title}</h2></div>

  const rolesSummary = isWide
    ? [['Date', mapping.date], ['Store / id', mapping.store || '—'], ['Value columns', `${valueColumns.length} selected`]]
    : [...REQUIRED, ...OPTIONAL].map(({ role, label }) => [label + (REQUIRED.some(r => r.role === role) ? ' *' : ''), mapping[role] ? `${mapping[role]} (${Math.round((profile.confidence?.[role] || 0) * 100)}%)` : '—'])

  const quality = profile.quality

  return <>
    <div className="head"><div><h1>Upload a CSV</h1><p>{info.dataset_id} · {fmt(profile.rows_sampled)} rows profiled · {isWide ? 'wide' : 'long'} layout</p></div></div>
    <Stepper current={step} />

    {step === 0 && <Card t="1. Upload CSV">
      <input type="file" accept=".csv,text/csv" onChange={upload} disabled={busy} />
      <p className="sub" style={{ marginTop: 10 }}>Current file: <b>{info.dataset_id}</b>. Choosing another file restarts the wizard.</p>
      {msg && <p className="sub" style={{ color: '#a92917' }}>{msg}</p>}
    </Card>}

    {step === 1 && <Card t="2. Understand data">
      <div className="stat"><div><b>{isWide ? 'Wide (matrix)' : 'Long (one row per obs.)'}</b><span>layout</span></div><div><b>{profile.date_format || '—'}</b><span>date format</span></div><div><b>{fmt(profile.rows_sampled)}</b><span>rows profiled</span></div></div>
      <p className="sub" style={{ marginTop: 6 }}><b>Detected roles:</b></p>
      <div className="scroll"><table className="tbl">
        <thead><tr><th>Role</th><th>Detected column</th></tr></thead>
        <tbody>{rolesSummary.map(([label, value]) => <tr key={label}><td>{label}</td><td>{value}</td></tr>)}</tbody>
      </table></div>
      <p className="sub" style={{ marginTop: 12 }}><b>Columns:</b></p>
      <div className="scroll"><table className="tbl">
        <thead><tr><th>Column</th><th>Type</th><th className="num">Unique</th><th className="num">Missing</th><th>Looks like</th></tr></thead>
        <tbody>{profile.columns.map(c => <tr key={c.name}><td><b>{c.name}</b></td><td>{c.dtype}</td><td className="num">{fmt(c.unique)}</td><td className="num">{Math.round(c.missing_fraction * 100)}%</td><td>{c.date_like ? 'date' : c.numeric ? 'numeric' : 'text'}</td></tr>)}</tbody>
      </table></div>
      <p className="sub" style={{ marginTop: 12 }}><b>Optional drivers detected:</b> {presentOptional.length ? presentOptional.join(', ') : 'none'}. <b>Missing:</b> {absentOptional.length ? absentOptional.join(', ') : 'none'}. Optional features that are absent or constant are automatically excluded from the models, and the next 28 days reuse the last observed price/temperature with a calendar-derived holiday lift.</p>
    </Card>}

    {step === 2 && (isWide
      ? <Card t="3. Column mapping" s="Wide layout: each value column becomes a product/series. Choose the date column, an optional id column and the value columns.">
          {missingRequired.length > 0 && <ul className="issues warn">{missingRequired.map(x => <li key={x}>Required: {x}</li>)}</ul>}
          <div className="scenario-inputs">
            <label>Date column<select className="numin" value={mapping.date || ''} onChange={e => setRole('date', e.target.value)}><option value="">— choose —</option>{columns.map(c => <option key={c} value={c}>{c}</option>)}</select></label>
            <label>Id column (optional)<select className="numin" value={mapping.store || ''} onChange={e => setRole('store', e.target.value)}><option value="">— none —</option>{idChoices.map(c => <option key={c} value={c}>{c}</option>)}</select></label>
          </div>
          <p className="sub" style={{ marginTop: 12 }}>Value columns ({valueColumns.length} selected):</p>
          <div className="item-picker">{profile.wide_columns.map(c => <label key={c}><input type="checkbox" checked={valueColumns.includes(c)} onChange={() => toggleValue(c)} /> {c}</label>)}</div>
        </Card>
      : <Card t="3. Column mapping" s="Confirm or change how each role maps to a column. Confidence comes from the column name and its values.">
          {missingRequired.length > 0 && <ul className="issues warn">{missingRequired.map(x => <li key={x}>Required role not set: {x}</li>)}</ul>}
          <div className="scroll"><table className="tbl">
            <thead><tr><th>Role</th><th>Column</th><th className="num">Confidence</th><th>Why</th></tr></thead>
            <tbody>{[...REQUIRED, ...OPTIONAL].map(({ role, label }) => {
              const conf = profile.confidence?.[role] ?? 0
              const required = REQUIRED.some(r => r.role === role)
              return <tr key={role}><td><b>{label}</b>{required ? <span style={{ color: '#a92917' }}> *</span> : null}</td>
                <td><select className="numin" value={mapping[role] || ''} onChange={e => setRole(role, e.target.value)}><option value="">{required ? '— choose —' : '— not present —'}</option>{columns.map(c => <option key={c} value={c}>{c}</option>)}</select></td>
                <td className="num">{mapping[role] ? <span style={{ color: confColor(conf), fontWeight: 600 }}>{Math.round(conf * 100)}%</span> : <span className="sub">—</span>}</td>
                <td className="sub">{profile.roles?.[role]?.[0]?.reasons?.join('; ') || ''}</td></tr>
            })}</tbody>
          </table></div>
        </Card>)}

    {step === 3 && <Card t="4. Data quality" s="Full-file scan in chunks — there is no raw row limit. Only the filtered subset is capped (200,000 rows).">
      {quality
        ? <>
            <div className="stat">
              <div><b>{fmt(quality.rows_scanned)}</b><span>rows scanned</span></div>
              <div><b>{fmt(quality.skus)}</b><span>entities</span></div>
              <div><b>{fmt(quality.stores)}</b><span>stores</span></div>
              <div><b>{quality.date_min || '—'} → {quality.date_max || '—'}</b><span>date range</span></div>
            </div>
            <div className="scroll"><table className="tbl">
              <thead><tr><th>Check</th><th className="num">Count</th></tr></thead>
              <tbody>
                <tr><td>Invalid dates</td><td className="num">{fmt(quality.invalid_dates)}</td></tr>
                <tr><td>Non-numeric sales</td><td className="num">{fmt(quality.invalid_sales)}</td></tr>
                <tr><td>Missing sales</td><td className="num">{fmt(quality.missing_sales)}</td></tr>
                <tr><td>Negative sales</td><td className="num">{fmt(quality.negative_sales)}</td></tr>
                <tr><td>Missing entity ids</td><td className="num">{fmt(quality.missing_product)}</td></tr>
              </tbody>
            </table></div>
          </>
        : <p className="sub">Quality scan unavailable (required roles not yet mapped).</p>}
      {profile.warnings?.length > 0 && <ul className="issues warn" style={{ marginTop: 12 }}>{profile.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
    </Card>}

    {step === 4 && <Card t="5. Store / SKU / date selection" s="Filtering happens before the 200,000-row forecasting cap, so narrow a large dataset here.">
      <div className="scenario-inputs">
        {profile.stores?.length > 0 && <label>Store<select className="numin" value={filters.store} onChange={e => setFilters(f => ({ ...f, store: e.target.value }))}><option value="">All stores</option>{profile.stores.map(s => <option key={s} value={s}>{s}</option>)}</select></label>}
        <label>Start date<input className="numin" type="date" min={profile.date_min || undefined} max={profile.date_max || undefined} value={filters.start} onChange={e => setFilters(f => ({ ...f, start: e.target.value }))} /></label>
        <label>End date<input className="numin" type="date" min={profile.date_min || undefined} max={profile.date_max || undefined} value={filters.end} onChange={e => setFilters(f => ({ ...f, end: e.target.value }))} /></label>
        {isWide && <label>Missing cells<select className="numin" value={filters.wideMissing} onChange={e => setFilters(f => ({ ...f, wideMissing: e.target.value }))}><option value="zero">Treat as zero</option><option value="dropped">Drop the row</option></select></label>}
      </div>
      {!isWide && profile.entities?.length > 0 && <>
        <p className="sub" style={{ marginTop: 12 }}>Products ({filters.products.length ? `${filters.products.length} selected` : 'all'}) — optional, from the profiled sample:</p>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <button className="srcbtn" type="button" onClick={() => setFilters(f => ({ ...f, products: profile.entities.slice(0, 25) }))}>Select first 25</button>
          <button className="srcbtn" type="button" onClick={() => setFilters(f => ({ ...f, products: [] }))}>Clear</button>
        </div>
        <div className="item-picker">{profile.entities.slice(0, 120).map(p => <label key={p}><input type="checkbox" checked={filters.products.includes(p)} onChange={() => toggleProduct(p)} /> {p}</label>)}{profile.entities.length > 120 && <small>Showing first 120 of {profile.entities.length}; leave unselected to use all.</small>}</div>
      </>}
      {msg && <p className="sub" style={{ color: '#a92917', marginTop: 10 }}>{msg}</p>}
      <div className="wnav"><button className="srcbtn" onClick={() => setStep(3)} disabled={busy}>Back</button><button className="act" style={{ marginLeft: 'auto' }} onClick={prepare} disabled={busy || missingRequired.length > 0}>Prepare selection & continue</button></div>
    </Card>}

    {step === 5 && prepared && <Card t="6. Inventory assumptions" s="The source may not include stock, lead time or cost; set them here. Suggested stock is a demo default from recent sales.">
      <div className="stat"><div><b>{fmt(prepared.detected.rows)}</b><span>daily rows</span></div><div><b>{prepared.detected.skus}</b><span>SKUs</span></div><div><b>{prepared.detected.date_min} → {prepared.detected.date_max}</b><span>range</span></div></div>
      {prepared.detected.report && <p className="sub"><b>Preparation:</b> layout {prepared.detected.layout}; date format {prepared.detected.report.date_format || '—'}; duplicates aggregated {prepared.detected.report.duplicates_aggregated ?? 0}; dropped for missing required {prepared.detected.report.dropped_missing_required ?? 0}{prepared.detected.report.wide_cells_filled_zero ? `; wide cells filled with 0: ${prepared.detected.report.wide_cells_filled_zero}` : ''}.</p>}
      {prepared.detected.warnings?.length > 0 && <ul className="issues warn">{prepared.detected.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
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
    </Card>}

    {step === 6 && prepared && <Card t="7. Start forecasting" s="Review the run and start. Each SKU is evaluated against all four models; the best is refit and forecast 28 days.">
      <div className="stat"><div><b>{prepared.detected.skus}</b><span>SKUs</span></div><div><b>{fmt(prepared.detected.rows)}</b><span>daily rows</span></div><div><b>{prepared.detected.date_min} → {prepared.detected.date_max}</b><span>range</span></div><div><b>{layout}</b><span>layout</span></div></div>
      {msg && <p className="sub" style={{ color: '#a92917' }}>{msg}</p>}
      <button className="act" onClick={start} disabled={busy}>{busy ? 'Starting…' : 'Start forecasting'}</button>
    </Card>}

    {step >= 1 && step <= 3 && <WizardNav onBack={step === 1 ? () => setStep(0) : () => setStep(step - 1)} onNext={() => setStep(step + 1)} nextDisabled={step === 2 && missingRequired.length > 0} busy={busy} />}
    {step === 5 && <WizardNav onBack={() => setStep(4)} onNext={() => setStep(6)} busy={busy} />}
    {step === 6 && <WizardNav onBack={() => setStep(5)} busy={busy} />}
  </>
}
