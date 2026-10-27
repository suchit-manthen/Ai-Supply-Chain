import { useMemo, useState } from 'react'
import { fmt } from './ui.jsx'
import { Stepper, WizardNav } from './Wizard.jsx'

const Card = ({ t, s, children }) => <section className="card"><h3>{t}</h3>{s && <p className="sub">{s}</p>}{children}</section>

const MODES = {
  favorita: {
    title: 'Favorita CSV',
    blurb: 'Upload the large Kaggle train.csv, then prepare a small store and item subset for the forecasting pipeline.',
    uploadHint: 'Required columns: date, store_nbr, item_nbr, unit_sales. Optional: onpromotion. The file is scanned in chunks; it is not loaded as one large table.',
    base: '/api/favorita',
    entitySingular: 'Item',
    entityPlural: 'items',
    listKey: 'store_items',
    requestKey: 'item_nbrs',
    inventoryNote: 'Favorita train.csv does not include on-hand stock, lead time or cost. Suggested stock is a planning default from recent sales.',
    schema: [['date', 'date'], ['item_nbr', 'product'], ['unit_sales', 'sales'], ['onpromotion', 'discount (promo flag)'], ['store_nbr', 'store (filter)']],
  },
  store_family: {
    title: 'Store + family CSV',
    blurb: 'Upload the store/family train.csv, then prepare a small store and product-family subset for the forecasting pipeline.',
    uploadHint: 'Required columns: date, store_nbr, family, sales. Optional: onpromotion. The file is scanned in chunks; it is not loaded as one large table.',
    base: '/api/store-family',
    entitySingular: 'Family',
    entityPlural: 'families',
    listKey: 'store_families',
    requestKey: 'families',
    inventoryNote: 'This source has no on-hand stock, lead time or cost. Suggested stock is a planning default from recent sales.',
    schema: [['date', 'date'], ['family', 'product'], ['sales', 'sales'], ['onpromotion', 'discount (promo flag)'], ['store_nbr', 'store (filter)']],
  },
}

export default function FavoritaUpload({ onStarted, onCancel, initialUpload = null, mode = 'favorita' }) {
  const cfg = MODES[mode] || MODES.favorita
  const [fileInfo, setFileInfo] = useState(initialUpload)
  const [step, setStep] = useState(initialUpload ? 1 : 0)
  const [store, setStore] = useState('')
  const [items, setItems] = useState([])
  const [chosen, setChosen] = useState([])
  const [query, setQuery] = useState('')
  const [dates, setDates] = useState({ start: initialUpload?.dataset?.date_min || '', end: initialUpload?.dataset?.date_max || '' })
  const [prepared, setPrepared] = useState(null)
  const [inv, setInv] = useState({})
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)

  async function upload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true); setMsg(null); setPrepared(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const result = await (await fetch(`${cfg.base}/upload`, { method: 'POST', body: fd })).json()
      if (!result.ok) throw new Error(result.error || 'Could not accept this CSV.')
      setFileInfo(result)
      setDates({ start: result.dataset.date_min || '', end: result.dataset.date_max || '' })
      setStore(''); setItems([]); setChosen([]); setStep(1)
    } catch (error) { setMsg(error.message || 'Upload failed. Is the backend running?') }
    setBusy(false)
  }

  async function selectStore(value) {
    setStore(value); setChosen([]); setItems([]); setMsg(null)
    if (!value || !fileInfo) return
    setBusy(true)
    try {
      const result = await (await fetch(`${cfg.base}/datasets/${fileInfo.dataset_id}?store_nbr=${encodeURIComponent(value)}`)).json()
      if (!result.ok) throw new Error(result.error || `Could not load store ${cfg.entityPlural}.`)
      setItems(result.dataset[cfg.listKey] || [])
    } catch (error) { setMsg(error.message) }
    setBusy(false)
  }

  function toggleItem(item) {
    setChosen(current => current.includes(item) ? current.filter(x => x !== item) : current.length < 8 ? [...current, item] : current)
  }

  async function prepare() {
    setBusy(true); setMsg(null)
    try {
      const response = await fetch(`${cfg.base}/datasets/${fileInfo.dataset_id}/prepare`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ store_nbr: store, [cfg.requestKey]: chosen, start_date: dates.start, end_date: dates.end }),
      })
      const result = await response.json()
      if (!result.ok) throw new Error(result.error || 'Could not prepare this selection.')
      setPrepared(result)
      const next = {}
      result.detected.products.forEach(p => { next[p.id] = { stock: p.suggested_stock, lead: 3, service_level: 0.95, on_order: 0, unit_cost: '' } })
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
        body: JSON.stringify({ source: 'uploaded', dataset_id: fileInfo.dataset_id, inventory: inv }),
      })).json()
      if (!result.ok) throw new Error(result.error || 'Could not start forecasting.')
      onStarted()
    } catch (error) { setMsg(error.message) }
    setBusy(false)
  }

  const metadata = fileInfo?.dataset
  const visibleItems = useMemo(() => items.filter(item => item.toLowerCase().includes(query.trim().toLowerCase())).slice(0, 120), [items, query])
  const setRow = (id, key, value) => setInv(v => ({ ...v, [id]: { ...v[id], [key]: value } }))

  if (!metadata) {
    return <>
      <div className="head"><div><h1>{cfg.title}</h1><p>{cfg.blurb}</p></div></div>
      <Stepper current={0} />
      <Card t="1. Upload CSV" s={cfg.uploadHint}>
        <input type="file" accept=".csv,text/csv" onChange={upload} disabled={busy} />
        <div style={{ marginTop: 12 }}><button className="srcbtn" onClick={onCancel} disabled={busy}>Cancel</button></div>
        {busy && <p className="sub" style={{ marginTop: 8 }}>Uploading…</p>}
        {msg && <p className="sub" style={{ color: '#a92917', marginTop: 8 }}>{msg}</p>}
      </Card>
    </>
  }

  return <>
    <div className="head"><div><h1>{cfg.title}</h1><p>{fileInfo.dataset_id} · scanned in chunks</p></div></div>
    <Stepper current={step} />

    {step === 0 && <Card t="1. Upload CSV">
      <input type="file" accept=".csv,text/csv" onChange={upload} disabled={busy} />
      <p className="sub" style={{ marginTop: 10 }}>Current file size {(fileInfo.size_bytes / (1024 * 1024)).toFixed(1)} MB. Choosing another file restarts.</p>
      {msg && <p className="sub" style={{ color: '#a92917' }}>{msg}</p>}
    </Card>}

    {step === 1 && <Card t="2. Understand data" s="Statistics are computed while reading the file in chunks.">
      <div className="stat"><div><b>{fmt(metadata.rows)}</b><span>rows</span></div><div><b>{metadata.date_min || '—'} → {metadata.date_max || '—'}</b><span>date range</span></div><div><b>{fmt(metadata.stores)}</b><span>stores</span></div><div><b>{fmt(metadata.skus)}</b><span>distinct {cfg.entityPlural}</span></div></div>
      <p className="sub"><b>Available columns:</b> {metadata.available_columns.join(', ')}</p>
    </Card>}

    {step === 2 && <Card t="3. Column mapping" s="This file matches a known schema, so the mapping is fixed and applied automatically.">
      <div className="scroll"><table className="tbl"><thead><tr><th>Source column</th><th>Maps to</th><th className="num">Confidence</th></tr></thead>
        <tbody>{cfg.schema.map(([src, dst]) => <tr key={src}><td><b>{src}</b></td><td>{dst}</td><td className="num" style={{ color: '#1d6238', fontWeight: 600 }}>100%</td></tr>)}</tbody>
      </table></div>
    </Card>}

    {step === 3 && <Card t="4. Data quality" s="Counts gathered during the chunked scan.">
      <div className="scroll"><table className="tbl"><thead><tr><th>Check</th><th className="num">Count</th></tr></thead>
        <tbody>
          <tr><td>Invalid dates</td><td className="num">{fmt(metadata.quality.invalid_dates)}</td></tr>
          <tr><td>Non-numeric sales</td><td className="num">{fmt(metadata.quality.invalid_sales)}</td></tr>
          <tr><td>Missing sales</td><td className="num">{fmt(metadata.quality.missing_sales)}</td></tr>
          <tr><td>Negative sales</td><td className="num">{fmt(metadata.quality.negative_unit_sales_rows ?? metadata.quality.negative_sales_rows)}</td></tr>
          <tr><td>Missing store/entity ids</td><td className="num">{fmt(metadata.quality.missing_store_or_item ?? metadata.quality.missing_store_or_family)}</td></tr>
          <tr><td>Missing promotion flags</td><td className="num">{fmt(metadata.quality.missing_promotion)}</td></tr>
        </tbody></table></div>
      {metadata.warnings.length > 0 && <ul className="issues warn" style={{ marginTop: 12 }}>{metadata.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
    </Card>}

    {step === 4 && <Card t="5. Store / SKU / date selection" s={`Choose one store, up to eight ${cfg.entityPlural}, and a date range. Duplicates are aggregated (sales summed, promotion OR-ed).`}>
      <div className="scenario-inputs">
        <label>Store<select className="numin" value={store} onChange={e => selectStore(e.target.value)}><option value="">Choose store</option>{metadata.store_options.map(s => <option key={s} value={s}>{s}</option>)}</select></label>
        <label>Start date<input className="numin" type="date" min={metadata.date_min || undefined} max={metadata.date_max || undefined} value={dates.start} onChange={e => setDates(d => ({ ...d, start: e.target.value }))} /></label>
        <label>End date<input className="numin" type="date" min={metadata.date_min || undefined} max={metadata.date_max || undefined} value={dates.end} onChange={e => setDates(d => ({ ...d, end: e.target.value }))} /></label>
      </div>
      {store && <>
        <p className="sub" style={{ marginTop: 12 }}>{items.length} {cfg.entityPlural} found for store {store}. Selected {chosen.length}/8.</p>
        <input className="numin" style={{ width: 240, textAlign: 'left' }} value={query} placeholder={`Filter ${cfg.entityPlural}`} onChange={e => setQuery(e.target.value)} />
        <div className="item-picker">{visibleItems.map(item => <label key={item}><input type="checkbox" checked={chosen.includes(item)} onChange={() => toggleItem(item)} /> {item}</label>)}{items.length > visibleItems.length && <small>Showing first 120 matches; refine the filter.</small>}</div>
      </>}
      {msg && <p className="sub" style={{ color: '#a92917', marginTop: 10 }}>{msg}</p>}
      <div className="wnav"><button className="srcbtn" onClick={() => setStep(3)} disabled={busy}>Back</button><button className="act" style={{ marginLeft: 'auto' }} onClick={prepare} disabled={busy || chosen.length === 0}>Prepare selection & continue</button></div>
    </Card>}

    {step === 5 && prepared && <Card t="6. Inventory assumptions" s={cfg.inventoryNote}>
      <div className="stat"><div><b>{fmt(prepared.detected.rows)}</b><span>daily rows</span></div><div><b>{prepared.detected.skus}</b><span>SKUs</span></div><div><b>{prepared.detected.date_min} → {prepared.detected.date_max}</b><span>range</span></div></div>
      <div className="scroll"><table className="tbl"><thead><tr><th>{cfg.entitySingular}</th><th className="num">Current stock</th><th className="num">On order</th><th className="num">Lead time (days)</th><th className="num">Unit cost</th><th className="num">Service level</th></tr></thead>
        <tbody>{prepared.detected.products.map(p => <tr key={p.id}><td><b>{p.id}</b></td>
          <td className="num"><input className="numin" type="number" min="0" value={inv[p.id]?.stock ?? 0} onChange={e => setRow(p.id, 'stock', e.target.value)} /></td>
          <td className="num"><input className="numin" type="number" min="0" value={inv[p.id]?.on_order ?? 0} onChange={e => setRow(p.id, 'on_order', e.target.value)} /></td>
          <td className="num"><input className="numin" type="number" min="1" value={inv[p.id]?.lead ?? 3} onChange={e => setRow(p.id, 'lead', e.target.value)} /></td>
          <td className="num"><input className="numin" type="number" min="0" step="0.01" value={inv[p.id]?.unit_cost ?? ''} onChange={e => setRow(p.id, 'unit_cost', e.target.value)} /></td>
          <td className="num"><select className="numin" value={inv[p.id]?.service_level ?? .95} onChange={e => setRow(p.id, 'service_level', e.target.value)}><option value={.9}>90%</option><option value={.95}>95%</option><option value={.99}>99%</option></select></td>
        </tr>)}</tbody></table></div>
    </Card>}

    {step === 6 && prepared && <Card t="7. Start forecasting" s="Each SKU is evaluated against all four models; the best is refit and forecast 28 days.">
      <div className="stat"><div><b>{prepared.detected.skus}</b><span>SKUs</span></div><div><b>{fmt(prepared.detected.rows)}</b><span>daily rows</span></div><div><b>{prepared.detected.date_min} → {prepared.detected.date_max}</b><span>range</span></div></div>
      {msg && <p className="sub" style={{ color: '#a92917' }}>{msg}</p>}
      <button className="act" onClick={start} disabled={busy}>{busy ? 'Starting…' : 'Start forecasting'}</button>
    </Card>}

    {step >= 1 && step <= 3 && <WizardNav onBack={step === 1 ? () => setStep(0) : () => setStep(step - 1)} onNext={() => setStep(step + 1)} busy={busy} />}
    {step === 5 && <WizardNav onBack={() => setStep(4)} onNext={() => setStep(6)} busy={busy} />}
    {step === 6 && <WizardNav onBack={() => setStep(5)} busy={busy} />}
  </>
}
