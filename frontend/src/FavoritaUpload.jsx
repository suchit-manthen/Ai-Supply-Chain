import { useMemo, useState } from 'react'
import { fmt } from './ui.jsx'

const Card = ({ t, s, children }) => <section className="card"><h3>{t}</h3>{s && <p className="sub">{s}</p>}{children}</section>

export default function FavoritaUpload({ onStarted, onCancel, onBack, initialUpload = null }) {
  const [fileInfo, setFileInfo] = useState(initialUpload)
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
    setBusy(true); setMsg(null); setFileInfo(null); setPrepared(null)
    try {
      const fd = new FormData(); fd.append('file', file)
      const response = await fetch('/api/favorita/upload', { method: 'POST', body: fd })
      const result = await response.json()
      if (!result.ok) throw new Error(result.error || 'Could not accept this CSV.')
      setFileInfo(result)
      setDates({ start: result.dataset.date_min || '', end: result.dataset.date_max || '' })
      setStore(''); setItems([]); setChosen([])
    } catch (error) { setMsg(error.message || 'Upload failed. Is the backend running?') }
    setBusy(false)
  }

  async function selectStore(value) {
    setStore(value); setChosen([]); setItems([]); setMsg(null)
    if (!value || !fileInfo) return
    setBusy(true)
    try {
      const result = await (await fetch(`/api/favorita/datasets/${fileInfo.dataset_id}?store_nbr=${encodeURIComponent(value)}`)).json()
      if (!result.ok) throw new Error(result.error || 'Could not load store items.')
      setItems(result.dataset.store_items || [])
    } catch (error) { setMsg(error.message) }
    setBusy(false)
  }

  function toggleItem(item) {
    setChosen(current => current.includes(item) ? current.filter(x => x !== item) : current.length < 8 ? [...current, item] : current)
  }

  async function prepare() {
    setBusy(true); setMsg(null)
    try {
      const response = await fetch(`/api/favorita/datasets/${fileInfo.dataset_id}/prepare`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ store_nbr: store, item_nbrs: chosen, start_date: dates.start, end_date: dates.end }),
      })
      const result = await response.json()
      if (!result.ok) throw new Error(result.error || 'Could not prepare this selection.')
      setPrepared(result)
      const next = {}
      result.detected.products.forEach(p => { next[p.id] = { stock: p.suggested_stock, lead: 3, service_level: 0.95, on_order: 0, unit_cost: '' } })
      setInv(next)
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

  return <>
    <div className="head"><div><h1>Favorita dataset</h1><p>Upload the large Kaggle train.csv, then prepare a small store and item subset for the existing forecasting workflow.</p></div></div>
    <div className="srcbtns" style={{ marginBottom: 16 }}><button className="srcbtn" onClick={onBack}>Use StockSense CSV</button><button className="srcbtn" onClick={onCancel}>Cancel</button></div>

    <Card t="1. Upload Favorita train.csv" s="Required columns: date, store_nbr, item_nbr, unit_sales. Optional: onpromotion. The file is scanned in chunks; it is not loaded as one large table.">
      <input type="file" accept=".csv,text/csv" onChange={upload} disabled={busy} />
      {busy && <p className="sub" style={{ marginTop: 8 }}>Processing selected data…</p>}
      {msg && <p className="sub" style={{ color: '#a92917', marginTop: 8 }}>{msg}</p>}
      {fileInfo && <p className="sub" style={{ color: '#1d6238', marginTop: 8 }}><b>CSV accepted.</b> File size {(fileInfo.size_bytes / (1024 * 1024)).toFixed(1)} MB.</p>}
    </Card>

    {metadata && <Card t="2. Dataset overview" s="Statistics are computed while reading the file in chunks.">
      <div className="stat"><div><b>{fmt(metadata.rows)}</b><span>rows</span></div><div><b>{metadata.date_min || '—'} → {metadata.date_max || '—'}</b><span>date range</span></div><div><b>{fmt(metadata.stores)}</b><span>stores</span></div><div><b>{fmt(metadata.skus)}</b><span>distinct items / SKUs</span></div></div>
      <p className="sub"><b>Available columns:</b> {metadata.available_columns.join(', ')}</p>
      {metadata.warnings.length > 0 && <ul className="issues warn">{metadata.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul>}
      <p className="sub">Quality counts: invalid dates {metadata.quality.invalid_dates}; invalid sales {metadata.quality.invalid_sales}; missing sales {metadata.quality.missing_sales}; negative unit-sales rows {metadata.quality.negative_unit_sales_rows}; missing store/item ids {metadata.quality.missing_store_or_item}; missing promotion flags {metadata.quality.missing_promotion}.</p>
    </Card>}

    {metadata && <Card t="3. Select a manageable subset" s="Choose one store, up to eight items, and a date range. Preparation aggregates duplicate date/store/item rows by summing unit_sales and combines promotion flags with logical OR.">
      <div className="scenario-inputs">
        <label>Store<select className="numin" value={store} onChange={e => selectStore(e.target.value)}><option value="">Choose store</option>{metadata.store_options.map(s => <option key={s} value={s}>{s}</option>)}</select></label>
        <label>Start date<input className="numin" type="date" min={metadata.date_min || undefined} max={metadata.date_max || undefined} value={dates.start} onChange={e => setDates(d => ({ ...d, start: e.target.value }))} /></label>
        <label>End date<input className="numin" type="date" min={metadata.date_min || undefined} max={metadata.date_max || undefined} value={dates.end} onChange={e => setDates(d => ({ ...d, end: e.target.value }))} /></label>
      </div>
      {store && <>
        <p className="sub" style={{ marginTop: 12 }}>{items.length} items found for store {store}. Selected {chosen.length}/8.</p>
        <input className="numin" style={{ width: 240, textAlign: 'left' }} value={query} placeholder="Filter item ids" onChange={e => setQuery(e.target.value)} />
        <div className="item-picker">{visibleItems.map(item => <label key={item}><input type="checkbox" checked={chosen.includes(item)} onChange={() => toggleItem(item)} /> {item}</label>)}{items.length > visibleItems.length && <small>Showing first 120 matches; refine the filter to find other item ids.</small>}</div>
        <button className="act" onClick={prepare} disabled={busy || chosen.length === 0}>Prepare selected items</button>
      </>}
    </Card>}

    {prepared && <>
      <Card t="4. Prepared dataset" s="This is the normalized, selected subset passed to the existing pipeline. Missing daily observations are not filled with zero; items with gaps are rejected so demand is not fabricated.">
        <div className="stat"><div><b>{fmt(prepared.detected.rows)}</b><span>aggregated daily rows</span></div><div><b>{prepared.detected.store_nbr}</b><span>store</span></div><div><b>{prepared.detected.skus}</b><span>selected SKUs</span></div><div><b>{prepared.detected.date_min} → {prepared.detected.date_max}</b><span>prepared date range</span></div><div><b>{prepared.detected.duplicate_rows_aggregated}</b><span>duplicate rows combined</span></div></div>
        <p className="sub"><b>Adapter quality:</b> {JSON.stringify(prepared.detected.quality)}</p>
        <p className="sub"><b>Prepared columns:</b> {prepared.detected.columns.join(', ')}</p>
        {prepared.detected.warnings.map((warning, i) => <p className="sub" key={i}>{warning}</p>)}
      </Card>
      <Card t="5. Inventory configuration" s="Favorita train.csv does not include on-hand stock, lead time or cost. Suggested stock is only a planning default based on recent observed sales; review these values before training.">
        <div className="scroll"><table className="tbl"><thead><tr><th>Item</th><th className="num">Current stock</th><th className="num">On order</th><th className="num">Lead time (days)</th><th className="num">Unit cost</th><th className="num">Service level</th></tr></thead>
          <tbody>{prepared.detected.products.map(p => <tr key={p.id}><td><b>{p.id}</b></td>
            <td className="num"><input className="numin" type="number" min="0" value={inv[p.id]?.stock ?? 0} onChange={e => setRow(p.id, 'stock', e.target.value)} /></td>
            <td className="num"><input className="numin" type="number" min="0" value={inv[p.id]?.on_order ?? 0} onChange={e => setRow(p.id, 'on_order', e.target.value)} /></td>
            <td className="num"><input className="numin" type="number" min="1" value={inv[p.id]?.lead ?? 3} onChange={e => setRow(p.id, 'lead', e.target.value)} /></td>
            <td className="num"><input className="numin" type="number" min="0" step="0.01" value={inv[p.id]?.unit_cost ?? ''} onChange={e => setRow(p.id, 'unit_cost', e.target.value)} /></td>
            <td className="num"><select className="numin" value={inv[p.id]?.service_level ?? .95} onChange={e => setRow(p.id, 'service_level', e.target.value)}><option value={.90}>90%</option><option value={.95}>95%</option><option value={.99}>99%</option></select></td>
          </tr>)}</tbody></table></div>
        {msg && <p className="sub" style={{ color: '#a92917' }}>{msg}</p>}
        <button className="act" style={{ marginTop: 12 }} onClick={start} disabled={busy}>Start forecasting selected subset</button>
      </Card>
    </>}
  </>
}
