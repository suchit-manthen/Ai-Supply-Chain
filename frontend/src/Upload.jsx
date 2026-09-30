import { useState } from 'react'
import { fmt } from './ui.jsx'
import FavoritaUpload from './FavoritaUpload.jsx'

const Card = ({ t, s, children }) => <section className="card"><h3>{t}</h3>{s && <p className="sub">{s}</p>}{children}</section>
const OPTIONAL_LABELS = { category: 'category', price: 'price', discount: 'discount', temp: 'temperature', holiday_lift: 'holiday lift' }

export default function Upload({ onStarted, onCancel }) {
  const [format, setFormat] = useState('stocksense')
  const [report, setReport] = useState(null)
  const [inv, setInv] = useState({})
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [favoritaUpload, setFavoritaUpload] = useState(null)

  async function onFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setBusy(true); setMsg(null); setReport(null)
    try {
      const fd = new FormData(); fd.append('file', f)
      const r = await (await fetch('/api/upload', { method: 'POST', body: fd })).json()
      if (r.adapter === 'favorita') {
        setFavoritaUpload(r)
        setFormat('favorita')
        setBusy(false)
        return
      }
      setReport(r)
      if (r.ok) {
        const init = {}
        r.detected.products.forEach(p => { init[p.id] = { stock: p.suggested_stock, lead: 3, service_level: 0.95, on_order: 0, unit_cost: p.price ? (p.price * 0.7).toFixed(2) : '' } })
        setInv(init)
      }
    } catch {
      setMsg('Upload failed. Is the backend running?')
    }
    setBusy(false)
  }

  const setRow = (id, key, value) => setInv(v => ({ ...v, [id]: { ...v[id], [key]: value } }))

  async function start() {
    setBusy(true); setMsg(null)
    try {
      const r = await (await fetch('/api/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: 'uploaded', dataset_id: report.dataset_id, inventory: inv }),
      })).json()
      if (!r.ok) { setMsg(r.error || 'Could not start training.'); setBusy(false); return }
      onStarted()
    } catch {
      setMsg('Could not start training.'); setBusy(false)
    }
  }

  const detected = report?.detected
  const missingOptional = report ? Object.entries(report.optional).filter(([, v]) => !v).map(([k]) => OPTIONAL_LABELS[k]) : []

  if (format === 'favorita') return <FavoritaUpload initialUpload={favoritaUpload} onStarted={onStarted} onCancel={onCancel} onBack={() => { setFavoritaUpload(null); setFormat('stocksense') }} />

  return (
    <>
      <div className="srcbtns" style={{ marginBottom: 16 }}>
        <button className="srcbtn on" type="button">StockSense CSV</button>
        <button className="srcbtn" type="button" onClick={() => setFormat('favorita')}>Favorita train.csv</button>
      </div>
      <div className="head">
        <div><h1>Upload your supermarket data</h1><p>Upload a CSV of historical sales — the system validates it, selects a model per SKU and builds the dashboard.</p></div>
      </div>

      <Card t="1. Choose a CSV file" s="Required columns: date, product, sales. Optional: category, price, discount, temp, holiday_lift. Headers are matched case-insensitively; use the exact sample format below for easiest import.">
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <input type="file" accept=".csv,text/csv" onChange={onFile} disabled={busy} />
          <a className="srcbtn" href="/api/sample-csv" download>Download sample CSV</a>
          <button className="srcbtn" onClick={onCancel} disabled={busy}>Cancel</button>
        </div>
        {busy && <p className="sub" style={{ marginTop: 10 }}>Processing…</p>}
        {msg && <p className="sub" style={{ color: '#a92917', marginTop: 10 }}>{msg}</p>}
      </Card>

      {report && !report.ok && (
        <Card t="Validation problems" s="Fix these in your CSV and upload again.">
          <ul className="issues">{report.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>
        </Card>
      )}

      {report && report.ok && (
        <>
          <Card t="2. Detected data" s="Review what was found before forecasting. Rows are normalized to date, product, sales, discount, price, temp and holiday_lift.">
            <div className="stat">
              <div><b>{fmt(detected.rows)}</b><span>rows</span></div>
              <div><b>{detected.skus}</b><span>SKUs</span></div>
              <div><b>{detected.date_min}</b><span>first date</span></div>
              <div><b>{detected.date_max}</b><span>last date</span></div>
            </div>
            <p className="sub">Optional fields {missingOptional.length
              ? <>missing — defaults applied: <b>{missingOptional.join(', ')}</b></>
              : <><b>all provided</b></>}.</p>
            {report.warnings?.length > 0 && <ul className="issues warn">{report.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
            <div className="scroll" style={{ marginTop: 10 }}><table className="tbl">
              <thead><tr>{Object.keys(report.preview[0] || {}).map(k => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{report.preview.map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{v}</td>)}</tr>)}</tbody>
            </table></div>
          </Card>

          <Card t="3. Inventory assumptions" s="The CSV has no stock or lead-time info, so set it here. The suggested stock is a clearly-labelled demo default (about one week of recent sales).">
            <div className="scroll"><table className="tbl">
              <thead><tr><th>SKU</th><th className="num">Current stock</th><th className="num">On order</th><th className="num">Lead time (days)</th><th className="num">Unit cost</th><th className="num">Service level</th></tr></thead>
              <tbody>{detected.products.map(p => (
                <tr key={p.id}>
                  <td><b>{p.id}</b>{p.category && p.category !== 'Unknown' ? <><br /><small style={{ color: '#5f6f66' }}>{p.category}</small></> : null}</td>
                  <td className="num"><input className="numin" type="number" min="0" value={inv[p.id]?.stock ?? 0} onChange={e => setRow(p.id, 'stock', e.target.value)} /></td>
                  <td className="num"><input className="numin" type="number" min="0" value={inv[p.id]?.on_order ?? 0} onChange={e => setRow(p.id, 'on_order', e.target.value)} /></td>
                  <td className="num"><input className="numin" type="number" min="1" value={inv[p.id]?.lead ?? 3} onChange={e => setRow(p.id, 'lead', e.target.value)} /></td>
                  <td className="num"><input className="numin" type="number" min="0.01" step="0.01" value={inv[p.id]?.unit_cost ?? ''} onChange={e => setRow(p.id, 'unit_cost', e.target.value)} /></td>
                  <td className="num">
                    <select className="numin" value={inv[p.id]?.service_level ?? 0.95} onChange={e => setRow(p.id, 'service_level', e.target.value)}>
                      <option value={0.90}>90%</option><option value={0.95}>95%</option><option value={0.99}>99%</option>
                    </select>
                  </td>
                </tr>))}
              </tbody></table></div>
            <div style={{ marginTop: 14 }}><button className="act" onClick={start} disabled={busy}>Start forecasting</button></div>
          </Card>
        </>
      )}
    </>
  )
}
