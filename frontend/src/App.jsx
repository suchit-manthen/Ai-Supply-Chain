import { useCallback, useEffect, useRef, useState } from 'react'
import Owner from './Owner.jsx'
import Analyst from './Analyst.jsx'
import Upload from './Upload.jsx'

const when = iso => (iso ? new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : null)

export default function App() {
  const [tab, setTab] = useState('owner')
  const [mode, setMode] = useState('dashboard')
  const [state, setState] = useState({ status: 'loading', progress: 0, data_source: 'synthetic' })
  const [data, setData] = useState({})
  const stopRef = useRef(false)
  const timerRef = useRef(null)

  const poll = useCallback(async function run() {
    try {
      const s = await (await fetch('/api/status')).json()
      if (stopRef.current) return
      if (s.status === 'ready') {
        const [owner, analyst] = await Promise.all([fetch('/api/owner').then(r => r.json()), fetch('/api/analyst').then(r => r.json())])
        setData({ owner, analyst }); setState({ status: 'ready', run_id: s.run_id, trained_at: s.trained_at, data_source: s.data_source })
      } else if (s.status === 'error') {
        setState({ status: 'error', error: s.error, data_source: s.data_source })
      } else if (s.status === 'untrained') {
        setState({ status: 'untrained', data_source: s.data_source })
      } else {
        setState(s); timerRef.current = setTimeout(run, 1200)
      }
    } catch {
      setState({ status: 'offline' }); timerRef.current = setTimeout(run, 2500)
    }
  }, [])

  useEffect(() => {
    stopRef.current = false
    poll()
    return () => { stopRef.current = true; clearTimeout(timerRef.current) }
  }, [poll])

  const beginTraining = (body) => {
    setMode('dashboard')
    setState({ status: 'training', progress: 0, retraining: true, data_source: state.data_source })
    return fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(() => poll())
      .catch(() => setState({ status: 'offline' }))
  }

  const showSynthetic = () => {
    if (state.status === 'ready' && state.data_source === 'synthetic') { setMode('dashboard'); return }
    beginTraining({ source: 'synthetic' })
  }
  const retrain = () => {
    setMode('dashboard')
    setState({ status: 'training', progress: 0, retraining: true, data_source: state.data_source })
    fetch('/api/retrain', { method: 'POST' }).then(() => poll()).catch(() => setState({ status: 'offline' }))
  }

  const sourceLabel = state.data_source === 'favorita' ? 'Favorita subset' : state.data_source === 'uploaded' ? 'Uploaded supermarket dataset' : 'Synthetic demo dataset'
  const training = state.status === 'training'

  return (
    <>
      <header className="top">
        <div className="brand">StockSense<small>AI supply chain demand forecasting</small></div>
        <nav className="tabs">
          <button className={'tab ' + (tab === 'owner' ? 'on' : '')} onClick={() => { setTab('owner'); setMode('dashboard') }}>Store owner</button>
          <button className={'tab ' + (tab === 'analyst' ? 'on' : '')} onClick={() => { setTab('analyst'); setMode('dashboard') }}>ML analyst</button>
        </nav>
        <div className="topmeta">
          <span className="badge">{state.data_source === 'favorita' ? 'Favorita data' : state.data_source === 'uploaded' ? 'Uploaded data' : 'Synthetic demo data'}</span>
          {state.status === 'ready' && when(state.trained_at) && <span className="updated">Updated {when(state.trained_at)}</span>}
          {state.status === 'ready' && <button className="retrain" onClick={retrain} disabled={training}>Retrain</button>}
        </div>
      </header>
      <main>
        <div className="sourcebar">
          <span className="src">Data source: <b>{sourceLabel}</b></span>
          <div className="srcbtns">
          <button className={'srcbtn ' + (mode === 'dashboard' && state.data_source === 'synthetic' ? 'on' : '')} onClick={showSynthetic} disabled={training}>Synthetic Demo</button>
            <button className={'srcbtn ' + (mode === 'upload' ? 'on' : '')} onClick={() => setMode('upload')} disabled={training}>Upload CSV</button>
          </div>
        </div>

        {mode === 'upload' ? (
          <Upload onStarted={() => { setMode('dashboard'); setState({ status: 'training', progress: 0, retraining: true, data_source: 'uploaded' }); poll() }} onCancel={() => setMode('dashboard')} />
        ) : state.status === 'ready' ? (
          <>
            {state.data_source === 'favorita'
              ? <div className="banner" style={{ background: '#e6f4ea', color: '#1d6238', borderColor: '#bfe3cb' }}><b>Favorita subset.</b> Forecast from the selected store, items and dates. The source does not contain unit prices, weather or stock levels; revenue uses a neutral placeholder price, and inventory values are user configured.</div>
              : state.data_source === 'uploaded'
              ? <div className="banner" style={{ background: '#e6f4ea', color: '#1d6238', borderColor: '#bfe3cb' }}><b>Uploaded dataset.</b> Forecasting from your uploaded supermarket sales CSV. Future external inputs follow the assumptions shown in upload warnings.</div>
              : <div className="banner"><b>Synthetic demo data — not real store results.</b> Sales are generated by a simulator with weekly and annual seasonality, holidays, promotions, weather and growth. Use this mode to demonstrate the workflow only.</div>}
            {tab === 'owner'
              ? <Owner d={data.owner} />
              : <Analyst d={data.analyst} run={{ run_id: state.run_id, trained_at: state.trained_at }} />}
          </>
        ) : (
          <div className="load">
            {state.status === 'untrained' ? (
              <>
                <h2>No trained run yet</h2>
                <p className="sub">The API serves the latest saved training run and does not retrain on startup. Start a run to evaluate the four models for each SKU and generate forecasts.</p>
                <button className="act" onClick={() => beginTraining({ source: 'synthetic' })}>Train models</button>
              </>
            ) : (
              <>
                <h2>{state.status === 'offline' ? 'Backend not reachable' : state.status === 'error' ? 'Training failed' : state.retraining ? 'Training models…' : 'Training forecasting models…'}</h2>
                <p className="sub">{state.status === 'offline' ? 'Start the Flask server: cd backend && python app.py' : state.error || 'ARIMA, Prophet, XGBoost and Random Forest are being fitted and validated.'}</p>
                <div className="bar"><i style={{ width: (state.progress || 3) + '%' }} /></div>
              </>
            )}
          </div>
        )}
      </main>
    </>
  )
}
