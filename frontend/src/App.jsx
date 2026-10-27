import { useCallback, useEffect, useRef, useState } from 'react'
import Owner from './Owner.jsx'
import Analyst from './Analyst.jsx'
import Upload from './Upload.jsx'
import Simulator from './Simulator.jsx'
import Login, { clearSession, createSession, readSession, saveSession } from './Login.jsx'
import { BrandMark, Icon } from './ui.jsx'

const when = iso => (iso ? new Date(iso).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : null)

const NAV_ITEMS = {
  owner: [
    { id: 'owner', label: 'Store overview', icon: 'store' },
    { id: 'simulator', label: 'Simulator', icon: 'sliders' },
  ],
  analyst: [
    { id: 'analyst', label: 'Model workbench', icon: 'chart' },
    { id: 'simulator', label: 'Simulator', icon: 'sliders' },
  ],
}

const sourceLabel = source => source === 'favorita' ? 'Favorita subset'
  : source === 'store_family' ? 'Store + family subset'
    : source === 'generic' ? 'Mapped CSV dataset'
      : source === 'uploaded' ? 'Uploaded supermarket dataset'
        : 'Synthetic demo dataset'

function LoginScreen({ onSignIn }) {
  return <Login onSignIn={account => onSignIn(createSession(account))} />
}

export default function App() {
  const [user, setUser] = useState(readSession)

  const signIn = session => {
    saveSession(session)
    setUser(session)
  }

  const signOut = () => {
    clearSession()
    setUser(null)
  }

  if (!user) return <LoginScreen onSignIn={signIn} />
  return <Workspace key={user.email} user={user} onSignOut={signOut} />
}

function Workspace({ user, onSignOut }) {
  const defaultTab = user.role === 'analyst' ? 'analyst' : 'owner'
  const [tab, setTab] = useState(defaultTab)
  const [mode, setMode] = useState('dashboard')
  const [state, setState] = useState({ status: 'loading', progress: 0, data_source: 'synthetic' })
  const [data, setData] = useState({})
  const stopRef = useRef(false)
  const timerRef = useRef(null)
  const navItems = NAV_ITEMS[user.role] || NAV_ITEMS.owner

  const poll = useCallback(async function run() {
    try {
      const s = await (await fetch('/api/status')).json()
      if (stopRef.current) return
      if (s.status === 'ready') {
        const [owner, analyst] = await Promise.all([fetch('/api/owner').then(r => r.json()), fetch('/api/analyst').then(r => r.json())])
        setData({ owner, analyst }); setState({ status: 'ready', run_id: s.run_id, trained_at: s.trained_at, duration_s: s.duration_s, data_source: s.data_source })
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

  const beginTraining = body => {
    setMode('dashboard')
    setState({ status: 'training', progress: 0, retraining: true, data_source: state.data_source })
    return fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(() => poll())
      .catch(() => setState({ status: 'offline' }))
  }

  const showSynthetic = () => {
    setTab(defaultTab)
    if (state.status === 'ready' && state.data_source === 'synthetic') { setMode('dashboard'); return }
    beginTraining({ source: 'synthetic' })
  }
  const retrain = () => {
    setMode('dashboard')
    setState({ status: 'training', progress: 0, retraining: true, data_source: state.data_source })
    fetch('/api/retrain', { method: 'POST' }).then(() => poll()).catch(() => setState({ status: 'offline' }))
  }

  const training = state.status === 'training'
  const activeSource = state.data_source || 'synthetic'
  const banner = activeSource === 'favorita'
    ? { tone: 'data', title: 'Favorita subset.', text: 'Forecast from the selected store, items and dates. The source does not contain unit prices, weather or stock levels; revenue uses a neutral placeholder price, and inventory values are user configured.' }
    : activeSource === 'store_family'
      ? { tone: 'data', title: 'Store + family subset.', text: 'Forecast from the selected store, product families and dates. This source has no unit prices, weather or stock levels; revenue uses a neutral placeholder price, and inventory values are user configured.' }
      : activeSource === 'generic'
        ? { tone: 'data', title: 'Mapped CSV.', text: 'Forecast from the columns you mapped. Unmapped optional fields use neutral placeholder values, and inventory values are user configured.' }
        : activeSource === 'uploaded'
          ? { tone: 'data', title: 'Uploaded dataset.', text: 'Forecasting from your uploaded supermarket sales CSV. Future external inputs follow the assumptions shown in upload warnings.' }
          : { tone: 'demo', title: 'Synthetic demo data — not real store results.', text: 'Sales are generated by a simulator with weekly and annual seasonality, holidays, promotions, weather and growth. Use this mode to demonstrate the workflow only.' }

  return (
    <div className="app-shell">
      <header className="appbar">
        <div className="appbar-inner">
          <button type="button" className="brand" onClick={() => { setTab(defaultTab); setMode('dashboard') }} aria-label="Go to workspace home">
            <BrandMark size={40} />
            <span className="brand-copy"><b>StockSense</b><small>AI supply chain forecasting</small></span>
          </button>

          <nav className="tabs" aria-label="Primary navigation">
            {navItems.map(item => <button key={item.id} className={'tab ' + (mode === 'dashboard' && tab === item.id ? 'on' : '')}
              onClick={() => { setTab(item.id); setMode('dashboard') }}>
              <Icon name={item.icon} size={16} /><span>{item.label}</span>
            </button>)}
          </nav>

          <div className="appbar-actions">
            <div className="topmeta">
              {user.role === 'analyst' && state.status === 'ready' && <button className="retrain" onClick={retrain} disabled={training}><Icon name="refresh" size={14} /> Retrain</button>}
            </div>
            <div className="account">
              <span className={`avatar ${user.role}`}>{user.initials || user.name?.slice(0, 2)?.toUpperCase() || 'SS'}</span>
              <span className="account-copy"><b>{user.name}</b><small>{user.roleLabel}</small></span>
              <button type="button" className="signout" onClick={onSignOut} title="Sign out"><Icon name="logout" size={16} /><span>Sign out</span></button>
            </div>
          </div>
        </div>
      </header>

      <main className="app-main">
        <section className="sourcebar" aria-label="Data source">
          <div className="context-id">
            <span className={'context-dot ' + (state.status === 'ready' ? 'live' : '')} />
            <div className="context-copy"><span className="context-label">Active dataset</span><b>{sourceLabel(activeSource)}</b></div>
            {state.status === 'ready' && when(state.trained_at) && <span className="context-run">Trained {when(state.trained_at)}</span>}
          </div>
          <div className="srcbtns">
            <button className={'srcbtn ' + (mode === 'dashboard' && activeSource === 'synthetic' ? 'on' : '')} onClick={showSynthetic} disabled={training}><Icon name="sparkle" size={15} /> Synthetic demo</button>
            <button className={'srcbtn ' + (mode === 'upload' ? 'on' : '')} onClick={() => setMode('upload')} disabled={training}><Icon name="upload" size={15} /> Upload CSV</button>
          </div>
        </section>

        {mode === 'upload' ? (
          <Upload onStarted={() => { setTab(defaultTab); setMode('dashboard'); setState({ status: 'training', progress: 0, retraining: true, data_source: 'uploaded' }); poll() }} onCancel={() => setMode('dashboard')} />
        ) : tab === 'simulator' ? (
          <Simulator />
        ) : state.status === 'ready' ? (
          <>
            <div className={'banner banner-' + banner.tone}><Icon name={banner.tone === 'demo' ? 'sparkle' : 'database'} size={17} /><div><b>{banner.title}</b> {banner.text}</div></div>
            {tab === 'owner'
              ? <Owner d={data.owner} />
              : <Analyst d={data.analyst} run={{ run_id: state.run_id, trained_at: state.trained_at }} />}
          </>
        ) : (
          <div className="load">
            {state.status === 'untrained' ? (
              <>
                <span className="load-icon"><Icon name="chart" size={24} /></span>
                <h2>No trained run yet</h2>
                <p className="sub">The API serves the latest saved training run and does not retrain on startup. Start a run to evaluate the four models for each SKU and generate forecasts.</p>
                <button className="act" onClick={() => beginTraining({ source: 'synthetic' })}>Train models</button>
              </>
            ) : (
              <>
                <span className="load-icon"><Icon name={state.status === 'offline' ? 'database' : 'gauge'} size={24} /></span>
                <h2>{state.status === 'offline' ? 'Backend not reachable' : state.status === 'error' ? 'Training failed' : state.retraining ? 'Training models…' : 'Training forecasting models…'}</h2>
                <p className="sub">{state.status === 'offline' ? 'Start the Flask server: cd backend && python app.py' : state.error || 'ARIMA, Prophet, XGBoost and Random Forest are being fitted and validated.'}</p>
                <div className="bar"><i style={{ width: (state.progress || 3) + '%' }} /></div>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
