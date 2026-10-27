import { useState } from 'react'
import { BrandMark, Icon } from './ui.jsx'

export const DEMO_ACCOUNTS = {
  owner: {
    role: 'owner',
    roleLabel: 'Store Owner',
    name: 'Maya Chen',
    initials: 'MC',
    email: 'owner@stocksense.demo',
    password: 'owner123',
    workspace: 'Greenway Market',
    note: 'Inventory, reorder points & supplier scenarios',
  },
  analyst: {
    role: 'analyst',
    roleLabel: 'ML Analyst',
    name: 'Alex Rivera',
    initials: 'AR',
    email: 'analyst@stocksense.demo',
    password: 'analyst123',
    workspace: 'Forecasting Lab',
    note: 'Model selection, validation & run monitoring',
  },
}

const SESSION_KEY = 'stocksense.session.v1'

export function createSession(account) {
  return {
    role: account.role,
    roleLabel: account.roleLabel,
    name: account.name,
    initials: account.initials,
    email: account.email,
    workspace: account.workspace,
    signedInAt: new Date().toISOString(),
  }
}

export function readSession() {
  try {
    const raw = window.localStorage.getItem(SESSION_KEY)
    if (!raw) return null
    const session = JSON.parse(raw)
    if (!session || !DEMO_ACCOUNTS[session.role] || !session.email) return null
    return session
  } catch {
    return null
  }
}

export function saveSession(session) {
  try {
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(session))
  } catch {
    // A private browser session may block storage; the in-memory session still works.
  }
}

export function clearSession() {
  try {
    window.localStorage.removeItem(SESSION_KEY)
  } catch {
    // Ignore storage access failures on sign-out.
  }
}

function RoleOption({ account, selected, onSelect }) {
  return <button type="button" role="radio" aria-checked={selected}
    className={`role-option ${account.role} ${selected ? 'on' : ''}`} onClick={() => onSelect(account.role)}>
    <span className="role-option-top">
      <span className="role-icon"><Icon name={account.role === 'owner' ? 'store' : 'chart'} size={17} /></span>
      <span className="role-check">{selected && <Icon name="check" size={12} />}</span>
    </span>
    <b>{account.roleLabel}</b>
    <small>{account.note}</small>
    <span className="role-creds"><span>{account.email}</span><span>{account.password}</span></span>
  </button>
}

export default function Login({ onSignIn }) {
  const [role, setRole] = useState('owner')
  const [email, setEmail] = useState(DEMO_ACCOUNTS.owner.email)
  const [password, setPassword] = useState(DEMO_ACCOUNTS.owner.password)
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const active = DEMO_ACCOUNTS[role]

  const chooseRole = nextRole => {
    const account = DEMO_ACCOUNTS[nextRole]
    setRole(nextRole)
    setEmail(account.email)
    setPassword(account.password)
    setShowPassword(false)
    setError('')
  }

  const submit = event => {
    event.preventDefault()
    const normalizedEmail = email.trim().toLowerCase()
    if (!normalizedEmail || !password) {
      setError('Enter your email address and password to continue.')
      return
    }
    const account = Object.values(DEMO_ACCOUNTS).find(item => item.email === normalizedEmail && item.password === password)
    if (!account) {
      setError('Those details do not match a demo account. Use the credentials shown on the role cards above.')
      return
    }
    if (account.role !== role) {
      setError(`Those credentials belong to the ${account.roleLabel} workspace. Select ${account.roleLabel} above and try again.`)
      return
    }
    setError('')
    onSignIn(account)
  }

  return <div className="auth-shell">
    <aside className="auth-story">
      <div className="auth-brand"><BrandMark size={44} /><div><b>StockSense</b><small>AI supply chain intelligence</small></div></div>

      <div className="auth-story-copy">
        <span className="auth-eyebrow"><i /> DEMAND PLANNING, SIMPLIFIED</span>
        <h1>Know what’s next.<span>Stock what matters.</span></h1>
        <p>Turn daily sales into a clear 28-day demand plan. StockSense compares four forecasting models, surfaces the right reorder points and lets you stress-test the week ahead.</p>
        <ul className="auth-feature-list">
          <li><span><Icon name="store" size={17} /></span><div><b>One clear store view</b><small>See what is running low, what to order and what can wait.</small></div></li>
          <li><span><Icon name="layers" size={17} /></span><div><b>Model selection that explains itself</b><small>ARIMA, Prophet, XGBoost and Random Forest evaluated per SKU.</small></div></li>
          <li><span><Icon name="gauge" size={17} /></span><div><b>What-if supply scenarios</b><small>Test demand spikes, supplier delays and stock changes before they happen.</small></div></li>
        </ul>
      </div>

      <div className="auth-proof" aria-label="Platform highlights">
        <div><b>28</b><span>day forecast horizon</span></div>
        <div><b>4</b><span>models compared</span></div>
        <div><b>95%</b><span>default service level</span></div>
      </div>
    </aside>

    <section className="auth-panel">
      <div className="auth-card">
        <div className="auth-card-head">
          <span className="auth-secure"><Icon name="shield" size={13} /> Secure demo workspace</span>
          <span className="auth-version">v1.0</span>
        </div>
        <h2>Welcome back</h2>
        <p className="auth-sub">Choose a workspace, then sign in to continue.</p>

        <div className="role-options" role="radiogroup" aria-label="Demo workspace">
          {Object.values(DEMO_ACCOUNTS).map(account => <RoleOption key={account.role} account={account} selected={role === account.role} onSelect={chooseRole} />)}
        </div>

        <form className="auth-form" onSubmit={submit} noValidate>
          <label htmlFor="login-email">Email address
            <span className="auth-input"><Icon name="mail" size={17} /><input id="login-email" type="email" autoComplete="username" value={email} onChange={event => { setEmail(event.target.value); setError('') }} placeholder="you@company.com" /></span>
          </label>
          <label htmlFor="login-password">Password
            <span className="auth-input"><Icon name="lock" size={17} /><input id="login-password" type={showPassword ? 'text' : 'password'} autoComplete="current-password" value={password} onChange={event => { setPassword(event.target.value); setError('') }} placeholder="Enter your password" /><button type="button" className="auth-reveal" onClick={() => setShowPassword(value => !value)} aria-label={showPassword ? 'Hide password' : 'Show password'}>{showPassword ? 'Hide' : 'Show'}</button></span>
          </label>
          {error && <div className="auth-error" role="alert">{error}</div>}
          <button className="auth-submit" type="submit"><span>Sign in as {active.roleLabel}</span><Icon name="arrow" size={18} /></button>
        </form>

        <p className="auth-note"><b>Demo credentials</b> are prefilled when you choose a workspace. This demo keeps your session in this browser only — it is not a production authentication system.</p>
      </div>
      <p className="auth-foot">StockSense · AI supply chain forecasting demo</p>
    </section>
  </div>
}
