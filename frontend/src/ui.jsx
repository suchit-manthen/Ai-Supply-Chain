export const fmt = n => Number(n).toLocaleString('en-US')
export const usd = n => '$' + fmt(Math.round(n))
export const short = d => new Date(d + 'T00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
export const STATUS_COLOR = { Critical: '#d43f2b', Reorder: '#d99a12', Overstock: '#3b5bdb', Healthy: '#2c7a4b' }
export const Kpi = ({ v, l, c }) => <div className="kpi" style={{ borderLeftColor: c }}><b>{v}</b><span>{l}</span></div>
export const Pill = ({ s }) => <span className={'pill ' + s}>{s === 'Critical' ? 'Shortage risk' : s === 'Reorder' ? 'Reorder now' : s}</span>
export const RISK_STYLE = { High: { bg: '#fde4df', fg: '#a92917' }, Elevated: { bg: '#fdf0cf', fg: '#8a5f00' }, Low: { bg: '#dff2e6', fg: '#1d6238' } }
export const RiskPill = ({ r }) => <span className="pill" style={{ background: (RISK_STYLE[r] || RISK_STYLE.Low).bg, color: (RISK_STYLE[r] || RISK_STYLE.Low).fg }}>{r}</span>
