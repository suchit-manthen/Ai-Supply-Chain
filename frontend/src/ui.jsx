export const fmt = n => Number(n).toLocaleString('en-US')
export const usd = n => '$' + fmt(Math.round(n))
export const short = d => new Date(d + 'T00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
export const STATUS_COLOR = { Critical: '#d43f2b', Reorder: '#d99a12', Overstock: '#3b5bdb', Healthy: '#2c7a4b' }
export const Kpi = ({ v, l, c }) => <div className="kpi" style={{ borderLeftColor: c }}><b>{v}</b><span>{l}</span></div>
export const Pill = ({ s }) => <span className={'pill ' + s}>{s === 'Critical' ? 'Shortage risk' : s === 'Reorder' ? 'Reorder now' : s}</span>
export const RISK_STYLE = { High: { bg: '#fde4df', fg: '#a92917' }, Elevated: { bg: '#fdf0cf', fg: '#8a5f00' }, Low: { bg: '#dff2e6', fg: '#1d6238' } }
export const RiskPill = ({ r }) => <span className="pill" style={{ background: (RISK_STYLE[r] || RISK_STYLE.Low).bg, color: (RISK_STYLE[r] || RISK_STYLE.Low).fg }}>{r}</span>

const ICONS = {
  store: <>
    <path d="M4 10v10h16V10" />
    <path d="M3 10l2-6h14l2 6" />
    <path d="M3 10a3 3 0 0 0 6 0 3 3 0 0 0 6 0 3 3 0 0 0 6 0" />
    <path d="M9 20v-6h6v6" />
  </>,
  chart: <>
    <path d="M4 20V10" />
    <path d="M10 20V4" />
    <path d="M16 20v-7" />
    <path d="M22 20H2" />
  </>,
  sliders: <>
    <path d="M4 6h16" />
    <path d="M4 12h16" />
    <path d="M4 18h16" />
    <circle cx="9" cy="6" r="2" />
    <circle cx="15" cy="12" r="2" />
    <circle cx="7" cy="18" r="2" />
  </>,
  upload: <>
    <path d="M12 16V4" />
    <path d="m7 9 5-5 5 5" />
    <path d="M4 20h16" />
  </>,
  mail: <>
    <rect x="3" y="5" width="18" height="14" rx="2" />
    <path d="m3 7 9 6 9-6" />
  </>,
  lock: <>
    <rect x="4" y="10" width="16" height="10" rx="2" />
    <path d="M8 10V7a4 4 0 0 1 8 0v3" />
  </>,
  shield: <>
    <path d="M12 3 5 6v5c0 4.5 3 8 7 10 4-2 7-5.5 7-10V6l-7-3Z" />
    <path d="m9 12 2 2 4-4" />
  </>,
  check: <path d="m5 12 4 4L19 6" />,
  arrow: <>
    <path d="M5 12h14" />
    <path d="m13 6 6 6-6 6" />
  </>,
  refresh: <>
    <path d="M20 7v5h-5" />
    <path d="M20 12a8 8 0 1 0-2.3 6.7" />
  </>,
  logout: <>
    <path d="M10 17l5-5-5-5" />
    <path d="M15 12H3" />
    <path d="M15 4h4a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-4" />
  </>,
  sparkle: <>
    <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" />
    <path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" />
  </>,
  database: <>
    <ellipse cx="12" cy="5" rx="8" ry="3" />
    <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
    <path d="M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
  </>,
  layers: <>
    <path d="m12 3 9 5-9 5-9-5 9-5Z" />
    <path d="m3 12 9 5 9-5" />
    <path d="m3 16 9 5 9-5" />
  </>,
  gauge: <>
    <path d="M4 15a8 8 0 1 1 16 0" />
    <path d="m12 15 4-4" />
    <path d="M4 19h16" />
  </>,
}

export function Icon({ name, size = 18, className = '' }) {
  return <svg className={className} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{ICONS[name] || ICONS.chart}</svg>
}

export function BrandMark({ size = 38 }) {
  return <span className="brand-mark" style={{ width: size, height: size }} aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round">
      <path d="m3 16 4.5-5 4 3L18 6" />
      <path d="M14 6h4v4" />
    </svg>
  </span>
}
