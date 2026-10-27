export const WIZARD_STEPS = [
  'Upload CSV',
  'Understand data',
  'Column mapping',
  'Data quality',
  'Store / SKU / date',
  'Inventory',
  'Start forecasting',
]

export function Stepper({ current }) {
  return (
    <ol className="steps">
      {WIZARD_STEPS.map((label, i) => (
        <li key={label} className={'step ' + (i < current ? 'done' : i === current ? 'on' : '')}>
          <span className="dot">{i < current ? '\u2713' : i + 1}</span>
          <span className="slbl">{label}</span>
        </li>
      ))}
    </ol>
  )
}

export function WizardNav({ onBack, onNext, backLabel = 'Back', nextLabel = 'Next', nextDisabled, busy }) {
  return (
    <div className="wnav">
      {onBack && <button className="srcbtn" onClick={onBack} disabled={busy}>{backLabel}</button>}
      {onNext && <button className="act" style={{ marginLeft: 'auto' }} onClick={onNext} disabled={nextDisabled || busy}>{nextLabel}</button>}
    </div>
  )
}
