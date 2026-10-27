import { useState } from 'react'
import GenericUpload from './GenericUpload.jsx'
import FavoritaUpload from './FavoritaUpload.jsx'

export default function Upload({ onStarted, onCancel }) {
  const [adapter, setAdapter] = useState(null)

  if (adapter) {
    return <FavoritaUpload mode={adapter.kind} initialUpload={adapter.result}
      onStarted={onStarted} onCancel={onCancel} onBack={() => setAdapter(null)} />
  }
  return <GenericUpload onStarted={onStarted} onCancel={onCancel}
    onAdapter={r => setAdapter({ kind: r.adapter, result: r })} />
}
