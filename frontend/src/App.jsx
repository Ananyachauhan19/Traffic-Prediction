import React from 'react'
import TrafficMap from './components/TrafficMap'

export default function App() {
  const start = [51.505, -0.09]
  const dest = [51.515, -0.1]
  const segments = [
    { id: 's1', coords: [[51.505, -0.09], [51.51, -0.095]] },
    { id: 's2', coords: [[51.51, -0.095], [51.515, -0.1]] }
  ]

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <h1 className="text-2xl font-semibold mb-4">Traffic Prediction Map</h1>
      <TrafficMap start={start} destination={dest} segments={segments} />
    </div>
  )
}
