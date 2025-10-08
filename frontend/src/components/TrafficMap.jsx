import React, { useEffect, useState, useRef } from 'react'
import { MapContainer, TileLayer, Polyline, Marker, Tooltip } from 'react-leaflet'
import L from 'leaflet'
import axios from 'axios'

// Minimal Tailwind classes are used in inline className attributes.

const COLORS = {
  0: 'green',
  1: 'yellow',
  2: 'red'
}

// helper to map color name to leaflet stroke color
const colorForLevel = (lvl) => {
  return lvl === 2 ? '#e11d48' : lvl === 1 ? '#f59e0b' : '#10b981'
}

export default function TrafficMap({ start, destination, segments }) {
  // start, destination: [lat, lng]
  // segments: [{id, coords: [[lat,lng],...]}]

  const [segmentStatus, setSegmentStatus] = useState({})
  const [alternateRoute, setAlternateRoute] = useState(null)
  const intervalRef = useRef(null)

  useEffect(() => {
    // fetch prediction for all segments once and periodicially
    const fetchData = async () => {
      try {
        // For demo: call backend for each segment. Your backend should accept coords or segment id.
        const newStatus = {}
        for (const seg of segments) {
          // Example payload — adapt to your API
          const payload = {
            entry_count: 0,
            exit_count: 0,
            road_width: 5,
            time_of_day: (new Date()).getHours(),
            weather: 0,
            avg_speed: 30
          }
          const res = await axios.post('/api/traffic/predict', payload)
          if (res && res.data) {
            newStatus[seg.id] = { level: parseInt(res.data.status || 0), prob: res.data.probability }
            if (res.data.alternate_route) {
              setAlternateRoute(res.data.alternate_route)
            }
          }
        }
        setSegmentStatus(newStatus)
      } catch (e) {
        console.error('Failed to fetch predictions', e)
      }
    }

    fetchData()
    intervalRef.current = setInterval(fetchData, 30_000)
    return () => clearInterval(intervalRef.current)
  }, [segments])

  const center = start || [0, 0]

  return (
    <div className="w-full h-full p-2">
      <MapContainer center={center} zoom={13} className="h-[600px] w-full rounded-lg shadow">
        <TileLayer
          attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Draw main route segments with color */}
        {segments.map((seg) => {
          const status = segmentStatus[seg.id]
          const lvl = status ? status.level : 0
          return (
            <Polyline
              key={seg.id}
              positions={seg.coords}
              pathOptions={{ color: colorForLevel(lvl), weight: 6 }}
            >
              <Tooltip>{`Segment ${seg.id} - ${lvl === 2 ? 'Heavy' : lvl === 1 ? 'Moderate' : 'Free'}`}</Tooltip>
            </Polyline>
          )
        })}

        {/* Draw alternate route if present (dashed) */}
        {alternateRoute && alternateRoute.features && alternateRoute.features.length > 0 && (
          <Polyline
            positions={alternateRoute.features[0].geometry.coordinates.map(c => [c[1], c[0]])}
            pathOptions={{ color: '#3b82f6', dashArray: '10 10', weight: 4 }}
          />
        )}

        {/* Start/destination markers */}
        {start && <Marker position={start}><Tooltip>Start</Tooltip></Marker>}
        {destination && <Marker position={destination}><Tooltip>Destination</Tooltip></Marker>}
      </MapContainer>
    </div>
  )
}
