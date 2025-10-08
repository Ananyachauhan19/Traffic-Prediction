Traffic frontend component

This folder contains a small React Leaflet component `TrafficMap.jsx` that:
- Displays route segments
- Colors segments by congestion level (green/yellow/red)
- Draws alternate route as dashed line
- Auto-refreshes every 30s by calling `/api/traffic/predict`

Usage:
- Add this component into your React/Next.js app and supply `start`, `destination`, and `segments` props:

```jsx
import TrafficMap from './src/components/TrafficMap'

const segments = [
  { id: 's1', coords: [[lat1, lng1], [lat2, lng2]] },
  { id: 's2', coords: [[lat2, lng2], [lat3, lng3]] }
]

<TrafficMap start={[lat1, lng1]} destination={[latN, lngN]} segments={segments} />
```

Notes:
- The component uses `/api/traffic/predict` POST endpoint; adapt payload as required by your backend.
- Tailwind classes in the component are minimal; configure Tailwind in your project for those to apply.
- Install dependencies: react, react-dom, leaflet, react-leaflet, axios, tailwindcss
