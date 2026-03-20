/**
 * Minimal API for Arduino/ESP32 parking sensor updates.
 * - POST /api/parking — body: { lotId, availableSpots } (from ESP32)
 * - GET  /api/parking — returns { [lotId]: availableSpots } for sensor lots
 * Run: cd server && npm install && npm start
 * Default port: 3001 (set PORT to override)
 */
import express from 'express';
import cors from 'cors';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors({ origin: true }));
app.use(express.json());

// In-memory store: lotId -> availableSpots (only for sensor-powered lots)
const sensorOccupancy = new Map();

app.post('/api/parking', (req, res) => {
  const { lotId, availableSpots } = req.body ?? {};
  if (typeof lotId !== 'string' || typeof availableSpots !== 'number') {
    return res.status(400).json({ error: 'Missing or invalid lotId / availableSpots' });
  }
  sensorOccupancy.set(lotId, Math.max(0, availableSpots));
  console.log(`[sensor] ${lotId} -> ${availableSpots} available`);
  res.json({ ok: true, lotId, availableSpots });
});

app.get('/api/parking', (_req, res) => {
  const data = Object.fromEntries(sensorOccupancy);
  res.json(data);
});

// Bind to all IPv4 interfaces so hotspot/other devices can reach the API.
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Parking sensor API at http://localhost:${PORT}`);
  console.log('  GET  /api/parking  - current sensor occupancy');
  console.log('  POST /api/parking  - update (body: { lotId, availableSpots })');
});
