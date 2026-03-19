import { useState, useEffect, useCallback } from 'react';
import { initialParkingData, ParkingLocation } from '@/data/parkingData';

export function useParkingData(updateInterval: number = 10000) {
  const [parkingData, setParkingData] = useState<ParkingLocation[]>(initialParkingData);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());

  // Locations whose availability is driven by external sensors/cameras (not random demo simulation).
  const SENSOR_UPDATED_IDS = ['clergy-st-w', 'beamish-munro-hall', 'ontario-brock-lot'];

  const simulateUpdate = useCallback(() => {
    setParkingData(currentData =>
      currentData.map(location => {
        // Skip random simulation for Arduino-powered locations
        if (SENSOR_UPDATED_IDS.includes(location.id)) return location;

        // Random chance to change availability (30% chance)
        if (Math.random() > 0.3) return location;

        // Random change between -3 and +3 spots
        const change = Math.floor(Math.random() * 7) - 3;
        const newAvailable = Math.max(0, Math.min(location.totalSpots, location.availableSpots + change));

        return {
          ...location,
          availableSpots: newAvailable,
        };
      })
    );
    setLastUpdated(new Date());
  }, []);

  // Apply sensor update to state (shared by API poll and CustomEvent)
  const applySensorUpdate = useCallback((lotId: string, availableSpots: number) => {
    setParkingData(prev =>
      prev.map(loc =>
        loc.id === lotId
          ? { ...loc, availableSpots: Math.max(0, Math.min(loc.totalSpots, availableSpots)) }
          : loc
      )
    );
    setLastUpdated(new Date());
  }, []);

  // Poll backend API for sensor data (ESP32/Arduino POST to server; we GET here)
  const apiUrl = import.meta.env.VITE_PARKING_API_URL as string | undefined;
  useEffect(() => {
    if (!apiUrl?.trim()) return;
    const fetchSensorData = async () => {
      try {
        const res = await fetch(`${apiUrl.replace(/\/$/, '')}/api/parking`);
        if (!res.ok) return;
        const data = (await res.json()) as Record<string, number>;
        for (const [lotId, availableSpots] of Object.entries(data)) {
          if (SENSOR_UPDATED_IDS.includes(lotId) && typeof availableSpots === 'number') {
            applySensorUpdate(lotId, availableSpots);
          }
        }
      } catch {
        // ignore network errors (e.g. server not running)
      }
    };
    fetchSensorData();
    const interval = setInterval(fetchSensorData, 5000);
    return () => clearInterval(interval);
  }, [apiUrl, applySensorUpdate]);

  // Listen for in-page Arduino/sensor updates (e.g. Serial bridge or testing)
  // Dispatch: window.dispatchEvent(new CustomEvent('parking-sensor-update', { detail: { lotId, availableSpots } }));
  useEffect(() => {
    const handler = (e: CustomEvent<{ lotId: string; availableSpots: number }>) => {
      const { lotId, availableSpots } = e.detail ?? {};
      if (!lotId || typeof availableSpots !== 'number') return;
      applySensorUpdate(lotId, availableSpots);
    };
    window.addEventListener('parking-sensor-update', handler as EventListener);
    return () => window.removeEventListener('parking-sensor-update', handler as EventListener);
  }, [applySensorUpdate]);

  useEffect(() => {
    const interval = setInterval(simulateUpdate, updateInterval);
    return () => clearInterval(interval);
  }, [simulateUpdate, updateInterval]);

  const totalAvailable = parkingData.reduce((sum, loc) => sum + loc.availableSpots, 0);
  const totalSpots = parkingData.reduce((sum, loc) => sum + loc.totalSpots, 0);
  
  const streets = parkingData.filter(loc => loc.type === 'street');
  const lots = parkingData.filter(loc => loc.type === 'lot');

  return {
    parkingData,
    lastUpdated,
    totalAvailable,
    totalSpots,
    streets,
    lots,
  };
}
