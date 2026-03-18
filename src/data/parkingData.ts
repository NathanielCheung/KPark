// Days: 0=Sunday, 1=Monday, ..., 6=Saturday
export interface ParkingHours {
  days: number[];
  start: string; // "09:30"
  end: string;   // "17:30"
}

export interface PriceTier {
  days?: number[];  // default: all days in parent
  startTime: string;
  endTime: string;
  rate: number;
  unit: 'hour' | 'halfhour' | 'flat';
  dailyMax?: number;
}

export interface ParkingPricing {
  permitOnly?: boolean;
  tiers: PriceTier[];
}

export interface ParkingLocation {
  id: string;
  name: string;
  type: 'street' | 'lot';
  totalSpots: number;
  availableSpots: number;
  lat: number;
  lng: number;
  address?: string;
  /** When payment/operations are required. Outside = closed for pay lots. */
  operatingHours?: ParkingHours[];
  /** null/undefined = free or street parking. permitOnly = no casual rates. */
  pricing?: ParkingPricing | null;
  /** Max stay in hours (e.g. 3, 4, 24) */
  maxStayHours?: number;
  /** For street parking: polyline path as [lat, lng][] for the street outline */
  path?: [number, number][];
  /** For street parking: exact positions of each spot (from sensors/Arduino). Availability is driven by availableSpots at render time. */
  spots?: { lat: number; lng: number }[];
  /** Lots/garages only: EV charging available */
  evCharging?: boolean | 'level2' | 'level3';
  /** Lots/garages only: Accessible parking spots available */
  accessibleParking?: boolean;
  /** Lots/garages only: Height clearance in meters (e.g. 2.1 = 7 ft) */
  heightClearanceM?: number;
}

// Downtown Kingston, Ontario coordinates
export const KINGSTON_CENTER = {
  lat: 44.2312,
  lng: -76.4860,
};

// Helper: parse "09:30" to minutes since midnight
function parseTime(t: string): number {
  const [h, m] = t.split(':').map(Number);
  return (h ?? 0) * 60 + (m ?? 0);
}

// Check if location is currently open (within operating hours)
export function isLocationOpen(location: ParkingLocation, date: Date = new Date()): boolean {
  if (location.type === 'street') return true; // Street parking always available (paid or free)
  if (!location.operatingHours?.length) return true;
  const day = date.getDay();
  const mins = date.getHours() * 60 + date.getMinutes();
  return location.operatingHours.some((h) => {
    if (!h.days.includes(day)) return false;
    const start = parseTime(h.start);
    let end = parseTime(h.end);
    if (end < start) end += 24 * 60; // overnight
    return mins >= start && mins < end;
  });
}

// Get human-readable hours string for next open/close
export function getHoursDisplay(location: ParkingLocation, date: Date = new Date()): string {
  if (!location.operatingHours?.length) return '24 hours';
  const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const groups = location.operatingHours
    .map((h) => ({
      days: h.days.sort((a, b) => a - b),
      start: h.start,
      end: h.end,
    }))
    .reduce((acc, curr) => {
      const key = `${curr.start}-${curr.end}`;
      const existing = acc.find((g) => g.start === curr.start && g.end === curr.end);
      if (existing) {
        existing.days = [...new Set([...existing.days, ...curr.days])];
      } else {
        acc.push({ days: [...curr.days], start: curr.start, end: curr.end });
      }
      return acc;
    }, [] as { days: number[]; start: string; end: string }[]);

  const fmt = (t: string) => {
    const [h, m] = t.split(':').map(Number);
    const hr = h! >= 12 ? (h! === 12 ? 12 : h! - 12) : h === 0 ? 12 : h!;
    return `${hr}:${String(m ?? 0).padStart(2, '0')} ${(h ?? 0) >= 12 ? 'p.m.' : 'a.m.'}`;
  };
  return groups
    .map((g) => {
      const dayStr = g.days.length === 7 ? 'Daily' : g.days.map((d) => dayNames[d]).join(', ');
      return `${dayStr} ${fmt(g.start)}–${fmt(g.end)}`;
    })
    .join('; ');
}

// Get current price at given time. Returns null if free, permit-only, or closed.
export function getCurrentPrice(location: ParkingLocation, date: Date = new Date()): {
  rate: number;
  unit: 'hour' | 'halfhour' | 'flat';
  dailyMax?: number;
  label: string;
} | null {
  if (!location.pricing?.tiers?.length || location.pricing.permitOnly) return null;
  const day = date.getDay();
  const mins = date.getHours() * 60 + date.getMinutes();

  for (const tier of location.pricing.tiers) {
    const days = tier.days ?? [0, 1, 2, 3, 4, 5, 6];
    if (!days.includes(day)) continue;
    let start = parseTime(tier.startTime);
    let end = parseTime(tier.endTime);
    if (end < start) end += 24 * 60;
    if (mins >= start && mins < end) {
      const unitLabel =
        tier.unit === 'hour' ? '/hr' : tier.unit === 'halfhour' ? '/30 min' : 'flat';
      const dailyStr = tier.dailyMax ? ` (max $${tier.dailyMax}/day)` : '';
      const label =
        tier.rate === 0 ? 'Free' : `$${tier.rate.toFixed(2)}${unitLabel}${dailyStr}`;
      return {
        rate: tier.rate,
        unit: tier.unit,
        dailyMax: tier.dailyMax,
        label,
      };
    }
  }
  return null;
}

// Format price for display
export function formatPrice(price: { rate: number; unit: string; dailyMax?: number }): string {
  const u = price.unit === 'hour' ? '/hr' : price.unit === 'halfhour' ? '/30 min' : 'flat';
  let s = `$${price.rate.toFixed(2)}${u}`;
  if (price.dailyMax) s += ` (max $${price.dailyMax}/day)`;
  return s;
}

export function getStatusColor(available: number, total: number): 'available' | 'low' | 'full' {
  if (available === 0) return 'full';
  const percentage = (available / total) * 100;
  if (percentage <= 20) return 'low';
  return 'available';
}

export type AccessibilityFilterState = {
  evCharging: boolean;
  accessibleParking: boolean;
  minHeightClearanceM: number;
  freeParking: boolean;
};

/** True if location is free: no pricing, or at least one tier has rate 0 (e.g. free on Sundays). */
export function isFreeParking(loc: ParkingLocation): boolean {
  if (!loc.pricing?.tiers?.length) return true;
  return loc.pricing.tiers.some(t => t.rate === 0);
}

export function filterLotsByAccessibility(
  lots: ParkingLocation[],
  filters: AccessibilityFilterState
): ParkingLocation[] {
  return lots.filter(loc => {
    if (loc.type !== 'lot') return true;
    if (filters.evCharging && !loc.evCharging) return false;
    if (filters.accessibleParking && !loc.accessibleParking) return false;
    if (
      filters.minHeightClearanceM > 0 &&
      (!loc.heightClearanceM || loc.heightClearanceM < filters.minHeightClearanceM)
    )
      return false;
    return true;
  });
}

export function getStatusLabel(status: 'available' | 'low' | 'full'): string {
  switch (status) {
    case 'available':
      return 'Available';
    case 'low':
      return 'Limited';
    case 'full':
      return 'Full';
  }
}

/** Haversine distance in meters */
function dist(
  lat1: number,
  lng1: number,
  lat2: number,
  lng2: number
): number {
  const R = 6371000;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
    Math.cos((lat2 * Math.PI) / 180) *
    Math.sin(dLng / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Get spot positions for a street, placed along the path away from vertices (intersections).
 * Uses path interpolation when no explicit spots, or when spots count doesn't match.
 */
export function getSpotPositions(location: ParkingLocation): { lat: number; lng: number }[] {
  const path = location.path;
  const total = location.totalSpots;
  if (!path || path.length < 2 || total <= 0) return [];

  const segLengths: number[] = [];
  let totalLen = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const d = dist(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1]);
    segLengths.push(d);
    totalLen += d;
  }
  if (totalLen === 0) return [];

  const result: { lat: number; lng: number }[] = [];
  for (let i = 0; i < total; i++) {
    const t = (i + 0.5) / total;
    let targetDist = t * totalLen;
    let accumulated = 0;
    for (let j = 0; j < segLengths.length; j++) {
      if (accumulated + segLengths[j] >= targetDist || j === segLengths.length - 1) {
        const segLen = segLengths[j] || 1;
        let frac = (targetDist - accumulated) / segLen;
        frac = Math.max(0.2, Math.min(0.8, frac));
        const p0 = path[j];
        const p1 = path[j + 1];
        result.push({
          lat: p0[0] + (p1[0] - p0[0]) * frac,
          lng: p0[1] + (p1[1] - p0[1]) * frac,
        });
        break;
      }
      accumulated += segLengths[j];
    }
  }
  return result;
}

// City of Kingston parking data (source: cityofkingston.ca)
const MON_SAT = [1, 2, 3, 4, 5, 6];
const MON_FRI = [1, 2, 3, 4, 5];
const ALL_DAYS = [0, 1, 2, 3, 4, 5, 6];

export const initialParkingData: ParkingLocation[] = [
  // Street parking (metered)
  {
    id: 'princess-st',
    name: 'Princess Street',
    type: 'street',
    totalSpots: 12,
    availableSpots: 3,
    lat: 44.2312,
    lng: -76.482,
    address: 'Princess St, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 3,
    path: [
      [44.231108023785566, -76.4794786975904],
      [44.23180539529639, -76.48392266041621],
      [44.23257486075278, -76.48894214599909],
      [44.23303717412506, -76.49191547269139],
      [44.23540062963406, -76.49780603151758],
    ],
  },

  {
    id: 'queen-st',
    name: 'Queen Street',
    type: 'street',
    totalSpots: 13,
    availableSpots: 5,
    lat: 44.233109940874726,
    lng: -76.48638773476445,
    address: 'Queen St, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 3,
    path: [
      [44.23200561051448, -76.47928777867283],
      [44.23406269317056, -76.49271445067188],
    ],
  },

  {
    id: 'johnson-st',
    name: 'Johnson Street',
    type: 'street',
    totalSpots: 20,
    availableSpots: 13,
    lat: 44.231166546574,
    lng: -76.49224382187798,
    address: 'Johnson St, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 3,
    path: [
      [44.22837914671274, -76.48149722234045],
      [44.23010096757585, -76.4852028181767],
      [44.23022079579275, -76.4857014125111],
      [44.231002850045655, -76.49107936751325],
      [44.23168362714523, -76.49554103259378],
      [44.231700688976105, -76.4959536829794],
      [44.23153671266368, -76.50082112834113],
      [44.23127309275924, -76.5089363070011],
      [44.23110280841988, -76.51193775160975],
      [44.23099724974182, -76.51355684822191],
    ],
  },

  {
    id: 'barrie-st',
    name: 'Barrie Street',
    type: 'street',
    totalSpots: 15,
    availableSpots: 2,
    lat: 44.22838242106414,
    lng: -76.49091737215626,
    address: 'Barrie Street, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2.5, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 2,
    path: [
      [44.22293932202447, -76.49033776879267],
      [44.22884877506044, -76.49092826377043],
      [44.2312475346331, -76.49119296426993],
      [44.235084557197695, -76.49019523137619],
      [44.236719132837095, -76.48982812224571],
      [44.23692337230281, -76.48980776035025],
      [44.239782650382494, -76.49072404566057]
    ],
  },

  {
    id: 'king-st-e',
    name: 'King Street East',
    type: 'street',
    totalSpots: 8,
    availableSpots: 0,
    lat: 44.2295,
    lng: -76.4785,
    address: 'King St E, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2.5, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 2,
    path: [
      [44.23319370452168, -76.48040516118641],
      [44.23044356550673, -76.4812464041255],
      [44.224829869221935, -76.48624128407852],
      [44.22275756416579, -76.4902897657242]
    ],
  },
  {
    id: 'division-st',
    name: 'Division Street',
    type: 'street',
    totalSpots: 10,
    availableSpots: 2,
    lat: 44.2335,
    lng: -76.487,
    address: 'Division St, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 3,
    path: [
      [44.227775223459616, -76.49298267296957],
      [44.228517694211945, -76.49301881767735],
      [44.22904432476582, -76.49304291414921],
      [44.2297522469386, -76.49300074532317],
      [44.23130619258026, -76.49298267296957],
      [44.23218242717138, -76.49274170824975],
      [44.23399230629249, -76.4931849817261],
      [44.234893142551215, -76.49340459540872],
      [44.237721444322375, -76.49409089069168],
      [44.23938925137372, -76.49442031072236],
      [44.24180433500541, -76.49495835147033],
      [44.25567212701776, -76.49704857533119],
      [44.25888186120801, -76.49759440316922],
    ],
  },
  {
    id: 'brock-st',
    name: 'Brock Street',
    type: 'street',
    totalSpots: 6,
    availableSpots: 4,
    lat: 44.2298,
    lng: -76.484,
    address: 'Brock St, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 3,
    path: [
      [44.23022034201438, -76.47980795378562],
      [44.231370647601295, -76.48736710102051],
      [44.23262940304673, -76.49560943695128],
    ],
  },
  {
    id: 'ontario-st',
    name: 'Ontario Street',
    type: 'street',
    totalSpots: 15,
    availableSpots: 7,
    lat: 44.228,
    lng: -76.478,
    address: 'Ontario St, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2.5, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 2,
    path: [
      [44.22420282531405, -76.48533125749618],
      [44.22843895311297, -76.48141836322341],
      [44.23023967517676, -76.4797655045475],
      [44.23037261147047, -76.47969804111425],
      [44.23319442044337, -76.47882101308795],
    ],
  },
  {
    id: 'wellington-st',
    name: 'Wellington Street',
    type: 'street',
    totalSpots: 8,
    availableSpots: 1,
    lat: 44.232,
    lng: -76.49,
    address: 'Wellington St, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 3,
    path: [
      [44.22599550121487, -76.4870173347717],
      [44.230681937881684, -76.48272904242307],
      [44.23184647354094, -76.48235699879477],
      [44.23336663016369, -76.4818654737942],
      [44.233521844855176, -76.48179615285066],
      [44.23437862258396, -76.48156219466854],
      [44.23528505500727, -76.48129357601424],
      [44.23559547386051, -76.4811809294819],
      [44.23581897442122, -76.48107694806698],
      [44.23602384918928, -76.48103362247818],
      [44.23619147346855, -76.48103362247818],
      [44.236644677463744, -76.48112893877489],
      [44.236812301342695, -76.48116360491115],
      [44.23742691313183, -76.48119826538249],
      [44.237472065534575, -76.48118659797572],
    ],
  },
  {
    id: 'clergy-st-w',
    name: 'Clergy St W',
    type: 'street',
    totalSpots: 6,
    availableSpots: 2,
    lat: 44.228894698301386,
    lng: -76.49201753033214,
    address: 'Clergy St W, Downtown Kingston',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' },
        { days: [0], startTime: '00:00', endTime: '24:00', rate: 0, unit: 'hour' },
      ],
    },
    maxStayHours: 3,
    path: [
      [44.228929622014476, -76.49304324557262],
      [44.2288597745883, -76.49099181509166],
    ],
  },
  // 3-hour limit lots
  {
    id: 'angrove-lot',
    name: 'Angrove Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 30,
    availableSpots: 8,
    lat: 44.233,
    lng: -76.49,
    address: '207 Wellington St',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: { tiers: [{ startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' }] },
    maxStayHours: 3,
  },
  {
    id: 'armstrong-lot',
    name: 'Armstrong Memorial Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 25,
    availableSpots: 5,
    lat: 44.2295,
    lng: -76.484,
    address: '289 Brock St',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: { tiers: [{ startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' }] },
    maxStayHours: 3,
  },
  {
    id: 'barrack-lot',
    name: 'Barrack Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 40,
    availableSpots: 12,
    lat: 44.232,
    lng: -76.49,
    address: '63 Barrack St',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: { tiers: [{ startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' }] },
    maxStayHours: 3,
  },
  {
    id: 'gorsline-lot',
    name: 'Gorsline Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 35,
    availableSpots: 10,
    lat: 44.231,
    lng: -76.485,
    address: 'MacDonnell at Princess St',
    operatingHours: [{ days: MON_FRI, start: '08:00', end: '17:00' }],
    pricing: { tiers: [{ days: MON_FRI, startTime: '08:00', endTime: '17:00', rate: 2, unit: 'hour' }] },
    maxStayHours: 3,
  },
  {
    id: 'library-lot',
    name: 'Kingston Frontenac Public Library',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 20,
    availableSpots: 4,
    lat: 44.23,
    lng: -76.488,
    address: '130 Johnson St',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: { tiers: [{ startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' }] },
    maxStayHours: 3,
  },
  {
    id: 'mckee-lot',
    name: 'McKee Memorial Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 45,
    availableSpots: 15,
    lat: 44.23,
    lng: -76.483,
    address: '256 Queen St',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: { tiers: [{ startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' }] },
    maxStayHours: 3,
  },
  {
    id: 'waterfront-lot',
    name: 'Waterfront Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 50,
    availableSpots: 18,
    lat: 44.22218,
    lng: -76.49111,
    address: 'Waterfront',
    operatingHours: [{ days: MON_SAT, start: '08:00', end: '17:00' }],
    pricing: { tiers: [{ startTime: '08:00', endTime: '17:00', rate: 2.5, unit: 'hour' }] },
    maxStayHours: 3,
  },
  // 4-hour limit
  {
    id: 'upper-robert-bruce',
    name: 'Upper Robert Bruce Memorial Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 60,
    availableSpots: 22,
    lat: 44.234,
    lng: -76.492,
    address: '7 Montreal St',
    operatingHours: [{ days: MON_SAT, start: '08:00', end: '17:30' }],
    pricing: { tiers: [{ startTime: '08:00', endTime: '17:30', rate: 2, unit: 'hour' }] },
    maxStayHours: 4,
  },
  // 24-hour garages / various limits
  {
    id: 'chown-garage',
    name: 'Chown Memorial Garage',
    type: 'lot',
    totalSpots: 150,
    availableSpots: 45,
    lat: 44.228,
    lng: -76.485,
    address: '197 Brock St',
    evCharging: 'level2',
    accessibleParking: true,
    heightClearanceM: 1.9,
    operatingHours: [{ days: ALL_DAYS, start: '00:00', end: '23:59' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '06:00', endTime: '18:00', rate: 2, unit: 'hour' },
        { days: MON_SAT, startTime: '18:00', endTime: '06:00', rate: 4, unit: 'flat' },
        { days: [0], startTime: '06:00', endTime: '06:00', rate: 4, unit: 'flat' },
      ],
    },
    maxStayHours: 24,
  },
  {
    id: 'frontenac-courthouse',
    name: 'Frontenac County Court House',
    type: 'lot',
    totalSpots: 80,
    availableSpots: 25,
    lat: 44.229,
    lng: -76.486,
    address: '5 Court St',
    accessibleParking: true,
    operatingHours: [{ days: MON_FRI, start: '08:00', end: '17:00' }],
    pricing: { tiers: [{ startTime: '08:00', endTime: '17:00', rate: 2.5, unit: 'hour' }] },
    maxStayHours: 24,
  },
  {
    id: 'frontenac-lot',
    name: 'Frontenac Lot',
    type: 'lot',
    totalSpots: 55,
    availableSpots: 18,
    lat: 44.23,
    lng: -76.489,
    address: '28 Barrack St',
    evCharging: 'level3',
    accessibleParking: true,
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: { tiers: [{ startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' }] },
    maxStayHours: 24,
  },
  {
    id: 'hanson-garage',
    name: 'Hanson Memorial Garage',
    type: 'lot',
    totalSpots: 120,
    availableSpots: 35,
    evCharging: 'level2',
    accessibleParking: true,
    heightClearanceM: 1.8,
    lat: 44.229,
    lng: -76.484,
    address: '105 Brock St',
    operatingHours: [{ days: ALL_DAYS, start: '00:00', end: '23:59' }],
    pricing: {
      tiers: [
        { days: MON_SAT, startTime: '06:00', endTime: '18:00', rate: 2, unit: 'hour' },
        { days: MON_SAT, startTime: '18:00', endTime: '06:00', rate: 4, unit: 'flat' },
        { days: [0], startTime: '06:00', endTime: '06:00', rate: 4, unit: 'flat' },
      ],
    },
    maxStayHours: 24,
  },
  {
    id: 'lower-robert-bruce',
    name: 'Lower Robert Bruce Memorial Lot',
    type: 'lot',
    totalSpots: 70,
    availableSpots: 28,
    lat: 44.232,
    lng: -76.493,
    address: '266 Bagot St',
    evCharging: 'level3',
    accessibleParking: true,
    heightClearanceM: 2.1,
    operatingHours: [{ days: MON_SAT, start: '08:00', end: '17:00' }],
    pricing: { tiers: [{ startTime: '08:00', endTime: '17:00', rate: 2, unit: 'hour' }] },
    maxStayHours: 24,
  },
  {
    id: 'ontario-brock-lot',
    name: 'Ontario and Brock Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 10,
    availableSpots: 10,
    lat: 44.228,
    lng: -76.481,
    address: 'Ontario St west of Princess',
    operatingHours: [{ days: ALL_DAYS, start: '00:00', end: '23:59' }],
    pricing: {
      tiers: [
        {
          days: MON_SAT,
          startTime: '07:00',
          endTime: '17:00',
          rate: 1.75,
          unit: 'halfhour',
          dailyMax: 17.5,
        },
        { days: MON_SAT, startTime: '17:00', endTime: '07:00', rate: 6.25, unit: 'flat' },
        { days: [0], startTime: '07:00', endTime: '07:00', rate: 6.25, unit: 'flat' },
      ],
    },
    maxStayHours: 24,
  },
  {
    id: 'beamish-munro-hall',
    name: 'Beamish Munro Hall',
    type: 'lot',
    totalSpots: 3,
    availableSpots: 3,
    lat: 44.228699442682995,
    lng: -76.49150134966044,
    address: '99 University Ave, Queen\'s University',
    evCharging: 'level2',
    accessibleParking: true,
    operatingHours: [{ days: ALL_DAYS, start: '00:00', end: '23:59' }],
    pricing: { tiers: [{ startTime: '07:00', endTime: '18:00', rate: 2.5, unit: 'hour' }] },
    maxStayHours: 4,
  },
  {
    id: 'pumphouse-lot',
    name: 'Pump House Lot',
    type: 'lot',
    totalSpots: 40,
    availableSpots: 0,
    lat: 44.22390,
    lng: -76.48504,
    address: '23 Ontario St',
    evCharging: 'level2',
    accessibleParking: true,
    operatingHours: [{ days: MON_FRI, start: '08:00', end: '17:00' }],
    pricing: { tiers: [{ startTime: '08:00', endTime: '17:00', rate: 2, unit: 'hour' }] },
    maxStayHours: 24,
  },
  {
    id: 'rideaucrest-lot',
    name: 'Rideaucrest Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 50,
    availableSpots: 15,
    lat: 44.231,
    lng: -76.495,
    address: '175 Rideau St',
    operatingHours: [{ days: ALL_DAYS, start: '00:00', end: '23:59' }],
    pricing: {
      tiers: [
        { days: MON_FRI, startTime: '08:00', endTime: '15:00', rate: 2, unit: 'hour' },
        { days: [1, 2, 3, 4], startTime: '15:00', endTime: '24:00', rate: 4, unit: 'flat' },
        { days: [5], startTime: '15:00', endTime: '24:00', rate: 4, unit: 'flat' },
        { days: [6, 0], startTime: '00:00', endTime: '24:00', rate: 4, unit: 'flat' },
        { days: [0], startTime: '00:00', endTime: '08:00', rate: 4, unit: 'flat' },
      ],
    },
    maxStayHours: 24,
  },
  {
    id: 'springer-lot',
    name: 'Springer Memorial Lot',
    accessibleParking: true,
    type: 'lot',
    totalSpots: 200,
    availableSpots: 65,
    lat: 44.229,
    lng: -76.486,
    address: '140 Queen St',
    operatingHours: [{ days: MON_SAT, start: '09:30', end: '17:30' }],
    pricing: { tiers: [{ startTime: '09:30', endTime: '17:30', rate: 2, unit: 'hour' }] },
    maxStayHours: 24,
  },
];
