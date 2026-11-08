import type { StyleSpecification } from 'maplibre-gl';

const STYLE_URL = 'https://tiles.openfreemap.org/styles/liberty';

export const LAND = '#f7f5f0';

const C = {
  park: '#b5e0a9',
  garden: '#d9efd1',
  wood: '#9dd393',
  grassLight: '#cdebc4',
  sand: '#f7f0da',
  hospital: '#fce4e2',
  water: '#9cc0f9',
  building: '#ebe9e4',
  buildingLine: '#dedbd4',
  minor: '#ffffff',
  minorCase: '#dddbd6',
  major: '#ffffff',
  majorCase: '#cdcbc6',
  motorway: '#fde9b0',
  motorwayCase: '#eec46a',
  rail: '#c4c7cc',
  label: '#3c4043',
  labelMinor: '#6b6f74',
  waterLabel: '#3a6bd1',
  halo: '#ffffff',
};

const DROPPED = new Set([
  'natural_earth', 'building-3d', 'poi_r20', 'poi_r7', 'poi_r1', 'poi_transit', 'park_outline', 'road_area_pattern', 'landuse_residential',
  'road_one_way_arrow', 'road_one_way_arrow_opposite', 'aeroway_fill', 'aeroway_runway', 'aeroway_taxiway', 'airport',
  'highway-shield-non-us', 'highway-shield-us-interstate', 'road_shield_us',
]);

type Loose = { id: string; type: string; maxzoom?: number; filter?: unknown; paint?: Record<string, unknown> };

function paint(layer: Loose): Record<string, unknown> | null {
  const { id, type } = layer;
  if (type === 'background') return { 'background-color': LAND };
  if (id === 'park') return { 'fill-color': C.park, 'fill-outline-color': C.park, 'fill-opacity': 1 };
  if (id === 'landcover_wood') return { 'fill-color': C.wood, 'fill-opacity': 1 };
  if (id === 'landcover_grass') {
    return { 'fill-color': ['match', ['get', 'subclass'], ['garden', 'flowerbed', 'shrubbery'], C.garden, C.park], 'fill-opacity': 1 };
  }
  if (/^landuse_(pitch|track|cemetery|school)$/.test(id)) return { 'fill-color': C.grassLight };
  if (id === 'landuse_hospital') return { 'fill-color': C.hospital };
  if (id === 'landcover_sand') return { 'fill-color': C.sand };
  if (id === 'water') return { 'fill-color': C.water };
  if (id.startsWith('waterway') && type === 'line') return { 'line-color': C.water };
  if (id === 'building') {
    return { 'fill-color': C.building, 'fill-outline-color': C.buildingLine, 'fill-opacity': ['interpolate', ['linear'], ['zoom'], 14.5, 0, 15.5, 1] };
  }
  if (type === 'line' && id.includes('rail')) return { 'line-color': C.rail };
  if (type === 'line' && /^(road|tunnel|bridge)_/.test(id)) {
    const kind = id.includes('motorway') ? 'motorway' : /trunk_primary|secondary_tertiary|_link/.test(id) ? 'major' : 'minor';
    return { 'line-color': C[(id.includes('casing') ? `${kind}Case` : kind) as keyof typeof C] };
  }
  if (type === 'symbol') {
    const text = id.startsWith('water') ? C.waterLabel : id.startsWith('label_') ? C.label : C.labelMinor;
    return { 'text-color': text, 'text-halo-color': C.halo };
  }
  return null;
}

function recolour(style: StyleSpecification): StyleSpecification {
  const layers = (style.layers as unknown as Loose[])
    .filter((l) => !DROPPED.has(l.id))
    .map((l) => {
      const p = paint(l);
      const out: Loose = p ? { ...l, paint: { ...l.paint, ...p } } : { ...l };
      if (l.id === 'building') delete out.maxzoom;
      if (l.id === 'park') out.filter = ['all', l.filter, ['match', ['get', 'class'], ['nature_reserve', 'national_park'], true, false]];
      if (l.id.endsWith('path_pedestrian')) delete out.paint?.['line-dasharray'];
      return out;
    });
  return { ...style, layers: layers as unknown as StyleSpecification['layers'] };
}

const OFFLINE: StyleSpecification = {
  version: 8,
  sources: {},
  layers: [{ id: 'background', type: 'background', paint: { 'background-color': LAND } }],
};

export async function basemapStyle(): Promise<{ style: StyleSpecification; online: boolean }> {
  try {
    const res = await fetch(STYLE_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return { style: recolour((await res.json()) as StyleSpecification), online: true };
  } catch {
    return { style: OFFLINE, online: false };
  }
}
