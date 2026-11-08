import { useEffect, useRef, useState, type RefObject } from 'react';
import {
  Map as MLMap,
  setWorkerUrl,
  type ExpressionSpecification,
  type FilterSpecification,
  type GeoJSONSource,
  type LngLatBoundsLike,
  type PaddingOptions,
} from 'maplibre-gl';
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?url';
import type { Feature, FeatureCollection, LineString, Point, Polygon } from 'geojson';
import { slowestAt, type BBox, type CitySpeeds, type Data, type LayerId, type Segment, type Speeds } from './data';
import { basemapStyle } from './basemap';

setWorkerUrl(workerUrl);

export const SERIES = { traffic: '#2a78d6', crossings: '#a62a8c', counts: '#1baf7a' } as const;
export const SPEED_RAMP: [number, string][] = [
  [0, '#5c1d06'], [6, '#9b3510'], [12, '#d9541f'], [20, '#f08a5b'], [32, '#f6b796'],
];
export const TRAFFIC_RAMP: [number, string][] = [
  [0, '#b7d3f6'], [0.4, '#6da7ec'], [0.7, '#2a78d6'], [0.9, '#1c5cab'], [1.1, '#0d366b'],
];
export const RULE = { agrees: '#6250d6', disagrees: '#d03b3b', held: '#1f9e89' } as const;
export const CONCERN_RAMP: [number, string][] = [[0, '#fdd5bf'], [1, '#e8602c'], [5, '#c2410c'], [20, '#7c2d12']];
const INK = '#1f1d1a';
const HIGHLIGHT = '#ffc629';
const HIGHLIGHT_SOFT = '#ffe7a3';
const EDINBURGH: BBox = [-3.45, 55.83, -2.95, 56.02];
const NOTHING: FilterSpecification = ['==', ['get', 'name'], '\u0000'];
const NO_NODE: FilterSpecification = ['==', ['get', 'node'], '\u0000'];
const ROUND = { 'line-cap': 'round', 'line-join': 'round' } as const;
const DASHED = { 'line-cap': 'butt', 'line-join': 'round' } as const;

export const DESIGN = { close: '#d03b3b', reallocate: '#0e8a83', signalise_crossing: '#1f1d1a' } as const;

/** What the map highlights for a design: closed and bus-lane streets by name, new signals by crossing node. */
export interface DesignView {
  closed: string[];
  buses: string[];
  signals: string[];
}

const GROUPS: Partial<Record<LayerId, string[]>> = {
  designs: ['design-bus-case', 'design-bus', 'design-closed-case', 'design-closed', 'design-signal', 'design-pins', 'design-label'],
  rules: ['rules-held', 'rules-glow', 'rules-wrong-glow', 'rules-case', 'rules', 'rules-wrong-case',
    'rules-wrong', 'rules-pins', 'rules-label'],
  concerns: ['concerns-glow', 'concerns-case', 'concerns', 'concerns-pins', 'concerns-label'],
  speeds: ['city-speeds-case', 'city-speeds', 'speeds-case', 'speeds', 'speeds-predicted'],
  traffic: ['city-traffic-case', 'city-traffic', 'traffic-case', 'traffic'],
  crossings: ['crossings'],
  counts: ['counts', 'counts-label'],
};

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export interface MapApi {
  showStreet(name: string): void;
  reset(): void;
  zoomIn(): void;
  zoomOut(): void;
}

interface Props {
  data: Data;
  speeds: Speeds | null;
  city: CitySpeeds | null;
  visible: Record<LayerId, boolean>;
  selected: string | null;
  design?: DesignView | null;
  bin: number;
  onSelect(name: string | null): void;
  onChange?(id: string): void;
  onError(message: string): void;
  apiRef: RefObject<MapApi | null>;
}

const EMPTY: FeatureCollection<LineString> = { type: 'FeatureCollection', features: [] };

type SegmentFeature = Feature<LineString, { id: string; rev: string | null; name: string }>;

/** Every segment with a bus speed in the bin, carrying its slower direction's speed (spd) and whether a bus was seen (obs). */
function speedFeatures(segments: SegmentFeature[], speeds: Speeds | null, bin: number): FeatureCollection<LineString> {
  if (!speeds) return EMPTY;
  return {
    type: 'FeatureCollection',
    features: segments.flatMap((f) => {
      const reading = slowestAt(speeds, [f.properties.id, f.properties.rev], bin);
      return reading ? [{ type: 'Feature' as const, geometry: f.geometry,
        properties: { name: f.properties.name, spd: reading.speed, obs: reading.observed } }] : [];
    }),
  };
}

/** The study area plus a margin of `share` of its size on every side, so the opening view shows some city around it. */
export function widen([w, s, e, n]: BBox, share = 0.2): BBox {
  const dx = (e - w) * share, dy = (n - s) * share;
  return [w - dx, s - dy, e + dx, n + dy];
}

type Pins = FeatureCollection<Point, { name: string; ok: boolean }>;

/** One point per ruled street, at the middle of its longest ruled segment; ok is false if OSM disagrees anywhere on it. */
export const rulePins = (segments: Feature<LineString, Segment>[]): Pins =>
  streetPins(segments, (s) => Boolean(s.rule), (s) => s.osm_ok !== false);

/** One point per street named in consultation material, at the middle of its longest named segment. */
export const concernPins = (segments: Feature<LineString, Segment>[]): Pins =>
  streetPins(segments, (s) => s.concern != null, () => true);

function streetPins(segments: Feature<LineString, Segment>[], keep: (s: Segment) => boolean, isOk: (s: Segment) => boolean): Pins {
  const streets = new Map<string, { longest: Feature<LineString, Segment>; ok: boolean }>();
  for (const f of segments) {
    const { name, len } = f.properties;
    if (!keep(f.properties) || !name) continue;
    const seen = streets.get(name);
    const ok = isOk(f.properties);
    if (!seen) streets.set(name, { longest: f, ok });
    else streets.set(name, { longest: len > seen.longest.properties.len ? f : seen.longest, ok: seen.ok && ok });
  }
  return {
    type: 'FeatureCollection',
    features: [...streets].map(([name, { longest, ok }]) => {
      const coords = longest.geometry.coordinates;
      return { type: 'Feature', geometry: { type: 'Point', coordinates: coords[Math.floor(coords.length / 2)] }, properties: { name, ok } };
    }),
  };
}

type ChangePins = FeatureCollection<Point, { id: string; kind: string; label: string }>;

const PIN_LABEL = { close: 'closed', reallocate: 'bus lane', signalise_crossing: 'new signals' } as const;

/** A clickable pin for every change on the design menu, on its street's longest car segment or its crossing,
 * and the box around the pins and the streets they change. */
export function changePins(data: Data): { pins: ChangePins; bbox: BBox | null } {
  const features: ChangePins['features'] = [];
  const points: number[][] = [];
  for (const c of data.designs?.candidates ?? []) {
    let at: number[] | undefined;
    if (c.kind === 'signalise_crossing') {
      at = data.crossings?.features.find((f) => String(f.properties.node) === c.node)?.geometry.coordinates;
    } else {
      const car = data.network.features.filter((f) => f.properties.name === c.street && f.properties.car);
      const longest = car.reduce<typeof car[number] | null>((a, f) => (!a || f.properties.len > a.properties.len ? f : a), null);
      at = longest?.geometry.coordinates[Math.floor(longest.geometry.coordinates.length / 2)];
      const box = c.street ? data.streets[c.street]?.bbox : undefined;
      if (box) points.push([box[0], box[1]], [box[2], box[3]]);
    }
    if (!at) continue;
    points.push(at);
    features.push({ type: 'Feature', geometry: { type: 'Point', coordinates: at },
      properties: { id: c.id, kind: c.kind, label: `${c.street ?? 'Crossing'} · ${PIN_LABEL[c.kind]}` } });
  }
  const pins: ChangePins = { type: 'FeatureCollection', features };
  if (!points.length) return { pins, bbox: null };
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  return { pins, bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] };
}

function padding(): PaddingOptions {
  const w = window.innerWidth;
  const panel = Math.min(390, Math.max(330, w * 0.27));
  return w > 760 ? { top: 56, bottom: 56, left: panel + 56, right: 88 } : { top: 24, bottom: 24, left: 24, right: 24 };
}

function studyMask([w, s, e, n]: BBox): Feature<Polygon> {
  const world = [[-4, 55.5], [-2.4, 55.5], [-2.4, 56.5], [-4, 56.5], [-4, 55.5]];
  const hole = [[w, s], [w, n], [e, n], [e, s], [w, s]];
  return { type: 'Feature', properties: {}, geometry: { type: 'Polygon', coordinates: [world, hole] } };
}

function byFlow(low: number, high: number): ExpressionSpecification {
  return ['interpolate', ['linear'], ['coalesce', ['get', 'flow'], 0], 0, low, 2000, high];
}

function trafficWidth(extra: number): ExpressionSpecification {
  return ['interpolate', ['linear'], ['zoom'], 13, byFlow(1.2 + extra / 2, 4.5 + extra / 2), 17, byFlow(2.5 + extra, 12 + extra)];
}

function byZoom(low: number, high: number): ExpressionSpecification {
  return ['interpolate', ['exponential', 1.4], ['zoom'], 13, low, 17, high];
}

function addLayers(map: MLMap, data: Data, online: boolean) {
  const labels = map.getStyle().layers.find((l) => l.type === 'symbol')?.id;
  map.addSource('network', { type: 'geojson', data: data.network });
  map.addSource('study', { type: 'geojson', data: studyMask(data.meta.bbox) });
  if (!online) {
    map.addLayer({ id: 'network-base', type: 'line', source: 'network', layout: ROUND,
      paint: { 'line-color': '#d6cdbd', 'line-width': 1.2 } });
  }
  map.addLayer({ id: 'study-mask', type: 'fill', source: 'study',
    paint: { 'fill-color': '#ffffff', 'fill-opacity': 0.45 } }, labels);
  map.addLayer({ id: 'study-outline', type: 'line', source: 'study',
    paint: { 'line-color': INK, 'line-width': 1.5, 'line-opacity': 0.6, 'line-dasharray': [4, 2.5] } }, labels);

  map.addLayer({ id: 'hover', type: 'line', source: 'network', filter: NOTHING, layout: ROUND,
    paint: { 'line-color': HIGHLIGHT_SOFT, 'line-blur': 2, 'line-width': byZoom(10, 24) } }, labels);
  map.addLayer({ id: 'selected', type: 'line', source: 'network', filter: NOTHING, layout: ROUND,
    paint: { 'line-color': HIGHLIGHT, 'line-blur': 0.5, 'line-width': byZoom(9, 22) } }, labels);

  const hasFlow: FilterSpecification = ['>', ['coalesce', ['get', 'flow'], 0], 0];
  const trafficColour = ['interpolate', ['linear'], ['get', 'vc'], ...TRAFFIC_RAMP.flat()] as ExpressionSpecification;
  if (data.cityTraffic) {
    map.addSource('city-traffic', { type: 'geojson', data: data.cityTraffic });
    map.addLayer({ id: 'city-traffic-case', type: 'line', source: 'city-traffic', layout: ROUND,
      paint: { 'line-color': '#ffffff', 'line-width': trafficWidth(2) } }, labels);
    map.addLayer({ id: 'city-traffic', type: 'line', source: 'city-traffic', layout: ROUND,
      paint: { 'line-color': trafficColour, 'line-width': trafficWidth(0) } }, labels);
  }
  map.addLayer({ id: 'traffic-case', type: 'line', source: 'network', filter: hasFlow, layout: ROUND,
    paint: { 'line-color': '#ffffff', 'line-width': trafficWidth(2) } }, labels);
  map.addLayer({ id: 'traffic', type: 'line', source: 'network', filter: hasFlow, layout: ROUND,
    paint: {
      'line-color': trafficColour,
      'line-width': trafficWidth(0),
    } }, labels);

  if (data.speeds) {
    map.addSource('speeds', { type: 'geojson', data: EMPTY });
    const colour = ['interpolate', ['linear'], ['get', 'spd'], ...SPEED_RAMP.flat()] as ExpressionSpecification;
    const seen = (observed: boolean): FilterSpecification => ['==', ['get', 'obs'], observed];
    if (data.city) {
      map.addSource('city-speeds', { type: 'geojson', data: EMPTY });
      map.addLayer({ id: 'city-speeds-case', type: 'line', source: 'city-speeds', layout: ROUND,
        paint: { 'line-color': '#ffffff', 'line-width': byZoom(3.5, 9) } }, labels);
      map.addLayer({ id: 'city-speeds', type: 'line', source: 'city-speeds', layout: ROUND,
        paint: { 'line-color': colour, 'line-width': byZoom(2, 6) } }, labels);
    }
    map.addLayer({ id: 'speeds-case', type: 'line', source: 'speeds', layout: ROUND,
      paint: { 'line-color': '#ffffff', 'line-width': byZoom(4.5, 11) } }, labels);
    map.addLayer({ id: 'speeds', type: 'line', source: 'speeds', filter: seen(true), layout: ROUND,
      paint: { 'line-color': colour, 'line-width': byZoom(2.5, 7) } }, labels);
    map.addLayer({ id: 'speeds-predicted', type: 'line', source: 'speeds', filter: seen(false), layout: DASHED,
      paint: { 'line-color': colour, 'line-width': byZoom(2.5, 7), 'line-dasharray': [1.2, 0.9] } }, labels);
  }

  const glow = (colour: string | ExpressionSpecification, opacity = 0.3) => ({
    'line-color': colour, 'line-opacity': opacity, 'line-width': byZoom(16, 34), 'line-blur': byZoom(5, 10),
  });
  const concernColour = ['interpolate', ['linear'], ['get', 'concern'], ...CONCERN_RAMP.flat()] as ExpressionSpecification;
  map.addLayer({ id: 'concerns-glow', type: 'line', source: 'network', filter: ['has', 'concern'], layout: ROUND,
    paint: glow(concernColour, 0.38) }, labels);
  map.addLayer({ id: 'concerns-case', type: 'line', source: 'network', filter: ['has', 'concern'], layout: ROUND,
    paint: { 'line-color': '#ffffff', 'line-width': byZoom(9, 17) } }, labels);
  map.addLayer({ id: 'concerns', type: 'line', source: 'network', filter: ['has', 'concern'], layout: ROUND,
    paint: {
      'line-color': concernColour,
      'line-width': byZoom(5.5, 12),
    } }, labels);
  map.addSource('concern-pins', { type: 'geojson', data: concernPins(data.network.features) });
  map.addLayer({ id: 'concerns-pins', type: 'circle', source: 'concern-pins', paint: {
    'circle-color': CONCERN_RAMP[2][1], 'circle-radius': ['interpolate', ['linear'], ['zoom'], 12, 5, 16, 8],
    'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2.5,
  } });

  map.addLayer({ id: 'rules-held', type: 'line', source: 'network', filter: ['==', ['get', 'held'], true], layout: ROUND,
    paint: { 'line-color': RULE.held, 'line-width': byZoom(3, 7), 'line-dasharray': [0, 2] } }, labels);
  const ruled = (agrees: boolean): FilterSpecification => ['all', ['has', 'rule'], ['==', ['get', 'osm_ok'], agrees]];
  map.addLayer({ id: 'rules-glow', type: 'line', source: 'network', filter: ruled(true), layout: ROUND,
    paint: glow(RULE.agrees, 0.4) }, labels);
  map.addLayer({ id: 'rules-wrong-glow', type: 'line', source: 'network', filter: ruled(false), layout: ROUND,
    paint: glow(RULE.disagrees, 0.4) }, labels);
  map.addLayer({ id: 'rules-case', type: 'line', source: 'network', filter: ruled(true), layout: ROUND,
    paint: { 'line-color': '#ffffff', 'line-width': byZoom(11, 19) } }, labels);
  map.addLayer({ id: 'rules', type: 'line', source: 'network', filter: ruled(true), layout: ROUND,
    paint: { 'line-color': RULE.agrees, 'line-width': byZoom(7, 14) } }, labels);
  map.addLayer({ id: 'rules-wrong-case', type: 'line', source: 'network', filter: ruled(false), layout: ROUND,
    paint: { 'line-color': '#ffffff', 'line-width': byZoom(11, 19) } }, labels);
  map.addLayer({ id: 'rules-wrong', type: 'line', source: 'network', filter: ruled(false), layout: DASHED,
    paint: { 'line-color': RULE.disagrees, 'line-width': byZoom(7, 14), 'line-dasharray': [1.4, 0.7] } }, labels);
  const pinColour: ExpressionSpecification = ['case', ['get', 'ok'], RULE.agrees, RULE.disagrees];
  map.addSource('rule-pins', { type: 'geojson', data: rulePins(data.network.features) });
  map.addLayer({ id: 'rules-pins', type: 'circle', source: 'rule-pins', paint: {
    'circle-color': pinColour, 'circle-radius': ['interpolate', ['linear'], ['zoom'], 12, 5, 16, 8],
    'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2.5,
  } });
  if (online) {
    map.addLayer({ id: 'rules-label', type: 'symbol', source: 'rule-pins',
      layout: {
        'text-field': ['get', 'name'], 'text-font': ['Noto Sans Bold'], 'text-size': 11.5, 'text-anchor': 'left',
        'text-offset': [0.9, 0], 'text-optional': true,
      },
      paint: { 'text-color': ['case', ['get', 'ok'], '#3b2f8f', '#9b2424'], 'text-halo-color': '#ffffff', 'text-halo-width': 1.8 } });
    map.addLayer({ id: 'concerns-label', type: 'symbol', source: 'concern-pins',
      layout: {
        'text-field': ['get', 'name'], 'text-font': ['Noto Sans Bold'], 'text-size': 11.5, 'text-anchor': 'left',
        'text-offset': [0.9, 0], 'text-optional': true,
      },
      paint: { 'text-color': '#9a3412', 'text-halo-color': '#ffffff', 'text-halo-width': 1.8 } });
  }

  for (const [id, colour, layout, dash] of [['design-bus', DESIGN.reallocate, ROUND, null],
                                            ['design-closed', DESIGN.close, DASHED, [1.2, 0.8]]] as const) {
    map.addLayer({ id: `${id}-case`, type: 'line', source: 'network', filter: NOTHING, layout: ROUND,
      paint: { 'line-color': '#ffffff', 'line-width': byZoom(8, 17) } }, labels);
    map.addLayer({ id, type: 'line', source: 'network', filter: NOTHING, layout,
      paint: { 'line-color': colour, 'line-width': byZoom(4.5, 11), ...(dash ? { 'line-dasharray': [...dash] } : {}) } }, labels);
  }

  map.addLayer({ id: 'network-hit', type: 'line', source: 'network', filter: ['!=', ['get', 'name'], ''],
    paint: { 'line-color': '#000000', 'line-opacity': 0, 'line-width': 16 } });

  if (data.crossings) {
    map.addSource('crossings', { type: 'geojson', data: data.crossings });
    const size = (low: number, high: number): ExpressionSpecification =>
      ['interpolate', ['linear'], ['get', 'delay'], 0, low, 60, high];
    map.addLayer({ id: 'crossings', type: 'circle', source: 'crossings', paint: {
      'circle-color': SERIES.crossings, 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1.5,
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 13, size(1.5, 4), 17, size(4, 11)],
    } });
    map.addLayer({ id: 'design-signal', type: 'circle', source: 'crossings', filter: NO_NODE, paint: {
      'circle-color': DESIGN.signalise_crossing, 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 3,
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 13, 6, 17, 12],
    } });
  }
  if (data.counts) {
    map.addSource('counts', { type: 'geojson', data: data.counts });
    map.addLayer({ id: 'counts', type: 'circle', source: 'counts', paint: {
      'circle-color': SERIES.counts, 'circle-radius': 7, 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2,
    } });
    if (online) {
      map.addLayer({ id: 'counts-label', type: 'symbol', source: 'counts',
        layout: {
          'text-field': ['concat', ['number-format', ['/', ['get', 'aadf'], 1000], { 'max-fraction-digits': 1 }], 'k/day'],
          'text-font': ['Noto Sans Bold'], 'text-size': 12, 'text-offset': [0, 1.1], 'text-anchor': 'top',
        },
        paint: { 'text-color': INK, 'text-halo-color': '#ffffff', 'text-halo-width': 1.6 } });
    }
  }
  if (data.designs) {
    map.addSource('design-pins', { type: 'geojson', data: changePins(data).pins });
    map.addLayer({ id: 'design-pins', type: 'circle', source: 'design-pins', paint: {
      'circle-color': ['match', ['get', 'kind'], 'close', DESIGN.close, 'reallocate', DESIGN.reallocate, DESIGN.signalise_crossing],
      'circle-stroke-color': '#ffffff', 'circle-stroke-width': 3,
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 12, 8, 16, 12],
    } });
    if (online) {
      map.addLayer({ id: 'design-label', type: 'symbol', source: 'design-pins', layout: {
        'text-field': ['get', 'label'], 'text-font': ['Noto Sans Bold'], 'text-size': 12.5, 'text-offset': [0, 1.3],
        'text-anchor': 'top', 'text-max-width': 12,
      }, paint: { 'text-color': INK, 'text-halo-color': '#ffffff', 'text-halo-width': 2 } });
    }
  }
}

export default function MapView({ data, speeds, city, visible, selected, design, bin, onSelect, onChange, onError, apiRef }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const onSelectRef = useRef(onSelect);
  const onChangeRef = useRef(onChange);
  const onErrorRef = useRef(onError);
  const shownDesign = useRef<DesignView | null>(null);
  const [ready, setReady] = useState(false);
  onSelectRef.current = onSelect;
  onChangeRef.current = onChange;
  onErrorRef.current = onError;

  useEffect(() => {
    let map: MLMap | null = null;
    let cancelled = false;
    const noWebgl = (detail: string) =>
      onErrorRef.current(`The map can't draw in this browser (${detail}). It needs WebGL 2: try Chrome, Edge or Firefox with hardware acceleration on.`);
    basemapStyle().then(({ style, online }) => {
      if (cancelled || !container.current) return;
      let m: MLMap;
      try {
        m = new MLMap({
          container: container.current, style, bounds: widen(data.meta.bbox) as LngLatBoundsLike,
          fitBoundsOptions: { padding: padding() }, maxBounds: EDINBURGH as LngLatBoundsLike,
          attributionControl: { compact: true }, dragRotate: false, pitchWithRotate: false, touchPitch: false,
        });
      } catch (e) {
        noWebgl(message(e));
        return;
      }
      map = m;
      m.touchZoomRotate.disableRotation();
      m.on('error', (e) => {
        const text = message(e.error);
        if (/webgl/i.test(text)) noWebgl(text);
        else console.warn('map:', text);
      });
      m.on('webglcontextlost', () => onErrorRef.current('The map lost its graphics context. Reload the page to redraw it.'));
      m.on('load', () => {
        try {
          addLayers(m, data, online);
        } catch (e) {
          onErrorRef.current(`Some map layers couldn't be drawn: ${message(e)}`);
        }
        let hovered = '';
        m.on('mousemove', 'network-hit', (e) => {
          const name = String(e.features?.[0]?.properties?.name ?? '');
          m.getCanvas().style.cursor = 'pointer';
          if (name !== hovered) m.setFilter('hover', ['==', ['get', 'name'], (hovered = name)]);
        });
        m.on('mouseleave', 'network-hit', () => {
          m.getCanvas().style.cursor = '';
          m.setFilter('hover', NOTHING);
          hovered = '';
        });
        m.on('mouseenter', 'design-pins', () => { m.getCanvas().style.cursor = 'pointer'; });
        m.on('mouseleave', 'design-pins', () => { m.getCanvas().style.cursor = ''; });
        m.on('click', (e) => {
          const pin = m.getLayer('design-pins') ? m.queryRenderedFeatures(e.point, { layers: ['design-pins'] })[0] : undefined;
          if (pin) {
            onChangeRef.current?.(String(pin.properties.id));
            return;
          }
          const hit = m.queryRenderedFeatures(e.point, { layers: ['network-hit'] })[0];
          onSelectRef.current(hit ? String(hit.properties.name) : null);
        });
        const fit = (bbox: BBox) => m.fitBounds(bbox as LngLatBoundsLike, { padding: padding(), maxZoom: 17.5, duration: 900 });
        apiRef.current = {
          showStreet: (name) => data.streets[name] && fit(data.streets[name].bbox),
          reset: () => fit(widen(data.meta.bbox)),
          zoomIn: () => m.zoomIn(),
          zoomOut: () => m.zoomOut(),
        };
        mapRef.current = m;
        setReady(true);
      });
    });
    return () => {
      cancelled = true;
      map?.remove();
      mapRef.current = null;
      apiRef.current = null;
      setReady(false);
    };
  }, [data, apiRef]);

  useEffect(() => {
    const m = mapRef.current;
    if (!ready || !m) return;
    for (const [layer, ids] of Object.entries(GROUPS) as [LayerId, string[]][]) {
      for (const id of ids) {
        if (m.getLayer(id)) m.setLayoutProperty(id, 'visibility', visible[layer] ? 'visible' : 'none');
      }
    }
  }, [ready, visible]);

  useEffect(() => {
    const m = mapRef.current;
    if (!ready || !m?.getLayer('selected')) return;
    m.setFilter('selected', selected ? ['==', ['get', 'name'], selected] : NOTHING);
  }, [ready, selected]);

  useEffect(() => {
    const m = mapRef.current;
    if (!ready || !m) return;
    const streets = (names: string[] = []): FilterSpecification =>
      names.length ? ['all', ['==', ['get', 'car'], true], ['in', ['get', 'name'], ['literal', names]]] : NOTHING;
    for (const [ids, names] of [[['design-bus-case', 'design-bus'], design?.buses],
                                [['design-closed-case', 'design-closed'], design?.closed]] as const) {
      for (const id of ids) if (m.getLayer(id)) m.setFilter(id, streets(names ? [...names] : []));
    }
    if (m.getLayer('design-signal')) {
      m.setFilter('design-signal', design?.signals.length ? ['in', ['get', 'node'], ['literal', design.signals]] : NO_NODE);
    }
    const entering = Boolean(design) && !shownDesign.current;
    shownDesign.current = design ?? null;
    const bbox = entering ? changePins(data).bbox : null;
    if (bbox) m.fitBounds(bbox as LngLatBoundsLike, { padding: padding(), maxZoom: 15, duration: 900 });
  }, [ready, data, design]);

  useEffect(() => {
    if (!ready) return;
    mapRef.current?.getSource<GeoJSONSource>('speeds')?.setData(speedFeatures(data.network.features, speeds, bin));
    mapRef.current?.getSource<GeoJSONSource>('city-speeds')
      ?.setData(speedFeatures(city?.network.features ?? [], city?.speeds ?? null, bin));
  }, [ready, data, speeds, city, bin]);

  return <div ref={container} className="map" />;
}
