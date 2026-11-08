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
import type { Feature, FeatureCollection, LineString, Polygon } from 'geojson';
import { slowestAt, type BBox, type Data, type LayerId, type Speeds } from './data';
import { basemapStyle } from './basemap';

setWorkerUrl(workerUrl);

export const SERIES = { traffic: '#2a78d6', crossings: '#4b463f', counts: '#1baf7a' } as const;
export const SPEED_RAMP: [number, string][] = [
  [0, '#5c1d06'], [6, '#9b3510'], [12, '#d9541f'], [20, '#f08a5b'], [32, '#f6b796'],
];
export const TRAFFIC_RAMP: [number, string][] = [
  [0, '#b7d3f6'], [0.4, '#6da7ec'], [0.7, '#2a78d6'], [0.9, '#1c5cab'], [1.1, '#0d366b'],
];
export const RULE = { agrees: '#6250d6', disagrees: '#d03b3b', held: '#8d877d' } as const;
export const CONCERN_RAMP: [number, string][] = [[0, '#f3c5d7'], [1, '#d55181'], [5, '#a8325f'], [20, '#6e1d3d']];
const INK = '#1f1d1a';
const HIGHLIGHT = '#ffc629';
const HIGHLIGHT_SOFT = '#ffe7a3';
const EDINBURGH: BBox = [-3.45, 55.83, -2.95, 56.02];
const NOTHING: FilterSpecification = ['==', ['get', 'name'], '\u0000'];
const ROUND = { 'line-cap': 'round', 'line-join': 'round' } as const;
const DASHED = { 'line-cap': 'butt', 'line-join': 'round' } as const;

const GROUPS: Partial<Record<LayerId, string[]>> = {
  rules: ['rules-held', 'rules-case', 'rules', 'rules-wrong-case', 'rules-wrong'],
  concerns: ['concerns'],
  speeds: ['city-speeds-case', 'city-speeds', 'speeds-case', 'speeds', 'speeds-predicted'],
  traffic: ['traffic-case', 'traffic'],
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
  visible: Record<LayerId, boolean>;
  selected: string | null;
  bin: number;
  onSelect(name: string | null): void;
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
  map.addLayer({ id: 'study-casing', type: 'line', source: 'study',
    paint: { 'line-color': '#ffffff', 'line-width': 6, 'line-opacity': 0.9 } }, labels);
  map.addLayer({ id: 'study-outline', type: 'line', source: 'study',
    paint: { 'line-color': INK, 'line-width': 2.2, 'line-dasharray': [3, 2] } }, labels);

  map.addLayer({ id: 'hover', type: 'line', source: 'network', filter: NOTHING, layout: ROUND,
    paint: { 'line-color': HIGHLIGHT_SOFT, 'line-blur': 2, 'line-width': byZoom(10, 24) } }, labels);
  map.addLayer({ id: 'selected', type: 'line', source: 'network', filter: NOTHING, layout: ROUND,
    paint: { 'line-color': HIGHLIGHT, 'line-blur': 0.5, 'line-width': byZoom(9, 22) } }, labels);

  const hasFlow: FilterSpecification = ['>', ['coalesce', ['get', 'flow'], 0], 0];
  map.addLayer({ id: 'traffic-case', type: 'line', source: 'network', filter: hasFlow, layout: ROUND,
    paint: { 'line-color': '#ffffff', 'line-width': trafficWidth(2) } }, labels);
  map.addLayer({ id: 'traffic', type: 'line', source: 'network', filter: hasFlow, layout: ROUND,
    paint: {
      'line-color': ['interpolate', ['linear'], ['get', 'vc'], ...TRAFFIC_RAMP.flat()] as ExpressionSpecification,
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

  map.addLayer({ id: 'concerns', type: 'line', source: 'network', filter: ['has', 'concern'], layout: ROUND,
    paint: {
      'line-color': ['interpolate', ['linear'], ['get', 'concern'], ...CONCERN_RAMP.flat()] as ExpressionSpecification,
      'line-width': byZoom(3, 12), 'line-opacity': 0.85,
    } }, labels);

  map.addLayer({ id: 'rules-held', type: 'line', source: 'network', filter: ['==', ['get', 'held'], true], layout: DASHED,
    paint: { 'line-color': RULE.held, 'line-width': byZoom(2.5, 7), 'line-dasharray': [2, 1.4] } }, labels);
  const ruled = (agrees: boolean): FilterSpecification => ['all', ['has', 'rule'], ['==', ['get', 'osm_ok'], agrees]];
  map.addLayer({ id: 'rules-case', type: 'line', source: 'network', filter: ruled(true), layout: ROUND,
    paint: { 'line-color': '#ffffff', 'line-width': byZoom(6, 15) } }, labels);
  map.addLayer({ id: 'rules', type: 'line', source: 'network', filter: ruled(true), layout: ROUND,
    paint: { 'line-color': RULE.agrees, 'line-width': byZoom(3, 10) } }, labels);
  map.addLayer({ id: 'rules-wrong-case', type: 'line', source: 'network', filter: ruled(false), layout: ROUND,
    paint: { 'line-color': '#ffffff', 'line-width': byZoom(7, 17) } }, labels);
  map.addLayer({ id: 'rules-wrong', type: 'line', source: 'network', filter: ruled(false), layout: DASHED,
    paint: { 'line-color': RULE.disagrees, 'line-width': byZoom(4, 11), 'line-dasharray': [1.4, 0.7] } }, labels);

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
}

export default function MapView({ data, visible, selected, bin, onSelect, onError, apiRef }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const onSelectRef = useRef(onSelect);
  const onErrorRef = useRef(onError);
  const [ready, setReady] = useState(false);
  onSelectRef.current = onSelect;
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
          container: container.current, style, bounds: data.meta.bbox as LngLatBoundsLike,
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
        m.on('click', (e) => {
          const hit = m.queryRenderedFeatures(e.point, { layers: ['network-hit'] })[0];
          onSelectRef.current(hit ? String(hit.properties.name) : null);
        });
        const fit = (bbox: BBox) => m.fitBounds(bbox as LngLatBoundsLike, { padding: padding(), maxZoom: 17.5, duration: 900 });
        apiRef.current = {
          showStreet: (name) => data.streets[name] && fit(data.streets[name].bbox),
          reset: () => fit(data.meta.bbox),
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
    if (!ready) return;
    mapRef.current?.getSource<GeoJSONSource>('speeds')?.setData(speedFeatures(data.network.features, data.speeds, bin));
    mapRef.current?.getSource<GeoJSONSource>('city-speeds')
      ?.setData(speedFeatures(data.city?.network.features ?? [], data.city?.speeds ?? null, bin));
  }, [ready, data, bin]);

  return <div ref={container} className="map" />;
}
