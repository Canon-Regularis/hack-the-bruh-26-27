import type { FeatureCollection, LineString, Point } from 'geojson';

export type Basis = 'observed' | 'estimated' | 'modelled' | 'predicted' | 'synthetic' | 'verified';
export type LayerId = 'rules' | 'concerns' | 'speeds' | 'traffic' | 'crossings' | 'counts' | 'designs';
export type BBox = [number, number, number, number];

export interface LayerMeta {
  label: string;
  basis: Basis;
  note: string;
  hour?: string;
  applied?: number;
  osm_wrong?: number;
  abstained?: number;
  share_predicted?: number;
  window?: [string, string];
  mae_kmh?: number;
  baseline_mae_kmh?: number;
  skill?: number;
  time_mae_kmh?: number;
  time_skill?: number;
  time_baseline_mae_kmh?: number;
  interval_kmh?: number;
  interval_target?: number;
  interval_coverage?: number;
  city_edges?: number;
}

export interface Meta {
  bbox: BBox;
  layers: Partial<Record<LayerId, LayerMeta>>;
}

export interface Segment {
  id: string;
  rev: string | null;
  name: string;
  cls: string;
  car: boolean;
  oneway: boolean;
  len: number;
  lanes: number;
  limit: number;
  flow?: number;
  vc?: number;
  f_t?: number;
  f_t0?: number;
  r_t?: number;
  r_t0?: number;
  rule?: string;
  osm_ok?: boolean;
  concern?: number;
  concern_theme?: string;
  held?: boolean;
}

export interface Crossing {
  node: string;
  edges: string[];
  control: string;
  width: number;
  delay: number;
  los: string;
  flow: number | null;
}

export interface CountPoint {
  id: string;
  road: string;
  aadf: number;
  year: number;
  counted: boolean;
  edge: string;
}

export interface Clause {
  id: string;
  streets: string[];
  restriction: string;
  quote: string;
  document: string;
  page: number;
  times: string;
  exemptions: string[] | string;
  status: 'applied' | 'keep_open' | 'abstained';
  reason: string | null;
  edges: number;
  osm_wrong_edges: number;
  length_m: number;
  osm_wrong_m: number;
  osm_says: string[];
  in_force?: string | null;
}

export interface ConcernRecord {
  theme: string;
  count: number;
  source: string;
}

export type Streets = Record<string, { bbox: BBox; segments: number }>;

/** Signal's prediction table: per directed edge, km/h per 15-minute bin (s) and "1"/"0" for observed/predicted (o). */
export interface Speeds {
  bins: string[];
  edges: Record<string, { s: (number | null)[]; o: string }>;
}

export interface SpeedReading {
  speed: number;
  observed: boolean;
}

/** Bus-observed speeds outside the study area: the segments buses moved on, and their speeds in the study bins. */
export interface CitySpeeds {
  network: FeatureCollection<LineString, { id: string; rev: string | null; name: string }>;
  speeds: Speeds;
}

export interface Data {
  meta: Meta;
  network: FeatureCollection<LineString, Segment>;
  streets: Streets;
  crossings: FeatureCollection<Point, Crossing> | null;
  counts: FeatureCollection<Point, CountPoint> | null;
  rules: Clause[] | null;
  concerns: Record<string, ConcernRecord[]> | null;
  speeds: Speeds | null;
  city: CitySpeeds | null;
  cityTraffic: FeatureCollection<LineString, { id: string; name: string; flow: number; vc: number }> | null;
  warnings: string[];
}

export interface StreetInfo {
  name: string;
  segments: Segment[];
  crossings: Crossing[];
  counts: CountPoint[];
  rules: Clause[];
  concerns: ConcernRecord[];
}

export class DataError extends Error {}

type Check = (value: unknown) => boolean;

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isCollection: Check = (v) => isObject(v) && Array.isArray(v.features);
const isMeta: Check = (v) => isObject(v) && Array.isArray(v.bbox) && v.bbox.length === 4 && isObject(v.layers);
const isRules: Check = (v) => isObject(v) && Array.isArray(v.clauses);
const isSpeeds: Check = (v) => isObject(v) && Array.isArray(v.bins) && isObject(v.edges);

async function get<T>(name: string, check: Check): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/data/${name}`);
  } catch {
    throw new DataError(`${name}: the server could not be reached`);
  }
  if (!res.ok) throw new DataError(`${name}: HTTP ${res.status}`);
  let value: unknown;
  try {
    value = await res.json();
  } catch {
    throw new DataError(`${name}: not valid JSON, so the file is probably missing`);
  }
  if (!check(value)) throw new DataError(`${name}: not in the expected format`);
  return value as T;
}

const OPTIONAL: { layer: LayerId; file: string; check: Check; pick: (v: never) => unknown }[] = [
  { layer: 'crossings', file: 'crossings.geojson', check: isCollection, pick: (v) => v },
  { layer: 'counts', file: 'counts.geojson', check: isCollection, pick: (v) => v },
  { layer: 'rules', file: 'rules.json', check: isRules, pick: (v: { clauses: Clause[] }) => v.clauses },
  { layer: 'concerns', file: 'concerns.json', check: isObject, pick: (v) => v },
  { layer: 'speeds', file: 'speeds.json', check: isSpeeds, pick: (v) => v },
];

/** Load the exported files. The map, network and street list are required and throw DataError;
 * an optional layer that fails to load is dropped from meta and reported in warnings instead. */
export async function loadData(): Promise<Data> {
  const meta = await get<Meta>('meta.json', isMeta);
  const [network, streets, ...optional] = await Promise.all([
    get<Data['network']>('network.geojson', isCollection),
    get<Streets>('streets.json', isObject),
    ...OPTIONAL.map(({ layer, file, check, pick }) =>
      meta.layers[layer] ? get<never>(file, check).then(pick, (e: Error) => e) : Promise.resolve(null)),
  ]);
  const warnings: string[] = [];
  const loaded = OPTIONAL.map(({ layer }, i) => {
    const value = optional[i];
    if (!(value instanceof Error)) return value;
    warnings.push(`${meta.layers[layer]!.label} left out: ${value.message}`);
    delete meta.layers[layer];
    return null;
  });
  const [crossings, counts, rules, concerns, speeds] =
    loaded as [Data['crossings'], Data['counts'], Data['rules'], Data['concerns'], Data['speeds']];
  let city: CitySpeeds | null = null;
  if (speeds && meta.layers.speeds?.city_edges) {
    try {
      const [cityNetwork, citySpeeds] = await Promise.all([
        get<CitySpeeds['network']>('city_network.geojson', isCollection),
        get<Speeds>('city_speeds.json', isSpeeds),
      ]);
      city = { network: cityNetwork, speeds: citySpeeds };
    } catch (e) {
      warnings.push(`City-wide bus speeds left out: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  let cityTraffic: Data['cityTraffic'] = null;
  if (meta.layers.traffic?.city_edges) {
    try {
      cityTraffic = await get<NonNullable<Data['cityTraffic']>>('city_traffic.geojson', isCollection);
    } catch (e) {
      warnings.push(`City-wide car traffic left out: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return { meta, network, streets, crossings, counts, rules, concerns, speeds, city, cityTraffic, warnings };
}

/** The slowest reading in a bin among the given directed edges, or null if none of them has one. */
export function slowestAt(speeds: Speeds, ids: (string | null)[], bin: number): SpeedReading | null {
  let slowest: SpeedReading | null = null;
  for (const id of ids) {
    const edge = id ? speeds.edges[id] : undefined;
    const speed = edge?.s[bin];
    if (edge && speed != null && (!slowest || speed < slowest.speed)) slowest = { speed, observed: edge.o[bin] === '1' };
  }
  return slowest;
}

/** A street's slowest reading in the bin across all its segments and directions, and per bin whether a bus was seen on it. */
export function streetSpeeds(speeds: Speeds, segments: Segment[], bin: number) {
  const ids = segments.flatMap((s) => [s.id, s.rev]);
  const edges = ids.flatMap((id) => (id && speeds.edges[id] ? [speeds.edges[id]] : []));
  return {
    now: slowestAt(speeds, ids, bin),
    seen: speeds.bins.map((_, i) => edges.some((e) => e.o[i] === '1')),
    busRoad: edges.length > 0,
  };
}

/** The bin in which buses were seen on the most edges, so the slider opens on real observations. */
export function busiestBin(speeds: Speeds): number {
  const seen = speeds.bins.map((_, i) => Object.values(speeds.edges).filter((e) => e.o[i] === '1').length);
  return Math.max(0, seen.indexOf(Math.max(...seen)));
}

export function indexStreets(data: Data): Map<string, StreetInfo> {
  const byName = new Map<string, StreetInfo>();
  const edgeName = new Map<string, string>();
  for (const { properties: s } of data.network.features) {
    if (!s.name) continue;
    let info = byName.get(s.name);
    if (!info) byName.set(s.name, (info = { name: s.name, segments: [], crossings: [], counts: [], rules: [], concerns: [] }));
    info.segments.push(s);
    edgeName.set(s.id, s.name);
    if (s.rev) edgeName.set(s.rev, s.name);
  }
  for (const { properties: c } of data.crossings?.features ?? []) {
    for (const name of new Set(c.edges.map((e) => edgeName.get(e)))) {
      if (name) byName.get(name)!.crossings.push(c);
    }
  }
  for (const { properties: p } of data.counts?.features ?? []) {
    const name = edgeName.get(p.edge);
    if (name) byName.get(name)!.counts.push(p);
  }
  for (const clause of data.rules ?? []) {
    for (const name of clause.streets) byName.get(name)?.rules.push(clause);
  }
  for (const [name, records] of Object.entries(data.concerns ?? {})) {
    byName.get(name)?.concerns.push(...records);
  }
  return byName;
}

export interface RuledStreet {
  name: string;
  live: number;
  held: number;
  disagrees: boolean;
}

export function ruledStreets(streets: Map<string, StreetInfo>): RuledStreet[] {
  return [...streets.values()]
    .filter((s) => s.rules.length > 0)
    .map(({ name, rules }) => {
      const live = rules.filter((c) => c.status !== 'abstained');
      return { name, live: live.length, held: rules.length - live.length, disagrees: live.some((c) => c.osm_wrong_edges > 0) };
    })
    .sort((a, b) => b.live - a.live || a.name.localeCompare(b.name));
}

export interface ConcernStreet {
  name: string;
  mentions: number;
  themes: string[];
}

export function concernStreets(streets: Map<string, StreetInfo>): ConcernStreet[] {
  return [...streets.values()]
    .filter((s) => s.concerns.length > 0)
    .map(({ name, concerns }) => ({
      name,
      mentions: concerns.reduce((n, c) => n + c.count, 0),
      themes: [...new Set(concerns.map((c) => c.theme))],
    }))
    .sort((a, b) => b.mentions - a.mentions || a.name.localeCompare(b.name));
}
