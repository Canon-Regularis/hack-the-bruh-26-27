import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DataError, busiestBin, concernStreets, indexStreets, loadData, ruledStreets, slowestAt, streetSpeeds,
  type Clause, type Segment, type Speeds,
} from './data';

const segment = (id: string, rev: string | null, name: string) => ({
  type: 'Feature',
  geometry: { type: 'LineString', coordinates: [[0, 0], [1, 1]] },
  properties: { id, rev, name, cls: 'tertiary', car: true, oneway: false, len: 10, lanes: 1, limit: 30 },
});

const clause = (id: string, streets: string[], status: Clause['status'] = 'applied'): Clause => ({
  id, streets, restriction: 'no_motor_vehicles', quote: 'Q', document: 'order', page: 8, times: 'at any time',
  exemptions: [], status, reason: status === 'abstained' ? 'exemptions are not listed' : null, edges: 2,
  osm_wrong_edges: 0, length_m: 10, osm_wrong_m: 0, osm_says: [],
});

const layer = (label: string) => ({ label, basis: 'modelled', note: '' });

function files(): Record<string, unknown> {
  return {
    'meta.json': {
      bbox: [-3.2, 55.94, -3.15, 55.96],
      layers: { crossings: layer('Pedestrian crossing delay'), counts: layer('DfT traffic counts'),
                rules: layer('Legal restrictions'), concerns: layer('Consultation concerns') },
    },
    'network.geojson': { type: 'FeatureCollection', features: [segment('0', '1', 'Lawnmarket'), segment('2', null, 'Castlehill'), segment('3', null, '')] },
    'streets.json': { Lawnmarket: { bbox: [0, 0, 1, 1], segments: 1 } },
    'crossings.geojson': { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: { type: 'Point', coordinates: [0, 0] },
      properties: { node: '9', edges: ['1', '2'], control: 'zebra', width: 6, delay: 3, los: 'A', flow: null } }] },
    'counts.geojson': { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: { type: 'Point', coordinates: [0, 0] },
      properties: { id: '7', road: 'A1', aadf: 900, year: 2025, counted: true, edge: '2' } }] },
    'rules.json': { clauses: [clause('a', ['Lawnmarket']), clause('b', ['Lawnmarket'], 'abstained'), clause('c', ['Nowhere'])] },
    'concerns.json': { Lawnmarket: [{ theme: 'noise', count: 2, source: 'Report p.3' }], Nowhere: [{ theme: 'other', count: 1, source: '' }] },
  };
}

type Reply = unknown | { status: number } | { html: true } | Error;

function serve(overrides: Record<string, Reply> = {}) {
  const all = { ...files(), ...overrides };
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const reply = all[url.replace('/data/', '')];
    if (reply instanceof Error) throw reply;
    if (reply === undefined || (typeof reply === 'object' && reply !== null && 'html' in reply)) {
      return new Response('<!doctype html><html></html>', { status: 200, headers: { 'content-type': 'text/html' } });
    }
    if (typeof reply === 'object' && reply !== null && 'status' in reply && Object.keys(reply).length === 1) {
      return new Response('', { status: (reply as { status: number }).status });
    }
    return new Response(JSON.stringify(reply), { status: 200 });
  }));
}

afterEach(() => vi.unstubAllGlobals());

describe('loadData', () => {
  it('loads every layer meta lists', async () => {
    serve();
    const data = await loadData();
    expect(data.warnings).toEqual([]);
    expect(Object.keys(data.meta.layers)).toEqual(['crossings', 'counts', 'rules', 'concerns']);
    expect(data.rules?.map((c) => c.id)).toEqual(['a', 'b', 'c']);
    expect(data.concerns?.Lawnmarket).toHaveLength(1);
  });

  it('drops an optional layer that fails and says why, keeping the rest', async () => {
    serve({ 'crossings.geojson': { status: 404 } });
    const data = await loadData();
    expect(data.crossings).toBeNull();
    expect(data.meta.layers.crossings).toBeUndefined();
    expect(data.counts?.features).toHaveLength(1);
    expect(data.warnings).toEqual(['Pedestrian crossing delay left out: crossings.geojson: HTTP 404']);
  });

  it('treats the dev server HTML page as a missing file', async () => {
    serve({ 'rules.json': { html: true } });
    const data = await loadData();
    expect(data.rules).toBeNull();
    expect(data.warnings[0]).toBe('Legal restrictions left out: rules.json: not valid JSON, so the file is probably missing');
  });

  it('rejects an optional file in the wrong shape', async () => {
    serve({ 'rules.json': { rows: [] } });
    const data = await loadData();
    expect(data.meta.layers.rules).toBeUndefined();
    expect(data.warnings[0]).toBe('Legal restrictions left out: rules.json: not in the expected format');
  });

  it('only fetches the optional files meta lists', async () => {
    const only = files();
    (only['meta.json'] as { layers: object }).layers = {};
    serve(only);
    const data = await loadData();
    expect([data.crossings, data.counts, data.rules, data.concerns]).toEqual([null, null, null, null]);
    expect(vi.mocked(fetch).mock.calls.map(([url]) => url)).toEqual(['/data/meta.json', '/data/network.geojson', '/data/streets.json']);
  });

  it('fails with a DataError when a required file is missing', async () => {
    serve({ 'network.geojson': { status: 404 } });
    await expect(loadData()).rejects.toThrow(new DataError('network.geojson: HTTP 404'));
  });

  it('fails with a DataError when meta is malformed', async () => {
    serve({ 'meta.json': { layers: {} } });
    await expect(loadData()).rejects.toBeInstanceOf(DataError);
  });

  it('fails with a DataError when the server is down', async () => {
    serve({ 'meta.json': new TypeError('Failed to fetch') });
    await expect(loadData()).rejects.toThrow('meta.json: the server could not be reached');
  });
});

describe('indexStreets', () => {
  it('gives each named street its segments, crossings, counts, clauses and concerns', async () => {
    serve();
    const streets = indexStreets(await loadData());
    expect([...streets.keys()]).toEqual(['Lawnmarket', 'Castlehill']);
    const lawnmarket = streets.get('Lawnmarket')!;
    expect(lawnmarket.rules.map((c) => c.id)).toEqual(['a', 'b']);
    expect(lawnmarket.concerns).toEqual([{ theme: 'noise', count: 2, source: 'Report p.3' }]);
    expect(lawnmarket.crossings.map((c) => c.node)).toEqual(['9']);
    const castlehill = streets.get('Castlehill')!;
    expect(castlehill.crossings.map((c) => c.node)).toEqual(['9']);
    expect(castlehill.counts.map((p) => p.id)).toEqual(['7']);
    expect(castlehill.rules).toEqual([]);
  });
});

const SPEEDS: Speeds = {
  bins: ['2026-11-07T21:00:00+00:00', '2026-11-07T21:15:00+00:00', '2026-11-07T21:30:00+00:00'],
  edges: {
    '0': { s: [12, 8, null], o: '100' },
    '1': { s: [6, 14, 9], o: '011' },
    '2': { s: [20, 20, 20], o: '000' },
    '3': { s: [30, 30, 30], o: '010' },
  },
};

function withSpeeds(extra: Record<string, Reply> = {}): Record<string, Reply> {
  const all = files();
  (all['meta.json'] as { layers: Record<string, unknown> }).layers.speeds = layer('Bus-observed speed');
  return { ...all, 'speeds.json': SPEEDS, ...extra };
}

const seg = (id: string, rev: string | null) => segment(id, rev, 'Street').properties as Segment;

describe('bus speeds', () => {
  it('loads the speed table when meta lists it', async () => {
    serve(withSpeeds());
    const data = await loadData();
    expect(data.speeds).toEqual(SPEEDS);
    expect(data.warnings).toEqual([]);
  });

  it('drops a speed table in the wrong shape and says why', async () => {
    serve(withSpeeds({ 'speeds.json': { bins: 'all night' } }));
    const data = await loadData();
    expect(data.speeds).toBeNull();
    expect(data.meta.layers.speeds).toBeUndefined();
    expect(data.warnings).toEqual(['Bus-observed speed left out: speeds.json: not in the expected format']);
  });

  it('picks the slower direction and says whether a bus saw it', () => {
    expect(slowestAt(SPEEDS, ['0', '1'], 0)).toEqual({ speed: 6, observed: false });
    expect(slowestAt(SPEEDS, ['0', '1'], 1)).toEqual({ speed: 8, observed: false });
    expect(slowestAt(SPEEDS, ['0', '1'], 2)).toEqual({ speed: 9, observed: true });
  });

  it('has no reading where no direction has a speed', () => {
    expect(slowestAt(SPEEDS, ['0', null], 2)).toBeNull();
    expect(slowestAt(SPEEDS, ['99'], 0)).toBeNull();
  });

  it('summarises a street: its slowest reading now and which quarter-hours a bus saw it', () => {
    expect(streetSpeeds(SPEEDS, [seg('0', '1')], 0)).toEqual({ now: { speed: 6, observed: false }, seen: [true, true, true], busRoad: true });
    expect(streetSpeeds(SPEEDS, [seg('2', null)], 1)).toEqual({ now: { speed: 20, observed: false }, seen: [false, false, false], busRoad: true });
    expect(streetSpeeds(SPEEDS, [seg('8', '9')], 0)).toEqual({ now: null, seen: [false, false, false], busRoad: false });
  });

  it('loads city-wide speeds when meta lists them', async () => {
    const all = withSpeeds({ 'city_network.geojson': { type: 'FeatureCollection', features: [segment('50', '51', 'Leith Walk')] },
      'city_speeds.json': { bins: SPEEDS.bins, edges: { '50': { s: [20, null, null], o: '100' } } } });
    (all['meta.json'] as { layers: { speeds: Record<string, unknown> } }).layers.speeds.city_edges = 1;
    serve(all);
    const data = await loadData();
    expect(data.city?.network.features).toHaveLength(1);
    expect(data.city?.speeds.edges['50'].s[0]).toBe(20);
    expect(data.warnings).toEqual([]);
  });

  it('keeps the study speeds when the city-wide files fail', async () => {
    const all = withSpeeds({ 'city_network.geojson': { status: 404 } });
    (all['meta.json'] as { layers: { speeds: Record<string, unknown> } }).layers.speeds.city_edges = 1;
    serve(all);
    const data = await loadData();
    expect(data.city).toBeNull();
    expect(data.speeds).toEqual(SPEEDS);
    expect(data.warnings).toEqual(['City-wide bus speeds left out: city_network.geojson: HTTP 404']);
  });

  it('does not fetch city-wide files unless meta lists them', async () => {
    serve(withSpeeds());
    const data = await loadData();
    expect(data.city).toBeNull();
    expect(vi.mocked(fetch).mock.calls.map(([url]) => url)).not.toContain('/data/city_speeds.json');
  });

  it('opens on the quarter-hour with the most bus sightings', () => {
    expect(busiestBin(SPEEDS)).toBe(1);
    expect(busiestBin({ bins: [], edges: {} })).toBe(0);
  });
});

describe('ruledStreets', () => {
  it('lists map streets named by a clause, with live and held counts', async () => {
    serve();
    expect(ruledStreets(indexStreets(await loadData()))).toEqual([{ name: 'Lawnmarket', live: 1, held: 1, disagrees: false }]);
  });

  it('puts streets with rules in force first and counts keep-open clauses as live', async () => {
    serve({ 'rules.json': { clauses: [clause('a', ['Castlehill'], 'abstained'), clause('b', ['Lawnmarket']),
      clause('c', ['Lawnmarket'], 'keep_open'), clause('d', ['Nowhere'])] } });
    expect(ruledStreets(indexStreets(await loadData()))).toEqual([
      { name: 'Lawnmarket', live: 2, held: 0, disagrees: false },
      { name: 'Castlehill', live: 0, held: 1, disagrees: false },
    ]);
  });

  it('flags a street where OpenStreetMap disagrees with a rule in force, not with a held one', async () => {
    serve({ 'rules.json': { clauses: [{ ...clause('a', ['Lawnmarket']), osm_wrong_edges: 2 },
      { ...clause('b', ['Castlehill'], 'abstained'), osm_wrong_edges: 1 }] } });
    expect(ruledStreets(indexStreets(await loadData())).map((s) => [s.name, s.disagrees])).toEqual([
      ['Lawnmarket', true], ['Castlehill', false],
    ]);
  });
});

describe('concernStreets', () => {
  it('lists map streets named in consultation material with their total mentions and themes', async () => {
    serve();
    expect(concernStreets(indexStreets(await loadData()))).toEqual([{ name: 'Lawnmarket', mentions: 2, themes: ['noise'] }]);
  });

  it('puts the most mentioned streets first and keeps each theme once', async () => {
    serve({ 'concerns.json': {
      Castlehill: [{ theme: 'rat_running', count: 1, source: '' }],
      Lawnmarket: [{ theme: 'noise', count: 2, source: '' }, { theme: 'parking', count: 1, source: '' }, { theme: 'noise', count: 1, source: '' }],
    } });
    expect(concernStreets(indexStreets(await loadData()))).toEqual([
      { name: 'Lawnmarket', mentions: 4, themes: ['noise', 'parking'] },
      { name: 'Castlehill', mentions: 1, themes: ['rat_running'] },
    ]);
  });
});
