import { badDaySentence, fmtChange, fmtSigned, whatItDoes } from './Designs';
import { fmtDay, fmtPeriod } from './ui';
import { changePins, concernPins, rulePins, widen } from './MapView';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  DataError, busiestBin, concernStreets, indexStreets, loadData, ruledStreets, slowestAt, streetSpeeds, versusToday,
  type Clause, type Designs, type Segment, type Speeds,
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

describe('fmtDay', () => {
  it('names the weekday and date in Edinburgh time', () => {
    expect(fmtDay('2026-11-06T08:00:00+00:00')).toBe('Friday 6 November');
    expect(fmtDay('2026-11-07T23:30:00-01:00')).toBe('Sunday 8 November');
  });
});

describe('speed periods', () => {
  const table = (bins: string[]) => ({ bins, edges: { '0': { s: bins.map(() => 10), o: bins.map(() => '1').join('') } } });
  const day = ['2026-11-06T08:00:00+00:00', '2026-11-06T08:15:00+00:00'];
  const night = ['2026-11-07T19:30:00+00:00'];

  function withPeriods(extra: Record<string, Reply> = {}) {
    const all = files();
    const speeds = { ...layer('Bus-observed speed'), window: [day[0], day[1]], mae_kmh: 7 };
    (all['meta.json'] as { layers: Record<string, unknown> }).layers.speeds = {
      ...speeds, periods: [{ window: speeds.window, mae_kmh: 7 }, { window: [night[0], night[0]], mae_kmh: 8, city_edges: 1 }],
    };
    serve({
      ...all, 'speeds.json': table(day), 'speeds_1.json': table(night),
      'city_network_1.geojson': { type: 'FeatureCollection', features: [] }, 'city_speeds_1.json': table(night), ...extra,
    });
  }

  it('loads each period after the default, with its own meta and city speeds', async () => {
    withPeriods();
    const data = await loadData();
    expect(data.periods.map((p) => p.speeds.bins)).toEqual([day, night]);
    expect(data.periods[1].meta).toMatchObject({ label: 'Bus-observed speed', mae_kmh: 8, window: [night[0], night[0]] });
    expect(data.periods[1].meta.periods).toBeUndefined();
    expect([data.periods[0].city, data.periods[1].city?.speeds.bins]).toEqual([null, night]);
    expect(data.warnings).toEqual([]);
  });

  it('keeps the default period when another fails to load', async () => {
    withPeriods({ 'speeds_1.json': { status: 404 } });
    const data = await loadData();
    expect(data.periods).toHaveLength(1);
    expect(data.warnings).toEqual(['Bus speeds for another period left out: speeds_1.json: HTTP 404']);
  });
});

describe('fmtPeriod', () => {
  it('labels a window by weekday and Edinburgh time, ending 15 minutes after the last bin', () => {
    expect(fmtPeriod(['2026-11-06T08:00:00+00:00', '2026-11-06T08:45:00+00:00'])).toBe('Fri 08:00–09:00');
  });
});

describe('widen', () => {
  it('adds a fifth of the box on every side', () => {
    expect(widen([0, 10, 10, 20])).toEqual([-2, 8, 12, 22]);
  });
});

const MENU = [{ id: 'close-a', kind: 'close', street: 'A Street', reason: 'complaint street', evidence: {} }];

function withDesigns(designs: unknown): Record<string, Reply> {
  const all = files();
  (all['meta.json'] as { layers: Record<string, unknown> }).layers.designs = { ...layer('Design options'), status: 'searching' };
  return { ...all, 'designs.json': designs };
}

describe('designs', () => {
  it('loads the menu while the search runs', async () => {
    serve(withDesigns({ status: 'searching', candidates: MENU }));
    const data = await loadData();
    expect(data.designs).toEqual({ status: 'searching', candidates: MENU });
    expect(data.warnings).toEqual([]);
  });

  it('drops a designs file in the wrong shape and says why', async () => {
    serve(withDesigns({ status: 'searching', menu: MENU }));
    const data = await loadData();
    expect(data.designs).toBeNull();
    expect(data.meta.layers.designs).toBeUndefined();
    expect(data.warnings).toEqual(['Design options left out: designs.json: not in the expected format']);
  });

  it('gives a design\'s scores as % change against doing nothing', () => {
    const designs: Designs = {
      status: 'done', candidates: [],
      designs: [{ mask: 0, candidates: [], feasible: true, mean_j_car: 200, mean_j_ped: 50, j_risk: 400 }],
    };
    const change = versusToday(designs, { mean_j_car: 210, mean_j_ped: 40, j_risk: 400 })!;
    expect(change.car).toBeCloseTo(5);
    expect(change.ped).toBeCloseTo(-20);
    expect(change.risk).toBeCloseTo(0);
    expect(versusToday({ ...designs, designs: [{ mask: 0, candidates: [], feasible: false }] }, { mean_j_car: 1 })).toBeNull();
  });
});

describe('rulePins', () => {
  const seg = (name: string, len: number, props: Partial<Segment>, coords: number[][]) => ({
    type: 'Feature' as const, geometry: { type: 'LineString' as const, coordinates: coords },
    properties: { ...segment('0', null, name).properties, len, ...props },
  });

  it('puts one pin per ruled street at the middle of its longest segment, red if OSM disagrees anywhere', () => {
    const pins = rulePins([
      seg('Lawnmarket', 10, { rule: 'no_motor_vehicles', osm_ok: true }, [[0, 0], [1, 1]]),
      seg('Lawnmarket', 30, { rule: 'no_motor_vehicles', osm_ok: false }, [[5, 5], [6, 6], [7, 7]]),
      seg('Castlehill', 20, { rule: 'one_way', osm_ok: true }, [[2, 2], [3, 3]]),
      seg('Bank Street', 50, {}, [[9, 9], [9, 8]]),
    ]);
    expect(pins.features.map((f) => [f.properties.name, f.properties.ok, f.geometry.coordinates])).toEqual([
      ['Lawnmarket', false, [6, 6]], ['Castlehill', true, [3, 3]],
    ]);
  });
});

describe('concernPins', () => {
  it('puts one pin per street named in consultation material', () => {
    const named = (name: string, concern?: number) => ({
      type: 'Feature' as const, geometry: { type: 'LineString' as const, coordinates: [[0, 0], [1, 1]] },
      properties: { ...segment('0', null, name).properties, ...(concern != null && { concern }) },
    });
    expect(concernPins([named('Cowgate', 1), named('Cowgate', 1), named('Bank Street')]).features.map((f) => f.properties))
      .toEqual([{ name: 'Cowgate', ok: true }]);
  });
});

describe('design wording', () => {
  it('shows small changes to two decimals and a change that rounds to nothing without a sign', () => {
    expect([fmtChange(-0.309), fmtChange(0.061), fmtChange(-0.0004), fmtChange(12.34), fmtChange(null)])
      .toEqual(['−0.31%', '+0.06%', '0.00%', '+12.3%', '–']);
  });

  it('says the bad-day result in plain words', () => {
    const tail = { sentence: '', x_pct: 112.9, y_pct: 112.9, same_design: true, scenario_count: 4 };
    expect(badDaySentence(tail)).toBe("The design that's best on an average day is also the best on the worst day, when car "
      + 'travel time is 2.1× an average day.');
    expect(badDaySentence({ ...tail, same_design: false, y_pct: 40 })).toContain('the design built for bad days loses 40.0%');
  });
});

describe('change pins', () => {
  const line = (name: string, car: boolean, len: number, coords: number[][]) => ({
    type: 'Feature' as const, geometry: { type: 'LineString' as const, coordinates: coords },
    properties: { ...(segment('0', null, name).properties as Segment), car, len },
  });
  const data = {
    network: { type: 'FeatureCollection', features: [
      line('George Street', true, 10, [[0, 0], [1, 1], [2, 2]]), line('George Street', true, 50, [[4, 4], [5, 5], [6, 6]]),
      line('George Street', false, 99, [[9, 9], [9, 9]]),
    ] },
    crossings: { type: 'FeatureCollection', features: [
      { type: 'Feature', geometry: { type: 'Point', coordinates: [7, 1] }, properties: { node: '42' } }] },
    streets: { 'George Street': { bbox: [0, 0, 6, 6], segments: 3 } },
    designs: { status: 'searching', candidates: [
      { id: 'close-george-street', kind: 'close', street: 'George Street', reason: '', evidence: {} },
      { id: 'signal-q', kind: 'signalise_crossing', node: '42', street: 'Q Street', reason: '', evidence: {} },
      { id: 'signal-gone', kind: 'signalise_crossing', node: '99', street: 'Gone', reason: '', evidence: {} },
    ] },
  } as unknown as Parameters<typeof changePins>[0];

  it('pins every change on its longest car segment or its crossing, and boxes the pins and streets', () => {
    const { pins, bbox } = changePins(data);
    expect(pins.features.map((f) => [f.properties.id, f.properties.label, f.geometry.coordinates])).toEqual([
      ['close-george-street', 'George Street · closed', [5, 5]], ['signal-q', 'Q Street · new signals', [7, 1]],
    ]);
    expect(bbox).toEqual([0, 0, 7, 6]);
  });

  it('has no pins and no box without a menu', () => {
    expect(changePins({ ...data, designs: null })).toEqual({ pins: { type: 'FeatureCollection', features: [] }, bbox: null });
  });

  it('says what each change does from the numbers it was picked on', () => {
    expect(whatItDoes({ id: 's', kind: 'signalise_crossing', street: 'Q', reason: '',
      evidence: { delay_s: 60, los: 'F', vehicle_flow_veh_h: 1647.5, signalised_delay_s: 37.4, signalised_los: 'E' } }))
      .toBe('Adds traffic lights so people cross on their own green phase. The modelled wait to cross drops from 60 s '
        + '(level of service F) to 37 s (E), beside 1,648 cars an hour.');
    expect(whatItDoes({ id: 'c', kind: 'close', street: 'George Street', reason: '', evidence: { flow_veh_h: 966 } }))
      .toContain('Stops through traffic on George Street. Today its busiest direction carries 966 cars an hour');
    expect(whatItDoes({ id: 'b', kind: 'reallocate', street: 'Frederick Street', reason: '', evidence: { flow_veh_h: 1525.9, vc: 1.526 } }))
      .toContain('1,526 cars an hour, at 153% of capacity');
  });

  it('shows signed whole numbers, with no sign on zero', () => {
    expect([fmtSigned(93.4), fmtSigned(-1098.2), fmtSigned(-0.3)]).toEqual(['+93', '−1,098', '0']);
  });
});
