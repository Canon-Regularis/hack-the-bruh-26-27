import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { busiestBin, indexStreets, loadData, ruledStreets, type Data, type LayerId, type RuledStreet } from './data';
import MapView, { CONCERN_RAMP, RULE, SERIES, SPEED_RAMP, TRAFFIC_RAMP, type MapApi } from './MapView';
import Inspector from './Inspector';
import { BasisChip, ErrorBoundary, Icon, fmt, fmtPct, fmtTime, reasonLabel, restrictionLabel } from './ui';

const PLAY_STEP_MS = 900;
const SPEED_TOP = SPEED_RAMP.at(-1)![0];

interface SpeedControls {
  bin: number;
  playing: boolean;
  onBin(bin: number): void;
  onPlay(): void;
}

function SpeedKey({ data, speed }: { data: Data; speed: SpeedControls }) {
  const speeds = data.speeds!;
  const meta = data.meta.layers.speeds!;
  const stops = SPEED_RAMP.map(([v, c]) => `${c} ${(v / SPEED_TOP) * 100}%`).join(', ');
  const [start, end] = meta.window ?? [speeds.bins[0], speeds.bins.at(-1)!];
  return (
    <div className="key">
      <div className="slider">
        <button className="icon-btn" onClick={speed.onPlay} aria-label={speed.playing ? 'Pause' : 'Play through the night'}>
          <Icon name={speed.playing ? 'pause' : 'play'} size={18} />
        </button>
        <input type="range" min={0} max={speeds.bins.length - 1} value={speed.bin} aria-label="Quarter-hour"
          onChange={(e) => speed.onBin(Number(e.target.value))} />
        <span className="time">{fmtTime(speeds.bins[speed.bin])}</span>
      </div>
      <div className="ramp" style={{ background: `linear-gradient(90deg, ${stops})` }} />
      <div className="ramp-ticks"><span>0</span><span style={{ left: `${(12 / SPEED_TOP) * 100}%` }}>12</span><span>{SPEED_TOP}+ km/h</span></div>
      <div className="key inline"><span className="swatch" style={{ background: SPEED_RAMP[2][1] }} /><p>A bus was seen in this quarter-hour</p></div>
      <div className="key inline"><span className="swatch dashed" style={{ color: SPEED_RAMP[2][1] }} /><p>Predicted by the model</p></div>
      {data.city && (
        <p>Across the rest of the city, roads show only speeds a bus actually measured; there are no predictions outside the study area.</p>
      )}
      {meta.share_predicted != null && (
        <p className="headline">
          {fmtPct(meta.share_predicted)} of bus-road time is predicted
          {meta.mae_kmh != null && <>, off by {meta.mae_kmh.toFixed(1)} km/h on average on streets the model never saw</>}
          {meta.skill != null && <> ({fmtPct(meta.skill)} better than the best simple guess)</>}.
        </p>
      )}
      {meta.interval_kmh != null && meta.interval_target != null && meta.interval_coverage != null && (
        <p>
          Each prediction comes with a {fmtPct(meta.interval_target)} range of ±{fmt(meta.interval_kmh)} km/h; on streets
          the model never saw, the real speed fell inside it {fmtPct(meta.interval_coverage)} of the time.
        </p>
      )}
      <p>{meta.note}. Darker is slower; the slower direction is shown. {fmtTime(start)}–{fmtTime(end, 15)}, Edinburgh time.</p>
    </div>
  );
}

type ActId = 'rules' | 'complaints' | 'traffic' | 'designs';

const ACTS: { id: ActId; title: string; blurb: string; layers: LayerId[] }[] = [
  { id: 'rules', title: 'Rules', blurb: 'What the traffic orders say, and where OpenStreetMap disagrees', layers: ['rules'] },
  { id: 'complaints', title: 'Complaints', blurb: 'Where consultation concerns cluster, by theme', layers: ['concerns'] },
  { id: 'traffic', title: 'Traffic', blurb: 'Bus-observed speeds, modelled car flows, crossings and counts', layers: ['speeds', 'traffic', 'crossings', 'counts'] },
  { id: 'designs', title: 'Designs', blurb: 'Street designs that obey the rules and hold up on bad days', layers: ['designs'] },
];

const LAYER_TITLE: Record<LayerId, string> = {
  rules: 'Legal restrictions', concerns: 'Consultation concerns', speeds: 'Bus-observed speed',
  traffic: 'Car traffic', crossings: 'Crossing delay', counts: 'DfT traffic counts', designs: 'Design changes',
};

const ALL_LAYERS = Object.keys(LAYER_TITLE) as LayerId[];

function Search({ names, onPick }: { names: string[]; onPick(name: string): void }) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    const starts = names.filter((n) => n.toLowerCase().startsWith(q));
    const contains = names.filter((n) => !n.toLowerCase().startsWith(q) && n.toLowerCase().includes(q));
    return [...starts, ...contains].slice(0, 8);
  }, [query, names]);

  const pick = (name: string) => {
    setQuery(name);
    setOpen(false);
    onPick(name);
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') setActive((i) => Math.min(i + 1, matches.length - 1));
    else if (e.key === 'ArrowUp') setActive((i) => Math.max(i - 1, 0));
    else if (e.key === 'Enter' && matches[active]) pick(matches[active]);
    else if (e.key === 'Escape') setOpen(false);
    else return;
    e.preventDefault();
  };

  return (
    <div className="search">
      <Icon name="search" />
      <input
        value={query}
        placeholder="Search Old Town streets"
        aria-label="Search streets"
        onChange={(e) => { setQuery(e.target.value); setOpen(true); setActive(0); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={onKey}
      />
      {query && (
        <button className="icon-btn" aria-label="Clear search" onClick={() => setQuery('')}><Icon name="close" size={18} /></button>
      )}
      {open && matches.length > 0 && (
        <ul className="results" role="listbox">
          {matches.map((name, i) => (
            <li key={name} role="option" aria-selected={i === active} className={i === active ? 'active' : ''}
              onMouseDown={(e) => { e.preventDefault(); pick(name); }} onMouseEnter={() => setActive(i)}>
              {name}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

interface KeyProps {
  layer: LayerId;
  data: Data;
  ruled: RuledStreet[];
  speed: SpeedControls;
  onPick(name: string): void;
}

function Key({ layer, data, ruled, speed, onPick }: KeyProps) {
  if (layer === 'speeds' && data.speeds) return <SpeedKey data={data} speed={speed} />;
  if (layer === 'traffic') {
    const stops = TRAFFIC_RAMP.map(([v, c]) => `${c} ${(v / 1.1) * 100}%`).join(', ');
    return (
      <div className="key">
        <div className="ramp" style={{ background: `linear-gradient(90deg, ${stops})` }} />
        <div className="ramp-ticks"><span>0%</span><span style={{ left: `${(0.7 / 1.1) * 100}%` }}>70%</span><span>110%+</span></div>
        <p>Load on the busier direction, % of capacity. Thicker lines carry more vehicles per hour.</p>
      </div>
    );
  }
  if (layer === 'crossings') {
    return <div className="key inline"><span className="dot" style={{ background: SERIES.crossings }} /><p>Bigger dot, longer wait to cross</p></div>;
  }
  if (layer === 'counts') {
    return <div className="key inline"><span className="dot" style={{ background: SERIES.counts }} /><p>Count point, labelled with vehicles per day</p></div>;
  }
  if (layer === 'rules') {
    const meta = data.meta.layers.rules!;
    const held = (data.rules ?? []).filter((c) => c.status === 'abstained');
    return (
      <div className="key">
        <div className="key inline"><span className="swatch" style={{ background: RULE.agrees }} /><p>In force, and OpenStreetMap agrees</p></div>
        <div className="key inline"><span className="swatch dashed" style={{ color: RULE.disagrees }} /><p>In force, but OpenStreetMap disagrees</p></div>
        <div className="key inline"><span className="swatch dashed" style={{ color: RULE.held }} /><p>Named in an order but not applied: see the abstention queue</p></div>
        <p>{meta.applied} applied · {meta.osm_wrong} where OpenStreetMap disagrees · {meta.abstained} abstained for review</p>
        {ruled.length > 0 && (
          <ul className="rule-streets" aria-label="Streets named in the orders">
            {ruled.map((s) => (
              <li key={s.name}>
                <button onClick={() => onPick(s.name)}>
                  <strong>{s.name}</strong>
                  <span>{[s.live && `${s.live} in force`, s.held && `${s.held} held`].filter(Boolean).join(' · ')}</span>
                </button>
              </li>
            ))}
          </ul>
        )}
        {held.length > 0 && (
          <details className="held">
            <summary>Abstention queue</summary>
            <ul>
              {held.map((c) => (
                <li key={c.id}><strong>{c.streets.join(', ')}: {restrictionLabel(c.restriction)}</strong><span>{reasonLabel(c.reason, c.restriction)}</span></li>
              ))}
            </ul>
          </details>
        )}
      </div>
    );
  }
  if (layer === 'concerns') {
    return (
      <div className="key">
        <div className="key inline"><span className="swatch" style={{ background: CONCERN_RAMP[1][1] }} /><p>Raised in published consultation material; darker means raised more often</p></div>
        <p>{data.meta.layers.concerns!.note}</p>
      </div>
    );
  }
  return null;
}

export default function App() {
  const [data, setData] = useState<Data | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [act, setAct] = useState<ActId>('traffic');
  const [hidden, setHidden] = useState<Set<LayerId>>(new Set(['crossings']));
  const [selected, setSelected] = useState<string | null>(null);
  const [notices, setNotices] = useState<string[]>([]);
  const [bin, setBin] = useState(0);
  const [playing, setPlaying] = useState(false);
  const api = useRef<MapApi | null>(null);
  const body = useRef<HTMLDivElement>(null);

  const load = useCallback(() => {
    setError(null);
    loadData().then(
      (loaded) => { setData(loaded); setNotices(loaded.warnings); setBin(loaded.speeds ? busiestBin(loaded.speeds) : 0); },
      (e) => setError(e instanceof Error ? e.message : String(e)),
    );
  }, []);
  useEffect(load, [load]);
  const notify = useCallback((text: string) => setNotices((all) => (all.includes(text) ? all : [...all, text])), []);

  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      const n = Number(e.key);
      if (n >= 1 && n <= ACTS.length) setAct(ACTS[n - 1].id);
      if (e.key === 'Escape') setSelected(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const streets = useMemo(() => (data ? indexStreets(data) : new Map()), [data]);
  const names = useMemo(() => [...streets.keys()].sort(), [streets]);
  const ruled = useMemo(() => ruledStreets(streets), [streets]);
  const current = ACTS.find((a) => a.id === act)!;
  const has = (layer: LayerId) => Boolean(data?.meta.layers[layer]);
  const bins = data?.speeds?.bins.length ?? 0;
  useEffect(() => {
    if (!playing || !bins) return;
    const timer = window.setInterval(() => setBin((b) => (b + 1) % bins), PLAY_STEP_MS);
    return () => window.clearInterval(timer);
  }, [playing, bins]);
  const speed: SpeedControls = { bin, playing, onBin: (b) => { setPlaying(false); setBin(b); }, onPlay: () => setPlaying((p) => !p) };
  const visible = useMemo(
    () => Object.fromEntries(ALL_LAYERS.map((l) => [l, current.layers.includes(l) && Boolean(data?.meta.layers[l]) && !hidden.has(l)])) as Record<LayerId, boolean>,
    [current, data, hidden],
  );
  const toggle = (layer: LayerId) =>
    setHidden((h) => { const next = new Set(h); if (next.has(layer)) next.delete(layer); else next.add(layer); return next; });
  const showStreet = (name: string) => { setSelected(name); api.current?.showStreet(name); };
  const info = selected ? streets.get(selected) : undefined;

  useEffect(() => {
    if (selected) body.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }, [selected]);

  return (
    <div className="app">
      <aside className={`sidebar ${info ? 'open' : ''}`}>
        <header className="side-head">
          <div className="brand">
            <span className="brand-mark">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
                <path d="M5 19c5 0 2.5-7 7-7s2-7 7-7" strokeWidth="2.5" strokeDasharray="0.01 4.1" />
                <circle cx="5" cy="19" r="2.4" fill="currentColor" stroke="none" />
                <circle cx="19" cy="5" r="2.4" fill="currentColor" stroke="none" />
              </svg>
            </span>
            <div><strong>Invisible Streets</strong><span>Edinburgh Old Town</span></div>
          </div>
          <Search names={names} onPick={showStreet} />
        </header>

        <nav className="tabs" aria-label="Story">
          {ACTS.map((a, i) => (
            <button key={a.id} aria-pressed={a.id === act} className={a.layers.some(has) ? '' : 'empty'}
              title={`${a.blurb} (press ${i + 1})`} onClick={() => setAct(a.id)}>
              {a.title}
            </button>
          ))}
        </nav>

        <div className="side-body" ref={body}>
          {info && data && (
            <ErrorBoundary key={info.name} fallback={(e) => (
              <aside className="inspector inspector-error" role="alert">
                <p>Couldn't show {info.name}: {e.message}</p>
                <button className="icon-btn" onClick={() => setSelected(null)} aria-label="Close"><Icon name="close" /></button>
              </aside>
            )}>
              <Inspector info={info} meta={data.meta} speeds={data.speeds} bin={bin} onClose={() => setSelected(null)} />
            </ErrorBoundary>
          )}

          <section className="act" aria-label="Layers">
            <p className="eyebrow">Act {ACTS.indexOf(current) + 1} of {ACTS.length}</p>
            <h2>{current.title}</h2>
            <p className="blurb">{current.blurb}</p>
            {current.layers.map((layer) => {
              const meta = data?.meta.layers[layer];
              return (
                <div key={layer} className={`layer ${meta ? '' : 'missing'}`}>
                  <label>
                    <input type="checkbox" disabled={!meta} checked={visible[layer]} onChange={() => toggle(layer)} />
                    <span className="layer-title">{LAYER_TITLE[layer]}</span>
                    {meta ? <BasisChip basis={meta.basis} /> : <span className="soon">No data yet</span>}
                  </label>
                  {visible[layer] && data && <Key layer={layer} data={data} ruled={ruled} speed={speed} onPick={showStreet} />}
                </div>
              );
            })}
          </section>
        </div>

        <p className="disclaimer">Numbers come from code. Nothing here is legally approved.</p>
      </aside>

      <main className="stage">
        {data && <MapView data={data} visible={visible} selected={selected} bin={bin} onSelect={setSelected} onError={notify} apiRef={api} />}

        <div className="card controls">
          <button className="icon-btn" aria-label="Zoom in" onClick={() => api.current?.zoomIn()}><Icon name="plus" /></button>
          <button className="icon-btn" aria-label="Zoom out" onClick={() => api.current?.zoomOut()}><Icon name="minus" /></button>
          <hr />
          <button className="icon-btn" aria-label="Show the whole study area" onClick={() => api.current?.reset()}><Icon name="recentre" /></button>
        </div>

        {notices.length > 0 && (
          <div className="card notice" role="status">
            <ul>{notices.map((n) => <li key={n}>{n}</li>)}</ul>
            <button className="icon-btn" onClick={() => setNotices([])} aria-label="Dismiss"><Icon name="close" size={18} /></button>
          </div>
        )}

        {!data && (
          <div className={`loading card ${error ? 'error' : ''}`} role={error ? 'alert' : 'status'}>
            {error ? (
              <>
                <p>Couldn't load the map data: {error}. From the repo root, run <code>python -m src.api.export</code>.</p>
                <button className="retry" onClick={load}>Try again</button>
              </>
            ) : 'Loading streets…'}
          </div>
        )}
      </main>
    </div>
  );
}
