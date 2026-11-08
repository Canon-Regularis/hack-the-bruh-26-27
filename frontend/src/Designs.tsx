import { versusToday, type Design, type DesignCandidate, type Designs } from './data';
import { DESIGN, type DesignView } from './MapView';
import { Icon, fmt, fmtPct } from './ui';

const ROLES: [string, string][] = [
  ['knee', 'Balanced'], ['car_best', 'Best for cars'], ['pedestrian_best', 'Best on foot'], ['voice_best', 'Best for complaints'],
];

export function changeTitle(c: DesignCandidate) {
  if (c.kind === 'close') return `Close ${c.street} to through traffic`;
  if (c.kind === 'reallocate') return `Bus lane on ${c.street}`;
  return `Signals at the ${c.street ?? 'crossing'} crossing`;
}

/** The streets and crossing nodes a set of candidates changes, for the map. */
export function designView(designs: Designs, ids: string[]): DesignView {
  const picked = designs.candidates.filter((c) => ids.includes(c.id));
  const streets = (kind: DesignCandidate['kind']) => picked.filter((c) => c.kind === kind && c.street).map((c) => c.street!);
  return {
    closed: streets('close'),
    buses: streets('reallocate'),
    signals: picked.filter((c) => c.kind === 'signalise_crossing' && c.node).map((c) => c.node!),
  };
}

/** The mask to show first: the balanced pick, else do nothing. */
export const defaultMask = (designs: Designs) => designs.picks?.knee?.mask ?? designs.picks?.car_best?.mask ?? 0;

/** A % change with enough decimals to show it: two below 1%, one above; a change that rounds to nothing has no sign. */
export function fmtChange(v: number | null | undefined) {
  if (v == null) return '–';
  const digits = Math.abs(v) < 1 ? 2 : 1;
  const shown = Math.abs(v).toFixed(digits);
  return Number(shown) === 0 ? `${(0).toFixed(digits)}%` : `${v > 0 ? '+' : '−'}${shown}%`;
}

const tone = (v: number | null | undefined) => (v == null || Math.abs(v) < 0.005 ? '' : v < 0 ? 'better' : 'worse');

/** #30's tail-risk result in plain words. */
export function badDaySentence(tail: NonNullable<Designs['tail_risk']>) {
  const times = (pct: number) => `${(1 + pct / 100).toFixed(1)}×`;
  if (tail.same_design) {
    return `The design that's best on an average day is also the best on the worst day, when car travel time is ` +
      `${times(tail.x_pct)} an average day.`;
  }
  return `On the worst day, the design that's best on average loses ${tail.x_pct.toFixed(1)}% against its average day; ` +
    `the design built for bad days loses ${tail.y_pct.toFixed(1)}%.`;
}

const roleNames = (designs: Designs, mask: number) =>
  ROLES.filter(([r]) => designs.picks?.[r]?.mask === mask).map(([, label]) => label).join(' · ');

function Mark({ c }: { c: DesignCandidate }) {
  if (c.kind === 'signalise_crossing') return <i className="signal-mark" />;
  return <i className={`line ${c.kind === 'close' ? 'dashed' : 'solid'}`} style={{ color: DESIGN[c.kind] }} />;
}

function Menu({ candidates, onStreet }: { candidates: DesignCandidate[]; onStreet(name: string): void }) {
  return (
    <ul className="menu">
      {candidates.map((c) => (
        <li key={c.id}>
          <button onClick={() => c.street && onStreet(c.street)}>
            <Mark c={c} />
            <span><strong>{changeTitle(c)}</strong><small>{c.reason}</small></span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function Scatter({ designs, selected, onSelect }: { designs: Designs; selected: number; onSelect(mask: number): void }) {
  const points = (designs.designs ?? []).flatMap((d) => {
    const v = versusToday(designs, d);
    return d.feasible && v?.car != null && v.ped != null ? [{ d, x: v.car, y: v.ped }] : [];
  });
  if (points.length < 2) return null;
  const [w, h, left, right, top, bottom] = [300, 170, 14, 14, 22, 22];
  const span = (vals: number[]) => {
    const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
    return hi - lo < 1e-9 ? [lo - 1, hi + 1] : [lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08];
  };
  const [x0, x1] = span(points.map((p) => p.x));
  const [y0, y1] = span(points.map((p) => p.y));
  const sx = (x: number) => left + ((x - x0) / (x1 - x0)) * (w - left - right);
  const sy = (y: number) => h - bottom - ((y - y0) / (y1 - y0)) * (h - top - bottom);
  const front = new Set(designs.trade_offs ?? []);
  return (
    <svg className="scatter" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Every design: change in car travel time against change in walking delay">
      <line x1={sx(0)} x2={sx(0)} y1={top} y2={h - bottom} className="axis" />
      <line x1={left} x2={w - right} y1={sy(0)} y2={sy(0)} className="axis" />
      <text x={w - right} y={h - 7} textAnchor="end">car travel time vs today →</text>
      <text x={left} y={14}>↑ walking delay vs today</text>
      {points.map(({ d, x, y }) => {
        const label = roleNames(designs, d.mask);
        const flip = sx(x) > w * 0.6;
        return (
          <g key={d.mask} className={`pt ${front.has(d.mask) ? 'front' : ''} ${d.mask === selected ? 'on' : ''}`}
            onClick={() => onSelect(d.mask)}>
            <circle cx={sx(x)} cy={sy(y)} r={d.mask === selected ? 7 : 5}>
              <title>{d.candidates.length ? d.candidates.join(', ') : 'Do nothing'}</title>
            </circle>
            {label && <text x={sx(x) + (flip ? -10 : 10)} y={sy(y) + 4} textAnchor={flip ? 'end' : 'start'}>{label}</text>}
          </g>
        );
      })}
    </svg>
  );
}

/** A signed whole number with its unit, e.g. +93 or −1,098; zero has no sign. */
export function fmtSigned(v: number) {
  const n = Math.round(v);
  return n === 0 ? '0' : `${n > 0 ? '+' : '−'}${fmt(Math.abs(n))}`;
}

const KIND_NAME = { close: 'Closed to through traffic', reallocate: 'Bus lane', signalise_crossing: 'New signals for people on foot' } as const;

/** What a change does, in plain words, from the numbers it was picked on. */
export function whatItDoes(c: DesignCandidate) {
  const e = c.evidence as Record<string, number | string | undefined>;
  const n = (key: string) => fmt(Number(e[key] ?? 0));
  if (c.kind === 'signalise_crossing') {
    const after = e.signalised_delay_s != null ? ` drops from ${n('delay_s')} s (level of service ${e.los}) to ${n('signalised_delay_s')} s (${e.signalised_los})`
      : ` is ${n('delay_s')} s today (level of service ${e.los})`;
    return `Adds traffic lights so people cross on their own green phase. The modelled wait to cross${after}, beside ${n('vehicle_flow_veh_h')} cars an hour.`;
  }
  if (c.kind === 'close') {
    return `Stops through traffic on ${c.street}. Today its busiest direction carries ${n('flow_veh_h')} cars an hour in the morning peak; the model sends them onto other streets.`;
  }
  return `Turns half of ${c.street}'s car capacity into a bus lane. Today its busiest direction carries ${n('flow_veh_h')} cars an hour, at ${fmtPct(Number(e.vc ?? 0))} of capacity.`;
}

function Scores({ designs, design }: { designs: Designs; design: Design }) {
  const today = designs.designs?.find((d) => d.mask === 0 && d.feasible);
  const v = versusToday(designs, design);
  const delta = (key: 'mean_j_car' | 'mean_j_ped' | 'j_risk', scale: number) =>
    today?.[key] != null && design[key] != null ? fmtSigned((design[key]! - today[key]!) / scale) : '–';
  const tiles: [string, string, string, number | null | undefined][] = [
    ['Car time', delta('mean_j_car', 3600), 'vehicle-hours', v?.car],
    ['Waiting at crossings', delta('mean_j_ped', 1), 's, all crossings', v?.ped],
    ['Car time on bad days', delta('j_risk', 3600), 'vehicle-hours', v?.risk],
  ];
  return (
    <div className="tiles">
      {tiles.map(([label, value, unit, change]) => (
        <div key={label} className={`tile ${tone(change)}`}>
          <strong>{value}</strong><small>{unit}</small><span>{label} · {fmtChange(change)}</span>
        </div>
      ))}
    </div>
  );
}

interface CardProps {
  designs: Designs;
  id: string;
  onClose(): void;
  onSelect(mask: number): void;
}

/** One change on the menu, opened from its pin: what it does, its effect on its own, and why it was picked. */
export function ChangeCard({ designs, id, onClose, onSelect }: CardProps) {
  const c = designs.candidates.find((x) => x.id === id);
  if (!c) return null;
  const alone = designs.designs?.find((d) => d.feasible && d.candidates.length === 1 && d.candidates[0] === id);
  const picks = ROLES.filter(([r]) => designs.picks?.[r]?.candidates.includes(id)).map(([, label]) => label);
  return (
    <aside className="inspector change-card" aria-label={`${changeTitle(c)} details`}>
      <header className="inspector-head">
        <div>
          <p className="eyebrow">{KIND_NAME[c.kind]}</p>
          <h2><mark>{c.street ?? 'Crossing'}</mark></h2>
          <p>{changeTitle(c)}</p>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="close" /></button>
      </header>
      <div className="inspector-body">
        <section className="section">
          <header><h3>What it does</h3></header>
          <p className="body">{whatItDoes(c)}</p>
        </section>
        {alone && (
          <section className="section">
            <header><h3>On its own, against today</h3></header>
            <Scores designs={designs} design={alone} />
            <button className="link" onClick={() => onSelect(alone.mask)}>Show this change on its own</button>
          </section>
        )}
        <section className="section">
          <header><h3>Why it was picked</h3></header>
          <p className="body">{c.reason}</p>
          {picks.length > 0 && <p className="note">Part of the recommended designs: {picks.join(', ')}.</p>}
        </section>
      </div>
    </aside>
  );
}

interface Props {
  designs: Designs;
  selected: number;
  onSelect(mask: number): void;
  onStreet(name: string): void;
}

export default function DesignsKey({ designs, selected, onSelect, onStreet }: Props) {
  const scenarios = designs.scenarios?.length ?? 4;
  if (designs.status !== 'done') {
    return (
      <div className="key">
        <p className="headline">
          Testing every combination of these {designs.candidates.length} changes on {scenarios} kinds of day. The best
          designs appear here when the search finishes.
        </p>
        <Menu candidates={designs.candidates} onStreet={onStreet} />
      </div>
    );
  }
  const design = designs.designs?.find((d) => d.mask === selected);
  const role = roleNames(designs, selected);
  const pick = Object.values(designs.picks ?? {}).find((p) => p.mask === selected);
  const changes = designs.candidates.filter((c) => design?.candidates.includes(c.id));
  const cut = (designs.designs ?? []).filter((d) => !d.feasible).length;
  return (
    <div className="key">
      {designs.tail_risk && <p className="headline">{badDaySentence(designs.tail_risk)}</p>}
      <div className="tabs picks" role="group" aria-label="Designs to compare">
        {ROLES.filter(([r]) => designs.picks?.[r]).map(([r, label]) => (
          <button key={r} aria-pressed={designs.picks![r].mask === selected} onClick={() => onSelect(designs.picks![r].mask)}>{label}</button>
        ))}
      </div>
      <Scatter designs={designs} selected={selected} onSelect={onSelect} />
      {cut > 0 && <p className="note">{cut} {cut === 1 ? 'design cuts' : 'designs cut'} a zone off the network, so {cut === 1 ? 'it is' : 'they are'} left out.</p>}
      {design && (
        <section className="design-card">
          <p className="eyebrow">{role || 'Design'} · {changes.length || 'no'} {changes.length === 1 ? 'change' : 'changes'}</p>
          {changes.length ? (
            <ul className="changes">{changes.map((c) => <li key={c.id}><Mark c={c} />{changeTitle(c)}</li>)}</ul>
          ) : <p className="empty">Today's streets, unchanged</p>}
          <Scores designs={designs} design={design} />
          {pick?.policy && (
            <details className="fold">
              <summary>{pick.policy.summary}</summary>
              <ul className="reasons">
                {pick.policy.clauses.map((c) => (
                  <li key={c.clause_id}><strong>{c.street}: {c.status.replace('_', ' ')}</strong><span>{c.detail || c.restriction.replaceAll('_', ' ')}</span></li>
                ))}
              </ul>
            </details>
          )}
          {pick?.emissions?.change_pct && (
            <p className="note">
              Emissions (modelled, display only): {Object.entries(pick.emissions.change_pct).filter(([, v]) => v != null)
                .map(([k, v]) => `${k} ${fmtChange(v)}`).join(' · ')}
            </p>
          )}
        </section>
      )}
      <details className="fold">
        <summary>Why these {designs.candidates.length} changes</summary>
        <Menu candidates={designs.candidates} onStreet={onStreet} />
      </details>
      <p className="note">
        Changes against today's streets in the whole city's modelled morning hour ({designs.hour}), weighted over
        {scenarios} kinds of day; the dashed lines cross at today. Click a pin on the map to see what each change does.
      </p>
      <button className="link" onClick={() => onSelect(0)}><Icon name="recentre" size={14} />Compare with today</button>
    </div>
  );
}
