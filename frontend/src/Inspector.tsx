import { Fragment, type ReactNode } from 'react';
import { streetSpeeds, type Basis, type Clause, type LayerId, type Meta, type Speeds, type StreetInfo } from './data';
import { TRAFFIC_RAMP } from './MapView';
import { BasisChip, Icon, fmt, fmtLength, fmtPct, fmtTime, reasonLabel, restrictionLabel, roadClass, themeLabel } from './ui';

const PENDING: [LayerId, string][] = [
  ['rules', 'Legal restrictions'],
  ['concerns', 'Consultation concerns'],
  ['speeds', 'Bus-observed speed'],
  ['traffic', 'Car traffic'],
  ['crossings', 'Pedestrian crossings'],
];

const CONTROL: Record<string, string> = {
  signalised: 'signalised', zebra: 'zebra', unsignalised: 'uncontrolled', 'unsignalised-one-stage': 'uncontrolled',
  'unsignalised-capped': 'uncontrolled',
};

const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);

function rampColour(vc: number) {
  return TRAFFIC_RAMP.filter(([stop]) => vc >= stop).at(-1)?.[1] ?? TRAFFIC_RAMP[0][1];
}

function Section({ title, basis, note, children }: { title: string; basis: Basis; note?: string; children: ReactNode }) {
  return (
    <section className="section">
      <header><h3>{title}</h3><BasisChip basis={basis} /></header>
      {children}
      {note && <p className="note">{note}</p>}
    </section>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return <div className="row"><span>{label}</span><strong>{children}</strong></div>;
}

function Stat({ label, value, unit, children }: { label: string; value: string; unit?: string; children?: ReactNode }) {
  return <div className="stat"><span>{label}</span><strong>{value}{unit && <small>{unit}</small>}{children}</strong></div>;
}

const losTone = (los: string) => ('AB'.includes(los) ? 'good' : 'CD'.includes(los) ? 'fair' : 'poor');

function Traffic({ info, meta }: { info: StreetInfo; meta: Meta }) {
  const layer = meta.layers.traffic!;
  const modelled = info.segments.filter((s) => s.vc !== undefined);
  const note = [layer.hour, layer.note].filter(Boolean).join(' · ');
  if (!modelled.length) {
    const text = info.segments.some((s) => s.car) ? 'Not part of the modelled car network' : 'Closed to cars on OpenStreetMap';
    return <Section title="Car traffic" basis={layer.basis}><p className="empty">{text}</p></Section>;
  }
  const peakFlow = Math.max(...modelled.map((s) => s.flow ?? 0));
  if (peakFlow === 0) {
    return (
      <Section title="Car traffic" basis={layer.basis} note={note}>
        <p className="empty">No cars in the modelled hour: the demand model sends no traffic along this street</p>
      </Section>
    );
  }
  const peakVc = Math.max(...modelled.map((s) => s.vc!));
  const t = sum(modelled.map((s) => (s.f_t ?? 0) + (s.r_t ?? 0)));
  const t0 = sum(modelled.map((s) => (s.f_t0 ?? 0) + (s.r_t0 ?? 0)));
  return (
    <Section title="Car traffic" basis={layer.basis} note={note}>
      <div className="stats">
        <Stat label="Busiest point" value={fmt(peakFlow)} unit="veh/h" />
        <div className="stat">
          <span>Peak load</span>
          <strong>{fmtPct(peakVc)}<small>of capacity</small></strong>
          <div className="meter" role="presentation">
            <span style={{ width: `${Math.min(peakVc, 1.2) / 1.2 * 100}%`, background: rampColour(peakVc) }} />
          </div>
        </div>
      </div>
      {t0 > 0 && <Row label="Travel time vs free flow">+{fmtPct(t / t0 - 1)}</Row>}
    </Section>
  );
}

function Crossings({ info, meta }: { info: StreetInfo; meta: Meta }) {
  const layer = meta.layers.crossings!;
  if (!info.crossings.length) {
    return <Section title="Pedestrian crossings" basis={layer.basis}><p className="empty">No crossings mapped on this street</p></Section>;
  }
  const kinds = new Map<string, number>();
  for (const c of info.crossings) kinds.set(CONTROL[c.control] ?? c.control, (kinds.get(CONTROL[c.control] ?? c.control) ?? 0) + 1);
  const worst = info.crossings.reduce((a, b) => (b.delay > a.delay ? b : a));
  return (
    <Section title="Pedestrian crossings" basis={layer.basis} note={layer.note}>
      <div className="stats">
        <Stat label="Crossings" value={fmt(info.crossings.length)} />
        <Stat label="Longest wait" value={fmt(worst.delay)} unit="s">
          <span className={`los los-${losTone(worst.los)}`}>LOS {worst.los}</span>
        </Stat>
      </div>
      <Row label="Control">{[...kinds].map(([k, n]) => `${n} ${k}`).join(' · ')}</Row>
    </Section>
  );
}

function BusSpeed({ info, meta, speeds, bin }: { info: StreetInfo; meta: Meta; speeds: Speeds; bin: number }) {
  const layer = meta.layers.speeds!;
  const { now, seen, busRoad } = streetSpeeds(speeds, info.segments, bin);
  if (!busRoad) {
    return <Section title="Bus-observed speed" basis={layer.basis}><p className="empty">No bus roads on this street</p></Section>;
  }
  const time = fmtTime(speeds.bins[bin]);
  const seenCount = seen.filter(Boolean).length;
  return (
    <Section title="Bus-observed speed" basis={layer.basis} note="Speed of buses, not of all traffic. Slower direction shown.">
      <div className="stats">
        <Stat label={`Slowest at ${time}`} value={now ? fmt(now.speed) : '–'} unit="km/h">
          {now && <span className={`kind kind-${now.observed ? 'observed' : 'predicted'}`}>{now.observed ? 'Seen' : 'Predicted'}</span>}
        </Stat>
        <Stat label="Quarter-hours with a bus" value={fmt(seenCount)} unit={`of ${seen.length}`} />
      </div>
      <div className="strip" role="img" aria-label={`Buses seen in ${seenCount} of ${seen.length} quarter-hours`}>
        {seen.map((s, i) => <span key={i} className={[s && 'seen', i === bin && 'now'].filter(Boolean).join(' ')} />)}
      </div>
    </Section>
  );
}

function OsmCheck({ clause }: { clause: Clause }) {
  if (clause.status !== 'applied') return null;
  if (!clause.osm_wrong_edges) return <p className="osm osm-ok">OpenStreetMap agrees</p>;
  return (
    <p className="osm osm-wrong">
      OpenStreetMap disagrees on {fmtLength(clause.osm_wrong_m)} of {fmtLength(clause.length_m)}: {clause.osm_says.join('; ')}
    </p>
  );
}

function Rules({ info, meta }: { info: StreetInfo; meta: Meta }) {
  const layer = meta.layers.rules!;
  if (!info.rules.length) {
    return <Section title="Legal restrictions" basis={layer.basis}><p className="empty">No verified clause names this street</p></Section>;
  }
  const live = info.rules.filter((c) => c.status !== 'abstained');
  const held = info.rules.filter((c) => c.status === 'abstained');
  return (
    <Section title="Legal restrictions" basis={layer.basis}>
      {live.map((c) => (
        <div key={c.id} className={`clause ${c.status === 'applied' && c.osm_wrong_edges ? 'clause-wrong' : ''}`}>
          <div className="clause-head"><strong>{restrictionLabel(c.restriction)}</strong><span>{c.times}</span></div>
          {c.in_force && <p className="clause-when">{c.in_force}</p>}
          <OsmCheck clause={c} />
          <blockquote>{c.quote}<cite>{c.document}, page {c.page}</cite></blockquote>
        </div>
      ))}
      {held.length > 0 && (
        <details className="fold">
          <summary>{held.length} {held.length === 1 ? 'clause' : 'clauses'} held back</summary>
          <ul className="reasons">
            {held.map((c) => (
              <li key={c.id}>
                <strong>{restrictionLabel(c.restriction)}</strong>
                <span>{reasonLabel(c.reason, c.restriction)}{c.in_force ? ` · ${c.in_force}` : ''}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </Section>
  );
}

function Concerns({ info, meta }: { info: StreetInfo; meta: Meta }) {
  const layer = meta.layers.concerns!;
  if (!info.concerns.length) {
    return <Section title="Consultation concerns" basis={layer.basis}><p className="empty">Not named in the consultation material we have</p></Section>;
  }
  const sources = [...new Set(info.concerns.map((c) => c.source).filter(Boolean))];
  return (
    <Section title="Consultation concerns" basis={layer.basis} note={sources.join(' · ')}>
      <div className="themes">
        {info.concerns.map((c, i) => <span key={i} className="theme">{themeLabel(c.theme)}<b>{fmt(c.count)}</b></span>)}
      </div>
    </Section>
  );
}

interface InspectorProps {
  info: StreetInfo;
  meta: Meta;
  speeds: Speeds | null;
  bin: number;
  focus: LayerId[];
  onClose(): void;
}

export default function Inspector({ info, meta, speeds, bin, focus, onClose }: InspectorProps) {
  const lengthByClass = new Map<string, number>();
  for (const s of info.segments) lengthByClass.set(s.cls, (lengthByClass.get(s.cls) ?? 0) + s.len);
  const cls = [...lengthByClass].sort((a, b) => b[1] - a[1])[0][0];
  const counts = meta.layers.counts;
  const sections: [LayerId, ReactNode][] = [
    ['rules', meta.layers.rules && <Rules info={info} meta={meta} />],
    ['concerns', meta.layers.concerns && <Concerns info={info} meta={meta} />],
    ['speeds', meta.layers.speeds && speeds && <BusSpeed info={info} meta={meta} speeds={speeds} bin={bin} />],
    ['traffic', meta.layers.traffic && <Traffic info={info} meta={meta} />],
    ['crossings', meta.layers.crossings && <Crossings info={info} meta={meta} />],
    ['counts', counts && info.counts.length > 0 && (
      <Section title="DfT traffic count" basis={counts.basis} note={counts.note}>
        {info.counts.map((p) => (
          <Row key={p.id} label={`${p.road} · point ${p.id} · ${p.year} ${p.counted ? 'count' : 'estimate'}`}>
            {fmt(p.aadf)}/day
          </Row>
        ))}
      </Section>
    )],
  ];
  return (
    <aside className="inspector" aria-label={`${info.name} details`}>
      <header className="inspector-head">
        <div>
          <p className="eyebrow">{roadClass(cls)}</p>
          <h2><mark>{info.name}</mark></h2>
          <p>{fmtLength(sum(info.segments.map((s) => s.len)))} in the study area</p>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="close" /></button>
      </header>
      <div className="inspector-body">
        {[...sections].sort(([a], [b]) => Number(focus.includes(b)) - Number(focus.includes(a)))
          .map(([id, node]) => <Fragment key={id}>{node}</Fragment>)}
        <ul className="pending">
          {PENDING.filter(([id]) => !meta.layers[id]).map(([id, title]) => (
            <li key={id}>{title}<span>No data yet</span></li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
