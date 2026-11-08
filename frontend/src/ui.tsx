import { Component, type ReactNode } from 'react';
import type { Basis } from './data';

const PATHS = {
  search: 'M10.5 17a6.5 6.5 0 1 0 0-13 6.5 6.5 0 0 0 0 13zM15.5 15.5 20 20',
  close: 'M6 6l12 12M18 6 6 18',
  plus: 'M12 5v14M5 12h14',
  minus: 'M5 12h14',
  recentre: 'M12 3v3M12 18v3M3 12h3M18 12h3M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z',
  play: 'M8 5.5v13l10.5-6.5z',
  pause: 'M9 5.5v13M15 5.5v13',
};

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}

const BASIS_LABEL: Record<Basis, string> = {
  observed: 'Observed', estimated: 'Estimated', modelled: 'Modelled', predicted: 'Predicted', synthetic: 'Synthetic',
  verified: 'Verified',
};

export function BasisChip({ basis }: { basis: Basis }) {
  return <span className={`chip chip-${basis}`}>{BASIS_LABEL[basis]}</span>;
}

const CLASS_LABEL: Record<string, string> = {
  motorway: 'Motorway', trunk: 'Trunk road', primary: 'Primary road', secondary: 'Secondary road',
  tertiary: 'Tertiary road', unclassified: 'Minor road', residential: 'Residential street',
  living_street: 'Living street', service: 'Service road', cycleway: 'Cycleway', footway: 'Footpath',
  pedestrian: 'Pedestrian street',
};

export const roadClass = (cls: string) => CLASS_LABEL[cls] ?? cls;

const RESTRICTION_LABEL: Record<string, string> = {
  no_motor_vehicles: 'No motor vehicles', one_way: 'One way', pedestrian_only: 'Pedestrians only',
  keep_open: 'Must stay open', no_buses: 'No buses', no_entry: 'No entry', max_weight_7_5t: 'Max weight 7.5 t',
  max_length_8m: 'Max length 8 m', no_right_turn: 'No right turn',
  no_motor_vehicles_part_street: 'No motor vehicles (part of the street)',
};

export const restrictionLabel = (r: string) => RESTRICTION_LABEL[r] ?? r.replaceAll('_', ' ');

export function reasonLabel(reason: string | null, restriction = '') {
  if (!reason) return '';
  if (restriction.endsWith('_part_street')) return 'Covers only part of the street, and the checker applies whole streets';
  if (restriction.endsWith('_phased')) return "Applies during works phases the order doesn't date";
  if (reason.startsWith('unknown restriction')) return "A restriction type the checker can't encode yet";
  return reason.charAt(0).toUpperCase() + reason.slice(1);
}

const THEME_LABEL: Record<string, string> = {
  crossing_delay: 'Crossing delay', road_safety: 'Road safety', rat_running: 'Through traffic',
  air_quality: 'Air quality', missing_cycle_link: 'Missing cycle link', accessibility: 'Accessibility',
  parking: 'Parking', severance: 'Severance', noise: 'Noise', other: 'Other',
};

export const themeLabel = (theme: string) => THEME_LABEL[theme] ?? theme.replaceAll('_', ' ');

interface BoundaryProps {
  fallback(error: Error, reset: () => void): ReactNode;
  children: ReactNode;
}

/** Shows fallback instead of its children when they throw while rendering, so one panel can't blank the page. */
export class ErrorBoundary extends Component<BoundaryProps, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error) {
    console.error(error);
  }

  reset = () => this.setState({ error: null });

  render() {
    return this.state.error ? this.props.fallback(this.state.error, this.reset) : this.props.children;
  }
}

const whole = new Intl.NumberFormat('en-GB', { maximumFractionDigits: 0 });
export const fmt = (n: number) => whole.format(n);
export const fmtPct = (ratio: number) => `${whole.format(ratio * 100)}%`;
export const fmtLength = (m: number) => (m >= 1000 ? `${(m / 1000).toFixed(1)} km` : `${whole.format(m)} m`);

const clock = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', timeZone: 'Europe/London' });
export const fmtTime = (iso: string, plusMinutes = 0) => clock.format(new Date(Date.parse(iso) + plusMinutes * 60_000));
