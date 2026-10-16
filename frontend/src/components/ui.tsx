import type { ReactNode } from "react";
import type { ApiError } from "../lib/api";
import type { BandName } from "../lib/types";

export function Card({ title, subtitle, children, actions }: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-line bg-surface p-5">
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div>
            {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-ink-2">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function PageHeader({ title, intro }: { title: string; intro?: ReactNode }) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
      {intro && <p className="mt-1 max-w-3xl text-sm text-ink-2">{intro}</p>}
    </header>
  );
}

export function StatTile({ label, value, note }: { label: string; value: ReactNode; note?: string }) {
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <p className="text-xs font-medium uppercase tracking-wide text-muted">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums text-ink">{value}</p>
      {note && <p className="mt-1 text-xs text-ink-2">{note}</p>}
    </div>
  );
}

const BAND_STYLE: Record<BandName, { icon: string; color: string; title: string }> = {
  Low: { icon: "●", color: "text-good", title: "Low risk band" },
  Moderate: { icon: "▲", color: "text-warning", title: "Moderate risk band" },
  High: { icon: "◆", color: "text-critical", title: "High risk band" },
};

/** Risk band = colour + icon + text, never colour alone. */
export function BandChip({ band }: { band: BandName }) {
  const s = BAND_STYLE[band];
  return (
    <span
      title={s.title}
      className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-2 px-2.5 py-0.5 text-xs font-medium text-ink"
    >
      <span aria-hidden className={`text-[11px] leading-none ${s.color}`}>{s.icon}</span>
      {band}
    </span>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 py-8 text-sm text-ink-2">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-line border-t-accent" />
      {label}
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-xl border border-critical/40 bg-surface p-4 text-sm">
      <p className="font-medium text-ink">Something went wrong</p>
      <p className="mt-1 text-ink-2">{error.message}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-3 rounded-md border border-line px-3 py-1.5 text-ink hover:bg-surface-2">
          Try again
        </button>
      )}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="py-6 text-center text-sm text-ink-2">{children}</p>;
}

export function SyntheticBanner() {
  return (
    <div role="note" className="border-b border-line bg-notice px-4 py-2 text-center text-xs text-notice-ink sm:text-sm">
      <strong>Synthetic demonstration.</strong> Every student and score here comes from generated
      data. Results measure agreement with a documented synthetic process, not real-world
      accuracy. Decision support only — never a basis for penalties.
    </div>
  );
}

export function InfoTip({ text }: { text: string }) {
  return (
    <span
      tabIndex={0}
      title={text}
      aria-label={text}
      className="ml-1 inline-flex h-4 w-4 cursor-help items-center justify-center rounded-full border border-line text-[10px] text-ink-2"
    >
      ?
    </span>
  );
}
