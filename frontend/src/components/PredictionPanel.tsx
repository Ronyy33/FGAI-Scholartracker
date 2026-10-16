import { MODEL_LABELS, MODELS, num } from "../lib/format";
import type { Factor, Prediction } from "../lib/types";
import { BandChip, Card, InfoTip } from "./ui";

type PanelPrediction = Omit<Prediction, "prn"> & { prn?: string };

export function RiskHeadline({ p }: { p: PanelPrediction }) {
  return (
    <div className="flex flex-wrap items-end gap-4">
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-muted">
          Academic Risk Score
          <InfoTip text="round(100 × estimated probability). A score of 54 means an estimated 54 % chance — a statement about one student, not about model accuracy." />
        </p>
        <p className="text-5xl font-semibold tabular-nums text-ink">{p.risk_score}</p>
      </div>
      <div className="pb-2">
        <BandChip band={p.risk_band} />
        <p className="mt-1 text-sm text-ink-2">
          Estimated {p.risk_score} % chance of a difficulty semester in semester {p.predicts_semester}
          {p.flagged ? " · above the review threshold" : ""}
        </p>
      </div>
    </div>
  );
}

export function ModelScores({ p }: { p: PanelPrediction }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs uppercase tracking-wide text-muted">
          <th className="py-1 font-medium">Model</th>
          <th className="py-1 text-right font-medium">Score</th>
          <th className="py-1 pl-3 font-medium">Band</th>
        </tr>
      </thead>
      <tbody>
        {MODELS.map((m) => (
          <tr key={m} className="border-t border-line">
            <td className="py-1.5 text-ink">
              {MODEL_LABELS[m]}
              {m === p.default_model && <span className="ml-1 text-xs text-muted">(default)</span>}
            </td>
            <td className="py-1.5 text-right tabular-nums text-ink">{p.models[m].risk_score}</td>
            <td className="py-1.5 pl-3"><BandChip band={p.models[m].risk_band} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function FactorList({ factors, note }: { factors: Factor[]; note: string | null }) {
  const max = Math.max(1, ...factors.map((f) => Math.abs(f.effect_points)));
  return (
    <div>
      {factors.length === 0 ? (
        <p className="text-sm text-ink-2">No group moves the estimate by a point or more.</p>
      ) : (
        <ul className="space-y-3">
          {factors.map((f) => (
            <li key={f.group}>
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="font-medium text-ink">{f.label}</span>
                <span className="tabular-nums text-ink">
                  {f.effect_points > 0 ? "+" : "−"}
                  {Math.abs(f.effect_points).toFixed(1)} points · {f.direction}
                </span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-surface-2" aria-hidden>
                <div
                  className={`h-1.5 rounded-full ${f.effect_points > 0 ? "bg-critical/70" : "bg-good/70"}`}
                  style={{ width: `${(100 * Math.abs(f.effect_points)) / max}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-ink-2">
                {f.indicators
                  .filter((i) => i.value !== null)
                  .slice(0, 4)
                  .map((i) => `${i.label}: ${num(i.value)} (typical ${num(i.typical_value)})`)
                  .join(" · ")}
              </p>
            </li>
          ))}
        </ul>
      )}
      {note && <p className="mt-4 text-xs text-muted">{note}</p>}
    </div>
  );
}

export function Suggestions({ p }: { p: PanelPrediction }) {
  const r = p.recommendations;
  if (!r) return null;
  return (
    <div>
      {r.items.length === 0 ? (
        <p className="text-sm text-ink-2">{r.message}</p>
      ) : (
        <ul className="space-y-3">
          {r.items.map((item) => (
            <li key={item.id} className="rounded-lg border border-line p-3 text-sm">
              <p className="text-ink-2">{item.observation}</p>
              <p className="mt-1 text-ink">{item.suggestion}</p>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-4 text-xs text-muted">{r.disclaimer}</p>
    </div>
  );
}

export function PredictionPanel({ p }: { p: PanelPrediction }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Risk estimate">
        <RiskHeadline p={p} />
        <div className="mt-5">
          <ModelScores p={p} />
        </div>
      </Card>
      <Card title="Main contributing factors" subtitle="Change in risk score versus a typical student">
        <FactorList factors={p.explanation} note={p.explanation_note} />
      </Card>
      <div className="lg:col-span-2">
        <Card title="Supportive suggestions" subtitle="Based on observed academic indicators">
          <Suggestions p={p} />
        </Card>
      </div>
    </div>
  );
}
