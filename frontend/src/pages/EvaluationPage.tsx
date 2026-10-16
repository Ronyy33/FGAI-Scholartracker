import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { BandChip, Card, ErrorState, Loading, PageHeader } from "../components/ui";
import { useApi } from "../lib/api";
import { MODEL_LABELS, MODELS, num, pct } from "../lib/format";
import { MODEL_STYLE, usePalette } from "../lib/palette";
import type { CurvePoint, Metrics, ModelName } from "../lib/types";

function CurveChart({ curves, xLabel, yLabel, diagonal, baseline }: {
  curves: Record<ModelName, CurvePoint[]>;
  xLabel: string;
  yLabel: string;
  diagonal?: boolean;
  baseline?: number;
}) {
  const palette = usePalette();
  const axis = { stroke: palette.axis, tick: { fill: palette.muted, fontSize: 12 } };
  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart margin={{ top: 8, right: 12, bottom: 16, left: 0 }}>
          <CartesianGrid stroke={palette.grid} />
          <XAxis type="number" dataKey="x" domain={[0, 1]} {...axis} label={{ value: xLabel, position: "insideBottom", offset: -8, fill: palette.ink2, fontSize: 12 }} />
          <YAxis type="number" dataKey="y" domain={[0, 1]} {...axis} label={{ value: yLabel, angle: -90, position: "insideLeft", fill: palette.ink2, fontSize: 12 }} />
          <Tooltip
            contentStyle={{ background: palette.surface, borderColor: palette.grid, color: palette.ink }}
            formatter={(v, name) => [num(Number(v), 3), String(name)]}
            labelFormatter={(x) => `${xLabel} ${num(Number(x), 3)}`}
          />
          <Legend verticalAlign="top" iconType="plainline" itemSorter={null} wrapperStyle={{ fontSize: 12, color: palette.ink2 }} />
          {diagonal && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={palette.muted} strokeDasharray="2 4" />}
          {baseline !== undefined && <ReferenceLine y={baseline} stroke={palette.muted} strokeDasharray="2 4" />}
          {MODELS.map((m) => (
            <Line key={m} data={curves[m]} dataKey="y" name={MODEL_LABELS[m]} stroke={palette.series[MODEL_STYLE[m].slot]}
              strokeDasharray={MODEL_STYLE[m].dash} strokeWidth={2} dot={false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const ci = (v: [number, number] | undefined, d = 3) => (v ? `[${v[0].toFixed(d)}, ${v[1].toFixed(d)}]` : "");

export function EvaluationPage() {
  const { data, error, loading, reload } = useApi<Metrics>("/api/metrics");
  if (loading) return <Loading />;
  if (error) return <ErrorState error={error} onRetry={reload} />;
  if (!data) return null;
  const ev = data.evaluation;
  const chosen = ev.models[ev.chosen_model];
  const reliability = Object.fromEntries(
    MODELS.map((m) => [m, data.curves.reliability[m].filter((r) => r.mean_predicted !== null && r.observed_rate !== null)
      .map((r) => ({ x: r.mean_predicted as number, y: r.observed_rate as number }))]),
  ) as Record<ModelName, CurvePoint[]>;

  return (
    <>
      <PageHeader
        title="Model evaluation"
        intro={`One-time evaluation on ${ev.test_rows} held-out student-semesters (${ev.test_students} students, ${pct(ev.positive_rate)} positive). Choices were frozen on validation data first. Synthetic data: these numbers describe agreement with the synthetic process only.`}
      />
      <div className="space-y-4">
        <Card title="Model comparison (test set, 95 % CI by student bootstrap)">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead className="text-left text-xs uppercase tracking-wide text-muted">
                <tr><th className="py-1 font-medium">Model</th><th className="font-medium">ROC-AUC</th><th className="font-medium">PR-AUC</th><th className="font-medium">Brier</th><th className="font-medium">ECE</th><th className="font-medium">F1 at threshold</th><th className="font-medium">Top-{ev.top_k} capture</th></tr>
              </thead>
              <tbody>
                {(["majority", ...MODELS] as const).map((m) => {
                  const r = ev.models[m];
                  return (
                    <tr key={m} className="border-t border-line">
                      <td className="py-1.5 text-ink">{m === "majority" ? "Majority baseline" : MODEL_LABELS[m]}{m === ev.chosen_model && <span className="ml-1 text-xs text-muted">(chosen)</span>}</td>
                      <td className="tabular-nums">{r.probability.roc_auc.toFixed(3)} <span className="text-xs text-muted">{ci(r.ci95.roc_auc)}</span></td>
                      <td className="tabular-nums">{r.probability.pr_auc.toFixed(3)} <span className="text-xs text-muted">{ci(r.ci95.pr_auc)}</span></td>
                      <td className="tabular-nums">{r.probability.brier.toFixed(4)}</td>
                      <td className="tabular-nums">{r.ece.toFixed(4)}</td>
                      <td className="tabular-nums">{num(r.threshold.f1, 3)}</td>
                      <td className="tabular-nums">{num(r.top_k.capture_pct, 1)} %</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-sm text-ink-2">
            Selection rule: lowest validation {data.selection.rule.primary_metric}; models within {data.selection.rule.tolerance} are tied and the simpler one wins.
            Tied: {data.selection.tied_within_tolerance.map((m) => MODEL_LABELS[m]).join(", ")} → <strong className="text-ink">{MODEL_LABELS[data.selection.chosen_model]}</strong>.
            Paired differences between the learned models all have intervals containing 0.
          </p>
        </Card>

        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="ROC curves" subtitle="Dotted diagonal = random ranking"><CurveChart curves={data.curves.roc} xLabel="False positive rate" yLabel="True positive rate" diagonal /></Card>
          <Card title="Precision–recall curves" subtitle={`Dotted line = random (positive rate ${ev.positive_rate.toFixed(2)})`}><CurveChart curves={data.curves.pr} xLabel="Recall" yLabel="Precision" baseline={ev.positive_rate} /></Card>
          <Card title="Calibration" subtitle="Observed rate vs predicted probability; diagonal = perfect"><CurveChart curves={reliability} xLabel="Mean predicted probability" yLabel="Observed rate" diagonal /></Card>
          <Card title={`Confusion matrix — ${MODEL_LABELS[ev.chosen_model]} at ${chosen.threshold.threshold.toFixed(3)}`}>
            <table className="mx-auto text-center text-sm">
              <thead><tr><th /><th className="px-4 py-2 font-medium text-ink-2">Predicted: not at risk</th><th className="px-4 py-2 font-medium text-ink-2">Predicted: at risk</th></tr></thead>
              <tbody>
                <tr><th className="py-3 pr-3 text-right font-medium text-ink-2">Actual: no difficulty</th><td className="border border-line px-6 py-3 text-lg tabular-nums">{chosen.threshold.tn}</td><td className="border border-line px-6 py-3 text-lg tabular-nums">{chosen.threshold.fp}</td></tr>
                <tr><th className="py-3 pr-3 text-right font-medium text-ink-2">Actual: difficulty</th><td className="border border-line px-6 py-3 text-lg tabular-nums">{chosen.threshold.fn}</td><td className="border border-line px-6 py-3 text-lg tabular-nums">{chosen.threshold.tp}</td></tr>
              </tbody>
            </table>
            <p className="mt-3 text-sm text-ink-2">
              Recall {pct(chosen.threshold.recall)} · precision {pct(chosen.threshold.precision)} · flags {pct(chosen.threshold.flagged_share)} of students.
              Accuracy ({pct(chosen.threshold.accuracy)}) is misleading here: calling everyone "fine" would already score {pct(1 - ev.positive_rate)}.
            </p>
          </Card>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <Card title={`Top-${ev.top_k} risk capture`} subtitle="If mentors can meet only the highest-risk 10 %">
            <p className="text-sm text-ink-2">
              The {ev.top_k} highest scores contain {chosen.top_k.positives_in_top_k} of the {chosen.top_k.total_positives} students who later struggled:
              capture {num(chosen.top_k.capture_pct, 1)} % (maximum possible {num((100 * ev.top_k) / chosen.top_k.total_positives, 1)} %), precision@K {num(chosen.top_k.precision_at_k_pct, 1)} %.
              This is a ranking measure, not accuracy.
            </p>
          </Card>
          <Card title="Do the bands mean what they say?" subtitle="Observed difficulty rate per band (test set)">
            <ul className="space-y-2 text-sm">
              {ev.risk_bands.map((b) => (
                <li key={b.band} className="flex items-center justify-between">
                  <span className="flex items-center gap-2"><BandChip band={b.band} /><span className="text-ink-2">scores {b.range}</span></span>
                  <span className="tabular-nums text-ink">{b.count} students · observed {pct(b.observed_rate)}</span>
                </li>
              ))}
            </ul>
          </Card>
        </div>

        <Card title="Provenance and limitations">
          <ul className="list-disc space-y-1 pl-5 text-sm text-ink-2">
            <li>Data: synthetic, generated from a documented process modelled on the college CSV format; no real student.</li>
            <li>Validation and test scores differ (ROC-AUC 0.93 vs 0.88); cross-validation shows this is normal variation for ~180-student samples.</li>
            <li>History adds little in this synthetic data, so the LSTM cannot outperform the snapshot models here.</li>
            <li>Before use at a real college: local data, re-validation, a fairness review and institutional approval.</li>
          </ul>
        </Card>
      </div>
    </>
  );
}
