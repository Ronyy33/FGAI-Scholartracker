import { Link } from "react-router-dom";
import { BandChip, Card, ErrorState, Loading, PageHeader, StatTile } from "../components/ui";
import { useApi } from "../lib/api";
import { MODEL_LABELS, pct } from "../lib/format";
import type { Overview } from "../lib/types";

export function OverviewPage() {
  const { data, error, loading, reload } = useApi<Overview>("/api/overview");
  return (
    <>
      <PageHeader
        title="ScholarTrack — students at risk of falling behind"
        intro="Estimates, for each enrolled student, the probability that next semester brings a failed subject or an SGPA below 6.0, so mentors can offer support early. Thresholds are illustrative."
      />
      {loading && <Loading />}
      {error && <ErrorState error={error} onRetry={reload} />}
      {data && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <StatTile label="Records processed" value={data.records_processed.toLocaleString()} note="student-semesters (synthetic)" />
            <StatTile label="Students" value={data.students.toLocaleString()} note="6 synthetic cohorts" />
            <StatTile label="Scored now" value={data.current_students_scored} note={`enrolled in ${data.current_academic_year}`} />
            <StatTile label="Above review threshold" value={data.flagged} note="flagged by the default model" />
          </div>

          <Card title="Risk bands of currently enrolled students" subtitle="Bands are demonstration defaults (configs/risk_bands.yaml)">
            <ul className="space-y-3">
              {data.risk_bands.map((b) => (
                <li key={b.band}>
                  <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-sm">
                    <span className="flex items-center gap-2 whitespace-nowrap">
                      <BandChip band={b.band} />
                      <span className="text-ink-2">scores {b.range[0]}–{b.range[1]}</span>
                    </span>
                    <span className="tabular-nums text-ink">
                      {b.count} students · {b.percent.toFixed(1)} %
                    </span>
                  </div>
                  <div className="mt-1 h-2 rounded-full bg-surface-2" aria-hidden>
                    <div className="h-2 rounded-full bg-accent" style={{ width: `${b.percent}%` }} />
                  </div>
                </li>
              ))}
            </ul>
            <Link to="/students" className="mt-4 inline-block text-sm text-accent hover:underline">
              Review students by risk →
            </Link>
          </Card>

          <div className="grid gap-4 md:grid-cols-2">
            <Card title="Model status">
              <dl className="grid grid-cols-2 gap-y-2 text-sm">
                <dt className="text-ink-2">Default model</dt>
                <dd className="text-ink">{MODEL_LABELS[data.model.default_model]}</dd>
                <dt className="text-ink-2">Test ROC-AUC</dt>
                <dd className="tabular-nums text-ink">{data.model.test_roc_auc.toFixed(3)}</dd>
                <dt className="text-ink-2">Test PR-AUC</dt>
                <dd className="tabular-nums text-ink">
                  {data.model.test_pr_auc.toFixed(3)} <span className="text-muted">(random: {data.model.test_positive_rate.toFixed(3)})</span>
                </dd>
                <dt className="text-ink-2">Run</dt>
                <dd className="truncate font-mono text-xs text-ink">{data.model.run_id}</dd>
              </dl>
              <Link to="/evaluation" className="mt-4 inline-block text-sm text-accent hover:underline">
                Full evaluation →
              </Link>
            </Card>
            <Card title="How to read a score">
              <p className="text-sm text-ink-2">
                A score of 54 means an estimated 54 % chance of a difficulty semester next term. In the
                synthetic test set, {pct(data.model.test_positive_rate)} of students had one. Scores
                prompt a supportive conversation; they never decide anything on their own.
              </p>
              <Link to="/about" className="mt-4 inline-block text-sm text-accent hover:underline">
                Method and limitations →
              </Link>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
