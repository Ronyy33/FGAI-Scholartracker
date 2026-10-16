import Papa from "papaparse";
import { useMemo, useState } from "react";
import { PredictionPanel } from "../components/PredictionPanel";
import { BandChip, Card, ErrorState, Loading, PageHeader, StatTile } from "../components/ui";
import { ApiError, apiFetch } from "../lib/api";
import { downloadText, predictionsToCsv } from "../lib/format";
import type { BatchResponse, Issue, Prediction } from "../lib/types";

const PREVIEW_COLUMNS = 8;
/** Direct identifiers other than the ID are not echoed back, even from the user's own file. */
const HIDDEN_COLUMN = /name/i;

interface Preview {
  columns: string[];
  rows: string[][];
  totalColumns: number;
  hidden: string[];
}

export function previewCsv(text: string): Preview {
  const parsed = Papa.parse<string[]>(text.replace(/^\uFEFF/, ""), { preview: 6, skipEmptyLines: true });
  const [header = [], ...rows] = parsed.data;
  const keep = header.map((c, i) => (HIDDEN_COLUMN.test(c) ? -1 : i)).filter((i) => i >= 0).slice(0, PREVIEW_COLUMNS);
  return {
    columns: keep.map((i) => header[i]),
    rows: rows.map((r) => keep.map((i) => r[i] ?? "")),
    totalColumns: header.length,
    hidden: header.filter((c) => HIDDEN_COLUMN.test(c)),
  };
}

function IssueTable({ issues }: { issues: Issue[] }) {
  const [severity, setSeverity] = useState<"all" | Issue["severity"]>("all");
  const shown = issues.filter((i) => severity === "all" || i.severity === severity);
  return (
    <>
      <label className="mb-3 block text-sm text-ink-2">
        Show
        <select value={severity} onChange={(e) => setSeverity(e.target.value as typeof severity)} className="ml-2 rounded-md border border-line bg-surface px-2 py-1 text-ink">
          <option value="all">all ({issues.length})</option>
          <option value="error">errors — row not used</option>
          <option value="invalid">invalid — value treated as missing</option>
          <option value="warning">warnings — value kept</option>
        </select>
      </label>
      <div className="max-h-72 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-surface text-left text-xs uppercase tracking-wide text-muted">
            <tr><th className="py-1 pr-3 font-medium">Severity</th><th className="pr-3 font-medium">Row</th><th className="pr-3 font-medium">Column</th><th className="font-medium">Problem</th></tr>
          </thead>
          <tbody>
            {shown.map((i, n) => (
              <tr key={n} className="border-t border-line align-baseline">
                <td className="py-1 pr-3 text-ink-2">{i.severity}</td>
                <td className="pr-3 tabular-nums">{i.row ?? "—"}</td>
                <td className="pr-3 font-mono text-xs">{i.column ?? "—"}</td>
                <td className="text-ink">{i.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [result, setResult] = useState<BatchResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [desc, setDesc] = useState(true);

  const sorted = useMemo<Prediction[]>(
    () => (result ? [...result.predictions].sort((a, b) => (desc ? b.risk_score - a.risk_score : a.risk_score - b.risk_score)) : []),
    [result, desc],
  );

  async function choose(f: File | null) {
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(f ? previewCsv(await f.text()) : null);
  }

  async function upload() {
    if (!file) return;
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      setResult(await apiFetch<BatchResponse>("/api/predict/batch", { method: "POST", body: form }));
    } catch (e) {
      setError(e as ApiError);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeader
        title="Batch prediction"
        intro="Upload a CSV in the college format (like the reference file: one row per student per semester). Each student's latest semester is scored for the next one. The file is processed in memory and not stored."
      />
      <div className="space-y-4">
        <Card title="1. Choose a file">
          <input
            type="file"
            accept=".csv,text/csv"
            aria-label="CSV file"
            onChange={(e) => choose(e.target.files?.[0] ?? null)}
            className="block text-sm text-ink-2 file:mr-3 file:rounded-md file:border file:border-line file:bg-surface-2 file:px-3 file:py-1.5 file:text-ink"
          />
          {preview && (
            <div className="mt-4">
              <p className="mb-2 text-sm text-ink-2">
                Preview: first {preview.rows.length} rows, {preview.columns.length} of {preview.totalColumns} columns
                {preview.hidden.length > 0 && ` (${preview.hidden.join(", ")} not shown; names are never used by the model)`}
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead><tr>{preview.columns.map((c) => <th key={c} className="border-b border-line px-2 py-1 text-left font-mono font-medium text-ink-2">{c}</th>)}</tr></thead>
                  <tbody>{preview.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className="border-b border-line px-2 py-1 font-mono">{v}</td>)}</tr>)}</tbody>
                </table>
              </div>
            </div>
          )}
          <button
            onClick={upload}
            disabled={!file || busy}
            className="mt-4 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            {busy ? "Scoring…" : "2. Validate and score"}
          </button>
        </Card>

        {busy && <Loading label="Validating and scoring…" />}
        {error && (
          <div>
            <ErrorState error={error} />
            {Array.isArray(error.details) && error.details.length > 0 && (
              <Card title="Problems found"><IssueTable issues={error.details as Issue[]} /></Card>
            )}
          </div>
        )}

        {result && (
          <>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              <StatTile label="Students scored" value={result.summary.students_scored} note={`${result.summary.rows_received} rows received`} />
              <StatTile label="Rows not used" value={result.summary.rows_rejected} note="errors: missing ID / semester, duplicates" />
              <StatTile label="Values treated as missing" value={result.summary.invalid_values} note="impossible values, never guessed" />
              <StatTile label="Above review threshold" value={result.summary.flagged} note={`High ${result.summary.band_counts.High} · Moderate ${result.summary.band_counts.Moderate} · Low ${result.summary.band_counts.Low}`} />
            </div>
            {result.issues.length > 0 && (
              <Card title="Validation report" subtitle="Row numbers refer to data rows (spreadsheet line = row + 1)">
                <IssueTable issues={result.issues} />
              </Card>
            )}
            <Card
              title="Predictions"
              subtitle="Select a row for its explanation and suggestions"
              actions={
                <button
                  onClick={() => downloadText("scholartrack_predictions.csv", predictionsToCsv(result.predictions))}
                  className="rounded-md border border-line px-3 py-1.5 text-sm text-ink hover:bg-surface-2"
                >
                  Download CSV
                </button>
              }
            >
              <div className="max-h-[32rem] overflow-auto">
                <table className="w-full min-w-[560px] text-sm">
                  <thead className="sticky top-0 bg-surface text-left text-xs uppercase tracking-wide text-muted">
                    <tr>
                      <th className="py-2 font-medium">Student ID</th>
                      <th className="font-medium">Semester</th>
                      <th className="text-right font-medium">
                        <button onClick={() => setDesc((d) => !d)} aria-label="Sort by risk" className="uppercase tracking-wide hover:text-ink">Risk {desc ? "↓" : "↑"}</button>
                      </th>
                      <th className="pl-3 font-medium">Band</th>
                      <th className="font-medium">Main factor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sorted.map((p) => (
                      <tr key={p.prn} className="cursor-pointer border-t border-line hover:bg-surface-2" onClick={() => setOpen(open === p.prn ? null : p.prn)}>
                        <td className="py-2 font-mono">{p.prn}</td>
                        <td className="text-ink-2">{p.semester} → {p.predicts_semester}</td>
                        <td className="text-right tabular-nums">{p.risk_score}</td>
                        <td className="pl-3"><BandChip band={p.risk_band} /></td>
                        <td className="text-ink-2">{p.explanation[0]?.label ?? "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {open && result.predictions.find((p) => p.prn === open) && (
                <div className="mt-4 border-t border-line pt-4">
                  <PredictionPanel p={result.predictions.find((p) => p.prn === open)!} />
                </div>
              )}
            </Card>
          </>
        )}
      </div>
    </>
  );
}
