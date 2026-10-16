import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PredictionPanel } from "../components/PredictionPanel";
import { BandChip, Card, Empty, ErrorState, Loading, PageHeader } from "../components/ui";
import { useApi } from "../lib/api";
import { num } from "../lib/format";
import { usePalette } from "../lib/palette";
import type { BandName, StudentDetail, StudentRow } from "../lib/types";

const PAGE_SIZE = 25;

type SortKey = "risk_score" | "prn" | "sgpa_current" | "attendance_pct" | "n_failed";

export function sortRows(rows: StudentRow[], key: SortKey, descending: boolean): StudentRow[] {
  const sign = descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    const x = a[key];
    const y = b[key];
    if (x === null) return 1;
    if (y === null) return -1;
    return (x < y ? -1 : x > y ? 1 : 0) * sign;
  });
}

export function StudentsPage() {
  const [query, setQuery] = useState("");
  const [band, setBand] = useState<BandName | "">("");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "risk_score", desc: true });
  const [page, setPage] = useState(0);
  const params = new URLSearchParams({ limit: "1000" });
  if (query) params.set("q", query);
  if (band) params.set("band", band);
  const { data, error, loading, reload } = useApi<{ total: number; items: StudentRow[] }>(
    `/api/students?${params.toString()}`,
  );
  const rows = useMemo(() => (data ? sortRows(data.items, sort.key, sort.desc) : []), [data, sort]);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const current = Math.min(page, pages - 1);
  const shown = rows.slice(current * PAGE_SIZE, (current + 1) * PAGE_SIZE);

  const header = (key: SortKey, label: string, right = false) => (
    <th className={`py-2 font-medium ${right ? "text-right" : "text-left"}`}>
      <button
        className="hover:text-ink"
        onClick={() => {
          setSort((s) => ({ key, desc: s.key === key ? !s.desc : true }));
          setPage(0);
        }}
        aria-label={`Sort by ${label}`}
      >
        {label}
        {sort.key === key ? (sort.desc ? " ↓" : " ↑") : ""}
      </button>
    </th>
  );

  return (
    <>
      <PageHeader
        title="Student risk analysis"
        intro="Currently enrolled synthetic students, scored for their next semester. Select a student for the explanation, history and supportive suggestions."
      />
      <Card>
        <div className="mb-4 flex flex-wrap gap-3">
          <label className="text-sm text-ink-2">
            Search ID
            <input
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setPage(0);
              }}
              placeholder="e.g. SYN24CE0127"
              className="ml-2 rounded-md border border-line bg-surface px-2 py-1 text-ink"
            />
          </label>
          <label className="text-sm text-ink-2">
            Band
            <select
              value={band}
              onChange={(e) => {
                setBand(e.target.value as BandName | "");
                setPage(0);
              }}
              className="ml-2 rounded-md border border-line bg-surface px-2 py-1 text-ink"
            >
              <option value="">All</option>
              <option>High</option>
              <option>Moderate</option>
              <option>Low</option>
            </select>
          </label>
          {data && <span className="self-center text-sm text-muted">{data.total} students</span>}
        </div>
        {loading && <Loading />}
        {error && <ErrorState error={error} onRetry={reload} />}
        {data && rows.length === 0 && <Empty>No student matches these filters.</Empty>}
        {rows.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead className="text-xs uppercase tracking-wide text-muted">
                <tr>
                  {header("prn", "Student ID")}
                  <th className="py-2 text-left font-medium">Semester</th>
                  {header("risk_score", "Risk", true)}
                  <th className="py-2 pl-3 text-left font-medium">Band</th>
                  {header("sgpa_current", "SGPA", true)}
                  {header("attendance_pct", "Attendance %", true)}
                  {header("n_failed", "Not passed", true)}
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => (
                  <tr key={r.prn} className="border-t border-line hover:bg-surface-2">
                    <td className="py-2">
                      <Link to={`/students/${r.prn}`} className="font-mono text-accent hover:underline">{r.prn}</Link>
                    </td>
                    <td className="py-2 text-ink-2">{r.semester} → {r.semester + 1}</td>
                    <td className="py-2 text-right tabular-nums text-ink">{r.risk_score}</td>
                    <td className="py-2 pl-3"><BandChip band={r.risk_band} /></td>
                    <td className="py-2 text-right tabular-nums">{num(r.sgpa_current)}</td>
                    <td className="py-2 text-right tabular-nums">{num(r.attendance_pct, 1)}</td>
                    <td className="py-2 text-right tabular-nums">{num(r.n_failed, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-3 flex items-center justify-between text-sm text-ink-2">
              <span className="tabular-nums">
                {current * PAGE_SIZE + 1}–{Math.min((current + 1) * PAGE_SIZE, rows.length)} of {rows.length}
              </span>
              <span className="flex gap-2">
                <button disabled={current === 0} onClick={() => setPage(current - 1)} className="rounded-md border border-line px-3 py-1 hover:bg-surface-2 disabled:opacity-40">Previous</button>
                <button disabled={current >= pages - 1} onClick={() => setPage(current + 1)} className="rounded-md border border-line px-3 py-1 hover:bg-surface-2 disabled:opacity-40">Next</button>
              </span>
            </div>
          </div>
        )}
      </Card>
    </>
  );
}

function HistoryChart({ data, dataKey, label, domain, ticks }: {
  data: StudentDetail["history"];
  dataKey: "sgpa" | "attendance_pct";
  label: string;
  domain: [number, number];
  ticks: number[];
}) {
  const palette = usePalette();
  return (
    <div>
      <p className="mb-1 text-sm font-medium text-ink">{label}</p>
      <div className="h-44">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid stroke={palette.grid} vertical={false} />
            <XAxis dataKey="semester" stroke={palette.axis} tick={{ fill: palette.muted, fontSize: 12 }} tickFormatter={(s) => `S${s}`} />
            <YAxis domain={domain} ticks={ticks} stroke={palette.axis} tick={{ fill: palette.muted, fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: palette.surface, borderColor: palette.grid, color: palette.ink }}
              formatter={(v) => [num(Number(v)), label]}
              labelFormatter={(s) => `Semester ${s}`}
            />
            <Line type="linear" dataKey={dataKey} stroke={palette.series[0]} strokeWidth={2} dot={{ r: 4 }} connectNulls isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function StudentDetailPage() {
  const { prn = "" } = useParams();
  const { data, error, loading, reload } = useApi<StudentDetail>(`/api/students/${encodeURIComponent(prn)}`);
  return (
    <>
      <PageHeader title={`Student ${prn}`} intro={data ? `Synthetic cohort ${data.cohort}` : undefined} />
      <Link to="/students" className="mb-4 inline-block text-sm text-accent hover:underline">← All students</Link>
      {loading && <Loading />}
      {error && <ErrorState error={error} onRetry={reload} />}
      {data && (
        <div className="space-y-4">
          {data.prediction ? <PredictionPanel p={data.prediction} /> : <Card><p className="text-sm text-ink-2">{data.message}</p></Card>}
          <Card title="Semester history" subtitle="Computed from the student's marks; the LSTM reads this sequence, the other models the latest semester">
            <div className="grid gap-4 md:grid-cols-2">
              <HistoryChart data={data.history} dataKey="sgpa" label="SGPA" domain={[0, 10]} ticks={[0, 2, 4, 6, 8, 10]} />
              <HistoryChart data={data.history} dataKey="attendance_pct" label="Attendance %" domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} />
            </div>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full min-w-[480px] text-sm">
                <thead className="text-left text-xs uppercase tracking-wide text-muted">
                  <tr><th className="py-1 font-medium">Semester</th><th className="font-medium">Year</th><th className="text-right font-medium">SGPA</th><th className="text-right font-medium">Marks %</th><th className="text-right font-medium">Attendance %</th><th className="text-right font-medium">Not passed</th></tr>
                </thead>
                <tbody>
                  {data.history.map((h) => (
                    <tr key={h.semester} className="border-t border-line">
                      <td className="py-1">{h.semester}</td>
                      <td>{h.academic_year}</td>
                      <td className="text-right tabular-nums">{num(h.sgpa)}</td>
                      <td className="text-right tabular-nums">{num(h.credit_weighted_pct, 1)}</td>
                      <td className="text-right tabular-nums">{num(h.attendance_pct, 1)}</td>
                      <td className="text-right tabular-nums">{num(h.n_failed, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </>
  );
}
