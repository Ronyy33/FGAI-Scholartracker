import { useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Scatter,
  ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { Card, Empty, ErrorState, Loading, PageHeader } from "../components/ui";
import { useApi } from "../lib/api";
import { num } from "../lib/format";
import { usePalette } from "../lib/palette";
import type { Analytics } from "../lib/types";

const DASHES = [undefined, "6 4", "2 3", "8 3 2 3", "1 3", "10 4"];

export function AnalyticsPage() {
  const [cohort, setCohort] = useState("");
  const [semester, setSemester] = useState("");
  const params = new URLSearchParams();
  if (cohort) params.set("cohort", cohort);
  if (semester) params.set("semester", semester);
  const { data, error, loading, reload } = useApi<Analytics>(`/api/analytics?${params.toString()}`);
  const palette = usePalette();
  const axis = { stroke: palette.axis, tick: { fill: palette.muted, fontSize: 12 } };
  const tooltip = { contentStyle: { background: palette.surface, borderColor: palette.grid, color: palette.ink } };

  const cohorts = data ? [...new Set(data.sgpa_trends.map((t) => t.cohort))] : [];
  const trendRows = data
    ? [...new Set(data.sgpa_trends.map((t) => t.semester))].sort((a, b) => a - b).map((s) => ({
        semester: s,
        ...Object.fromEntries(data.sgpa_trends.filter((t) => t.semester === s).map((t) => [String(t.cohort), t.mean_sgpa])),
      }))
    : [];

  return (
    <>
      <PageHeader
        title="Analytics"
        intro="Aggregate views of the synthetic population. Groups with fewer than 10 records are suppressed so that no small group can be singled out."
      />
      <div className="mb-4 flex flex-wrap gap-3" role="group" aria-label="Filters">
        <label className="text-sm text-ink-2">
          Cohort
          <select value={cohort} onChange={(e) => setCohort(e.target.value)} className="ml-2 rounded-md border border-line bg-surface px-2 py-1 text-ink">
            <option value="">All</option>
            {(data?.filters.cohorts ?? []).map((c) => <option key={c}>{c}</option>)}
          </select>
        </label>
        <label className="text-sm text-ink-2">
          Semester
          <select value={semester} onChange={(e) => setSemester(e.target.value)} className="ml-2 rounded-md border border-line bg-surface px-2 py-1 text-ink">
            <option value="">All</option>
            {[1, 2, 3, 4, 5, 6, 7, 8].map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
      </div>
      {loading && <Loading />}
      {error && <ErrorState error={error} onRetry={reload} />}
      {data && (
        <div className="grid gap-4 xl:grid-cols-2">
          <Card title="Risk score distribution" subtitle="Currently enrolled students (cohort filter applies)">
            {data.risk_distribution.every((d) => d.count === 0) ? (
              <Empty>No currently enrolled students in this selection.</Empty>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data.risk_distribution} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                    <CartesianGrid stroke={palette.grid} vertical={false} />
                    <XAxis dataKey="from" {...axis} tickFormatter={(v) => `${v}`} />
                    <YAxis {...axis} allowDecimals={false} />
                    <Tooltip {...tooltip} formatter={(v) => [v, "students"]} labelFormatter={(v) => `Scores ${v}–${Number(v) + 9}`} />
                    <Bar dataKey="count" fill={palette.series[0]} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </Card>

          <Card title="Attendance versus marks" subtitle={`Student-semesters per 5 × 5 % cell; ${data.attendance_vs_marks.suppressed_cells} small cells suppressed`}>
            {data.attendance_vs_marks.cells.length === 0 ? (
              <Empty>Every cell is below the minimum group size.</Empty>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: -8 }}>
                    <CartesianGrid stroke={palette.grid} />
                    <XAxis type="number" dataKey="attendance_from" name="Attendance %" domain={[40, 100]} {...axis} />
                    <YAxis type="number" dataKey="marks_from" name="Credit-weighted marks %" domain={[20, 100]} {...axis} />
                    <ZAxis type="number" dataKey="count" range={[20, 400]} name="Student-semesters" />
                    <Tooltip {...tooltip} cursor={{ strokeDasharray: "3 3" }} />
                    <Scatter data={data.attendance_vs_marks.cells} fill={palette.series[0]} fillOpacity={0.6} stroke={palette.surface} />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            )}
            <p className="mt-2 text-xs text-muted">Bubble area = number of student-semesters. Association only, not cause.</p>
          </Card>

          <Card title="Mean SGPA by semester and cohort" subtitle="Groups with ≥ 10 students">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trendRows} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                  <CartesianGrid stroke={palette.grid} vertical={false} />
                  <XAxis dataKey="semester" {...axis} tickFormatter={(s) => `S${s}`} />
                  <YAxis domain={[5, 9]} {...axis} />
                  <Tooltip {...tooltip} formatter={(v, name) => [num(Number(v)), `Cohort ${name}`]} labelFormatter={(s) => `Semester ${s}`} />
                  <Legend iconType="plainline" itemSorter={null} formatter={(name) => `Cohort ${name}`} wrapperStyle={{ fontSize: 12, color: palette.ink2 }} />
                  {cohorts.map((c, i) => (
                    <Line key={c} dataKey={String(c)} stroke={palette.series[i % 6]} strokeDasharray={DASHES[i % 6]} strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card title={`Subjects in semester ${data.subjects.semester}`} subtitle="Mean subject total and share of results below 40 %">
            <div className="max-h-72 overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface text-left text-xs uppercase tracking-wide text-muted">
                  <tr><th className="py-1 font-medium">Subject</th><th className="text-right font-medium">Results</th><th className="text-right font-medium">Mean %</th><th className="text-right font-medium">Below 40 %</th><th className="text-right font-medium">Attendance %</th></tr>
                </thead>
                <tbody>
                  {data.subjects.items.map((s) => (
                    <tr key={s.subject_code} className="border-t border-line">
                      <td className="py-1 font-mono text-xs">{s.subject_code}</td>
                      <td className="text-right tabular-nums">{s.n}</td>
                      <td className="text-right tabular-nums">{num(s.mean_total_pct, 1)}</td>
                      <td className="text-right tabular-nums">{num(s.below_40_pct, 1)}</td>
                      <td className="text-right tabular-nums">{num(s.mean_attendance, 1)}</td>
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
