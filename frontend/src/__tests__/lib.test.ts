import { afterEach, describe, expect, it, vi } from "vitest";
import { apiFetch } from "../lib/api";
import { num, pct, predictionsToCsv } from "../lib/format";
import type { Prediction } from "../lib/types";
import { previewCsv } from "../pages/BatchPage";
import { sortRows } from "../pages/StudentsPage";

const prediction = (prn: string, score: number): Prediction => ({
  prn, semester: 5, predicts_semester: 6, default_model: "logistic_regression",
  probability: score / 100, risk_score: score, risk_band: score >= 60 ? "High" : "Low", flagged: score >= 31,
  models: {
    logistic_regression: { probability: score / 100, risk_score: score, risk_band: "Low", flagged: false },
    mlp: { probability: 0.1, risk_score: 10, risk_band: "Low", flagged: false },
    lstm: { probability: 0.2, risk_score: 20, risk_band: "Low", flagged: false },
  },
  indicators: {},
  explanation: [{ group: "results", label: "Current-semester results", effect_points: 12.5, direction: "raises risk", indicators: [] }],
  explanation_note: null,
  recommendations: { items: [{ id: "attendance", observation: "o", suggestion: "s" }], message: null, disclaimer: "d" },
});

describe("formatting", () => {
  it("formats percentages and numbers, with a dash for missing values", () => {
    expect(pct(0.204)).toBe("20.4 %");
    expect(num(7.1234)).toBe("7.12");
    expect(num(3)).toBe("3");
    expect(num(null)).toBe("—");
  });

  it("exports one CSV row per student with all model scores and factors", () => {
    const csv = predictionsToCsv([prediction("SYN1", 72), prediction("SYN2", 12)]);
    const lines = csv.trim().split(/\r?\n/);
    expect(lines).toHaveLength(3);
    expect(lines[0]).toContain("risk_score");
    expect(lines[0]).toContain("lstm_risk_score");
    expect(lines[1]).toContain("SYN1");
    expect(lines[1]).toContain("Current-semester results (+12.5)");
  });
});

describe("CSV preview", () => {
  it("strips the byte-order mark and caps the preview size", () => {
    const header = Array.from({ length: 12 }, (_, i) => `c${i}`).join(",");
    const rows = Array.from({ length: 10 }, (_, r) => Array.from({ length: 12 }, () => r).join(","));
    const preview = previewCsv(`\uFEFF${header}\n${rows.join("\n")}`);
    expect(preview.columns[0]).toBe("c0");
    expect(preview.columns).toHaveLength(8);
    expect(preview.totalColumns).toBe(12);
    expect(preview.rows).toHaveLength(5);
  });

  it("never shows name columns", () => {
    const preview = previewCsv("PRN,student_name,semester\nB1,Some Person,3\n");
    expect(preview.columns).toEqual(["PRN", "semester"]);
    expect(preview.rows).toEqual([["B1", "3"]]);
    expect(preview.hidden).toEqual(["student_name"]);
    expect(JSON.stringify(preview)).not.toContain("Some Person");
  });
});

describe("sorting students", () => {
  it("sorts by risk and keeps missing values last", () => {
    const rows = [
      { prn: "A", cohort: 2024, semester: 5, risk_score: 10, risk_band: "Low" as const, flagged: false, sgpa_current: null, attendance_pct: 80, n_failed: 0 },
      { prn: "B", cohort: 2024, semester: 5, risk_score: 90, risk_band: "High" as const, flagged: true, sgpa_current: 4.1, attendance_pct: 60, n_failed: 2 },
    ];
    expect(sortRows(rows, "risk_score", true).map((r) => r.prn)).toEqual(["B", "A"]);
    expect(sortRows(rows, "sgpa_current", false).map((r) => r.prn)).toEqual(["B", "A"]);
  });
});

describe("apiFetch", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("surfaces the API's own error message and code", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ error: { code: "file_too_large", message: "The file exceeds 2048 KB." } }),
      { status: 413 })));
    await expect(apiFetch("/api/predict/batch")).rejects.toMatchObject({
      status: 413, code: "file_too_large", message: "The file exceeds 2048 KB.",
    });
  });

  it("reports an unreachable server clearly", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("failed"); }));
    await expect(apiFetch("/api/health")).rejects.toMatchObject({ code: "network" });
  });
});
