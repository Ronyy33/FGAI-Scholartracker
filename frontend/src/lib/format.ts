import Papa from "papaparse";
import type { ModelName, Prediction } from "./types";

export const MODEL_LABELS: Record<ModelName, string> = {
  logistic_regression: "Logistic regression",
  mlp: "MLP",
  lstm: "LSTM",
};

export const MODELS: ModelName[] = ["logistic_regression", "mlp", "lstm"];

export function pct(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : `${(100 * value).toFixed(digits)} %`;
}

export function num(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

/** Predictions as CSV text for download (one row per student). */
export function predictionsToCsv(predictions: Prediction[]): string {
  return Papa.unparse(
    predictions.map((p) => ({
      PRN: p.prn,
      scored_semester: p.semester,
      predicts_semester: p.predicts_semester,
      risk_score: p.risk_score,
      risk_band: p.risk_band,
      probability: p.probability,
      flagged_at_threshold: p.flagged ? "yes" : "no",
      default_model: p.default_model,
      ...Object.fromEntries(MODELS.map((m) => [`${m}_risk_score`, p.models[m].risk_score])),
      main_factors: p.explanation.map((f) => `${f.label} (${f.effect_points > 0 ? "+" : ""}${f.effect_points})`).join("; "),
      suggestions: (p.recommendations?.items ?? []).map((r) => r.id).join("; "),
    })),
  );
}

export function downloadText(filename: string, text: string, type = "text/csv"): void {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
