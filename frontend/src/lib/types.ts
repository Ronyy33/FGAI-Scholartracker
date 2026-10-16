// Shapes of the ScholarTrack API responses (see /api/docs for the authoritative schema).

export type BandName = "Low" | "Moderate" | "High";
export type ModelName = "logistic_regression" | "mlp" | "lstm";

export interface ModelScore {
  probability: number;
  risk_score: number;
  risk_band: BandName;
  flagged: boolean;
}

export interface Indicator {
  feature: string;
  label: string;
  value: number | null;
  typical_value: number;
  unit: string;
}

export interface Factor {
  group: string;
  label: string;
  effect_points: number;
  direction: "raises risk" | "lowers risk";
  indicators: Indicator[];
}

export interface Recommendations {
  items: { id: string; observation: string; suggestion: string }[];
  message: string | null;
  disclaimer: string;
}

export interface Prediction {
  prn: string;
  semester: number;
  predicts_semester: number;
  default_model: ModelName;
  probability: number;
  risk_score: number;
  risk_band: BandName;
  flagged: boolean;
  models: Record<ModelName, ModelScore>;
  indicators: Record<string, number | null>;
  explanation: Factor[];
  explanation_note: string | null;
  recommendations: Recommendations | null;
}

export interface Issue {
  severity: "error" | "invalid" | "warning";
  code: string;
  row: number | null;
  column: string | null;
  value: string | null;
  message: string;
}

export interface Overview {
  synthetic_data: boolean;
  records_processed: number;
  students: number;
  current_academic_year: string;
  current_students_scored: number;
  flagged: number;
  risk_bands: { band: BandName; range: [number, number]; count: number; percent: number }[];
  model: {
    run_id: string;
    default_model: ModelName;
    test_roc_auc: number;
    test_pr_auc: number;
    test_positive_rate: number;
  };
}

export interface StudentRow {
  prn: string;
  cohort: number;
  semester: number;
  risk_score: number;
  risk_band: BandName;
  flagged: boolean;
  sgpa_current: number | null;
  attendance_pct: number | null;
  n_failed: number | null;
}

export interface StudentDetail {
  prn: string;
  cohort: number;
  history: {
    semester: number;
    academic_year: string;
    sgpa: number | null;
    credit_weighted_pct: number | null;
    attendance_pct: number | null;
    n_failed: number | null;
  }[];
  prediction: (Omit<Prediction, "prn"> & { prn?: string }) | null;
  message: string | null;
}

export interface BatchResponse {
  synthetic_model: boolean;
  summary: {
    rows_received: number;
    rows_used: number;
    rows_rejected: number;
    students_scored: number;
    invalid_values: number;
    warnings: number;
    band_counts: Record<BandName, number>;
    flagged: number;
  };
  predictions: Prediction[];
  issues: Issue[];
}

export interface Analytics {
  min_group_size: number;
  filters: { cohort: number | null; semester: number | null; cohorts: number[]; semesters: number[] };
  risk_distribution: { from: number; to: number; count: number }[];
  attendance_vs_marks: {
    cells: { attendance_from: number; marks_from: number; count: number }[];
    suppressed_cells: number;
  };
  sgpa_trends: { cohort: number; semester: number; mean_sgpa: number; n: number }[];
  subjects: {
    semester: number;
    items: { subject_code: string; n: number; mean_total_pct: number; below_40_pct: number; mean_attendance: number }[];
  };
}

export interface CurvePoint {
  x: number;
  y: number;
}

export interface ReliabilityRow {
  bin: string;
  count: number;
  mean_predicted: number | null;
  observed_rate: number | null;
}

export interface ModelEvaluation {
  probability: { roc_auc: number; pr_auc: number; brier: number; log_loss: number; positive_rate: number };
  ece: number;
  threshold: {
    threshold: number; tp: number; fp: number; fn: number; tn: number;
    accuracy: number; precision: number; recall: number; f1: number; specificity: number; flagged_share: number;
  };
  top_k: { k: number; positives_in_top_k: number; total_positives: number; capture_pct: number; precision_at_k_pct: number };
  ci95: Record<string, [number, number]>;
}

export interface Metrics {
  evaluation: {
    run_id: string;
    test_rows: number;
    test_students: number;
    positive_rate: number;
    chosen_model: ModelName;
    top_k: number;
    models: Record<ModelName | "majority", ModelEvaluation>;
    risk_bands: { band: BandName; range: string; count: number; observed_rate: number }[];
  };
  curves: { roc: Record<ModelName, CurvePoint[]>; pr: Record<ModelName, CurvePoint[]>; reliability: Record<ModelName, ReliabilityRow[]> };
  selection: {
    rule: { primary_metric: string; tolerance: number; simplicity_order: ModelName[] };
    chosen_model: ModelName;
    tied_within_tolerance: ModelName[];
  };
}
