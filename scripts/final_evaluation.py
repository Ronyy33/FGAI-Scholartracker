"""Phase 7: calibration, thresholds, model selection, and the one-time test evaluation.

Usage:
    uv run python scripts/final_evaluation.py --stage validation
        Decide everything on VALIDATION data and freeze it in <run>/selection.json (and a copy in
        reports/model_selection.json, committed BEFORE the test stage, so Git history shows the
        choice was made without test results).
    uv run python scripts/final_evaluation.py --stage test --final
        Score the TEST partition once with the frozen choices; write reports/evaluation_report.md,
        reports/evaluation_metrics.json and reports/figures/*.png; record the evaluation in the
        run manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve

from scholartrack.evaluation.calibration import (
    apply_calibration,
    choose_calibration,
    expected_calibration_error,
    reliability_table,
)
from scholartrack.evaluation.classification import (
    choose_threshold,
    student_bootstrap,
    threshold_metrics,
    top_k_capture,
)
from scholartrack.evaluation.metrics import probability_metrics
from scholartrack.evaluation.plots import (
    LABELS,
    calibration_plot,
    confusion_plot,
    curve_plot,
    risk_band_plot,
)
from scholartrack.evaluation.runner import latest_run_dir, predict_partition
from scholartrack.paths import CONFIG_DIR, REPORTS_DIR
from scholartrack.reporting import fmt, md_table
from scholartrack.risk import load_risk_bands, risk_band, risk_score

MODELS = ("majority", "logistic_regression", "mlp", "lstm")
LEARNED = ("logistic_regression", "mlp", "lstm")
FIGURES = REPORTS_DIR / "figures"


def validation_stage(run_dir: Path, cfg: dict) -> dict:
    _, val = predict_partition(run_dir, "validation")
    y, groups = val["label"].to_numpy(), val["student_key"].to_numpy()
    models = {}
    for name in MODELS:
        p = val[name].to_numpy()
        if name == "majority":  # a constant cannot be recalibrated meaningfully
            calibration = {"method": "none"}
        else:
            calibration = choose_calibration(
                y, p, groups, cfg["calibration"]["cv_folds"], cfg["calibration"]["min_improvement"]
            )
        pc = apply_calibration(p, calibration)
        threshold = choose_threshold(y, pc, cfg["threshold"]["criterion"])
        models[name] = {
            "calibration": calibration,
            "threshold": threshold,
            "validation": {
                **probability_metrics(y, pc),
                "ece": expected_calibration_error(y, pc, cfg["calibration"]["n_bins"]),
                **threshold_metrics(y, pc, threshold),
            },
        }
    rule = cfg["selection"]
    losses = {m: models[m]["validation"][rule["primary_metric"]] for m in LEARNED}
    best = min(losses.values())
    tied = [m for m in LEARNED if losses[m] - best <= rule["tolerance"]]
    chosen = next(m for m in rule["simplicity_order"] if m in tied)
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "rule": rule,
        "tied_within_tolerance": tied,
        "chosen_model": chosen,
        "models": models,
        "evaluation_config": cfg,
    }


def _curves(y, scores: dict[str, np.ndarray]):
    roc, pr = {}, {}
    for name, p in scores.items():
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        roc[name], pr[name] = (fpr, tpr), (recall, precision)
    return roc, pr


def test_stage(run_dir: Path, cfg: dict, selection: dict) -> tuple[dict, str]:
    run, test = predict_partition(run_dir, "test")
    y, groups = test["label"].to_numpy(), test["student_key"].to_numpy()
    k = int(round(cfg["top_k_fraction"] * len(y)))
    n_boot, seed = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
    scores = {
        m: apply_calibration(test[m].to_numpy(), selection["models"][m]["calibration"])
        for m in MODELS
    }

    results = {}
    for name in MODELS:
        p, thr = scores[name], selection["models"][name]["threshold"]

        def stats(idx, p=p, thr=thr):
            prob = probability_metrics(y[idx], p[idx])
            th = threshold_metrics(y[idx], p[idx], thr)
            cap = top_k_capture(y[idx], p[idx], int(round(cfg["top_k_fraction"] * len(idx))))
            return {
                "roc_auc": prob["roc_auc"],
                "pr_auc": prob["pr_auc"],
                "log_loss": prob["log_loss"],
                "brier": prob["brier"],
                "precision": th["precision"],
                "recall": th["recall"],
                "f1": th["f1"],
                "top_k_capture_pct": cap["capture_pct"],
            }

        results[name] = {
            "probability": probability_metrics(y, p),
            "ece": expected_calibration_error(y, p, cfg["calibration"]["n_bins"]),
            "threshold": threshold_metrics(y, p, thr),
            "top_k": top_k_capture(y, p, k),
            "ci95": student_bootstrap(groups, stats, n_boot, seed) if name != "majority" else {},
            "reliability": reliability_table(y, p, cfg["calibration"]["n_bins"]).to_dict("records"),
            "inference_ms_per_1000_rows": 1000
            * 1000
            * test.attrs["inference_seconds"][name]
            / len(y),
        }

    chosen = selection["chosen_model"]
    differences = {}
    for other in LEARNED:
        if other == chosen:
            continue

        def diff(idx, other=other):
            a = probability_metrics(y[idx], scores[chosen][idx])
            b = probability_metrics(y[idx], scores[other][idx])
            return {m: a[m] - b[m] for m in ("roc_auc", "pr_auc", "log_loss", "brier")}

        point = diff(np.arange(len(y)))
        differences[other] = {
            "point": point,
            "ci95": student_bootstrap(groups, diff, n_boot, seed + 1),
        }

    bands = load_risk_bands()
    band = risk_band(risk_score(scores[chosen]), bands)
    band_table = (
        pd.DataFrame({"band": band, "label": y})
        .groupby("band", sort=False)
        .agg(count=("label", "size"), observed_rate=("label", "mean"))
    )
    band_table = band_table.reindex([b.name for b in bands]).fillna(0).reset_index()
    band_table["range"] = [f"{b.minimum}–{b.maximum}" for b in bands]

    FIGURES.mkdir(parents=True, exist_ok=True)
    learned_scores = {m: scores[m] for m in LEARNED}
    roc, pr = _curves(y, learned_scores)
    curve_plot(roc, "roc", y.mean(), FIGURES / "roc_test.png", "test set (synthetic)")
    curve_plot(pr, "pr", y.mean(), FIGURES / "pr_test.png", "test set (synthetic)")
    calibration_plot(
        {m: pd.DataFrame(results[m]["reliability"]) for m in LEARNED},
        FIGURES / "calibration_test.png",
        "test set (synthetic)",
    )
    t = results[chosen]["threshold"]
    confusion_plot(
        t["tn"],
        t["fp"],
        t["fn"],
        t["tp"],
        FIGURES / "confusion_test.png",
        f"{LABELS[chosen]} at threshold {t['threshold']:.3f} (test)",
    )
    risk_band_plot(band_table, FIGURES / "risk_bands_test.png", LABELS[chosen])

    payload = {
        "run_id": run_dir.name,
        "evaluated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "synthetic_data": True,
        "test_rows": int(len(y)),
        "test_students": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
        "chosen_model": chosen,
        "top_k": k,
        "models": results,
        "paired_differences_vs_chosen": differences,
        "risk_bands": band_table.to_dict("records"),
        "training_seconds": {
            m: v.get("seconds")
            for m, v in json.loads((run_dir / "metrics.json").read_text())["chosen"].items()
        },
    }
    return payload, build_report(payload, selection)


def _ci(result: dict, metric: str, digits: int = 3) -> str:
    lo, hi = result["ci95"].get(metric, (np.nan, np.nan))
    return f"[{fmt(lo, digits)}, {fmt(hi, digits)}]" if lo == lo else "—"


def build_report(m: dict, selection: dict) -> str:
    chosen = m["chosen_model"]
    lines: list[str] = []
    w = lines.append
    w("# Evaluation Report — Test Set\n")
    w(
        f"> Generated by `scripts/final_evaluation.py` for run `{m['run_id']}`. **Synthetic "
        "data**: every number measures agreement with the documented synthetic label process, "
        "not real-world predictive ability. Calibration, thresholds and the model choice were "
        "fixed on validation data *before* this test evaluation (`reports/model_selection.json`, "
        "committed beforehand).\n"
    )
    w("## 1. Test set\n")
    w(
        f"{m['test_rows']} labelled student-semesters from {m['test_students']} students never "
        f"used for training or selection; positive (difficulty next semester) rate "
        f"**{m['positive_rate'] * 100:.1f} %**.\n"
    )

    w("## 2. Model selection (decided on validation)\n")
    sel_rows = []
    for name in MODELS:
        s = selection["models"][name]
        sel_rows.append(
            {
                "model": LABELS[name],
                "calibration": s["calibration"]["method"],
                "threshold": fmt(s["threshold"], 3),
                "val log loss": fmt(s["validation"]["log_loss"], 4),
                "val ECE": fmt(s["validation"]["ece"], 4),
                "val F1": fmt(s["validation"]["f1"], 3),
            }
        )
    w(md_table(pd.DataFrame(sel_rows)) + "\n")
    rule = selection["rule"]
    w(
        f"Rule: lowest validation {rule['primary_metric']}; models within {rule['tolerance']} "
        f"of the best are tied, and ties go to the simpler model "
        f"({' > '.join(rule['simplicity_order'])}). Tied: "
        f"{', '.join(LABELS[t] for t in selection['tied_within_tolerance'])}. "
        f"**Chosen: {LABELS[chosen]}.**\n"
    )

    w("## 3. Probability and ranking metrics (95 % CI, bootstrap by student)\n")
    rows = []
    for name in MODELS:
        r = m["models"][name]
        rows.append(
            {
                "model": LABELS[name],
                "ROC-AUC": f"{fmt(r['probability']['roc_auc'], 3)} {_ci(r, 'roc_auc')}",
                "PR-AUC": f"{fmt(r['probability']['pr_auc'], 3)} {_ci(r, 'pr_auc')}",
                "Brier": f"{fmt(r['probability']['brier'], 4)} {_ci(r, 'brier', 4)}",
                "log loss": f"{fmt(r['probability']['log_loss'], 4)} {_ci(r, 'log_loss', 4)}",
                "ECE": fmt(r["ece"], 4),
            }
        )
    w(md_table(pd.DataFrame(rows)) + "\n")
    w(
        f"PR-AUC of a random ranking equals the positive rate ({m['positive_rate']:.3f}). The "
        "majority baseline has no confidence interval: it is a constant.\n"
    )
    w("![ROC curves](figures/roc_test.png) ![Precision–recall curves](figures/pr_test.png)\n")

    w("### 3.1 Are the differences real? (chosen model minus other; 95 % CI)\n")
    drows = []
    for other, d in m["paired_differences_vs_chosen"].items():
        drows.append(
            {
                "comparison": f"{LABELS[chosen]} − {LABELS[other]}",
                **{
                    metric: f"{d['point'][metric]:+.4f} [{d['ci95'][metric][0]:+.4f}, "
                    f"{d['ci95'][metric][1]:+.4f}]"
                    for metric in ("roc_auc", "pr_auc", "log_loss", "brier")
                },
            }
        )
    w(md_table(pd.DataFrame(drows)) + "\n")
    w(
        "An interval that contains 0 means the test data cannot distinguish the two models on "
        "that metric. For log loss and Brier, negative favours the chosen model.\n"
    )

    w("## 4. Decisions at the validation-chosen threshold\n")
    trows = []
    for name in MODELS:
        t, r = m["models"][name]["threshold"], m["models"][name]
        trows.append(
            {
                "model": LABELS[name],
                "threshold": fmt(t["threshold"], 3),
                "accuracy": fmt(t["accuracy"], 3),
                "precision": f"{fmt(t['precision'], 3)} {_ci(r, 'precision')}",
                "recall": f"{fmt(t['recall'], 3)} {_ci(r, 'recall')}",
                "F1": f"{fmt(t['f1'], 3)} {_ci(r, 'f1')}",
                "specificity": fmt(t["specificity"], 3),
                "flagged": f"{t['flagged_share'] * 100:.1f} %",
            }
        )
    w(md_table(pd.DataFrame(trows)) + "\n")
    maj = m["models"]["majority"]["threshold"]
    w(
        'Accuracy is misleading with imbalanced classes: predicting "no difficulty" for '
        f"everyone would already be {(1 - m['positive_rate']) * 100:.1f} % accurate. The majority "
        f"baseline's threshold flags everyone (recall {fmt(maj['recall'], 2)}, precision = the "
        "positive rate); its F1 is the floor any useful model must beat.\n"
    )
    t = m["models"][chosen]["threshold"]
    w(
        f"Confusion matrix of the chosen model ({LABELS[chosen]}): TP {t['tp']}, FP {t['fp']}, "
        f"FN {t['fn']}, TN {t['tn']}.\n"
    )
    w("![Confusion matrix](figures/confusion_test.png)\n")

    positives = m["models"][chosen]["top_k"]["total_positives"]
    max_capture = min(100.0, 100 * m["top_k"] / max(1, positives))
    w(f"## 5. Top-K risk capture (K = {m['top_k']}, 10 % of test student-semesters)\n")
    krows = []
    for name in MODELS:
        c, r = m["models"][name]["top_k"], m["models"][name]
        krows.append(
            {
                "model": LABELS[name],
                "K": c["k"],
                "actual positives in top K": c["positives_in_top_k"],
                "all actual positives": c["total_positives"],
                "capture %": f"{fmt(c['capture_pct'], 1)} {_ci(r, 'top_k_capture_pct', 1)}",
                "precision@K %": fmt(c["precision_at_k_pct"], 1),
            }
        )
    w(md_table(pd.DataFrame(krows)) + "\n")
    w(
        "Capture = positives found among the K highest risk scores ÷ all positives × 100. With "
        f"K = {m['top_k']} and {m['models'][chosen]['top_k']['total_positives']} positives, the "
        f"maximum possible capture is {max_capture:.1f} %. "
        "It is a ranking measure for limited mentoring capacity, not accuracy.\n"
    )

    w("## 6. Calibration\n")
    rel = pd.DataFrame(m["models"][chosen]["reliability"])
    w(md_table(rel, digits=3) + "\n")
    w(
        f"Reliability table for {LABELS[chosen]} (test). ECE = expected calibration error: the "
        "count-weighted average gap between predicted and observed rates.\n"
    )
    w("![Calibration](figures/calibration_test.png)\n")

    w("## 7. Risk bands (chosen model, test set)\n")
    bands = pd.DataFrame(m["risk_bands"])[["band", "range", "count", "observed_rate"]]
    w(md_table(bands, digits=3) + "\n")
    w(
        "Bands are demonstration defaults (`configs/risk_bands.yaml`). If the scores are "
        "calibrated, the observed rate in each band should fall inside its score range.\n"
    )
    w("![Risk bands](figures/risk_bands_test.png)\n")

    w("## 8. Cost\n")
    crows = [
        {
            "model": LABELS[n],
            "training seconds (chosen setting)": fmt(m["training_seconds"].get(n), 1),
            "inference ms per 1,000 rows": fmt(m["models"][n]["inference_ms_per_1000_rows"], 2),
        }
        for n in MODELS
    ]
    w(md_table(pd.DataFrame(crows)) + "\n")
    w("CPU only (4 threads). Timings vary by machine; they are shown for relative comparison.\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["validation", "test"], required=True)
    parser.add_argument("--final", action="store_true", help="required for --stage test")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load((CONFIG_DIR / "evaluation.yaml").read_text())
    run_dir = latest_run_dir() if args.run is None else latest_run_dir().parent / args.run
    selection_path = run_dir / "selection.json"

    if args.stage == "validation":
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("test_partition_evaluated"):
            raise SystemExit("This run's test set was already scored; its selection is frozen.")
        selection = validation_stage(run_dir, cfg)
        text = json.dumps(selection, indent=2) + "\n"
        selection_path.write_text(text)
        (REPORTS_DIR / "model_selection.json").write_text(text)
        print(
            f"Chosen model: {selection['chosen_model']} "
            f"(tied: {selection['tied_within_tolerance']})"
        )
        for name, s in selection["models"].items():
            v = s["validation"]
            print(
                f"  {name:20s} calibration={s['calibration']['method']:5s} "
                f"threshold={s['threshold']:.3f} val log loss={v['log_loss']:.4f} "
                f"ECE={v['ece']:.4f} F1={v['f1']:.3f}"
            )
        return

    if not args.final:
        raise SystemExit("Refusing to score the test set without --final.")
    if not selection_path.exists():
        raise SystemExit("Run --stage validation first: choices must be frozen before testing.")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("test_partition_evaluated"):
        raise SystemExit("The test set of this run was already evaluated once.")
    selection = json.loads(selection_path.read_text())
    payload, report = test_stage(run_dir, cfg, selection)
    (REPORTS_DIR / "evaluation_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (REPORTS_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")
    manifest["test_partition_evaluated"] = True
    manifest.setdefault("test_evaluations_utc", []).append(payload["evaluated_utc"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote reports/evaluation_report.md, reports/evaluation_metrics.json, "
        f"{len(list(FIGURES.glob('*.png')))} figures; test evaluation recorded in manifest."
    )


if __name__ == "__main__":
    main()
