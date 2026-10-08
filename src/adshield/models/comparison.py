"""Honest model-vs-baseline comparison report.

Renders ``reports/model_comparison.md`` from a run's metrics. The verdict is
computed from explicit margins, so the report cannot claim a win the numbers
do not show.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

Verdict = str  # "clearly_better" | "not_clearly_better" | "worse"


def verdict(
    model: dict[str, Any], rules: dict[str, Any], min_margin: float, min_minority_rows: int = 0
) -> Verdict:
    """Compare model and rule baseline on the minority-class view.

    Uses threshold-free PR-AUC and F1 at each system's own operating point.
    The minority-class view is used because with the proxy label the fraud
    class can be the majority, which makes target-class F1 near 1 for any
    system that flags almost everything.

    - ``inconclusive``: fewer than ``min_minority_rows`` minority-class rows in
      the test set; differences that small are noise, so no claim is made.
    - ``clearly_better``: model leads on both by at least ``min_margin``.
    - ``worse``: model trails on either by at least ``min_margin``.
    - ``not_clearly_better``: anything else (tie, mixed, or small gains).
    """
    n_minority = round(model["minority_prevalence"] * model["n_rows"])
    if n_minority < min_minority_rows:
        return "inconclusive"
    d_ap = model["pr_auc_minority"] - rules["pr_auc_minority"]
    d_f1 = model["minority_view"]["f1"] - rules["minority_view"]["f1"]
    if d_ap >= min_margin and d_f1 >= min_margin:
        return "clearly_better"
    if d_ap <= -min_margin or d_f1 <= -min_margin:
        return "worse"
    return "not_clearly_better"


_VERDICT_TEXT = {
    "inconclusive": "**Inconclusive.** The test set has too few minority-class clicks for the metric "
    "differences below to be trusted. Re-run on the full dataset before drawing any conclusion.",
    "clearly_better": "The trained model **clearly beats** the rule baseline on this test set "
    "(both PR-AUC and F1 improve by at least the margin).",
    "not_clearly_better": "The trained model does **NOT clearly beat** the rule baseline on this test "
    "set. Differences are within the margin or point in different directions; "
    "the rule engine remains a competitive, fully explainable alternative.",
    "worse": "The trained model is **WORSE** than the rule baseline on at least one headline "
    "metric by more than the margin. Do not ship it in place of the rules on this evidence.",
}


def _row(name: str, m: dict[str, Any], view: str) -> str:
    v = m[view]
    return f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} |"


def _minority_row(name: str, m: dict[str, Any]) -> str:
    v = m["minority_view"]
    return (
        f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} "
        f"| {m['pr_auc_minority']:.4f} | {m['roc_auc']:.4f} |"
    )


def render_report(metrics: dict[str, Any]) -> str:
    """Markdown comparison of the trained model, the rule baseline and a flag-everything reference."""
    test = metrics["test"]
    model, rules, rules_cfg, flag_all = test["model"], test["rules_tuned"], test["rules"], test["flag_all"]
    margin = metrics["comparison_min_margin"]
    v = verdict(model, rules, margin, metrics["comparison_min_minority_rows"])
    minority = (
        "non-converting clicks (the fraud proxy)"
        if model["minority_is_target"]
        else ("converting clicks (real conversions)")
    )
    lines = [
        "# Model vs rule-baseline comparison",
        "",
        f"- Model version: `{metrics['version']}`  |  rules config hash: `{test['rules_config_hash']}`",
        f"- Data: `{metrics['data_source']}`, {metrics['data']['n_rows']:,} clicks",
        f"- Chronological split (never shuffled): train < {metrics['data']['val_start']} "
        f"<= val < {metrics['data']['test_start']} <= test",
        f"- Test set: {test['n_rows']:,} clicks, fraud-proxy prevalence {test['prevalence']:.4%}",
        f"- Operating points, both tuned on the validation period for minority-class F1: model threshold "
        f"{metrics['threshold']:.4f}, rules threshold {test['rules_tuned_threshold']:g}. "
        f"The rules' configured threshold ({test['rules_threshold']:g}, `configs/rules.yaml`) "
        "is shown separately.",
        "",
        "## Verdict",
        "",
        _VERDICT_TEXT[v],
        "",
        f"Minority-class test rows: {round(model['minority_prevalence'] * model['n_rows']):,} "
        f"(a verdict needs at least {metrics['comparison_min_minority_rows']:,}). "
        f"Margin used: {margin:g} absolute on both PR-AUC and F1 (minority-class view), "
        "against the validation-tuned rule baseline.",
        "",
        "## Fraud-proxy class (label 1 = click never converted)",
        "",
        "| System | Precision | Recall | F1 |",
        "| --- | --- | --- | --- |",
        _row("Trained model", model, "target_view"),
        _row("Rule baseline (validation-tuned)", rules, "target_view"),
        _row("Rule baseline (configured threshold)", rules_cfg, "target_view"),
        _row("Flag every click", flag_all, "target_view"),
        "",
        f"## Minority-class view: {minority}",
        "",
        "| System | Precision | Recall | F1 | PR-AUC | ROC-AUC |",
        "| --- | --- | --- | --- | --- | --- |",
        _minority_row("Trained model", model),
        _minority_row("Rule baseline (validation-tuned)", rules),
        _minority_row("Rule baseline (configured threshold)", rules_cfg),
        f"| Random scorer (reference) | - | - | - | {model['minority_prevalence']:.4f} | 0.5000 |",
        "",
        "## Confusion matrices at the operating points (rows actual, columns predicted)",
        "",
        "| System | TN | FP | FN | TP |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, m in (
        ("Trained model", model),
        ("Rule baseline (validation-tuned)", rules),
        ("Rule baseline (configured threshold)", rules_cfg),
        ("Flag every click", flag_all),
    ):
        c = m["confusion"]
        lines.append(f"| {name} | {c['tn']:,} | {c['fp']:,} | {c['fn']:,} | {c['tp']:,} |")
    lines += [
        "",
        "## How to read this",
        "",
        "- PR-AUC is threshold-free and compared against the class prevalence (a random scorer "
        "scores about the prevalence). ROC-AUC is shown for completeness but is flattering under "
        "heavy imbalance, so it is not used for the verdict.",
        "- The label is a proxy: `is_attributed == 0` means the click did not convert, which is not "
        "proof of fraud. When most clicks are non-converting, fraud-proxy precision and F1 look "
        "excellent for any system, including 'flag every click'; judge systems against that row.",
        "- Rule weights in `configs/rules.yaml` were not fitted to this data; the model's "
        "hyperparameters were tuned on the validation period only. The test period was used once.",
        "- Per-trial tuning results are in `trials.json` in the model directory.",
        "",
    ]
    return "\n".join(lines)


def write_report(metrics: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics))
    return path


RESULTS_START = "<!-- RESULTS:START (generated by `python -m adshield.models.comparison`; do not edit) -->"
RESULTS_END = "<!-- RESULTS:END -->"


def readme_results_section(metrics: dict[str, Any]) -> str:
    """README results block from the same metrics as the full report (4 decimals, as in the report)."""
    test = metrics["test"]
    systems = (
        ("Rule baseline (validation-tuned threshold)", test["rules_tuned"]),
        ("Rule baseline (configured threshold)", test["rules"]),
        ("Trained model (XGBoost)", test["model"]),
        ("Flag every click (reference)", test["flag_all"]),
    )
    v = verdict(
        test["model"],
        test["rules_tuned"],
        metrics["comparison_min_margin"],
        metrics["comparison_min_minority_rows"],
    )
    data = metrics["data"]
    lines = [
        f"Run `{metrics['version']}` on `{metrics['data_source']}` ({data['n_rows']:,} clicks). "
        f"Held-out test period: {test['n_rows']:,} clicks from {data['test_start']} onward, "
        f"fraud-proxy prevalence {test['prevalence']:.4%}. Split is chronological, never shuffled.",
        "",
        "| System | Precision | Recall | F1 | PR-AUC (fraud proxy) | PR-AUC (minority class) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in systems:
        t = m["target_view"]
        lines.append(
            f"| {name} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} "
            f"| {m['pr_auc']:.4f} | {m['pr_auc_minority']:.4f} |"
        )
    minority = (
        "converting clicks" if test["model"]["minority_is_target"] is False else "non-converting clicks"
    )
    lines += [
        "",
        f"Precision, recall and F1 are for the fraud-proxy class (click never converted). The minority class "
        f"here is {minority}; a random scorer's PR-AUC would equal its prevalence, "
        f"{test['model']['minority_prevalence']:.4f}.",
        "",
        f"**Verdict:** {_VERDICT_TEXT[v]}",
        "",
        "Full tables, confusion matrices and operating points: `reports/model_comparison.md`.",
    ]
    return "\n".join(lines)


def update_readme(readme: str | Path, metrics: dict[str, Any]) -> None:
    """Replace the text between the RESULTS markers in ``readme``. Raises if the markers are missing."""
    path = Path(readme)
    text = path.read_text()
    if RESULTS_START not in text or RESULTS_END not in text:
        raise ValueError(f"{path} does not contain the RESULTS:START / RESULTS:END markers")
    head, rest = text.split(RESULTS_START, 1)
    _, tail = rest.split(RESULTS_END, 1)
    path.write_text(f"{head}{RESULTS_START}\n{readme_results_section(metrics)}\n{RESULTS_END}{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill the README results table from a run's metrics.json.")
    parser.add_argument("metrics", type=Path, help="models/<version>/metrics.json")
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument(
        "--report", type=Path, default=None, help="also (re)write the full markdown report here"
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    update_readme(args.readme, metrics)
    if args.report:
        write_report(metrics, args.report)


if __name__ == "__main__":
    main()
