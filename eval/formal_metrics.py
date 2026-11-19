"""Compute fixed-threshold metrics, paired image bootstrap intervals and figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve

from src.datasets import EVAL_SUITES
from eval.formal_predict import SPLITS, manifest_lf_sha256, sha256

SEEDS = (5421, 2026)
CONDITIONS = ("A", "B", "C")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
SUITE_LABELS = ("Clean", "JPEG 70", "JPEG 50", "Half size", "Blur", "Bias control")
SPLIT_LABELS = dict(zip(SPLITS, ("Seen", "Unseen ADM", "Unseen GLIDE")))


def scores(labels, probabilities):
    predicted = probabilities >= 0.5
    tp = (predicted & labels).sum(axis=-1)
    fp = (predicted & ~labels).sum(axis=-1)
    pos = labels.sum(axis=-1)
    neg = labels.shape[-1] - pos
    tn, fn = neg - fp, pos - tp
    ranks = rankdata(probabilities, axis=-1)
    auc = ((ranks * labels).sum(axis=-1) - pos * (pos + 1) / 2) / (pos * neg)
    return np.stack(((tp + tn) / (pos + neg), tp / np.maximum(tp + fp, 1), tp / pos,
                     2 * tp / (2 * tp + fp + fn), auc), axis=-1)


def intervals(samples):
    return np.quantile(samples, [0.025, 0.975], axis=0).T


def aligned_predictions(predictions, manifest, split):
    expected = manifest.loc[manifest.split == split].set_index("image_id").sort_index()
    cases = {}
    for (condition, seed, suite), group in predictions.loc[predictions.split == split].groupby(
        ["condition", "seed", "transformation"]
    ):
        ordered = group.set_index("image_id", verify_integrity=True).sort_index()
        if not ordered[["label", "generator", "rel_path"]].equals(expected[["label", "generator", "rel_path"]]):
            raise ValueError(f"Prediction rows disagree with the manifest: {split}/{condition}/{seed}/{suite}")
        cases[(condition, seed, suite)] = ordered.p_ai.to_numpy()
    keys = [(condition, seed, suite) for condition in CONDITIONS for seed in SEEDS for suite in EVAL_SUITES]
    if set(cases) != set(keys):
        raise ValueError(f"Missing or unexpected prediction groups in {split}")
    shape = (len(CONDITIONS), len(SEEDS), len(EVAL_SUITES), len(expected))
    return expected.label.to_numpy(dtype=bool), np.stack([cases[key] for key in keys]).reshape(shape)


def evaluate(predictions, manifest, draws, seed):
    rng = np.random.default_rng(seed)
    metrics, contrasts = [], []
    for split in SPLITS:
        y, p = aligned_predictions(predictions, manifest, split)
        estimates = scores(y, p)
        strata = [np.flatnonzero(y == label) for label in (False, True)]
        samples = []
        for start in range(0, draws, 32):
            count = min(32, draws - start)
            # Each draw uses identical image IDs across conditions, seeds and transformations.
            index = np.concatenate([rng.choice(ids, (count, len(ids)), replace=True) for ids in strata], axis=1)
            samples.append(scores(y[index], p[..., index]))
        sampled = np.concatenate(samples, axis=-2)
        for c, condition in enumerate(CONDITIONS):
            for t, suite in enumerate(EVAL_SUITES):
                for s, run in enumerate((*SEEDS, "mean")):
                    point = estimates[c, s, t] if s < 2 else estimates[c, :, t].mean(axis=0)
                    boot = sampled[c, s, t] if s < 2 else sampled[c, :, t].mean(axis=0)
                    row = dict(split=split, transformation=suite, condition=condition, seed=run, n=len(y))
                    if s < 2:
                        pred = p[c, s, t] >= 0.5
                        row.update(tn=int((~pred & ~y).sum()), fp=int((pred & ~y).sum()),
                                   fn=int((~pred & y).sum()), tp=int((pred & y).sum()))
                    for metric, value, (low, high) in zip(METRICS, point, intervals(boot)):
                        row.update({metric: float(value), f"{metric}_low": float(low), f"{metric}_high": float(high)})
                    metrics.append(row)
                comparisons = []
                if c:
                    comparisons.append((f"{condition}-{CONDITIONS[c-1]}", estimates[c-1, :, t], sampled[c-1, :, t]))
                if t:
                    comparisons.append((f"{condition}: {suite}-clean", estimates[c, :, 0], sampled[c, :, 0]))
                for label, reference, reference_samples in comparisons:
                    point = (estimates[c, :, t] - reference).mean(axis=0)
                    boot = (sampled[c, :, t] - reference_samples).mean(axis=0)
                    for metric, value, (low, high) in zip(METRICS, point, intervals(boot)):
                        contrasts.append(dict(split=split, transformation=suite, comparison=label,
                                              metric=metric, delta=float(value), low=float(low), high=float(high)))
        print(f"Summarized {split}", flush=True)
    return metrics, contrasts


def make_figures(predictions, metrics, contrasts, folder):
    means = metrics[metrics.seed == "mean"]
    colors = dict(zip(CONDITIONS, ("#315b9c", "#a04f16", "#187461")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    for split, ax in zip(SPLITS, axes):
        for condition, color in colors.items():
            rows = means[(means.split == split) & (means.condition == condition)].set_index("transformation").loc[list(EVAL_SUITES)]
            ax.plot(range(6), rows.roc_auc, "-o", color=color, label=condition)
            ax.fill_between(range(6), rows.roc_auc_low, rows.roc_auc_high, color=color, alpha=0.12)
        ax.set_xticks(range(6), SUITE_LABELS, rotation=35, ha="right")
        ax.set_title(SPLIT_LABELS[split])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("ROC-AUC (two-seed mean)")
    axes[-1].legend()
    fig.savefig(folder / "robustness_auc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    grid = np.linspace(0, 1, 501)
    for split, ax in zip(SPLITS, axes):
        clean = predictions[(predictions.split == split) & (predictions.transformation == "clean")]
        for condition, color in colors.items():
            curves = []
            for _, run in clean[clean.condition == condition].groupby("seed"):
                fpr, tpr, _ = roc_curve(run.label, run.p_ai)
                curves.append(np.interp(grid, fpr, tpr))
            ax.plot(grid, np.mean(curves, axis=0), color=color, label=condition)
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":")
        ax.set(xlabel="False positive rate", title=SPLIT_LABELS[split], xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("True positive rate (two-seed mean)")
    axes[-1].legend()
    fig.savefig(folder / "clean_roc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, layout="constrained")
    for split, ax in zip(SPLITS, axes):
        for offset, label, color in [(-0.13, "B-A", colors["B"]), (0.13, "C-B", colors["C"])]:
            rows = contrasts[(contrasts.split == split) & (contrasts.comparison == label)
                             & (contrasts.metric == "roc_auc")].set_index("transformation").loc[list(EVAL_SUITES)]
            positions = np.arange(6) + offset
            ax.hlines(positions, rows.low * 100, rows.high * 100, color=color)
            ax.scatter(rows.delta * 100, positions, color=color, label=label)
        ax.axvline(0, color="grey", linestyle=":")
        ax.set(yticks=range(6), yticklabels=SUITE_LABELS, xlabel="ROC-AUC difference (percentage points)",
               title=SPLIT_LABELS[split])
    axes[0].invert_yaxis()
    axes[-1].legend()
    fig.savefig(folder / "paired_auc_differences.png", dpi=180)
    plt.close(fig)


def export_errors(predictions, manifest, folder):
    clean = predictions[predictions.transformation == "clean"].merge(
        manifest[["image_id", "class_id", "class_name"]], on="image_id", validate="many_to_one"
    )
    clean["error"] = (clean.p_ai >= 0.5).astype(int) != clean.label
    keys = ["split", "condition", "seed", "generator", "label"]
    by_source = clean.groupby(keys).error.agg(n="size", errors="sum", error_rate="mean").reset_index()
    by_source.to_csv(folder / "clean_errors_by_source.csv", index=False, lineterminator="\n")
    errors = clean[clean.error].copy()
    errors["error_type"] = np.where(errors.label == 0, "false_positive", "false_negative")
    errors["wrong_class_probability"] = np.where(errors.label == 0, errors.p_ai, 1 - errors.p_ai)
    examples = errors.sort_values(["wrong_class_probability", "image_id"], ascending=[False, True]).groupby(
        ["split", "condition", "seed", "error_type"], sort=False
    ).head(5)
    examples.drop(columns="error").to_csv(folder / "clean_error_examples.csv", index=False, lineterminator="\n")


def main(args):
    provenance = json.loads((args.results / "provenance.json").read_text(encoding="utf-8"))
    if (sha256(args.manifest) != provenance["manifest_sha256"]
            and manifest_lf_sha256(args.manifest) != provenance.get("manifest_lf_sha256")):
        raise ValueError("Manifest differs from the one used for inference")
    predictions = pd.read_csv(args.results / "predictions.csv")
    manifest = pd.read_csv(args.manifest)
    if (len(predictions) != 162000 or set(predictions.split) != set(SPLITS)
            or not predictions.p_ai.between(0, 1).all()):
        raise ValueError("Expected exactly 162,000 valid held-out probabilities")
    metrics, contrasts = evaluate(predictions, manifest, args.draws, args.seed)
    metric_table, contrast_table = pd.DataFrame(metrics), pd.DataFrame(contrasts)
    metric_table.to_csv(args.results / "metrics.csv", index=False, lineterminator="\n")
    contrast_table.to_csv(args.results / "paired_intervals.csv", index=False, lineterminator="\n")
    payload = dict(positive_label=1, threshold=0.5, bootstrap_draws=args.draws,
                   bootstrap_seed=args.seed, confidence_level=0.95, metrics=metrics, paired=contrasts)
    (args.results / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_figures(predictions, metric_table, contrast_table, args.results)
    export_errors(predictions, manifest, args.results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("Images/manifest.csv"))
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=5421)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("--draws must be at least 2")
    main(args)
