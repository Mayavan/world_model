from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_rollout_by_step(path: Path) -> dict[int, list[dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = {}
    for row in read_csv_rows(path):
        step = int(row["step"])
        grouped.setdefault(step, []).append(
            {
                "horizon": int(row["horizon"]),
                "mse": float(row["mse"]),
                "psnr": float(row["psnr"]),
            }
        )
    for rows in grouped.values():
        rows.sort(key=lambda x: x["horizon"])
    return grouped


def read_val_metrics(path: Path) -> list[dict[str, float]]:
    rows = []
    for row in read_csv_rows(path):
        rows.append(
            {
                "step": int(row["step"]),
                "val_loss": float(row["val_loss"]),
                "val_mse": float(row["val_mse"]),
                "val_psnr": float(row["val_psnr"]),
                "val_ssim": float(row["val_ssim"]),
            }
        )
    return rows


def summarize_run(run_dir: Path) -> dict[str, object]:
    val_rows = read_val_metrics(run_dir / "val_metrics.csv")
    rollout_by_step = group_rollout_by_step(run_dir / "val_rollout_metrics.csv")
    final_val = val_rows[-1]
    best_val = max(val_rows, key=lambda x: x["val_psnr"])
    final_rollout = rollout_by_step[max(rollout_by_step.keys())]
    best_rollout = min(
        rollout_by_step.values(),
        key=lambda rows: mean(item["mse"] for item in rows),
    )
    return {
        "run_dir": str(run_dir),
        "final_val": final_val,
        "best_val": best_val,
        "final_rollout": final_rollout,
        "best_rollout": best_rollout,
    }


def plot_rollout_curves(
    *,
    baseline_rows: list[dict[str, float]],
    experiment_rows: list[dict[str, float]],
    out_dir: Path,
    title_suffix: str,
) -> None:
    horizons = [row["horizon"] for row in baseline_rows]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(horizons, [row["mse"] for row in baseline_rows], label="baseline", linewidth=2)
    plt.plot(horizons, [row["mse"] for row in experiment_rows], label="rollout-aware", linewidth=2)
    plt.xlabel("Horizon")
    plt.ylabel("MSE")
    plt.title(f"Rollout MSE vs horizon ({title_suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(horizons, [row["psnr"] for row in baseline_rows], label="baseline", linewidth=2)
    plt.plot(horizons, [row["psnr"] for row in experiment_rows], label="rollout-aware", linewidth=2)
    plt.xlabel("Horizon")
    plt.ylabel("PSNR")
    plt.title(f"Rollout PSNR vs horizon ({title_suffix})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f"rollout_curves_{title_suffix.replace(' ', '_').lower()}.png", dpi=160)
    plt.close()


def plot_val_curves(
    *,
    baseline_val: list[dict[str, float]],
    experiment_val: list[dict[str, float]],
    out_dir: Path,
) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot([r["step"] for r in baseline_val], [r["val_mse"] for r in baseline_val], label="baseline")
    plt.plot([r["step"] for r in experiment_val], [r["val_mse"] for r in experiment_val], label="rollout-aware")
    plt.xlabel("Step")
    plt.ylabel("Validation MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([r["step"] for r in baseline_val], [r["val_psnr"] for r in baseline_val], label="baseline")
    plt.plot([r["step"] for r in experiment_val], [r["val_psnr"] for r in experiment_val], label="rollout-aware")
    plt.xlabel("Step")
    plt.ylabel("Validation PSNR")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_dir / "validation_curves.png", dpi=160)
    plt.close()


def write_report(
    *,
    baseline_summary: dict[str, object],
    experiment_summary: dict[str, object],
    out_dir: Path,
) -> None:
    baseline_final = baseline_summary["final_rollout"]
    experiment_final = experiment_summary["final_rollout"]
    interesting_horizons = [1, 5, 10, 20, 32]

    def lookup(rows: list[dict[str, float]], horizon: int, key: str) -> float:
        for row in rows:
            if row["horizon"] == horizon:
                return float(row[key])
        raise KeyError(horizon)

    lines = []
    lines.append("# Rollout-aware training experiment report\n")
    lines.append(f"- Baseline: `{baseline_summary['run_dir']}`")
    lines.append(f"- Experiment: `{experiment_summary['run_dir']}`\n")
    lines.append("## Final validation metrics\n")
    lines.append(
        f"- Baseline final val_mse={baseline_summary['final_val']['val_mse']:.6f}, "
        f"val_psnr={baseline_summary['final_val']['val_psnr']:.3f}, "
        f"val_ssim={baseline_summary['final_val']['val_ssim']:.4f}"
    )
    lines.append(
        f"- Experiment final val_mse={experiment_summary['final_val']['val_mse']:.6f}, "
        f"val_psnr={experiment_summary['final_val']['val_psnr']:.3f}, "
        f"val_ssim={experiment_summary['final_val']['val_ssim']:.4f}\n"
    )
    lines.append("## Final rollout metrics by selected horizons\n")
    lines.append("| Horizon | Baseline MSE | Experiment MSE | Baseline PSNR | Experiment PSNR |")
    lines.append("|---:|---:|---:|---:|---:|")
    for h in interesting_horizons:
        lines.append(
            "| "
            f"{h} | {lookup(baseline_final, h, 'mse'):.6f} | {lookup(experiment_final, h, 'mse'):.6f} | "
            f"{lookup(baseline_final, h, 'psnr'):.3f} | {lookup(experiment_final, h, 'psnr'):.3f} |"
        )
    lines.append("")
    lines.append("## Notes\n")
    lines.append(
        "- Compare `validation_curves.png` for overall supervised validation behavior."
    )
    lines.append(
        "- Compare `rollout_curves_final_step.png` for long-horizon open-loop behavior at the final checkpoint."
    )
    lines.append(
        "- Compare `rollout_curves_best_mean_rollout.png` for the best rollout checkpoint found within each run."
    )

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline)
    experiment_dir = Path(args.experiment)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = summarize_run(baseline_dir)
    experiment_summary = summarize_run(experiment_dir)

    plot_val_curves(
        baseline_val=read_val_metrics(baseline_dir / "val_metrics.csv"),
        experiment_val=read_val_metrics(experiment_dir / "val_metrics.csv"),
        out_dir=out_dir,
    )
    plot_rollout_curves(
        baseline_rows=baseline_summary["final_rollout"],
        experiment_rows=experiment_summary["final_rollout"],
        out_dir=out_dir,
        title_suffix="final step",
    )
    plot_rollout_curves(
        baseline_rows=baseline_summary["best_rollout"],
        experiment_rows=experiment_summary["best_rollout"],
        out_dir=out_dir,
        title_suffix="best mean rollout",
    )
    write_report(
        baseline_summary=baseline_summary,
        experiment_summary=experiment_summary,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
