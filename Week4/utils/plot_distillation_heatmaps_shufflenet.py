import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_base_model(name: str) -> str:
    # Remove distillation suffix "_alpha..._T..."
    return re.sub(r"_alpha.*$", "", name)


def parse_alpha(name: str) -> float:
    m = re.search(r"_alpha([0-9]*\.?[0-9]+)", name)
    return float(m.group(1)) if m else np.nan


def parse_temperature(name: str) -> int:
    m = re.search(r"_T([0-9]+)", name)
    return int(m.group(1)) if m else np.nan


def plot_heatmap(
    pivot: pd.DataFrame,
    title: str,
    out_path: str,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    cbar_label: str = "Accuracy",  # NEW
):
    """
    pivot: DataFrame indexed by alpha (rows) and with columns=temperature (cols).
    """
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    alphas = pivot.index.to_list()
    temps = pivot.columns.to_list()
    Z = pivot.values

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Alpha (α)")

    ax.set_xticks(np.arange(len(temps)))
    ax.set_xticklabels([str(t) for t in temps])

    ax.set_yticks(np.arange(len(alphas)))
    ax.set_yticklabels([f"{a:g}" for a in alphas])

    if annotate:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)  # NEW

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(csv_paths, annotate=False, show=False):
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        df["model_name"] = df["Model Name"]
        df["accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")

        # NEW: baseline vs distilled split
        df["is_distilled"] = df["model_name"].str.contains(r"_alpha", regex=True)

        # NEW: base model for ALL rows (baseline and distilled)
        df["base_model"] = df["model_name"].apply(parse_base_model)

        # Only distilled runs for the heatmaps
        distilled = df[df["is_distilled"]].copy()

        distilled["alpha"] = distilled["model_name"].apply(parse_alpha)
        distilled["temperature"] = distilled["model_name"].apply(parse_temperature)

        distilled = distilled.dropna(subset=["alpha", "temperature", "accuracy"])

        # If multiple runs exist per (model, alpha, T), average them
        distilled = (
            distilled.groupby(["base_model", "alpha", "temperature"], as_index=False)["accuracy"]
            .mean()
        )

        # -----------------------------
        # NEW: Compute baseline accuracy per model and attach to distilled runs
        # -----------------------------
        baseline = df[~df["is_distilled"]].copy()
        baseline = baseline.dropna(subset=["accuracy"])

        # Baseline model name may already be the base model, but we use base_model for safety
        baseline_acc = (
            baseline.groupby("base_model", as_index=False)["accuracy"]
            .max()  # if multiple baselines exist, keep the best one
            .rename(columns={"accuracy": "baseline_acc"})
        )

        distilled = distilled.merge(baseline_acc, on="base_model", how="left")
        distilled = distilled.dropna(subset=["baseline_acc"])

        # NEW: gain per run
        distilled["accuracy_gain"] = distilled["accuracy"] - distilled["baseline_acc"]

        # Global vmin/vmax for ACCURACY heatmaps
        acc_vmin = distilled["accuracy"].min()
        acc_vmax = distilled["accuracy"].max()

        # NEW: global vmin/vmax for GAIN heatmaps
        gain_vmin = distilled["accuracy_gain"].min()
        gain_vmax = distilled["accuracy_gain"].max()

        out_dir = os.path.dirname(csv_path) or "."

        # -----------------------------
        # 1) Heatmap per model (accuracy) + (gain)
        # -----------------------------
        for base_model, g in distilled.groupby("base_model"):
            # Accuracy heatmap
            pivot_acc = g.pivot(index="alpha", columns="temperature", values="accuracy")
            out_path_acc = os.path.join(out_dir, f"heatmap_{base_model}.png")
            plot_heatmap(
                pivot_acc,
                title=f"{base_model} - Distillation Accuracy",
                out_path=out_path_acc,
                vmin=acc_vmin,
                vmax=acc_vmax,
                annotate=annotate,
                cbar_label="Accuracy",
            )
            print(f"Saved: {out_path_acc}")

            # NEW: Gain heatmap
            pivot_gain = g.pivot(index="alpha", columns="temperature", values="accuracy_gain")
            out_path_gain = os.path.join(out_dir, f"heatmap_{base_model}_GAIN.png")
            plot_heatmap(
                pivot_gain,
                title=f"{base_model} - Accuracy Gain vs Baseline",
                out_path=out_path_gain,
                vmin=gain_vmin,
                vmax=gain_vmax,
                annotate=annotate,
                cbar_label="Δ Accuracy",
            )
            print(f"Saved: {out_path_gain}")

        # -----------------------------
        # 2) Average heatmap across models (accuracy) + (gain)
        # -----------------------------
        avg_acc = distilled.groupby(["alpha", "temperature"], as_index=False)["accuracy"].mean()
        pivot_avg_acc = avg_acc.pivot(index="alpha", columns="temperature", values="accuracy")

        out_path_avg_acc = os.path.join(out_dir, "heatmap_ALL_MODELS_AVG.png")
        plot_heatmap(
            pivot_avg_acc,
            title="All Models - Average Distillation Accuracy",
            out_path=out_path_avg_acc,
            vmin=acc_vmin,
            vmax=acc_vmax,
            annotate=annotate,
            cbar_label="Accuracy",
        )
        print(f"Saved: {out_path_avg_acc}")

        # NEW: average gain heatmap across models
        avg_gain = distilled.groupby(["alpha", "temperature"], as_index=False)["accuracy_gain"].mean()
        pivot_avg_gain = avg_gain.pivot(index="alpha", columns="temperature", values="accuracy_gain")

        out_path_avg_gain = os.path.join(out_dir, "heatmap_ALL_MODELS_AVG_GAIN.png")
        plot_heatmap(
            pivot_avg_gain,
            title="All Models - Average Accuracy Gain vs Baseline",
            out_path=out_path_avg_gain,
            vmin=gain_vmin,
            vmax=gain_vmax,
            annotate=annotate,
            cbar_label="Δ Accuracy",
        )
        print(f"Saved: {out_path_avg_gain}")

        # Optional quick display
        if show:
            pivot_avg_acc = pivot_avg_acc.sort_index(axis=0).sort_index(axis=1)
            plt.figure(figsize=(6.5, 4.8))
            plt.imshow(pivot_avg_acc.values, origin="lower", aspect="auto", vmin=acc_vmin, vmax=acc_vmax)
            plt.title("All Models - Average Distillation Accuracy")
            plt.xlabel("Temperature (T)")
            plt.ylabel("Alpha (α)")
            plt.xticks(np.arange(len(pivot_avg_acc.columns)), [str(t) for t in pivot_avg_acc.columns])
            plt.yticks(np.arange(len(pivot_avg_acc.index)), [f"{a:g}" for a in pivot_avg_acc.index])
            plt.colorbar(label="Accuracy")
            plt.tight_layout()
            plt.show()

            pivot_avg_gain = pivot_avg_gain.sort_index(axis=0).sort_index(axis=1)
            plt.figure(figsize=(6.5, 4.8))
            plt.imshow(pivot_avg_gain.values, origin="lower", aspect="auto", vmin=gain_vmin, vmax=gain_vmax)
            plt.title("All Models - Average Accuracy Gain vs Baseline")
            plt.xlabel("Temperature (T)")
            plt.ylabel("Alpha (α)")
            plt.xticks(np.arange(len(pivot_avg_gain.columns)), [str(t) for t in pivot_avg_gain.columns])
            plt.yticks(np.arange(len(pivot_avg_gain.index)), [f"{a:g}" for a in pivot_avg_gain.index])
            plt.colorbar(label="Δ Accuracy")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distillation heatmaps per model + average.")
    parser.add_argument("csv_paths", type=str, nargs="+", help="Path(s) to CSV file(s).")
    parser.add_argument("--annotate", action="store_true", help="Write accuracy values inside the cells.", default=True)
    parser.add_argument("--show", action="store_true", help="Show the average heatmap interactively.")
    args = parser.parse_args()

    main(args.csv_paths, annotate=args.annotate, show=args.show)
