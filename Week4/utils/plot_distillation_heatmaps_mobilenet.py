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
    xlabel: str = "Number of Parameters"
):
    """
    pivot: DataFrame indexed by alpha (rows) and with columns=temperature (cols).
    """
    # Ensure consistent ordering
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    alphas = pivot.index.to_list()
    ys = pivot.columns.to_list()
    Z = pivot.values  # shape: (len(alphas), len(temps))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Alpha (α)")

    # Ticks/labels
    ax.set_xticks(np.arange(len(ys)))
    ax.set_xticklabels([str(t) for t in ys])

    ax.set_yticks(np.arange(len(alphas)))
    ax.set_yticklabels([f"{a:g}" for a in alphas])

    # Optional value annotations (can get busy; keep off for slides)
    if annotate:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(csv_paths, annotate=False):
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # Normalize columns (as in your previous script)
        df["model_name"] = df["Model Name"]
        df["accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
        df["params"] = pd.to_numeric(df["Parameters"], errors="coerce")

        df["is_distilled"] = df["model_name"].str.contains(r"_alpha", regex=True)


        # Only distilled runs for the heatmaps
        distilled = df[df["model_name"].str.contains(r"_alpha", regex=True)].copy()

        distilled["base_model"] = distilled["model_name"].apply(parse_base_model)
        distilled["alpha"] = distilled["model_name"].apply(parse_alpha)

        # Drop rows without distillation runs
        distilled = distilled.dropna(subset=["alpha", "accuracy"])

        # Optional: if multiple runs exist per (model, alpha, T), take the mean (or max)
        # distilled = distilled.groupby(
        #     ["base_model", "alpha", "params"], as_index=False)["accuracy"].mean()

        # Compute global vmin/vmax so all per-model heatmaps share the same color scale
        global_vmin = distilled["accuracy"].min()
        global_vmax = distilled["accuracy"].max()

        out_dir = os.path.dirname(csv_path) or "."

        # Heatmap
        pivot = distilled.pivot(index="alpha", columns="params", values="accuracy")
        out_path = os.path.join(out_dir, f"heatmap_distillation_mobilenet.png")
        plot_heatmap(
            pivot,
            title=f"Distillation Accuracy",
            out_path=out_path,
            vmin=global_vmin,
            vmax=global_vmax,
            annotate=annotate,
        )
        print(f"Saved: {out_path}")


        # -----------------------------
        # difference between baseline accuracy and distillation experiment acc
        # -----------------------------
        # Baseline accuracy per base model (rows WITHOUT "_alpha")
        baseline = df[~df["is_distilled"]].copy()
        baseline["base_model"] = baseline["model_name"]  # baseline name is already the base
        baseline_acc = (
            baseline.groupby("base_model")["accuracy"]
            .max()  # in case duplicates exist
            .reset_index()
            .rename(columns={"accuracy": "baseline_acc"})
        )

        # Attach baseline accuracy to each distilled run
        distilled_with_base = distilled.merge(baseline_acc, on="base_model", how="left")

        # Compute gain per run
        distilled_with_base["accuracy_gain"] = (
            distilled_with_base["accuracy"] - distilled_with_base["baseline_acc"]
        )

        # (Optional) Heatmap of gain instead of accuracy
        gain_vmin = distilled_with_base["accuracy_gain"].min()
        gain_vmax = distilled_with_base["accuracy_gain"].max()

        pivot_gain = distilled_with_base.pivot(index="alpha", columns="params", values="accuracy_gain")
        out_path_gain = os.path.join(out_dir, "heatmap_distillation_gain_mobilenet.png")
        plot_heatmap(
            pivot_gain,
            title="Distillation Gain vs Baseline (Accuracy Δ)",
            out_path=out_path_gain,
            vmin=gain_vmin,
            vmax=gain_vmax,
            annotate=annotate,
            xlabel="Number of Parameters",
        )
        print(f"Saved: {out_path_gain}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distillation heatmaps per model + average.")
    parser.add_argument("csv_paths", type=str, nargs="+", help="Path(s) to CSV file(s).")
    parser.add_argument("--annotate", action="store_true", help="Write accuracy values inside the cells.", default=True)
    args = parser.parse_args()

    main(args.csv_paths, annotate=args.annotate)
