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
    annotate: bool = False,
):
    """
    pivot: DataFrame indexed by alpha (rows) and with columns=temperature (cols).
    """
    # Ensure consistent ordering
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    alphas = pivot.index.to_list()
    temps = pivot.columns.to_list()
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
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Alpha (α)")

    # Ticks/labels
    ax.set_xticks(np.arange(len(temps)))
    ax.set_xticklabels([str(t) for t in temps])

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


def main(csv_paths, annotate=False, show=False):
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)

        # Normalize columns (as in your previous script)
        df["model_name"] = df["Model Name"]
        df["accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")

        # Only distilled runs for the heatmaps
        distilled = df[df["model_name"].str.contains(r"_alpha", regex=True)].copy()

        distilled["base_model"] = distilled["model_name"].apply(parse_base_model)
        distilled["alpha"] = distilled["model_name"].apply(parse_alpha)
        distilled["temperature"] = distilled["model_name"].apply(parse_temperature)

        # Drop rows that failed parsing
        distilled = distilled.dropna(subset=["alpha", "temperature", "accuracy"])

        # Optional: if multiple runs exist per (model, alpha, T), take the mean (or max)
        distilled = (
            distilled.groupby(["base_model", "alpha", "temperature"], as_index=False)["accuracy"]
            .mean()
        )

        # Compute global vmin/vmax so all per-model heatmaps share the same color scale
        global_vmin = distilled["accuracy"].min()
        global_vmax = distilled["accuracy"].max()

        out_dir = os.path.dirname(csv_path) or "."

        # -----------------------------
        # 1) Heatmap per model
        # -----------------------------
        for base_model, g in distilled.groupby("base_model"):
            pivot = g.pivot(index="alpha", columns="temperature", values="accuracy")
            out_path = os.path.join(out_dir, f"heatmap_{base_model}.png")
            plot_heatmap(
                pivot,
                title=f"{base_model} - Distillation Accuracy",
                out_path=out_path,
                vmin=global_vmin,
                vmax=global_vmax,
                annotate=annotate,
            )
            print(f"Saved: {out_path}")

        # -----------------------------
        # 2) Average heatmap across models
        # -----------------------------
        avg = distilled.groupby(["alpha", "temperature"], as_index=False)["accuracy"].mean()
        pivot_avg = avg.pivot(index="alpha", columns="temperature", values="accuracy")

        out_path_avg = os.path.join(out_dir, "heatmap_ALL_MODELS_AVG.png")
        plot_heatmap(
            pivot_avg,
            title="All Models - Average Distillation Accuracy",
            out_path=out_path_avg,
            vmin=global_vmin,   # or use pivot_avg min/max if you prefer
            vmax=global_vmax,
            annotate=annotate,
        )
        print(f"Saved: {out_path_avg}")

        # Optional quick display of the average heatmap (if running interactively)
        if show:
            # Re-open and show the avg plot quickly
            pivot_avg = pivot_avg.sort_index(axis=0).sort_index(axis=1)
            plt.figure(figsize=(6.5, 4.8))
            plt.imshow(pivot_avg.values, origin="lower", aspect="auto", vmin=global_vmin, vmax=global_vmax)
            plt.title("All Models - Average Distillation Accuracy")
            plt.xlabel("Temperature (T)")
            plt.ylabel("Alpha (α)")
            plt.xticks(np.arange(len(pivot_avg.columns)), [str(t) for t in pivot_avg.columns])
            plt.yticks(np.arange(len(pivot_avg.index)), [f"{a:g}" for a in pivot_avg.index])
            plt.colorbar(label="Accuracy")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distillation heatmaps per model + average.")
    parser.add_argument("csv_paths", type=str, nargs="+", help="Path(s) to CSV file(s).")
    parser.add_argument("--annotate", action="store_true", help="Write accuracy values inside the cells.")
    parser.add_argument("--show", action="store_true", help="Show the average heatmap interactively.")
    args = parser.parse_args()

    main(args.csv_paths, annotate=args.annotate, show=args.show)
