import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import os


# set size parameters of the plots to be big enought to be slide-readable
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
})


def main(data_paths):
    for data_path in data_paths:
        df = pd.read_csv(data_path)

        # Ensure numeric
        df["params"] = pd.to_numeric(df["Parameters"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")

        # Identify baseline vs distilled
        df["model_name"] = df["Model Name"]
        df["is_distilled"] = df["model_name"].str.contains(r"_alpha", regex=True)

        # Base model name (remove distillation suffix)
        def base_name(name: str) -> str:
            return re.sub(r"_alpha.*$", "", name)

        df["base_model"] = df["model_name"].apply(base_name)

        # Baselines (pick best if multiple)
        baselines = (
            df[~df["is_distilled"]]
            .sort_values(["base_model", "accuracy"], ascending=[True, False])
            .groupby("base_model", as_index=False)
            .first()
            .rename(columns={"accuracy": "baseline_acc", "params": "baseline_params", "model_name": "baseline_run"})
        )

        # Best distilled per base model
        best_distilled = (
            df[df["is_distilled"]]
            .sort_values(["base_model", "accuracy"], ascending=[True, False])
            .groupby("base_model", as_index=False)
            .first()
            .rename(
                columns={
                    "accuracy": "distill_acc",
                    "params": "distill_params",
                    "model_name": "best_distill_run",
                }
            )
        )

        merged = baselines.merge(best_distilled, on="base_model", how="left")
        merged = merged.dropna(subset=["distill_acc", "baseline_acc"])

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(8, 5))

        # 1) All distilled runs (faint, no annotations)
        distilled = df[df["is_distilled"]].copy()
        plt.scatter(
            distilled["params"],
            distilled["accuracy"],
            label="All distillation runs",
            alpha=0.25,
            s=25,
        )

        # 2) Baseline points (highlight)
        plt.scatter(
            merged["baseline_params"],
            merged["baseline_acc"],
            label="Baseline (no distillation)",
            alpha=0.95,
            s=80,
        )

        # 3) Best distilled points (highlight)
        plt.scatter(
            merged["distill_params"],
            merged["distill_acc"],
            label="Best distillation (max accuracy)",
            alpha=0.95,
            s=80,
        )

        # Connect baseline -> best distilled per model
        for _, row in merged.iterrows():
            x0, y0 = row["baseline_params"], row["baseline_acc"]
            x1, y1 = row["distill_params"], row["distill_acc"]

            # Draw connecting line
            plt.plot(
                [x0, x1],
                [y0, y1],
                linestyle="--",
                alpha=0.6,
                color="blue"
            )

            # Accuracy gain
            gain = y1 - y0

            # Midpoint of the line
            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2

            # Annotate gain slightly to the left
            plt.annotate(
                f"+{gain:.3f}" if gain > 0 else f"{gain:.3f}",
                (xm, ym),
                textcoords="offset points",
                xytext=(-4, 0),   # left shift
                ha="right",
                va="center",
                fontsize=9,
                alpha=0.9,
                color="green" if gain > 0 else "red"
            )


        # Annotate ONLY baseline + best distilled
        for _, row in merged.iterrows():
            # Baseline label
            plt.annotate(
                row["base_model"],
                (row["baseline_params"], row["baseline_acc"]),
                textcoords="offset points",
                xytext=(6, -3),
                ha="left",
                alpha=0.95,
                fontsize=10
            )

            # Best distillation label (anchored to best point!)
            # plt.annotate(
            #     row["best_distill_run"],
            #     (row["distill_params"], row["distill_acc"]),
            #     textcoords="offset points",
            #     xytext=(6, -3),
            #     ha="left",
            #     fontsize=9,
            #     alpha=0.95,
            # )

        plt.xlabel("Number of Parameters")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Model Size: Baseline vs Best Distillation")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.95,
        )
        plt.tight_layout()
        plt.margins(x=0.15, y=0.1)

        out_path = os.path.join(os.path.dirname(data_path), "distillation_improvement.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {out_path}")

        # Optional summary
        summary = merged.copy()
        summary["abs_gain"] = summary["distill_acc"] - summary["baseline_acc"]
        print(summary[["base_model", "baseline_acc", "distill_acc", "abs_gain", "best_distill_run"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Distillation sweep results")
    parser.add_argument(
        "sweep_csv_path",
        type=str,
        nargs="+",
        help="Paths to the distillation sweep csv result filepath",
    )
    args = parser.parse_args()
    main(args.sweep_csv_path)


"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import os


def main(data_paths):
    for data_path in data_paths:
        # -----------------------------
        # Load data
        # -----------------------------
        df = pd.read_csv(data_path)

        # Ensure numeric
        df["params"] = pd.to_numeric(df["Parameters"], errors="coerce")
        df["accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")

        # -----------------------------
        # Identify baseline vs distilled
        # Baseline: no "_alpha" in name
        # Distilled: has "_alpha"
        # -----------------------------
        df["model_name"] = df["Model Name"]
        df["is_distilled"] = df["model_name"].str.contains(r"_alpha", regex=True)

        # Extract base model name:
        # - If distilled: remove "_alpha..._T..." suffix
        # - If baseline: model_name is already base
        def base_name(name: str) -> str:
            return re.sub(r"_alpha.*$", "", name)

        df["base_model"] = df["model_name"].apply(base_name)

        # -----------------------------
        # For each base model:
        # - baseline row (should be exactly 1; if multiple, pick best accuracy)
        # - best distilled row (highest accuracy)
        # -----------------------------
        baselines = (
            df[~df["is_distilled"]]
            .sort_values(["base_model", "accuracy"], ascending=[True, False])
            .groupby("base_model", as_index=False)
            .first()
            .rename(columns={"accuracy": "baseline_acc", "params": "baseline_params"})
        )

        best_distilled = (
            df[df["is_distilled"]]
            .sort_values(["base_model", "accuracy"], ascending=[True, False])
            .groupby("base_model", as_index=False)
            .first()
            .rename(columns={"accuracy": "distill_acc", "params": "distill_params", "model_name": "best_distill_run"})
        )

        merged = baselines.merge(best_distilled, on="base_model", how="left")
        merged.dropna(inplace=True)

        # -----------------------------
        # Plot
        # -----------------------------
        plt.figure(figsize=(8, 5))

        # Baseline points
        plt.scatter(
            merged["baseline_params"],
            merged["baseline_acc"],
            label="Baseline (no distillation)",
            alpha=0.9,
        )

        # Best distilled points (only where available)
        has_distill = merged["distill_acc"].notna()
        plt.scatter(
            merged.loc[has_distill, "distill_params"],
            merged.loc[has_distill, "distill_acc"],
            label="Best distillation (max accuracy)",
            alpha=0.9,
        )

        # Connect baseline -> best distilled per model
        for _, row in merged.loc[has_distill].iterrows():
            plt.plot(
                [row["baseline_params"], row["distill_params"]],
                [row["baseline_acc"], row["distill_acc"]],
                linestyle="--",
                alpha=0.5,
                color="green"
            )

        # Annotate model names
        for _, row in merged.iterrows():
            plt.annotate(
                row["base_model"],
                (row["baseline_params"], row["baseline_acc"]),
                textcoords="offset points",
                xytext=(6, -4),
                ha="left",
                fontsize=9,
                alpha=0.9,
            )
            plt.annotate(
                row["best_distill_run"],
                (row["baseline_params"], row["distill_acc"]),
                textcoords="offset points",
                xytext=(6, -4),
                ha="left",
                fontsize=9,
                alpha=0.9,
            )

        plt.xlabel("Number of Parameters")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Model Size: Baseline vs Best Distillation")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()

        plt.tight_layout()
        
        out_path = os.path.join(os.path.dirname(data_path), "distillation_improvement.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Plot saved to {out_path}")

        # -----------------------------
        # Optional: print a small summary table
        # -----------------------------
        summary = merged.copy()
        summary["abs_gain"] = summary["distill_acc"] - summary["baseline_acc"]
        print(summary[["base_model", "baseline_acc", "distill_acc", "abs_gain", "best_distill_run"]].to_string(index=False))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description='Plot Distillation sweep results')
    parser.add_argument(
        'sweep_csv_path', type=str, nargs='+',
        help='Paths to the distillation sweep csv result filepath (e.g., results/sweeps/hbuck2cj/model_comparison_hbuck2cj.csv)')

    args = parser.parse_args()
    main(args.sweep_csv_path)

"""


