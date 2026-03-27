from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="camels_workspace")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = root / "analysis_outputs"

    summary_df = pd.read_csv(outdir / "metrics_summary.csv")
    by_basin_df = pd.read_csv(outdir / "metrics_by_basin.csv")
    pred_test = pd.read_csv(outdir / "test_predictions.csv", parse_dates=["date"])

    # Figure 1: basin-wise NSE distribution
    plt.figure(figsize=(10, 6))
    for model in by_basin_df["model"].unique():
        vals = by_basin_df.loc[by_basin_df["model"] == model, "NSE"].dropna().values
        vals = pd.Series(vals).sort_values().reset_index(drop=True)
        plt.plot(vals.values, label=model)

    plt.xlabel("Basins sorted by NSE")
    plt.ylabel("NSE")
    plt.title("Basin-wise NSE distribution across models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig_nse_distribution.png", dpi=300)
    plt.close()

    # Figure 2: overall pooled NSE
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["model"], summary_df["NSE"])
    plt.ylabel("Pooled test NSE")
    plt.title("Overall pooled test NSE by model")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outdir / "fig_overall_nse.png", dpi=300)
    plt.close()

    # Figure 3: hydrographs for 3 best basins using best model
    best_model = summary_df.iloc[0]["model"]
    best3 = (
        by_basin_df[by_basin_df["model"] == best_model]
        .sort_values("NSE", ascending=False)
        .head(3)["gauge_id"]
        .tolist()
    )

    for gauge_id in best3:
        g = pred_test[pred_test["gauge_id"] == gauge_id].copy()
        plt.figure(figsize=(11, 4))
        plt.plot(g["date"], g["target_q_next1"], label="Observed")
        plt.plot(g["date"], g[best_model], label=best_model)
        plt.title(f"Hydrograph comparison — Gauge {gauge_id}")
        plt.ylabel("Runoff (mm/day)")
        plt.xlabel("Date")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"fig_hydrograph_{gauge_id}.png", dpi=300)
        plt.close()

    # Basin-wise summary table
    summary_basin = (
        by_basin_df.groupby("model")
        .agg(
            median_NSE=("NSE", "median"),
            mean_NSE=("NSE", "mean"),
            std_NSE=("NSE", "std"),
            min_NSE=("NSE", "min"),
            max_NSE=("NSE", "max"),
            median_KGE=("KGE", "median"),
            negative_NSE_basins=("NSE", lambda x: int((x < 0).sum())),
            basins=("gauge_id", "nunique"),
        )
        .reset_index()
        .sort_values("median_NSE", ascending=False)
    )
    summary_basin.to_csv(outdir / "metrics_by_basin_summary.csv", index=False)

    # Winning model counts
    winners = (
        by_basin_df.sort_values(["gauge_id", "NSE"], ascending=[True, False])
        .groupby("gauge_id")
        .head(1)
        .reset_index(drop=True)
    )
    winner_counts = winners["model"].value_counts().reset_index()
    winner_counts.columns = ["model", "winning_basins"]
    winner_counts.to_csv(outdir / "winning_model_counts.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(winner_counts["model"], winner_counts["winning_basins"])
    plt.ylabel("Number of basins won")
    plt.title("Best-performing model by basin (NSE)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(outdir / "fig_winning_model_counts.png", dpi=300)
    plt.close()

    print("[done] saved figures and basin summary to:", outdir)


if __name__ == "__main__":
    main()
