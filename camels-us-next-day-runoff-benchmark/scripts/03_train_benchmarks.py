from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def nse(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    if len(obs) == 0:
        return np.nan
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((sim - obs) ** 2) / denom


def kge(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    if len(obs) < 2:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1) if np.std(obs, ddof=1) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def kge_components(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    r = np.corrcoef(obs, sim)[0, 1] if len(obs) >= 2 else np.nan
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1) if len(obs) >= 2 and np.std(obs, ddof=1) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if len(obs) >= 1 and np.mean(obs) != 0 else np.nan
    return r, alpha, beta


def metric_block(obs, pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(obs, pred))),
        "MAE": float(mean_absolute_error(obs, pred)),
        "NSE": float(nse(obs, pred)),
        "KGE": float(kge(obs, pred)),
    }


def evaluate_by_basin(df, pred_col):
    rows = []
    for gauge_id, g in df.groupby("gauge_id"):
        m = metric_block(g["target_q_next1"].values, g[pred_col].values)
        m["gauge_id"] = gauge_id
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="camels_workspace")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = root / "analysis_outputs"
    panel_file = outdir / "camels_50_model_panel.parquet"

    if not panel_file.exists():
        raise FileNotFoundError(f"Missing {panel_file}. Run 02_build_panel.py first.")

    model_df = pd.read_parquet(panel_file)
    model_df["date"] = pd.to_datetime(model_df["date"])

    train_df = model_df[(model_df["date"] >= "1990-01-01") & (model_df["date"] <= "1999-12-31")].copy()
    val_df   = model_df[(model_df["date"] >= "2000-01-01") & (model_df["date"] <= "2004-12-31")].copy()
    test_df  = model_df[(model_df["date"] >= "2005-01-01") & (model_df["date"] <= "2014-12-31")].copy()

    exclude = {"gauge_id", "date", "q_mm_day", "target_q_next1"}
    feature_cols = [c for c in model_df.columns if c not in exclude]

    X_train = train_df[feature_cols]
    y_train = train_df["target_q_next1"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target_q_next1"].values

    models = {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=8,
                max_iter=200,
                random_state=42,
            )),
        ]),
    }

    pred_test = test_df[["gauge_id", "date", "q_mm_day", "target_q_next1"]].copy()
    pred_test["Persistence"] = test_df["q_mm_day"].values

    summary_rows = []
    by_basin_outputs = []

    m = metric_block(pred_test["target_q_next1"], pred_test["Persistence"])
    r, alpha, beta = kge_components(pred_test["target_q_next1"], pred_test["Persistence"])
    m.update({"model": "Persistence", "r": r, "alpha": alpha, "beta": beta})
    summary_rows.append(m)

    mb = evaluate_by_basin(pred_test.assign(pred=pred_test["Persistence"]), "pred")
    mb["model"] = "Persistence"
    by_basin_outputs.append(mb)

    for name, pipe in models.items():
        print(f"Fitting {name} ...")
        pipe.fit(X_train, y_train)
        pred_test[name] = pipe.predict(X_test)

        m = metric_block(pred_test["target_q_next1"], pred_test[name])
        r, alpha, beta = kge_components(pred_test["target_q_next1"], pred_test[name])
        m.update({"model": name, "r": r, "alpha": alpha, "beta": beta})
        summary_rows.append(m)

        mb = evaluate_by_basin(pred_test.assign(pred=pred_test[name]), "pred")
        mb["model"] = name
        by_basin_outputs.append(mb)

    summary_df = pd.DataFrame(summary_rows)[["model", "RMSE", "MAE", "NSE", "KGE", "r", "alpha", "beta"]]
    by_basin_df = pd.concat(by_basin_outputs, ignore_index=True)[["model", "gauge_id", "RMSE", "MAE", "NSE", "KGE"]]

    summary_df = summary_df.sort_values("NSE", ascending=False)
    by_basin_df = by_basin_df.sort_values(["model", "NSE"], ascending=[True, False])

    summary_df.to_csv(outdir / "metrics_summary.csv", index=False)
    by_basin_df.to_csv(outdir / "metrics_by_basin.csv", index=False)
    pred_test.to_csv(outdir / "test_predictions.csv", index=False)

    print(summary_df)
    print("[done] saved metrics and predictions to:", outdir)


if __name__ == "__main__":
    main()
