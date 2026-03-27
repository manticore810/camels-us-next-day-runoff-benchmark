from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def load_camels_us_attributes(data_dir: Path, basins=None) -> pd.DataFrame:
    txt_files = sorted((data_dir / "camels_attributes_v2.0").glob("camels_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No camels_*.txt files found in {data_dir / 'camels_attributes_v2.0'}")

    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file, sep=";", header=0, dtype={"gauge_id": str})
        df_temp = df_temp.set_index("gauge_id")
        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)
    df.index = df.index.astype(str).str.zfill(8)

    if "huc_02" in df.columns:
        df["huc"] = df["huc_02"].astype(str).str.zfill(2)
        df = df.drop(columns=["huc_02"])

    if basins is not None:
        basins = [str(b).zfill(8) for b in basins]
        df = df.loc[basins]

    return df


def load_camels_us_forcings(data_dir: Path, basin: str, forcings: str = "daymet"):
    forcing_path = data_dir / "basin_mean_forcing" / forcings
    file_path = list(forcing_path.glob(f"**/{basin}_*_forcing_leap.txt"))
    if not file_path:
        raise FileNotFoundError(f"No forcing file for basin {basin} in {forcing_path}")

    file_path = file_path[0]
    with open(file_path, "r", encoding="utf-8") as fp:
        fp.readline()
        fp.readline()
        area = int(fp.readline().strip())
        df = pd.read_csv(fp, sep=r"\s+")

    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + "/" + df["Mnth"].astype(str) + "/" + df["Day"].astype(str),
        format="%Y/%m/%d",
        errors="coerce",
    )
    df = df.set_index("date")
    return df, area


def load_camels_us_discharge(data_dir: Path, basin: str, area_m2: int):
    discharge_path = data_dir / "usgs_streamflow"
    file_path = list(discharge_path.glob(f"**/{basin}_streamflow_qc.txt"))
    if not file_path:
        raise FileNotFoundError(f"No streamflow file for basin {basin} in {discharge_path}")

    col_names = ["basin", "Year", "Mnth", "Day", "QObs", "flag"]
    df = pd.read_csv(file_path[0], sep=r"\s+", header=None, names=col_names)

    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + "/" + df["Mnth"].astype(str) + "/" + df["Day"].astype(str),
        format="%Y/%m/%d",
        errors="coerce",
    )
    df = df.set_index("date")

    df["QObs(mm/d)"] = 28316846.592 * df["QObs"] * 86400 / (area_m2 * 10**6)
    df.loc[df["QObs(mm/d)"] < 0, "QObs(mm/d)"] = np.nan

    return df["QObs(mm/d)"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="camels_workspace")
    parser.add_argument("--forcing", type=str, default="daymet")
    args = parser.parse_args()

    root = Path(args.root)
    data_dir = root / "subset"
    outdir = root / "analysis_outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    basins = pd.read_csv(data_dir / "basin_list.csv", dtype={"gauge_id": str})
    basin_ids = basins["gauge_id"].astype(str).str.zfill(8).tolist()

    attrs = load_camels_us_attributes(data_dir, basins=basin_ids).reset_index().rename(columns={"index": "gauge_id"})
    attrs["gauge_id"] = attrs["gauge_id"].astype(str).str.zfill(8)

    candidate_static = [
        "area_gages2", "elev_mean", "slope_mean", "p_mean", "pet_mean", "aridity",
        "frac_snow", "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
        "forest_frac", "gvf_max", "gvf_diff", "lai_max", "lai_diff",
        "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity",
        "soil_conductivity", "max_water_content", "sand_frac", "silt_frac", "clay_frac",
        "carbonate_rocks_frac", "geol_permeability", "huc"
    ]
    static_cols = [c for c in candidate_static if c in attrs.columns]

    all_frames = []
    diagnostics = []

    for basin in basin_ids:
        try:
            forcing_df, area_m2 = load_camels_us_forcings(data_dir, basin, forcings=args.forcing)
            qobs = load_camels_us_discharge(data_dir, basin, area_m2)

            df = forcing_df.copy()
            df["QObs(mm/d)"] = qobs
            df = df.reset_index()
            df["gauge_id"] = basin
            all_frames.append(df)

            diagnostics.append({
                "gauge_id": basin,
                "forcing_rows": len(forcing_df),
                "qobs_rows": len(qobs),
                "final_rows": len(df),
                "status": "ok",
            })
        except Exception as e:
            diagnostics.append({
                "gauge_id": basin,
                "forcing_rows": np.nan,
                "qobs_rows": np.nan,
                "final_rows": np.nan,
                "status": f"error: {e}",
            })

    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(outdir / "merge_diagnostics.csv", index=False)

    if not all_frames:
        raise RuntimeError("No basin data loaded successfully. Check merge_diagnostics.csv.")

    panel = pd.concat(all_frames, ignore_index=True)
    panel["gauge_id"] = panel["gauge_id"].astype(str).str.zfill(8)
    panel = panel[(panel["date"] >= "1990-01-01") & (panel["date"] <= "2014-12-31")].copy()

    rename_map = {
        "QObs(mm/d)": "q_mm_day",
        "prcp(mm/day)": "prcp",
        "srad(W/m2)": "srad",
        "tmax(C)": "tmax",
        "tmin(C)": "tmin",
        "vp(Pa)": "vp",
        "dayl(s)": "dayl",
        "swe(mm)": "swe",
    }
    panel = panel.rename(columns=rename_map)
    panel = panel.merge(attrs[["gauge_id"] + static_cols], on="gauge_id", how="left")

    available_core = ["gauge_id", "date", "q_mm_day"] + [c for c in ["prcp", "tmax", "tmin", "vp", "srad", "dayl", "swe"] if c in panel.columns]
    panel = panel[available_core + static_cols].copy()

    feature_frames = []
    for gauge_id, g in panel.groupby("gauge_id", sort=False):
        g = g.sort_values("date").copy()

        for lag in [1, 2, 3, 7]:
            g[f"q_lag{lag}"] = g["q_mm_day"].shift(lag)

        if "prcp" in g.columns:
            for win in [3, 7, 15, 30]:
                g[f"prcp_sum_{win}"] = g["prcp"].rolling(win).sum()

        for win in [3, 7, 15, 30]:
            g[f"q_mean_{win}"] = g["q_mm_day"].rolling(win).mean()

        if "tmin" in g.columns:
            for win in [3, 7, 30]:
                g[f"tmin_mean_{win}"] = g["tmin"].rolling(win).mean()

        if "tmax" in g.columns:
            for win in [3, 7, 30]:
                g[f"tmax_mean_{win}"] = g["tmax"].rolling(win).mean()

        if "srad" in g.columns:
            for win in [7, 30]:
                g[f"srad_mean_{win}"] = g["srad"].rolling(win).mean()

        if "vp" in g.columns:
            for win in [7, 30]:
                g[f"vp_mean_{win}"] = g["vp"].rolling(win).mean()

        doy = g["date"].dt.dayofyear
        g["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        g["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        g["target_q_next1"] = g["q_mm_day"].shift(-1)
        feature_frames.append(g)

    panel = pd.concat(feature_frames, ignore_index=True)

    dynamic_features = [c for c in [
        "prcp", "tmin", "tmax", "srad", "vp", "dayl", "swe",
        "q_lag1", "q_lag2", "q_lag3", "q_lag7",
        "prcp_sum_3", "prcp_sum_7", "prcp_sum_15", "prcp_sum_30",
        "q_mean_3", "q_mean_7", "q_mean_15", "q_mean_30",
        "tmin_mean_3", "tmin_mean_7", "tmin_mean_30",
        "tmax_mean_3", "tmax_mean_7", "tmax_mean_30",
        "srad_mean_7", "srad_mean_30",
        "vp_mean_7", "vp_mean_30",
        "doy_sin", "doy_cos"
    ] if c in panel.columns]

    base_cols = ["gauge_id", "date", "q_mm_day", "target_q_next1"]
    existing_static_cols = [c for c in static_cols if c in panel.columns]

    model_cols = base_cols + dynamic_features + existing_static_cols
    model_df = panel[model_cols].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna(subset=dynamic_features + ["target_q_next1"]).reset_index(drop=True)

    model_df.to_parquet(outdir / "camels_50_model_panel.parquet", index=False)
    model_df.to_csv(outdir / "camels_50_model_panel.csv", index=False)

    print("[done] saved panel to:", outdir / "camels_50_model_panel.parquet")
    print("Shape:", model_df.shape)


if __name__ == "__main__":
    main()
