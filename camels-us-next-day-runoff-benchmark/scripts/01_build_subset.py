from pathlib import Path
import argparse
import shutil
import zipfile
import requests
import pandas as pd

GAUGES = [
    "01013500", "01022500", "01030500", "01031500", "01047000",
    "01052500", "01054200", "01055000", "01057000", "01073000",
    "01078000", "01118300", "01121000", "01123000", "01139000",
    "01187300", "01466500", "01516500", "01543000", "01548500",
    "01557500", "01606500", "02028500", "02112360", "02235200",
    "02296500", "02312200", "02408540", "02422500", "02465493",
    "03015500", "03049000", "03070500", "03182500", "03280700",
    "03488000", "03498500", "03500000", "04063700", "04105700",
    "06404000", "07056000", "08070200", "09312600", "10172700",
    "11143000", "12115500", "12488500", "14154500", "14400000"
]

ZENODO_RECORD = "https://zenodo.org/records/15529996/files"
TIMESERIES_ZIP_URL = f"{ZENODO_RECORD}/basin_timeseries_v1p2_metForcing_obsFlow.zip?download=1"
ATTRIBUTE_FILES = [
    "camels_clim.txt",
    "camels_geol.txt",
    "camels_hydro.txt",
    "camels_name.txt",
    "camels_soil.txt",
    "camels_topo.txt",
    "camels_vege.txt",
    "readme.txt",
]


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest}")
        return

    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def merge_attributes(attr_dir: Path) -> pd.DataFrame:
    txt_files = sorted(attr_dir.glob("camels_*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No camels_*.txt files found in {attr_dir}")

    dfs = []
    for fp in txt_files:
        df = pd.read_csv(fp, sep=";", dtype={"gauge_id": str})
        df["gauge_id"] = df["gauge_id"].astype(str).str.zfill(8)
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        dup_cols = [c for c in df.columns if c in merged.columns and c != "gauge_id"]
        df = df.drop(columns=dup_cols, errors="ignore")
        merged = merged.merge(df, on="gauge_id", how="outer")

    return merged


def extract_selected(zip_path: Path, subset_dir: Path, forcing: str) -> None:
    gauges = set(GAUGES)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for member in names:
            member_lower = member.lower()
            filename = Path(member).name

            forcing_hit = (
                f"/basin_mean_forcing/{forcing}/" in member_lower
                and member_lower.endswith(".txt")
                and any(g in filename for g in gauges)
            )

            flow_hit = (
                "/usgs_streamflow/" in member_lower
                and member_lower.endswith(".txt")
                and any(g in filename for g in gauges)
            )

            keep_misc = (
                filename.lower() == "readme.txt"
                or "gauge_information" in member_lower
                or "camels_name" in member_lower
            )

            if not (forcing_hit or flow_hit or keep_misc):
                continue

            parts = Path(member).parts
            rel_path = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            target = subset_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)

            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="camels_workspace")
    parser.add_argument("--forcing", type=str, default="daymet", choices=["daymet", "nldas", "maurer"])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    downloads_dir = root / "downloads"
    subset_dir = root / "subset"
    attr_dir = subset_dir / "camels_attributes_v2.0"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)
    attr_dir.mkdir(parents=True, exist_ok=True)

    zip_path = downloads_dir / "basin_timeseries_v1p2_metForcing_obsFlow.zip"

    if args.download:
        download_file(TIMESERIES_ZIP_URL, zip_path)
        for fname in ATTRIBUTE_FILES:
            download_file(f"{ZENODO_RECORD}/{fname}?download=1", attr_dir / fname)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {zip_path}. Either place the official CAMELS zip there or rerun with --download."
        )

    for fname in ATTRIBUTE_FILES:
        if not (attr_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing attribute file {(attr_dir / fname)}. Download or place the official files first."
            )

    print("[extract] extracting selected basins")
    extract_selected(zip_path, subset_dir, args.forcing)

    merged_attrs = merge_attributes(attr_dir)
    merged_attrs.to_csv(attr_dir / "camels_attributes_merged.csv", index=False)

    basin_list = merged_attrs[merged_attrs["gauge_id"].isin(GAUGES)].copy()
    keep_cols = [c for c in ["gauge_id", "gauge_name", "huc_02", "area_gages2"] if c in basin_list.columns]
    if not keep_cols:
        keep_cols = ["gauge_id"]
    basin_list[keep_cols].sort_values("gauge_id").to_csv(subset_dir / "basin_list.csv", index=False)

    with open(subset_dir / "inventory.txt", "w", encoding="utf-8") as f:
        f.write(f"Selected gauges: {len(GAUGES)}\n")
        f.write(f"Forcing product: {args.forcing}\n")

    print("[done] subset prepared at:", subset_dir)


if __name__ == "__main__":
    main()
