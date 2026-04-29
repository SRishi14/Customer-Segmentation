
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    print(f"[1/6] Loading {path}...")
    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    df = df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice"})
    print(f"       {len(df):,} raw rows loaded")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Cleaning...")
    initial = len(df)
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.drop_duplicates()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)
    print(f"       {len(df):,} clean rows ({initial - len(df):,} dropped)")
    return df.reset_index(drop=True)


def build_rfm(clean_df: pd.DataFrame) -> pd.DataFrame:
    print("[3/6] Building RFM features...")
    snapshot = clean_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Primary country per customer = country they transact in most
    primary_country = (clean_df.groupby(["CustomerID", "Country"]).size()
                       .reset_index(name="n")
                       .sort_values(["CustomerID", "n"], ascending=[True, False])
                       .drop_duplicates("CustomerID")[["CustomerID", "Country"]])

    rfm = clean_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        TotalItems=("Quantity", "sum"),
    ).reset_index()
    rfm = rfm.merge(primary_country, on="CustomerID", how="left")
    print(f"       {len(rfm):,} unique customers")
    return rfm


def cluster(rfm: pd.DataFrame, k: int) -> pd.DataFrame:
    print(f"[4/6] Clustering (k={k})...")
    # Log + standardize
    log_feats = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
    scaled = StandardScaler().fit_transform(log_feats)

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    rfm["Cluster"] = km.fit_predict(scaled)

    # PCA for Tableau scatter
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled)
    rfm["PC1"], rfm["PC2"] = coords[:, 0], coords[:, 1]
    print(f"       PCA explained variance: {pca.explained_variance_ratio_.sum():.1%}")
    return rfm


def name_segments(rfm: pd.DataFrame, k: int) -> pd.DataFrame:
    print("[5/6] Naming segments...")
    profile = rfm.groupby("Cluster").agg(
        R=("Recency", "mean"), F=("Frequency", "mean"), M=("Monetary", "mean"),
    )
    score = -profile["R"].rank() + profile["F"].rank() + profile["M"].rank()
    ordered = score.sort_values(ascending=False).index.tolist()

    persona_names = {
        2: ["High Value", "Low Value"],
        3: ["Champions", "Loyal Customers", "At Risk"],
        4: ["Champions", "Loyal Customers", "At Risk", "Hibernating"],
        5: ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Hibernating"],
        6: ["Champions", "Loyal Customers", "Potential Loyalists", "Needs Attention", "At Risk", "Hibernating"],
    }
    names = persona_names.get(k, [f"Segment {i+1}" for i in range(k)])
    mapping = {cid: names[rank] for rank, cid in enumerate(ordered)}
    rfm["Segment"] = rfm["Cluster"].map(mapping)

    # An ordering column so Tableau can sort segments from best to worst
    rank_map = {name: i for i, name in enumerate(names)}
    rfm["SegmentRank"] = rfm["Segment"].map(rank_map)

    for name in names:
        n = (rfm["Segment"] == name).sum()
        print(f"       {name}: {n:,} customers")
    return rfm


def add_rfm_quintiles(rfm: pd.DataFrame) -> pd.DataFrame:
    """Classic RFM scoring 1-5. 1 = best for all three axes."""
    # Recency: lower is better → rank ascending, then bin
    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5,
                             labels=[1, 2, 3, 4, 5]).astype(int)
    # Frequency / Monetary: higher is better → rank descending
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first", ascending=False),
                             5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first", ascending=False),
                             5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    )
    return rfm


def export(rfm: pd.DataFrame, transactions: pd.DataFrame, out_dir: Path):
    print("[6/6] Exporting CSVs...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Customer-level file
    customers_path = out_dir / "customers_segmented.csv"
    cols = ["CustomerID", "Country", "Segment", "SegmentRank", "Cluster",
            "Recency", "Frequency", "Monetary", "TotalItems",
            "R_Score", "F_Score", "M_Score", "RFM_Score",
            "FirstPurchase", "LastPurchase", "PC1", "PC2"]
    rfm[cols].to_csv(customers_path, index=False)
    print(f"       {customers_path}  ({customers_path.stat().st_size/1e6:.1f} MB, {len(rfm):,} rows)")

    # Transaction-level file with Segment joined in
    enriched = transactions.merge(
        rfm[["CustomerID", "Segment", "SegmentRank", "R_Score", "F_Score", "M_Score"]],
        on="CustomerID", how="left",
    )
    tx_path = out_dir / "transactions_enriched.csv"
    enriched.to_csv(tx_path, index=False)
    print(f"       {tx_path}  ({tx_path.stat().st_size/1e6:.1f} MB, {len(enriched):,} rows)")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("D:\/Projects/Customer segmentation/online_retail_II.xlsx"),
                        help="Path to the UCI .xlsx file")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of clusters")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Could not find {args.input}. "
            "Download online_retail_II.xlsx from "
            "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
        )

    raw = load_raw(args.input)
    clean_tx = clean(raw)
    rfm = build_rfm(clean_tx)
    rfm = cluster(rfm, args.k)
    rfm = name_segments(rfm, args.k)
    rfm = add_rfm_quintiles(rfm)
    export(rfm, clean_tx, args.output)

    print("\nDone. Open Tableau and connect to the two CSV files in ./output/.")
    print("Then follow TABLEAU_GUIDE.md to build the dashboard.")


if __name__ == "__main__":
    main()
