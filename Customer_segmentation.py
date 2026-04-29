import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 13
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

xlsx_path = "D:/Projects/Customer segmentation/online_retail_II.xlsx"
df_2009 = pd.read_excel(xlsx_path, sheet_name="Year 2009-2010")
df_2010 = pd.read_excel(xlsx_path, sheet_name="Year 2010-2011")
df = pd.concat([df_2009, df_2010], ignore_index=True)

# Normalize column names (the two sheets use slightly different conventions)
df.columns = [c.strip().replace(" ", "") for c in df.columns]
rename_map = {"Invoice": "InvoiceNo", "Price": "UnitPrice", "CustomerID": "CustomerID"}
df = df.rename(columns=rename_map)
# Some versions ship it as 'Customer ID' -> 'CustomerID' after our strip+replace above

print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
df.head()
df.info()
df.describe(include="all").T

# Check for missing values

initial_rows = len(df)

# 1. Missing CustomerID
df = df.dropna(subset=["CustomerID"])

# 2. Cancellations
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# 3. Invalid quantities / prices
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# 4. Duplicates
df = df.drop_duplicates()

# 5. TotalPrice + correct dtypes
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype(int)

print(f"Initial rows:   {initial_rows:,}")
print(f"Cleaned rows:   {len(df):,}")
print(f"Rows dropped:   {initial_rows - len(df):,} ({(initial_rows - len(df))/initial_rows:.1%})")
print(f"Unique customers: {df['CustomerID'].nunique():,}")
print(f"Date range: {df['InvoiceDate'].min().date()} -> {df['InvoiceDate'].max().date()}")


# EDA: Revenue by country and over time

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Top 10 countries by revenue
country_rev = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False).head(10)
country_rev.plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("Top 10 Countries by Revenue")
axes[0].set_xlabel("Revenue (£)")
axes[0].invert_yaxis()

# Monthly revenue
monthly = df.set_index("InvoiceDate")["TotalPrice"].resample("ME").sum()
monthly.plot(ax=axes[1], marker="o", color="darkorange")
axes[1].set_title("Monthly Revenue")
axes[1].set_xlabel("")
axes[1].set_ylabel("Revenue (£)")

plt.tight_layout()
plt.show()

print(f"UK share of revenue: {country_rev.iloc[0] / df['TotalPrice'].sum():.1%}")

top_products = (df.groupby("Description")
                  .agg(Quantity=("Quantity", "sum"), Revenue=("TotalPrice", "sum"))
                  .sort_values("Revenue", ascending=False)
                  .head(10))
print(top_products)



# RFM features

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"Snapshot date: {snapshot_date.date()}")

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
).reset_index()

print(f"RFM table shape: {rfm.shape}")
rfm.describe()

# Distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"], ["#4C72B0", "#55A868", "#C44E52"]):
    sns.histplot(rfm[col], bins=50, ax=ax, color=color)
    ax.set_title(f"{col} distribution")
plt.tight_layout()
plt.show()

# Log transform + standardize
rfm_log = rfm.copy()
rfm_log[["Recency", "Frequency", "Monetary"]] = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log[["Recency", "Frequency", "Monetary"]])
rfm_scaled = pd.DataFrame(rfm_scaled, columns=["Recency", "Frequency", "Monetary"], index=rfm.index)

# Visualize post-transform
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col, color in zip(axes, ["Recency", "Frequency", "Monetary"], ["#4C72B0", "#55A868", "#C44E52"]):
    sns.histplot(rfm_scaled[col], bins=50, ax=ax, color=color)
    ax.set_title(f"{col} (log + standardized)")
plt.tight_layout()
plt.show()

# Determine optimal k using Elbow and Silhouette methods

k_range = range(2, 11)
inertias, silhouettes = [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(rfm_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(list(k_range), inertias, "o-", color="steelblue")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(list(k_range), silhouettes, "o-", color="darkorange")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette")

plt.tight_layout()
plt.show()

for k, inert, sil in zip(k_range, inertias, silhouettes):
    print(f"k={k}  inertia={inert:8.1f}  silhouette={sil:.4f}")

# Fit final model with chosen k

K_FINAL = 4

kmeans = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

rfm["Cluster"].value_counts().sort_index()

# Cluster profiling

profile = rfm.groupby("Cluster").agg(
    Recency=("Recency", "mean"),
    Frequency=("Frequency", "mean"),
    Monetary=("Monetary", "mean"),
).round(1)
profile["Count"] = rfm.groupby("Cluster").size()
profile.sort_values("Monetary", ascending=False)

# Heatmap of cluster centers (scaled values) — shows how each cluster differs
center_df = pd.DataFrame(
    kmeans.cluster_centers_, columns=["Recency", "Frequency", "Monetary"]
)
plt.figure(figsize=(8, 4))
sns.heatmap(center_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Cluster centers (scaled RFM)\nRed = high, Blue = low")
plt.ylabel("Cluster")
plt.show()

# Rank clusters by a simple composite score: low R is good, high F and M are good
score = (-profile["Recency"].rank()
         + profile["Frequency"].rank()
         + profile["Monetary"].rank())
ordered = score.sort_values(ascending=False).index.tolist()

# Apply names in order from best to worst
default_names = ["Champions", "Loyal Customers", "At Risk", "Hibernating"]
if K_FINAL != 4:
    default_names = [f"Segment {i+1}" for i in range(K_FINAL)]

name_map = {cluster_id: default_names[rank] for rank, cluster_id in enumerate(ordered)}
rfm["Segment"] = rfm["Cluster"].map(name_map)

segment_profile = rfm.groupby("Segment").agg(
    Customers=("Recency", "count"),
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    TotalRevenue=("Monetary", "sum"),
).round(1)
segment_profile["RevenueShare"] = (segment_profile["TotalRevenue"] / segment_profile["TotalRevenue"].sum() * 100).round(1).astype(str) + "%"
segment_profile.sort_values("TotalRevenue", ascending=False)


# Visualize segments in 2D using PCA

# PCA projection for 2D scatter
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(rfm_scaled)
rfm["PC1"], rfm["PC2"] = coords[:, 0], coords[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter colored by segment
palette = sns.color_palette("Set2", n_colors=K_FINAL)
for i, seg in enumerate(rfm["Segment"].unique()):
    sub = rfm[rfm["Segment"] == seg]
    axes[0].scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.6, label=seg, color=palette[i])
axes[0].set_title("Customer segments (2D PCA projection)")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
axes[0].legend()

# Segment size bar
seg_counts = rfm["Segment"].value_counts()
seg_counts.plot(kind="bar", ax=axes[1], color=palette[:len(seg_counts)])
axes[1].set_title("Customers per segment")
axes[1].set_ylabel("Customers")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.show()

# Revenue share vs customer share — the "who drives revenue" chart
summary = rfm.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    Revenue=("Monetary", "sum"),
)
summary["CustomerShare"] = summary["Customers"] / summary["Customers"].sum()
summary["RevenueShare"] = summary["Revenue"] / summary["Revenue"].sum()

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(summary))
w = 0.38
ax.bar(x - w/2, summary["CustomerShare"] * 100, w, label="Customer share", color="#4C72B0")
ax.bar(x + w/2, summary["RevenueShare"] * 100, w, label="Revenue share",  color="#C44E52")
ax.set_xticks(x)
ax.set_xticklabels(summary.index, rotation=20)
ax.set_ylabel("Share (%)")
ax.set_title("Customer share vs Revenue share per segment")
ax.legend()
plt.tight_layout()
plt.show()

summary.round(3)

# Save results

output = rfm[["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Segment"]]
output.to_csv("customer_segments.csv", index=False)
print(f"Saved {len(output):,} customers to customer_segments.csv")
output.head()