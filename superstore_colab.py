"""
=============================================================
  Superstore Analytics & Talent Intelligence Pipeline
  GOOGLE COLAB VERSION
  Author: Bala | MUJ B.Tech CS (Data Science) 2023-2027
=============================================================
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import sqlite3
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, confusion_matrix,
    silhouette_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Colab output folder ──────────────────────
OUTPUT_DIR = "/content/outputs"
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("  SUPERSTORE ANALYTICS & TALENT INTELLIGENCE PIPELINE")
print("=" * 60)


# ─────────────────────────────────────────────
#  Generate synthetic Superstore dataset
# ─────────────────────────────────────────────
def generate_superstore_data(n=9994, seed=42):
    rng = np.random.default_rng(seed)
    categories = ["Furniture", "Office Supplies", "Technology"]
    sub_cats = {
        "Furniture":       ["Bookcases", "Chairs", "Furnishings", "Tables"],
        "Office Supplies": ["Appliances", "Art", "Binders", "Envelopes",
                            "Fasteners", "Labels", "Paper", "Storage", "Supplies"],
        "Technology":      ["Accessories", "Copiers", "Machines", "Phones"],
    }
    segments   = ["Consumer", "Corporate", "Home Office"]
    regions    = ["East", "West", "Central", "South"]
    ship_modes = ["First Class", "Same Day", "Second Class", "Standard Class"]

    cat_arr  = rng.choice(categories, n)
    seg_arr  = rng.choice(segments, n)
    reg_arr  = rng.choice(regions, n)
    ship_arr = rng.choice(ship_modes, n)
    qty_arr  = rng.integers(1, 15, n)
    disc_arr = rng.choice(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], n,
        p=[0.4, 0.15, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.02],
    )
    base_sales = {"Furniture": 400, "Office Supplies": 60, "Technology": 600}
    sales_arr = np.array([
        rng.exponential(base_sales[c]) * qty_arr[i] * (1 - disc_arr[i] * 0.5)
        for i, c in enumerate(cat_arr)
    ]).clip(5, 10000).round(2)
    profit_arr = (sales_arr * (
        rng.normal(0.18, 0.12, n) - disc_arr * 0.5
    )).round(2)
    sub_cat_arr = [rng.choice(sub_cats[c]) for c in cat_arr]

    return pd.DataFrame({
        "Ship_Mode":    ship_arr,
        "Segment":      seg_arr,
        "Category":     cat_arr,
        "Sub_Category": sub_cat_arr,
        "Region":       reg_arr,
        "Sales":        sales_arr,
        "Quantity":     qty_arr,
        "Discount":     disc_arr,
        "Profit":       profit_arr,
    })


# ══════════════════════════════════════════════
#  LAB 1 — Data Collection & SQL Import
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 1 — Data Collection & SQL Import")
print("─" * 50)

df_raw = generate_superstore_data()

conn = sqlite3.connect(":memory:")
df_raw.to_sql("superstore", conn, if_exists="replace", index=False)

queries = {
    "Total Records":      "SELECT COUNT(*) AS total FROM superstore",
    "By Category":        "SELECT Category, COUNT(*) AS count FROM superstore GROUP BY Category",
    "Avg Sales by Region":"SELECT Region, ROUND(AVG(Sales),2) AS avg_sales FROM superstore GROUP BY Region ORDER BY avg_sales DESC",
}
for label, q in queries.items():
    print(f"\n[SQL] {label}:")
    print(pd.read_sql_query(q, conn).to_string(index=False))

print(f"\nShape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print("Lab 1 ✓")


# ══════════════════════════════════════════════
#  LAB 2 — Data Cleaning
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 2 — Data Cleaning")
print("─" * 50)

df = df_raw.copy()
print(f"Missing values:\n{df.isnull().sum()}")
df.drop_duplicates(inplace=True)
print(f"Rows after dedup: {len(df)}")

for col in ["Ship_Mode", "Segment", "Category", "Region"]:
    df[col] = df[col].astype("category")
df["Quantity"] = df["Quantity"].astype(int)
for col in ["Sales", "Profit", "Discount"]:
    df[col] = df[col].astype(float)

Q1, Q3 = df["Sales"].quantile(0.25), df["Sales"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["Sales"] < Q1 - 1.5*IQR) | (df["Sales"] > Q3 + 1.5*IQR)]
print(f"Outliers in Sales: {len(outliers)}")

cleaned_path = os.path.join(OUTPUT_DIR, "superstore_cleaned.csv")
df.to_csv(cleaned_path, index=False)
print(f"Saved → {cleaned_path}")
print("Lab 2 ✓")


# ══════════════════════════════════════════════
#  LAB 3 — Feature Engineering
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 3 — Feature Engineering & Transformation")
print("─" * 50)

df["Profit_Margin"]    = (df["Profit"] / df["Sales"] * 100).round(2)
df["Revenue_per_Unit"] = (df["Sales"] / df["Quantity"]).round(2)
df["Discount_Category"] = pd.cut(
    df["Discount"],
    bins=[-0.01, 0, 0.2, 0.5, 1],
    labels=["No Discount", "Low", "Medium", "High"],
)

for col in ["Sales", "Profit"]:
    mn, mx = df[col].min(), df[col].max()
    df[f"{col}_Norm"] = ((df[col] - mn) / (mx - mn)).round(4)

scaler = StandardScaler()
df[["Sales_Zscore", "Profit_Zscore"]] = scaler.fit_transform(df[["Sales", "Profit"]]).round(4)

le = LabelEncoder()
for col in ["Region", "Category", "Segment"]:
    df[f"{col}_Label"] = le.fit_transform(df[col])

print("New features: Profit_Margin, Revenue_per_Unit, Discount_Category")
print(df[["Sales", "Profit", "Profit_Margin", "Revenue_per_Unit",
          "Sales_Norm", "Sales_Zscore"]].head(3).to_string(index=False))

engineered_path = os.path.join(OUTPUT_DIR, "superstore_engineered.csv")
df.to_csv(engineered_path, index=False)
print(f"Saved → {engineered_path}")
print("Lab 3 ✓")


# ══════════════════════════════════════════════
#  LAB 4 — EDA with SQL + pandas
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 4 — Exploratory Data Analysis")
print("─" * 50)

conn2 = sqlite3.connect(":memory:")
df.to_sql("superstore_eng", conn2, if_exists="replace", index=False)

eda_queries = {
    "Sales by Category": """
        SELECT Category,
               ROUND(SUM(Sales),2) AS Total_Sales,
               ROUND(AVG(Sales),2) AS Avg_Sales
        FROM superstore_eng GROUP BY Category ORDER BY Total_Sales DESC""",
    "Profit by Region": """
        SELECT Region,
               ROUND(SUM(Profit),2) AS Total_Profit,
               ROUND(AVG(Profit),2) AS Avg_Profit
        FROM superstore_eng GROUP BY Region ORDER BY Total_Profit DESC""",
    "Discount Impact": """
        SELECT Discount_Category,
               ROUND(AVG(Profit),2) AS Avg_Profit,
               COUNT(*) AS Records
        FROM superstore_eng GROUP BY Discount_Category""",
}
for label, q in eda_queries.items():
    print(f"\n[SQL] {label}:")
    print(pd.read_sql_query(q, conn2).to_string(index=False))

print(f"\nCorrelation Matrix:")
print(df[["Sales", "Profit", "Quantity", "Discount"]].corr().round(3))
print("Lab 4 ✓")


# ══════════════════════════════════════════════
#  LAB 5 — Static Visualizations
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 5 — Static Visualizations")
print("─" * 50)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Superstore — EDA Plots", fontsize=16, fontweight="bold")

cat_sales  = df.groupby("Category")["Sales"].sum().reset_index()
reg_profit = df.groupby("Region")["Profit"].sum().reset_index()

axes[0,0].bar(cat_sales["Category"], cat_sales["Sales"],
              color=["#4C72B0","#DD8452","#55A868"])
axes[0,0].set_title("Sales by Category"); axes[0,0].set_ylabel("Sales ($)")

axes[0,1].bar(reg_profit["Region"], reg_profit["Profit"],
              color=["green" if p>0 else "red" for p in reg_profit["Profit"]])
axes[0,1].set_title("Profit by Region"); axes[0,1].set_ylabel("Profit ($)")

axes[0,2].hist(df["Sales"], bins=40, color="steelblue", edgecolor="white")
axes[0,2].set_title("Sales Distribution")

axes[1,0].hist(df["Profit"], bins=40, color="tomato", edgecolor="white")
axes[1,0].set_title("Profit Distribution")

seg_data = [df[df["Segment"]==s]["Profit"].values for s in df["Segment"].cat.categories]
axes[1,1].boxplot(seg_data, labels=df["Segment"].cat.categories)
axes[1,1].set_title("Profit by Segment")

for cat, color in zip(df["Category"].cat.categories, ["#4C72B0","#DD8452","#55A868"]):
    sub = df[df["Category"]==cat]
    axes[1,2].scatter(sub["Sales"], sub["Profit"], alpha=0.3, s=10, color=color, label=cat)
axes[1,2].set_title("Sales vs Profit"); axes[1,2].legend(fontsize=7)

plt.tight_layout()
static_plot_path = os.path.join(PLOTS_DIR, "lab5_static_plots.png")
plt.savefig(static_plot_path, dpi=150, bbox_inches="tight")
plt.show()  # ← shows inline in Colab
print(f"Saved → {static_plot_path}")
print("Lab 5 ✓")


# ══════════════════════════════════════════════
#  LAB 6 — Interactive Dashboard + Power BI Export
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 6 — Interactive Dashboard + Power BI Export")
print("─" * 50)

pbi_cat     = df.groupby("Category").agg(
    Total_Sales=("Sales","sum"), Avg_Profit=("Profit","mean"),
    Record_Count=("Sales","count")).reset_index().round(2)
pbi_region  = df.groupby("Region").agg(
    Total_Sales=("Sales","sum"), Total_Profit=("Profit","sum"),
    Avg_Discount=("Discount","mean")).reset_index().round(2)
pbi_segment = df.groupby("Segment").agg(
    Total_Sales=("Sales","sum"),
    Avg_Profit_Margin=("Profit_Margin","mean")).reset_index().round(2)
pbi_discount = df.groupby("Discount_Category", observed=True).agg(
    Avg_Sales=("Sales","mean"), Avg_Profit=("Profit","mean"),
    Count=("Sales","count")).reset_index().round(2)

pbi_path = os.path.join(OUTPUT_DIR, "powerbi_ready_superstore.xlsx")
with pd.ExcelWriter(pbi_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Raw_Data", index=False)
    pbi_cat.to_excel(writer, sheet_name="Sales_by_Category", index=False)
    pbi_region.to_excel(writer, sheet_name="Profit_by_Region", index=False)
    pbi_segment.to_excel(writer, sheet_name="Sales_by_Segment", index=False)
    pbi_discount.to_excel(writer, sheet_name="Discount_Impact", index=False)
print(f"Power BI Excel → {pbi_path}")

fig_dash = make_subplots(rows=2, cols=2,
    subplot_titles=["Sales by Category","Profit by Region",
                    "Sales vs Profit","Profit by Segment"])
fig_dash.add_trace(go.Bar(x=pbi_cat["Category"], y=pbi_cat["Total_Sales"],
    marker_color=["#4C72B0","#DD8452","#55A868"]), row=1, col=1)
fig_dash.add_trace(go.Bar(x=pbi_region["Region"], y=pbi_region["Total_Profit"],
    marker_color=["green" if v>0 else "red" for v in pbi_region["Total_Profit"]]), row=1, col=2)
for cat, color in zip(df["Category"].cat.categories, ["#4C72B0","#DD8452","#55A868"]):
    sub = df[df["Category"]==cat].sample(min(300,len(df)), random_state=42)
    fig_dash.add_trace(go.Scatter(x=sub["Sales"], y=sub["Profit"], mode="markers",
        marker=dict(size=4, color=color, opacity=0.5), name=str(cat)), row=2, col=1)
for seg in df["Segment"].cat.categories:
    fig_dash.add_trace(go.Box(y=df[df["Segment"]==seg]["Profit"], name=str(seg)), row=2, col=2)
fig_dash.update_layout(height=650, title_text="Superstore Interactive Dashboard", showlegend=False)

dashboard_path = os.path.join(OUTPUT_DIR, "dashboard.html")
fig_dash.write_html(dashboard_path)
fig_dash.show()  # ← shows inline in Colab
print(f"Dashboard HTML → {dashboard_path}")
print("Lab 6 ✓")


# ══════════════════════════════════════════════
#  LAB 7 — Regression Models
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 7 — Regression Models")
print("─" * 50)

features = ["Sales","Quantity","Discount","Region_Label","Category_Label","Segment_Label"]
X, y = df[features], df["Profit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

results_reg = {}

slr = LinearRegression()
slr.fit(X_train[["Sales"]], y_train)
slr_pred = slr.predict(X_test[["Sales"]])
results_reg["Simple LR"] = {
    "R2":   round(r2_score(y_test, slr_pred), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, slr_pred)), 4),
}

mlr = LinearRegression()
mlr.fit(X_train, y_train)
mlr_pred = mlr.predict(X_test)
results_reg["Multiple LR"] = {
    "R2":   round(r2_score(y_test, mlr_pred), 4),
    "RMSE": round(np.sqrt(mean_squared_error(y_test, mlr_pred)), 4),
}

print("\nRegression Results:")
print(pd.DataFrame(results_reg).T)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, mlr_pred, alpha=0.3, s=10, color="steelblue")
lims = [min(y_test.min(), mlr_pred.min()), max(y_test.max(), mlr_pred.max())]
ax.plot(lims, lims, "r--", lw=1.5)
ax.set_xlabel("Actual Profit"); ax.set_ylabel("Predicted Profit")
ax.set_title("Actual vs Predicted Profit (Multiple LR)")
reg_plot_path = os.path.join(PLOTS_DIR, "lab7_regression.png")
plt.savefig(reg_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print("Lab 7 ✓")


# ══════════════════════════════════════════════
#  LAB 8 — Classification Models
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 8 — Classification Models")
print("─" * 50)

df["Profitable"] = (df["Profit"] > 0).astype(int)
print(f"Profitable orders: {df['Profitable'].mean()*100:.1f}%")

Xc, yc = df[features], df["Profitable"]
Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42)

classifiers = {
    "Decision Tree":  DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":            SVC(kernel="rbf", random_state=42),
}
results_cls = {}
for name, clf in classifiers.items():
    clf.fit(Xc_train, yc_train)
    preds = clf.predict(Xc_test)
    acc = round(accuracy_score(yc_test, preds), 4)
    results_cls[name] = acc
    print(f"\n{name} — Accuracy: {acc}")
    print(confusion_matrix(yc_test, preds))

cls_df = pd.DataFrame(list(results_cls.items()), columns=["Model","Accuracy"])

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(cls_df["Model"], cls_df["Accuracy"],
              color=["#4C72B0","#DD8452","#55A868"], width=0.5)
for bar, val in zip(bars, cls_df["Accuracy"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.4f}", ha="center", fontsize=9)
ax.set_ylim(0, 1.1); ax.set_title("Classification Accuracy Comparison")
cls_plot_path = os.path.join(PLOTS_DIR, "lab8_classification.png")
plt.savefig(cls_plot_path, dpi=150, bbox_inches="tight")
plt.show()
print("Lab 8 ✓")


# ══════════════════════════════════════════════
#  LAB 9 — Clustering
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 9 — Clustering Techniques")
print("─" * 50)

cluster_cols = ["Sales","Profit","Quantity","Discount"]
X_clust = StandardScaler().fit_transform(df[cluster_cols])

wss = [KMeans(n_clusters=k, n_init=25, random_state=42).fit(X_clust).inertia_
       for k in range(1, 11)]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(1,11), wss, "bo-")
axes[0].axvline(3, color="red", linestyle="--", label="k=3")
axes[0].set_title("Elbow Method"); axes[0].legend()

kmeans    = KMeans(n_clusters=3, n_init=25, random_state=42)
km_labels = kmeans.fit_predict(X_clust)
df["KMeans_Cluster"] = km_labels
km_sil = round(silhouette_score(X_clust, km_labels), 4)

hc        = AgglomerativeClustering(n_clusters=3, linkage="ward")
hc_labels = hc.fit_predict(X_clust)
hc_sil    = round(silhouette_score(X_clust, hc_labels), 4)

sample_idx = np.random.choice(len(X_clust), 100, replace=False)
Z = linkage(X_clust[sample_idx], method="ward")
dendrogram(Z, ax=axes[1], no_labels=True, color_threshold=5)
axes[1].axhline(5, color="red", linestyle="--")
axes[1].set_title("Hierarchical Dendrogram (n=100)")

clust_plot_path = os.path.join(PLOTS_DIR, "lab9_clustering.png")
plt.savefig(clust_plot_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"K-Means Silhouette: {km_sil} | Hierarchical Silhouette: {hc_sil}")
print("\nCluster Profiles:")
print(df.groupby("KMeans_Cluster")[cluster_cols].mean().round(2))
print("Lab 9 ✓")


# ══════════════════════════════════════════════
#  LAB 10 — Automation Pipeline
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 10 — Automation Pipeline")
print("─" * 50)

def generate_automated_report(df, output_path):
    conn_auto = sqlite3.connect(":memory:")
    df.to_sql("data", conn_auto, if_exists="replace", index=False)

    summary = pd.read_sql_query("""
        SELECT Category, Region,
               ROUND(SUM(Sales),2)    AS Total_Sales,
               ROUND(AVG(Profit),2)   AS Avg_Profit,
               ROUND(AVG(Discount),3) AS Avg_Discount,
               COUNT(*)               AS Records,
               ROUND(SUM(CASE WHEN Profit>0 THEN 1.0 ELSE 0 END)/COUNT(*)*100,1) AS Profitable_Pct
        FROM data GROUP BY Category, Region ORDER BY Total_Sales DESC
    """, conn_auto)

    anomalies = pd.read_sql_query("""
        SELECT Category, Region,
               ROUND(AVG(Discount),3) AS Avg_Discount,
               ROUND(AVG(Profit),2)   AS Avg_Profit,
               COUNT(*) AS Count
        FROM data WHERE Discount >= 0.4 AND Profit < 0
        GROUP BY Category, Region ORDER BY Avg_Profit ASC LIMIT 10
    """, conn_auto)

    top5 = pd.read_sql_query("""
        SELECT Category, Region,
               ROUND(SUM(Sales),2)  AS Total_Sales,
               ROUND(SUM(Profit),2) AS Total_Profit
        FROM data GROUP BY Category, Region
        ORDER BY Total_Profit DESC LIMIT 5
    """, conn_auto)

    overall = {
        "total_records":         int(len(df)),
        "total_sales":           round(df["Sales"].sum(), 2),
        "total_profit":          round(df["Profit"].sum(), 2),
        "avg_profit_margin_pct": round(df["Profit_Margin"].mean(), 2),
        "profitable_orders_pct": round((df["Profit"]>0).mean()*100, 1),
        "anomalies_flagged":     int(len(anomalies)),
    }

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([overall]).to_excel(writer, sheet_name="Overall_Metrics", index=False)
        summary.to_excel(writer, sheet_name="Category_Region_Summary", index=False)
        anomalies.to_excel(writer, sheet_name="Anomaly_Alerts", index=False)
        top5.to_excel(writer, sheet_name="Top_Performers", index=False)
        cls_df.to_excel(writer, sheet_name="Model_Performance", index=False)

    json_path = output_path.replace(".xlsx", ".json")
    with open(json_path, "w") as f:
        json.dump(overall, f, indent=2)

    return overall, json_path

auto_report_path = os.path.join(OUTPUT_DIR, "auto_report.xlsx")
metrics, json_path = generate_automated_report(df, auto_report_path)

print(f"Excel report → {auto_report_path}")
print(f"JSON report  → {json_path}")
print("\nKey Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")
print("Lab 10 ✓")


# ══════════════════════════════════════════════
#  LAB 11 — AI Talent Intelligence (Claude API)
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("LAB 11 — AI Talent Intelligence (Claude API)")
print("─" * 50)

import urllib.request, ssl

# ── Paste your Anthropic API key below ──────
ANTHROPIC_API_KEY = ""   # ← add your key here to enable real AI output

def call_claude(prompt, system=""):
    if not ANTHROPIC_API_KEY:
        return "[Add your ANTHROPIC_API_KEY at the top of Lab 11 to get real AI output]"
    url = "https://api.anthropic.com/v1/messages"
    payload = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("content-type", "application/json")
    req.add_header("x-api-key", ANTHROPIC_API_KEY)
    req.add_header("anthropic-version", "2023-06-01")
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            return json.loads(resp.read())["content"][0]["text"]
    except Exception as e:
        return f"API error: {e}"

analytics_context = f"""
Superstore Dataset: {len(df):,} orders

Overall:
- Total Sales: ${df['Sales'].sum():,.2f}
- Total Profit: ${df['Profit'].sum():,.2f}
- Avg Profit Margin: {df['Profit_Margin'].mean():.2f}%
- Profitable Orders: {(df['Profit']>0).mean()*100:.1f}%

Sales by Category:
{df.groupby('Category')['Sales'].sum().round(2).to_string()}

Profit by Region:
{df.groupby('Region')['Profit'].sum().round(2).to_string()}

Key finding: High-discount (≥40%) orders average -$281 profit
Best classifier: Decision Tree at {results_cls['Decision Tree']:.2%} accuracy
"""

print("Calling Claude API for business insights...")
insights = call_claude(
    analytics_context,
    system="You are a senior BI analyst. Give 3-4 concise, actionable bullet points based on the data. Be specific with numbers."
)
print(f"\n[AI Business Insights]\n{insights}")

talent_prompt = """
A company is hiring for data analytics / talent acquisition roles.
Skills needed: SQL, Python, Power BI, data visualization, ML/AI, automation.
Generate a structured talent brief:
1. Top 3 skills to prioritize (with rationale)
2. One sharp screening question per skill
3. One risk flag for the hiring team
Keep it under 200 words.
"""
print("\nRunning Talent Intelligence Agent...")
talent_output = call_claude(talent_prompt)
print(f"\n[Talent Intelligence Output]\n{talent_output}")

ai_path = os.path.join(OUTPUT_DIR, "ai_insights.txt")
with open(ai_path, "w") as f:
    f.write("BUSINESS INSIGHTS\n" + "="*40 + "\n" + insights)
    f.write("\n\nTALENT INTELLIGENCE\n" + "="*40 + "\n" + talent_output)
print(f"\nSaved → {ai_path}")
print("Lab 11 ✓")


# ══════════════════════════════════════════════
#  DOWNLOAD ALL OUTPUTS (Colab only)
# ══════════════════════════════════════════════
print("\n" + "─" * 50)
print("Downloading output files to your computer...")
print("─" * 50)

from google.colab import files

output_files = [
    os.path.join(OUTPUT_DIR, "superstore_cleaned.csv"),
    os.path.join(OUTPUT_DIR, "superstore_engineered.csv"),
    os.path.join(OUTPUT_DIR, "powerbi_ready_superstore.xlsx"),
    os.path.join(OUTPUT_DIR, "auto_report.xlsx"),
    os.path.join(OUTPUT_DIR, "dashboard.html"),
    os.path.join(PLOTS_DIR,  "lab5_static_plots.png"),
    os.path.join(PLOTS_DIR,  "lab7_regression.png"),
    os.path.join(PLOTS_DIR,  "lab8_classification.png"),
    os.path.join(PLOTS_DIR,  "lab9_clustering.png"),
]

for f_path in output_files:
    if os.path.exists(f_path):
        files.download(f_path)
        print(f"  ↓ {os.path.basename(f_path)}")

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE ✅")
print("=" * 60)
print(f"""
Labs completed:
  ✅ Lab 1  — SQL data ingestion
  ✅ Lab 2  — Data cleaning
  ✅ Lab 3  — Feature engineering
  ✅ Lab 4  — SQL + pandas EDA
  ✅ Lab 5  — Static visualizations
  ✅ Lab 6  — Interactive dashboard + Power BI export
  ✅ Lab 7  — Regression models
  ✅ Lab 8  — Classification models
  ✅ Lab 9  — Clustering
  ✅ Lab 10 — Automated report pipeline
  ✅ Lab 11 — AI talent intelligence
""")
