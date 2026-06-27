# Superstore Analytics & Talent Intelligence Pipeline

An end-to-end data analytics pipeline built in Python, covering SQL-driven EDA, machine learning models, interactive dashboards, automated reporting, and an AI-powered talent intelligence module using the Claude API.

> Built as part of B.Tech CS (Data Science) coursework at Manipal University Jaipur — extended with industry-relevant tooling aligned to real-world data analyst and talent intelligence roles.

---

## Tech Stack

| Area | Tools |
|---|---|
| Data & SQL | Python, pandas, SQLite, openpyxl |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn (regression, classification, clustering) |
| Automation | Custom pipeline scripts, JSON export, Excel automation |
| AI / Agentic | Claude API (Anthropic) |
| BI Export | Power BI-ready multi-sheet Excel |

---

## Project Structure

```
superstore-talent-analytics/
├── src/                        # Modular lab scripts
│   ├── lab1_sql_import.py      # Data ingestion + SQL queries
│   ├── lab2_cleaning.py        # Missing values, dedup, outlier detection
│   ├── lab3_features.py        # Feature engineering & normalization
│   ├── lab4_eda.py             # SQL + pandas EDA
│   ├── lab5_static_viz.py      # matplotlib / seaborn plots
│   ├── lab6_interactive_viz.py # Plotly dashboard + Power BI Excel export
│   ├── lab7_regression.py      # Linear regression models
│   ├── lab8_classification.py  # Decision Tree, Random Forest, SVM
│   ├── lab9_clustering.py      # K-Means + Hierarchical clustering
│   ├── lab10_automation.py     # Auto-report generation pipeline
│   └── lab11_ai_talent.py      # Claude API talent intelligence agent
├── notebooks/
│   └── superstore_pipeline.ipynb  # Full pipeline as Jupyter notebook
├── outputs/
│   └── plots/                  # Generated visualizations
├── requirements.txt
└── README.md
```

---

## Key Features

### SQL-Driven Analytics (Labs 1, 4, 10)
- Data loaded into SQLite and queried with raw SQL for all aggregations
- Demonstrates: category-level sales, regional profit breakdown, discount impact analysis, anomaly detection

### Power BI-Ready Export (Lab 6)
- Multi-sheet Excel output with structured tabs: `Raw_Data`, `Sales_by_Category`, `Profit_by_Region`, `Discount_Impact`, `Anomaly_Alerts`
- Import directly into Power BI Desktop with zero cleanup

### Machine Learning Models (Labs 7–9)
| Model | Task | Result |
|---|---|---|
| Simple Linear Regression | Predict Profit from Sales | R² = 0.25 |
| Multiple Linear Regression | Predict Profit (multi-feature) | R² = 0.36 |
| Decision Tree | Classify Profitable orders | Accuracy = 81.9% |
| Random Forest | Classify Profitable orders | Accuracy = 79.9% |
| K-Means (k=3) | Customer/order segmentation | Silhouette = 0.33 |

### Automation Pipeline (Lab 10)
- Fully automated report generation: ingests data → runs SQL → detects anomalies → exports Excel + JSON
- JSON output is API-ready for downstream consumption

### AI Talent Intelligence Agent (Lab 11)
- Calls Claude API with structured analytics context
- Generates actionable business insights and a talent screening brief
- Demonstrates agentic AI workflow design

---

## Cluster Insights (Lab 9 — K-Means)

| Cluster | Avg Sales | Avg Profit | Avg Discount | Profile |
|---|---|---|---|---|
| 0 | $1,894 | **-$157** | 0.47 | High-discount loss segment ⚠️ |
| 1 | $1,010 | $121 | 0.08 | Low-value profitable orders ✅ |
| 2 | $7,328 | $1,204 | 0.08 | High-value premium orders 🌟 |

---

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/superstore-talent-analytics.git
cd superstore-talent-analytics

# Install dependencies
pip install -r requirements.txt

# Run individual labs
python src/lab1_sql_import.py

# Or open the full notebook
jupyter notebook notebooks/superstore_pipeline.ipynb
```

> **Note:** Lab 11 (AI Talent Intelligence) requires an Anthropic API key.
> Set it as an environment variable: `export ANTHROPIC_API_KEY=your_key_here`

---

## Dataset

Uses a synthetically generated Superstore-style dataset (9,994 records) with columns:
`Ship_Mode`, `Segment`, `Category`, `Sub_Category`, `Region`, `Sales`, `Quantity`, `Discount`, `Profit`

Mirrors the structure of the widely-used [Sample Superstore dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final).

---

## Skills Demonstrated

- ✅ SQL querying and aggregation (SQLite + pandas)
- ✅ Data cleaning, feature engineering, normalization
- ✅ Statistical EDA and correlation analysis
- ✅ Interactive dashboards (Plotly)
- ✅ Power BI-ready Excel pipelines (openpyxl)
- ✅ Regression, classification, and clustering (scikit-learn)
- ✅ Automated reporting with anomaly detection
- ✅ AI/agentic API integration (Claude API)

---

## Author

**Balasubrahmanyam**
B.Tech Computer Science (Data Science) — Manipal University Jaipur, Batch 2023–2027

