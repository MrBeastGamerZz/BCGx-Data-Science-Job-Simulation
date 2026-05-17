# 🔍 BCG X — Data Science Job Simulation
### Predicting Customer Churn for PowerCo | End-to-End Data Science Pipeline

<br>

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=for-the-badge)

<br>

> **Virtual Experience Programme** — BCGX via [Forage](https://www.theforage.com/simulations/bcg/data-science-ccdz)  
> Simulated the role of a Junior Data Scientist at BCG X, working on a real-world customer churn problem for energy utility client **PowerCo**.

---

## 📌 Project Overview

PowerCo — a major gas and electricity supplier to SMEs — was experiencing rising customer churn in an increasingly competitive energy market. The company hypothesised that **price sensitivity** was the primary driver of customer attrition.

As a BCG X Data Scientist, I was tasked with:
- Investigating whether price sensitivity truly drives churn
- Building a predictive model to identify at-risk customers before they leave
- Translating findings into actionable business recommendations for senior stakeholders

**Key Finding:** Price sensitivity is *a* contributor to churn — but not the primary driver. Customer tenure, contract timing, and net margin were stronger predictors, fundamentally changing the recommended retention strategy.

---

## 📜 Programme Details

| | |
|---|---|
| **Programme** | BCGX Data Science Job Simulation |
| **Platform** | Forage |
| **Difficulty** | Advanced |
| **Duration** | ~8–9 hours, self-paced |
| **Status** | ✅ Completed |
| **Location** | Bengaluru, Karnataka, India |
| **Certificate** | [View Certificate](certificates/) |
| **Simulation Link** | [The Forage — BCGX Data Science](https://www.theforage.com/simulations/bcg/data-science-ccdz) |


---

## 📁 Repository Structure

```
BCGx-Data-Science-Job-Simulation/
│
├── 📂 eda/                            # Task 2 — Exploratory Data Analysis
│   └── Task 2 -eda_starter.ipynb
│
├── 📂 featured_eng/                   # Task 3 — Feature Engineering
│   └── Task 3 -feature_engineering.ipynb
│
├── 📂 model/                          # Task 4 — Modeling & Evaluation
│   └── Task 4 -modeling_starter.ipynb
│
├── 📄 Data Description (1).pdf        # Dataset column definitions
├── 📄 Executive Summary Template.pptx
└── 📄 README.md
```

---

## 🗂️ Tasks Completed

| # | Task | Description | Deliverable |
|---|---|---|---|
| 1 | Background | BCG X methodology & client brief | — |
| 2 | Problem Framing | Define data needs & analytical approach | Professional email |
| 3 | EDA & Cleaning | Explore data, fix quality issues, visualise distributions | Jupyter Notebook |
| 4 | Feature Engineering | Create 20+ predictive features from raw data | Jupyter Notebook |
| 5 | Modeling & Evaluation | Train & evaluate Random Forest churn classifier | Jupyter Notebook |
| 6 | Insights & Recommendations | Executive summary for senior stakeholders | BCG-style slide |

---

## 📊 Datasets

Two datasets provided by PowerCo, analysed across **14,606 customers** and **193,002 pricing records**:

**`client_data.csv`** — 27 columns including:
- Customer profile (industry, channel, tenure, contract dates)
- Energy & gas consumption (12-month history + forecasts)
- Financial metrics (net margin, gross margin, subscribed power)
- Target variable: `churn` — whether the customer churned in the next 3 months

**`price_data.csv`** — 8 columns including:
- Monthly variable and fixed prices across off-peak, peak, and mid-peak periods
- Date range: Jan–Dec 2015 (12 months of price history per customer)

---

## 🔬 Methodology

Following BCG X's 5-step data science framework:

### 1️⃣ Business Understanding & Problem Framing
- Identified data requirements across 5 categories: customer profile, pricing & contracts, consumption, service interactions, and churn labels
- Drafted an internal stakeholder email outlining the analytical approach and data requests

### 2️⃣ Exploratory Data Analysis & Data Cleaning

**Key findings:**
- **9.7% churn rate** — significant class imbalance (13,187 stayed vs 1,419 churned)
- Hidden `'MISSING'` text values in `channel_sales` (3,725) and `origin_up` (64) — retained as predictive categories
- Date columns stored as `object` — converted to `datetime64`
- `has_gas` stored as `'t'`/`'f'` — converted to boolean
- All consumption columns heavily **right-skewed** — log transform applied in Feature Engineering
- Churn rate varies noticeably by sales channel — indicating channel as a meaningful predictor

**Visualisations produced:**
- Stacked percentage bar charts (churn rate by sales channel and origin contract)
- Stacked histograms comparing Retention vs Churn across 6 numeric columns
- Box plots for consumption outlier analysis
- Price trend line charts across variable and fixed components over time

### 3️⃣ Feature Engineering

Created **20+ new features** from raw data using the BCG 4-step framework (Remove → Expand → Combine → Merge):

| Feature Group | Features | Business Rationale |
|---|---|---|
| **Price Signal (Estelle's)** | `offpeak_diff_dec_january_energy/power` | Year-on-year price jump signal |
| **Price Period Spreads** | 6 max monthly diff features | Worst spread experienced — direct price sensitivity measure |
| **Contract Timing** | `months_to_end`, `months_activ`, `months_modif_prod`, `months_renewal` | Contract expiry = peak churn risk window |
| **Log Transforms** | 10 skewed consumption & forecast columns | Fixes right-skew distribution for model stability |
| **Encoding** | `has_gas` → 0/1 + one-hot `channel_sales` & `origin_up` | Converts categoricals; drops rare categories (<100 customers) |

**Key decisions:**
- Precise `convert_months()` helper — calendar-accurate vs. dividing raw days
- Applied `log10(x+1)` — handles zeros safely, more interpretable at scale
- Critical: sorted price data by `['id', 'price_date']` before `.first()`/`.last()` to correctly capture Jan and Dec prices
- Kept `'MISSING'` as its own category — its absence may itself signal churn risk

### 4️⃣ Modeling & Evaluation

**Model:** Random Forest Classifier

```python
RandomForestClassifier(
    n_estimators=1000,        # Stable ensemble size
    class_weight='balanced',  # Compensates for 9.7% class imbalance
    max_depth=10,             # Prevents overfitting
    random_state=42           # Reproducibility
)
```

**Train/Test Split:** 75% / 25% · `random_state=42`

**Evaluation approach:**

| Metric | Justification |
|---|---|
| **Recall** ⭐ Primary | Missing a real churner = permanent revenue loss |
| **Precision** | Controls unnecessary discount spend |
| **F1 Score** | Balanced view across imbalanced classes |
| **AUC-ROC** | Overall model discriminative power |
| **Confusion Matrix** | Full breakdown of TP / TN / FP / FN |

> **Why not Accuracy?** A model predicting "stayed" for every customer achieves 90.3% accuracy while catching zero churners. Accuracy is meaningless on imbalanced datasets.

**False Negative vs False Positive — Business Cost:**
- **FN (missed churner):** Customer leaves → permanent revenue loss ❌
- **FP (flagged non-churner):** Unnecessary discount → small recoverable cost ⚠️
- **Conclusion:** Maximise Recall — catching real churners is the priority

### 5️⃣ Insights & Recommendations

**Hypothesis verdict:**

> ❌ Price sensitivity is **NOT** the primary driver of churn at PowerCo.

Feature importance analysis showed:
- **Top drivers:** Net margin, months until contract end, customer tenure
- **Price features:** Contributed — but ranked mid-table, not dominant

**Recommendations for PowerCo:**
1. Deploy monthly churn scoring across all active customers
2. Prioritise customers with contracts expiring within 90 days
3. Tailor interventions by risk type — price offers vs. loyalty incentives
4. Abandon blanket price discounting — expensive, misdirected, not data-supported
5. Retrain model quarterly as market conditions evolve

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.7+` | Core language |
| `pandas` | Data manipulation, groupby, merging |
| `numpy` | Numerical ops, log transforms |
| `matplotlib` | Charts, histograms, trend lines |
| `seaborn` | Box plots, heatmaps |
| `scikit-learn` | Modeling, metrics, train/test split |
| `Jupyter Notebook` | All deliverable notebooks |

---

## 📈 Project Numbers at a Glance

| | |
|---|---|
| 👥 Customers analysed | 14,606 |
| 💰 Price records processed | 193,002 |
| 📉 Overall churn rate | 9.7% |
| 🔧 Features engineered | 20+ |
| 🌲 Model trees | 1,000 |
| ✅ Tasks completed | 6 / 6 |

---

## 📬 Connect

<p>
  <a href="https://www.linkedin.com/in/manjunathareddyn/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin" alt="LinkedIn">
  </a>
  &nbsp;
  <a href="https://github.com/MrBeastGamerZz">
    <img src="https://img.shields.io/badge/GitHub-MrBeastGamerZz-181717?style=flat-square&logo=github" alt="GitHub">
  </a>
</p>

---

<p align="center">
  <i>Built as part of the BCG X Data Science Virtual Experience Programme on Forage</i>
</p>

