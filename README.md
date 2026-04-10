# 🏦 Fintech ML: Merchant Health Scoring & Transaction Fraud Detection

> **End-to-end ML pipeline on real-world payment transaction data — combining unsupervised merchant risk scoring with supervised fraud classification using XGBoost + Isolation Forest.**

---

## 📌 Project Overview

Payment ecosystems face two distinct but interconnected risks: **merchant-level churn risk** (which merchants are likely to become financially unstable or exit the platform?) and **transaction-level fraud** (which individual transactions are suspicious?). Most projects solve one or the other. This project solves both — in a single, cohesive ML pipeline — and produces actionable business intelligence at both the merchant portfolio level and the individual transaction level.

**Dataset:** PaySim-style online payments fraud detection dataset
**Scale:** 138,115 transactions | 68,439 unique merchants/accounts analyzed

---

## 🧩 Problem Statement

### Sub-Problem 1 — Merchant Health Scoring
A fintech platform needs to proactively identify which merchants in its network are at risk of churning, defaulting, or becoming financially unhealthy — *before* they actually do. Traditional approaches look at simple transaction volume. This project builds a composite, multi-signal **Merchant Health Score** that synthesizes behavioral, financial, temporal, and transactional signals into a single 0–100 score per merchant.

**Business question:** *Which merchants represent the highest risk to platform GMV, and how much GMV is actually at stake?*

### Sub-Problem 2 — Transaction Fraud Classification
Fraud in payment networks is an extreme class imbalance problem. In this dataset, fraudulent transactions represent a tiny fraction of total volume — yet each one is high-stakes. The goal is to build a model that:
- Detects fraud with high recall (missing a fraud case is expensive)
- Maintains precision (false positives erode merchant trust and operational capacity)
- Works reliably under real-world class imbalance without cheating via data leakage

---

## 🏗️ Architecture & Methodology

### Project 1 — Merchant Health Scoring System

#### Feature Engineering (13 signals across 5 dimensions)

| Dimension | Features |
|---|---|
| **RFM (Recency / Frequency / Monetary)** | Recency (steps since last transaction), Frequency (total transaction count), Monetary (total GMV) |
| **CLV Signal** | Decay-weighted CLV proxy: `Monetary × (1 / (1 + Recency/max_step))` — penalizes dormant merchants |
| **Revenue Risk** | Revenue volatility (std of transaction amounts), Growth trend (second-half GMV minus first-half GMV) |
| **Behavioral Risk** | Transaction type mix (% TRANSFER/CASH_OUT — high-risk types), Customer concentration (top recipient ratio), Balance drain ratio |
| **Temporal Activity** | Cohort age, Active days in last 7 days, Time-windowed engagement signals |

#### Scoring Methodology
Each feature is min-max normalized to a 0–100 score. Negative-direction features (high recency = bad, high volatility = bad) are inverted before normalization so that **higher score always = healthier merchant**. The final Health Score is the unweighted mean across all 13 normalized signals, producing a fully interpretable composite index.

#### Risk Segmentation
Merchants are bucketed into four risk tiers via quantile-based segmentation:
- **High Risk** (bottom 25%) — immediate intervention candidates
- **Medium Risk** (25th–50th percentile)
- **Low Risk** (50th–75th percentile)
- **Very Low Risk** (top 25%) — platform's healthiest merchants

---

### Project 2 — Fraud Detection Pipeline

#### Feature Engineering (25 features, zero leakage)

**Balance integrity features:**
- `balance_error_orig` — detects cases where origin balance doesn't reconcile post-transaction
- `balance_error_dest` — flags destination-side anomalies (a known strong fraud signal in payment data)

**Velocity & behavioral features:**
- Transaction count in last 24 steps, last 5 steps (burst detection)
- Time since last transaction per originator (sudden reactivation signal)
- Amount sum over rolling 48-step window

**User risk profile features:**
- Per-user historical average and std of amounts (z-score deviation)
- New recipient flag (`is_new_recipient`) — first-time sender/receiver pairs carry higher fraud risk
- Destination account risk metrics (avg received, transaction count)

**Temporal features:**
- Hour of day (encoded from step modulo 24)
- Unusual hour flag (midnight–5am, 10pm–midnight)

**Anomaly signal:**
- Isolation Forest anomaly score fed as a feature into XGBoost — creating a hybrid unsupervised + supervised pipeline

#### Model Training

**Hyperparameter Optimization:**
Bayesian search via **Optuna** — 20 trials optimizing AUC-PR (area under precision-recall curve) as the objective, since AUC-PR is the appropriate metric for imbalanced classification (unlike AUC-ROC, which can be misleadingly high on skewed datasets).

**Cross-validation:** 3-fold Stratified K-Fold, preserving fraud class distribution across folds.

**Class imbalance handling:** `scale_pos_weight` set to the negative/positive class ratio, giving XGBoost explicit awareness of the imbalance rather than relying on post-hoc resampling.

**Final model:** XGBoost with GPU-accelerated training (`tree_method='hist'`, `device='cuda'`).

---

## 📊 Results

### Merchant Health Scoring

| Metric | Value |
|---|---|
| Total merchants analyzed | 68,439 |
| High-risk merchants identified | 17,110 (25% of portfolio) |
| At-risk GMV represented | ₹2,176.74M (~₹2.18 Billion) |
| Health Score range | 0 – 100 (composite, interpretable) |
| Signals used | 13 features across 5 business dimensions |

> **Business impact framing:** One quarter of the entire merchant portfolio is flagged as high-risk, collectively accounting for ₹2.18B in GMV. This gives a risk team a clear, ranked, data-driven intervention list — rather than gut-feel account management.

### Fraud Detection (XGBoost + Isolation Forest Hybrid)

| Metric | Value |
|---|---|
| AUC-ROC | **0.9994** |
| AUC-PR (primary metric) | **0.9606** |
| Precision (fraud class) | **85%** |
| Recall (fraud class) | **96%** |
| F1-score (fraud class) | **0.90** |
| Anomalous transactions flagged (Isolation Forest) | **11%** of total volume |

> **Why AUC-PR matters more than AUC-ROC here:** With extreme class imbalance, a naive model predicting "not fraud" for everything achieves near-perfect AUC-ROC. AUC-PR of **0.9606** directly measures the model's ability to find fraud cases without drowning operations in false positives — this is the metric that translates to real compliance cost savings.

> **96% recall** means the model catches 96 out of every 100 actual fraud cases. In production, each missed fraud case has a direct financial and regulatory cost. **85% precision** means 85% of flagged transactions are genuine fraud — a false positive rate that keeps human review queues manageable.

---

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| ML — Supervised | `xgboost`, `scikit-learn` |
| ML — Unsupervised | `sklearn.ensemble.IsolationForest` |
| Hyperparameter tuning | `optuna` (Bayesian optimization) |
| Evaluation | `classification_report`, `roc_auc_score`, `precision_recall_curve`, `auc` |
| Visualization | `matplotlib`, `seaborn` |
| Compute | GPU-accelerated XGBoost (CUDA) |

---

## 📁 Repository Structure

```
├── notebook/
│   └── merchant_health_fraud_detection.ipynb   # Full end-to-end pipeline
├── data/
│   └── data.csv                                 # PaySim-style transaction data
└── README.md
```

## 💡 Key Design Decisions & Learnings

**1. AUC-PR over AUC-ROC as optimization target**
Training XGBoost to maximize AUC-PR rather than AUC-ROC is a deliberate choice. On imbalanced fraud data, AUC-ROC rewards a model for being right about the majority class. AUC-PR forces the model to actually find the rare positive class — which is the entire point.

**2. Isolation Forest as a feature, not just a standalone detector**
Rather than running Isolation Forest separately, its anomaly score is fed as a feature into XGBoost. This creates a hybrid architecture where the supervised model can learn *when* to trust the unsupervised signal and when to override it — consistently outperforming either approach alone.

**3. No target encoding for fraud label in feature set**
Features like `nameOrig_fraud_rate` were computed but deliberately excluded from the final feature set used for evaluation to prevent data leakage. Every feature was constructed to be inferable from historical data prior to the transaction in question.

**4. Composite Health Score interpretability**
The 0–100 Health Score is designed to be business-readable. Every contributing signal is directionally consistent (higher = healthier), which means a non-technical risk manager can understand why a merchant scored 32 vs 78 by looking at which sub-scores pulled the composite down — enabling explainability without SHAP or LIME overhead.

---

## 📈 Business Value Summary

| Output | Stakeholder | Value |
|---|---|---|
| Merchant Health Score (0–100) | Risk / Account Management teams | Prioritized, data-driven intervention list across 68K merchants |
| ₹2.18B at-risk GMV identification | Finance / CFO | Quantified portfolio risk in business currency (not just model metrics) |
| 96% fraud recall | Fraud & Compliance | Near-exhaustive fraud capture before financial loss occurs |
| 11% anomaly flag rate | Operations | Scalable unsupervised pre-screening layer for transaction review queues |

---

*Built as a portfolio demonstration of applied ML in fintech — covering the full lifecycle from raw transaction data to business-ready risk intelligence.*
