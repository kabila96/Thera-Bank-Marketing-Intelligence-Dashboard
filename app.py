from __future__ import annotations

from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Thera Bank Marketing Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Bank_Personal_Loan_Modelling.csv"
OUTPUT_DIR = BASE_DIR / "bank_loan_outputs_portfolio"

PROFIT_PER_SUCCESS = 1200
CONTACT_COST = 25
DEFAULT_THRESHOLD = 0.50
PROFIT_THRESHOLD = 0.16

EDUCATION_MAP = {1: "Undergraduate", 2: "Graduate", 3: "Advanced / Professional"}
YN_MAP = {0: "No", 1: "Yes"}
FEATURE_COLUMNS = [
    "Age",
    "Experience",
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "Securities Account",
    "CD Account",
    "Online",
    "CreditCard",
]
MODEL_NAME_MAP = {
    "KNN": KNeighborsClassifier,
    "Logistic Regression": LogisticRegression,
    "Naive Bayes": GaussianNB,
}

# -----------------------------
# STYLE
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #07111f 0%, #0d1b2a 100%);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .hero {
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(135deg, rgba(25,55,109,0.95), rgba(11,87,164,0.82));
        padding: 1.5rem 1.6rem;
        border-radius: 24px;
        margin-bottom: 1rem;
        box-shadow: 0 15px 40px rgba(0,0,0,0.22);
    }
    .hero h1 {
        color: #f8fbff;
        margin: 0 0 0.4rem 0;
        font-size: 2.2rem;
        line-height: 1.1;
    }
    .hero p {
        color: #dbeafe;
        font-size: 1rem;
        margin-bottom: 0.4rem;
    }
    .credit {
        display: inline-block;
        color: #d1fae5;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.25);
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-top: 0.35rem;
    }
    .kpi-card {
        padding: 1rem 1rem 0.8rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        min-height: 130px;
    }
    .kpi-label {
        color: #93c5fd;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        color: #f8fafc;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.15;
    }
    .kpi-sub {
        color: #cbd5e1;
        font-size: 0.88rem;
        margin-top: 0.45rem;
    }
    .section-card {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.025);
        border-radius: 22px;
        padding: 1rem 1rem 0.6rem 1rem;
        margin-bottom: 1rem;
    }
    .tight-note {
        color: #cbd5e1;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 0.85rem;
        border-radius: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# LOADERS
# -----------------------------

def _safe_read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_lift_table(df: pd.DataFrame, prob_col: str = "predicted_probability", target_col: str = "Personal Loan") -> pd.DataFrame:
    out = df[[prob_col, target_col]].copy().sort_values(prob_col, ascending=False).reset_index(drop=True)
    out["decile"] = pd.qcut(np.arange(len(out)), 10, labels=[f"D{i}" for i in range(1, 11)])
    grouped = out.groupby("decile", observed=False).agg(
        customers=(target_col, "size"),
        responders=(target_col, "sum"),
    ).reset_index()
    grouped["response_rate"] = grouped["responders"] / grouped["customers"]
    baseline = out[target_col].mean()
    grouped["lift"] = grouped["response_rate"] / baseline
    grouped["cum_customers"] = grouped["customers"].cumsum()
    grouped["cum_responders"] = grouped["responders"].cumsum()
    grouped["cum_customer_pct"] = grouped["cum_customers"] / grouped["customers"].sum() * 100
    grouped["cum_gain_pct"] = grouped["cum_responders"] / grouped["responders"].sum() * 100
    return grouped


def _compute_roi_curve(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        metrics = compute_roi_table(scores, float(threshold), PROFIT_PER_SUCCESS, CONTACT_COST)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return pd.DataFrame(rows)


def _build_artifacts_if_missing(raw: pd.DataFrame):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    needed = [
        OUTPUT_DIR / "customer_targeting_scores.csv",
        OUTPUT_DIR / "cv_model_comparison.csv",
        OUTPUT_DIR / "test_model_comparison.csv",
        OUTPUT_DIR / "lift_gain_decile_table.csv",
        OUTPUT_DIR / "roi_threshold_table.csv",
        OUTPUT_DIR / "data_audit.json",
    ]
    if all(p.exists() for p in needed):
        return

    df = raw.copy()
    df["Experience"] = df["Experience"].clip(lower=0)
    X = df[FEATURE_COLUMNS].copy()
    y = df["Personal Loan"].astype(int)

    model_specs = {
        "KNN": (
            KNeighborsClassifier(n_neighbors=5, weights="distance", p=1),
            {"smote__k_neighbors": 5, "model__n_neighbors": 5, "model__weights": "distance", "model__p": 1},
        ),
        "Logistic Regression": (
            LogisticRegression(C=1.0, solver="liblinear", penalty="l2", max_iter=5000, random_state=42),
            {"smote__k_neighbors": 5, "model__C": 1.0, "model__solver": "liblinear", "model__penalty": "l2"},
        ),
        "Naive Bayes": (
            GaussianNB(var_smoothing=1e-9),
            {"smote__k_neighbors": 5, "model__var_smoothing": 1e-9},
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    cv_rows, test_rows = [], []
    test_prob_store = {}

    for model_name, (model_obj, best_params) in model_specs.items():
        pipeline = _build_model_by_name(model_name, best_params.copy())

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        oof_pred = (oof >= 0.5).astype(int)

        cv_rows.append({
            "model": model_name,
            "cv_precision_mean": precision_score(y_train, oof_pred, zero_division=0),
            "cv_recall_mean": recall_score(y_train, oof_pred, zero_division=0),
            "cv_f1_mean": f1_score(y_train, oof_pred, zero_division=0),
            "cv_roc_auc_mean": roc_auc_score(y_train, oof),
            "best_params": str(best_params),
        })

        pipeline.fit(X_train, y_train)
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        test_prob_store[model_name] = test_prob

        test_rows.append({
            "model": model_name,
            "test_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_recall": recall_score(y_test, test_pred, zero_division=0),
            "test_f1": f1_score(y_test, test_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, test_prob),
        })

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    best_name = test_df.sort_values("test_f1", ascending=False).iloc[0]["model"]

    scores = X_test.copy()
    scores["Personal Loan"] = y_test.values
    scores["predicted_probability"] = test_prob_store[best_name]
    if "ID" in df.columns:
        scores["ID"] = df.loc[X_test.index, "ID"].values

    lift_df = _compute_lift_table(scores)
    roi_df = _compute_roi_curve(scores)

    audit = {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "negative_experience_count": int((raw["Experience"] < 0).sum()) if "Experience" in raw.columns else 0,
        "total_missing_values": int(raw.isna().sum().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "positive_class_count": int(raw["Personal Loan"].sum()),
        "positive_class_rate": float(raw["Personal Loan"].mean()),
    }

    scores.to_csv(OUTPUT_DIR / "customer_targeting_scores.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "cv_model_comparison.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_model_comparison.csv", index=False)
    lift_df.to_csv(OUTPUT_DIR / "lift_gain_decile_table.csv", index=False)
    roi_df.to_csv(OUTPUT_DIR / "roi_threshold_table.csv", index=False)
    with open(OUTPUT_DIR / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    summary = build_insight_report(raw, test_df, lift_df, roi_df, audit)
    (OUTPUT_DIR / "executive_summary.txt").write_text(summary, encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(DATA_PATH)
    _build_artifacts_if_missing(raw)
    scores = pd.read_csv(OUTPUT_DIR / "customer_targeting_scores.csv")
    cv = pd.read_csv(OUTPUT_DIR / "cv_model_comparison.csv")
    test = pd.read_csv(OUTPUT_DIR / "test_model_comparison.csv")
    lift = pd.read_csv(OUTPUT_DIR / "lift_gain_decile_table.csv")
    roi = pd.read_csv(OUTPUT_DIR / "roi_threshold_table.csv")
    audit = _safe_read_json(OUTPUT_DIR / "data_audit.json")
    return raw, scores, cv, test, lift, roi, audit


def prepare_raw(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "Experience" in df.columns:
        df["Experience"] = df["Experience"].clip(lower=0)
        df["Experience_Clean"] = df["Experience"]
    else:
        df["Experience_Clean"] = np.nan
    if "Education" in df.columns:
        df["Education Label"] = df["Education"].map(EDUCATION_MAP).fillna(df["Education"].astype(str))
    for col in ["Securities Account", "CD Account", "Online", "CreditCard", "Personal Loan"]:
        if col in df.columns:
            df[f"{col} Label"] = df[col].map(YN_MAP)
    return df


def format_money(x: float) -> str:
    return f"${x:,.0f}"


def make_decile_labels(df: pd.DataFrame, prob_col: str = "predicted_probability") -> pd.DataFrame:
    out = df.copy()
    ranks = out[prob_col].rank(method="first", ascending=False)
    out["decile"] = pd.qcut(ranks, 10, labels=[f"D{i}" for i in range(1, 11)])
    return out


def compute_roi_table(scores: pd.DataFrame, threshold: float, profit_per_success: float, contact_cost: float) -> dict:
    pred = (scores["predicted_probability"] >= threshold).astype(int)
    actual = scores["Personal Loan"].astype(int)

    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())

    targeted = int(pred.sum())
    precision = tp / targeted if targeted else 0.0
    recall = tp / actual.sum() if actual.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    revenue = tp * profit_per_success
    campaign_cost = targeted * contact_cost
    net_profit = revenue - campaign_cost

    return {
        "targeted_customers": targeted,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "revenue": revenue,
        "campaign_cost": campaign_cost,
        "net_profit": net_profit,
    }


def model_winner(test: pd.DataFrame, metric: str = "test_f1") -> str:
    row = test.sort_values(metric, ascending=False).iloc[0]
    return str(row["model"])


def _build_model_by_name(name: str, params: dict):
    smote_neighbors = int(params.pop("smote__k_neighbors", 5))

    if name == "KNN":
        model = KNeighborsClassifier(
            n_neighbors=int(params.get("model__n_neighbors", 5)),
            weights=params.get("model__weights", "distance"),
            p=int(params.get("model__p", 2)),
        )
        return ImbPipeline([
            ("smote", SMOTE(random_state=42, k_neighbors=smote_neighbors)),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    if name == "Logistic Regression":
        model = LogisticRegression(
            C=float(params.get("model__C", 1.0)),
            solver=params.get("model__solver", "liblinear"),
            penalty=params.get("model__penalty", "l2"),
            max_iter=5000,
            random_state=42,
        )
        return ImbPipeline([
            ("smote", SMOTE(random_state=42, k_neighbors=smote_neighbors)),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    if name == "Naive Bayes":
        model = GaussianNB(var_smoothing=float(params.get("model__var_smoothing", 1e-9)))
        return ImbPipeline([
            ("smote", SMOTE(random_state=42, k_neighbors=smote_neighbors)),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    raise ValueError(f"Unsupported model name: {name}")


@st.cache_resource(show_spinner=False)
def train_best_model(raw_df: pd.DataFrame, cv_results: pd.DataFrame, test_results: pd.DataFrame):
    model_name = model_winner(test_results, metric="test_f1")
    params_str = cv_results.loc[cv_results["model"] == model_name, "best_params"].iloc[0]
    params = ast.literal_eval(params_str)
    pipeline = _build_model_by_name(model_name, params.copy())

    df = raw_df.copy()
    df["Experience"] = df["Experience"].clip(lower=0)
    X = df[FEATURE_COLUMNS].copy()
    y = df["Personal Loan"].astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    pipeline.fit(X, y)
    return pipeline, model_name, oof_proba


def build_prediction_input(defaults: dict | None = None) -> dict:
    defaults = defaults or {}
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", 18, 100, int(defaults.get("Age", 40)))
        experience = st.number_input("Experience (years)", 0, 60, int(max(0, defaults.get("Experience", 12))))
        income = st.number_input("Income ($000)", 0, 500, int(defaults.get("Income", 60)))
    with col2:
        family = st.selectbox("Family size", [1, 2, 3, 4], index=max(0, min(3, int(defaults.get("Family", 2)) - 1)))
        ccavg = st.number_input("Avg monthly credit card spend ($000)", 0.0, 20.0, float(defaults.get("CCAvg", 1.5)), step=0.1)
        education = st.selectbox("Education", [1, 2, 3], index=max(0, min(2, int(defaults.get("Education", 2)) - 1)), format_func=lambda x: EDUCATION_MAP[x])
    with col3:
        mortgage = st.number_input("Mortgage ($000)", 0, 700, int(defaults.get("Mortgage", 0)))
        securities = st.selectbox("Securities Account", [0, 1], index=int(defaults.get("Securities Account", 0)), format_func=lambda x: YN_MAP[x])
        cd_account = st.selectbox("CD Account", [0, 1], index=int(defaults.get("CD Account", 0)), format_func=lambda x: YN_MAP[x])
    with col4:
        online = st.selectbox("Online Banking", [0, 1], index=int(defaults.get("Online", 1)), format_func=lambda x: YN_MAP[x])
        creditcard = st.selectbox("Bank Credit Card", [0, 1], index=int(defaults.get("CreditCard", 1)), format_func=lambda x: YN_MAP[x])

    return {
        "Age": age,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities Account": securities,
        "CD Account": cd_account,
        "Online": online,
        "CreditCard": creditcard,
    }


@st.cache_data(show_spinner=False)
def compute_global_explainability(_model, raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["Experience"] = df["Experience"].clip(lower=0)
    X = df[FEATURE_COLUMNS].copy()
    y = df["Personal Loan"].astype(int)
    sample_n = min(800, len(X))
    X_sample = X.sample(sample_n, random_state=42)
    y_sample = y.loc[X_sample.index]
    result = permutation_importance(
        _model,
        X_sample,
        y_sample,
        n_repeats=8,
        random_state=42,
        scoring="roc_auc",
    )
    imp = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": result.importances_mean,
        "Std": result.importances_std,
    }).sort_values("Importance", ascending=False)
    return imp


def build_local_explanation(user_input: dict, raw_df: pd.DataFrame) -> pd.DataFrame:
    explain_rows = []
    medians = {
        "Age": float(raw_df["Age"].median()),
        "Experience": float(raw_df["Experience"].clip(lower=0).median()),
        "Income": float(raw_df["Income"].median()),
        "Family": float(raw_df["Family"].median()),
        "CCAvg": float(raw_df["CCAvg"].median()),
        "Education": float(raw_df["Education"].median()),
        "Mortgage": float(raw_df["Mortgage"].median()),
        "Securities Account": float(raw_df["Securities Account"].median()),
        "CD Account": float(raw_df["CD Account"].median()),
        "Online": float(raw_df["Online"].median()),
        "CreditCard": float(raw_df["CreditCard"].median()),
    }

    for feature in FEATURE_COLUMNS:
        value = float(user_input[feature]) if feature != "Education" else int(user_input[feature])
        median_value = medians[feature]
        note = "Near portfolio midpoint"
        direction_score = 0

        if value > median_value:
            note = "Above portfolio midpoint"
            direction_score = 1
        elif value < median_value:
            note = "Below portfolio midpoint"
            direction_score = -1

        display_value = EDUCATION_MAP.get(int(value), int(value)) if feature == "Education" else value
        display_median = EDUCATION_MAP.get(int(round(median_value)), int(round(median_value))) if feature == "Education" else round(median_value, 2)

        explain_rows.append({
            "Feature": feature,
            "Customer Value": display_value,
            "Portfolio Median": display_median,
            "Relative Position": note,
            "Direction Score": direction_score,
        })

    return pd.DataFrame(explain_rows)


def build_insight_report(raw_df: pd.DataFrame, test_results: pd.DataFrame, lift_df: pd.DataFrame, roi_df: pd.DataFrame, audit: dict, prediction_summary: dict | None = None) -> str:
    response_rate = raw_df["Personal Loan"].mean()
    best_row = test_results.sort_values("test_f1", ascending=False).iloc[0]
    best_auc_row = test_results.sort_values("test_roc_auc", ascending=False).iloc[0]
    profit_peak = roi_df.loc[roi_df["net_profit"].idxmax()]
    top_decile = lift_df.iloc[0]
    report = f"""# Thera Bank Marketing Intelligence Report

Created by Powell A. Ndlovu

## Executive Summary
- Customers analysed: {len(raw_df):,}
- Historical loan conversion rate: {response_rate:.1%}
- Best deployment model by test F1: {best_row['model']} (F1={best_row['test_f1']:.3f}, Recall={best_row['test_recall']:.3f}, ROC-AUC={best_row['test_roc_auc']:.3f})
- Highest ROC-AUC model: {best_auc_row['model']} ({best_auc_row['test_roc_auc']:.3f})
- Profit-optimal threshold from analysis: {profit_peak['threshold']:.2f}
- Estimated peak net profit on test scoring file: {format_money(float(profit_peak['net_profit']))}
- First decile cumulative gain: {top_decile['cum_gain_pct']:.1f}% of responders captured in the first 10% of ranked customers
- First decile lift: {top_decile['lift']:.2f}x baseline

## Data Integrity Notes
- Negative Experience values flagged in audit: {audit.get('negative_experience_count', 'n/a')}
- Missing values total: {audit.get('total_missing_values', 'n/a')}
- Duplicate rows: {audit.get('duplicate_rows', 'n/a')}

## Strategic Interpretation
This is not an accuracy contest. The positive class is rare, so the commercially useful model is the one that ranks likely responders well enough to improve targeting efficiency. Lift, gain, recall, and profit matter more than vanity accuracy.

## Recommended Campaign Action
1. Use the ranked customer list rather than untargeted outreach.
2. Start with the top decile when budget is tight.
3. Use the profit threshold near {profit_peak['threshold']:.2f} for aggressive revenue capture.
4. Recalibrate threshold if contact cost or loan value changes.

## Segment Clues from the Data
- Mean income: {raw_df['Income'].mean():.1f}
- Mean CCAvg: {raw_df['CCAvg'].mean():.2f}
- Mean mortgage: {raw_df['Mortgage'].mean():.1f}
- Customers with CD Account: {raw_df['CD Account'].mean():.1%}
- Customers using Online Banking: {raw_df['Online'].mean():.1%}

"""
    if prediction_summary:
        report += f"""## Live Prediction Snapshot
- Predicted probability of acceptance: {prediction_summary['probability']:.1%}
- Risk tier: {prediction_summary['tier']}
- Recommended action: {prediction_summary['action']}
- Expected value from contacting this customer: {format_money(prediction_summary['expected_value'])}

"""
    report += "## Closing Note\nThe strongest use of this tool is not to predict everyone. It is to decide whom to contact first, why, and at what threshold the campaign creates the most return.\n"
    return report


raw, scores, cv_df, test_df, lift_df, roi_df, audit = load_data()
raw = prepare_raw(raw)
scores = make_decile_labels(scores)
best_model, best_model_name, oof_probs = train_best_model(raw, cv_df, test_df)
global_importance_df = compute_global_explainability(best_model, raw)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🏦 Control Center")
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Overview",
        "Prediction Panel",
        "Explainability Panel",
        "Customer Intelligence",
        "Model Command Center",
        "Campaign ROI Simulator",
        "Lift & Gain Strategy",
    ],
)

st.sidebar.markdown("---")
selected_education = st.sidebar.multiselect(
    "Filter by education",
    options=sorted(raw["Education Label"].dropna().unique().tolist()),
    default=sorted(raw["Education Label"].dropna().unique().tolist()),
)
selected_loan = st.sidebar.multiselect(
    "Filter by historical loan response",
    options=["No", "Yes"],
    default=["No", "Yes"],
)
income_range = st.sidebar.slider(
    "Income range",
    min_value=int(raw["Income"].min()),
    max_value=int(raw["Income"].max()),
    value=(int(raw["Income"].min()), int(raw["Income"].max())),
)

filt = raw[
    raw["Education Label"].isin(selected_education)
    & raw["Personal Loan Label"].isin(selected_loan)
    & raw["Income"].between(income_range[0], income_range[1])
].copy()

# -----------------------------
# HERO
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Thera Bank Marketing Intelligence Dashboard</h1>
        <p>From raw customer records to profit-ranked targeting, this dashboard transforms a classroom classification task into an executive decision system.</p>
        <p>Designed to answer the only questions leadership really cares about: <b>who to target, why them, and how much return the campaign creates</b>.</p>
        <div class="credit">Created by Powell A. Ndlovu</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# KPI TOP BAR
# -----------------------------
response_rate = raw["Personal Loan"].mean()
neg_exp_count = int((pd.read_csv(DATA_PATH)["Experience"] < 0).sum()) if "Experience" in pd.read_csv(DATA_PATH).columns else 0
best_f1_model = model_winner(test_df, metric="test_f1")
best_auc_model = model_winner(test_df, metric="test_roc_auc")
profit_peak = roi_df.loc[roi_df["net_profit"].idxmax()]

a, b, c, d = st.columns(4)
with a:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Portfolio Context</div><div class='kpi-value'>{len(raw):,}</div><div class='kpi-sub'>Customers in modelling dataset</div></div>",
        unsafe_allow_html=True,
    )
with b:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Base Conversion Rate</div><div class='kpi-value'>{response_rate:.1%}</div><div class='kpi-sub'>Only {int(raw['Personal Loan'].sum())} accepted historically</div></div>",
        unsafe_allow_html=True,
    )
with c:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Best Model</div><div class='kpi-value'>{best_f1_model}</div><div class='kpi-sub'>Deployment model by holdout F1</div></div>",
        unsafe_allow_html=True,
    )
with d:
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>Peak Profit Threshold</div><div class='kpi-value'>{profit_peak['threshold']:.2f}</div><div class='kpi-sub'>{format_money(profit_peak['net_profit'])} estimated net profit</div></div>",
        unsafe_allow_html=True,
    )

# -----------------------------
# PAGES
# -----------------------------
if page == "Executive Overview":
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("### Executive brief")
        st.info(
            f"This dataset is imbalanced, with only {response_rate:.1%} positive responses. That means a naive accuracy-first model can look good while still being commercially weak. "
            f"The dashboard therefore prioritizes ranking quality, recall, lift, and profit optimization."
        )

        metric_cols = st.columns(3)
        metric_cols[0].metric("Best test F1", best_f1_model, f"{test_df['test_f1'].max():.3f}")
        metric_cols[1].metric("Best ROC-AUC", best_auc_model, f"{test_df['test_roc_auc'].max():.3f}")
        metric_cols[2].metric("Top-decile responder concentration", f"{lift_df.iloc[0]['cum_gain_pct']:.1f}%", "captured in first 10%")

        fig_resp = px.histogram(
            filt,
            x="Income",
            color="Personal Loan Label",
            nbins=35,
            barmode="overlay",
            opacity=0.75,
            title="Income distribution by historical loan response",
        )
        fig_resp.update_layout(height=430)
        st.plotly_chart(fig_resp, use_container_width=True)

        group_cols = st.columns(2)
        with group_cols[0]:
            edu_conv = (
                filt.groupby("Education Label", as_index=False)["Personal Loan"]
                .mean()
                .sort_values("Personal Loan", ascending=False)
            )
            fig_edu = px.bar(
                edu_conv,
                x="Education Label",
                y="Personal Loan",
                title="Conversion rate by education segment",
                text_auto=".1%",
            )
            fig_edu.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_edu, use_container_width=True)

        with group_cols[1]:
            acct_conv = pd.DataFrame(
                {
                    "Feature": ["CD Account", "Securities Account", "Online", "CreditCard"],
                    "Conversion Rate": [
                        filt.groupby("CD Account")["Personal Loan"].mean().get(1, np.nan),
                        filt.groupby("Securities Account")["Personal Loan"].mean().get(1, np.nan),
                        filt.groupby("Online")["Personal Loan"].mean().get(1, np.nan),
                        filt.groupby("CreditCard")["Personal Loan"].mean().get(1, np.nan),
                    ],
                }
            )
            fig_acct = px.bar(acct_conv, x="Feature", y="Conversion Rate", title="Conversion rate among account-holder segments", text_auto=".1%")
            fig_acct.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig_acct, use_container_width=True)

    with right:
        st.markdown("### Decision signal panel")
        best_row = test_df.sort_values("test_f1", ascending=False).iloc[0]
        deployment_msg = (
            f"**Deployment candidate:** {best_row['model']}\n\n"
            f"Test F1 = **{best_row['test_f1']:.3f}**, Recall = **{best_row['test_recall']:.3f}**, ROC-AUC = **{best_row['test_roc_auc']:.3f}**"
        )
        st.success(deployment_msg)

        top_decile = lift_df.iloc[0]
        st.markdown("#### Top-decile targeting impact")
        st.metric("Responders captured in first decile", f"{top_decile['cum_gain_pct']:.1f}%")
        st.metric("Lift in first decile", f"{top_decile['lift']:.2f}x")
        st.metric("Top-decile response rate", f"{top_decile['response_rate']:.1%}")

        summary_path = OUTPUT_DIR / "executive_summary.txt"
        summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else build_insight_report(raw, test_df, lift_df, roi_df, audit)
        st.markdown("#### Analyst memo")
        st.code(summary_text, language="text")

        default_report = build_insight_report(raw, test_df, lift_df, roi_df, audit)
        st.download_button(
            "⬇ Download Data Insights Report",
            data=default_report.encode("utf-8"),
            file_name="thera_bank_data_insights_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

elif page == "Prediction Panel":
    st.markdown("### Prediction panel")
    st.caption("Add a new customer profile, score it with the best model, and turn the output into a campaign decision.")
    st.success(f"Live scoring model: {best_model_name}")

    sample_mode = st.radio("Prefill input", ["Manual entry", "Use a historical customer sample"], horizontal=True)
    defaults = None
    if sample_mode == "Use a historical customer sample":
        sample_id = st.selectbox("Choose sample customer ID", raw["ID"].sort_values().tolist(), index=0)
        defaults = raw.loc[raw["ID"] == sample_id, FEATURE_COLUMNS].iloc[0].to_dict()

    user_input = build_prediction_input(defaults)
    pred_df = pd.DataFrame([user_input])
    probability = float(best_model.predict_proba(pred_df)[0, 1])
    predicted_class = int(probability >= PROFIT_THRESHOLD)
    expected_value = probability * PROFIT_PER_SUCCESS - CONTACT_COST
    percentile_rank = float((oof_probs <= probability).mean() * 100)

    if probability >= 0.50:
        tier = "Immediate Target"
        action = "Prioritize this customer in the next wave."
    elif probability >= PROFIT_THRESHOLD:
        tier = "Profitable Target"
        action = "Contact if the campaign is profit-led and budget allows."
    elif probability >= 0.10:
        tier = "Watchlist"
        action = "Do not lead with this customer. Consider nurturing or bundling offers."
    else:
        tier = "Low Priority"
        action = "Skip in the current campaign and preserve budget."

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted acceptance probability", f"{probability:.1%}")
    m2.metric("Decision at profit threshold", "Target" if predicted_class else "Hold")
    m3.metric("Expected value of contact", format_money(expected_value))
    m4.metric("Probability percentile vs customer base", f"{percentile_rank:.1f}%")

    insight_left, insight_right = st.columns([1.0, 1.0])
    with insight_left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Loan acceptance likelihood"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 10], "color": "#3b4252"},
                    {"range": [10, PROFIT_THRESHOLD * 100], "color": "#1f3a5f"},
                    {"range": [PROFIT_THRESHOLD * 100, 50], "color": "#2c7fb8"},
                    {"range": [50, 100], "color": "#22c55e"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 4}, "value": PROFIT_THRESHOLD * 100},
            },
        ))
        gauge.update_layout(height=340)
        st.plotly_chart(gauge, use_container_width=True)

    with insight_right:
        st.markdown("#### Recommendation")
        st.info(
            f"**Tier:** {tier}\n\n"
            f"**Action:** {action}\n\n"
            f"**Interpretation:** This customer sits above the {PROFIT_THRESHOLD:.2f} profit threshold. "
            if predicted_class else
            f"**Tier:** {tier}\n\n**Action:** {action}\n\n**Interpretation:** This customer sits below the {PROFIT_THRESHOLD:.2f} profit threshold."
        )
        compare_df = pd.DataFrame(
            {
                "Metric": ["Customer probability", "Portfolio mean probability", "Profit threshold"],
                "Value": [probability, float(scores["predicted_probability"].mean()), PROFIT_THRESHOLD],
            }
        )
        compare_df["Value"] = compare_df["Value"].map(lambda x: f"{x:.1%}")
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

    prediction_summary = {
        "probability": probability,
        "tier": tier,
        "action": action,
        "expected_value": expected_value,
    }
    live_report = build_insight_report(raw, test_df, lift_df, roi_df, audit, prediction_summary=prediction_summary)
    st.download_button(
        "⬇ Download Data Insights Report with This Prediction",
        data=live_report.encode("utf-8"),
        file_name="thera_bank_data_insights_report_with_prediction.md",
        mime="text/markdown",
        use_container_width=True,
    )


elif page == "Explainability Panel":
    st.markdown("### Explainability panel")
    st.caption("This panel explains what is driving model decisions globally across the portfolio and locally for a customer profile.")

    left_exp, right_exp = st.columns([1.0, 1.0])

    with left_exp:
        st.markdown("#### Global feature influence")
        fig_imp = px.bar(
            global_importance_df.sort_values("Importance", ascending=True),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Permutation importance on ROC-AUC",
        )
        fig_imp.update_layout(height=420)
        st.plotly_chart(fig_imp, use_container_width=True)
        st.dataframe(global_importance_df, use_container_width=True, hide_index=True)

    with right_exp:
        st.markdown("#### Local customer explanation")
        st.caption("Enter a customer profile to see how it compares with the portfolio midpoint.")
        sample_mode_local = st.radio(
            "Prefill explanation input",
            ["Manual entry", "Use a historical customer sample"],
            horizontal=True,
            key="exp_mode",
        )
        defaults_local = None
        if sample_mode_local == "Use a historical customer sample":
            sample_id_local = st.selectbox(
                "Choose sample customer ID",
                raw["ID"].sort_values().tolist(),
                index=0,
                key="exp_sample",
            )
            defaults_local = raw.loc[raw["ID"] == sample_id_local, FEATURE_COLUMNS].iloc[0].to_dict()

        user_input_local = build_prediction_input(defaults_local)
        local_explain_df = build_local_explanation(user_input_local, raw)

        pred_df_local = pd.DataFrame([user_input_local])
        probability_local = float(best_model.predict_proba(pred_df_local)[0, 1])

        st.metric("Predicted acceptance probability", f"{probability_local:.1%}")

        directional = local_explain_df.copy()
        directional["Influence Label"] = directional["Direction Score"].map({
            1: "Higher than midpoint",
            0: "Near midpoint",
            -1: "Lower than midpoint",
        })
        st.dataframe(
            directional[["Feature", "Customer Value", "Portfolio Median", "Relative Position", "Influence Label"]],
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        strongest_high = directional[directional["Direction Score"] == 1]["Feature"].head(3).tolist()
        strongest_low = directional[directional["Direction Score"] == -1]["Feature"].head(3).tolist()

        upward_text = ", ".join(strongest_high) if strongest_high else "None flagged"
        downward_text = ", ".join(strongest_low) if strongest_low else "None flagged"

        st.info(
            f"Potential upward signals: {upward_text}.\n\nPotential downward signals: {downward_text}."
        )

elif page == "Customer Intelligence":
    st.markdown("### Customer intelligence explorer")
    st.caption("Use this page to identify who looks commercially attractive before pushing them into a campaign.")

    score_view = scores.copy()
    score_view["Education Label"] = score_view["Education"].map(EDUCATION_MAP)
    score_view["Priority Tier"] = pd.cut(
        score_view["predicted_probability"],
        bins=[-0.001, 0.10, 0.25, 0.50, 1.0],
        labels=["Low", "Watchlist", "High", "Immediate Target"],
    )
    score_view["CD Account Label"] = score_view["CD Account"].map(YN_MAP)
    score_view["Online Label"] = score_view["Online"].map(YN_MAP)

    c1, c2, c3 = st.columns([0.9, 0.9, 1.2])
    with c1:
        prob_cut = st.slider("Minimum predicted probability", 0.0, 1.0, 0.16, 0.01)
    with c2:
        priority_filter = st.multiselect(
            "Priority tier",
            options=score_view["Priority Tier"].dropna().unique().tolist(),
            default=score_view["Priority Tier"].dropna().unique().tolist(),
        )
    with c3:
        search_id = st.text_input("Find specific customer ID", placeholder="e.g. 318")

    filtered_scores = score_view[
        (score_view["predicted_probability"] >= prob_cut)
        & (score_view["Priority Tier"].isin(priority_filter))
    ].copy()

    if search_id.strip():
        try:
            filtered_scores = filtered_scores[filtered_scores["ID"] == int(search_id.strip())]
        except ValueError:
            st.warning("Customer ID must be numeric.")

    colx, coly = st.columns([1.1, 0.9])
    with colx:
        scatter = px.scatter(
            filtered_scores,
            x="Income",
            y="CCAvg",
            size="Mortgage",
            color="predicted_probability",
            hover_data=["ID", "Age", "Family", "Education Label", "CD Account Label", "Online Label"],
            title="Targetable customers: income vs credit-card spend",
        )
        scatter.update_layout(height=500)
        st.plotly_chart(scatter, use_container_width=True)

    with coly:
        tier_counts = filtered_scores["Priority Tier"].value_counts(dropna=False).rename_axis("Priority Tier").reset_index(name="Customers")
        fig_tier = px.funnel(tier_counts, y="Priority Tier", x="Customers", title="Priority funnel")
        st.plotly_chart(fig_tier, use_container_width=True)

        seg = (
            filtered_scores.groupby(["Education Label", "CD Account Label"], as_index=False)
            .agg(customers=("ID", "count"), avg_probability=("predicted_probability", "mean"), avg_income=("Income", "mean"))
            .sort_values("avg_probability", ascending=False)
        )
        st.dataframe(seg, use_container_width=True, hide_index=True)

    st.markdown("#### Priority list")
    display_cols = [
        "ID", "Age", "Income", "Family", "CCAvg", "Education Label", "Mortgage",
        "CD Account Label", "Online Label", "predicted_probability", "Priority Tier",
    ]
    st.dataframe(
        filtered_scores[display_cols].sort_values("predicted_probability", ascending=False),
        use_container_width=True,
        height=420,
        hide_index=True,
    )

elif page == "Model Command Center":
    st.markdown("### Model command center")
    st.caption("This section exists to show rigor, not just results. It highlights validation discipline, tuning outcomes, and deployment tradeoffs.")

    left, right = st.columns([0.95, 1.05])
    with left:
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(name="CV F1", x=cv_df["model"], y=cv_df["cv_f1_mean"]))
        fig_cv.add_trace(go.Bar(name="CV ROC-AUC", x=cv_df["model"], y=cv_df["cv_roc_auc_mean"]))
        fig_cv.update_layout(barmode="group", title="Cross-validation comparison", height=420)
        st.plotly_chart(fig_cv, use_container_width=True)

        fig_test = go.Figure()
        for metric in ["test_precision", "test_recall", "test_f1"]:
            fig_test.add_trace(go.Bar(name=metric.replace("test_", "").title(), x=test_df["model"], y=test_df[metric]))
        fig_test.update_layout(barmode="group", title="Holdout test tradeoffs", height=420)
        st.plotly_chart(fig_test, use_container_width=True)

    with right:
        st.markdown("#### Best-parameter ledger")
        params_df = cv_df[["model", "best_params"]].copy()
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        st.markdown("#### Performance table")
        st.dataframe(test_df.sort_values("test_f1", ascending=False), use_container_width=True, hide_index=True)

        st.warning(
            "SMOTE helps the model see the minority class during training, but it does not change reality. It must stay inside cross-validation folds to avoid leakage."
        )

    st.markdown("#### Evidence gallery")
    img1, img2 = st.columns(2)
    with img1:
        roc_path = OUTPUT_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC curve comparison")
    with img2:
        pr_path = OUTPUT_DIR / "precision_recall_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="Precision-recall view for imbalanced classification")

elif page == "Campaign ROI Simulator":
    st.markdown("### Campaign ROI simulator")
    st.caption("This is where the model becomes a business tool. Move the threshold and unit economics, then watch the campaign profile change.")

    s1, s2, s3 = st.columns(3)
    with s1:
        threshold = st.slider("Scoring threshold", 0.01, 0.99, float(PROFIT_THRESHOLD), 0.01)
    with s2:
        profit_per_success = st.number_input("Profit per successful loan ($)", min_value=100, max_value=10000, value=PROFIT_PER_SUCCESS, step=50)
    with s3:
        contact_cost = st.number_input("Campaign contact cost per customer ($)", min_value=1, max_value=500, value=CONTACT_COST, step=1)

    roi_metrics = compute_roi_table(scores, threshold, profit_per_success, contact_cost)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Targeted customers", f"{roi_metrics['targeted_customers']:,}")
    m2.metric("Estimated net profit", format_money(roi_metrics["net_profit"]))
    m3.metric("Precision", f"{roi_metrics['precision']:.1%}")
    m4.metric("Recall", f"{roi_metrics['recall']:.1%}")

    breakdown_col, curve_col = st.columns([0.9, 1.1])
    with breakdown_col:
        conf_df = pd.DataFrame(
            {
                "Outcome": ["True Positive", "False Positive", "False Negative", "True Negative"],
                "Count": [roi_metrics["tp"], roi_metrics["fp"], roi_metrics["fn"], roi_metrics["tn"]],
            }
        )
        fig_conf = px.treemap(conf_df, path=["Outcome"], values="Count", title="Campaign outcome mix")
        st.plotly_chart(fig_conf, use_container_width=True)

        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": ["Revenue", "Campaign Cost", "Net Profit", "F1 Score"],
                    "Value": [
                        format_money(roi_metrics["revenue"]),
                        format_money(roi_metrics["campaign_cost"]),
                        format_money(roi_metrics["net_profit"]),
                        f"{roi_metrics['f1']:.3f}",
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with curve_col:
        dynamic_roi = roi_df.copy()
        dynamic_roi["revenue"] = dynamic_roi["tp"] * profit_per_success
        dynamic_roi["campaign_cost"] = dynamic_roi["targeted_customers"] * contact_cost
        dynamic_roi["net_profit"] = dynamic_roi["revenue"] - dynamic_roi["campaign_cost"]
        fig_roi = px.line(dynamic_roi, x="threshold", y="net_profit", markers=True, title="Threshold vs estimated net profit")
        fig_roi.add_vline(x=threshold, line_dash="dash", annotation_text=f"Current {threshold:.2f}")
        st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown("#### Recommended action frame")
    if threshold <= 0.20:
        st.success("Aggressive targeting mode: this favors recall and revenue capture, accepting higher false-positive cost.")
    elif threshold <= 0.50:
        st.info("Balanced targeting mode: this moderates spend while still capturing a sizeable share of likely responders.")
    else:
        st.warning("Conservative targeting mode: this protects budget, but you will miss a larger share of real responders.")

elif page == "Lift & Gain Strategy":
    st.markdown("### Lift & gain strategy")
    st.caption("These charts answer whether the model actually ranks customers well enough to improve marketing efficiency.")

    left, right = st.columns(2)
    with left:
        fig_lift = px.line(lift_df, x="decile", y="lift", markers=True, title="Lift by decile")
        fig_lift.add_hline(y=1.0, line_dash="dot")
        st.plotly_chart(fig_lift, use_container_width=True)
        lift_path = OUTPUT_DIR / "lift_chart.png"
        if lift_path.exists():
            st.image(str(lift_path), caption="Saved lift chart artifact")

    with right:
        fig_gain = px.line(lift_df, x="cum_customer_pct", y="cum_gain_pct", markers=True, title="Cumulative gain curve")
        fig_gain.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Baseline", line=dict(dash="dash")))
        st.plotly_chart(fig_gain, use_container_width=True)
        gain_path = OUTPUT_DIR / "gain_chart.png"
        if gain_path.exists():
            st.image(str(gain_path), caption="Saved cumulative gain artifact")

    st.markdown("#### Decile strategy table")
    strategy = lift_df[["decile", "customers", "responders", "response_rate", "lift", "cum_gain_pct"]].copy()
    strategy["response_rate"] = strategy["response_rate"].map(lambda x: f"{x:.1%}")
    strategy["cum_gain_pct"] = strategy["cum_gain_pct"].map(lambda x: f"{x:.1f}%")
    strategy["lift"] = strategy["lift"].map(lambda x: f"{x:.2f}x")
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    st.info(
        f"The first decile alone captures {lift_df.iloc[0]['cum_gain_pct']:.1f}% of responders. If leadership only funds a small campaign, the model says start there."
    )
