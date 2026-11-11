"""
Customer Churn Prediction - training pipeline.

Dataset: IBM Telco Customer Churn (Kaggle)
Download "WA_Fn-UseC_-Telco-Customer-Churn.csv" and save it as data/telco.csv

Run:  python train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# ------------------------------------------------------------------ 1. LOAD & CLEAN
df = pd.read_csv("data/telco.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID"])
print(f"Rows: {len(df)} | Churn rate: {df['Churn'].mean():.1%}")

# ------------------------------------------------------------------ 2. FEATURE ENGINEERING
df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                           labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]).astype(str)

X, y = df.drop(columns="Churn"), df["Churn"]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ------------------------------------------------------------------ 3. COMPARE MODELS (5-fold CV)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_results[name] = round(scores.mean(), 4)
    print(f"{name:22s} CV ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")

# ------------------------------------------------------------------ 4. TUNE BEST TREE MODEL
param_dist = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.02, 0.05, 0.1],
    "model__max_depth": [2, 3, 4],
    "model__subsample": [0.7, 0.85, 1.0],
}
search = RandomizedSearchCV(
    Pipeline([("pre", pre), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
    param_dist, n_iter=15, scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
search.fit(X_train, y_train)
best = search.best_estimator_
print("Best params:", search.best_params_, "| CV ROC-AUC:", round(search.best_score_, 4))

# ------------------------------------------------------------------ 5. THRESHOLD TUNING
# Missing a churner (false negative) costs more than a wasted retention offer,
# so we choose the threshold that maximises F2 (recall weighted higher than precision).
proba = best.predict_proba(X_test)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, proba)
f2 = (5 * prec * rec) / (4 * prec + rec + 1e-9)
best_thr = float(thr[np.argmax(f2[:-1])])
pred_default = (proba >= 0.5).astype(int)
pred_tuned = (proba >= best_thr).astype(int)

def metrics(pred):
    return {"precision": round(precision_score(y_test, pred), 3),
            "recall": round(recall_score(y_test, pred), 3),
            "f1": round(f1_score(y_test, pred), 3)}

results = {
    "cv_roc_auc": cv_results,
    "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
    "threshold_0.5": metrics(pred_default),
    f"threshold_{best_thr:.2f}": metrics(pred_tuned),
    "best_params": search.best_params_,
    "chosen_threshold": round(best_thr, 3),
}
print(json.dumps(results, indent=2, default=str))
print(classification_report(y_test, pred_tuned, target_names=["Stay", "Churn"]))
json.dump(results, open(OUT / "metrics.json", "w"), indent=2, default=str)

# ------------------------------------------------------------------ 6. PLOTS
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ConfusionMatrixDisplay.from_predictions(y_test, pred_tuned, display_labels=["Stay", "Churn"],
                                        cmap="Blues", ax=ax[0])
ax[0].set_title(f"Confusion Matrix (threshold={best_thr:.2f})")
RocCurveDisplay.from_predictions(y_test, proba, ax=ax[1])
ax[1].set_title("ROC Curve")
plt.tight_layout(); plt.savefig(OUT / "evaluation.png", dpi=150); plt.close()

perm = permutation_importance(best, X_test, y_test, scoring="roc_auc",
                              n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values().tail(10)
plt.figure(figsize=(7, 4.5))
sns.barplot(x=imp.values, y=imp.index, color="#2a6fdb")
plt.title("Top 10 Churn Drivers (Permutation Importance)")
plt.xlabel("Drop in ROC-AUC when feature is shuffled")
plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()
print("Top drivers:", list(imp.index[::-1][:5]))

# ------------------------------------------------------------------ 7. SAVE MODEL
joblib.dump({"model": best, "threshold": best_thr}, OUT / "churn_model.joblib")
print("Saved model and plots to outputs/")
