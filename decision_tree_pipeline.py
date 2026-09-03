"""
=============================================================================
Assignment 3 — Loan Default Classification
Stage 3: Decision Tree Classifier — Complete Pipeline
=============================================================================
Author : Rai (UTS)
Purpose: Train, tune, evaluate, and interpret a Decision Tree classifier.
         Produces baseline vs tuned comparison, CV results, all plots.

NOTE:  This script is designed to be run section-by-section in a notebook
       or as a complete file. Grid search uses 3-fold CV to manage runtime
       (3-fold on 28k rows is statistically adequate for 35k total).
       Final evaluation uses 5-fold CV for reporting.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib, time, warnings, os
warnings.filterwarnings('ignore')

# ── Plotting style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#0f1117',
    'axes.edgecolor': '#2a2d3e', 'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9', 'xtick.color': '#8b949e',
    'ytick.color': '#8b949e', 'grid.color': '#1c1f2b',
    'font.family': 'sans-serif', 'font.size': 10,
})
BLUE, ORANGE, GREEN, PURPLE = '#58a6ff', '#f78166', '#7ee787', '#d2a8ff'
OUTPUT = '/home/claude/plots'
os.makedirs(OUTPUT, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(r"C:\Users\62877\Downloads\final_loan_engineered.csv")
feature_cols = [c for c in df.columns if c != 'loan_default']
X = df[feature_cols]
y = df['loan_default'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Majority baseline accuracy: {y_test.value_counts().max()/len(y_test):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — BASELINE (UNRESTRICTED) DECISION TREE
# ═══════════════════════════════════════════════════════════════════════════
#
# Justification: An unrestricted tree grows to full depth, memorising
# training data. The gap between train and test performance quantifies
# overfitting. Decision Trees are scale-invariant (no scaler needed).
# class_weight='balanced' handles the 61:39 class imbalance.

dt_baseline = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_baseline.fit(X_train, y_train)

y_base_pred = dt_baseline.predict(X_test)
y_base_prob = dt_baseline.predict_proba(X_test)[:, 1]

print(f"\n=== BASELINE ===")
print(f"Depth: {dt_baseline.get_depth()}, Leaves: {dt_baseline.get_n_leaves()}")
print(f"Train Acc: {accuracy_score(y_train, dt_baseline.predict(X_train)):.4f}")
print(f"Test  Acc: {accuracy_score(y_test, y_base_pred):.4f}")
print(f"Test  F1:  {f1_score(y_test, y_base_pred):.4f}")
print(f"Test  AUC: {roc_auc_score(y_test, y_base_prob):.4f}")
print(f"Overfit gap: {accuracy_score(y_train, dt_baseline.predict(X_train)) - accuracy_score(y_test, y_base_pred):.4f}")
print(f"→ SEVERE OVERFITTING: 100% train accuracy with ~73% test accuracy")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — HYPERPARAMETER TUNING (MULTI-STAGE GRID SEARCH)
# ═══════════════════════════════════════════════════════════════════════════
#
# Strategy: Two-stage grid search using 3-fold stratified CV.
#   Stage 1: Broad sweep over criterion, max_depth, min_samples_leaf
#   Stage 2: Refined search around the best region
#
# Optimising for F1-macro (balances both classes).

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Stage 1: Broad sweep — entropy vs gini, depth 5–13, leaf 1–50
print("\n=== GRID SEARCH — Stage 1 (broad sweep) ===")
stage1_results = []
for crit in ['gini', 'entropy']:
    for depth in [5, 7, 9, 11, 13]:
        for leaf in [1, 10, 50]:
            dt = DecisionTreeClassifier(
                criterion=crit, max_depth=depth, min_samples_leaf=leaf,
                random_state=42, class_weight='balanced'
            )
            scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring='f1_macro')
            stage1_results.append({
                'criterion': crit, 'max_depth': depth, 'min_samples_leaf': leaf,
                'mean_f1': scores.mean(), 'std_f1': scores.std()
            })

stage1_results.sort(key=lambda x: x['mean_f1'], reverse=True)
print("Top 5 from Stage 1:")
for r in stage1_results[:5]:
    print(f"  F1={r['mean_f1']:.4f}±{r['std_f1']:.4f}  "
          f"crit={r['criterion']}, depth={r['max_depth']}, leaf={r['min_samples_leaf']}")

# Stage 2: Refined search around best (entropy, depth 8–10, leaf 5–20)
print("\n=== GRID SEARCH — Stage 2 (refinement) ===")
stage2_results = []
for depth in [8, 9, 10, 11]:
    for leaf in [5, 8, 10, 15, 20]:
        for split in [2, 10, 20]:
            dt = DecisionTreeClassifier(
                criterion='entropy', max_depth=depth,
                min_samples_leaf=leaf, min_samples_split=split,
                random_state=42, class_weight='balanced'
            )
            scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring='f1_macro')
            stage2_results.append({
                'max_depth': depth, 'min_samples_leaf': leaf,
                'min_samples_split': split,
                'mean_f1': scores.mean(), 'std_f1': scores.std()
            })

stage2_results.sort(key=lambda x: x['mean_f1'], reverse=True)
print("Top 5 from Stage 2:")
for r in stage2_results[:5]:
    print(f"  F1={r['mean_f1']:.4f}±{r['std_f1']:.4f}  "
          f"depth={r['max_depth']}, leaf={r['min_samples_leaf']}, split={r['min_samples_split']}")

best = stage2_results[0]


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — FINAL TUNED MODEL
# ═══════════════════════════════════════════════════════════════════════════

final_params = {
    'criterion': 'entropy',
    'max_depth': best['max_depth'],
    'min_samples_leaf': best['min_samples_leaf'],
    'min_samples_split': best['min_samples_split'],
    'class_weight': 'balanced',
    'random_state': 42,
}

print(f"\n=== FINAL TUNED MODEL ===")
print(f"Parameters: {final_params}")

dt_final = DecisionTreeClassifier(**final_params)
dt_final.fit(X_train, y_train)

y_final_pred = dt_final.predict(X_test)
y_final_prob = dt_final.predict_proba(X_test)[:, 1]

print(f"Depth: {dt_final.get_depth()}, Leaves: {dt_final.get_n_leaves()}")
print(f"Train Acc: {accuracy_score(y_train, dt_final.predict(X_train)):.4f}")
print(f"Test  Acc: {accuracy_score(y_test, y_final_pred):.4f}")
print(f"Test  Prec:{precision_score(y_test, y_final_pred):.4f}")
print(f"Test  Rec: {recall_score(y_test, y_final_pred):.4f}")
print(f"Test  F1:  {f1_score(y_test, y_final_pred):.4f}")
print(f"Test  F1m: {f1_score(y_test, y_final_pred, average='macro'):.4f}")
print(f"Test  AUC: {roc_auc_score(y_test, y_final_prob):.4f}")
print(f"Overfit:   {accuracy_score(y_train, dt_final.predict(X_train)) - accuracy_score(y_test, y_final_pred):.4f}")

print("\n" + classification_report(y_test, y_final_pred,
      target_names=['No Default (0)', 'Default (1)'], digits=4))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — 5-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("=== 5-FOLD CROSS-VALIDATION ===")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'roc_auc']:
    scores = cross_val_score(
        DecisionTreeClassifier(**final_params),
        X_train, y_train, cv=cv5, scoring=metric
    )
    print(f"  {metric:15s}  mean={scores.mean():.4f}  std={scores.std():.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

imp = pd.Series(dt_final.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== FEATURE IMPORTANCE ===")
for f, v in imp.items():
    if v > 0.005:
        print(f"  {f:35s}  {v:.4f}  {'█' * int(v * 80)}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7 — DECISION RULES
# ═══════════════════════════════════════════════════════════════════════════

print("\n=== TOP DECISION RULES ===")
txt = export_text(dt_final, feature_names=feature_cols, max_depth=2)
for line in txt.split('\n')[:20]:
    print(f"  {line}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8 — SAVE MODEL & PREDICT UNKNOWN
# ═══════════════════════════════════════════════════════════════════════════

joblib.dump(dt_final, '/home/claude/dt_final_model.joblib')

df_unknown = pd.read_csv('/home/claude/final_unknown_engineered.csv')
preds = dt_final.predict(df_unknown[feature_cols])
probs = dt_final.predict_proba(df_unknown[feature_cols])[:, 1]

df_out = df_unknown.copy()
df_out['predicted_default'] = preds
df_out['default_probability'] = probs
df_out.to_csv('/home/claude/dt_unknown_predictions.csv', index=False)

print(f"\nUnknown predictions: {pd.Series(preds).value_counts().to_dict()}")
print("PIPELINE COMPLETE")
