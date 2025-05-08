# ------------------------------------------------------------
# One-Stop pilot: eye + text → comprehension (LightGBM)
# ------------------------------------------------------------
import random
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ---------- 1. File path ----------
IA_FILE = "ia_Paragraph_ordinary.csv"  # path to the OneStop IA file

# ---------- 2. Columns to keep ----------
eye_cols = [
    "IA_FIRST_FIXATION_DURATION",
    "IA_SECOND_FIXATION_DURATION",
    "IA_THIRD_FIXATION_DURATION",
    "IA_LAST_FIXATION_DURATION",
    "IA_REGRESSION_IN_COUNT",
    "IA_REGRESSION_OUT_COUNT",
    "IA_FIRST_SACCADE_AMPLITUDE",
    "IA_LAST_SACCADE_AMPLITUDE",
    "IA_FIXATION_COUNT",
    "IA_DWELL_TIME",
    "IA_FIRST_RUN_DWELL_TIME",
    "IA_SKIP",
    "IA_AVERAGE_FIX_PUPIL_SIZE",
    "IA_MAX_FIX_PUPIL_SIZE",
    "IA_SELECTIVE_REGRESSION_PATH_DURATION",
    "IA_REGRESSION_PATH_DURATION",
]
text_cols = ["word_length_no_punctuation", "subtlex_frequency", "gpt2_surprisal"]
answer_cols = ["selected_answer_position", "correct_answer_position"]
group_cols = ["participant_id", "trial_index"]

use_cols = group_cols + eye_cols + text_cols + answer_cols

# ---------- 3. Aggregate IA file & build features & label ----------
print("Aggregating IA file → paragraph-level features …")
chunks = pd.read_csv(IA_FILE, usecols=use_cols, sep=",", chunksize=500_000)
records = []

for chunk in chunks:
    # 3.1 build binary label & drop raw answer columns
    chunk["correct"] = (
        chunk["selected_answer_position"] == chunk["correct_answer_position"]
    ).astype(int)
    chunk = chunk.drop(columns=answer_cols)

    # 3.2 coerce to numeric, drop non-numeric
    cols_to_num = eye_cols + text_cols
    chunk[cols_to_num] = chunk[cols_to_num].apply(pd.to_numeric, errors="coerce")

    # 3.3 log-transform skewed features
    for col in [
        "IA_DWELL_TIME",
        "IA_FIRST_RUN_DWELL_TIME",
        "IA_REGRESSION_PATH_DURATION",
        "IA_FIRST_FIXATION_DURATION",
    ]:
        chunk[col] = np.log1p(chunk[col])

    # 3.4 compute rate-normalized gaze features
    chunk["words_in_para"] = chunk.groupby(group_cols)[
        "word_length_no_punctuation"
    ].transform("count")
    chunk["regressions_per_word"] = (
        chunk["IA_REGRESSION_IN_COUNT"] + chunk["IA_REGRESSION_OUT_COUNT"]
    ) / chunk["words_in_para"]
    chunk["fixations_per_word"] = chunk["IA_FIXATION_COUNT"] / chunk["words_in_para"]

    # add new rate features
    eye_cols_extra = ["regressions_per_word", "fixations_per_word"]
    cols_all = eye_cols + eye_cols_extra + text_cols + ["correct"]
    # 3.5 aggregate to participant × paragraph
    grp = chunk.groupby(group_cols)[cols_all].median().reset_index()
    records.append(grp)

df = pd.concat(records, ignore_index=True)
print(f"Final table shape: {df.shape}")

# ---------- 4. Prepare X, y, groups ----------
# combine original eye/text with extra rate features
X = (
    df[eye_cols + eye_cols_extra + text_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)
y = df["correct"].round().astype(int)
grp = df["participant_id"]

# ---------- 5. Participant-wise 5-fold CV setup ----------
cv = GroupKFold(n_splits=5)


# ---------- 6. Logistic baselines for unimodal AUC ----------
def quick_auc(Xsub):
    a = []
    for tr, te in cv.split(Xsub, y, grp):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xsub.iloc[tr])
        Xte = scaler.transform(Xsub.iloc[te])
        clf = LogisticRegression(max_iter=5000, solver="lbfgs")
        clf.fit(Xtr, y.iloc[tr])
        a.append(roc_auc_score(y.iloc[te], clf.predict_proba(Xte)[:, 1]))
    return np.mean(a)


eye_only_auc = quick_auc(X[eye_cols + eye_cols_extra])
text_only_auc = quick_auc(X[text_cols])

print("Eye-only  :", round(eye_only_auc, 3))
print("Text-only :", round(text_only_auc, 3))

# ---------- 7. LightGBM multimodal CV ----------
lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=600,
    n_jobs=-1,
)

aucs = []
feat_imp = np.zeros(X.shape[1])

print("Running LightGBM CV …")
for tr, te in cv.split(X, y, grp):
    train_set = lgb.Dataset(X.iloc[tr], y.iloc[tr])
    valid_set = lgb.Dataset(X.iloc[te], y.iloc[te], reference=train_set)
    gbm = lgb.train(lgb_params, train_set, valid_sets=[valid_set], verbose_eval=False)
    proba = gbm.predict(X.iloc[te])
    aucs.append(roc_auc_score(y.iloc[te], proba))
    feat_imp += gbm.feature_importance(importance_type="gain")

print(f"\nMean multimodal AUC across folds: {np.mean(aucs):.3f}")

# ---------- 8. Top-10 feature importances ----------
importance = (
    pd.Series(feat_imp / cv.get_n_splits(), index=X.columns)
    .sort_values(ascending=False)
    .head(10)
)
print("\nTop-10 multimodal features:")
print(importance.round(4))
