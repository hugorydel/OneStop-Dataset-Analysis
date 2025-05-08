# ------------------------------------------------------------
# One-Stop pilot: integrated multimodal strategies
# ------------------------------------------------------------
import random
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# ---------- 1. File path & columns ----------
IA_FILE = "ia_Paragraph_ordinary.csv"  # path to the OneStop IA file
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

# ---------- 2. Aggregate & feature engineering ----------
print("Aggregating IA file → features …")
chunks = pd.read_csv(IA_FILE, usecols=use_cols, sep=",", chunksize=500_000)
records = []
for chunk in chunks:
    # label
    chunk["correct"] = (
        chunk["selected_answer_position"] == chunk["correct_answer_position"]
    ).astype(int)
    chunk.drop(columns=answer_cols, inplace=True)
    # numeric coercion
    chunk[eye_cols + text_cols] = chunk[eye_cols + text_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    # log-transform skewed
    for col in [
        "IA_DWELL_TIME",
        "IA_FIRST_RUN_DWELL_TIME",
        "IA_REGRESSION_PATH_DURATION",
        "IA_FIRST_FIXATION_DURATION",
    ]:
        chunk[col] = np.log1p(chunk[col])
    # rate features
    chunk["words_in_para"] = chunk.groupby(group_cols)[
        "word_length_no_punctuation"
    ].transform("count")
    chunk["regressions_per_word"] = (
        chunk["IA_REGRESSION_IN_COUNT"] + chunk["IA_REGRESSION_OUT_COUNT"]
    ) / chunk["words_in_para"]
    chunk["fixations_per_word"] = chunk["IA_FIXATION_COUNT"] / chunk["words_in_para"]
    # aggregate
    cols_all = (
        eye_cols
        + ["regressions_per_word", "fixations_per_word"]
        + text_cols
        + ["correct"]
    )
    grp = chunk.groupby(group_cols)[cols_all].median().reset_index()
    records.append(grp)
df = pd.concat(records, ignore_index=True)
print(f"Data shape: {df.shape}")

# ---------- 3. Interaction features ----------
for txt in ["subtlex_frequency", "gpt2_surprisal"]:
    for eye in ["IA_FIRST_FIXATION_DURATION", "IA_REGRESSION_PATH_DURATION"]:
        df[f"{eye}_x_{txt}"] = df[eye] * df[txt]
interaction_cols = [c for c in df.columns if "_x_" in c]

# ---------- 4. Prepare X, y, groups ----------
eye_extra = ["regressions_per_word", "fixations_per_word"]
X_eye = df[eye_cols + eye_extra].replace([np.inf, -np.inf], np.nan).fillna(0)
X_text = df[text_cols].fillna(0)
X_both = (
    df[eye_cols + eye_extra + text_cols + interaction_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)
y = df["correct"].round().astype(int)
grp = df["participant_id"]

# ---------- 5. CV splitter ----------
cv = GroupKFold(n_splits=5)


# ---------- 6. Unimodal logistic baselines ----------
def quick_auc(Xsub):
    aucs = []
    for tr, te in cv.split(Xsub, y, grp):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xsub.iloc[tr])
        Xte = scaler.transform(Xsub.iloc[te])
        clf = LogisticRegression(max_iter=5000, solver="lbfgs")
        clf.fit(Xtr, y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], clf.predict_proba(Xte)[:, 1]))
    return np.mean(aucs)


print("Eye-only  AUC:", round(quick_auc(X_eye), 3))
print("Text-only AUC:", round(quick_auc(X_text), 3))

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
    verbose=-1,
)
aucs, feat_imp = [], np.zeros(X_both.shape[1])
print("Running LightGBM CV …")
for tr, te in cv.split(X_both, y, grp):
    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(X_both.iloc[tr], y.iloc[tr])
    proba = clf.predict_proba(X_both.iloc[te])[:, 1]
    aucs.append(roc_auc_score(y.iloc[te], proba))
    feat_imp += clf.booster_.feature_importance(importance_type="gain")
print("Multimodal AUC:", round(np.mean(aucs), 3))

# ---------- 8. Stacking ensemble ----------
oof = np.zeros((len(X_both), 3))
for i, (cols, Xsub) in enumerate(
    [
        (X_eye.columns, X_eye),
        (X_text.columns, X_text),
        (X_both.columns, X_both),
    ]
):
    for tr, te in cv.split(Xsub, y, grp):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xsub.iloc[tr], y.iloc[tr])
        oof[te, i] = m.predict_proba(Xsub.iloc[te])[:, 1]
meta = LogisticRegression(max_iter=1000).fit(oof, y)
print("Stacked AUC:", round(roc_auc_score(y, meta.predict_proba(oof)[:, 1]), 3))

# ---------- 9. Hyperparameter random search ----------
param_grid = {
    "num_leaves": [15, 31, 63, 127],
    "max_depth": [-1, 5, 10, 15],
    "learning_rate": [0.01, 0.03, 0.05],
    "feature_fraction": [0.6, 0.8, 1.0],
}
best_auc, best_p = 0, None
trials = random.sample(list(product(*param_grid.values())), 30)
for nl, md, lr, ff in trials:
    p = lgb_params.copy()
    p.update(num_leaves=nl, max_depth=md, learning_rate=lr, feature_fraction=ff)
    tmp = []
    for tr, te in cv.split(X_both, y, grp):
        m = lgb.LGBMClassifier(**p)
        m.fit(X_both.iloc[tr], y.iloc[tr])
        tmp.append(roc_auc_score(y.iloc[te], m.predict_proba(X_both.iloc[te])[:, 1]))
    auc = np.mean(tmp)
    if auc > best_auc:
        best_auc, best_p = auc, p.copy()
print("Best tuned AUC:", round(best_auc, 3), "with params:", best_p)

# ---------- 10. Feature importances ----------
imp = (
    pd.Series(feat_imp / cv.get_n_splits(), index=X_both.columns)
    .sort_values(ascending=False)
    .head(10)
)
print("\nTop-10 features:")
print(imp.round(4))
