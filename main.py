# ------------------------------------------------------------
# One-Stop pilot: eye + text → comprehension (Random Forest)
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler  # ← NEW


# ----------------- helper: logistic-baseline AUC -----------------
def quick_auc(Xsub):
    """
    Participant-wise 5-fold CV using Logistic Regression.
    Scales features inside each fold to avoid convergence warnings.
    """
    a = []
    for tr, te in cv.split(Xsub, y, grp):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xsub.iloc[tr])
        Xte = scaler.transform(Xsub.iloc[te])

        clf = LogisticRegression(max_iter=5000, solver="lbfgs")  # ↑ iterations
        clf.fit(Xtr, y.iloc[tr])
        a.append(roc_auc_score(y.iloc[te], clf.predict_proba(Xte)[:, 1]))
    return np.mean(a)


# ---------- 1.  File path ----------
IA_FILE = "ia_Paragraph_ordinary.csv"  # 5-GB TSV/CSV

# ---------- 2.  Columns to keep ----------
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

# ---------- 3.  Aggregate IA file & build label ----------
print("Aggregating IA file → paragraph-level means …")
chunks = pd.read_csv(IA_FILE, usecols=use_cols, sep=",", chunksize=500_000)
records = []

for chunk in chunks:
    chunk["correct"] = (
        chunk["selected_answer_position"] == chunk["correct_answer_position"]
    ).astype(int)
    chunk = chunk.drop(columns=answer_cols)

    cols_to_num = eye_cols + text_cols
    chunk[cols_to_num] = chunk[cols_to_num].apply(pd.to_numeric, errors="coerce")

    # optional log-transform on skewed columns
    for col in [
        "IA_DWELL_TIME",
        "IA_FIRST_RUN_DWELL_TIME",
        "IA_REGRESSION_PATH_DURATION",
    ]:
        chunk[col] = np.log1p(chunk[col])

    grp = (
        chunk.groupby(group_cols)[eye_cols + text_cols + ["correct"]]
        .median()  # more robust than mean
        .reset_index()
    )
    records.append(grp)

df = pd.concat(records, ignore_index=True)
print(f"Final table shape: {df.shape}")

# ---------- 4.  Prepare X, y, groups ----------
X = df[eye_cols + text_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df["correct"].round().astype(int)
grp = df["participant_id"]

# ---------- 5.  Participant-wise 5-fold CV ----------
cv = GroupKFold(n_splits=5)
aucs = []
feat_imp = np.zeros(X.shape[1])

print("Running Random-Forest CV …")
for tr, te in cv.split(X, y, grp):
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    clf.fit(X.iloc[tr], y.iloc[tr])
    proba = clf.predict_proba(X.iloc[te])[:, 1]
    aucs.append(roc_auc_score(y.iloc[te], proba))
    feat_imp += clf.feature_importances_

# ---------- 6.  Baseline AUCs ----------
print("Eye-only  :", quick_auc(X[eye_cols]))
print("Text-only :", quick_auc(X[text_cols]))
print(f"\nMean RF AUC across folds: {np.mean(aucs):.3f}")

# ---------- 7.  Top-10 RF feature importances ----------
importance = (
    pd.Series(feat_imp / cv.get_n_splits(), index=X.columns)
    .sort_values(ascending=False)
    .head(10)
)
print("\nTop-10 RF features:")
print(importance.round(4))
