# train.py — sklearn-only, server-friendly
import os, joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/HPLC data.csv"
MODEL_PATH = "artifacts/model.pkl"
os.makedirs("artifacts", exist_ok=True)

def map_dx_to_3(s):
    s = s.astype(str).str.lower()
    y = pd.Series(np.nan, index=s.index)
    # \b və non-capturing (?:...) istifadə edirik; regex xəbərdarlığı yox olur
    y[s.str.contains(r"\b(?:disease|major|intermedia|severe|hbh|bart)\b",  na=False, case=False)] = 2
    y[s.str.contains(r"\b(?:minor|trait|carrier)\b",na=False, case=False) & y.isna()] = 1
    y[s.str.contains(r"\b(?:normal|healthy|control)\b",na=False, case=False) & y.isna()] = 0

    return y.astype("Int64")

def upsample_train_only(X_tr, y_tr, random_state=42):
    """Class imbalance üçün sadə upsampling (yalnız train-də)."""
    df_tr = X_tr.copy()
    df_tr["__y__"] = y_tr.values
    counts = df_tr["__y__"].value_counts()
    max_n = counts.max()
    parts = []
    rng = np.random.RandomState(random_state)
    for cls, n in counts.items():
        block = df_tr[df_tr["__y__"] == cls]
        if n < max_n:
            extra = block.sample(max_n - n, replace=True, random_state=rng)
            block = pd.concat([block, extra], ignore_index=True)
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    y_new = out["__y__"].astype(int).copy()
    X_new = out.drop(columns=["__y__"]).copy()
    return X_new, y_new

# 1) Load
assert os.path.exists(DATA_PATH), "Dataset not found at data/HPLC data.csv"
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().replace(" ","_").replace("-","_") for c in df.columns]

# 2) Target
y = map_dx_to_3(df["Diagnosis"])
mask = y.notna()
df, y = df.loc[mask].reset_index(drop=True), y.loc[mask].astype(int)

# 3) Features
drop_cols = [c for c in ["Sl_No","Diagnosis"] if c in df.columns]
X = df.drop(columns=drop_cols).copy()
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 4) Upsample yalnız train-də
X_tr_up, y_tr_up = upsample_train_only(X_tr, y_tr, random_state=42)

# 5) Modellər (class_weight balans üçün əlavə kömək edir)
models = {
    "LogReg": Pipeline([("prep", preprocess),
                        ("clf", LogisticRegression(max_iter=800, class_weight="balanced", random_state=42))]),
    "RandomForest": Pipeline([("prep", preprocess),
                              ("clf", RandomForestClassifier(
                                  n_estimators=400, min_samples_leaf=2,
                                  class_weight="balanced", random_state=42, n_jobs=-1))]),
}

# 6) Ən yaxşısını seç (macro-F1 ilə)
from sklearn.metrics import f1_score
best_name, best_pipe, best_f1 = None, None, -1
for name, pipe in models.items():
    pipe.fit(X_tr_up, y_tr_up)
    f1m = f1_score(y_te, pipe.predict(X_te), average="macro")
    if f1m > best_f1:
        best_name, best_pipe, best_f1 = name, pipe, f1m

joblib.dump({"model": best_pipe}, MODEL_PATH)
print(f"Saved {MODEL_PATH} with model: {best_name}, F1_macro={best_f1:.3f}")

