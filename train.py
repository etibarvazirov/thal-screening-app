# train.py
import pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = "data/HPLC data.csv"
MODEL_PATH = "artifacts/model.pkl"
os.makedirs("artifacts", exist_ok=True)

def map_dx_to_3(s):
    s = s.astype(str).str.lower()
    y = pd.Series(np.nan, index=s.index)
    y[s.str.contains(r"(disease|major|intermedia|severe|hbh|bart)", na=False)] = 2
    y[s.str.contains(r"(minor|trait|carrier)", na=False) & y.isna()] = 1
    y[s.str.contains(r"(normal|healthy|control)", na=False) & y.isna()] = 0
    return y.astype("Int64")

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().replace(" ","_").replace("-","_") for c in df.columns]
y = map_dx_to_3(df["Diagnosis"])
mask = y.notna()
df, y = df.loc[mask].reset_index(drop=True), y.loc[mask].astype(int)

X = df.drop(columns=[c for c in ["Sl_No","Diagnosis"] if c in df.columns]).copy()
num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in X.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

pipe = ImbPipeline([
    ("prep", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("clf", MLPClassifier(hidden_layer_sizes=(160,80),
                          activation="relu",
                          learning_rate_init=1e-3,
                          alpha=2e-4,
                          max_iter=400,
                          early_stopping=True,
                          random_state=42))
])

pipe.fit(X_tr, y_tr)
joblib.dump({"model": pipe, "num_cols": num_cols, "cat_cols": cat_cols}, MODEL_PATH)
print("Saved:", MODEL_PATH)
