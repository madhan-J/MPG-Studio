"""
train.py - Train and compare 4 regression models for MPG prediction,
then save the best full pipeline (preprocessing + model) for deployment.

Usage:  python train.py [path/to/auto-mpg.csv]
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".csv") else "auto-mpg.csv"

NUM_FEATURES = ["cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year"]
CAT_FEATURES = ["origin", "brand"]

BRAND_FIX = {"chevy": "chevrolet", "chevroelt": "chevrolet", "vw": "volkswagen",
             "vokswagen": "volkswagen", "volkswagen": "volkswagen",
             "toyouta": "toyota", "maxda": "mazda",
             "mercedes-benz": "mercedes", "mercedes": "mercedes"}


# ----------------------------------------------------------------------
# 1. Load + clean
# ----------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Missing values are often stored as '?' (e.g. in horsepower) -> NaN
    df = df.replace("?", np.nan)
    for col in ["mpg", "cylinders", "displacement", "horsepower",
                "weight", "acceleration", "model_year", "origin"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Brand = first word of car name (with typo fixes)
    brand = df["car_name"].astype(str).str.lower().str.strip().str.split().str[0]
    df["brand"] = brand.replace(BRAND_FIX)

    # origin is a category (1/2/3), not an ordered number
    df["origin"] = df["origin"].map({1: "USA", 2: "Europe", 3: "Japan"})

    df = df.dropna(subset=["mpg"])          # target cannot be missing
    return df


# ----------------------------------------------------------------------
# 2. Preprocessing (imputation + scaling + encoding)
# ----------------------------------------------------------------------
def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=3)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_FEATURES),
        ("cat", categorical, CAT_FEATURES),
    ])


# ----------------------------------------------------------------------
# 3. Candidate models + small hyper-parameter grids
# ----------------------------------------------------------------------
CANDIDATES = {
    "Ridge Regression": (Ridge(), {"model__alpha": [0.1, 1, 10, 50]}),
    "Random Forest": (
        RandomForestRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [200, 400], "model__max_depth": [None, 10]},
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"model__n_estimators": [200, 400],
         "model__learning_rate": [0.03, 0.1], "model__max_depth": [2, 3]},
    ),
    "SVR (RBF)": (SVR(), {"model__C": [1, 10, 50], "model__gamma": ["scale", 0.05]}),
}


def train_and_save(df, log=print):
    """Train all candidates on a cleaned dataframe, save best pipeline + metadata."""
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = [], {}

    for name, (model, grid) in CANDIDATES.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        search = GridSearchCV(pipe, grid, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        results.append({
            "Model": name,
            "CV RMSE": round(-search.best_score_, 3),
            "Test RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
            "Test MAE": round(float(mean_absolute_error(y_test, pred)), 3),
            "Test R2": round(float(r2_score(y_test, pred)), 3),
            "Best Params": str(search.best_params_),
        })
        fitted[name] = search.best_estimator_
        log(f"{name:20s} CV RMSE={results[-1]['CV RMSE']}  "
            f"Test R2={results[-1]['Test R2']}")

    res = pd.DataFrame(results).sort_values("CV RMSE").reset_index(drop=True)
    best_name = res.loc[0, "Model"]       # chosen on CV (not on test) to avoid leakage
    log("\n" + res.drop(columns="Best Params").to_string(index=False))
    log(f"\nBest model: {best_name}")

    # Refit the winner on ALL data for deployment
    final = fitted[best_name]
    final.fit(X, y)
    joblib.dump(final, "mpg_model.joblib")

    meta = {
        "best_model": best_name,
        "brands": sorted(df["brand"].dropna().unique().tolist()),
        "ranges": {c: [float(df[c].min()), float(df[c].max()),
                       float(df[c].median())] for c in NUM_FEATURES},
        "results": res.to_dict(orient="records"),
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log("Saved mpg_model.joblib and model_meta.json")


def main():
    train_and_save(load_data(CSV_PATH))


if __name__ == "__main__":
    main()
