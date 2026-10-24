"""Compare 4 algorithms (Ridge, SVR, Random Forest, Gradient Boosting), pick the best by
cross-validated RMSE, refit on all data.
Run `python train.py` to regenerate model.joblib + metrics.json."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).parent
NUM = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
CAT = ["origin"]


def clean(df):
    """Normalise column names, turn '?' into NaN, drop rows without target/origin."""
    df = df.copy()
    df.columns = [c.strip().lower().replace("_", " ") for c in df.columns]
    missing = [c for c in ["mpg"] + NUM + CAT if c not in df.columns]
    if missing:
        raise ValueError("Missing column(s): " + ", ".join(missing))
    for c in ["mpg"] + NUM + CAT:
        df[c] = pd.to_numeric(df[c], errors="coerce")   # '?' -> NaN
    df = df.dropna(subset=["mpg", "origin"])
    df["origin"] = df["origin"].astype(int)
    return df


def pipe(model):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT)])
    return Pipeline([("prep", prep), ("model", model)])


CANDIDATES = {
    "Ridge Regression": (Ridge(), {"model__alpha": [0.1, 1, 10]}),
    "Support Vector Regressor": (SVR(epsilon=0.2), {"model__C": [5, 20, 60], "model__gamma": ["scale", 0.05]}),
    "Random Forest": (RandomForestRegressor(random_state=42, n_jobs=-1),
                      {"model__n_estimators": [200], "model__max_depth": [None, 12]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=42),
                          {"model__n_estimators": [200, 400], "model__learning_rate": [0.03, 0.08],
                           "model__max_depth": [3]}),
}


def train(save=False, df=None):
    df = clean(pd.read_csv(HERE / "auto-mpg.csv") if df is None else df)
    X, y = df[NUM + CAT], df["mpg"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    cv, results, searches = KFold(5, shuffle=True, random_state=42), [], {}
    for name, (m, grid) in CANDIDATES.items():
        gs = GridSearchCV(pipe(m), grid, cv=cv, scoring="neg_root_mean_squared_error").fit(X_tr, y_tr)
        pred = gs.predict(X_te)
        searches[name] = gs
        results.append(dict(name=name, cv_rmse=round(float(-gs.best_score_), 3),
            rmse=round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
            mae=round(float(mean_absolute_error(y_te, pred)), 3), r2=round(float(r2_score(y_te, pred)), 4)))
    best = min(results, key=lambda r: r["cv_rmse"])["name"]
    final = clone(searches[best].best_estimator_).fit(X, y)       # refit winner on ALL data
    meta = dict(best=best, results=results, n=len(df), n_missing_hp=int(df["horsepower"].isna().sum()),
        ranges={c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in NUM})
    if save:
        joblib.dump(final, HERE / "model.joblib")
        json.dump(meta, open(HERE / "metrics.json", "w"))
    return final, meta


if __name__ == "__main__":
    _, m = train(save=True)
    for r in m["results"]:
        print(r)
    print("Deployed:", m["best"])
