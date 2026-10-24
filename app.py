"""
app.py - Streamlit app for MPG prediction.
Run:  streamlit run app.py     (after running  python train.py)
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("mpg_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


if not (os.path.exists("mpg_model.joblib") and os.path.exists("model_meta.json")):
    st.title("🚗 MPG Predictor - first-time setup")
    import train
    if os.path.exists("auto-mpg.csv"):      # dataset shipped with the repo -> train automatically
        with st.spinner("First run: training and tuning 4 models (about a minute)..."):
            train.train_and_save(train.load_data("auto-mpg.csv"), log=lambda *_: None)
        load_artifacts.clear()
        st.rerun()
    st.info("No trained model found. Upload `auto-mpg.csv` to train all 4 models "
            "and deploy the best one.")
    up = st.file_uploader("auto-mpg.csv", type="csv")
    if up is not None and st.button("Train models", type="primary"):
        with st.spinner("Training and tuning 4 models (about a minute)..."):
            logs = []
            train.train_and_save(train.load_data(up), log=logs.append)
        st.code("\n".join(logs))
        load_artifacts.clear()
        st.success("Done! Reload the page to start predicting.")
        st.button("Reload")
    st.stop()

model, meta = load_artifacts()

R = meta["ranges"]
st.title("🚗 Automobile Fuel Efficiency (MPG) Predictor")
st.caption(f"Deployed model: **{meta['best_model']}**")

tab_pred, tab_models = st.tabs(["Predict", "Model comparison"])

with tab_pred:
    c1, c2 = st.columns(2)
    with c1:
        cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
        displacement = st.number_input("Displacement (cu. in.)", 50.0, 600.0,
                                       float(R["displacement"][2]), step=1.0)
        weight = st.number_input("Weight (lbs)", 1000.0, 6000.0,
                                 float(R["weight"][2]), step=10.0)
        acceleration = st.number_input("Acceleration (0-60 mph, sec)", 5.0, 30.0,
                                       float(R["acceleration"][2]), step=0.1)
    with c2:
        hp_known = st.checkbox("I know the horsepower", value=True)
        horsepower = st.number_input("Horsepower", 40.0, 400.0,
                                     float(R["horsepower"][2]), step=1.0,
                                     disabled=not hp_known)
        model_year = st.slider("Model year (19xx)", 70, 82, 76)
        origin = st.selectbox("Origin", ["USA", "Europe", "Japan"])
        brand = st.selectbox("Brand", meta["brands"],
                             index=meta["brands"].index("ford")
                             if "ford" in meta["brands"] else 0)

    if st.button("Predict MPG", type="primary"):
        row = pd.DataFrame([{
            "cylinders": cylinders,
            "displacement": displacement,
            # Unknown horsepower -> NaN; the pipeline imputes it automatically
            "horsepower": horsepower if hp_known else np.nan,
            "weight": weight,
            "acceleration": acceleration,
            "model_year": model_year,
            "origin": origin,
            "brand": brand,
        }])
        mpg = float(model.predict(row)[0])
        st.success(f"Predicted fuel efficiency: **{mpg:.1f} MPG**  "
                   f"(~{mpg * 0.4251:.1f} km/L)")
        if mpg < 15:
            st.info("Thirsty engine: low efficiency.")
        elif mpg > 30:
            st.info("Very fuel efficient.")

with tab_models:
    st.write("All models were tuned with 5-fold cross-validation; "
             "the one with the lowest CV RMSE was deployed.")
    df_res = pd.DataFrame(meta["results"]).drop(columns=["Best Params"])
    st.dataframe(df_res, hide_index=True)
