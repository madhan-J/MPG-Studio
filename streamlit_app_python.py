import json
from pathlib import Path
import joblib, pandas as pd, streamlit as st
from train import train

HERE = Path(__file__).parent
st.set_page_config(page_title="MPG Studio", page_icon="🏎️", layout="wide")


@st.cache_resource
def load_default():
    try:  # saved artefacts; if scikit-learn versions differ on the server, retrain
        m, meta = joblib.load(HERE / "model.joblib"), json.load(open(HERE / "metrics.json"))
        m.predict(pd.DataFrame([{"cylinders": 4, "displacement": 150, "horsepower": None, "weight": 2500,
                                 "acceleration": 15, "model year": 75, "origin": 1}]))  # sanity check
        return m, meta
    except Exception:
        return train()


if "art" not in st.session_state:
    st.session_state.art = load_default()

with st.sidebar:
    st.header("📂 Your dataset")
    up = st.file_uploader("Upload auto-mpg.csv", type="csv")
    if up and st.button("Train on this file", type="primary"):
        try:
            with st.spinner("Training and tuning 4 models..."):
                st.session_state.art = train(df=pd.read_csv(up))
            st.success("New model deployed")
        except Exception as e:
            st.error(str(e))
model, meta = st.session_state.art
R = meta["ranges"]

st.markdown("<h1 id='t'>MPG Studio</h1>", unsafe_allow_html=True)
st.caption(f"Deployed model: **{meta['best']}** · trained on {meta['n']} cars ({meta['n_missing_hp']} missing horsepower values auto-filled)")

tab_p, tab_m = st.tabs(["Live prediction", "Model comparison"])
with tab_p:
    c1, c2, c3 = st.columns(3)
    cyl = c1.number_input("Cylinders", 1, 16, int(R["cylinders"][2]), step=1)
    disp = c1.number_input("Engine displacement (cu. in.)", 0.0, 1000.0, float(R["displacement"][2]), step=1.0)
    known = c2.checkbox("I know the horsepower", True)
    hp = c2.number_input("Horsepower", 0.0, 1000.0, float(R["horsepower"][2]), step=1.0, disabled=not known)
    wt = c2.number_input("Weight (lbs)", 0.0, 10000.0, float(R["weight"][2]), step=10.0)
    acc = c3.number_input("Acceleration (sec)", 0.0, 40.0, float(R["acceleration"][2]), step=0.1)
    yr = c3.number_input("Model year (e.g. 76)", 60, 100, int(R["model year"][2]), step=1)
    org = c3.selectbox("Origin", [1, 2, 3], format_func={1: "USA", 2: "Europe", 3: "Japan"}.get)

    row = pd.DataFrame([{"cylinders": cyl, "displacement": disp, "horsepower": hp if known else None,
                         "weight": wt, "acceleration": acc, "model year": yr, "origin": org}])
    mpg = float(max(4, min(55, model.predict(row)[0])))

    # the whole look changes with the result
    if mpg < 18:   hue, name, dur = 12, "Thirsty beast", 2
    elif mpg < 27: hue, name, dur = 45, "Balanced cruiser", 6
    else:          hue, name, dur = 160, "Eco champion", 14
    off = 251.3 * (1 - min(mpg, 50) / 50)
    css = """<style>
    #t{font-size:clamp(2.2rem,6vw,4rem);margin:0;background:linear-gradient(90deg,hsl(__H__ 95% 62%),hsl(calc(__H__ + 50) 95% 60%));
       -webkit-background-clip:text;background-clip:text;color:transparent}
    .stApp{background:radial-gradient(60% 50% at 15% 10%,hsl(__H__ 90% 28% / .55),transparent),
           radial-gradient(50% 50% at 90% 90%,hsl(calc(__H__ + 60) 90% 26% / .5),transparent),#07070f}
    .streaks{position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.35;
       background:repeating-linear-gradient(90deg,transparent 0 90px,hsl(__H__ 95% 65% / .5) 90px 92px,transparent 92px 230px);
       background-size:230px 100%;animation:sl __D__s linear infinite}
    @keyframes sl{to{background-position:-230px 0}}
    .gw{max-width:340px;margin:10px auto;position:relative}
    .gw svg{width:100%;overflow:visible}.arc{fill:none;stroke-width:14;stroke-linecap:round}
    #f{stroke:url(#g);stroke-dasharray:251.3;stroke-dashoffset:__O__;animation:fi 1.1s cubic-bezier(.2,.8,.2,1);
       filter:drop-shadow(0 0 8px hsl(__H__ 95% 60%))}
    @keyframes fi{from{stroke-dashoffset:251.3}}
    .rd{position:absolute;left:0;right:0;bottom:0;text-align:center;font-size:3.4rem;font-weight:800;line-height:1}
    .rd small{display:block;font-size:.75rem;letter-spacing:.14em;color:#aab0d8;font-weight:400}
    .tg{text-align:center;font-weight:700;font-size:1.3rem;color:hsl(__H__ 95% 65%);text-shadow:0 0 14px hsl(__H__ 95% 55% / .6)}
    </style><div class='streaks'></div>
    <div class='gw'><svg viewBox='0 0 200 112'><defs><linearGradient id='g'><stop offset='0' stop-color='#ff5b3a'/>
    <stop offset='.5' stop-color='#ffc83a'/><stop offset='1' stop-color='#2dffb0'/></linearGradient></defs>
    <path class='arc' stroke='#ffffff18' d='M20 100 A80 80 0 0 1 180 100'/><path class='arc' id='f' d='M20 100 A80 80 0 0 1 180 100'/></svg>
    <div class='rd'>__M__<small>MILES PER GALLON</small></div></div>
    <div class='tg'>__N__ · ≈ __K__ km/L</div>"""
    st.markdown(css.replace("__H__", str(hue)).replace("__D__", str(dur)).replace("__O__", f"{off:.1f}")
                .replace("__M__", f"{mpg:.1f}").replace("__N__", name).replace("__K__", f"{mpg * 0.4251:.1f}"),
                unsafe_allow_html=True)
    if not known:
        st.info("Horsepower left unknown → the pipeline filled it with the dataset median.")

with tab_m:
    st.write("Each model was tuned with 5-fold cross-validation; the lowest CV RMSE is deployed (👑).")
    df = pd.DataFrame(meta["results"]).rename(columns={"name": "Model", "cv_rmse": "CV RMSE",
                                                        "rmse": "Test RMSE", "mae": "Test MAE", "r2": "Test R²"})
    df["Model"] = [("👑 " if n == meta["best"] else "") + n for n in df["Model"]]
    st.dataframe(df, hide_index=True)
