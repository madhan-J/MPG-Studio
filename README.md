# MPG Studio – Fuel Efficiency Predictor (Streamlit)

Main app: streamlit_app.py  (designed UI in index.html: loads the bundled dataset, trains 4 models in the browser,
live prediction from typed values, upload your own CSV to retrain; theme and effects change with the result)
Pure-Python alternative: streamlit_app_python.py  (same idea with Streamlit widgets; predictions run through the
scikit-learn pipeline on the server; upload a CSV in the sidebar to retrain)
ML work: train.py (4 algorithms compared: Ridge, SVR, Random Forest, Gradient Boosting; best by CV RMSE deployed),
auto-mpg.csv, model.joblib, metrics.json

Run locally:  pip install -r requirements.txt && streamlit run streamlit_app.py
Deploy: push this folder to GitHub -> share.streamlit.io -> New app -> main file: streamlit_app.py
(For the Python-pipeline version choose streamlit_app_python.py as the main file.)
