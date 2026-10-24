# MPG Predictor (Streamlit)

Predicts a car's fuel efficiency (MPG). Trains 4 models (Ridge, Random Forest,
Gradient Boosting, SVR) with 5-fold CV and deploys the best one.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Add your `auto-mpg.csv` to this folder (next to `app.py`).
2. Push all files to a GitHub repository.
3. On share.streamlit.io click "Create app", pick the repo, set main file to `app.py`, Deploy.

The model trains automatically on the first launch. If `auto-mpg.csv` is not in the repo,
the app shows an upload box so you can train from the browser instead.
