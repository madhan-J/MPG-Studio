from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="MPG Studio", page_icon="🏎️", layout="wide")
st.markdown("<style>.block-container{padding:0!important;max-width:100%!important}header,footer{visibility:hidden}</style>", unsafe_allow_html=True)
components.html((Path(__file__).parent / "index.html").read_text(encoding="utf-8"), height=1300, scrolling=True)
