import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Prediction", page_icon="📈", layout="centered")
st.title("📈 Sales Prediction (Linear Regression)")

# โหลดโมเดล
@st.cache_resource
def load_model():
    model = joblib.load("model-reg-xxx.pkl")
    feature_order = ["youtube", "tiktok", "instagram"]
    return model, feature_order

try:
    model, feature_order = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Cannot load model-reg-xxx.pkl: {e}")
    st.stop()

st.markdown("กรอกงบโฆษณาแต่ละช่องทางแล้วคลิก **Predict**")

col1, col2, col3 = st.columns(3)
with col1:
    youtube = st.number_input("youtube", min_value=0.0, value=50.0, step=1.0)
with col2:
    tiktok = st.number_input("tiktok", min_value=0.0, value=50.0, step=1.0)
with col3:
    instagram = st.number_input("instagram", min_value=0.0, value=50.0, step=1.0)

if st.button("Predict"):
    X = pd.DataFrame([[youtube, tiktok, instagram]], columns=feature_order)
    y_hat = model.predict(X)[0]
    st.success(f"Estimated sales: **{y_hat:.2f}**")
