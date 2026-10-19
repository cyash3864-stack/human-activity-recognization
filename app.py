import os
import sys

from streamlit.runtime.scriptrunner import get_script_run_ctx

if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", os.path.abspath(__file__), *sys.argv[1:]]
    raise SystemExit(stcli.main())

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "har_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "har_scaler.joblib")
X_TEST_PATH = os.path.join(BASE_DIR, "X_test_raw.csv")
Y_TEST_PATH = os.path.join(BASE_DIR, "y_test.csv")

CLASS_NAMES = ["Walking", "Walking Up", "Walking Down", "Sitting", "Standing", "Laying"]
N_FEATURES = 561

st.set_page_config(page_title="Activity Guesser", page_icon="🚶", layout="wide")

@st.cache_resource
def load_artifacts():
    return keras.models.load_model(MODEL_PATH), joblib.load(SCALER_PATH)

@st.cache_data
def load_samples():
    if os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
        X = pd.read_csv(X_TEST_PATH, header=None)
        y = pd.read_csv(Y_TEST_PATH, header=None).iloc[:, 0].astype(int).values
        return X, y
    return None, None

def predict_activity(values):
    model, scaler = load_artifacts()
    arr = np.asarray(values, dtype=np.float32).reshape(1, -1)
    probs = model.predict(scaler.transform(arr), verbose=0)[0]
    return CLASS_NAMES[int(np.argmax(probs))], probs

st.title("🏃 Human Activity Guesser")
st.caption("A simple ANN class project demo")

left, right = st.columns([1, 1])
with left:
    st.markdown("### What is this?")
    st.write("This webpage is a live demo of your notebook model. The model was trained to recognize human activity from smartphone sensor data.")
    st.write("You are not expected to understand the raw numbers. Just use the slider and read the result.")
with right:
    st.markdown("### What does it use to predict?")
    st.write("- Phone accelerometer/gyroscope sensor features")
    st.write("- 561 numbers from one time window")
    st.write("- A trained neural network")

st.markdown("---")

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    st.error("Run `python train_model.py` first.")
    st.stop()

X_samples, y_samples = load_samples()
if X_samples is None:
    st.warning("Sample data not found. Run training again.")
    st.stop()

row = st.slider("Choose a saved phone-sensor sample", 0, len(X_samples)-1, 0)
actual = CLASS_NAMES[y_samples[row]]
predicted, probs = predict_activity(X_samples.iloc[row].values)

c1, c2, c3 = st.columns(3)
c1.metric("Sample number", row)
c2.metric("Actual activity", actual)
c3.metric("AI prediction", predicted)

if predicted == actual:
    st.success("✅ Correct prediction")
else:
    st.error("❌ Wrong prediction")

st.markdown("### Why this sample is what it is")
st.write(f"The saved label says this sensor window was **{actual}**. The ANN looked at the sensor features and guessed **{predicted}**.")

st.markdown("### Model confidence")
prob_df = pd.DataFrame({"Activity": CLASS_NAMES, "Confidence": probs}).set_index("Activity")
st.bar_chart(prob_df)
st.dataframe(prob_df.style.format({"Confidence": "{:.1%}"}), use_container_width=True)

with st.expander("Show first few raw sensor features"):
    st.write(X_samples.iloc[row].head(30).to_frame(name="value"))
