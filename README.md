# Human Activity Recognition with ANN

This project uses the UCI HAR dataset and a TensorFlow/Keras ANN to classify activities.

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This creates:
- `har_model.keras`
- `har_scaler.joblib`
- `X_test_raw.csv`
- `y_test.csv`

## Run the app

```bash
python app.py
```

This starts the Streamlit server and opens the app in your browser. The terminal
also prints the local URL (usually `http://localhost:8501`). You can still launch
it with `streamlit run app.py` if you prefer.
