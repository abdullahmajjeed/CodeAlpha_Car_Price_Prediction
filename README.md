# 🚗 Consumer Vehicle Valuation Predictor Engine

A professional, end-to-end Machine Learning web application designed to predict the fair market selling price of used vehicles. This project utilizes an optimized Gradient Boosting Regressor backend coupled with a real-time, interactive Streamlit frontend dashboard.

---

## 📊 Project Workflow & Data Science Pipeline

1. **Data Cleaning & Verification**: The pipeline ingests raw vehicular records, strictly evaluates data types, and applies runtime handling routines for missing values (`NaN` fields) directly on numerical and categorical matrices.
2. **Feature Engineering**: Converts absolute temporal columns (`Year`) into dynamic structural depreciation metrics (`Car_Age`) relative to the evaluation timeframe.
3. **Manual Feature Transformation**: Implements strict, reliable One-Hot Encoding directly on categorical variations (`Fuel_Type`, `Selling_type`, `Transmission`) to eliminate high-variance noise without complex wrapper dependencies.
4. **Model Architecture**: Trains a production-optimized **Gradient Boosting Regressor** with tailored tuning parameters (`n_estimators=150`, `learning_rate=0.1`) to ensure extreme predictive alignment.
5. **Artifact Serialization**: Exports the finalized, trained model architecture (`car_price_predictor.pkl`) alongside its feature column state (`model_columns.pkl`) for real-time edge processing on the frontend.

---

## 🛠️ Tech Stack & Dependencies

- **Interactive Dashboard:** Streamlit
- **Core Processing Engine:** Python 3.x
- **Data Engineering:** Pandas, NumPy
- **Machine Learning Architecture:** Scikit-Learn
- **Serialization Handling:** Joblib

---

## ⚙️ How to Setup & Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPO_NAME
```

### 2. Install Required Dependencies
Ensure you have the core packages installed in your environment:
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### 3. Generate the Production Artifacts
Open and execute all code cells within the sanitized Jupyter Notebook to output your synchronized serialized arrays:
```bash
jupyter notebook car_price_prediction.ipynb
```

### 4. Boot Up the Web Interface
Launch the production Streamlit engine instance from your terminal:
```bash
streamlit run car_price_prediction.py
```

---

## 🚀 Model Deployment State & Artifacts
- **`car_price_predictor.pkl`**: The compiled, serialized "brain" containing trained decision parameters.
- **`model_columns.pkl`**: A hardcoded structure index that ensures web dashboard payload arrays perfectly align with the core matrix layout to prevent runtime crashes.
