"""
Car Selling Price Predictor — Streamlit App
Updated version fully compatible with Python 3.13 and scikit-learn 1.3+
"""

import io
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import pandas as pd
import numpy as np
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price Predictor", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper functions
# ---------------------------

def gdrive_to_direct(drive_url: str) -> str:
    if not drive_url:
        return ""
    parsed = urlparse(drive_url)
    qs = parse_qs(parsed.query)
    if "id" in qs:
        file_id = qs["id"][0]
    else:
        parts = parsed.path.split("/")
        if "d" in parts:
            d_idx = parts.index("d")
            if d_idx + 1 < len(parts):
                file_id = parts[d_idx + 1]
            else:
                return drive_url
        else:
            return drive_url
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def load_data_from_drive(drive_url: str) -> pd.DataFrame:
    direct = gdrive_to_direct(drive_url)
    try:
        import gdown
        out = "temp_car_dataset.csv"
        gdown.download(direct, out, quiet=True)
        df = pd.read_csv(out)
        os.remove(out)
        return df
    except Exception as e:
        try:
            return pd.read_csv(direct)
        except Exception as e2:
            raise RuntimeError(f"Could not download/read the file automatically.\n"
                               f"gdown error: {e}\nread_csv error: {e2}\n"
                               f"Please download the CSV manually and upload it via the app.") from e2

def preprocess_df(df: pd.DataFrame, reference_year: int = None, fit_encoder=None):
    df = df.copy()
    expected_columns = {"Car_Name", "Year", "Selling_Price", "Present_Price",
                        "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"}
    if not expected_columns.issubset(set(df.columns)):
        missing = expected_columns - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.dropna(subset=["Selling_Price", "Present_Price", "Kms_Driven", "Year"])

    # Numeric conversions
    df["Kms_Driven"] = pd.to_numeric(df["Kms_Driven"], errors="coerce")
    df["Present_Price"] = pd.to_numeric(df["Present_Price"], errors="coerce")
    df["Selling_Price"] = pd.to_numeric(df["Selling_Price"], errors="coerce")
    df["Owner"] = pd.to_numeric(df["Owner"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Kms_Driven", "Present_Price", "Selling_Price", "Owner", "Year"])

    # --- Brand & Model parsing ---
    df["Car_Name"] = df["Car_Name"].astype(str)
    df["Brand"] = df["Car_Name"].str.split(" ").str[0]
    df["Model"] = df["Car_Name"].str.split(" ", n=1).str[1].fillna("Unknown")

    # Car age
    if reference_year is None:
        reference_year = datetime.now().year
    df["Car_Age"] = (reference_year - df["Year"]).clip(lower=0, upper=50)

    X = df[["Present_Price", "Kms_Driven", "Owner", "Car_Age",
            "Brand", "Model", "Fuel_Type", "Seller_Type", "Transmission"]]
    y = df["Selling_Price"].copy()

    # Drop extreme outliers
    ratio = y / (X["Present_Price"].replace(0, np.nan))
    mask = (ratio < 5) | (ratio.isna())
    X, y = X[mask], y[mask]

    num_cols = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
    cat_cols = ["Brand", "Model", "Fuel_Type", "Seller_Type", "Transmission"]

    if fit_encoder is None:
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ])
        preprocessor.fit(X)
    else:
        preprocessor = fit_encoder

    X_trans = preprocessor.transform(X)

    try:
        cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    except Exception:
        cat_feature_names = []
    feature_names = num_cols + cat_feature_names

    X_trans_df = pd.DataFrame(X_trans, columns=feature_names, index=X.index)
    return X_trans_df, y.loc[X_trans_df.index], preprocessor, feature_names

def train_model(X, y, do_grid_search=False, random_state=42):
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    if do_grid_search:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 8, 12],
            "min_samples_split": [2, 5]
        }
        gs = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, verbose=0)
        gs.fit(X, y)
        return gs.best_estimator_, gs
    else:
        rf.fit(X, y)
        return rf, None

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2, "y_true": y_test, "y_pred": preds}

def plot_true_vs_pred(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    minv = min(y_true.min(), y_pred.min())
    maxv = max(y_true.max(), y_pred.max())
    ax.plot([minv, maxv], [minv, maxv], linestyle="--")
    ax.set_xlabel("True Selling Price")
    ax.set_ylabel("Predicted Selling Price")
    ax.set_title("True vs Predicted Selling Price")
    st.pyplot(fig)

def plot_feature_importances(model, feature_names, top_n=12):
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        df_fi = pd.DataFrame({"feature": feature_names, "importance": fi})
        df_fi = df_fi.sort_values("importance", ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(df_fi))))
        ax.barh(df_fi["feature"][::-1], df_fi["importance"][::-1])
        ax.set_title("Feature Importances")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model type.")

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🚗 Car Selling Price Predictor — Streamlit Demo")

with st.sidebar:
    st.header("Load dataset")
    drive_link_input = st.text_input("Google Drive share link", 
                                    value="https://drive.google.com/file/d/1yFuNVPXM5CH6g0TthYKcTGrZCCJo6n8Z/view?usp=drive_link")
    uploaded_file = st.file_uploader("Or upload dataset CSV", type=["csv"])
    ref_year = st.number_input("Reference year for 'Car_Age'", value=datetime.now().year, step=1)
    run_train = st.button("Train model")
    retrain_grid = st.checkbox("Perform light Grid Search", value=False)
    st.markdown("---")
    st.header("Model file")
    model_filename = st.text_input("Model filename to save/load", value="car_price_model.joblib")
    if st.button("Load saved model from disk"):
        if os.path.exists(model_filename):
            saved = joblib.load(model_filename)
            if isinstance(saved, dict) and ("model" in saved and "preprocessor" in saved):
                st.session_state["model_bundle"] = saved
                st.success("Loaded model bundle from disk.")
            else:
                st.error("Saved file doesn't contain expected bundle.")
        else:
            st.error(f"File not found: {model_filename}")
    if st.button("Clear trained model from session"):
        if "model_bundle" in st.session_state:
            del st.session_state["model_bundle"]
            st.success("Cleared session model.")
        else:
            st.info("No trained model in session.")

# ---------------------------
# Load dataset
# ---------------------------
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Uploaded dataset loaded.")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
elif drive_link_input:
    try:
        with st.spinner("Downloading dataset from Google Drive..."):
            df = load_data_from_drive(drive_link_input)
        st.success("Dataset downloaded and loaded.")
    except Exception as e:
        st.warning("Automatic download failed. Please upload CSV manually.")
        st.write(e)

# ---------------------------
# Parse Brand & Model
# ---------------------------
if df is not None and "Car_Name" in df.columns:
    df["Car_Name"] = df["Car_Name"].astype(str)
    df["Brand"] = df["Car_Name"].str.split(" ").str[0]
    df["Model"] = df["Car_Name"].str.split(" ", n=1).str[1].fillna("Unknown")

# ---------------------------
# Preprocessing & Training
# ---------------------------
if df is not None:
    st.subheader("Raw data preview")
    st.dataframe(df.head(10))

    st.markdown("### Dataset summary")
    st.write(df.describe(include="all"))

    try:
        X_all, y_all, preproc_all, feature_names = preprocess_df(df, reference_year=ref_year)
        st.markdown("### Preprocessed data (first 10 rows)")
        st.dataframe(pd.concat([X_all.reset_index(drop=True), y_all.reset_index(drop=True)], axis=1).head(10))
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        X_all = y_all = preproc_all = None

    if run_train and X_all is not None:
        st.info("Training model — this may take some seconds.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
            model, gs = train_model(X_train, y_train, do_grid_search=retrain_grid)
            results = evaluate_model(model, X_test, y_test)

            st.session_state["model_bundle"] = {
                "model": model,
                "preprocessor": preproc_all,
                "feature_names": feature_names,
                "reference_year": ref_year
            }

            st.success("Training completed.")
            cols = st.columns(3)
            cols[0].metric("MAE", f"{results['mae']:.3f}")
            cols[1].metric("RMSE", f"{results['rmse']:.3f}")
            cols[2].metric("R²", f"{results['r2']:.3f}")

            st.subheader("True vs Predicted (test set)")
            plot_true_vs_pred(results["y_true"], results["y_pred"])
            st.subheader("Feature importances")
            plot_feature_importances(model, feature_names)

            if gs is not None:
                st.info("GridSearch best params:")
                st.write(gs.best_params_)

            if st.checkbox("Save trained model to disk"):
                joblib.dump(st.session_state["model_bundle"], model_filename)
                st.success(f"Saved model bundle to {model_filename}")

        except Exception as e:
            st.error(f"Training failed: {e}")

# ---------------------------
# Single record prediction
# ---------------------------
st.markdown("---")
st.header("Predict price for a car")

bundle = st.session_state.get("model_bundle", None)
if bundle is None:
    st.info("No trained model. Train or load a model first.")
elif df is not None:
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    reference_year = bundle.get("reference_year", datetime.now().year)

    left, right = st.columns(2)
    with left:
        present_price = st.number_input("Present (showroom) price (in lakhs)", value=5.0, step=0.1, format="%.2f")
        kms_driven = st.number_input("Kilometers driven", value=50000, step=100)
        owner = st.selectbox("Number of previous owners", [0,1,2,3])
        year = st.number_input("Year of manufacture", value=2015, step=1, min_value=1900, max_value=datetime.now().year)
        brands = sorted(df['Brand'].unique())
        models = sorted(df['Model'].unique())
        brand = st.selectbox("Brand", options=brands)
        model_name = st.selectbox("Model", options=models)

    with right:
        fuel_type = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        seller_type = st.selectbox("Seller type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    if st.button("Predict selling price"):
        car_age = reference_year - year
        input_df = pd.DataFrame([{
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Car_Age": car_age,
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type,
            "Transmission": transmission,
            "Brand": brand,
            "Model": model_name
        }])
        try:
            X_input = preprocessor.transform(input_df)
            pred = model.predict(X_input)[0]
            st.success(f"Estimated selling price: ₹ {pred:.2f} lakhs")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------
# Batch prediction
# ---------------------------
st.markdown("---")
st.header("Batch prediction (CSV upload)")

batch_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_uploader")
if batch_file is not None:
    if "model_bundle" not in st.session_state:
        st.error("No trained model available.")
    else:
        try:
            batch_df = pd.read_csv(batch_file)
            required_cols = ["Car_Name","Year","Present_Price","Kms_Driven","Fuel_Type","Seller_Type","Transmission","Owner"]
            missing_cols = [c for c in required_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Batch CSV missing columns: {missing_cols}")
            else:
                batch_df["Car_Name"] = batch_df["Car_Name"].astype(str)
                batch_df["Brand"] = batch_df["Car_Name"].str.split(" ").str[0]
                batch_df["Model"] = batch_df["Car_Name"].str.split(" ", n=1).str[1].fillna("Unknown")

                Xb, yb, _, _ = preprocess_df(batch_df,
                                              reference_year=bundle.get("reference_year", datetime.now().year),
                                              fit_encoder=bundle["preprocessor"])
                pred_vals = bundle["model"].predict(Xb)
                result_df = batch_df.loc[Xb.index].copy()
                result_df["Predicted_Selling_Price"] = pred_vals
                st.markdown("### Predictions (first 10 rows)")
                st.dataframe(result_df.head(10))
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
