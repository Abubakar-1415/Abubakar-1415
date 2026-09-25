import json
from pathlib import Path
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
REPORTS = BASE / "reports"

st.set_page_config(
    page_title="Healthcare Resource & Patient Analytics Portal",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .title {font-size: 2.1rem; font-weight: 700; margin-bottom: 0.2rem;}
    .subtitle {color: #667085; margin-bottom: 1rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.2);
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def require_file(path):
    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()


def get_model(artifact):
    return artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact


def get_encoder(artifact):
    return artifact.get("label_encoder") if isinstance(artifact, dict) else None


def get_preprocessor(artifact):
    return artifact.get("preprocessor") if isinstance(artifact, dict) else None


def prepare_input(df, artifact):
    if isinstance(artifact, dict) and artifact.get("feature_columns"):
        cols = list(artifact["feature_columns"])
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                "Model expects missing feature(s): " + ", ".join(missing)
            )
        return df[cols].copy()
    return df.copy()


def report(title, filename):
    st.subheader(title)
    path = REPORTS / filename
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.info(f"{filename} was not found. Generate it from the ML notebook.")


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_models():
    paths = [
        ART / "classification_model.joblib",
        ART / "regression_model.joblib",
        ART / "kmeans_cluster_model.joblib",
    ]
    for p in paths:
        require_file(p)

    return (
        joblib.load(paths[0]),
        joblib.load(paths[1]),
        joblib.load(paths[2]),
    )


@st.cache_data
def load_data():
    path = ART / "dashboard_data.csv"
    require_file(path)
    df = pd.read_csv(path)

    for col in ["Date of Admission", "Discharge Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


@st.cache_data
def load_metadata():
    path = ART / "metadata.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    cls_art, reg_art, clu_art = load_models()
    data = load_data()
    metadata = load_metadata()

    cls_model = get_model(cls_art)
    reg_model = get_model(reg_art)
    clu_model = get_model(clu_art)

    label_encoder = get_encoder(cls_art)
    clu_preprocessor = get_preprocessor(clu_art)

except Exception as e:
    st.error("The application could not load the saved ML artifacts.")
    st.exception(e)
    st.stop()

st.markdown(
    '<div class="title">🏥 Healthcare Resource & Patient Analytics Portal</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">End-to-end Machine Learning Analytics Dashboard</div>',
    unsafe_allow_html=True,
)
st.caption("Classification • Regression • Patient Segmentation • Model Evaluation")

st.warning(
    "Analytics demonstration only. The Test Results prediction is not a medical diagnosis "
    "and must not be used for clinical treatment decisions."
)

st.sidebar.header("👤 Patient Prediction")
st.sidebar.caption("Enter patient information and run the trained models.")


def options(column):
    if column not in data.columns:
        return ["Not available"]
    return sorted(data[column].dropna().astype(str).unique().tolist())


age = st.sidebar.slider("Age", 1, 100, 45)
gender = st.sidebar.selectbox("Gender", options("Gender"))
blood = st.sidebar.selectbox("Blood Type", options("Blood Type"))
condition = st.sidebar.selectbox("Medical Condition", options("Medical Condition"))
insurance = st.sidebar.selectbox("Insurance Provider", options("Insurance Provider"))
admission_type = st.sidebar.selectbox("Admission Type", options("Admission Type"))
medication = st.sidebar.selectbox("Medication", options("Medication"))
admission_date = st.sidebar.date_input("Date of Admission", value=date(2024, 1, 15))

default_bill = (
    float(data["Billing Amount"].median())
    if "Billing Amount" in data.columns
    else 0.0
)
billing_amount = st.sidebar.number_input(
    "Billing Amount",
    min_value=0.0,
    value=default_bill,
    step=100.0,
)

run_prediction = st.sidebar.button(
    "🚀 Run Prediction",
    type="primary",
    use_container_width=True,
)


patient = pd.DataFrame([{
    "Gender": gender,
    "Blood Type": blood,
    "Medical Condition": condition,
    "Insurance Provider": insurance,
    "Admission Type": admission_type,
    "Medication": medication,
    "Age": age,
    "Admission Year": admission_date.year,
    "Admission Month": admission_date.month,
    "Admission DayOfWeek": admission_date.weekday(),
    "Admission Quarter": ((admission_date.month - 1) // 3) + 1,
}])

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Patients", f"{len(data):,}")

if "Length of Stay" in data.columns:
    c2.metric("Average Length of Stay", f"{data['Length of Stay'].mean():.1f} days")
else:
    c2.metric("Average Length of Stay", "N/A")

if "Billing Amount" in data.columns:
    c3.metric("Average Billing", f"${data['Billing Amount'].mean():,.0f}")
else:
    c3.metric("Average Billing", "N/A")

if "Test Results" in data.columns:
    abnormal = data["Test Results"].eq("Abnormal").mean() * 100
    c4.metric("Abnormal Test Results", f"{abnormal:.1f}%")
else:
    c4.metric("Abnormal Test Results", "N/A")

tab1, tab2, tab3 = st.tabs([
    "📊 Overview",
    "🤖 Patient Prediction",
    "🔎 Model Evidence",
])

with tab1:
    st.subheader("Healthcare Dataset Overview")

    left, right = st.columns(2)

    with left:
        st.write("**Medical Condition Distribution**")
        if "Medical Condition" in data.columns:
            st.bar_chart(data["Medical Condition"].value_counts())

    with right:
        st.write("**Admission Type Distribution**")
        if "Admission Type" in data.columns:
            st.bar_chart(data["Admission Type"].value_counts())

    if {"Medical Condition", "Length of Stay"}.issubset(data.columns):
        st.write("**Average Length of Stay by Medical Condition**")
        los = (
            data.groupby("Medical Condition")["Length of Stay"]
            .mean()
            .sort_values(ascending=False)
            .round(2)
        )
        st.bar_chart(los)

    left, right = st.columns(2)

    with left:
        st.write("**Test Result Distribution**")
        if "Test Results" in data.columns:
            st.bar_chart(data["Test Results"].value_counts())

    with right:
        st.write("**Age Distribution**")
        if "Age" in data.columns:
            st.line_chart(data["Age"].value_counts().sort_index())

    with st.expander("View Dataset Sample"):
        st.dataframe(data.head(20), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Individual Patient Prediction")

    if not run_prediction:
        st.info("Configure the patient in the sidebar and click 🚀 Run Prediction.")

    if run_prediction:
        try:
            # Classification
            cls_input = prepare_input(patient, cls_art)
            probabilities = cls_model.predict_proba(cls_input)[0]
            pred_index = int(np.argmax(probabilities))

            if label_encoder is not None:
                predicted_label = label_encoder.inverse_transform([pred_index])[0]
                classes = list(label_encoder.classes_)
            else:
                classes = list(cls_model.classes_)
                predicted_label = classes[pred_index]

            # Regression
            reg_input = prepare_input(patient, reg_art)
            predicted_los = max(0.0, float(reg_model.predict(reg_input)[0]))

            # Clustering
            cluster_input = pd.DataFrame([{
                "Age": age,
                "Billing Amount": billing_amount,
                "Length of Stay": max(1.0, predicted_los),
                "Gender": gender,
                "Medical Condition": condition,
                "Admission Type": admission_type,
                "Medication": medication,
            }])

            if clu_preprocessor is not None:
                cluster_input = clu_preprocessor.transform(cluster_input)

            cluster_id = int(clu_model.predict(cluster_input)[0])

            # Results
            a, b, c = st.columns(3)
            a.metric("Predicted Test Result", str(predicted_label))
            b.metric("Estimated Length of Stay", f"{predicted_los:.1f} days")
            c.metric("Patient Segment", f"Cluster {cluster_id}")

            st.divider()

            st.subheader("Classification Probability")

            probability_df = pd.DataFrame({
                "Class": classes,
                "Probability (%)": np.round(probabilities * 100, 2),
            })

            st.bar_chart(
                probability_df.set_index("Class")["Probability (%)"]
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("View Patient Input"):
                st.dataframe(
                    patient,
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander("View Model Information"):
                st.write("Classification:", type(cls_model).__name__)
                st.write("Regression:", type(reg_model).__name__)
                st.write("Clustering:", type(clu_model).__name__)

            st.warning(
                "The classification target is the dataset's Test Results field. "
                "This is an ML analytics output, not a medical diagnosis."
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)

with tab3:
    st.subheader("Model Evaluation & Evidence")

    report("ROC Curve & AUC", "classification_roc.png")
    report("Regression — Actual vs Predicted", "regression_actual_vs_predicted.png")
    report("Patient Segmentation — PCA", "kmeans_pca.png")

    if (REPORTS / "shap_global_importance.png").exists():
        report("SHAP Feature Importance", "shap_global_importance.png")

    with st.expander("Project Metadata"):
        if metadata:
            st.json(metadata)
        else:
            st.info("metadata.json was not found.")


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Healthcare ML Project • Classification • Regression • Clustering • "
    "Model Evaluation • For analytics/educational demonstration only"
)
