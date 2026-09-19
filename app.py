"""
app.py
------
Streamlit UI for the Medical Cost & Risk Prediction system, built from the
Medical_Cost_&_Risk_Analysis_System notebook.

Loads the pipelines saved by train_model.py (which already contain the
ColumnTransformer preprocessing, so raw categorical values like
sex='male' can be passed straight in — no manual encoding here).

Run train_model.py once first (it needs insurance.csv in the same folder),
then:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Medical Cost & Risk Prediction", layout="wide")
st.title("Medical Cost & Risk Prediction App")

REQUIRED_FILES = ['pipe_reg.pkl', 'classification_models.pkl', 'model_metadata.pkl']
missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(
        "Missing model files: " + ", ".join(missing) +
        "\n\nRun `python train_model.py` first (with insurance.csv in this folder) "
        "to generate them."
    )
    st.stop()


@st.cache_resource
def load_artifacts():
    pipe_reg = joblib.load('pipe_reg.pkl')
    trained_models = joblib.load('classification_models.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return pipe_reg, trained_models, metadata


pipe_reg, trained_models, metadata = load_artifacts()

best_model_name = metadata['best_model_name']
eval_df = metadata['eval_df']
best_model_instance = trained_models[best_model_name]

with st.expander("Model performance (from training run)"):
    st.dataframe(eval_df, use_container_width=True)
    st.bar_chart(eval_df.set_index('Model')['Accuracy'])
    st.caption(f"Best classification model selected: **{best_model_name}**")

# --- Sidebar inputs ---
st.sidebar.header("Patient Input Features")


def user_input_features():
    age = st.sidebar.slider('Age', 18, 64, 30)
    sex = st.sidebar.selectbox('Sex', ['female', 'male'])
    bmi = st.sidebar.slider('BMI', 15.0, 53.0, 25.0)
    children = st.sidebar.slider('Children', 0, 5, 0)
    smoker = st.sidebar.selectbox('Smoker', ['no', 'yes'])
    region = st.sidebar.selectbox('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

    # Raw values — the saved pipelines' ColumnTransformer does the
    # one-hot encoding and scaling internally.
    data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region,
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# --- Predictions ---
st.subheader('Prediction Results')

predicted_cost = pipe_reg.predict(input_df)[0]
st.write(f"**Predicted Medical Cost:** ${predicted_cost:,.2f}")

predicted_class = best_model_instance.predict(input_df)[0]
risk_probability = best_model_instance.predict_proba(input_df)[0][predicted_class] * 100
risk_level = "High Risk" if predicted_class == 1 else "Low Risk"

st.write(f"**Risk Classification ({best_model_name} Model):** {risk_level}")
st.write(f"**Model Confidence:** {risk_probability:.2f}%")

st.subheader('Insurance System Recommendation')
if predicted_class == 1:
    st.error("⚠️ **[Financial Alert]:** This client is classified as High Risk.")
    st.warning("**Recommendation:** Review the predicted cost and consider appropriate pricing based on business rules.")
else:
    st.success("✅ **[Lower Cost Status]:** This client is classified as Low Risk.")
    st.info("**Recommendation:** The client falls into the lower-cost classification according to the model.")

# --- Batch prediction ---
st.subheader("Batch Prediction (upload CSV)")
st.caption("CSV must have columns: age, sex, bmi, children, smoker, region")

uploaded_file = st.file_uploader("Upload patients CSV", type=['csv'])
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    batch_costs = pipe_reg.predict(batch_df)
    batch_preds = best_model_instance.predict(batch_df)
    batch_probs = best_model_instance.predict_proba(batch_df)

    results = batch_df.copy()
    results['Predicted Cost'] = batch_costs
    results['Risk Classification'] = ["High Risk" if p == 1 else "Low Risk" for p in batch_preds]
    results['Model Confidence (%)'] = [batch_probs[i][batch_preds[i]] * 100 for i in range(len(batch_preds))]

    st.dataframe(results, use_container_width=True)
    st.download_button(
        "Download results as CSV",
        results.to_csv(index=False).encode('utf-8'),
        "batch_predictions.csv",
        "text/csv"
    )
