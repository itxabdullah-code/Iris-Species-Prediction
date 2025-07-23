import streamlit as st
import numpy as np
import pickle

# --------------------------------------------------
# 📦 Load pre‑trained models (your actual saved model filenames)
# --------------------------------------------------
logistic_model = pickle.load(open("L_model", "rb"))
svm_model      = pickle.load(open("svm_model", "rb"))
dt_model       = pickle.load(open("DT_model", "rb"))
rf_model       = pickle.load(open("RF_model", "rb"))

# --------------------------------------------------
# 🖥️  Streamlit page configuration & CSS styling
# --------------------------------------------------
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="centered")

st.markdown(
    """
    <style>
    body  {background-color:#F6F6F9;}
    h1    {background:linear-gradient(to right,#0d47a1,#1976d2,#42a5f5);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            font-size:42px; text-align:center; margin-bottom:10px;}
    .stButton>button{background-color:#0d47a1;color:white;padding:0.6rem 1.2rem;border:none;border-radius:8px;font-size:16px;}
    .stButton>button:hover{background-color:#1976d2;}
    .sidebar .sidebar-content{background-color:#E3F2FD;padding:1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 🏷️  App title & description
# --------------------------------------------------
st.markdown("""<h1>Iris Species Prediction</h1>""", unsafe_allow_html=True)
st.markdown("### Enter the Iris flower measurements below to predict its species.")

# --------------------------------------------------
# 🔧 Sidebar: choose model
# --------------------------------------------------
st.sidebar.title("Model Selection")
model_option = st.sidebar.radio("Select Model:", (
    "Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest"))

st.sidebar.markdown("---")
st.sidebar.write("Models trained on the classic Iris dataset (150 samples, 3 species).")

# --------------------------------------------------
#  User inputs for flower measurements
# --------------------------------------------------
st.markdown("---")
st.markdown("### 🌿 Flower Measurements (cm)")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8, 0.1)
sepal_width  = st.slider("Sepal Width",  2.0, 4.5, 3.0, 0.1)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0, 0.1)
petal_width  = st.slider("Petal Width",  0.1, 2.5, 1.3, 0.1)

# --------------------------------------------------
# 🔢 Prepare data for prediction
# --------------------------------------------------
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# --------------------------------------------------
# 🚀 Prediction button
# --------------------------------------------------
if st.button("🌸 Predict Species"):
    if model_option == "Logistic Regression":
        pred = logistic_model.predict(input_data)
    elif model_option == "Support Vector Machine":
        pred = svm_model.predict(input_data)
    elif model_option == "Decision Tree":
        pred = dt_model.predict(input_data)
    else:
        pred = rf_model.predict(input_data)

    species = species_map.get(int(pred[0]), "Unknown")
    st.markdown("---")
    st.success(f"🔮 Predicted Species: **{species}**")