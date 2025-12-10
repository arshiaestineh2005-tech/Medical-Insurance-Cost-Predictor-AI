import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="Medical Insurance AI",
    page_icon="🏥",
    layout="wide"
)

# Styling
st.markdown("""
<style>
    .stMetric {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #b3e5fc;
    }
</style>
""", unsafe_allow_html=True)

# 2. Model Training Function
@st.cache_resource
def load_data_and_train():
    try:
        # The csv file must be in the same directory
        df = pd.read_csv('insurance.csv')
    except:
        st.error("File 'insurance.csv' not found! Please ensure it is in the same folder.")
        st.stop()
        
    # Preprocessing
    encoders = {}
    # Convert text columns to numbers
    for col in ['sex', 'smoker', 'region']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # Train XGBoost Regressor
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    
    return model, encoders, df, r2, mae

model, encoders, df, r2_val, mae_val = load_data_and_train()

# 3. Sidebar (User Inputs)
with st.sidebar:
    st.header("📋 Patient Details")
    
    age = st.slider("Age", 18, 100, 30)
    bmi = st.slider("BMI (Body Mass Index)", 15.0, 55.0, 25.0)
    children = st.slider("Number of Children", 0, 5, 0)
    
    st.markdown("---")
    
    sex = st.selectbox("Sex", ['male', 'female'])
    smoker = st.selectbox("Smoker?", ['yes', 'no'])
    region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])
    
    btn = st.button("Calculate Insurance Cost 💰", type="primary")

# 4. Prepare Input Data
input_data = pd.DataFrame({
    'age': [age],
    'sex': [encoders['sex'].transform([sex])[0]],
    'bmi': [bmi],
    'children': [children],
    'smoker': [encoders['smoker'].transform([smoker])[0]],
    'region': [encoders['region'].transform([region])[0]]
})

# 5. Main Dashboard
st.title("🏥 AI Medical Insurance Cost Estimator")
st.markdown(f"Model Accuracy (R2 Score): **{r2_val:.2f}** | Mean Error: **${mae_val:.0f}**")
st.divider()

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("💵 Prediction Result")
    
    if btn:
        prediction = model.predict(input_data)[0]
        
        st.success(f"Estimated Annual Insurance Cost:")
        st.metric(label="Predicted Charges", value=f"${prediction:,.2f}")
        
        # Smart Analysis
        if smoker == 'yes':
            st.warning("⚠️ Insight: Smoking significantly increases the insurance cost.")
        if bmi > 30:
            st.info("ℹ️ Insight: High BMI (Obesity) is a key factor in cost increase.")
            
    else:
        st.info("Enter patient details in the sidebar and click the button to see the estimate.")

with col2:
    st.subheader("📊 Impact of BMI on Charges")
    # Interactive Chart
    fig = px.scatter(
        df, x='bmi', y='charges', color='smoker',
        title="Relationship: BMI, Smoking & Charges",
        color_discrete_map={0: 'blue', 1: 'red'},
        labels={'0': 'Non-Smoker', '1': 'Smoker'}
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Developed with XGBoost & Streamlit")