# 🏥 Medical Insurance Cost Estimator

> **An AI-powered Regression tool to predict medical insurance charges with high precision.**

---

## 📊 Overview

Accurate pricing is the backbone of the insurance industry. This application leverages **Machine Learning (XGBoost Regressor)** to analyze patient biometrics and lifestyle choices, providing instant cost estimations.

Unlike simple calculators, this AI dashboard identifies non-linear relationships (e.g., how smoking amplifies the cost for older patients) and visualizes the financial impact of health factors like BMI.

### Key Features:
* **Precision Pricing:** Uses XGBoost Regression to predict annual charges based on 6 key factors.
* **Interactive Simulation:** Adjust patient details (Age, BMI, Smoking status) via sidebar controls.
* **Smart Insights:** Automatically flags high-risk factors (e.g., Obesity + Smoking).
* **Data Visualization:** Interactive scatter plots showing the correlation between BMI and costs.

---

## 🛠️ Technology Stack

* **Core Logic:** Python 3.x
* **Machine Learning:** XGBoost (Regression), Scikit-Learn
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly Express

---

## 💻 Installation & Usage

To run this project locally, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/arshiaestineh2005-tech/Medical-Insurance-Cost-Prediction-AI.git](https://github.com/arshiaestineh2005-tech/Medical-Insurance-Cost-Prediction-AI.git)
cd Medical-Insurance-Cost-Prediction-AI
**2. Install requirements
pip install -r requirements.txt
**3. Run the App
streamlit run insurance_app.py
The application will open in your browser at http://localhost:8501.


📂 File Structure
 ├── insurance_app.py     # Main application code
├── insurance.csv        # Dataset source
├── requirements.txt     # Project dependencies
├── dashboard.png        # App Screenshot
└── README.md            # Documentation

📬 Contact
Arshia Estineh Machine Learning Engineer | AI Solutions

📧 Email: arshiaestineh2005@icloud.com

🐙 GitHub: arshiaestineh2005-tech

Built with ❤️ using XGBoost & Streamlit.