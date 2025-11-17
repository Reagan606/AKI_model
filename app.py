import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# Load the model
model = joblib.load('k09_model.pkl')

# Define feature options
icd10_pcs = {
    1: "Neurosurgery", 
    2: 'Cardiac and Major Vascular Surgery',    
    3: 'Vascular Surgery', 
    4: 'Thoracic Surgery',
    5: "Gastrointestinal Surgery",
    6: "Hepatobiliary and Pancreatic Surgery", 
    7: 'Urologic Surgery',    
    8: 'Gynecology', 
    9: 'Endocrine Surgery',
    10: "Ophthalmology",
    11: "Otolaryngology", 
    12: 'Orthopedic Surgery',    
    13: 'Plastic and Soft Tissue Surgery', 
    14: 'General & Trauma Surgery'
}



sex = {
    1: "Male", 
    2: 'Female'
}




# Define feature names (确保与模型训练时的特征顺序一致)
feature_names = ["icd10_pcs","op_time","creatinine","eGFR","sex","total_bilirubin","bun","albumin","chloride"]




# Streamlit user interface
st.title("Prediction Model for Postoperative Acute Kidney Injury")

# 输入字段



icd10_pcs = st.selectbox("Surgery type:", options=list(icd10_pcs.keys()), format_func=lambda x: icd10_pcs[x])


op_time = st.number_input("Operative time(min):", min_value=0, max_value=3000, value=300)

creatinine = st.number_input("Creatinine(mg/dL):", min_value=0.00, max_value=6.00, value=3.00, format="%.2f")

eGFR = st.number_input("eGFR(ml/min/1.73m²):", min_value=0.00, max_value=200.00, value=90.00, format="%.2f")

sex = st.selectbox("Gender:", options=list(sex.keys()), format_func=lambda x: sex[x])

total_bilirubin = st.number_input("Total bilirubin(mg/dL):", min_value=0.0, max_value=10.0, value=3.0, format="%.1f")

bun = st.number_input("Blood urea nitrogen(mg/dL):", min_value=0, max_value=100, value=25)

albumin = st.number_input("Albumin(g/dL):", min_value=0.0, max_value=10.0, value=2.0, format="%.1f")

chloride= st.number_input("Chloride(mmol/L):", min_value=0, max_value=200, value=60)


# 确保特征顺序正确
feature_values = [icd10_pcs, op_time, creatinine, eGFR, sex, total_bilirubin, bun, albumin, chloride]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
        
    # 显示结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100




    if predicted_class == 1:
        advice = (
            f"✅ Prediction result: Postoperative AKI is unlikely (Probability: {probability:.1f}%)"
            """
            **Recommendations:**
            - Continue with the current treatment as prescribed
            - Regularly monitor renal function and urine protein levels
            - Maintain a low-salt, low-fat diet
            """
        )
    else:
        advice = (
            f"⚠️ Prediction result: Postoperative AKI risk is high (Probability: {probability:.1f}%)"
            """
            **Recommendations:**
            - Consult with your doctor about possible adjustments to the treatment plan
            - Consider additional therapeutic interventions
            - Closely monitor changes in your condition
            """
        )

    st.write(advice)     
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)


    st.image("shap_force_plot.png")

