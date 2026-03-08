import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. 加载模型
model = joblib.load('k09_model.pkl')

# 定义字典映射（保持不变）
surgery_options = {
    1: "Neurosurgery", 2: 'Cardiac and Major Vascular Surgery', 3: 'Vascular Surgery', 
    4: 'Thoracic Surgery', 5: "Gastrointestinal Surgery", 6: "Hepatobiliary and Pancreatic Surgery", 
    7: 'Urologic Surgery', 8: 'Gynecology', 9: 'Endocrine Surgery', 10: "Ophthalmology",
    11: "Otolaryngology", 12: 'Orthopedic Surgery', 13: 'Plastic and Soft Tissue Surgery', 
    14: 'General & Trauma Surgery'
}
gender_options = {1: "Male", 2: 'Female'}

feature_names = ["icd10_pcs","op_time","creatinine","eGFR","sex","total_bilirubin","bun","albumin","chloride"]

# Streamlit UI
st.title("Prediction Model for Postoperative Acute Kidney Injury")

# 输入字段
icd10_pcs = st.selectbox("Surgery type:", options=list(surgery_options.keys()), format_func=lambda x: surgery_options[x])
op_time = st.number_input("Operative time(min):", min_value=0, max_value=3000, value=300)
creatinine = st.number_input("Creatinine(mg/dL):", min_value=0.00, max_value=6.00, value=3.00, format="%.2f")
eGFR = st.number_input("eGFR(ml/min/1.73m²):", min_value=0.00, max_value=200.00, value=90.00, format="%.2f")
sex = st.selectbox("Gender:", options=list(gender_options.keys()), format_func=lambda x: gender_options[x])
total_bilirubin = st.number_input("Total bilirubin(mg/dL):", min_value=0.0, max_value=10.0, value=3.0, format="%.1f")
bun = st.number_input("Blood urea nitrogen(mg/dL):", min_value=0, max_value=100, value=25)
albumin = st.number_input("Albumin(g/dL):", min_value=0.0, max_value=10.0, value=2.0, format="%.1f")
chloride = st.number_input("Chloride(mmol/L):", min_value=0, max_value=200, value=60)

feature_values = [icd10_pcs, op_time, creatinine, eGFR, sex, total_bilirubin, bun, albumin, chloride]
features = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # 2. 获取正类概率 (AKI 风险概率)
    # 通常 predict_proba 返回 [p0, p1]，我们要的是 p1 (AKI 概率)
    predicted_proba_raw = model.predict_proba(features)[0]
    aki_probability = predicted_proba_raw[1] 
    
    # 3. 应用最优阈值 F1* = 0.227007
    best_threshold = 0.227007
    
    if aki_probability >= best_threshold:
        predicted_class = 1
        status_text = "High Risk"
        color = "red"
    else:
        predicted_class = 0
        status_text = "Low Risk"
        color = "green"

    # 显示判定结果
    st.subheader(f"Prediction Result: :{color}[{status_text}]")
    st.write(f"**Calculated AKI Probability:** {aki_probability*100:.1f}%")
    st.write(f"*(Based on optimized threshold: {best_threshold:.3f})*")

    # 4. 根据判定结果给出建议
    if predicted_class == 0:
        advice = """
        ✅ **AKI is unlikely**
        - Continue with the current treatment as prescribed.
        - Regularly monitor renal function and urine protein levels.
        - Maintain a low-salt, low-fat diet.
        """
    else:
        advice = """
        ⚠️ **AKI risk is high**
        - Consult with the attending physician about adjustments to the treatment plan.
        - Consider additional therapeutic interventions or protective measures.
        - Closely monitor urine output and biomarkers.
        """
    st.markdown(advice)

    # 5. SHAP 可解释性分析
    st.divider()
    st.subheader("Feature Contribution (SHAP Explanation)")
    
    explainer = shap.TreeExplainer(model)
    # 计算当前样本的 SHAP 值
    shap_val = explainer.shap_values(features)
    
    # 兼容处理：GBM 等模型在某些版本下返回列表，某些返回数组
    if isinstance(shap_val, list):
        # 取正类（AKI）对应的 SHAP 值
        current_shap_values = shap_val[1]
    else:
        current_shap_values = shap_val

    # 绘图
    fig, ax = plt.subplots()
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        current_shap_values[0],
        features,
        matplotlib=True,
        show=False
    )
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=200)
    st.image("shap_force_plot.png")
