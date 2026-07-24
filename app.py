
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

st.set_page_config(page_title="Accident Risk Predictor", layout="wide")

# UI Styling
st.markdown("""
<style>
    .main { background-color: #fcfcfc; }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3.5em;
        background-color: #f1c40f; color: #2c3e50;
        font-weight: bold; font-size: 1.2em;
    }
    .header-box {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        color: #2c3e50; padding: 2.5rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_train():
    try:
        file_path = 'Dataset Klasifikasi - Copy.csv'
        if not os.path.exists(file_path):
            st.error(f"File '{file_path}' tidak ditemukan di GitHub. Pastikan file ada di repositori.")
            return None, None, None, None, None
            
        df = pd.read_csv(file_path, sep=';', encoding='latin1')
        df.columns = df.columns.str.strip().str.lower()

        df['kecepatan'] = pd.to_numeric(df['kecepatan'], errors='coerce')
        median_speed = df['kecepatan'].median() if not df['kecepatan'].isna().all() else 40
        df['kecepatan'] = df['kecepatan'].fillna(median_speed).astype(int)

        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
        df['age'] = df['age'].replace(['nan', '(blanks)', 'none'], 'tidak diketahui')

        selected_features = ['cuaca', 'tipe cahaya', 'direction', 'kelas jalan', 'geometri jalan', 'tipe jalan', 'kecepatan', 'kecamatan', 'age', 'jenis kelamin', 'jenis kendaraan', 'atribut_keselamatan', 'kepemilikan_sim']

        X_all = pd.get_dummies(df[selected_features], dtype=int)
        y_all = df['jenis luka']

        X_train, _, y_train, _ = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

        smt = SMOTETomek(random_state=42)
        X_train_res, y_train_res = smt.fit_resample(X_train, y_train)

        m_smote = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)
        m_smote.fit(X_train_res, y_train_res)

        m_balanced = BalancedRandomForestClassifier(n_estimators=100, max_depth=10, sampling_strategy='all', random_state=42, n_jobs=1)
        m_balanced.fit(X_train, y_train)

        return df, selected_features, X_all.columns, m_smote, m_balanced
    except Exception as e:
        st.error(f"Error saat training: {e}")
        return None, None, None, None, None

data_bundle = load_data_and_train()
df, selected_features, fitur_model, model_smote, model_balanced = data_bundle

if df is not None:
    st.markdown("<div class='header-box'><h1>Accident Risk Predictor</h1><p>Estimasi Tingkat Keparahan LAKA Gresik</p></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    input_data = {}
    for i, col in enumerate(selected_features):
        target_col = col1 if i < 7 else col2
        if col == 'kecepatan':
            input_data[col] = target_col.slider(col.title(), int(df[col].min()), int(df[col].max()), int(df[col].median()))
        else:
            options = sorted([str(x) for x in df[col].unique()])
            input_data[col] = target_col.selectbox(col.title(), options)

    if st.button("MULAI ANALISIS RISIKO"):
        data = pd.DataFrame([input_data])
        data_enc = pd.get_dummies(data).reindex(columns=fitur_model, fill_value=0)

        probs_balanced = model_balanced.predict_proba(data_enc)[0]
        fatal_idx = list(model_balanced.classes_).index('kecelakaan fatal')

        if probs_balanced[fatal_idx] >= 0.37:
            final_probs = np.zeros(len(model_balanced.classes_))
            final_probs[fatal_idx] = probs_balanced[fatal_idx]
            rem = 1.0 - final_probs[fatal_idx]
            others = [i for i in range(len(model_balanced.classes_)) if i != fatal_idx]
            sum_others = probs_balanced[others].sum() if probs_balanced[others].sum() > 0 else 1
            for idx in others: final_probs[idx] = (probs_balanced[idx] / sum_others) * rem
        else:
            p_smote = model_smote.predict_proba(data_enc)[0]
            final_probs = (p_smote * 0.6) + (probs_balanced * 0.4)

        res = dict(zip(model_balanced.classes_, final_probs))
        st.write("### Hasil Estimasi:")
        st.json(res)
