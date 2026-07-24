
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Accident Risk Predictor", layout="wide")

# CSS Tema Kuning Korporat
st.markdown("""
<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'>
<style>
    .main { background-color: #fcfcfc; }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3.5em;
        background-color: #f1c40f; color: #2c3e50;
        font-weight: bold; border: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1.2em;
    }
    .header-box {
        background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%);
        color: #2c3e50; padding: 2.5rem; border-radius: 15px;
        text-align: center; margin-bottom: 2rem;
    }
    .result-card {
        background: white; border-left: 10px solid #f1c40f;
        padding: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    .stSelectbox label, .stSlider label {
        font-size: 1.2rem !important; font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data_and_train():
    file_path = 'Dataset Klasifikasi - Copy.csv'
    df = pd.read_csv(file_path, sep=';', encoding='latin1')
    df.columns = df.columns.str.strip().str.lower()
    
    # Preprocessing Kecepatan
    df['kecepatan'] = pd.to_numeric(df['kecepatan'], errors='coerce')
    median_speed = df['kecepatan'].median() if not df['kecepatan'].isna().all() else 40
    df['kecepatan'] = df['kecepatan'].fillna(median_speed).astype(int)

    # Preprocessing Kategori
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df['age'] = df['age'].replace(['nan', '(blanks)', 'none'], 'tidak diketahui')

    selected_features = ['cuaca', 'tipe cahaya', 'direction', 'kelas jalan', 'geometri jalan', 'tipe jalan', 'kecepatan', 'kecamatan', 'age', 'jenis kelamin', 'jenis kendaraan', 'atribut_keselamatan', 'kepemilikan_sim']

    X_all = pd.get_dummies(df[selected_features], dtype=int)
    y_all = df['jenis luka']

    # SINKRONISASI: Gunakan pembagian data yang sama dengan Colab
    X_train, _, y_train, _ = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)

    # Model SMOTE (Harus dilatih pada X_train agar persentase sama dengan Colab)
    m_smote = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2, class_weight='balanced', random_state=42)
    m_smote.fit(X_train, y_train)

    # Model Balanced RF
    m_balanced = BalancedRandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, sampling_strategy='all', random_state=42)
    m_balanced.fit(X_train, y_train)

    return df, selected_features, X_all.columns, m_smote, m_balanced

try:
    df, selected_features, fitur_model, model_smote, model_balanced = load_data_and_train()
    st.markdown("<div class='header-box'><h1><i class='fa-solid fa-triangle-exclamation'></i> Accident Risk Predictor</h1><p style='font-size: 1.4em; font-weight: 600;'>Estimasi Tingkat Keparahan Kecelakaan Kab. Gresik</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    input_data = {}
    icon_map = {'direction': '🧭', 'tipe cahaya': '☀️', 'cuaca': '☁️', 'kelas jalan': '🛣️', 'geometri jalan': '📐', 'tipe jalan': '🚦', 'kecepatan': '🏎️', 'kecamatan': '🏙️', 'age': '👤', 'jenis kelamin': '🚻', 'jenis kendaraan': '🚗', 'atribut_keselamatan': '🛡️', 'kepemilikan_sim': '🪪'}

    for i, col in enumerate(selected_features):
        target_col = col1 if i < 7 else col2
        label = f"{icon_map.get(col, '•')} {col.title()}"
        if col == 'kecepatan':
            val_min, val_max, val_default = int(df[col].min()), int(df[col].max()), int(df[col].median())
            if val_min == val_max: val_max = val_min + 1
            input_data[col] = target_col.slider(label, val_min, val_max, val_default)
        else:
            options = sorted([str(x) for x in df[col].unique()])
            input_data[col] = target_col.selectbox(label, options)

    if st.button("MULAI ANALISIS RISIKO"):
        data = pd.DataFrame([input_data])
        data_enc = pd.get_dummies(data).reindex(columns=fitur_model, fill_value=0)

        # Logika Prediksi Hybrid (Sama dengan Colab)
        probs_balanced = model_balanced.predict_proba(data_enc)[0]
        fatal_idx = list(model_balanced.classes_).index('kecelakaan fatal')

        threshold_fatal = 0.37
        if probs_balanced[fatal_idx] >= threshold_fatal:
            final_probs = np.zeros(len(model_balanced.classes_))
            final_probs[fatal_idx] = probs_balanced[fatal_idx]
            rem = 1.0 - final_probs[fatal_idx]
            others = [i for i in range(len(model_balanced.classes_)) if i != fatal_idx]
            for idx in others: final_probs[idx] = (probs_balanced[idx] / probs_balanced[others].sum()) * rem
        else:
            p_smote = model_smote.predict_proba(data_enc)[0]
            final_probs = (p_smote * 0.6) + (probs_balanced * 0.4)

        res = dict(zip(model_balanced.classes_, final_probs))

        st.markdown("<div class='result-card'><h2><i class='fa-solid fa-square-poll-vertical'></i> Hasil Estimasi Keparahan</h2></div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1.5, 1])
        with c1:
            categories = ['kecelakaan berat', 'kecelakaan fatal', 'kecelakaan ringan']
            values = [res.get(cat, 0) for cat in categories]
            labels = [cat.upper() for cat in categories]
            colors = ['#f39c12', '#e74c3c', '#27ae60']

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, 
                   explode=[0.05, 0.05, 0.05], shadow=True, textprops={'weight':'bold', 'fontsize': 14})
            st.pyplot(fig)
        with c2:
            for cat in categories:
                st.metric(label=cat.title(), value=f"{res.get(cat, 0)*100:.1f}%")

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
