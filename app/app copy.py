# ===============================================
# ✅ SALES & DEPARTURE PREDICTION DASHBOARD (100% MATCH COLAB)
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# -----------------------------
# 1️⃣ Load Model & Data
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'model_global_rf.pkl')
HISTORICAL_PATH = os.path.join(BASE_DIR, '..', 'data', 'data_h-raya_final_clean.csv')

# Load model RF (global, single output: Keberangkatan)
model = joblib.load(MODEL_PATH)

# Load data historis untuk distribusi lead_time
df = pd.read_csv(HISTORICAL_PATH)
df['Keberangkatan'] = pd.to_datetime(df['Keberangkatan'])
df['Tanggal_Beli'] = pd.to_datetime(df['Tanggal_Beli'])

# Load or fit LabelEncoders for Segmentasi and Layanan
from sklearn.preprocessing import LabelEncoder

le_segmentasi = LabelEncoder()
le_segmentasi.fit(df['Segmentasi'])

le_layanan = LabelEncoder()
le_layanan.fit(df['Layanan'])

# -----------------------------
# 2️⃣ UI — Input
# -----------------------------

st.title("📈 Prediksi H- Lebaran (MATCH Colab Flow)")

tahun = st.number_input("Tahun Prediksi", min_value=2026, max_value=2029, value=2026)
trayek = st.selectbox("Pilih Trayek", df['Trayek'].unique())

# -----------------------------
# 3️⃣ Prediksi Keberangkatan (Model A)
# -----------------------------

# Ambil median Segmentasi & Layanan encoder
df_trayek = df[df['Trayek'] == trayek].copy()
seg_str = df_trayek['Segmentasi'].mode().iloc[0]
lay_str = df_trayek['Layanan'].mode().iloc[0]

segmentasi_enc = le_segmentasi.transform([seg_str])[0]
layanan_enc = le_layanan.transform([lay_str])[0]

H_range = np.arange(0, 11)[::-1]  # H0 s/d H10 mundur

X_pred = pd.DataFrame({
    'H_minus': H_range,
    'Tahun': tahun,
    'Segmentasi_Enc': segmentasi_enc,
    'Layanan_Enc': layanan_enc
})

# Pastikan urutan sama
X_pred = X_pred[model.feature_names_in_]

Y_pred = model.predict(X_pred)
Y_pred = np.atleast_2d(Y_pred).T if Y_pred.ndim == 1 else Y_pred

df_keberangkatan = pd.DataFrame({
    'H_minus': H_range,
    'Tanggal_Keberangkatan': pd.Timestamp(f"{tahun}-03-20") - pd.to_timedelta(H_range, unit='D'),
    'Prediksi_Keberangkatan': Y_pred[:, 0].round().astype(int)
})

# -----------------------------
# 4️⃣ Simulasi Pembelian (Model B)
# -----------------------------

# Buat distribusi lead_time historis trayek terpilih
df_trayek['lead_time'] = (df_trayek['Keberangkatan'] - df_trayek['Tanggal_Beli']).dt.days
lead_time_dist = df_trayek[df_trayek['lead_time'] > 0].groupby('lead_time').size().reset_index(name='Count')
lead_time_dist['Proporsi'] = lead_time_dist['Count'] / lead_time_dist['Count'].sum()

# Simulasi pembelian dari prediksi keberangkatan
pembelian_simulasi = []

for _, row in df_keberangkatan.iterrows():
    tanggal_keberangkatan = row['Tanggal_Keberangkatan']
    total_keberangkatan = row['Prediksi_Keberangkatan']

    for _, lt_row in lead_time_dist.iterrows():
        lt = int(lt_row['lead_time'])
        proporsi = lt_row['Proporsi']

        tanggal_beli = tanggal_keberangkatan - pd.Timedelta(days=lt)
        estimasi_beli = total_keberangkatan * proporsi

        pembelian_simulasi.append({
            'Tanggal_Beli': tanggal_beli,
            'Keberangkatan': tanggal_keberangkatan,
            'Lead_Time': lt,
            'Estimasi_Pembelian': estimasi_beli
        })

df_pembelian = pd.DataFrame(pembelian_simulasi)
df_pembelian_final = df_pembelian.groupby('Tanggal_Beli').agg({
    'Estimasi_Pembelian': 'sum'
}).reset_index()
df_pembelian_final['Estimasi_Pembelian'] = df_pembelian_final['Estimasi_Pembelian'].round().astype(int)

# -----------------------------
# 5️⃣ Tampilkan
# -----------------------------

st.subheader("📦 Prediksi Keberangkatan:")
st.dataframe(df_keberangkatan)
st.line_chart(df_keberangkatan.set_index('H_minus')['Prediksi_Keberangkatan'])

st.subheader("💳 Simulasi Pembelian:")
st.dataframe(df_pembelian_final)
st.line_chart(df_pembelian_final.set_index('Tanggal_Beli')['Estimasi_Pembelian'])

st.caption("✅ Alur sama persis dengan Colab: RF ➜ Distribusi ➜ Simulasi.")