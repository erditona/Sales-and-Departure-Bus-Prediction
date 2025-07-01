# ===============================================
# ✅ SALES & DEPARTURE PREDICTION DASHBOARD (Dropdown Mode: H- / H+)
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------
# 1️⃣ Setup & Load Model + Data
# -----------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODEL_HMINUS = os.path.join(BASE_DIR, '..', 'model', 'model_global_rf_hminus.pkl')
MODEL_HPLUS = os.path.join(BASE_DIR, '..', 'model', 'model_global_rf_hplus.pkl')
HMINUS_DATA = os.path.join(BASE_DIR, '..', 'data', 'data_h_minus_final_clean.csv')
HPLUS_DATA = os.path.join(BASE_DIR, '..', 'data', 'data_h_plus_final_clean.csv')

# Load models
model_hminus = joblib.load(MODEL_HMINUS)
model_hplus = joblib.load(MODEL_HPLUS)

# Load datasets
df_hminus = pd.read_csv(HMINUS_DATA)
df_hminus['Keberangkatan'] = pd.to_datetime(df_hminus['Keberangkatan'])
df_hminus['Tanggal_Beli'] = pd.to_datetime(df_hminus['Tanggal_Beli'])

df_hplus = pd.read_csv(HPLUS_DATA)
df_hplus['Keberangkatan'] = pd.to_datetime(df_hplus['Keberangkatan'])
df_hplus['Tanggal_Beli'] = pd.to_datetime(df_hplus['Tanggal_Beli'])

# -----------------------------------------------
# 2️⃣ Single Global LabelEncoders
# -----------------------------------------------

all_segmentasi = pd.concat([df_hminus['Segmentasi'], df_hplus['Segmentasi']]).unique()
all_layanan = pd.concat([df_hminus['Layanan'], df_hplus['Layanan']]).unique()

le_segmentasi = LabelEncoder().fit(all_segmentasi)
le_layanan = LabelEncoder().fit(all_layanan)

# -----------------------------------------------
# 3️⃣ UI: Dropdown Input
# -----------------------------------------------

st.title("🚀 Prediksi Keberangkatan & Pembelian Tiket Lebaran Tahun 2026")

mode = st.selectbox("📌 Pilih Mode Prediksi:", ['H- Lebaran', 'H+ Lebaran'])

if mode == 'H- Lebaran':
    df_use = df_hminus
    model = model_hminus
    h_col = 'H_minus'
    tanggal_lebaran = pd.Timestamp("2026-03-20")
else:
    df_use = df_hplus
    model = model_hplus
    h_col = 'H_plus'
    tanggal_lebaran = pd.Timestamp("2026-03-21")

# tahun = st.number_input("📅 Tahun Prediksi", min_value=2026, max_value=2026, value=2026)
trayek = st.selectbox("🛣️ Pilih Trayek", df_use['Trayek'].unique())

# -----------------------------------------------
# 4️⃣ Model A: Prediksi Keberangkatan
# -----------------------------------------------

df_trayek = df_use[df_use['Trayek'] == trayek].copy()
seg_str = df_trayek['Segmentasi'].mode().iloc[0]
lay_str = df_trayek['Layanan'].mode().iloc[0]

seg_enc = le_segmentasi.transform([seg_str])[0]
lay_enc = le_layanan.transform([lay_str])[0]

H_range = np.arange(0, 11)[::-1]

X_pred = pd.DataFrame({
    h_col: H_range,
    'Tahun': 2026,
    'Segmentasi_Enc': seg_enc,
    'Layanan_Enc': lay_enc
})[model.feature_names_in_]

Y_pred = model.predict(X_pred)
Y_pred = np.atleast_2d(Y_pred).T if Y_pred.ndim == 1 else Y_pred

df_keberangkatan = pd.DataFrame({
    h_col: H_range,
    'Tanggal_Keberangkatan': tanggal_lebaran + pd.to_timedelta(H_range, unit='D') if mode == 'H+ Lebaran'
    else tanggal_lebaran - pd.to_timedelta(H_range, unit='D'),
    'Prediksi_Keberangkatan': Y_pred[:, 0].round().astype(int)
})

# -----------------------------------------------
# 5️⃣ Model B: Simulasi Pembelian
# -----------------------------------------------

df_trayek['lead_time'] = (df_trayek['Keberangkatan'] - df_trayek['Tanggal_Beli']).dt.days
lead_time_dist = df_trayek[df_trayek['lead_time'] > 0].groupby('lead_time').size().reset_index(name='Count')
lead_time_dist['Proporsi'] = lead_time_dist['Count'] / lead_time_dist['Count'].sum()

if lead_time_dist.empty:
    df_use['lead_time'] = (df_use['Keberangkatan'] - df_use['Tanggal_Beli']).dt.days
    lead_time_dist = df_use[df_use['lead_time'] > 0].groupby('lead_time').size().reset_index(name='Count')
    lead_time_dist['Proporsi'] = lead_time_dist['Count'] / lead_time_dist['Count'].sum()

pembelian_simulasi = []
for _, row in df_keberangkatan.iterrows():
    tgl_keb = row['Tanggal_Keberangkatan']
    total_keb = row['Prediksi_Keberangkatan']

    for _, lt_row in lead_time_dist.iterrows():
        lt = int(lt_row['lead_time'])
        proporsi = lt_row['Proporsi']
        tanggal_beli = tgl_keb - pd.Timedelta(days=lt)
        estimasi_beli = total_keb * proporsi

        pembelian_simulasi.append({
            'Tanggal_Beli': tanggal_beli,
            'Keberangkatan': tgl_keb,
            'Lead_Time': lt,
            'Estimasi_Pembelian': estimasi_beli
        })

df_pembelian = pd.DataFrame(pembelian_simulasi)
df_pembelian_final = df_pembelian.groupby('Tanggal_Beli').agg({
    'Estimasi_Pembelian': 'sum'
}).reset_index()
df_pembelian_final['Estimasi_Pembelian'] = df_pembelian_final['Estimasi_Pembelian'].round().astype(int)

# -----------------------------------------------
# 6️⃣ Output
# -----------------------------------------------

st.subheader(f"📦 Prediksi Keberangkatan ({mode}):")
st.dataframe(df_keberangkatan)
st.line_chart(df_keberangkatan.set_index(h_col)['Prediksi_Keberangkatan'])

st.subheader(f"💳 Simulasi Pembelian ({mode}):")
st.dataframe(df_pembelian_final)
st.line_chart(df_pembelian_final.set_index('Tanggal_Beli')['Estimasi_Pembelian'])

st.caption("✅ Alur: Random Forest ➜ Distribusi ➜ Simulasi Pembelian — Single Encoder Global")
