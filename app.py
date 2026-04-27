import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN & CACHING
# ==========================================
# Layout 'wide' membuat tampilan memenuhi layar (fullscreen)
st.set_page_config(page_title="Jaya Jaya Institut Analytics", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load('model/student_model_tuned.pkl')

model = load_model()

# ==========================================
# 2. HEADER & TABS
# ==========================================
st.title("🎓 Jaya Jaya Institut: Student Attrition Early Warning System")
st.markdown("Sistem prediktif berbasis *Machine Learning* untuk mendeteksi potensi *dropout* mahasiswa secara dini.")
st.markdown("---")

# Membuat dua tab utama
tab1, tab2 = st.tabs(["🔍 Prediksi Individu", "📊 Insight Model"])

# ==========================================
# 3. TAB 1: FORM PREDIKSI (KOLOM RAPI)
# ==========================================
with tab1:
    st.subheader("Input Profil Mahasiswa")
    
    # Membagi form menjadi 3 kolom agar layout lebih lebar dan profesional
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💰 Faktor Finansial**")
        tuition = st.selectbox("Status SPP (Tuition Up to Date)", options=[1, 0], format_func=lambda x: "Lancar (1)" if x == 1 else "Menunggak (0)")
        fin_risk = st.selectbox("Tingkat Risiko Finansial", options=[0, 1, 2], format_func=lambda x: "Menengah (0)" if x==0 else ("Rendah (1)" if x==1 else "Tinggi (2)"))
    
    with col2:
        st.markdown("**📚 Performa Akademik (Tahun 1)**")
        total_units = st.slider("Total SKS Lulus", min_value=0, max_value=40, value=10)
        avg_grade = st.slider("Rata-rata IPK / Nilai", min_value=0.0, max_value=20.0, value=12.0)
        
    with col3:
        st.markdown("**👤 Demografi Awal**")
        admission = st.number_input("Nilai Masuk (Admission Grade)", min_value=90.0, max_value=200.0, value=120.0)
        age_group = st.selectbox("Kelompok Usia Masuk", options=[0, 1, 2], format_func=lambda x: "Reguler/Remaja (0)" if x==0 else ("Dewasa Muda (1)" if x==1 else "Dewasa (2)"))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tombol menggunakan lebar penuh
    if st.button("🚀 Jalankan Prediksi Algoritma", use_container_width=True):
        
        # Dictionary 25 Fitur Input
        input_dict = {
            'Marital_status': 1, 'Application_mode': 1, 'Application_order': 1, 'Course': 9085, 
            'Daytime_evening_attendance': 1, 'Previous_qualification': 1, 'Previous_qualification_grade': 130.0, 
            'Mothers_qualification': 19, 'Fathers_qualification': 19, 'Admission_grade': admission, 
            'Displaced': 1, 'Educational_special_needs': 0, 'Debtor': 0, 'Tuition_fees_up_to_date': tuition, 
            'Gender': 0, 'Scholarship_holder': 0, 'International': 0, 
            'Curricular_units_1st_sem_evaluations': 8, 'Curricular_units_2nd_sem_credited': 0, 
            'Curricular_units_2nd_sem_enrolled': 6, 'Curricular_units_2nd_sem_evaluations': 8, 
            'Financial_Risk': fin_risk, 'Total_Approved_Units': total_units, 
            'Average_Grade_1st_Year': avg_grade, 'Age_Group': age_group
        }
        df_input = pd.DataFrame([input_dict])
        
        # Eksekusi Prediksi & Probabilitas
        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0] # Mengambil skor persentase
        
        st.markdown("---")
        
        # Menampilkan Hasil Akhir dan Probabilitas Bersebelahan
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.subheader("Hasil Keputusan")
            if prediction == 0:
                st.error("🔴 DROPOUT")
                st.caption("Peringkat Risiko: KRITIS")
            elif prediction == 2:
                st.success("🟢 GRADUATE")
                st.caption("Peringkat Risiko: AMAN")
            else:
                st.warning("🟡 ENROLLED")
                st.caption("Peringkat Risiko: WASPADA")
                
        with res_col2:
            st.subheader("Keyakinan Model (Probabilitas)")
            # Visualisasi Progress Bar untuk persentase
            st.write(f"Risiko Dropout: **{probabilities[0]:.1%}**")
            st.progress(float(probabilities[0]))
            
            st.write(f"Potensi Tetap Aktif (Enrolled): **{probabilities[1]:.1%}**")
            st.progress(float(probabilities[1]))
            
            st.write(f"Potensi Lulus (Graduate): **{probabilities[2]:.1%}**")
            st.progress(float(probabilities[2]))


# ==========================================
# 4. TAB 2: INSIGHT MODEL (VISUALISASI)
# ==========================================
with tab2:
    st.subheader("Bagaimana Algoritma Berpikir?")
    st.write("Grafik di bawah ini menunjukkan variabel mana saja yang memiliki bobot terbesar dalam menentukan status mahasiswa. Visualisasi ini di-generate langsung secara dinamis dari model Anda.")
    
    # Mengekstrak Feature Importance dari Random Forest
   # Ambil nilai pengaruh dari model
    importances = model.feature_importances_

    # (Solusi Otomatis) Ambil nama fitur langsung dari model yang disimpan!
    feature_names = model.feature_names_in_

    # Gabungkan menjadi DataFrame
    fi_df = pd.DataFrame({
        'Fitur': feature_names, 
        'Pengaruh (%)': importances * 100
    })

    
    # Membuat DataFrame untuk Plotting
    fi_df = pd.DataFrame({'Fitur': feature_names, 'Pengaruh (%)': importances * 100})
    fi_df = fi_df.sort_values(by='Pengaruh (%)', ascending=False).head(10)
    
    # Plotting menggunakan Seaborn
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='Pengaruh (%)', y='Fitur', data=fi_df, palette='magma', ax=ax)
    ax.set_title("Top 10 Fitur Paling Berpengaruh", fontweight='bold')
    st.pyplot(fig)
