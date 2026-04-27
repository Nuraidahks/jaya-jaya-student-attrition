import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KONFIGURASI HALAMAN & CACHING
# ==========================================
st.set_page_config(page_title="Jaya Jaya Institut Analytics", page_icon="🎓", layout="wide")

@st.cache_resource
def load_model():
    # Pastikan file model ada di folder 'model/'
    return joblib.load('model/student_model_tuned.pkl')

model = load_model()

# ==========================================
# 2. HEADER
# ==========================================
st.title("🎓 Jaya Jaya Institut: Student Attrition Early Warning System")
st.markdown("Sistem prediktif untuk mendeteksi potensi *dropout* berdasarkan tren akademik per semester.")
st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Prediksi Individu", "📊 Insight Model"])

# ==========================================
# 3. TAB 1: FORM PREDIKSI
# ==========================================
with tab1:
    st.subheader("Input Profil Mahasiswa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💰 Faktor Ekonomi & Dasar**")
        tuition = st.selectbox("Status SPP (Tuition Up to Date)", options=[1, 0], 
                                format_func=lambda x: "Lancar (1)" if x == 1 else "Menunggak (0)")
        debtor = st.selectbox("Punya Hutang/Tunggakan?", options=[0, 1],
                                format_func=lambda x: "Tidak (0)" if x == 0 else "Ya (1)")
        scholarship = st.selectbox("Penerima Beasiswa?", options=[0, 1],
                                format_func=lambda x: "Tidak (0)" if x == 0 else "Ya (1)")
        admission_grade = st.number_input("Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=120.0)
        age_at_enrollment = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=60, value=20)

    with col2:
        st.markdown("**📚 Performa Akademik Semester 1**")
        sem1_approved = st.number_input("Unit SKS Lulus (Sem 1)", min_value=0, max_value=30, value=6)
        sem1_grade = st.slider("IPK / Nilai (Sem 1)", min_value=0.0, max_value=20.0, value=12.0)
        sem1_evals = st.number_input("Jumlah Evaluasi (Sem 1)", min_value=0, max_value=30, value=8)

    with col3:
        st.markdown("**📚 Performa Akademik Semester 2**")
        sem2_approved = st.number_input("Unit SKS Lulus (Sem 2)", min_value=0, max_value=30, value=6)
        sem2_grade = st.slider("IPK / Nilai (Sem 2)", min_value=0.0, max_value=20.0, value=12.0)
        sem2_evals = st.number_input("Jumlah Evaluasi (Sem 2)", min_value=0, max_value=30, value=8)

    st.markdown("---")
    
    if st.button("🚀 Jalankan Prediksi Algoritma", use_container_width=True):
        
        # --- PROSES FEATURE ENGINEERING (Harus sama dengan notebook training) ---
        # 1. Financial Risk
        if scholarship == 0 and debtor == 1: fin_risk = 2 # Tinggi
        elif scholarship == 1 and debtor == 0: fin_risk = 1 # Rendah
        else: fin_risk = 0 # Menengah
        
        # 2. Total Approved Units
        total_approved = sem1_approved + sem2_approved
        
        # 3. Age Group
        if age_at_enrollment <= 21: age_group = 0 # Reguler
        elif age_at_enrollment <= 30: age_group = 1 # Dewasa Muda
        else: age_group = 2 # Dewasa

        # --- MEMBUAT DATAFRAME INPUT ---
        input_data = {
            'Marital_status': 1, 'Application_mode': 1, 'Application_order': 1, 'Course': 9147,
            'Daytime_evening_attendance': 1, 'Previous_qualification': 1, 'Previous_qualification_grade': 120.0,
            'Mothers_qualification': 1, 'Fathers_qualification': 1, 'Gender': 0, 'International': 0,
            'Displaced': 1, 'Educational_special_needs': 0, 
            'Debtor': debtor, 
            'Tuition_fees_up_to_date': tuition,
            'Scholarship_holder': scholarship,
            'Age_at_enrollment': age_at_enrollment,
            'Curricular_units_1st_sem_grade': sem1_grade,
            'Curricular_units_1st_sem_approved': sem1_approved,
            'Curricular_units_1st_sem_evaluations': sem1_evals,
            'Curricular_units_2nd_sem_grade': sem2_grade,
            'Curricular_units_2nd_sem_approved': sem2_approved,
            'Curricular_units_2nd_sem_evaluations': sem2_evals,
            # Menambahkan kolom yang terlewat
            'Curricular_units_2nd_sem_credited': 0,
            'Curricular_units_2nd_sem_enrolled': 6,
            'Admission_grade': admission_grade,
            'Financial_Risk': fin_risk,
            'Total_Approved_Units': total_approved,
            'Age_Group': age_group
        }
        
        df_input = pd.DataFrame([input_data])

        # --- SINKRONISASI KOLOM OTOMATIS (BULLETPROOF) ---
        try:
            expected_features = model.feature_names_in_
            
            # Jika ada kolom yang diminta model tapi belum ada di df_input, otomatis isi dengan 0
            for col in expected_features:
                if col not in df_input.columns:
                    df_input[col] = 0
                    
            # Pastikan urutan kolom persis sama dengan model
            df_input = df_input[expected_features]
            
            # Eksekusi Prediksi
            prediction = model.predict(df_input)[0]
            probabilities = model.predict_proba(df_input)[0]
            
            # Tampilan Hasil
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
                st.subheader("Keyakinan Model")
                st.write(f"Risiko Dropout: **{probabilities[0]:.1%}**")
                st.progress(float(probabilities[0]))
                st.write(f"Potensi Lulus: **{probabilities[2]:.1%}**")
                st.progress(float(probabilities[2]))
                
        except Exception as e:
            st.error(f"Terjadi kesalahan pada struktur data: {e}")

# ==========================================
# 4. TAB 2: INSIGHT MODEL
# ==========================================
with tab2:
    st.subheader("Fitur Paling Berpengaruh (Feature Importance)")
    
    # Ambil importances langsung dari model
    importances = model.feature_importances_
    feature_names = model.feature_names_in_

    fi_df = pd.DataFrame({'Fitur': feature_names, 'Pengaruh (%)': importances * 100})
    fi_df = fi_df.sort_values(by='Pengaruh (%)', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Pengaruh (%)', y='Fitur', data=fi_df, hue='Pengaruh (%)', palette='viridis', ax=ax, legend=False)
    plt.tight_layout()
    st.pyplot(fig)
