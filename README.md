# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech (Human Resources Division)

Link Looker Studio: https://datastudio.google.com/reporting/140be2c6-1cb8-42c4-ba1c-1b8d2860a5dc
Akses Aplikasi Streamlit: https://jaya-jaya-student-attrition-nuraidahks-prototype.streamlit.app/


## Business Understanding

Jaya Jaya Institut saat ini menghadapi tantangan berupa tingginya angka mahasiswa yang putus kuliah (dropout), yakni mencapai lebih dari 32%. Hal ini berisiko menurunkan reputasi institusi dan mengganggu stabilitas pendapatan kampus. Oleh karena itu, institusi membutuhkan sebuah Sistem Deteksi Dini (Early Warning System) berbasis Machine Learning yang mampu mengidentifikasi mahasiswa berisiko tinggi secara akurat (dengan target Recall >75%) berdasarkan data akademik dan finansial di tahun pertama. Sistem ini diintegrasikan ke dalam dashboard interaktif agar manajemen dapat melakukan intervensi preventif, dengan target menurunkan tingkat dropout menjadi maksimal 20% pada akhir 1 tahun akademik ke depan.

### Permasalahan Bisnis

1. Faktor-faktor apa saja (baik dari segi demografi, akademik, maupun finansial) yang paling membedakan mahasiswa yang berhasil lulus (Graduate) dengan mahasiswa yang putus kuliah (Dropout)?

2. Apakah kita dapat membangun model Machine Learning yang mampu memprediksi status akhir mahasiswa (Dropout, Enrolled, Graduate) secara dini menggunakan data tahun pertama pendaftaran mereka?

3. Berdasarkan insight dan hasil prediksi dari model, rekomendasi intervensi atau tindakan preventif seperti apa yang dapat diterapkan oleh manajemen Jaya Jaya Institut untuk menekan angka dropout?

### Cakupan Proyek

1. Business Understanding (Pemahaman Bisnis)
Tingginya tingkat mahasiswa putus kuliah (dropout) di Jaya Jaya Institut yang mencapai lebih dari 32%, mengancam reputasi akademik dan stabilitas finansial institusi. Untuk itu, proyek ini bertujuan membangun sistem deteksi dini (Early Warning System) untuk mengidentifikasi mahasiswa yang memiliki probabilitas tinggi untuk dropout berdasarkan profil awal dan performa akademik tahun pertama.

2. Data Understanding (Pemahaman Data)
Eksplorasi Data (EDA) dengan menganalisis dataset yang terdiri dari 4.424 rekam jejak mahasiswa. Dalam identifikasi pola ditemukan bahwa nilai 0 pada semester 1 dan 2 bukanlah sebuah anomali (error), melainkan sinyal kuat dari mahasiswa yang tidak mengikuti ujian atau telah keluar. Untuk itu, dilakukan analisis korelasi dengan memetakan hubungan antara faktor akademik (SKS, IPK, program studi), faktor demografi (usia, gender, status perkawinan), dan faktor finansial (status beasiswa, tunggakan SPP) terhadap status kelulusan mahasiswa.

3. Data Preparation (Persiapan Data)
Pada data preparation dilakukan feature engineering untuk membuat variabel turunan yang dapat memperkuat daya prediksi model. feature engineering meliput financial risk, total approved units, dan age group.
a. Financial_Risk, indikator risiko keuangan gabungan antara status hutang dan kepemilikan beasiswa.
b. Total_Approved_Units, total akumulasi beban SKS yang berhasil diselesaikan di tahun pertama.
c. Age_Group, pengelompokan usia pendaftar menjadi Reguler, Dewasa Muda, dan Dewasa.

Selain itu juga dilakukan Feature Selection dengan mereduksi dimensi dengan membuang fitur makroekonomi (GDP, Tingkat Inflasi, Pengangguran) dan fitur dengan kardinalitas tinggi yang dapat memicu noise pada model. Terakhir, Data Transformation dilakukan untuk mengubah data kategorikal dan teks (Dropout, Enrolled, Graduate) menjadi format numerik menggunakan teknik Label Encoding.

4. Modeling (Pemodelan Machine Learning)
a. Pemilihan algoritma menggunakan Random Forest Classifier yang tangguh dalam menangani data non-linear dan interaksi multivariabel.
b. Hyperparameter Tuning dilakukan untuk mengoptimalkan algoritma menggunakan GridSearchCV untuk menguji puluhan kombinasi parameter secara otomatis.
c. Penanganan Imbalanced Data menggunakan parameter class_weight='balanced' untuk memastikan algoritma tetap adil dan akurat dalam mendeteksi kelas minoritas (Enrolled) tanpa harus membuang data berharga dari kelas mayoritas (Graduate).

5. Evaluation (Evaluasi Model)
a. Pengujian Metrik untuk mengevaluasi model menggunakan data uji (Test Set) dengan hasil tingkat akurasi akhir mencapai ~78.33%.
b. Recall Optimization menunjukkan model berhasil mencapai target Recall >75% untuk kelas Dropout, memastikan kemampuan sistem dalam memprediksikan sebanyak mungkin mahasiswa berisiko.
c. Interpretasi Model (Feature Importance) untuk mengekstrak bobot algoritma untuk menyimpulkan bahwa performa akademik semester 1 & 2, admission grade atau nilai masuk ke universitas serta kelancaran pembayaran SPP adalah prediktor absolut terkuat untuk kelulusan mahasiswa.

6. Deployment & Business Intelligence (Implementasi & Visualisasi)
Proyek ini tidak berhenti pada script Python, melainkan diimplementasikan ke dalam dua produk akhir yang siap pakai (production-ready):

 - Aplikasi Prediksi Individu (Streamlit): Membangun aplikasi web interaktif menggunakan antarmuka Python (app.py). Aplikasi ini dilengkapi dengan perhitungan tingkat keyakinan (Prediction Probability) secara real-time untuk memudahkan Dosen Pembimbing melakukan asesmen profil satu mahasiswa secara cepat.

 - Dashboard Monitoring Eksekutif (Looker Studio): Menyiapkan dataset khusus (looker_dataset_with_ai.csv) yang menggabungkan data aktual dengan prediksi AI. Dashboard ini menyajikan:

    a. Tabel "Actionable List" (Daftar prioritas intervensi mahasiswa berisiko).
    b. Evaluasi kinerja model secara live (Confusion Matrix / AI Accuracy Tracker).
    c. Visualisasi sebaran nilai masuk mahasiswa dan performa akademik.


### Persiapan

Sumber data: Dataset students performance yang mencakup data demografi & latar belakang personal, data jalur masuk & program studi, latar belakang keluarga, status finansial & ekonomi makro, dan performa akademik (https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

Instalasi

1. Setup Environment - Anaconda
```bash
conda create --name jaya-ds python=3.9
conda activate jaya-ds
pip install -r requirements.txt

2. Setup Environment - Shell/Terminal
```bash
mkdir jaya-jaya-dashboard
cd jaya-jaya-dashboard
pipenv install
pipenv shell
pip install -r requirements.txt

3. Run Streamlit App
```bash
streamlit run app.py
```

### Business Dashboard
1. Scoreboard (Indikator Kinerja Utama)
ringkasan eksekutif yang memberikan gambaran cepat tentang status kelulusan mahasiswa di institusi.
Total Mahasiswa: Menampilkan volume populasi yang sedang dianalisis.
Dropout Rate: Metrik paling krusial yang menunjukkan jumlah mahasiswa yang berhenti kuliah. Angka ini menjadi indikator utama keberhasilan sistem intervensi.
Average Admission Grade: Menunjukkan kualitas input mahasiswa secara keseluruhan.


2. Distribusi Mahasiswa (Status Overview)
Visualisasi ini ditampilakan dalam bentuk Donut Chart menunjukkan proporsi mahasiswa dalam tiga kategori: Dropout, Enrolled (Masih Aktif), dan Graduate (Lulus). Ini membantu pimpinan melihat seberapa jumlah mahasiswa droupout dibandingkan dengan mereka yang berhasil mencapai kelulusan.

3. Dampak Kelancaran SPP (Tuition Status Analysis)
Grafik ini menyoroti hubungan antara administrasi keuangan dan keberlanjutan studi.Terlihat jelas pola di mana mahasiswa yang menunggak SPP memiliki kecenderungan dropout yang jauh lebih tinggi.

4. Admission Grade vs Status (Analisis Seleksi Masuk)
Menggunakan visualisasi Boxplot, bagian ini menunjukkan sebaran nilai ujian masuk untuk tiap kelompok status.

5. Status Kelulusan Berdasarkan Kepemilikan Beasiswa
Visualisasi ditampilkan dengan grafik Stacked Bar Chart untuk membandingkan tingkat kelulusan antara penerima beasiswa dan non-beasiswa. Penerima beasiswa memiliki tingkat dropout yang lebih rendah. 

6. Prediction vs Actual Matrix AI (Confusion Matrix)
Ini adalah bagian teknis yang disajikan secara bisnis untuk menunjukkan transparansi model AI. Tabel ini memetakan tebakan AI terhadap kondisi nyata. Misalnya, berapa banyak mahasiswa yang diprediksi dropout dan ternyata memang benar-benar dropout.


7. Korelasi Akademik (Academic Trend)
Melalui Scatter Plot yang memetakan nilai Semester 1 vs Semester 2, kita dapat melihat tren performa akademik.


## Conclusion
Berdasarkan Exploratory Data Analysis (EDA) dan ekstraksi Feature Importance dari model algoritma, ditemukan tiga pilar utama yang menentukan keberhasilan studi mahasiswa:

1. Performa Akademik Tahun Pertama adalah Prediktor Mutlak. Nilai rata-rata dan jumlah SKS yang berhasil diselesaikan pada Semester 1 dan Semester 2 memiliki korelasi terkuat terhadap status kelulusan. Mahasiswa yang gagal beradaptasi secara akademik di tahun pertama hampir dapat dipastikan akan mengalami dropout.

2. Kestabilan Finansial sebagai Pelindung (Protective Factor) karena status kelancaran pembayaran uang kuliah (Tuition fees up to date) dan kepemilikan beasiswa (Scholarship holder) sangat memengaruhi retensi mahasiswa. Mahasiswa yang menunggak SPP di semester awal memiliki probabilitas dropout yang sangat tinggi, terlepas dari nilai akademik mereka.

3. Standar Masuk (Admission Grade) Berkorelasi, namun bukan penentu akhir status mahasiswa. Meskipun mahasiswa dengan status dropout rata-rata memiliki nilai ujian masuk yang lebih rendah, hal tersebut bisa diatasi jika mereka mendapatkan dukungan finansial dan pendampingan akademik yang tepat di semester awal.

### Rekomendasi Action Items (Optional)

1. Sistem Early Warning Akademik dan intervensi akademik berbasis prioritas
    Mengaktifkan notifikasi otomatis ke Dosen Pembimbing Akademik jika mahasiswa gagal mencapai target SKS tertentu di Semester 1 dan 2. Dosen Pembimbing Akademik diwajibkan menjadwalkan sesi konseling khusus pada akhir Semester 1 dan 2 bagi mahasiswa yang diprediksi Dropout oleh sistem AI, guna mengevaluasi beban SKS atau metode belajar mereka

2. Restrukturisasi kebijakan finansial
    Menawarkan skema cicilan pembayaran uang kuliah yang lebih ringan dan fleksibel (installment plan) bagi mahasiswa yang terdeteksi menunggak (nilai Tuition_fees_up_to_date = 0) di bulan-bulan pertama untuk mencegah mahasiswa dropout/keluar karena alasan tunggakan menumpuk.

3. Optimalisasi Program Beasiswa 
    Dengan mempertahankan atau bahkan memperluas kuota beasiswa bagi mahasiswa berprestasi dengan latar belakang ekonomi menengah ke bawah, karena data membuktikan bahwa beasiswa sangat efektif menekan angka dropout.
