# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech (Human Resources Division)

## Business Understanding

Jaya Jaya Institut adalah sebuah institusi pendidikan tinggi yang telah beroperasi selama 26 tahun dan berdiri sejak tahun 2000. Sebagai lembaga pendidikan yang memiliki rekam jejak panjang, institusi ini berdedikasi untuk mencetak lulusan-lulusan berprestasi yang siap berkontribusi di berbagai sektor industri dan masyarakat. Jaya Jaya Institut menawarkan berbagai program studi dan terus berupaya menjaga standar kualitas pendidikan yang tinggi. Bagi institusi ini, mempertahankan kelangsungan studi setiap mahasiswa sejak pendaftaran hingga berhasil lulus (graduate) bukan sekadar tanggung jawab moral pendidikan, melainkan juga pilar utama yang menyokong keberhasilan operasional dan reputasi institusi di dunia pendidikan tinggi.

## Permasalahan Bisnis

Saat ini, Jaya Jaya Institut sedang menghadapi tantangan kritis berupa tingginya tingkat mahasiswa yang putus kuliah atau dropout. Berdasarkan data operasional, rasio mahasiswa yang meninggalkan bangku kuliah sebelum lulus tergolong sangat tinggi. Permasalahan ini bukan sekadar statistik akademik, melainkan sebuah krisis institusional yang membawa dampak bisnis yang serius.

Urgensi dan Efek Bisnis:
1. Ketidakstabilan Finansial
    Setiap mahasiswa yang dropout merepresentasikan hilangnya potensi pendapatan institusi secara langsung, khususnya dari Sumbangan Pembinaan Pendidikan (SPP) rutin. Hilangnya pendapatan ini secara masif akan sangat mengganggu cash flow yang dibutuhkan untuk operasional harian, pemeliharaan fasilitas, dan pembayaran gaji staf pengajar dan pegawai.
2. Penurunan Akreditasi dan Reputasi
    Tingkat kelulusan (graduation rate) dan tingkat putus kuliah (dropout rate) adalah metrik utama yang dievaluasi oleh badan akreditasi nasional. Angka dropout yang tinggi secara langsung akan mengancam nilai akreditasi kampus dan merusak citra Jaya Jaya Institut di mata calon mahasiswa baru, orang tua, maupun pihak industri penyedia beasiswa.
3. Inefisiensi Sumber Daya Kampus
    Institusi telah mengalokasikan banyak sumber daya (misalnya waktu dosen, fasilitas laboratorium, hingga subsidi operasional) untuk setiap mahasiswa yang diterima. Kegagalan mahasiswa untuk lulus berarti investasi sumber daya tersebut menjadi terbuang sia-sia atau disebut sunk cost.

Risiko Jangka Panjang:
Jika permasalahan ini diabaikan dan terus berlanjut tanpa adanya sistem mitigasi yang terstruktur, Jaya Jaya Institut berisiko mengalami efek domino. Penurunan reputasi akan menyebabkan anjloknya jumlah pendaftaran mahasiswa baru di tahun-tahun berikutnya. Dalam skenario terburuk, hal ini dapat mengancam kelangsungan hidup atau kebangkrutan institusi. Oleh karena itu, penyelesaian masalah ini sangat mendesak. Institusi membutuhkan sebuah sistem proaktif, yaitu Sistem Peringatan Dini (Early Warning System) berbasis data untuk mendeteksi mahasiswa yang memiliki probabilitas tinggi untuk dropout sejak dini, sehingga pihak manajemen dapat melakukan tindakan intervensi, bimbingan, atau bantuan finansial sebelum mahasiswa tersebut benar-benar keluar.

## Cakupan Proyek

1. Business Understanding (Pemahaman Bisnis)
Tingginya tingkat mahasiswa putus kuliah (dropout) di Jaya Jaya Institut yang mencapai 39,1%, mengancam reputasi akademik dan stabilitas finansial institusi. Untuk itu, proyek ini bertujuan membangun sistem deteksi dini (Early Warning System) untuk mengidentifikasi mahasiswa yang memiliki probabilitas tinggi untuk dropout berdasarkan profil awal dan performa akademik tahun pertama.

2. Data Understanding (Pemahaman Data)
Eksplorasi Data (EDA) dengan menganalisis dataset yang terdiri dari 4.424 rekam jejak mahasiswa. Dalam identifikasi pola ditemukan bahwa nilai 0 pada semester 1 dan 2 bukanlah sebuah anomali (error), melainkan sinyal kuat dari mahasiswa yang tidak mengikuti ujian atau telah keluar. Untuk itu, dilakukan analisis korelasi dengan memetakan hubungan antara faktor akademik (SKS, IPK, program studi), faktor demografi (usia, gender, status perkawinan), dan faktor finansial (status beasiswa, tunggakan SPP) terhadap status kelulusan mahasiswa.

3. Data Preparation (Persiapan Data)
Pada data preparation dilakukan feature engineering untuk membuat variabel turunan yang dapat memperkuat daya prediksi model. feature engineering meliput financial risk, total approved units, dan age group.
a. Financial_Risk, indikator risiko keuangan gabungan antara status hutang dan kepemilikan beasiswa.
b. Total_Approved_Units, total akumulasi beban SKS yang berhasil diselesaikan di tahun pertama.
c. Age_Group, pengelompokan usia pendaftar menjadi Reguler, Dewasa Muda, dan Dewasa.
Selain itu juga dilakukan Feature Selection dengan mereduksi dimensi dengan membuang fitur makroekonomi (GDP, Tingkat Inflasi, Pengangguran) dan fitur dengan kardinalitas tinggi yang dapat memicu noise pada model. Terakhir, Data Transformation dilakukan untuk mengubah data kategorikal dan teks (Dropout dan Graduate) menjadi format numerik menggunakan teknik Label Encoding.

5. Modeling (Pemodelan Machine Learning)
a. Pemilihan algoritma menggunakan Random Forest Classifier yang tangguh dalam menangani data non-linear dan interaksi multivariabel.
b. Hyperparameter Tuning dilakukan untuk mengoptimalkan algoritma menggunakan GridSearchCV untuk menguji puluhan kombinasi parameter secara otomatis.
c. Penanganan Imbalanced Data menggunakan parameter class_weight='balanced' untuk memastikan algoritma tetap adil dan akurat

6. Evaluation (Evaluasi Model)
Model akhir dikembangkan menggunakan algoritma Random Forest dengan pendekatan klasifikasi biner (Dropout vs Graduate). Model ini berhasil mencapai tingkat Akurasi sebesar 90,63% dengan Recall untuk kelas Dropout sebesar 81%, yang berarti sistem ini sangat andal dalam mendeteksi mahasiswa yang berisiko dropout sebelum hal itu terjadi
c. Interpretasi Model (Feature Importance) untuk mengekstrak bobot algoritma untuk menyimpulkan bahwa performa akademik semester 1 & 2, admission grade atau nilai masuk ke universitas serta kelancaran pembayaran SPP adalah prediktor absolut terkuat untuk kelulusan mahasiswa.

7. Deployment & Business Intelligence (Implementasi & Visualisasi)
Proyek ini tidak berhenti pada script Python, melainkan diimplementasikan ke dalam dua produk akhir yang siap pakai (production-ready):

 - Aplikasi Prediksi Individu (Streamlit): Membangun aplikasi web interaktif menggunakan antarmuka Python (app.py). Aplikasi ini dilengkapi dengan perhitungan tingkat keyakinan (Prediction Probability) secara real-time untuk memudahkan Dosen Pembimbing melakukan asesmen profil satu mahasiswa secara cepat.

 - Dashboard Monitoring Eksekutif (Looker Studio): Menyiapkan dataset khusus (looker_dataset_with_ai.csv) yang menggabungkan data aktual dengan prediksi AI. Dashboard ini menyajikan visualisasi sebaran nilai masuk mahasiswa dan performa akademik dan evaluasi kinerja model secara live (Confusion Matrix / AI Accuracy Tracker).

### Persiapan
Sumber data: Dataset students performance yang mencakup data demografi & latar belakang personal, data jalur masuk & program studi, latar belakang keluarga, status finansial & ekonomi makro, dan performa akademik (https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)

Setup Environment:
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

## Business Dashboard
1. Scoreboard (Indikator Kinerja Utama)
ringkasan eksekutif yang memberikan gambaran cepat tentang status kelulusan mahasiswa di institusi.
Total Mahasiswa: Menampilkan volume populasi yang sedang dianalisis.
Dropout Rate: Metrik paling krusial yang menunjukkan jumlah mahasiswa yang berhenti kuliah. Angka ini menjadi indikator utama keberhasilan sistem intervensi.
Average Admission Grade: Menunjukkan kualitas input mahasiswa secara keseluruhan.

2. Distribusi Mahasiswa (Status Overview)
Visualisasi ini ditampilakan dalam bentuk Donut Chart menunjukkan proporsi mahasiswa dalam tiga kategori: Dropout, Enrolled dan Graduate (Lulus). Ini membantu pimpinan melihat seberapa jumlah mahasiswa droupout dibandingkan dengan mereka yang berhasil mencapai kelulusan.

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

Link Untuk mengakses Dashboard Looker Studio: https://datastudio.google.com/reporting/140be2c6-1cb8-42c4-ba1c-1b8d2860a5dc

## Menjalankan Sistem Machine Learning

Sistem Machine Learning untuk prediksi dropout mahasiswa ini telah di-deploy menjadi sebuah prototype aplikasi web interaktif menggunakan antarmuka Streamlit Community Cloud. Aplikasi ini dirancang agar dapat digunakan dengan mudah oleh pengguna non-teknis, seperti Dosen Pembimbing Akademik atau staf manajemen kampus.

🔗 Tautan Prototype Aplikasi:
https://jaya-jaya-student-attrition-nuraidahks-prototype.streamlit.app/

Panduan Penggunaan Prototype (User Manual)
Aplikasi ini menggunakan metode Full Dynamic Input, yang artinya hasil prediksi akan sangat bergantung pada kelengkapan profil mahasiswa yang diinput secara manual. Berikut adalah langkah-langkahnya:
1.  Akses Aplikasi: Buka tautan di atas melalui browser.
2. Input Data Mahasiswa: Di sisi kiri halaman, terdapat tiga kategori panel lipat (expander) yang harus diisi:
    Panel 1 (Demografi): Isi data dasar seperti usia, jenis kelamin, dan status pernikahan.
    Panel 2 (Akademik & Ekonomi): Masukkan kode program studi, jalur pendaftaran, serta status finansial (beasiswa, hutang, dan kelancaran SPP).
    Panel 3 (Performa Semester 1 & 2): Masukkan jumlah SKS yang diambil, SKS yang lulus, dan nilai rata-rata IPK untuk kedua semester awal.

3. Eksekusi Prediksi: Klik tombol "🚀 Jalankan Prediksi Algoritma".
4. Interpretasi Hasil ditunjukkan dengansSistem akan menampilkan status DROPOUT (Merah) jika berisiko tinggi, atau GRADUATE (Hijau) jika berisiko rendah. Lihat bagian "Keyakinan Model" untuk mengetahui seberapa besar probabilitas (%) mahasiswa tersebut masuk ke dalam kategori tertentu.

Catatan penting:
Hal Penting yang Harus Diperhatikan
- Pastikan Admission Grade menggunakan skala 0-200 dan IPK Semester menggunakan skala 0.0 - 20.0 sesuai standar dataset.
- Untuk pengisian "Kode" (seperti kode jurusan atau kualifikasi orang tua), silakan merujuk pada lampiran Data Dictionary di dokumen teknis ini untuk menghindari kesalahan interpretasi oleh model AI.
- Model saat ini memiliki akurasi sebesar 90,63% pada data uji. Meskipun tinggi, hasil prediksi sistem ini harus digunakan sebagai alat bantu pendukung keputusan, bukan penentu mutlak kebijakan akademik.

## Conclusion
Berdasarkan Exploratory Data Analysis (EDA) dan ekstraksi Feature Importance dari model algoritma, ditemukan tiga pilar utama yang menentukan keberhasilan studi mahasiswa:

1. Analisis Feature Importance menunjukkan bahwa Curricular Units 2nd Sem Approved dan Tuition fees up to date adalah dua prediktor terkuat. Hal ini mengonfirmasi bahwa intervensi harus difokuskan pada bantuan finansial dan pendampingan akademik intensif di tahun pertama.

2. Kestabilan Finansial sebagai Pelindung (Protective Factor) karena status kelancaran pembayaran uang kuliah (Tuition fees up to date) dan kepemilikan beasiswa (Scholarship holder) sangat memengaruhi retensi mahasiswa. Mahasiswa yang menunggak SPP di semester awal memiliki probabilitas dropout yang sangat tinggi, terlepas dari nilai akademik mereka.

3. Standar Masuk (Admission Grade) Berkorelasi, namun bukan penentu akhir status mahasiswa. Meskipun mahasiswa dengan status dropout rata-rata memiliki nilai ujian masuk yang lebih rendah, hal tersebut bisa diatasi jika mereka mendapatkan dukungan finansial dan pendampingan akademik yang tepat di semester awal.

### Rekomendasi Action Items (Optional)

1. Sistem Early Warning Akademik dan intervensi akademik berbasis prioritas
    Mengaktifkan notifikasi otomatis ke Dosen Pembimbing Akademik jika mahasiswa gagal mencapai target SKS tertentu di Semester 1 dan 2. Dosen Pembimbing Akademik diwajibkan menjadwalkan sesi konseling khusus pada akhir Semester 1 dan 2 bagi mahasiswa yang diprediksi Dropout oleh sistem AI, guna mengevaluasi beban SKS atau metode belajar mereka

2. Restrukturisasi kebijakan finansial
    Menawarkan skema cicilan pembayaran uang kuliah yang lebih ringan dan fleksibel (installment plan) bagi mahasiswa yang terdeteksi menunggak (nilai Tuition_fees_up_to_date = 0) di bulan-bulan pertama untuk mencegah mahasiswa dropout/keluar karena alasan tunggakan menumpuk.

3. Optimalisasi Program Beasiswa 
    Dengan mempertahankan atau bahkan memperluas kuota beasiswa bagi mahasiswa berprestasi dengan latar belakang ekonomi menengah ke bawah, karena data membuktikan bahwa beasiswa sangat efektif menekan angka dropout.
