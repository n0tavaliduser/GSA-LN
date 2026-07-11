# Ringkasan Eksperimen GSALN (Global-Spatial Attention Layer Network)

Berdasarkan arsitektur pada `basicsr/archs/gsaln_arch.py` serta hasil pengujian dan konfigurasi eksperimen (x2, x3, x4), berikut adalah susunan laporan ringkasan:

## 1. Judul
**Peningkatan Kinerja Image Super-Resolution Menggunakan Arsitektur GSALN dengan Triple-Attention dan Contrast-Aware Pixel Attention**

## 2. Latar Belakang
Teknik *Image Super-Resolution* (SR) bertujuan untuk merekonstruksi citra beresolusi tinggi (HR) dari input citra beresolusi rendah (LR). Meskipun berbagai arsitektur CNN memberikan hasil yang menjanjikan, banyak pendekatan konvensional yang mengabaikan detail frekuensi tinggi dan *long-range dependencies* (konteks global), sehingga hasil rekonstruksi seringkali tampak terlalu mulus (*oversmoothed*) dan kehilangan ketajaman pada tekstur atau tepi (*edge*). Untuk mengatasi tantangan tersebut, model **GSALN (Global-Spatial Attention Layer Network)** diusulkan. Model ini menggunakan arsitektur pemrosesan berbasis *Triple-Attention*, yang mengeksplorasi *Channel Attention* (FCA-FFT), *Spatial Attention* (memanfaatkan *Large Kernel Attention* (LKA) agar dapat menangkap jangkauan fitur spasial yang leluasa), dan *Frequency Attention* berbasis fourier (memaksimalkan ekstraksi informasi frekuensi/batas tepi). Selain itu, untuk menstabilkan konvergensi pembelajaran sekaligus mendongkrak kualitas restorasi, model menerapkan metode *Learnable Residual Scaling* (LRS) dan *Contrast-Aware Pixel Attention* (CAPA) yang memberikan atensi adaptif lebih lanjut pada piksel terindikasi memiliki eror (*MSE*) tertinggi.

## 3. Rumusan Masalah
1. Bagaimana mengintegrasikan mekanisme fitur *Triple-Attention* (Spatial LKA, Channel FCA-FFT, dan Frequency) untuk mencegah hilangnya komponen detail tak kasatmata dan mengatasi masalah keburaman tekstur pada tugas SR?
2. Bagaimana peran taktis dari komponen penguat (booster) seperti *Contrast-Aware Pixel Attention* (CAPA) dan parameter *Learnable Residual Scaling* (LRS) pada tingkat kemampuan adaptasi model?
3. Setinggi apa performa hasil restorasi resolusi jaringan GSALN jika dinilai dalam simulasi upscaling berskala **x2**, **x3**, dan **x4**?

## 4. Batasan Masalah
1. **Arsitektur Model**: Penjenjangan model difokuskan secara spesifik pada jaringan arsitektur SR *GSALN* (berisi 32 *Residual Blocks*, 64 *feature channels*) dengan pemanfaatan modul fitur modifikasi seperti LKA, CAPA, LRS, dan tanpa menggunakan struktur konvensional fiksasi parameter tertentu seperti *Batch Normalization*.
2. **Dataset**: Set data latih (training) difokuskan eksklusif pada dataset pencitraan berkualitas tinggi **DF2K**, sementara evaluasi validasi model dieksekusi dengan dataset *benchmark* **Set5**.
3. **Konfigurasi Pelatihan**: Konfigurasi batasan iterasi difiksasi maksimal pada 600.000 iterasi, dengan komputasi kerugian fungsi dasar menggunakan *L1Loss* dan optimasi parameter ditangani oleh paduan *Adam Optimizer* dengan skema *Cosine Annealing Restart LR*.
4. **Faktor Skala Upsampling**: Ruang lingkup eksplorasi pembatasan unjuk kerja dipersempit pada tiga dimensi perbesaran saja, melingkupi perbesaran spasial skala **x2**, **x3**, dan **x4**.
5. **Metrik Evaluasi**: Ukuran tingkat keberhasilan rekonstruksi dibatasi hanya bertumpu pada metrik pengujian kuantitatif fundamental standar yaitu ***PSNR*** (*Peak Signal-to-Noise Ratio*) dan ***SSIM*** (*Structural Similarity Index*).
   *(Berdasarkan pengujian, hasil terbaik validasi pada dataset Set5 masing-masing mencapai: Skala x2 dengan PSNR 38.14 dB / SSIM 0.9609, Skala x3 dengan PSNR 34.56 dB / SSIM 0.9285, dan Skala x4 dengan PSNR 32.34 dB / SSIM 0.8967).*
