Ringkasan Eksperimen GSALN untuk Image Super-Resolution

Judul
Perancangan dan Evaluasi GSALN Berbasis Triple-Attention untuk Image Super-Resolution pada Skala Pembesaran x2, x3, dan x4

Latar Belakang
Image super-resolution merupakan salah satu tugas penting dalam pengolahan citra digital yang bertujuan merekonstruksi citra beresolusi tinggi dari masukan citra beresolusi rendah. Permasalahan ini banyak ditemui pada citra hasil kompresi, citra pengawasan, citra medis, maupun citra lama yang mengalami penurunan kualitas. Kebutuhan akan metode super-resolution yang mampu meningkatkan detail visual secara akurat menjadi semakin penting karena kualitas citra sangat memengaruhi proses analisis lanjutan, baik oleh manusia maupun sistem berbasis kecerdasan buatan.

Meskipun berbagai pendekatan berbasis convolutional neural network telah menunjukkan hasil yang baik, masih terdapat kendala dalam mempertahankan detail tepi, tekstur halus, dan informasi berfrekuensi tinggi. Model yang hanya menekankan ekstraksi fitur lokal cenderung menghasilkan citra yang tampak halus, tetapi kehilangan ketajaman struktur penting. Di sisi lain, citra tidak hanya memerlukan pemahaman lokal, melainkan juga konteks global agar hubungan antarbagiannya tetap konsisten setelah proses pembesaran dilakukan.

Untuk menjawab tantangan tersebut, arsitektur GSALN dirancang dengan memadukan backbone residual dan mekanisme atensi pada beberapa dimensi fitur. Backbone residual digunakan untuk memperkuat aliran informasi dasar dan mempermudah proses pembelajaran fitur. Setelah itu, fitur diperkaya melalui Channel Attention yang memanfaatkan deskriptor domain frekuensi berbasis FFT sehingga model dapat menimbang kanal-kanal yang paling relevan terhadap proses rekonstruksi.

Selain perhatian pada dimensi kanal, GSALN juga menerapkan Spatial Attention melalui modul Large Kernel Attention (LKA). Modul ini membantu jaringan menangkap konteks spasial yang lebih luas tanpa menambah biaya komputasi secara berlebihan. Dengan cakupan reseptif yang lebih besar, model dapat memahami hubungan jarak jauh antarwilayah citra, sehingga struktur global dan pola visual penting dapat dipertahankan dengan lebih baik saat citra diperbesar.

GSALN kemudian memperkuat representasi fitur melalui Frequency Attention yang memanfaatkan pemetaan Fourier dan content-aware gating untuk menyoroti informasi frekuensi rendah, menengah, dan tinggi secara adaptif. Arsitektur ini juga dilengkapi dengan Contrast-Aware Pixel Attention (CAPA) agar model lebih fokus pada area berkontras tinggi seperti tepi dan tekstur, serta Learnable Residual Scaling (LRS) untuk menstabilkan penggabungan residual sebelum proses upsampling menggunakan PixelShuffle. Pada tahap evaluasi, pengujian mengikuti alur super-resolution standar dan menyediakan opsi test-time augmentation x8.

Berdasarkan uraian tersebut, penelitian mengenai GSALN menjadi relevan untuk dilakukan karena menawarkan pendekatan yang menggabungkan perhatian terhadap informasi kanal, spasial, dan frekuensi dalam satu arsitektur. Pendekatan ini diharapkan mampu memberikan rekonstruksi citra yang lebih baik pada berbagai skala pembesaran. Oleh karena itu, diperlukan kajian lebih lanjut untuk menganalisis rancangan model, konfigurasi pelatihan, dan mekanisme evaluasi yang digunakan pada GSALN dalam tugas image super-resolution.

Rumusan Masalah
1. Bagaimana merancang arsitektur super-resolution yang mampu memadukan informasi kanal, spasial, dan frekuensi agar detail citra hasil rekonstruksi tetap tajam?
2. Bagaimana peran modul CAPA, LRS, dan frequency attention berbasis Fourier dalam meningkatkan kualitas rekonstruksi citra pada GSALN?
3. Bagaimana kinerja GSALN pada skala pembesaran x2, x3, dan x4 jika dievaluasi menggunakan metrik PSNR dan SSIM pada dataset Set5?

Tujuan Penelitian
1. Menghasilkan model single image super-resolution (SISR) baru berbasis arsitektur GSALN yang memadukan backbone residual, Channel Attention, Spatial Attention, dan Frequency Attention.
2. Menganalisis peran modul CAPA, LRS, dan mekanisme perhatian berbasis Fourier dalam mendukung proses rekonstruksi citra beresolusi tinggi.
3. Mengevaluasi kinerja model GSALN pada skala pembesaran x2, x3, dan x4 menggunakan metrik PSNR dan SSIM pada dataset Set5.

Batasan Masalah
1. Penelitian hanya membahas model GSALN dengan konfigurasi utama num_feat=64, num_blocks=32, num_in_ch=3, num_out_ch=3, dan freq_mb=true.
2. Proses pelatihan menggunakan dataset DF2K, sedangkan validasi hanya menggunakan dataset benchmark Set5.
3. Skala pembesaran yang dianalisis dibatasi pada x2, x3, dan x4, dengan ukuran patch pelatihan berturut-turut 192, 288, dan 384.
4. Pengaturan pelatihan dibatasi pada batch_size_per_gpu=32, optimizer Adam, fungsi loss L1, scheduler CosineAnnealingRestartLR, warmup_iter=5000, dan total_iter=600000.
5. Evaluasi kuantitatif dibatasi pada PSNR dan SSIM pada kanal-Y, dengan crop_border mengikuti faktor skala, yaitu 2 untuk x2, 3 untuk x3, dan 4 untuk x4.
6. Penelitian difokuskan pada perancangan model, proses pelatihan, dan evaluasi kuantitatif GSALN untuk tugas image super-resolution, tanpa membahas implementasi di luar skenario eksperimen yang telah ditetapkan.

Manfaat Penelitian
1. Memberikan pemahaman mengenai penerapan arsitektur berbasis triple-attention dalam tugas image super-resolution.
2. Menjadi referensi dalam pengembangan model super-resolution yang memanfaatkan informasi kanal, spasial, dan frekuensi secara terpadu.
3. Memberikan gambaran mengenai konfigurasi pelatihan dan evaluasi yang dapat digunakan untuk menguji performa model pada beberapa skala pembesaran.
4. Menambah wawasan akademik mengenai pemanfaatan mekanisme attention dan residual learning untuk meningkatkan kualitas rekonstruksi citra.
