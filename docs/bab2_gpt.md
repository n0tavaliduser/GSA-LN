BAB II
KAJIAN PUSTAKA

2.1 Penelitian Terdahulu

Penelitian pada bidang image super-resolution, khususnya single image super-resolution (SISR), telah berkembang dari pendekatan berbasis convolutional neural network (CNN) menuju pendekatan yang memanfaatkan attention, transformer, dan diffusion model. Perkembangan tersebut menunjukkan bahwa peningkatan kualitas rekonstruksi citra tidak lagi hanya bergantung pada kedalaman jaringan, tetapi juga pada kemampuan model dalam menjaga detail fitur, menangkap konteks global, serta memulihkan informasi frekuensi tinggi. Oleh karena itu, kajian terhadap penelitian terdahulu diperlukan untuk memahami arah pengembangan metode, kelebihan setiap pendekatan, serta celah penelitian yang masih dapat dikembangkan.

Tabel 2.1 Penelitian Terdahulu pada Bidang Super-Resolution

| No. | Peneliti | Model/Metode | Fokus Penelitian | Hasil atau Kontribusi Utama |
| --- | --- | --- | --- | --- |
| 1 | Chung dan Kim (2025) | Unsupervised Image Super-Resolution via Domain Translation | Super-resolution citra satelit tanpa supervisi | Menawarkan pendekatan domain translation real-to-synthetic untuk super-resolution citra satelit resolusi tinggi. |
| 2 | Su et al. (2023) | Feature Preserving and Enhancing Network | Preservasi dan penguatan fitur | Menekankan pentingnya mempertahankan fitur penting agar hasil rekonstruksi lebih tajam. |
| 3 | Karwowska dan Wierzbicki (2023) | MCWESRGAN | Super-resolution berbasis GAN untuk citra satelit | Mengembangkan model GAN yang ditujukan untuk meningkatkan kualitas citra satelit. |
| 4 | Xiao et al. (2024) | EDiffSR | Diffusion model efisien untuk super-resolution | Menunjukkan bahwa diffusion model dapat diterapkan secara lebih efisien pada super-resolution citra penginderaan jauh. |
| 5 | Magid et al. (2021) | Dynamic High-Pass Filtering and Multi-Spectral Attention | Penguatan detail frekuensi tinggi | Menekankan peran high-pass filtering dan spectral attention untuk memperbaiki detail citra. |
| 6 | Chen et al. (2021) | Attention in Attention Network | Attention bertingkat untuk super-resolution | Menunjukkan efektivitas integrasi attention di dalam attention pada rekonstruksi citra. |
| 7 | Liang et al. (2021) | Model Adaptation for SISR | Adaptasi model pada SISR | Menyoroti pentingnya adaptasi model terhadap karakteristik data pada tugas super-resolution. |
| 8 | Lan et al. (2021) | MADNet | Model ringan dan cepat untuk SISR | Mengembangkan jaringan yang berorientasi pada efisiensi komputasi dan kecepatan inferensi. |
| 9 | Liu et al. (2022) | Self-Attention Negative Feedback Network | Real-time image super-resolution | Menggabungkan self-attention dengan mekanisme negative feedback untuk kebutuhan super-resolution waktu nyata. |
| 10 | Liu et al. (2022) | DSMA | Reference-based super-resolution dengan multi-attention | Menggunakan dual-view supervised learning dan multi-attention untuk memperkaya rekonstruksi citra. |
| 11 | Chen et al. (2023) | Cross Aggregation Transformer dan Dual Aggregation Transformer | Agregasi fitur berbasis transformer | Menunjukkan efektivitas transformer dalam pencampuran dan agregasi fitur untuk restorasi dan super-resolution. |
| 12 | Saharia et al. (2023) | Iterative Refinement for Image Super-Resolution | Super-resolution berbasis diffusion | Menawarkan rekonstruksi citra melalui penyempurnaan bertahap menggunakan diffusion process. |

Muhammad et al. (2022) melalui kajian komprehensif mengenai deep learning untuk image super-resolution menunjukkan bahwa pendekatan pembelajaran mendalam telah menjadi arus utama dalam pengembangan metode super-resolution. Tinjauan tersebut memperlihatkan bahwa fokus penelitian bergerak dari jaringan CNN dasar menuju model yang lebih adaptif, lebih dalam, dan lebih kaya mekanisme atensi. Hal ini menegaskan bahwa pengembangan model SISR baru perlu mempertimbangkan tidak hanya struktur ekstraksi fitur, tetapi juga strategi pemulihan detail citra yang semakin kompleks.

Su et al. (2023) menekankan pentingnya preservasi dan penguatan fitur dalam proses super-resolution. Arah penelitian ini relevan karena salah satu masalah utama pada super-resolution adalah hilangnya detail halus setelah citra diperbesar. Penelitian tersebut memperlihatkan bahwa kualitas hasil rekonstruksi sangat dipengaruhi oleh kemampuan model dalam menjaga informasi fitur yang penting sejak tahap ekstraksi awal hingga tahap rekonstruksi akhir.

Pada jalur penelitian berbasis attention, Chen et al. (2021) menunjukkan bahwa mekanisme attention yang disusun secara bertingkat mampu meningkatkan kemampuan model dalam menyeleksi informasi yang paling relevan untuk rekonstruksi citra. Temuan ini memperkuat pandangan bahwa super-resolution tidak cukup hanya mengandalkan konvolusi biasa, tetapi juga memerlukan modul seleksi fitur yang dapat mengarahkan fokus model pada area citra yang penting, seperti tepi, tekstur, dan pola struktur tertentu.

Magid et al. (2021) memperluas pendekatan attention dengan menambahkan dynamic high-pass filtering dan multi-spectral attention. Penelitian ini penting karena detail citra yang menentukan kualitas visual umumnya berada pada komponen frekuensi tinggi. Dengan demikian, gagasan untuk memperhatikan informasi frekuensi secara eksplisit menjadi sangat relevan dalam pengembangan model SISR, terutama ketika tujuan penelitian adalah menghasilkan model baru yang lebih peka terhadap detail halus dan ketajaman tekstur.

Di sisi efisiensi model, Lan et al. (2021) melalui MADNet memperlihatkan bahwa SISR juga dapat diarahkan pada model yang cepat dan ringan. Pendekatan ini penting karena banyak model super-resolution berkinerja tinggi cenderung memiliki beban komputasi besar. Penelitian tersebut memberi pemahaman bahwa pengembangan model baru sebaiknya tetap mempertimbangkan keseimbangan antara akurasi rekonstruksi dan efisiensi komputasi, terlebih jika model nantinya diarahkan untuk penggunaan yang lebih luas.

Liu et al. (2022) mengusulkan self-attention negative feedback network untuk real-time image super-resolution. Pendekatan ini menunjukkan bahwa mekanisme attention dapat dikombinasikan dengan strategi umpan balik untuk memperbaiki kestabilan dan efisiensi proses rekonstruksi. Kajian ini memperlihatkan bahwa attention tidak hanya bermanfaat untuk meningkatkan representasi fitur, tetapi juga dapat diarahkan untuk mendukung proses inferensi yang lebih responsif.

Pada skenario yang lebih kaya informasi, Liu et al. (2022) melalui DSMA memperkenalkan pendekatan reference-based super-resolution dengan dual-view supervised learning dan multi-attention mechanism. Penelitian ini menegaskan bahwa pemanfaatan banyak sumber informasi dan banyak mekanisme perhatian dapat memperkaya hasil rekonstruksi. Meskipun demikian, pendekatan reference-based berbeda dari SISR murni, sehingga masih terdapat kebutuhan akan model SISR yang kuat tanpa ketergantungan pada citra referensi tambahan.

Perkembangan berikutnya bergerak ke arah transformer. Chen et al. (2023) pada Cross Aggregation Transformer serta Dual Aggregation Transformer menunjukkan bahwa pencampuran dan agregasi fitur lintas jendela dan lintas dimensi dapat meningkatkan kemampuan restorasi citra. Pendekatan transformer menawarkan keunggulan dalam menangkap dependensi global, namun umumnya memiliki kompleksitas yang lebih tinggi. Oleh sebab itu, penelitian model SISR baru tetap perlu mempertimbangkan alternatif yang mampu memperoleh konteks global secara efektif tanpa mengorbankan efisiensi secara berlebihan.

Selain CNN, attention, dan transformer, diffusion model juga mulai digunakan dalam super-resolution. Saharia et al. (2023) menunjukkan bahwa iterative refinement dapat menghasilkan rekonstruksi citra yang berkualitas tinggi, sedangkan Xiao et al. (2024) mengembangkan diffusion probabilistic model yang lebih efisien untuk citra penginderaan jauh. Kedua penelitian ini memperlihatkan bahwa diffusion memiliki potensi besar, tetapi juga menunjukkan bahwa desain model yang efisien tetap menjadi isu penting dalam praktik super-resolution.

Pada domain citra penginderaan jauh dan hiperspektral, Chung dan Kim (2025), Karwowska dan Wierzbicki (2023), Deng et al. (2022), serta Ziaja et al. (2025) menunjukkan bahwa super-resolution juga berkembang secara kuat pada domain khusus. Fokus mereka mencakup pembelajaran tanpa supervisi, GAN, degradation learning, dan penggabungan multi-image super-resolution. Rangkaian penelitian ini menunjukkan bahwa karakteristik data sangat memengaruhi rancangan model, sehingga penelitian baru perlu merumuskan arsitektur yang tetap kuat pada tugas SISR umum sekaligus fleksibel terhadap berbagai skala pembesaran.

Berdasarkan penelitian-penelitian terdahulu tersebut, dapat dilihat bahwa masih terdapat ruang pengembangan untuk model SISR baru yang menggabungkan kekuatan backbone residual, perhatian terhadap kanal, perhatian spasial, dan perhatian frekuensi dalam satu arsitektur yang terintegrasi. Banyak penelitian sebelumnya menonjol pada satu sisi tertentu, misalnya preservasi fitur, efisiensi, attention, transformer, atau diffusion. Oleh karena itu, penelitian ini menempatkan diri pada upaya menghasilkan model SISR baru yang mengombinasikan berbagai kekuatan tersebut secara lebih terarah untuk mendukung rekonstruksi citra pada skala x2, x3, dan x4.

2.2 Deep Learning

Deep learning adalah cabang machine learning yang menggunakan jaringan saraf tiruan berlapis-lapis untuk mempelajari representasi data secara hierarkis. Pada citra, lapisan awal umumnya mempelajari pola sederhana seperti tepi dan tekstur, sedangkan lapisan yang lebih dalam mempelajari pola yang lebih kompleks. Keunggulan utama deep learning terletak pada kemampuannya melakukan feature learning secara otomatis, sehingga model tidak lagi bergantung sepenuhnya pada rekayasa fitur manual.

Dalam konteks image super-resolution, deep learning menjadi pendekatan yang dominan karena mampu memodelkan hubungan nonlinier antara citra resolusi rendah dan citra resolusi tinggi. Muhammad et al. (2022) menjelaskan bahwa perkembangan model deep learning telah meningkatkan performa super-resolution secara signifikan dibandingkan pendekatan tradisional. Dengan dukungan optimisasi berbasis backpropagation dan ketersediaan perangkat komputasi modern, deep learning memungkinkan pembentukan model yang lebih adaptif terhadap keragaman pola degradasi citra.

Penggunaan deep learning pada penelitian ini menjadi penting karena tujuan utama penelitian adalah menghasilkan model SISR baru. Dengan deep learning, proses pembelajaran dapat diarahkan untuk mengekstraksi, memperkuat, dan merekonstruksi informasi visual penting secara end-to-end. Hal ini sangat sesuai dengan kebutuhan pengembangan model yang tidak hanya akurat, tetapi juga mampu mengintegrasikan berbagai mekanisme perhatian dalam satu sistem.

2.3 Computer Vision

Computer vision adalah bidang ilmu yang berfokus pada bagaimana komputer memperoleh, memproses, menganalisis, dan memahami informasi visual dari gambar atau video. Tujuan utama computer vision adalah membuat sistem komputer mampu menafsirkan konten visual secara bermakna, seperti mengenali objek, mendeteksi pola, melakukan segmentasi, maupun memperbaiki kualitas citra. Dengan demikian, computer vision mencakup tugas tingkat tinggi seperti klasifikasi dan deteksi, serta tugas tingkat rendah seperti denoising, deblurring, dan super-resolution.

Image super-resolution termasuk ke dalam low-level computer vision karena berorientasi pada perbaikan kualitas representasi visual. Berbeda dari tugas pengenalan objek yang berfokus pada semantik, super-resolution berfokus pada pemulihan detail piksel dan struktur citra. Namun demikian, hasil super-resolution yang baik dapat berkontribusi langsung pada performa tugas computer vision lainnya, karena citra yang lebih jelas akan lebih mudah dianalisis baik oleh manusia maupun model otomatis.

Perkembangan modern computer vision menunjukkan adanya pergeseran dari pendekatan berbasis filter manual ke pendekatan berbasis pembelajaran mendalam. Dalam jalur ini, tugas super-resolution menjadi salah satu bidang yang aktif berkembang karena berfungsi sebagai tahap peningkatan kualitas data visual. Oleh sebab itu, kajian tentang SISR tidak dapat dilepaskan dari kerangka besar computer vision, khususnya pada aspek restorasi citra dan peningkatan mutu visual.

2.4 Single Image Super-Resolution

Single image super-resolution adalah proses merekonstruksi satu citra resolusi tinggi dari satu citra resolusi rendah. Berbeda dari multi-image super-resolution yang memanfaatkan beberapa citra masukan, SISR hanya mengandalkan satu citra sehingga menjadi masalah yang lebih menantang. Tantangan utama SISR terletak pada sifatnya yang ill-posed, karena banyak kemungkinan citra resolusi tinggi dapat dihasilkan dari satu citra resolusi rendah yang sama.

Secara umum, citra resolusi rendah dapat dipandang sebagai hasil degradasi dari citra resolusi tinggi akibat proses blur, downsampling, dan noise. Tugas model SISR adalah mempelajari pemetaan balik dari citra resolusi rendah menuju citra resolusi tinggi sedekat mungkin dengan kondisi sebenarnya. Oleh karena itu, model SISR yang baik harus mampu menebak detail yang hilang secara masuk akal sambil tetap menjaga struktur global citra.

Penelitian SISR modern banyak memanfaatkan CNN, attention, transformer, dan diffusion untuk meningkatkan kualitas rekonstruksi. Liang et al. (2021) menekankan bahwa adaptasi model penting dalam SISR, sedangkan Lan et al. (2021) menunjukkan pentingnya desain yang efisien. Di sisi lain, Su et al. (2023), Chen et al. (2021), dan Liu et al. (2022) menunjukkan bahwa preservasi fitur dan attention memainkan peran besar dalam peningkatan kualitas hasil super-resolution.

2.4.1 Tantangan dalam SISR

Tantangan utama dalam SISR meliputi hilangnya informasi frekuensi tinggi, kesulitan mempertahankan ketajaman tepi, serta kebutuhan untuk menangkap konteks lokal dan global secara bersamaan. Model yang terlalu berfokus pada informasi lokal cenderung menghasilkan citra yang halus tetapi kurang tajam. Sebaliknya, model yang terlalu kompleks dapat meningkatkan biaya komputasi secara signifikan.

Selain itu, perbedaan karakteristik data dan faktor skala pembesaran juga menjadi tantangan tersendiri. Skala x2, x3, dan x4 memiliki tingkat kesulitan rekonstruksi yang berbeda. Semakin besar skala pembesaran, semakin besar pula detail informasi yang harus dipulihkan oleh model. Oleh karena itu, model SISR perlu dirancang agar tetap stabil dan efektif pada beberapa faktor skala.

2.4.2 Attention dalam SISR

Attention dalam SISR berfungsi untuk membantu model memfokuskan proses pembelajaran pada bagian fitur yang paling penting. Attention dapat diterapkan pada kanal, posisi spasial, maupun domain frekuensi. Chen et al. (2021), Magid et al. (2021), dan Liu et al. (2022) memperlihatkan bahwa mekanisme attention dapat meningkatkan kemampuan model dalam menyeleksi fitur yang relevan untuk proses rekonstruksi.

Dalam konteks penelitian ini, konsep attention menjadi landasan utama karena model yang dikembangkan diarahkan untuk mengintegrasikan perhatian kanal, spasial, dan frekuensi. Dengan pendekatan tersebut, diharapkan model dapat lebih peka terhadap detail citra sekaligus tetap mempertahankan konsistensi struktur global.

2.5 Convolutional Neural Network

Convolutional Neural Network (CNN) adalah arsitektur jaringan saraf yang dirancang untuk memproses data berbentuk grid, seperti citra. Operasi utama pada CNN adalah konvolusi, yaitu proses penyaringan lokal yang memungkinkan model mengekstraksi pola visual penting dari input. Dalam pengolahan citra, CNN sangat efektif untuk mengenali tepi, tekstur, kontur, dan pola spasial lainnya.

Pada tugas SISR, CNN berperan sebagai fondasi utama untuk mengekstraksi fitur dari citra resolusi rendah dan mengubahnya menjadi representasi yang lebih kaya. Banyak model super-resolution memanfaatkan beberapa lapisan konvolusi dan residual block untuk memperdalam ekstraksi fitur. Su et al. (2023) dan Lan et al. (2021) menunjukkan bahwa struktur berbasis CNN masih sangat relevan karena mampu memberikan keseimbangan antara kemampuan representasi dan efisiensi komputasi.

Namun, CNN juga memiliki keterbatasan, terutama pada cakupan receptive field yang cenderung lokal. Keterbatasan ini dapat menyebabkan model kurang efektif dalam menangkap hubungan jarak jauh antarbagian citra. Karena itu, berbagai penelitian menambahkan attention, large kernel operation, atau transformer agar model tidak hanya kuat pada detail lokal, tetapi juga pada pemahaman konteks global.

Dalam penelitian ini, CNN tetap menjadi dasar arsitektur karena sifatnya yang stabil dan efektif untuk ekstraksi fitur citra. Akan tetapi, CNN tidak digunakan secara tunggal, melainkan dipadukan dengan residual learning dan mekanisme attention agar mampu menjawab keterbatasan yang umum muncul pada model super-resolution berbasis konvolusi murni.

2.6 Metrik Performa

Evaluasi performa model super-resolution umumnya dilakukan menggunakan metrik kuantitatif yang membandingkan citra hasil rekonstruksi dengan citra referensi resolusi tinggi. Metrik ini diperlukan agar kualitas model dapat diukur secara objektif. Pada penelitian super-resolution, dua metrik yang paling umum digunakan adalah Peak Signal-to-Noise Ratio (PSNR) dan Structural Similarity Index Measure (SSIM).

2.6.1 Peak Signal-to-Noise Ratio

PSNR adalah metrik yang mengukur tingkat kesamaan antara citra hasil rekonstruksi dan citra referensi berdasarkan kesalahan piksel. Nilai PSNR dihitung dari Mean Squared Error (MSE). Semakin kecil nilai MSE, semakin besar nilai PSNR, yang berarti citra hasil rekonstruksi semakin mendekati citra acuan.

Secara matematis, MSE dapat dituliskan sebagai:

MSE = (1 / MN) x jumlah dari seluruh (I(i,j) - K(i,j)) kuadrat

dengan I(i,j) adalah piksel citra referensi, K(i,j) adalah piksel citra hasil rekonstruksi, dan M serta N adalah ukuran citra.

Nilai PSNR dapat dirumuskan sebagai:

PSNR = 10 x log10 ((MAX kuadrat) / MSE)

dengan MAX menyatakan nilai piksel maksimum. Pada citra 8-bit, nilai MAX umumnya adalah 255. Semakin tinggi PSNR, semakin baik kualitas rekonstruksi secara numerik. Namun, PSNR lebih menekankan kesamaan piksel dan belum tentu selalu sejalan dengan persepsi visual manusia.

2.6.2 Structural Similarity Index Measure

SSIM adalah metrik yang mengukur kemiripan struktur antara citra hasil rekonstruksi dan citra referensi. Berbeda dari PSNR yang menekankan kesalahan piksel, SSIM mempertimbangkan luminansi, kontras, dan struktur citra. Oleh karena itu, SSIM sering digunakan sebagai pelengkap PSNR agar penilaian kualitas citra menjadi lebih representatif.

Secara umum, SSIM dapat dinyatakan sebagai:

SSIM(x,y) = ((2µxµy + C1)(2σxy + C2)) / ((µx kuadrat + µy kuadrat + C1)(σx kuadrat + σy kuadrat + C2))

dengan µx dan µy menyatakan rata-rata intensitas, σx dan σy menyatakan variansi, σxy menyatakan kovariansi, sedangkan C1 dan C2 adalah konstanta stabilisasi. Nilai SSIM berada pada rentang 0 sampai 1. Semakin mendekati 1, semakin mirip struktur citra hasil rekonstruksi terhadap citra referensi.

Dalam penelitian ini, PSNR dan SSIM digunakan sebagai metrik utama karena keduanya telah menjadi standar evaluasi dalam banyak penelitian super-resolution. Penggunaan dua metrik tersebut memungkinkan penilaian kualitas model dilakukan dari dua sisi, yaitu akurasi piksel dan kemiripan struktur visual.

2.7 Sintesis Kajian Pustaka dan Posisi Penelitian

Berdasarkan kajian pustaka yang telah diuraikan, dapat disimpulkan bahwa penelitian super-resolution telah berkembang menuju model yang semakin kaya representasi fitur dan semakin kuat dalam memanfaatkan mekanisme attention. CNN tetap menjadi fondasi penting, tetapi kebutuhan untuk menangkap konteks global dan informasi frekuensi telah mendorong penggunaan modul attention, transformer, dan diffusion. Meskipun demikian, belum semua pendekatan mampu menggabungkan perhatian kanal, perhatian spasial, dan perhatian frekuensi secara seimbang dalam kerangka SISR yang terarah.

Posisi penelitian ini berada pada upaya menghasilkan model SISR baru yang memanfaatkan kekuatan backbone residual berbasis CNN dan mengintegrasikannya dengan mekanisme perhatian pada beberapa dimensi fitur. Dengan demikian, penelitian ini diharapkan dapat memberikan kontribusi pada pengembangan model super-resolution yang tidak hanya kuat dalam preservasi detail lokal, tetapi juga peka terhadap struktur global dan informasi frekuensi yang penting untuk rekonstruksi citra.
