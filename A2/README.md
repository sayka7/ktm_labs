# A2. GPU va sun'iy intellekt tezlatkichlaridan foydalangan holda parallel hisoblashni tashkil etish

Bu virtual laboratoriyada siz matritsa ko'paytirish va konvolyutsiyani — neyron tarmoqning asosiy yadrolarini — bir necha darajada parallellashtirasiz va bir xil metodika (isinish, takror, median) bilan o'lchaysiz: sof Python → NumPy vektorlash → Numba @njit va prange (OpenMP muqobili) → kesh bloklash → PyTorch iplari → multiprocessing → Amdahl qonuni. Muhim: sandbox'da CUDA va GPU mavjud emas, shuning uchun real labdagi CUDA yadrolari va Tensor Core o'lchovlari o'rniga 2 yadroli CPU'da parallelizm o'rganiladi; har qadam oxirida «GPU bo'lsa shu kod qanday o'zgaradi» izohi berilgan. Massivlar kichik (n ≤ 1024), chunki Numba birinchi kompilyatsiyasi va 2 yadro vaqt chegarasini talab qiladi.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/A2

## Bosqichlar

### 1-qadam. Sof Python vs NumPy vektorlash

Uch sikldagi sof Python matritsa ko'paytirish (n=64, 128) va NumPy BLAS chaqiruvi bir xil metodika bilan o'lchanadi, GFLOP/s va tezlanish hisoblanadi; so'ng 1M elementli GELU uchun list comprehension va vektorlangan NumPy solishtiriladi. Kutilgan natija: matmul'da 250–500×, GELU'da ~10× tezlanish; NumPy'ning oraliq massivlar tufayli xotira trafigi ko'rsatiladi. n va N ni o'zgartirib tezlanish qanday o'sishini kuzating.

Fayl: `s1_1_qadam_sof_python_vs_numpy_vektorlash.py` — ishga tushirish: `python3 s1_1_qadam_sof_python_vs_numpy_vektorlash.py`

Kutilgan natija: Sof Python ~0,04 GFLOP/s, NumPy FP64 10–20 GFLOP/s, FP32 n=1024 da ~45 GFLOP/s; GELU 236 ms vs 21 ms.

Tajriba qilib ko'ring:

- matmul_python'ni ijk tartibiga o'zgartiring — sof Python'da ham kesh ta'siri sezilarmi?
- GELU'ni np.float64 bilan qayta o'lchang: xotira trafigi 2 baravar oshganda vaqt qanday o'zgaradi?
- NumPy matmul uchun n=2048 qo'shing va GFLOP/s barqaror qoladimi, tekshiring.

### 2-qadam. Numba @njit va prange — OpenMP'ning Python muqobili

@njit matmul (ikj) kompilyatsiya qilinadi, parallel=True + prange varianti 1 va 2 ipda (numba.set_num_threads) o'lchanadi, NumPy BLAS bilan solishtiriladi; alohida reduktsiya (sum x²) misoli 2 ipda xotira o'tkazuvchanligi cheklovini ko'rsatadi. Kutilgan natija: JIT kompilyatsiya ~0,3–1 s, prange 2 ipda ~1,6–1,9× tezlanish, BLAS yana 1,5× tezroq. Bu real labdagi `#pragma omp parallel for` va `reduction(+:s)` ning to'g'ridan-to'g'ri muqobili.

Fayl: `s2_2_qadam_numba_njit_va_prange_openmpning_.py` — ishga tushirish: `python3 s2_2_qadam_numba_njit_va_prange_openmpning_.py`

Kutilgan natija: Kompilyatsiya vaqtlari, threading layer nomi, 4 qatorli jadval (seq ~16 GFLOP/s, prange 2 ip ~27, BLAS ~39), reduktsiya 11 → 22 GB/s.

Tajriba qilib ko'ring:

- prange'ni ichki (j) siklga ko'chiring — tezlanish nima uchun yo'qoladi (ip yaratish xarajati)?
- n=512 va n=1024 uchun o'lchab, 2 ip samaradorligi (tezlanish/2) o'lcham bilan qanday o'zgarishini jadvalga yozing.
- fastmath=True bayrog'ini qo'shing va GFLOP/s hamda maks xato qanday o'zgarishini solishtiring.

### 3-qadam. Kesh bloklash (tiled) vs naive matritsa ko'paytirish

Uch algoritm bir ipda solishtiriladi: naive ijk (B ustun bo'ylab — har qadam yangi kesh satri), ikj (qator bo'ylab, vektorlashadi) va bloklangan (tiled) variant; so'ng blok o'lchami 8…256 ta'siri o'lchanadi. Kutilgan natija: ijk → ikj ~10× tezlanish; kichik bloklar sikl xarajati tufayli sekin, 128–256 da ikj darajasiga yetadi. Kodda Numba'ning muhim nozik jihati ko'rsatilgan: range(jj, jj+bs) indekslari manfiy-tekshiruv tufayli vektorlashmaydi, slice bilan hal qilinadi.

Fayl: `s3_3_qadam_kesh_bloklash_tiled_vs_naive_mat.py` — ishga tushirish: `python3 s3_3_qadam_kesh_bloklash_tiled_vs_naive_mat.py`

Kutilgan natija: n=256/512/1024 jadvali: naive ~1,7 GFLOP/s, ikj ~15–17, tiled ~12; blok o'lchami jadvali 5 → 15 GFLOP/s.

Tajriba qilib ko'ring:

- matmul_tiled'da slice o'rniga to'g'ridan-to'g'ri C[i, j] indekslashga qaytib, GFLOP/s 2 ga tushishini o'zingiz tekshiring.
- n=2048 (48 MB, keshdan katta) uchun faqat ikj va tiled bs=128 ni o'lchang — endi bloklash foyda beradimi?
- Blokni ikki darajali qiling (tashqi 256, ichki 32) va natijani yozing.

### 4-qadam. Konvolyutsiyani to'rt usulda parallellashtirish

3×3 konvolyutsiya (16→32 kanal, 96×96) Numba to'g'ridan-to'g'ri sikl (1 ip), prange (1 va 2 ip), NumPy im2col+GEMM va PyTorch conv2d (oneDNN, 1 va 2 ip, batch 8) bilan bajariladi; barcha natijalar etalon bilan solishtiriladi. Kutilgan natija: prange 2 ipda ~2×, im2col ~16×, torch 2 ip ~20× tezlanish; xato < 1e-5. Bu cuDNN'ning implicit GEMM g'oyasini CPU'da ko'rsatadi.

Fayl: `s4_4_qadam_konvolyutsiyani_tort_usulda_para.py` — ishga tushirish: `python3 s4_4_qadam_konvolyutsiyani_tort_usulda_para.py`

Kutilgan natija: 81,4 MFLOP; jadval: Numba 1 ip ~2 GFLOP/s, prange 2 ip ~4, im2col ~33, torch 2 ip ~46 GFLOP/s; maks xato ~5e-6.

Tajriba qilib ko'ring:

- prange'ni chiqish kanallari (co) o'rniga qatorlar (i) bo'yicha qo'ying — yuk muvozanati o'zgaradimi?
- Yadroni 5×5 qilib FLOP va vaqtni qayta o'lchang; im2col matritsasi hajmi necha baravar oshadi?
- CIN=64, COUT=64, 224×224 (mustaqil ish topshirig'i) uchun faqat torch variantini o'lchang va GFLOP/s ni solishtiring.

### 5-qadam. PyTorch: GPU o'rniga ko'p yadro — 1 vs 2 ip, formatlar, yadro birlashtirish

torch.cuda.is_available() False ekani ko'rsatiladi va GPU'dagi o'zgarishlar izohlanadi. CPU'da: (a) matmul n=128/512/1024 uchun torch.set_num_threads(1) vs (2) tezlanishi; (b) FP32 vs BF16 vs INT8 (torch._int_mm) — CPU'da maxsus buyruqlar (AVX512-BF16/VNNI) bo'lmasa past formatlar sekinroq, bu apparat–format mosligi (Co-design) darsidir; (c) GELU eager (7 alohida amal) vs birlashtirilgan F.gelu — xotira trafigi 7× kam. Kutilgan natija: 2 ip katta n da ~2×, BF16 CPU'da 20× sekin, F.gelu ~3× tez.

Fayl: `s5_5_qadam_pytorch_gpu_orniga_kop_yadro_1_v.py` — ishga tushirish: `python3 s5_5_qadam_pytorch_gpu_orniga_kop_yadro_1_v.py`

Kutilgan natija: Iplar jadvali (1024 da 1,98× tezlanish), formatlar jadvali (FP32 ~50 GFLOP/s, BF16 ~2, INT8 ~4), GELU: eager ~130–220 ms, F.gelu ~45–80 ms.

Tajriba qilib ko'ring:

- n=4097 kabi 8 ga karrali bo'lmagan o'lchamni qo'shing — CPU'da ham unumdorlik tushadimi (GPU'dagi Tensor Core sharti bilan solishtiring)?
- GELU uchun N ni 1M ga tushiring: eager va fused farqi saqlanadimi (kesh ichida ishlaganda)?
- torch.float16 matmul'ni CPU'da sinab ko'ring va BF16 bilan vaqt/xato bo'yicha solishtiring.

### 6-qadam. Ma'lumot parallelligi: multiprocessing bilan 2 jarayon

600 ta sintetik kadr uchun Python+NumPy oldindan ishlov (Sobel, filtr, gistogramma) ketma-ket va multiprocessing.Pool (fork, 1 va 2 jarayon, chunksize 1/25/150) bilan bajariladi; tezlanish, samaradorlik va Pool yaratish xarajati o'lchanadi, natija ketma-ket bilan taqqoslanib tekshiriladi. Kutilgan natija: 2 jarayonda ~1,9–2,4× (chunksize kattaroq bo'lsa yaxshiroq), Pool ochish ~10–50 ms. Bu DDP'dagi «har qurilma o'z ulushini ishlaydi» g'oyasining CPU muqobili.

Fayl: `s6_6_qadam_malumot_parallelligi_multiproces.py` — ishga tushirish: `python3 s6_6_qadam_malumot_parallelligi_multiproces.py`

Kutilgan natija: Ketma-ket ~0,6 s; jadval 6 qator (jarayonlar × chunksize), 2 jarayonda ~2× tezlanish; Pool xarajati xabari.

Tajriba qilib ko'ring:

- KADRLAR=60 qiling — qisqa vazifada tezlanish saqlanadimi? Pool xarajati ulushini hisoblang.
- pool.map o'rniga pool.imap_unordered ishlatib, birinchi natija kelgunicha vaqtni (kechikish) o'lchang.
- kadr_ishlov ichidagi takror sonini 6 dan 1 ga tushiring: IPC xarajati hisobga nisbatan qanday o'sadi?

### 7-qadam. Amdahl va Gustafson qonunlari: nazariya va 2 yadroda o'lchov

S(n)=1/((1−p)+p/n) jadvali p=0,5…0,99 uchun chiqariladi; so'ng «dastur» = Python ketma-ket qism + torch matmul 1 va 2 ipda o'lchanib, S(2) dan parallel ulush p baholanadi va shu p bilan 8 hamda 1024 yadrodagi (GPU) tezlanish bashorat qilinadi. Grafikda Amdahl egri chiziqlari, o'lchangan nuqta va Gustafson chiziqlari ko'rsatiladi. Kutilgan natija: faqat matmul p≈1, ketma-ket qism qo'shilganda p≈0,97 → 1024 yadroda tezlanish atigi ~30×.

Fayl: `s7_7_qadam_amdahl_va_gustafson_qonunlari_na.py` — ishga tushirish: `python3 s7_7_qadam_amdahl_va_gustafson_qonunlari_na.py`

Kutilgan natija: Amdahl jadvali (p=0,95: n=inf da 20), o'lchov jadvali S(2) ~1,9–2,0, p baholari, 2 panelli grafik.

Tajriba qilib ko'ring:

- ketma_ket_qism sikl sonini 60 000 dan 600 000 ga oshiring — p va S(1024) qanday tushadi?
- Matritsa o'lchamini 768 dan 1536 ga oshiring (Gustafson: masala hajmi o'sadi) — p qanday o'zgaradi?
- S(2) o'lchovini 5 marta takrorlab p bahosining tarqalishini (min/max) yozing.

## Qo'shimcha topshiriqlar (10 ta, jami 100 ball)

| № | Topshiriq | Daraja | Ball |
|---|---|---|---|
| 1 | **Sof Python vs NumPy: o'lcham egri chizig'i** — 1-qadam skriptida n = 16, 32, 64, 128, 192 uchun sof Python va NumPy matmul vaqtlarini o'lchab, log-log grafikda chizing; sof Python uchun vaqt ~n³ ekanini (log-log og'ish koeffitsiyenti) hisoblab ko'rsating. | oson | 5 |
| 2 | **prange samaradorligi** — 2-qadamda n = 128, 256, 512, 1024 uchun prange 1 va 2 ipda tezlanish va parallel samaradorlikni (tezlanish/2) jadvalga yozing; kichik n da samaradorlik past bo'lishining sababini ip yaratish xarajati orqali tushuntiring. | oson | 5 |
| 3 | **Blok o'lchami tadqiqoti** — 3-qadamda bs = 8, 16, 32, 64, 128, 256, 512 uchun n=1024 va n=2048 da GFLOP/s o'lchang, grafik chizing va har n uchun eng yaxshi blokni hamda uning uch bloki egallagan KB hajmini L1/L2 kesh o'lchamlari bilan solishtiring. | oson | 5 |
| 4 | **Konvolyutsiya: parallellashtirish o'qi** — 4-qadam Numba konvolyutsiyasini uch xil prange o'qi (chiqish kanali, qator, kanal+qator collapse) bilan amalga oshiring; 1 va 2 ipda vaqt, GFLOP/s va tezlanishni jadvalga kiriting, yuk muvozanati va kesh lokalligi nuqtai nazaridan eng yaxshi variantni asoslang. | o'rta | 10 |
| 5 | **im2col xotira va vaqt savdosi** — 4-qadam im2col variantida CIN=16/32/64 va 96×96/160×160 kirishlar uchun cols matritsasining hajmini (MB) va vaqtni o'lchang; to'g'ridan-to'g'ri Numba va torch conv2d bilan solishtirib, qaysi hajmdan boshlab im2col xotira cheklovi tufayli foydasiz bo'lishini hisoblab ko'rsating. | o'rta | 10 |
| 6 | **Format va iplar matritsasi** — 5-qadamda FP32, BF16, FP16 va INT8 formatlarida n = 256, 512, 1024 va 1, 2 ip uchun GFLOP/s va nisbiy xato jadvalini tuzing; CPU'da past formatlar nima uchun tezlashmasligini va GPU Tensor Core'da (A2 nazariy qismidagi jadval) qanday nisbat kutilishini solishtirib yozing. | o'rta | 10 |
| 7 | **Yadro birlashtirish qo'lda** — 5-qadam GELU'sini Numba @njit(parallel=True) bilan bitta o'tishda (fused) yozing; eager torch, F.gelu va Numba fused variantlarini 1 va 2 ipda 8M element uchun o'lchab samarali GB/s ni hisoblang va xotira trafigi nazariy hisobi bilan solishtiring. | o'rta | 10 |
| 8 | **Multiprocessing vs threading vs Numba** — 6-qadam kadr_ishlov vazifasini uch usulda parallellashtiring: multiprocessing.Pool(2), concurrent.futures.ThreadPoolExecutor(2) va Numba prange; har biri uchun tezlanish, natija to'g'riligi va xotira sarfini (psutil) o'lchang, GIL ta'sirini threading natijasida ko'rsating va qaysi usul qachon tanlanishini jadval bilan asoslang. | qiyin | 15 |
| 9 | **Amdahl: o'lchangan p va bashorat tekshiruvi** — 7-qadam metodikasini 2–4-qadamlardagi barcha parallel variantlarga (prange matmul, prange konvolyutsiya, torch matmul, multiprocessing) qo'llab har biri uchun p ni baholang; ketma-ket qism ulushini qo'lda o'lchab (profil) Amdahl bahosi bilan solishtiring, farq sabablarini (xotira o'tkazuvchanligi, sinxronizatsiya) tushuntiring va jamlovchi grafik chizing. | qiyin | 15 |
| 10 | **CPU Roofline va GPU bashorati** — Barcha qadamlardagi o'lchovlarni (sof Python, NumPy, prange, tiled, im2col, torch conv2d, GELU eager/fused) arifmetik intensivlik–unumdorlik tekisligiga joylashtiring: sandbox CPU uchun cho'qqi GFLOP/s ni torch FP32 matmul'ning eng yaxshi natijasidan, xotira o'tkazuvchanligini reduktsiya (2-qadam) natijasidan oling; har nuqta qaysi chegara bilan cheklanganini tushuntiring va A2 matnidagi RTX 3060 ko'rsatkichlari bilan GPU'da kutilgan tezlanishni bashorat qiling. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.
