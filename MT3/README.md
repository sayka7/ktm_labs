# MT3. Sun'iy intellektga asoslangan zamonaviy kompyuter tizimlari arxitekturalarini (CPU, GPU, TPU, NPU va FPGA) qiyosiy tahlil qilish va ularning AI ilovalaridagi samaradorligini baholash

📄 Bosma qo'llanma (bosqichlar, kod, mashqlar, 20 variant): `MT3_laboratoriya_qollanma.pdf`
Bu mustaqil ishda siz besh xil hisoblash arxitekturasini analitik modellar va simulyatsiyalar orqali solishtirasiz: rasmiy cho'qqi qiymatlari bilan Roofline modeli, TPU sistolik massivining takt darajasidagi simulyatori, FPGA quvuri va fixed-point arifmetikasi, NPU uchun INT8 kvantlash, energiya va TCO kalkulyatori hamda yakuniy ko'p mezonli tanlov matritsasi. Real tezlatkichlar (GPU, TPU, NPU, FPGA) sandbox'da yo'q, shuning uchun ular modellar bilan baholanadi; yagona real o'lchov qurilmasi — 2 yadroli CPU, uning natijalari boshqa arxitekturalarga ekstrapolyatsiya qilinadi. Jadvaldagi barcha narx, quvvat va unumdorlik raqamlari taxminiy.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/MT3

## Bosqichlar

### 1-qadam. Arxitekturalar pasporti va Roofline modeli

CPU (AMD EPYC 9654), GPU (NVIDIA H100 SXM5), TPU (Google TPU v4), NPU (Hailo-8), chekka SoC (Jetson AGX Orin) va FPGA (AMD Alveo U250) uchun ochiq spetsifikatsiyalardagi cho'qqi qiymatlar (INT8/FP16 TOP/s, xotira o'tkazuvchanligi, quvvat) jadvali tuziladi. Har biri uchun ridge nuqtasi (cho'qqi / o'tkazuvchanlik) va TOP/s/Vt hisoblanadi. Keyin 6 ta tipik AI yadrosi (katta GEMM, ResNet conv3x3, BERT FFN, LLM dekod GEMV, depthwise conv, ReLU) uchun arifmetik intensivlik formuladan topiladi va Roofline bo'yicha erishiladigan unumdorlik hamda chegara turi (hisoblash yoki xotira) ko'rsatiladi. Log-log Roofline grafigi chiziladi. Barcha cho'qqi qiymatlari taxminiy — ular avlod, chastota va rejimga qarab o'zgaradi.

Fayl: `s1_1_qadam_arxitekturalar_pasporti_va_roofl.py` — ishga tushirish: `python3 s1_1_qadam_arxitekturalar_pasporti_va_roofl.py`

Kutilgan natija: Pasport jadvali, AI jadvali (GEMM ~2700, GEMV ~2 op/bayt), Roofline jadvali: katta GEMM hamma joyda H, GEMV/ReLU hamma joyda X; LLM dekod token/s ~ BW/7 GB; grafik.

Tajriba qilib ko'ring:

- Jadvalga NVIDIA A100 yoki L4 ni qo'shing (ma'lumot varag'idan qiymatlarni o'zingiz toping) va ridge nuqtasini solishtiring.
- BERT FFN da token sonini 128 dan 2048 ga oshiring — H100 qachon hisoblash chegarasiga o'tadi?
- Yuklamalarni FP16 (2 bayt/element) bilan qayta hisoblang: AI va chegara turi qanday o'zgaradi?

### 2-qadam. TPU sistolik massivi: takt darajasidagi simulyator va MXU foydalanish modeli

NumPy'da output-stationary sistolik massiv takt-ma-takt simulyatsiya qilinadi: A satrlari chapdan kechikish (skew) bilan, B ustunlari yuqoridan oqib kiradi, registrlar har taktda siljiydi. Natija A@B bilan aynan mos kelishi, taktlar soni K+R+C-2 formulasiga tengligi va PE foydalanish ulushi tekshiriladi; t=6 dagi diagonal to'lqin fronti ko'rsatiladi. So'ngra TPU v4 ning 128x128 MXU (8 dona, 1,05 GHz = 275 TFLOP/s) uchun weight-stationary analitik model bilan real qatlamlarda (BERT, ResNet, LLM dekod, toq o'lcham) foydalanish va samarali TFLOP/s hisoblanadi.

Fayl: `s2_2_qadam_tpu_sistolik_massivi_takt_daraja.py` — ishga tushirish: `python3 s2_2_qadam_tpu_sistolik_massivi_takt_daraja.py`

Kutilgan natija: Simulyatorda xato 0; K=256 da foydalanish ~90 %; MXU: BERT 4096 token ~94 %, GEMV M=1 ~0,4 %, toq N=130 ~45 %; grafik.

Tajriba qilib ko'ring:

- Massivni weight-stationary rejimga o'zgartiring (B oldindan PE'larga yuklanadi) va taktlar formulasini chiqaring.
- MXU o'lchamini 256x256 ga oshiring (TPU v6e kabi) — kichik qatlamlarda foydalanish qanday o'zgaradi?
- Toq o'lcham N=130 uchun 128 ga moslash (padding) va 2 ta plitka o'rniga N ni 128 ga qisqartirish samarasini solishtiring.

### 3-qadam. FPGA: quvur (pipeline), II va fixed-point aniqlik simulyatsiyasi

Birinchi qismda 4 bosqichli MAC quvuri vaqt diagrammasi bilan simulyatsiya qilinadi: initiation interval (II) 1, 2, 4 bo'lganda o'tkazuvchanlik qanday tushishi ko'rsatiladi, Alveo U250 (~12288 DSP, 300 MHz) uchun INT16/INT8/FP32 rejimlarda TOP/s bahosi chiqariladi. Ikkinchi qismda digits ustida o'qitilgan MLP 64-32-10 butunlay butun sonli (fixed-point) arifmetikada bajariladi: Qm.n formatlari, to'yinish (saturation), siljitish orqali masshtab va akkumulyator kengligi. Har format uchun aniqlik, SQNR va to'yingan og'irliklar soni jadvalda.

Fayl: `s3_3_qadam_fpga_quvur_pipeline_ii_va_fixed_.py` — ishga tushirish: `python3 s3_3_qadam_fpga_quvur_pipeline_ii_va_fixed_.py`

Kutilgan natija: II=1 da 8 kirish 11 taktda; INT8 packing ~12 TOP/s (taxminiy); 16 bitda aniqlik suzuvchi nuqta bilan bir xil (~0,97), Q0.7 da to'yinish va keskin tushish, 12 bitli akkumulyatorda ~0,6.

Tajriba qilib ko'ring:

- Quvurga 2 taktli xotira o'qish bosqichini qo'shing (5 bosqich) — kechikish va o'tkazuvchanlik qanday o'zgaradi?
- Yashirin qatlamni 128 neyronga oshiring: 8 bitli akkumulyator yetarli bo'lishi uchun nechta bit kerak? (log2(K) qoidasi)
- Har qatlamga alohida Q format tanlang (qatlam bo'yicha diapazon) va aniqlikni solishtiring.

### 4-qadam. NPU uchun INT8 kvantlash: aniqlik, hajm, kechikish va grafik bo'linishi

Kichik CNN digits ustida o'qitilib, PyTorch eager statik kvantlash (qnnpack) bilan INT8 ga o'tkaziladi: conv+relu birlashtiriladi, per-tensor va per-channel og'irlik kvantlash, 8 va 256 namunali kalibrlash solishtiriladi. Aniqlik, model hajmi va batch 1/64 kechikishi o'lchanadi. So'ng har qatlam og'irliklarining simmetrik INT8 SQNR'i hisoblanadi. Oxirida NPU qo'llamaydigan operatorlar (SiLU, NMS) CPU'ga qaytganda PCIe orqali qatnov kechikishi modellashtiriladi va SiLU -> ReLU almashtirish samarasi ko'rsatiladi.

Fayl: `s4_4_qadam_npu_uchun_int8_kvantlash_aniqlik.py` — ishga tushirish: `python3 s4_4_qadam_npu_uchun_int8_kvantlash_aniqlik.py`

Kutilgan natija: FP32 va INT8 aniqligi deyarli teng (~0,95), hajm ~3,5x kichik; CPU'da batch=1 da INT8 tez bo'lmasligi mumkin, batch 64 da tezroq; SQNR ~40-47 dB; bo'linish modelida o'tishlar 5 -> 3, kechikish ~3,0 -> 2,4 ms.

Tajriba qilib ko'ring:

- Kalibrlash uchun HistogramObserver ishlating va aniqlik farqini o'lchang.
- Modelga SiLU qo'shing va kvantlangan model qaysi qatlamda dequant/quant qilishini ko'ring (print(q)).
- PCIE_GBS ni 0,5 GB/s (USB) ga o'zgartiring — Edge TPU USB kabi qurilmada bo'linish narxi qancha?

### 5-qadam. Real CPU'da GEMM va konvolyutsiya o'lchovlari, boshqa arxitekturalarga ekstrapolyatsiya

Sandbox CPU (2 yadro) uchun haqiqiy Roofline quriladi: xotira o'tkazuvchanligi STREAM triad bilan, amaliy cho'qqi esa eng yaxshi FP32 GEMM bilan o'lchanadi. GEMM 64...1024, GEMV, conv3x3, conv1x1, depthwise va ReLU uchun GFLOP/s, arifmetik intensivlik va Roofline'ga nisbatan samaradorlik jadvali chiqadi. Keyin har yadro samaradorligi saqlanadi deb faraz qilinib, 1-qadam pasportidagi arxitekturalarda kutilgan vaqt (ishga tushirish xarajati bilan) ekstrapolyatsiya qilinadi. Bu — birinchi yaqinlashish, aniq bashorat emas.

Fayl: `s5_5_qadam_real_cpuda_gemm_va_konvolyutsiya.py` — ishga tushirish: `python3 s5_5_qadam_real_cpuda_gemm_va_konvolyutsiya.py`

Kutilgan natija: Triad ~20-40 GB/s, GEMM cho'qqisi ~40-60 GFLOP/s (mashinaga bog'liq); depthwise samaradorligi < 10 %; ekstrapolyatsiyada kichik yadrolar tezlatkichlarda 5-10 us 'pol'ga uriladi.

Tajriba qilib ko'ring:

- torch.set_num_threads(1) bilan qayta o'lchang — cho'qqi va ridge nuqtasi qanday o'zgaradi?
- GEMM'ni float64 da o'lchang (AI 2 marta kamayadi) va Roofline'dagi joyini ko'ring.
- Ishga tushirish xarajatini 0 qiling — qaysi yadrolar uchun bashorat keskin o'zgaradi?

### 6-qadam. Energiya samaradorligi (TOP/s/Vt, mJ/tasvir) va 3 yillik TCO

ResNet-50 sinfidagi tasnif (~8,2 GOP/tasvir) va 5000 tasvir/s talab uchun har arxitekturada kerakli qurilmalar soni, umumiy quvvat, energiya (mJ/tasvir), CAPEX, elektr xarajati (PUE bilan) va egalik/bulut TCO hisoblanadi; natija bitta mln tasvir narxiga keltiriladi. Elektr narxi bo'yicha sezgirlik, 30 % yuklamada bulut va egalik solishtiruvi hamda chekka stsenariysi (120 tasvir/s, 30 Vt) ko'rib chiqiladi. Narx, quvvat va real foydalanish koeffitsientlari — taxminiy faraz, ular kod boshida o'zgartiriladi.

Fayl: `s6_6_qadam_energiya_samaradorligi_top_s_vt_.py` — ishga tushirish: `python3 s6_6_qadam_energiya_samaradorligi_top_s_vt_.py`

Kutilgan natija: TOP/s/Vt: NPU ~10, GPU ~2,8, CPU ~0,08; mJ/tasvir NPU ~3,5, GPU ~10, CPU ~470; 24/7 da GPU egaligi bulutdan arzon, 30 % yuklamada bulut afzal; chekkada faqat NPU mos.

Tajriba qilib ko'ring:

- Real foydalanish koeffitsientlarini MLPerf Inference natijalari asosida yangilang (manbani hisobotda ko'rsating).
- Talabni 500 tasvir/s ga tushiring — qaysi arxitektura eng arzon bo'ladi?
- Jetson'ni 30 Vt rejimida (unumdorlik ~0,5x) hisoblang va chekka cheklovini qayta tekshiring.

### 7-qadam. Yakuniy tanlov matritsasi: WSM, TOPSIS va vaznlar sezgirligi

Oldingi qadamlar natijalari 6 mezonli jadvalga yig'iladi (unumdorlik, kechikish, energiya, TCO, ekotizim, moslashuvchanlik) va qat'iy cheklovlar (quvvat, internetsiz ishlash) qo'llanadi. Uch ssenariy (data-markaz, chekka kamera, moliya) uchun vaznli yig'indi (WSM) va TOPSIS reytinglari hisoblanadi; Dirichlet taqsimoti bilan vaznlar 2000 marta tebratilib, 1-o'rin barqarorligi baholanadi. Natija issiqlik xaritasi ko'rinishidagi tanlov matritsasi.

Fayl: `s7_7_qadam_yakuniy_tanlov_matritsasi_wsm_to.py` — ishga tushirish: `python3 s7_7_qadam_yakuniy_tanlov_matritsasi_wsm_to.py`

Kutilgan natija: Data-markaz va moliyada GPU 1-o'rin (barqarorlik 80-98 %), chekkada Jetson/NPU — usullar kelishmasligi ko'rsatiladi; matritsa va grafik.

Tajriba qilib ko'ring:

- Data-markaz ssenariysiga 'quvvat_vt >= 100' (faqat server qurilmalari) cheklovini qo'shing va reytingni qayta ko'ring.
- Ekotizim va moslashuvchanlik ekspert baholarini o'zgartiring — tavsiya qachon o'zgaradi?
- To'rtinchi ssenariy (LLM dekod xizmati) qo'shing: xotira o'tkazuvchanligi mezoni bilan.

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
|---|---|---|---|
| 1 | **Pasport jadvalini tekshirish** — 1-qadam jadvalidagi har qurilma uchun cho'qqi INT8/FP16 qiymati, xotira o'tkazuvchanligi va quvvatni ishlab chiqaruvchining rasmiy ma'lumot varag'i (datasheet) bilan solishtiring; farq bo'lsa tuzating va manbalar ro'yxatini (havola yoki hujjat nomi, sana) jadval ostida bering. Ridge nuqtalarini qayta hisoblang. | oson | 5 |
| 2 | **O'z yuklamangizning AI si** — 1-qadamga o'zingiz tanlagan modelning 3 ta qatlamini (masalan, YOLO conv, Transformer attention QK^T, embedding qidiruv) qo'shing: amallar va baytlar formulasini yozing, AI ni hisoblang va har arxitekturada chegara turini (H/X) jadvalda ko'rsating. | oson | 5 |
| 3 | **Sistolik massiv to'g'riligi va to'lqin fronti** — 2-qadam simulyatorini 12x12 massiv va K=12, 48, 192 uchun ishga tushiring, taktlar formulasini tasdiqlang va t=5, 11, 17 dagi faol PE xaritalarini chiqaring. PE foydalanishning K ga bog'liqligi grafigini topshiring. | oson | 5 |
| 4 | **MXU uchun padding strategiyasi** — 2-qadam modelida N = 100...300 (qadam 10) uchun MXU foydalanishini hisoblang, 128 ga karrali bo'lmagan o'lchamlardagi 'arra tishi' grafigini chizing. Toq o'lchamli qatlam uchun 3 strategiyani (padding, o'lchamni qisqartirish, ikki qatlamni birlashtirish) samarali TFLOP/s bo'yicha solishtiring. | o'rta | 10 |
| 5 | **FPGA fixed-point dizaynini tanlash** — 3-qadamdagi MLP uchun har qatlamga alohida (og'irlik, faollashtirish, akkumulyator) bit kengligini tanlang: aniqlik pasayishi <= 0,5 p.p. bo'lgan eng tejamkor konfiguratsiyani toping. DSP soni va o'tkazuvchanlikni (300 MHz, II=1) modelda baholang va jadvalda bering. | o'rta | 10 |
| 6 | **Kalibrlash va observer tadqiqoti** — 4-qadamda MinMax, MovingAverageMinMax va Histogram observerlarini 4, 16, 64, 256 namunali kalibrlash bilan solishtiring (aniqlik, kalibrlash vaqti). Natijani 4x3 jadvalda va qisqa xulosada bering: NPU kompilyatoriga qaysi kalibrlash to'plamini berardingiz? | o'rta | 10 |
| 7 | **Sandbox CPU Roofline'ini aniqlashtirish** — 5-qadamda triad o'lchovini 1, 4, 16, 64 MB massivlar bilan takrorlang (kesh ierarxiyasi ko'rinsin) va GEMM'ni 1 va 2 ip bilan o'lchang. Kesh darajalari uchun alohida 'shift' chiziqli Roofline grafigini chizing va GEMV ning 100 % dan yuqori samaradorligini tushuntiring. | o'rta | 10 |
| 8 | **TCO modeliga o'z mintaqangiz narxlari** — 6-qadam modelini O'zbekiston sharoitiga moslang: elektr narxi (amaldagi tarif, manba bilan), import bojxona xarajatlari, sovutish (PUE) va xizmat ko'rsatish (yillik 10 % CAPEX) qo'shing. 3 ta talab darajasi (500, 5000, 50000 tasvir/s) uchun TCO va 1 mln tasvir narxini hisoblang; natija qaysi arxitekturani tanlashga ta'sir qilishini tahlil qiling. Barcha farazlarni taxminiy deb belgilang. | qiyin | 15 |
| 9 | **Ekstrapolyatsiyani tekshirish** — 5-qadam usuli bilan ONNX Runtime'da kichik CNN (yoki digits MLP) inferensini sandbox CPU'da o'lchang, so'ng uni qatlam bo'yicha Roofline modelida jamlab boshqa arxitekturalar uchun bashorat qiling. Bashoratni ochiq manbadagi (MLPerf Inference yoki ishlab chiqaruvchi benchmark) mos model natijasi bilan solishtiring va xato manbalarini (ishga tushirish xarajati, fusion, kesh) tahlil qiling. | qiyin | 15 |
| 10 | **To'liq qiyosiy hisobot va tanlov matritsasi** — O'z AI ilovangiz (masalan, shahar kameralari, tibbiy tasvir, chatbot) uchun 1-7 qadamlarni moslab, CPU/GPU/TPU/NPU/FPGA bo'yicha yakuniy hisobot tayyorlang: pasport, Roofline, kvantlash ta'siri, energiya va TCO, kamida 3 ssenariyli tanlov matritsasi va vaznlar sezgirligi. Tavsiyani raqamlar bilan asoslang (kamida 400 so'z), barcha taxminiy qiymatlarni alohida belgilang. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Tibbiyot: KT tasvirlarida o'smani segmentlash serveri (oson)

Soha: Tibbiyot (radiologiya)

Viloyat diagnostika markazi KT kesimlarini U-Net bilan segmentlovchi server sotib olmoqchi. Kuniga ~3000 kesim, har biri 512x512, natija 2 daqiqa ichida kerak. Bemor ma'lumoti binodan chiqmasligi sababli bulut faqat taqqoslash uchun ko'riladi. Siz CPU, GPU va FPGA ni Roofline, energiya va TCO bo'yicha solishtirib, tavsiya berasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | U-Net conv3x3 512x512x64->64 va 1x1 qatlam |
| Talab | 3000 kesim/kun, javob <= 120 s |
| Cheklov | bulutsiz, 1 kVt gacha |
| Solishtiriladi | CPU vs GPU vs FPGA |
| Fixed-point format | Q2.5 / Q4.11 |
| Narx farazi | elektr 0,08 $/kVt*soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (U-Net conv3x3 512x512x64->64 va 1x1 qatlam) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 3- va 4-qadamlar: fixed-point (Q2.5 / Q4.11) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
4. 6-qadam: talab (3000 kesim/kun, javob <= 120 s) va cheklov (bulutsiz, 1 kVt gacha) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (elektr 0,08 $/kVt*soat); sezgirlik tahlili.
5. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 2-variant. Agro: dron tasvirlarida begona o'tni aniqlash (oson)

Soha: Qishloq xo'jaligi

Paxta dalasida dron 30 kadr/s 1280x720 tasvir oladi, begona o'tlar real vaqtda aniqlanib purkagichga signal beriladi. Dron batareyasi chekli: hisoblash moduli 15 Vt dan oshmasligi kerak. Hailo-8 kabi NPU va Jetson sinfidagi SoC bilan bort CPU ni solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | YOLO uslubidagi depthwise conv + 1x1 |
| Talab | 30 kadr/s, kechikish <= 50 ms |
| Cheklov | <= 15 Vt, internetsiz |
| Solishtiriladi | CPU vs NPU vs Jetson |
| Fixed-point format | Q1.6 / Q3.4 |
| Narx farazi | batareya 100 Vt*soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs NPU vs Jetson + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (YOLO uslubidagi depthwise conv + 1x1) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 3- va 4-qadamlar: fixed-point (Q1.6 / Q3.4) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
4. 6-qadam: talab (30 kadr/s, kechikish <= 50 ms) va cheklov (<= 15 Vt, internetsiz) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (batareya 100 Vt*soat); sezgirlik tahlili.
5. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 3-variant. Transport: avtomobil raqamini tanish (ANPR) shlyuzi (oson)

Soha: Transport

Shahar chorrahalarida 16 ta kamera, har biridan 10 kadr/s; raqam aniqlash va OCR mahalliy shlyuzda bajariladi. Shlyuz shkafda turadi, sovutish passiv, 60 Vt gacha. Arxitekturalarni kadr/s/Vt va 5 yillik TCO bo'yicha baholaysiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | conv3x3 + kichik LSTM/GEMM (OCR) |
| Talab | 160 kadr/s |
| Cheklov | <= 60 Vt, passiv sovutish |
| Solishtiriladi | CPU vs Jetson vs FPGA |
| Fixed-point format | Q3.4 |
| Narx farazi | 5 yil, 0,10 $/kVt*soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs Jetson vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (conv3x3 + kichik LSTM/GEMM (OCR)) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 3- va 4-qadamlar: fixed-point (Q3.4) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
4. 6-qadam: talab (160 kadr/s) va cheklov (<= 60 Vt, passiv sovutish) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (5 yil, 0,10 $/kVt*soat); sezgirlik tahlili.
5. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 4-variant. Telekom: tarmoq trafigidagi anomaliyalarni aniqlash (oson)

Soha: Telekom

Operator yadro tarmog'ida har soniyada 2 mln paket metadatasi avtoenkoder bilan tekshiriladi. Kechikish 1 ms dan oshsa paket navbati to'lib ketadi. FPGA (SmartNIC) bilan GPU va CPU ni GEMV-og'ir yuklamada solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | MLP avtoenkoder 64-32-8-32-64 (GEMV, batch kichik) |
| Talab | 2 mln namuna/s, p99 <= 1 ms |
| Cheklov | server shkafi, 300 Vt |
| Solishtiriladi | CPU vs GPU vs FPGA |
| Fixed-point format | Q1.14 (16 bit) |
| Narx farazi | elektr 0,12 $/kVt*soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (MLP avtoenkoder 64-32-8-32-64 (GEMV, batch kichik)) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 3- va 4-qadamlar: fixed-point (Q1.14 (16 bit)) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
4. 6-qadam: talab (2 mln namuna/s, p99 <= 1 ms) va cheklov (server shkafi, 300 Vt) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (elektr 0,12 $/kVt*soat); sezgirlik tahlili.
5. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 5-variant. Energetika: quyosh paneli nuqsonlarini termal tasvirda topish (oson)

Soha: Energetika

Quyosh elektr stansiyasida 40 ming panel, dron termal kamerasi bilan haftasiga bir marta suratga olinadi (~200 ming tasvir). Ishlov kechasi paketli bajariladi, kechikish muhim emas, narx muhim. Bulut TPU, egalikdagi GPU va CPU klasterini solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | ResNet-18 sinfidagi tasnif (1,8 GMAC) |
| Talab | 200 ming tasvir 8 soatda |
| Cheklov | kechikish muhim emas |
| Solishtiriladi | CPU vs GPU vs TPU |
| Fixed-point format | BF16 |
| Narx farazi | bulut TPU soatbay |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs TPU + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (ResNet-18 sinfidagi tasnif (1,8 GMAC)) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 3- va 4-qadamlar: fixed-point (BF16) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
4. 6-qadam: talab (200 ming tasvir 8 soatda) va cheklov (kechikish muhim emas) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (bulut TPU soatbay); sezgirlik tahlili.
5. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 6-variant. Moliya: kartadagi firibgarlikni real vaqtda baholash (oson)

Soha: Moliya (bank)

Bank protsessing markazida har tranzaksiya 5 ms ichida gradient boosting + kichik MLP bilan baholanishi kerak; cho'qqida 8000 tranzaksiya/s. Ishonchlilik uchun ikki ma'lumot markazi. CPU, GPU va FPGA ni p99 kechikish va TCO bo'yicha solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | MLP 256-128-1 + daraxtlar (batch 1) |
| Talab | 8000 tranz./s, p99 <= 5 ms |
| Cheklov | 2 DC, zaxira 2N |
| Solishtiriladi | CPU vs GPU vs FPGA |
| Fixed-point format | Q3.12 |
| Narx farazi | 3 yil, 2 ta DC |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (MLP 256-128-1 + daraxtlar (batch 1)) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 3- va 4-qadamlar: fixed-point (Q3.12) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
4. 6-qadam: talab (8000 tranz./s, p99 <= 5 ms) va cheklov (2 DC, zaxira 2N) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (3 yil, 2 ta DC); sezgirlik tahlili.
5. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 7-variant. Ta'lim: imtihon videonazorati (proktoring) xizmati (oson)

Soha: Ta'lim

Universitet onlayn imtihonida 2000 talaba bir vaqtda, har biridan 2 kadr/s yuz va ko'z harakati tahlili qilinadi. Imtihonlar yiliga 30 kun, qolgan vaqt yuklama past. Egalikdagi GPU serveri va bulut GPU/TPU ni yuklama koeffitsienti bo'yicha solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | MobileNetV2 (0,3 GMAC) + yuz nuqtalari |
| Talab | 4000 kadr/s, 30 kun/yil |
| Cheklov | mavsumiy yuklama |
| Solishtiriladi | GPU egalik vs bulut GPU vs TPU |
| Fixed-point format | INT8 |
| Narx farazi | yuklama koeffitsienti 8 % |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (GPU egalik vs bulut GPU vs TPU + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (MobileNetV2 (0,3 GMAC) + yuz nuqtalari) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 3- va 4-qadamlar: fixed-point (INT8) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
4. 6-qadam: talab (4000 kadr/s, 30 kun/yil) va cheklov (mavsumiy yuklama) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (yuklama koeffitsienti 8 %); sezgirlik tahlili.
5. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 8-variant. Sanoat: konveyerda metall quyma nuqsonini aniqlash (o'rta)

Soha: Sanoat (metallurgiya)

Quyish sexida konveyer 2 m/s tezlikda, 4 kamera 120 kadr/s. Nuqsonli detal 20 ms ichida ajratilishi kerak, sex harorati yuqori va chang ko'p. FPGA (deterministik kechikish) va Jetson ni CPU bilan solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | conv3x3 256x256x32->32, 5 qatlam |
| Talab | 480 kadr/s, kechikish <= 20 ms (qat'iy) |
| Cheklov | IP65 shkaf, 100 Vt |
| Solishtiriladi | CPU vs Jetson vs FPGA |
| MXU/sistolik holat | N=32 |
| Fixed-point format | Q2.5 / Q2.13 |
| Narx farazi | elektr 0,07 $/kVt*soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs Jetson vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (conv3x3 256x256x32->32, 5 qatlam) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (N=32) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (Q2.5 / Q2.13) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (480 kadr/s, kechikish <= 20 ms (qat'iy)) va cheklov (IP65 shkaf, 100 Vt) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (elektr 0,07 $/kVt*soat); sezgirlik tahlili.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 9-variant. Xavfsizlik: aeroportda bagaj rentgen tasvirini skrining (o'rta)

Soha: Xavfsizlik

Aeroport bagaj liniyasida soatiga 1800 sumka, har biri 4 proyeksiyali rentgen tasvir. Taqiqlangan buyumni aniqlash detektori 1 s ichida operatorga ko'rsatishi kerak. GPU serveri, FPGA va CPU ni ishonchlilik va TCO bo'yicha taqqoslaysiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | Faster R-CNN sinfidagi detektor (~90 GMAC) |
| Talab | 2 tasvir/s, javob <= 1 s |
| Cheklov | ishonchlilik 99,99 % |
| Solishtiriladi | CPU vs GPU vs FPGA |
| MXU/sistolik holat | M=4 |
| Fixed-point format | Q4.11 |
| Narx farazi | 7 yil xizmat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (Faster R-CNN sinfidagi detektor (~90 GMAC)) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (M=4) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (Q4.11) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (2 tasvir/s, javob <= 1 s) va cheklov (ishonchlilik 99,99 %) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (7 yil xizmat); sezgirlik tahlili.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 10-variant. Chakana savdo: do'kon javonlarini monitoring qilish (o'rta)

Soha: Chakana savdo

Supermarketlar tarmog'i 120 do'konda javonlardagi bo'sh joylarni har 5 daqiqada tekshiradi; har do'konda 20 kamera. Variant: har do'konda chekka qurilma yoki markaziy GPU serveri. Energiya va trafik narxi bilan 3 yillik TCO hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | EfficientDet-D0 (2,5 GMAC) |
| Talab | 120 do'kon x 20 kamera / 5 daq |
| Cheklov | do'konda 20 Vt gacha |
| Solishtiriladi | NPU (har do'kon) vs markaziy GPU vs CPU |
| MXU/sistolik holat | M=8 |
| Fixed-point format | INT8 per-channel |
| Narx farazi | trafik 0,05 $/GB |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (NPU (har do'kon) vs markaziy GPU vs CPU + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (EfficientDet-D0 (2,5 GMAC)) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (M=8) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (INT8 per-channel) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (120 do'kon x 20 kamera / 5 daq) va cheklov (do'konda 20 Vt gacha) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (trafik 0,05 $/GB); sezgirlik tahlili.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 11-variant. Logistika: ombor robotlari uchun yo'l rejalashtirish (o'rta)

Soha: Logistika

Avtomatlashtirilgan omborda 150 robot, har biri 10 Gs chastotada lidar nuqtalar bulutini kichik PointNet bilan tahlil qiladi. Hisoblash robot bortida (Jetson) yoki markaziy serverda (Wi-Fi kechikishi 15 ms) bo'lishi mumkin. Variantlarni kechikish va energiya bo'yicha solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | PointNet: GEMM 1024x64->128 va max-pool |
| Talab | 150 robot x 10 Gs |
| Cheklov | bort batareyasi, 25 Vt |
| Solishtiriladi | Jetson (bort) vs GPU (markaz) vs CPU |
| MXU/sistolik holat | M=1024 |
| Fixed-point format | FP16 |
| Narx farazi | Wi-Fi RTT 15 ms |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (Jetson (bort) vs GPU (markaz) vs CPU + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (PointNet: GEMM 1024x64->128 va max-pool) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (M=1024) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (FP16) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (150 robot x 10 Gs) va cheklov (bort batareyasi, 25 Vt) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (Wi-Fi RTT 15 ms); sezgirlik tahlili.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 12-variant. Shahar: aqlli svetofor uchun transport oqimini bashorat qilish (o'rta)

Soha: Shahar infratuzilmasi

Shahar 300 chorrahasida har 10 s da GNN/LSTM bilan keyingi 5 daqiqa oqimi bashorat qilinadi. Hisob markaziy data-markazda, kechikish 2 s gacha. GEMV-og'ir LSTM yuklamada xotira o'tkazuvchanligi hal qiluvchi ekanini Roofline bilan ko'rsatasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | LSTM 512 yashirin, 300 ketma-ketlik |
| Talab | 30 bashorat/s, kechikish <= 2 s |
| Cheklov | data-markaz |
| Solishtiriladi | CPU vs GPU vs TPU |
| MXU/sistolik holat | M=300 |
| Fixed-point format | BF16 |
| Narx farazi | PUE 1,6 |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs TPU + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (LSTM 512 yashirin, 300 ketma-ketlik) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (M=300) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (BF16) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (30 bashorat/s, kechikish <= 2 s) va cheklov (data-markaz) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (PUE 1,6); sezgirlik tahlili.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 13-variant. Ekologiya: havo sifati sensorlarida chekka tahlil (o'rta)

Soha: Ekologiya

Shahar bo'ylab 500 ta havo sifati stansiyasi quyosh paneli bilan ishlaydi (o'rtacha 5 Vt). Har stansiya har daqiqada 1D-CNN bilan ifloslanish manbasini tasniflaydi. Mikro-NPU, kichik CPU (Raspberry Pi sinfi) va bulutga yuborishni energiya bo'yicha solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | 1D-CNN 6 qatlam, 1440 nuqta |
| Talab | 1 tasnif/daq/stansiya |
| Cheklov | <= 5 Vt, quyosh |
| Solishtiriladi | CPU (RPi) vs NPU vs bulut |
| MXU/sistolik holat | N=24 |
| Fixed-point format | Q1.6 (8 bit) |
| Narx farazi | LTE trafik 2 $/GB |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU (RPi) vs NPU vs bulut + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (1D-CNN 6 qatlam, 1440 nuqta) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (N=24) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (Q1.6 (8 bit)) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (1 tasnif/daq/stansiya) va cheklov (<= 5 Vt, quyosh) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (LTE trafik 2 $/GB); sezgirlik tahlili.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 14-variant. Sport: futbol o'yinida o'yinchilarni kuzatish (o'rta)

Soha: Sport

Stadionda 8 kamera 50 kadr/s 4K oqim beradi, o'yinchi va to'p koordinatalari 100 ms kechikish bilan efirga chiqariladi. O'yinlar haftasiga 2 marta, 2 soatdan. Mobil GPU serveri va bulut GPU ni foydalanish koeffitsienti bilan baholaysiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | YOLOv8-m sinfidagi detektor (~40 GMAC) |
| Talab | 400 kadr/s, <= 100 ms |
| Cheklov | mobil stend, 2 kVt |
| Solishtiriladi | GPU egalik vs bulut GPU vs FPGA |
| MXU/sistolik holat | M=8 |
| Fixed-point format | FP16 / INT8 |
| Narx farazi | haftada 4 soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (GPU egalik vs bulut GPU vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (YOLOv8-m sinfidagi detektor (~40 GMAC)) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (M=8) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (FP16 / INT8) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (400 kadr/s, <= 100 ms) va cheklov (mobil stend, 2 kVt) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (haftada 4 soat); sezgirlik tahlili.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 15-variant. Media: video kontentni avtomatik subtitrlash (o'rta)

Soha: Media

Teleradiokompaniya arxivida 20 ming soat video; Whisper sinfidagi model bilan o'zbek tilida subtitr yaratiladi. Kodlovchi qism hisobga, dekoder xotiraga og'ir. Bulut TPU, GPU va CPU klasterida arxivni 30 kunda qayta ishlash narxini solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | Transformer kodlovchi (GEMM 1500x512x2048) + dekoder GEMV |
| Talab | 20 ming soat / 30 kun |
| Cheklov | faqat narx va muddat |
| Solishtiriladi | CPU vs GPU vs TPU |
| MXU/sistolik holat | M=1500 (kodlovchi), M=1 (dekoder) |
| Fixed-point format | BF16 |
| Narx farazi | bulut soatbay |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs TPU + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (Transformer kodlovchi (GEMM 1500x512x2048) + dekoder GEMV) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (M=1500 (kodlovchi), M=1 (dekoder)) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (BF16) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (20 ming soat / 30 kun) va cheklov (faqat narx va muddat) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (bulut soatbay); sezgirlik tahlili.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan)

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 16-variant. Tibbiyot: EKG signalidan aritmiyani taqiladigan qurilmada aniqlash (qiyin)

Soha: Tibbiyot (kardiologiya)

Taqiladigan EKG qurilmasi 250 Gs signalni 1D-CNN bilan har 10 s da tahlil qiladi, batareya 7 kun yetishi shart (400 mAh, 3,7 V). Mikro-NPU, FPGA (past quvvatli) va MCU-CPU uchun energiya byudjeti va fixed-point aniqlikni hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | 1D-CNN 8 qatlam, 2500 nuqta |
| Talab | 1 tahlil/10 s |
| Cheklov | batareya 7 kun, < 10 mVt o'rtacha |
| Solishtiriladi | CPU (MCU) vs NPU vs FPGA |
| MXU/sistolik holat | N=16 |
| Fixed-point format | Q1.6 / Q1.14 |
| Narx farazi | akkumulyator 1,48 Vt*soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU (MCU) vs NPU vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (1D-CNN 8 qatlam, 2500 nuqta) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (N=16) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (Q1.6 / Q1.14) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (1 tahlil/10 s) va cheklov (batareya 7 kun, < 10 mVt o'rtacha) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (akkumulyator 1,48 Vt*soat); sezgirlik tahlili.
7. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan); Manbalar ro'yxati (datasheet, MLPerf) va farazlar jadvali

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 17-variant. Neft-gaz: quvurda oqishni akustik signal bo'yicha aniqlash (qiyin)

Soha: Neft-gaz

Magistral quvur bo'ylab har 5 km da akustik sensor, 48 kGs signal spektrogrammasi CNN bilan tahlil qilinadi. Stansiyalar uzoqda, quvvat quyosh+akkumulyator (30 Vt), aloqa sun'iy yo'ldosh orqali qimmat. Chekkada FPGA va Jetson ni solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | spektrogramma CNN 128x128, 4 qatlam |
| Talab | har stansiya 2 tahlil/s |
| Cheklov | <= 30 Vt, sun'iy yo'ldosh aloqasi |
| Solishtiriladi | Jetson vs FPGA vs NPU |
| MXU/sistolik holat | N=64 |
| Fixed-point format | Q3.12 |
| Narx farazi | aloqa 5 $/MB |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (Jetson vs FPGA vs NPU + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (spektrogramma CNN 128x128, 4 qatlam) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (N=64) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (Q3.12) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (har stansiya 2 tahlil/s) va cheklov (<= 30 Vt, sun'iy yo'ldosh aloqasi) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (aloqa 5 $/MB); sezgirlik tahlili.
7. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan); Manbalar ro'yxati (datasheet, MLPerf) va farazlar jadvali

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 18-variant. Bank: chatbot uchun LLM inferens klasteri (qiyin)

Soha: Moliya (mijozlarga xizmat)

Bank mijozlarga 7B parametrli LLM asosidagi chatbotni ichki serverda joylashtirmoqchi: cho'qqida 200 parallel suhbat, javob tezligi kamida 15 token/s. Dekod bosqichi xotira o'tkazuvchanligi bilan cheklangani sababli HBM'li GPU, TPU va CPU ni token/s/$ bo'yicha solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | LLM 7B: dekod GEMV 4096x4096, prefill GEMM |
| Talab | 200 suhbat x 15 token/s |
| Cheklov | ma'lumot ichkarida |
| Solishtiriladi | CPU vs GPU vs TPU |
| MXU/sistolik holat | dekod M=1..64 |
| Fixed-point format | INT8 og'irlik / FP16 |
| Narx farazi | 3 yil, 0,10 $/kVt*soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs TPU + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (LLM 7B: dekod GEMV 4096x4096, prefill GEMM) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (dekod M=1..64) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (INT8 og'irlik / FP16) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (200 suhbat x 15 token/s) va cheklov (ma'lumot ichkarida) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (3 yil, 0,10 $/kVt*soat); sezgirlik tahlili.
7. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan); Manbalar ro'yxati (datasheet, MLPerf) va farazlar jadvali

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 19-variant. Kosmik monitoring: sun'iy yo'ldosh tasvirlarida ekin turlarini tasniflash (qiyin)

Soha: Masofaviy zondlash

Davlat agentligi har 5 kunda respublika hududining 10 m aniqlikdagi multispektral tasvirlarini (13 kanal) yangilaydi: ~45 ming kvadrat 256x256. Tasniflash 12 soatda tugashi kerak. Egalikdagi GPU, bulut TPU va FPGA ni katta batch rejimida solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | U-Net yengil, 13 kanal kirish |
| Talab | 45 ming plitka / 12 soat |
| Cheklov | paketli, kechikish muhim emas |
| Solishtiriladi | GPU vs TPU vs FPGA |
| MXU/sistolik holat | toq K=13*9=117 |
| Fixed-point format | BF16 / INT8 |
| Narx farazi | 5 kunda 1 ishga tushirish |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (GPU vs TPU vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (U-Net yengil, 13 kanal kirish) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (toq K=13*9=117) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (BF16 / INT8) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (45 ming plitka / 12 soat) va cheklov (paketli, kechikish muhim emas) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (5 kunda 1 ishga tushirish); sezgirlik tahlili.
7. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan); Manbalar ro'yxati (datasheet, MLPerf) va farazlar jadvali

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)

## 20-variant. Kiberxavfsizlik: zararli fayllarni bayt-ketma-ketlik bo'yicha aniqlash (qiyin)

Soha: Kiberxavfsizlik

Korporativ pochta shlyuzida kuniga 1,5 mln ilova MalConv uslubidagi 1D-CNN bilan tekshiriladi, cho'qqi 60 fayl/s, kechikish 300 ms. Shlyuz 1U serverga sig'ishi kerak (bitta PCIe slot). GPU, FPGA va CPU ni slot, quvvat va TCO bo'yicha solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Yuklama/model | 1D-CNN embedding 8 + conv 512 filtr, 2 MB kirish |
| Talab | 60 fayl/s, p95 <= 300 ms |
| Cheklov | 1U, bitta PCIe, 250 Vt |
| Solishtiriladi | CPU vs GPU vs FPGA |
| MXU/sistolik holat | M=1..16 |
| Fixed-point format | Q3.4 (8 bit) |
| Narx farazi | elektr 0,10 $/kVt*soat |

Qadamlar:

1. 1-qadam kabi pasport jadvalini tuzing (CPU vs GPU vs FPGA + kamida yana bitta arxitektura), ridge nuqtasi va TOP/s/Vt ni hisoblang; manbalarni ko'rsating.
2. Variant yuklamasi (1D-CNN embedding 8 + conv 512 filtr, 2 MB kirish) uchun amallar va baytlar formulasini yozing, AI ni hisoblang va Roofline grafigida chegara turini (H/X) ko'rsating.
3. 2-qadam sistolik simulyatorini variant holati (M=1..16) uchun ishlating: to'g'rilik, taktlar va MXU foydalanish foizi.
4. 3- va 4-qadamlar: fixed-point (Q3.4 (8 bit)) yoki INT8 kvantlashda kichik muqobil modelning (digits/sintetik) aniqlik pasayishi va SQNR'i; NPU bo'linishi kerak bo'lsa modellang.
5. 5-qadam kabi sandbox CPU'da variant yadrosiga o'xshash GEMM/conv ni o'lchab, samaradorlikni topib, tanlangan arxitekturalarga ekstrapolyatsiya qiling.
6. 6-qadam: talab (60 fayl/s, p95 <= 300 ms) va cheklov (1U, bitta PCIe, 250 Vt) bo'yicha qurilmalar soni, quvvat, mJ/inferens va TCO (elektr 0,10 $/kVt*soat); sezgirlik tahlili.
7. 7-qadam: variant ssenariysi uchun vaznlar va qat'iy cheklovlar bilan WSM/TOPSIS tanlov matritsasi, Monte-Karlo barqarorligi va yakuniy tavsiya.

Topshiriladi: Kod (.py, har qadam alohida); Pasport va natijalar jadvallari; Roofline va TCO grafiklari (PNG); Xulosa (kamida 150 so'z, taxminiy raqamlar belgilangan); Manbalar ro'yxati (datasheet, MLPerf) va farazlar jadvali

Baholash: To'g'ri ishlaydigan kod — 30, Hisob-kitob va jadvallar — 25, Tahlil, tavsiya va asoslash — 25, Hisobot sifati — 20 (jami 100)
