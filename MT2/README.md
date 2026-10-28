# MT2. AI modelini optimallashtirish: latency, throughput va energiya samaradorligi tahlili (ONNX Runtime, TensorRT, OpenVINO)

📄 Bosma qo'llanma (bosqichlar, kod, mashqlar, 20 variant): `MT2_laboratoriya_qollanma.pdf`
Bu mustaqil ishda akselerometr signallari bo'yicha harakat turini (yurish, yugurish, zina, yiqilish) aniqlovchi depthwise-separable 1D CNN chekka qurilma va server uchun optimallashtiriladi: to'g'ri o'lchov metodikasi, ONNX eksport va operatorlar profili, FP16/BF16/INT8 formatlari, dinamik paketlash va navbat simulyatsiyasi, parallellik (oqimlar, sessiyalar), energiya modeli va DVFS, yakuniy Pareto tahlili. Real labdan farqi: GPU, TensorRT va OpenVINO yo'q — ONNX Runtime CPU'da o'lchanadi, TensorRT/OpenVINO'ning mos sozlamalari va buyruqlari izoh sifatida beriladi, energiya esa quvvat modeli bilan baholanadi (raqamlar taxminiy).

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/MT2

## Bosqichlar

### 1-qadam. Etalon model va o'lchov metodikasi

Sintetik HAR ma'lumoti (3 o'q x 128 nuqta, 6 sinf, sensor shovqini bilan) yaratiladi va HARNet (depthwise-separable 1D CNN, ~8 ming parametr) 5 epoxa o'qitiladi. Asosiy e'tibor — **to'g'ri o'lchash**: taymer aniqligi, isinish (birinchi chaqiruv sekin), bir konfiguratsiyani 5 marta takrorlab p50/p95/p99/p99.9 va variatsiya koeffitsienti (CV). PyTorch eager va TorchScript (trace + freeze) solishtiriladi; kechikish gistogrammasi uzun dumni ko'rsatadi.

Fayl: `s1_1_qadam_etalon_model_va_olchov_metodikas.py` — ishga tushirish: `python3 s1_1_qadam_etalon_model_va_olchov_metodikas.py`

Kutilgan natija: Test aniqligi ~0,92; 1-chaqiruv ~1,5 ms, keyingilari ~0,55 ms; TorchScript p50 ~0,37 ms (eager ~0,56 ms), CV < 1% (taxminiy).

Tajriba qilib ko'ring:

- Isinishni 0 ga tushirib, takror sonini 50 qiling — p99 qanday buziladi?
- torch.set_num_threads(1) bilan qayta o'lchang.
- HARNet(w=64) bilan parametrlar, aniqlik va kechikish qanday o'zgaradi?

### 2-qadam. ONNX eksport, grafik optimallashtirish va operatorlar profili

Model ONNX'ga (opset 17, dinamik batch) eksport qilinadi. ORT ning 4 ta optimallashtirish darajasi uchun optimallashtirilgan grafik faylga saqlanib, tugunlar va operatorlar ro'yxati (Conv+Relu → FusedConv) solishtiriladi — bu TensorRT'ning engine qurishdagi va OpenVINO'ning kompilyatsiyadagi qatlam birlashtirishiga o'xshaydi. So'ngra `enable_profiling` bilan har operatorning vaqt ulushi hisoblanadi (trtexec --dumpProfile analogi) va Amdahl qonuni bo'yicha optimallashtirish nuqtasi aniqlanadi.

Fayl: `s2_2_qadam_onnx_eksport_grafik_optimallasht.py` — ishga tushirish: `python3 s2_2_qadam_onnx_eksport_grafik_optimallasht.py`

Kutilgan natija: 14 tugun → 11 (FusedConv:3); ORT PyTorch eager'dan ~8 marta tez (~0,07 ms); profilda Conv/FusedConv vaqtning ~80% ini oladi (taxminiy).

Tajriba qilib ko'ring:

- `do_constant_folding=False` bilan eksport qiling — BatchNorm tugunlari paydo bo'ladimi?
- Profilni batch 1 da qiling: operatorlar ulushi qanday o'zgaradi (ustama ulushi)?
- MaxPool o'rniga stride=2 li Conv ishlatib, profil va aniqlikni solishtiring.

### 3-qadam. Hisoblash aniqligi formatlari: FP32, FP16/BF16, INT8 dinamik va statik

O'qitilgan model 6 ko'rinishda solishtiriladi: ORT FP32, FP16 vazn yaxlitlash simulyatsiyasi (TensorRT --fp16 / OpenVINO FP16 IR ning sonli ta'siri), INT8 dinamik, INT8 statik QDQ (MinMax va Percentile) va QOperator (QLinearConv) formatlari, qo'shimcha ravishda PyTorch BF16 autocast. Aniqlikdan tashqari kvantlash xatosining sezgirroq o'lchovlari — FP32 bilan bashorat mosligi, max|Δlogit| va KL divergensiya — hamda hajm, batch 1 kechikishi va batch 64 throughput o'lchanadi.

Fayl: `s3_3_qadam_hisoblash_aniqligi_formatlari_fp.py` — ishga tushirish: `python3 s3_3_qadam_hisoblash_aniqligi_formatlari_fp.py`

Kutilgan natija: Aniqlik barcha formatlarda ~0,92 (±0,5 p.p.), moslik >= 99%; INT8 statik hajmi ~16–24 KB (FP32 ~33 KB), throughput ~1,9x; dinamik INT8 bu modelda ~0,4x sekin (taxminiy).

Tajriba qilib ko'ring:

- Kalibrlash to'plamini 320 dan 32 ga tushiring — moslik va KL qanday o'zgaradi?
- `per_channel=False` bilan statik kvantlang va depthwise Conv xatosini solishtiring.
- `CalibrationMethod.Entropy` qo'shing va uch kalibrlash usulini taqqoslang.

### 4-qadam. Paket va oqim tahlili: S(b) modeli va dinamik paketlash simulyatsiyasi

Batch 1–128 uchun xizmat vaqti o'lchanadi va chiziqli model S(b) = c0 + c1·b moslanadi (c0 — har chaqiruv ustamasi, c1 — namuna narxi). So'ngra Puasson oqimi bilan navbat simulyatsiyasi: paketsiz, dinamik paketlash (b<=16, kutish 0,3 ms) va (b<=64, kutish 1 ms) siyosatlari turli yuklamada p50/p99 va throughput bo'yicha solishtiriladi (Triton dynamic_batching, OpenVINO avtomatik paketlash g'oyasi). Xuddi shu simulyatsiya GPU uslubidagi S(b) (katta c0, kichik c1 — faraziy) uchun takrorlanadi.

Fayl: `s4_4_qadam_paket_va_oqim_tahlili_s_b_modeli.py` — ishga tushirish: `python3 s4_4_qadam_paket_va_oqim_tahlili_s_b_modeli.py`

Kutilgan natija: Bu CPU'da c0 kichik (~0,02 ms), throughput batch bilan ~15k → ~18,5k inf/s; GPU uslubida dinamik paketlash SLO ichidagi yuklamani ~1000 dan ~60 000 so'rov/s gacha oshiradi (taxminiy).

Tajriba qilib ko'ring:

- max_wait ni 0,1 / 0,5 / 2 ms qilib, p99–throughput egri chizig'ini chizing.
- Puasson o'rniga «portlovchi» (bursty) oqim (masalan, har 10 ms da 50 so'rov) bilan simulyatsiya qiling.
- Ikki ishchi (2 nusxa) uchun simulyatorni kengaytiring.

### 5-qadam. Parallellik: intra-op oqimlar, sessiyalar va LATENCY/THROUGHPUT rejimlari

Kengroq model (w=64) uchun 5 konfiguratsiya batch 1 va 32 da o'lchanadi: 1 sessiya/1 oqim, intra=2, umumiy sessiyaga 2 Python oqimi, 2 sessiya x 1 oqim va ortiqcha obuna (intra=2 x 2 oqim). Natija OpenVINO'ning NUM_STREAMS va PERFORMANCE_HINT (LATENCY/THROUGHPUT), TensorRT'ning bir nechta execution context + CUDA stream tushunchalari bilan jadvalda bog'lanadi.

Fayl: `s5_5_qadam_parallellik_intra_op_oqimlar_ses.py` — ishga tushirish: `python3 s5_5_qadam_parallellik_intra_op_oqimlar_ses.py`

Kutilgan natija: Eng past kechikish — intra=2 (1 oqim); eng yuqori throughput — 2 sessiya x 1 oqim (~2x); ortiqcha obunada p99 keskin oshadi (taxminiy).

Tajriba qilib ko'ring:

- ORT_PARALLEL rejimi va inter_op_num_threads=2 ni qo'shib ko'ring — bu ketma-ket grafikda foyda beradimi?
- 3 va 4 ishchi oqim bilan sinab, throughput to'yinish nuqtasini toping.
- Batch 8 uchun ham jadvalni to'ldiring.

### 6-qadam. Energiya samaradorligi: J/inferens, EDP, DVFS va quvvat rejimlari

RAPL/nvidia-smi yo'qligi sababli quvvat modeli ishlatiladi: P = P_bosh + (P_max − P_bosh)·bandlik, bandlik = CPU vaqti / (devor vaqti·yadro). FP32 va INT8 statik, 1/2 oqim, batch 1/32 uchun µJ/inferens (to'liq va dinamik), inf/J va EDP (energiya x kechikish) hisoblanadi. So'ngra DVFS simulyatsiyasi: P_din ~ f·V², kechikish ~ 1/f — SLO ichida eng tejamkor chastota va «race-to-idle» sharti; oxirida Jetson uslubidagi nvpmodel rejimlari (faraziy raqamlar) bo'yicha tanlov.

Fayl: `s6_6_qadam_energiya_samaradorligi_j_inferen.py` — ishga tushirish: `python3 s6_6_qadam_energiya_samaradorligi_j_inferen.py`

Kutilgan natija: Eng kam energiya — INT8 statik, 1 oqim, batch 32 (~500 µJ/inf, taxminiy); 2 oqim batch 1 da energiya oshadi; DVFS optimumi ~1,2 GHz, statik quvvat katta bo'lsa optimum yuqori chastotaga siljiydi.

Tajriba qilib ko'ring:

- P_BOSH va P_MAX ni o'z noutbukingiz TDP'siga moslang — eng tejamkor konfiguratsiya o'zgaradimi?
- psutil.Process().cpu_times() bilan user/system vaqtni ajrating.
- DVFS da yuk 500 va 8000 so'rov/s bo'lganda optimal chastotani toping.

### 7-qadam. Yakuniy Pareto tahlili, ssenariy bo'yicha tanlov va TensorRT/OpenVINO yo'l xaritasi

6 konfiguratsiya (PyTorch eager, ORT FP32 1/2 oqim, INT8 dinamik, INT8 statik 1/2 oqim) bitta jadvalga yig'iladi: aniqlik, p99 (batch 1), throughput (batch 32), µJ/inferens va hajm. To'rt mezonli Pareto chegarasi aniqlanadi va uch ssenariy (bilakuzuk, kasalxona serveri, real vaqt signali) uchun cheklovlar va maqsad funksiyasi bo'yicha konfiguratsiya tanlanadi. Oxirida GPU/NPU li stendda bajariladigan trtexec, ovc/benchmark_app va quvvat o'lchash buyruqlari beriladi.

Fayl: `s7_7_qadam_yakuniy_pareto_tahlili_ssenariy_.py` — ishga tushirish: `python3 s7_7_qadam_yakuniy_pareto_tahlili_ssenariy_.py`

Kutilgan natija: Pareto chegarasida INT8 statik konfiguratsiyalar; bilakuzuk uchun INT8 statik 1 oqim, server uchun eng yuqori throughput'li variant tanlanadi; grafikda qizil nuqtalar — Pareto (raqamlar taxminiy).

Tajriba qilib ko'ring:

- Ssenariyga «aniqlik >= FP32» shartini qo'shing — tanlov o'zgaradimi?
- Pareto'ni faqat (p99, energiya) juftligida hisoblang va natijani solishtiring.
- HARNet(w=16) ni qo'shib, kichikroq model Pareto chegarasiga chiqadimi tekshiring.

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
|---|---|---|---|
| 1 | **O'lchov barqarorligi** — 1-qadamda isinish soni (0, 10, 50, 200) va takror soni (50, 400, 2000) kombinatsiyalari uchun p50, p99 va CV ni 5 martadan o'lchang. Jadval va 3–5 jumlali xulosada ishonchli o'lchov uchun minimal sozlamalarni asoslang. | oson | 5 |
| 2 | **Operator profilining batch'ga bog'liqligi** — 2-qadam profilini batch 1, 8, 64 va 256 da bajaring. Har operator ulushini ustunli grafikda ko'rsating va batch oshganda qaysi operator(lar) ulushi o'sishini, qat'iy ustama qanday kamayishini izohlang. | oson | 5 |
| 3 | **Format tanlovi jadvali** — 3-qadam natijalari asosida 6 format uchun «hajm / aniqlik / moslik / tezlik» jadvalini tuzing va har formatni qaysi apparatda (Tensor Core'li GPU — TensorRT, VNNI li Intel CPU — OpenVINO, ARM chekka qurilma) ishlatish mantiqli ekanini 1–2 jumladan asoslang. | oson | 5 |
| 4 | **Kalibrlash usuli va to'plami** — 3-qadamda statik INT8 ni MinMax, Percentile (99,9 / 99,99 / 99,999) va Entropy usullari bilan, 32, 128, 320 va 1000 namunali kalibrlash to'plamlarida bajaring. Aniqlik, FP32 bilan moslik va KL ni jadvalga yozing va eng barqaror kombinatsiyani tanlang. | o'rta | 10 |
| 5 | **Dinamik paketlashni sozlash** — 4-qadam simulyatorida max_b (4, 16, 64) va max_wait (0,1; 0,3; 1; 3 ms) panjarasini bu CPU va GPU uslubidagi S(b) uchun bajaring. p99 <= 2 ms SLO ostida maksimal yuklamani beradigan sozlamani toping va natijani issiqlik xaritasi (heatmap) bilan ko'rsating. | o'rta | 10 |
| 6 | **Parallellik va energiya birgalikda** — 5-qadam konfiguratsiyalari uchun 6-qadam quvvat modeli bilan µJ/inferens va EDP ni hisoblang. «Eng tez» va «eng tejamkor» konfiguratsiya bir xilmi? Natijani jadval va 1 ta grafik bilan bering, OpenVINO LATENCY/THROUGHPUT ko'rsatmalari bilan bog'lab izohlang. | o'rta | 10 |
| 7 | **Model kengligi bo'yicha masshtablash** — HARNet(w) ni w = 8, 16, 32, 64, 128 uchun o'qitib (3-qadam usulida), har biri uchun aniqlik, parametrlar, FP32 va INT8 statik p99 va throughput ni o'lchang. Aniqlik–kechikish egri chizig'ini chizing va «yetarli darajada kichik» modelni tanlang. | o'rta | 10 |
| 8 | **Real quvvat o'lchashga tayyorgarlik** — 6-qadamdagi model asosidagi energiya bahosini real o'lchov bilan solishtirish metodikasini ishlab chiqing: o'z kompyuteringizda (agar mavjud bo'lsa) RAPL (perf stat -e power/energy-pkg/) yoki tashqi quvvat o'lchagich bilan FP32 va INT8 uchun J/inferens ni o'lchang, yoki o'lchash imkoni bo'lmasa, quvvat modelining sezgirlik tahlilini (P_bosh, P_max ±50%) bajaring. Xatolik manbalari va noaniqlik intervali bilan hisobot bering. | qiyin | 15 |
| 9 | **Navbat nazariyasi bilan tekshirish** — 4-qadamdagi paketsiz siyosat uchun M/D/1 navbat formulasidan (Pollaczek–Khinchine) o'rtacha kutish vaqtini hisoblang va simulyatsiya natijasi bilan 5 ta yuklama nuqtasida solishtiring. Dinamik paketlash uchun nazariy throughput chegarasini 1/c1 orqali chiqaring va simulyatsiyadagi to'yinish bilan solishtiring; tafovutlarni izohlang. | qiyin | 15 |
| 10 | **To'liq optimallashtirish hisoboti** — O'zingiz tanlagan boshqa model va ma'lumot (masalan, sklearn digits uchun MLP yoki sintetik EKG uchun 1D CNN) uchun 1–7 qadamlarning barchasini takrorlang: o'lchov metodikasi, eksport va profil, kamida 4 format, paket/oqim tahlili, parallellik, energiya va DVFS, Pareto va ssenariy tanlovi. 4–6 betlik hisobotda «TensorRT yoki OpenVINO bo'lsa nima o'zgaradi» bo'limini buyruqlar bilan yozing. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Tibbiyot: EKG aritmiya detektori — ONNX eksport va o'lchov metodikasi (oson)

Soha: Tibbiyot (kardiologiya)

Portativ EKG qurilmasi 1 kanalli signalni 360 Hz da yozadi va 1 soniyalik oyna bo'yicha aritmiyani aniqlashi kerak. Qurilma ishlab chiqaruvchisi ONNX formatini qabul qiladi va p99 kechikish 10 ms dan oshmasligini talab qiladi. Sizdan modelni o'qitish, to'g'ri o'lchov metodikasi bilan kechikishni o'lchash va ORT darajalarini solishtirish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik EKG (sinus + QRS shakli), 4 sinf |
| Kirish | 1 x 360 |
| Model | 1D CNN 3 qatlam (16-32-32) |
| Batch | 1, 8 |
| SLO | p99 <= 10 ms |
| Takror | 5 x 500 |

Qadamlar:

1. Lab s1 kabi sintetik EKG generatori va 1D CNN ni o'qiting, aniqlikni bering.
2. Isinish va takror bilan PyTorch eager va TorchScript kechikishini (p50/p99/CV) o'lchang.
3. Lab s2 kabi ONNX eksport va 4 ORT darajasi: tugunlar, operatorlar, p50/p99.
4. ORT profil bilan eng og'ir operatorni toping.
5. Xulosa: SLO bajariladimi, qaysi daraja va rejim tavsiya etiladi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 2-variant. Agro: tuproq namligi sensorlari bo'yicha sug'orish tasnifi — INT8 dinamik (oson)

Soha: Agro

Dala sensori (8 ta o'lchov: namlik, harorat, EC, ...) bo'yicha sug'orish kerak/kerak emas qarori Raspberry Pi'da qabul qilinadi. Model kichik bo'lishi va kam energiya sarflashi kerak. Sizdan MLP ni INT8 dinamik kvantlash, hajm, aniqlik va tezlikni solishtirish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | make_classification (8 belgi, 3 sinf, 5000) |
| Model | MLP 8-128-64-3 |
| Kvantlash | PyTorch va ORT dinamik INT8 |
| Batch | 1, 64 |
| Maqsad hajm | < 20 KB |
| Dvigatel | qnnpack |

Qadamlar:

1. MLP ni o'qitib, FP32 aniqlik, hajm va p50 ni o'lchang.
2. PyTorch quantize_dynamic (qnnpack) bilan INT8: hajm, aniqlik, tezlik.
3. ONNX eksport va ORT quantize_dynamic: xuddi shu o'lchovlar.
4. Lab s3 kabi FP32 bilan moslik va KL ni hisoblang.
5. Xulosa: Raspberry Pi uchun qaysi variant (ARM'da INT8 qanday bo'lishi mumkin — taxminiy izoh).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 3-variant. Transport: yo'l belgilari tasnifi CNN — statik INT8 va kalibrlash (o'rta)

Soha: Transport

Avtomobil kamerasi uchun 8x8 dan kattaroq (16x16) sintetik yo'l belgilari tasvirlarini tasniflovchi CNN optimallashtirilmoqda. Aniqlik 1 p.p. dan ko'p tushmasligi kerak. Sizdan statik INT8 ni uch kalibrlash usuli va ikki format (QDQ, QOperator) bilan sinash so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik 16x16 shakllar (doira, uchburchak, kvadrat, ...), 6 sinf |
| Model | CNN 2 Conv + 1 FC |
| Kalibrlash | MinMax, Percentile 99,99, Entropy |
| Format | QDQ, QOperator |
| Kalibrlash to'plami | 64 va 512 |
| Chegara | aniqlik pasayishi <= 1 p.p. |

Qadamlar:

1. Sintetik tasvir generatori va CNN ni o'qiting, ONNX'ga eksport qiling.
2. Lab s3 kabi 3 kalibrlash usuli x 2 format x 2 to'plam hajmi bilan statik INT8 yarating.
3. Har biri uchun aniqlik, moslik, KL, hajm va throughput (batch 64) ni jadvalga yozing.
4. Operatorlar ro'yxatini (QLinearConv, Q/DQ) taqqoslang.
5. Talabni qanoatlantiradigan eng tez konfiguratsiyani tanlang.
6. Xulosa: TensorRT INT8 kalibrlash keshi bilan bog'lab izohlang.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 4-variant. Telekom: trafik anomaliyasi autoenkoderi — paket va throughput (o'rta)

Soha: Telekom

Operator tarmoq oqimlarini (20 belgi) real vaqtda tekshiradi; soniyasiga 50 000 oqim keladi. Autoenkoder rekonstruksiya xatosi bo'yicha anomaliyani aniqlaydi. Sizdan batch bo'yicha S(b) modelini qurish, dinamik paketlash simulyatsiyasi bilan kerakli nusxalar sonini topish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik oqim belgilari (20), 2% anomaliya |
| Model | Autoenkoder 20-64-8-64-20 |
| Batch | 1–1024 |
| Yuklama | 50 000 oqim/s |
| SLO | p99 <= 5 ms |
| Siyosat | paketsiz, b<=128 kutish 1 ms |

Qadamlar:

1. Autoenkoderni o'qiting va anomaliya aniqlash ROC-AUC sini bering.
2. Lab s4 kabi batch 1–1024 da S(b) ni o'lchab, c0 va c1 ni toping.
3. Dinamik paketlash simulyatsiyasida yuklama 10k–60k uchun p99 va throughput.
4. SLO ni saqlash uchun kerakli nusxalar sonini hisoblang.
5. Lab s5 kabi 2 sessiya x 1 oqim konfiguratsiyasi bilan tekshiring.
6. Xulosa: paketlash va nusxalar soni bo'yicha tavsiya.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 5-variant. Energetika: quyosh paneli quvvati prognozi — regressiyada kvantlash xatosi (oson)

Soha: Energetika

Quyosh elektr stansiyasi har 15 daqiqada keyingi soat quvvatini prognoz qiladi; model kichik kontrollerda ishlaydi. Regressiyada kvantlash xatosi to'g'ridan-to'g'ri MAE ga ta'sir qiladi. Sizdan FP32 va INT8 variantlarda MAE va kechikishni solishtirish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik quyosh nurlanishi + bulut (24 belgi) |
| Model | MLP 24-64-32-1 |
| Kvantlash | ORT dinamik va statik |
| Metrika | MAE (kW), max xato |
| Batch | 1 |
| Kalibrlash | 200 namuna |

Qadamlar:

1. Sintetik ma'lumot va MLP regressorni o'qiting, MAE ni bering.
2. ONNX eksport va ORT FP32 kechikishi.
3. Dinamik va statik INT8: MAE, max xato, hajm, kechikish.
4. Kvantlash xatosining taqsimotini gistogrammada ko'rsating.
5. Xulosa: regressiyada INT8 ishlatish mumkinmi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 6-variant. Moliya: kredit skoring — daraxtli modeldan MLP distillyatsiyasi va energiya (qiyin)

Soha: Moliya

Bank skoring modeli (gradient boosting) ONNX'ga to'g'ridan-to'g'ri eksport qilinmaydi (skl2onnx yo'q). Yechim — bilimni MLP ga distillyatsiya qilish va uni optimallashtirish. Sizdan distillyatsiya sifati, ORT optimallashtirish, INT8, paket tahlili va energiya bahosi bilan to'liq hisobot so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | make_classification (30 belgi, 20 000, 10% defolt) |
| O'qituvchi | HistGradientBoosting |
| Talaba | MLP 30-128-64-2 |
| Metrika | ROC-AUC, o'qituvchi bilan moslik |
| Batch | 1, 256 |
| Energiya | P_bosh 10 W, P_max 45 W (faraz) |

Qadamlar:

1. Boosting modelini o'qiting va ROC-AUC ni bering.
2. O'qituvchi ehtimollari (soft label) bo'yicha MLP ni distillyatsiya qiling; AUC va moslikni solishtiring.
3. MLP ni ONNX'ga eksport qilib, ORT darajalari va profilni bajaring.
4. INT8 dinamik/statik: AUC, hajm, throughput.
5. Lab s4 kabi S(b) va dinamik paketlash (yuklama 5000 so'rov/s).
6. Lab s6 kabi µJ/inferens va EDP ni barcha variantlar uchun hisoblang.
7. Xulosa: Pareto va server uchun tanlov.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)

## 7-variant. Ta'lim: kichik transformer bloki — MatMul kvantlash va parallellik (qiyin)

Soha: Ta'lim

Ta'lim platformasi talabaning 32 ta oxirgi javobi ketma-ketligi bo'yicha keyingi javob to'g'riligini bashorat qiladi (knowledge tracing). Model — bitta transformer bloki. Sizdan eksport, fusion, MatMul kvantlash, parallellik konfiguratsiyalari va energiyani tahlil qilish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik javoblar ketma-ketligi (32 qadam, 50 ko'nikma) |
| Model | Embedding 64 + TransformerEncoderLayer (4 bosh) + Linear |
| Kvantlash | dinamik INT8 (MatMul) |
| Parallellik | intra 1/2, 2 sessiya |
| Batch | 1, 32 |
| SLO | p99 <= 3 ms |

Qadamlar:

1. Sintetik ketma-ketliklar va modelni o'qiting, AUC ni bering.
2. ONNX (opset 17) eksport; ORT ALL darajasidagi operatorlarni va fusion'ni (Attention/LayerNorm) tahlil qiling.
3. Dinamik INT8: AUC, hajm, tezlik (MatMul og'ir modelda farq CNN dan kattami).
4. Lab s5 kabi 5 parallellik konfiguratsiyasi batch 1 va 32 da.
5. Lab s6 kabi energiya bahosi.
6. Xulosa: TensorRT da transformer uchun FP16 va plaginlar haqida izoh.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)

## 8-variant. Sanoat: podshipnik vibratsiyasi 1D CNN — profil va sezgir qatlam (o'rta)

Soha: Sanoat

Zavod nasoslaridagi vibratsiya sensori (2 kHz, 512 nuqta) bo'yicha nosozlik turini aniqlash modeli sanoat kompyuterida ishlaydi. Sizdan operator profili orqali og'ir qatlamni topish va statik INT8 da sezgir qatlamni FP32 da qoldirish tahlilini bajarish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik vibratsiya (asosiy chastota + nosozlik garmonikalari), 4 sinf |
| Kirish | 1 x 512 |
| Model | 1D CNN 4 qatlam |
| Kvantlash | statik QDQ, nodes_to_exclude |
| Profil | batch 1 va 64 |
| Chegara | aniqlik pasayishi <= 0,5 p.p. |

Qadamlar:

1. Generator va 1D CNN ni o'qiting.
2. Lab s2 kabi profil bilan operatorlar vaqt ulushini batch 1 va 64 da toping.
3. To'liq statik INT8 aniqligi va moslikni o'lchang.
4. Har Conv ni navbatma-navbat FP32 da qoldirib, aniqlikni o'lchang.
5. «INT8 + 1 FP32 qatlam» va to'liq INT8 ni tezlik/aniqlik bo'yicha solishtiring.
6. Xulosa: sezgir qatlam va uning sababi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 9-variant. Kiberxavfsizlik: login hodisalari MLP — keng model va hajm (oson)

Soha: Xavfsizlik

Kirish tizimi har login urinishini (16 belgi) tekshiradi. Model xotirada kichik bo'lishi kerak, chunki u har serverda ishlaydi. Sizdan kenglik bo'yicha (128, 512) ikki MLP ni INT8 dinamik kvantlash va hajm nisbatini tahlil qilish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | make_classification (16 belgi, 3% hujum) |
| Modellar | MLP 16-128-2 va 16-512-512-2 |
| Kvantlash | ORT dinamik INT8 |
| Metrika | recall, hajm nisbati |
| Batch | 1, 128 |

Qadamlar:

1. Ikki MLP ni o'qiting va recall ni bering.
2. ONNX eksport va FP32 hajm/kechikish.
3. Dinamik INT8: hajm nisbati (4x ga qanchalik yaqin), recall, tezlik.
4. Natijalarni jadvalda bering.
5. Xulosa: keng modelda kvantlash foydasi nega kattaroq.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 10-variant. Chakana savdo: embedding + MLP tavsiya modeli — katta paket (o'rta)

Soha: Chakana savdo

Onlayn do'kon tavsiya nomzodlarini qayta saralovchi modelni bir so'rovda 200 ta nomzod uchun ishlatadi. Shuning uchun katta paket (200–1000) asosiy rejim. Sizdan embedding (Gather) va MLP li modelni eksport qilish, katta paketda throughput va kvantlash ta'sirini o'lchash so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik foydalanuvchi-mahsulot juftlari (5000 mahsulot) |
| Model | Embedding 5000x32 + MLP 64-64-1 |
| Batch | 200, 500, 1000 |
| Kvantlash | dinamik INT8 (Gather va MatMul) |
| SLO | 200 nomzod <= 5 ms |
| Metrika | AUC, nDCG@10 |

Qadamlar:

1. Modelni o'qiting, AUC va nDCG@10 ni bering.
2. ONNX eksport; embedding jadvali hajmi umumiy hajmning qancha qismi.
3. Batch 200/500/1000 da S(b) va throughput.
4. Dinamik INT8: hajm, AUC, nDCG, tezlik.
5. Lab s5 kabi intra=1/2 bilan SLO bajarilishi.
6. Xulosa: katta paket rejimida qaysi konfiguratsiya.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 11-variant. Logistika: qo'l skanerida yetkazish vaqti prognozi — energiya va batareya (o'rta)

Soha: Logistika

Kuryerning qo'l skaneri (ARM, batareya 3000 mAh) har buyurtma uchun yetkazish vaqtini prognoz qiladi. Batareya bir smenaga (10 soat) yetishi kerak. Sizdan energiya modelini skaner parametrlariga moslash, FP32/INT8 da J/inferens va batareya umrini hisoblash so'raladi.

| Parametr | Qiymat |
|---|---|
| Model | MLP 12-64-32-1 |
| Inferens chastotasi | har 2 soniyada 1 (fon) + 200 buyurtma |
| Quvvat modeli | P_bosh 0,4 W, P_max 2,5 W (faraz) |
| Batareya | 3000 mAh x 3,7 V |
| Format | FP32, INT8 statik |
| DVFS | 0,6–1,8 GHz |

Qadamlar:

1. Modelni o'qiting, ONNX va INT8 statik variantlarni yarating.
2. Lab s6 kabi bandlik va µJ/inferens ni ikkala format uchun o'lchang.
3. Skaner quvvat modeliga moslab smena davomida model iste'mol qiladigan energiyani hisoblang.
4. DVFS simulyatsiyasida eng tejamkor chastotani toping (SLO 20 ms).
5. Batareya ulushini (model / butun qurilma) baholang.
6. Xulosa: model optimallashtirishning batareya umriga real ta'siri (taxminiy).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 12-variant. Shahar: chorraha tiqilinchi bashorati — oqimlar va Little qonuni (oson)

Soha: Shahar

Shahar trafik markazi 300 chorraha uchun har 5 soniyada tiqilinch darajasini bashorat qiladi (soniyasiga 60 so'rov). Sizdan kichik modelni ORT da o'lchash va Little qonuni bilan nechta nusxa kerakligini hisoblash so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik trafik matritsasi (12 belgi) |
| Model | MLP 12-32-3 |
| Yuklama | 60 so'rov/s (cho'qqida 600) |
| Oqimlar | intra 1, 2 |
| SLO | p99 <= 20 ms |

Qadamlar:

1. Modelni o'qiting va ONNX'ga eksport qiling.
2. ORT intra=1 va 2 da p50/p99 ni o'lchang.
3. Little qonuni (N = X·R) bilan oddiy va cho'qqi yuklama uchun nusxalar sonini hisoblang.
4. Natijani jadvalda bering.
5. Xulosa: bu yuklama uchun optimallashtirish kerakmi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 13-variant. Ekologiya: havo sifati tasnifi — INT8 va siyraklashtirish (pruning) (o'rta)

Soha: Ekologiya

Shahar havo sifati stansiyasi (10 sensor) har daqiqada havo sifati sinfini aniqlaydi. Model mikrokompyuterda ishlaydi. Sizdan INT8 kvantlash va magnituda bo'yicha siyraklashtirishni (50%, 80%) hajm, aniqlik va tezlik bo'yicha taqqoslash so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik sensor o'lchovlari, 5 sinf |
| Model | MLP 10-256-128-5 |
| Pruning | torch.nn.utils.prune 50% va 80% |
| Kvantlash | ORT statik INT8 |
| Metrika | aniqlik, nol vaznlar ulushi, hajm (zip bilan) |
| Batch | 1, 64 |

Qadamlar:

1. Modelni o'qiting.
2. 50% va 80% magnituda pruning; aniqlik va fine-tune (1 epoxa) dan keyingi aniqlik.
3. ONNX eksport; siqilgan (zip) hajm va kechikish — zich yadroda pruning tezlik beradimi?
4. Statik INT8 va «pruning + INT8» kombinatsiyasi.
5. Barcha variantlar jadvali va Pareto grafigi.
6. Xulosa: TensorRT 2:4 siyraklik va OpenVINO siyrak yadrolari haqida izoh.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 14-variant. Sport: akselerometr + giroskop harakat tanish — CNN+GRU, energiya va Pareto (qiyin)

Soha: Sport

Sport soati 6 kanalli (akselerometr + giroskop) signal bo'yicha mashq turini aniqlaydi. Model qurilmada uzluksiz ishlaydi, shuning uchun energiya asosiy mezon. Sizdan CNN+GRU modelini eksport qilish, statik kvantlash (per-tensor vs per-channel), parallellik, DVFS va Pareto tahlili so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik 6 x 100 signal, 8 mashq turi |
| Model | Conv1d 32 + GRU 32 + Linear |
| Kvantlash | statik QDQ per-tensor va per-channel, dinamik |
| Batch | 1 |
| Energiya | P_bosh 0,05 W, P_max 0,6 W (faraz) |
| SLO | p99 <= 15 ms |

Qadamlar:

1. Lab s1 kabi generator va CNN+GRU ni o'qiting; o'lchov metodikasi bilan eager kechikishini o'lchang.
2. ONNX eksport (GRU operatori) va profil — GRU ulushi.
3. Per-tensor, per-channel statik va dinamik INT8: aniqlik, moslik, KL, tezlik.
4. Lab s5 kabi parallellik (intra 1/2) va lab s6 kabi µJ/inferens.
5. DVFS simulyatsiyasida soat protsessori uchun eng tejamkor chastota.
6. Lab s7 kabi Pareto va soat uchun tanlov.
7. Xulosa: qurilmada (NPU bo'lsa) nima o'zgarishi mumkin.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)

## 15-variant. Media: sharh kayfiyati so'zlar sumkasi MLP — dinamik kvantlash (oson)

Soha: Media

Kino portali foydalanuvchi sharhlarining kayfiyatini (ijobiy/salbiy/neytral) aniqlaydi. Matn 2000 so'zli lug'at bo'yicha so'zlar sumkasi vektoriga aylantiriladi. Sizdan MLP ni dinamik kvantlash va hajm/tezlikni solishtirish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik sharhlar (lug'at 2000 so'z) |
| Model | MLP 2000-128-3 |
| Kvantlash | ORT dinamik INT8 |
| Batch | 1, 32 |
| Metrika | aniqlik, hajm, p50 |

Qadamlar:

1. Sintetik sharh generatori va MLP ni o'qiting.
2. ONNX eksport, FP32 hajm va kechikish.
3. Dinamik INT8: hajm (~4x), aniqlik, tezlik.
4. Natijalar jadvali.
5. Xulosa: birinchi qatlam (2000x128) hajmning qancha qismi va nega.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 16-variant. Tibbiyot: implant yurak monitori — uch mezonli Pareto va hajm chegarasi (o'rta)

Soha: Tibbiyot (implant)

Teri ostiga o'rnatiladigan yurak monitori 64 KB xotiraga ega va batareyasi 3 yil ishlashi kerak. Sizdan EKG oynasi bo'yicha 1D CNN ning bir nechta kenglikdagi variantlarini FP32/INT8 da o'lchab, hajm cheklovi ostida aniqlik–energiya Pareto chegarasini topish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik EKG oynalari (1 x 256), 3 sinf |
| Modellar | 1D CNN w = 4, 8, 16, 32 |
| Format | FP32, INT8 statik |
| Hajm chegarasi | <= 64 KB |
| Energiya | quvvat modeli (faraz) |
| Metrika | sezgirlik (recall) >= 0,95 |

Qadamlar:

1. Generator va 4 kenglikdagi modellarni o'qiting.
2. Har biri uchun FP32 va INT8 statik ONNX yarating.
3. Hajm, recall, kechikish va µJ/inferens ni o'lchang.
4. Hajm <= 64 KB va recall >= 0,95 shartlari ostida Pareto chegarasini toping.
5. Tanlangan model uchun 3 yillik energiya byudjetini taxminiy hisoblang.
6. Xulosa.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 17-variant. Aviatsiya: dvigatel qoldiq resursi (RUL) prognozi — ONNX va energiya (o'rta)

Soha: Aviatsiya

Samolyot bort kompyuteri dvigatelning 14 sensori bo'yicha qoldiq resursni (sikllar) prognoz qiladi. Sertifikatlash talablari tufayli FP32 dan farq kichik bo'lishi kerak. Sizdan regressiya modelini optimallashtirish va energiya/kechikishni baholash so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik degradatsiya egri chiziqlari (14 sensor, 30 qadam oyna) |
| Model | 1D CNN + Linear (regressiya) |
| Format | FP32, FP16 vazn (simul.), INT8 statik |
| Metrika | RMSE (sikl), max |farq| FP32 dan |
| Chegara | max |farq| <= 2 sikl |
| Batch | 1 |

Qadamlar:

1. Generator va regressorni o'qiting, RMSE ni bering.
2. ONNX eksport, ORT darajalari va profil.
3. FP16 vazn simulyatsiyasi va INT8 statik: RMSE va FP32 dan max farq.
4. Lab s6 kabi energiya bahosi va EDP.
5. Chegarani qanoatlantiradigan eng tejamkor format.
6. Xulosa: sertifikatlashda kvantlash xavfi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 18-variant. Fintech: to'lov firibgarligi — kvantlashning recall va chegaraga ta'siri (qiyin)

Soha: Fintech

To'lov tizimi har tranzaksiyani 2 ms ichida tekshirishi kerak; soniyasiga 20 000 tranzaksiya. Kvantlash ehtimollarni siljitib, chegaradagi recall ni o'zgartirishi mumkin. Sizdan kvantlashning qaror chegarasiga ta'sirini, dinamik paketlash va nusxalar rejasini to'liq tahlil qilish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | make_classification (40 belgi, 0,3% firibgarlik) |
| Model | MLP 40-256-128-1 |
| Kvantlash | statik MinMax, Percentile, Entropy |
| Metrika | recall@FPR 0,5%, PR-AUC |
| Yuklama | 20 000 so'rov/s |
| SLO | p99 <= 2 ms |

Qadamlar:

1. Modelni o'qiting, FP32 PR-AUC va recall@FPR 0,5% ni bering.
2. Uch kalibrlash usuli bilan statik INT8: PR-AUC, recall, chegara siljishi.
3. Kvantlangan model uchun chegarani qayta kalibrlab, recall tiklanishini ko'rsating.
4. Lab s4 kabi S(b) va dinamik paketlash simulyatsiyasi 20 000 so'rov/s da.
5. Lab s5 kabi parallellik va kerakli nusxalar soni.
6. Lab s6 kabi energiya va yakuniy Pareto.
7. Xulosa: kvantlashdan keyin chegarani qayta tekshirish qoidasi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)

## 19-variant. Oziq-ovqat sanoati: konveyerda nuqsonli mahsulot — ORT va 500 dona/s (oson)

Soha: Oziq-ovqat sanoati

Konveyer kamerasi har mahsulotni (sintetik 24x24 kulrang tasvir) yaroqli/nuqsonli deb ajratadi; tezlik soniyasiga 500 dona. Sizdan kichik CNN ni ORT da o'lchash va talab bajarilishini tekshirish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik 24x24 tasvirlar (dog', yoriq), 2 sinf |
| Model | CNN 2 Conv (8-16) + FC |
| Batch | 1, 16 |
| Talab | 500 dona/s, p99 <= 10 ms |
| Oqimlar | intra 1, 2 |

Qadamlar:

1. Generator va CNN ni o'qiting, aniqlik va recall (nuqsonli) ni bering.
2. ONNX eksport va ORT ALL darajasida p50/p99.
3. Batch 1 va 16 da throughput.
4. intra=1 va 2 da talab bajarilishini tekshiring.
5. Xulosa: bitta CPU yetadimi, zaxira qancha.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 20-variant. Konchilik: NIR spektr bo'yicha ruda tarkibi — to'liq optimallashtirish zanjiri (qiyin)

Soha: Konchilik

Boyitish fabrikasida konveyerdagi rudaning NIR spektri (256 to'lqin uzunligi) bo'yicha metall ulushi bashorat qilinadi. Natijalar texnologik jarayonni boshqaradi, shuning uchun aniqlik, kechikish va ishonchlilik muhim. Sizdan to'liq optimallashtirish zanjiri va natijalarning ishonch intervallarini berish so'raladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik spektrlar (3 mineral aralashmasi + shovqin), 256 nuqta |
| Model | 1D CNN regressor |
| Format | FP32, INT8 dinamik, INT8 statik |
| Metrika | RMSE (%), R2 |
| O'lchov | 5 takror, bootstrap 95% CI |
| SLO | p99 <= 5 ms, 200 spektr/s |

Qadamlar:

1. Lab s1 kabi generator, model va o'lchov metodikasi (5 takror, CV).
2. ONNX eksport, darajalar va operatorlar profili.
3. Uch format: RMSE, R2, moslik, hajm.
4. Lab s4–s5 kabi paket, dinamik paketlash va parallellik tahlili.
5. Lab s6 kabi energiya va DVFS.
6. p50/p99 va RMSE farqlari uchun bootstrap 95% ishonch intervallari; farqlar statistik ahamiyatlimi.
7. Lab s7 kabi Pareto va fabrika uchun yakuniy tavsiya.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)
