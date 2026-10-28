# A3. Sun'iy intellekt tizimlarini optimallashtirish va unumdorligini baholash

Bu virtual laboratoriyada siz o'qitilgan neyron tarmoqni ishlab chiqarishga tayyorlash zanjirini to'liq bajarasiz: PyTorch modelini ONNX'ga eksport qilish, ONNX Runtime'da grafik optimallashtirish darajalari va paket o'lchamini o'lchash, PyTorch va ONNX Runtime'da INT8 kvantlash (dinamik va statik, turli kalibrlash usullari), so'ngra aniqlik-tezlik-hajm jadvali va Pareto tahlili. Real labdan farqi: ResNet-50 va ImageNet o'rniga sklearn digits (8x8) va kichik CNN/MLP, GPU/TensorRT/OpenVINO yo'q — ular o'rniga CPU muqobillari va «GPU'da bu qadam qanday bo'ladi» izohlari; sandbox CPU'da AVX/VNNI yo'qligi sababli INT8 tezlanishi kutilmaydi (bu o'zi muhim saboq).

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/A3

## Bosqichlar

### 1-qadam. Etalon: PyTorch FP32 o'qitish va ONNX eksport

Kichik CNN sklearn digits to'plamida 3 epoxa o'qitiladi, batch 1 kechikishi (p50/p95) o'lchanadi, so'ngra model `torch.onnx.export` bilan dinamik paket o'qi va opset 17 bilan eksport qilinadi. `onnx.checker` grafikni tekshiradi, ONNX Runtime chiqishi PyTorch bilan `np.allclose` orqali solishtiriladi. O'zgartirib ko'ring: epoxa sonini, `dynamic_axes` ni olib tashlab kirish shaklini, `do_constant_folding=False` da tugunlar sonini.

Fayl: `s1_1_qadam_etalon_pytorch_fp32_oqitish_va_o.py` — ishga tushirish: `python3 s1_1_qadam_etalon_pytorch_fp32_oqitish_va_o.py`

Kutilgan natija: 3 epoxadan keyin ~94 % aniqlik, PyTorch eager p50 ~0,15 ms; ONNX grafikda 10 tugun (BatchNorm Conv'ga birlashgan), maksimal farq ~1e-6, «MOS».

Tajriba qilib ko'ring:

- `model.eval()` ni eksportdan oldin olib tashlang (train rejimida) — farq va tugunlar qanday o'zgaradi?
- Kanallar sonini 8/16 dan 32/64 ga oshiring: parametrlar, hajm va latency qanday o'sadi?
- `opset_version=13` bilan eksport qiling va operatorlar ro'yxatini solishtiring.

### 2-qadam. ONNX Runtime: optimallashtirish darajalari, paket, oqimlar, Little qonuni

ONNX Runtime'ning 4 ta grafik optimallashtirish darajasi (DISABLE_ALL, BASIC, EXTENDED, ALL) batch 1 da o'lchanadi va optimallashtirilgan grafikdagi tugunlar soni sanaladi; keyin paket 1/16/64/256 uchun throughput va Little qonuni (N = X·R), intra_op oqimlar 1 va 2 taqqoslanadi (GPU parallelligi o'rniga). Oxirida 2000 so'rov/s talabi uchun nusxalar soni rejalashtiriladi.

Fayl: `s2_2_qadam_onnx_runtime_optimallashtirish_d.py` — ishga tushirish: `python3 s2_2_qadam_onnx_runtime_optimallashtirish_d.py`

Kutilgan natija: ORT ALL PyTorch eager'dan 6–9 marta tez; EXTENDED/ALL da tugunlar 10 → 7–8; batch 256 da throughput 10+ marta yuqori, latency ham oshadi; 2 oqim ~1,9x.

Tajriba qilib ko'ring:

- `takror` ni 300 dan 2000 ga oshiring — p99 barqarorlashadimi?
- `inter_op_num_threads` va `execution_mode = ORT_PARALLEL` qo'shib ko'ring.
- Little qonuni rejasiga p50 o'rniga p99 ni emas, o'rtachani qo'ysangiz nusxalar soni qanday o'zgaradi va nega p99 to'g'riroq?

### 3-qadam. PyTorch'da INT8 dinamik kvantlash (Linear)

MLP (Linear og'ir model) `torch.ao.quantization.quantize_dynamic` bilan INT8 ga o'tkaziladi: kvantlash formulasi q = round(w/s) qo'lda tekshiriladi, hajm (state_dict baytlari), aniqlik va batch 1/256 kechikishi FP32 bilan solishtiriladi. Sandbox CPU'da AVX yo'qligi sababli `qnnpack` dvigateli majburan tanlanadi — real serverda bu qator kerak emas.

Fayl: `s3_3_qadam_pytorchda_int8_dinamik_kvantlash.py` — ishga tushirish: `python3 s3_3_qadam_pytorchda_int8_dinamik_kvantlash.py`

Kutilgan natija: Hajm ~3,8 marta kichik (335 → 89 KB), aniqlik o'zgarmaydi (±0,3 p.p.), tezlik bu CPU'da 0,7–1,0x (VNNI yo'q).

Tajriba qilib ko'ring:

- Qatlam kengligini 256 dan 1024 ga oshiring — hajm nisbati 4x ga yaqinlashadimi, tezlik nisbati o'zgaradimi?
- `dtype=torch.float16` bilan `quantize_dynamic` chaqirib ko'ring (CPU'da qo'llab-quvvatlanadimi?).
- Per-channel masshtab bilan qo'lda kvantlab, qayta tiklash xatosini per-tensor bilan solishtiring.

### 4-qadam. ONNX Runtime'da kvantlash: dinamik va statik (MinMax, Percentile, shovqin)

CNN ONNX modeli `quant_pre_process` dan o'tkazilib, `quantize_dynamic` va `quantize_static` (QDQ format) bilan kvantlanadi: per-channel MinMax, per-channel Percentile, per-tensor MinMax va ataylab noto'g'ri — tasodifiy shovqin bilan kalibrlash (A3 matnidagi 3-xato). Har variant uchun hajm, aniqlik, p50, throughput va Q/DQ tugunlar soni jadvalga yig'iladi.

Fayl: `s4_4_qadam_onnx_runtimeda_kvantlash_dinamik.py` — ishga tushirish: `python3 s4_4_qadam_onnx_runtimeda_kvantlash_dinamik.py`

Kutilgan natija: Hajm 17 → 11–13 KB; real kalibrlash aniqlikni saqlaydi (±0,3 p.p.), shovqinli kalibrlash 1 p.p. atrofida yo'qotadi; INT8 bu CPU'da tezroq emas.

Tajriba qilib ko'ring:

- Kalibrlash to'plamini 320 dan 32 va 1000 ga o'zgartiring — aniqlik qanday?
- `CalibrationMethod.Entropy` qo'shing va uch usulni solishtiring.
- `nodes_to_exclude` bilan birinchi Conv'ni FP32 da qoldiring (sezgir qatlam) — aniqlik tiklanadimi?

### 5-qadam. Yakuniy jadval, Pareto tahlili va grafik

6 ta konfiguratsiya (PyTorch FP32, PyTorch INT8 dinamik, ORT FP32, ORT INT8 dinamik, ORT statik MinMax/Percentile) bitta jadvalga yig'iladi: hajm, aniqlik, p50 (batch 1), throughput (batch 256), tezlanish. Pareto-optimal nuqtalar (latency kichik, aniqlik katta) aniqlanadi va ikki o'qli grafik chiziladi: latency–aniqlik va hajm–throughput.

Fayl: `s5_5_qadam_yakuniy_jadval_pareto_tahlili_va.py` — ishga tushirish: `python3 s5_5_qadam_yakuniy_jadval_pareto_tahlili_va.py`

Kutilgan natija: ORT variantlari PyTorch'dan 4–7 marta tez; Pareto chegarasida ORT FP32 va statik Percentile; grafikda qizil nuqtalar — Pareto.

Tajriba qilib ko'ring:

- Pareto mezonini (hajm, aniqlik) juftligiga o'zgartiring — qaysi konfiguratsiyalar qoladi?
- Uchinchi o'q sifatida hajmni nuqta kattaligi bilan ko'rsating.
- O'qitish epoxasini 1 ga tushiring: aniqlik farqlari kvantlash tufaylimi yoki o'qitish shovqinimi?

### 6-qadam. Energiya bahosi, nusxalar rejasi va GPU/TensorRT yo'l xaritasi

RAPL/nvidia-smi yo'qligi sababli quvvat MODEL bilan baholanadi: P = P_bosh + (P_to'liq − P_bosh)·CPU_bandlik, bandlik `time.process_time()/devor vaqti` dan olinadi. FP32 va INT8 dinamik, 1 va 2 oqim, batch 1 va 64 uchun µJ/inferens (to'liq va dinamik) hamda inferens/s/W hisoblanadi; Little qonuni bilan 5000 so'rov/s uchun nusxalar va quvvat; oxirida GPU'da TensorRT FP16/INT8 va quvvat o'lchash qadamlari (buyruqlar va kutilgan raqamlar).

Fayl: `s6_6_qadam_energiya_bahosi_nusxalar_rejasi_.py` — ishga tushirish: `python3 s6_6_qadam_energiya_bahosi_nusxalar_rejasi_.py`

Kutilgan natija: Batch 64 da energiya batch 1 dan 5–15 marta kam; 2 oqim batch 1 da energiya ortadi (bandlik 2x, tezlik oshmaydi); ustunli grafik.

Tajriba qilib ko'ring:

- P_TOLIK_W va P_BOSH_W ni o'z noutbukingiz TDP'siga moslang — xulosa o'zgaradimi?
- `psutil.Process().cpu_times()` bilan user/system vaqtni ajrating.
- Batch 256 va 1024 qo'shing: µJ/inferens qachon to'yinadi?

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
|---|---|---|---|
| 1 | **Opset va tugunlar** — 1-qadam kodida modelni opset 13, 15 va 17 bilan eksport qiling. Har biri uchun tugunlar soni, operatorlar ro'yxati, fayl hajmi va PyTorch bilan maksimal farqni jadvalga yozing. Qaysi opset'da farq bor-yo'qligini izohlang. | oson | 5 |
| 2 | **Optimallashtirish darajalari jadvali** — 2-qadamni `takror=1000` bilan 3 marta qayta bajaring va har daraja uchun p50/p95/p99 o'rtachasini va tarqalishini (min–max) hisoblang. Qaysi daraja barqaror tezlanish beradi? Natijani 4 qatorli jadval va 2–3 jumla xulosa bilan topshiring. | oson | 5 |
| 3 | **Kvantlash formulasi** — 3-qadamdagi qo'lda kvantlash kodini asimmetrik variantga (s = (max−min)/255, z = round(−min/s)) o'zgartiring va simmetrik bilan qayta tiklash xatosini (max, o'rtacha) uchala Linear qatlam uchun solishtiring. Qaysi qatlamda farq katta va nega (vaznlar taqsimotini gistogramma bilan ko'rsating)? | oson | 5 |
| 4 | **Kalibrlash to'plami hajmi** — 4-qadamda statik kvantlashni 8, 32, 128, 320 va 1000 namunali kalibrlash to'plamlari bilan (MinMax va Percentile) bajaring. Aniqlik va kvantlash vaqtini jadvalga yozing, aniqlik–hajm grafigini chizing va kalibrlash to'plamining minimal yetarli hajmini asoslang. | o'rta | 10 |
| 5 | **Sezgir qatlam tahlili** — 4-qadam kodida `quantize_static` ning `nodes_to_exclude` parametri bilan har bir Conv/Gemm tugunini navbatma-navbat FP32 da qoldiring (qolganlari INT8). Har konfiguratsiya uchun aniqlik va hajmni o'lchang, eng sezgir qatlamni aniqlang va «INT8 + 1 ta FP32 qatlam» aralash variantini to'liq INT8 bilan solishtiring. | o'rta | 10 |
| 6 | **Paket va kechikish byudjeti** — 2-qadamni batch 1, 2, 4, 8, 16, 32, 64, 128, 256 uchun bajarib, latency va throughput egri chiziqlarini chizing. p99 kechikish 0,1 ms dan oshmasligi talabi bilan maksimal paket o'lchamini toping va Little qonuni orqali 10 000 so'rov/s uchun kerakli nusxalar sonini ikki variantda (batch 1 va topilgan maksimal batch) hisoblang. | o'rta | 10 |
| 7 | **MLP vs CNN: dinamik kvantlash farqi** — 3- va 4-qadamlar asosida MLP (Linear og'ir) va CNN (Conv og'ir) uchun dinamik INT8 kvantlashning hajm, aniqlik va tezlik ta'sirini bir xil metodika bilan o'lchang (PyTorch va ORT ikkalasida). A3 matnidagi «dinamik kvantlash MatMul og'ir modellarga mos» tezisini o'z raqamlaringiz bilan tasdiqlang yoki rad eting. | o'rta | 10 |
| 8 | **QAT bilan aniqlikni tiklash** — 4-qadamdagi shovqin bilan kalibrlangan (yoki ataylab per-tensor, 4 bitli ta'sirni simulyatsiya qilish uchun vaznlarni round(w/s) bilan buzilgan) modelni oling va PyTorch'da soxta kvantlash (fake quantization: forward'da vaznlarni kvantlab-qayta tiklash, backward oddiy) bilan 2 epoxa qayta o'qiting. INT8 aniqligini FP32 ga 0,5 p.p. ichida yaqinlashtiring; o'qitish egri chizig'i va yakuniy jadval bilan hisobot bering. | qiyin | 15 |
| 9 | **Latency taqsimoti va ishonch intervali** — 5-qadamdagi 6 konfiguratsiyani har biri 5 marta, 1000 takror bilan o'lchang. p50 va p99 uchun bootstrap (1000 qayta tanlash) 95 % ishonch intervalini hisoblang, konfiguratsiyalar orasidagi farq statistik ahamiyatlimi — aniqlang (intervallar kesishadimi). Shuningdek, 450 ta test namunasida 0,2 p.p. aniqlik farqining ishonch intervalini (binomial) baholang va «qancha namuna kerak» savoliga javob bering. | qiyin | 15 |
| 10 | **To'liq Pareto hisoboti o'z modelingiz uchun** — O'zingiz tanlagan boshqa arxitektura (masalan, 3 qatlamli kengroq CNN yoki `make_classification` dagi 100 belgili MLP) uchun 1–6 qadamlarning barchasini takrorlang: eksport, ORT darajalari, PyTorch va ORT kvantlash (kamida 2 kalibrlash usuli), yakuniy jadval, Pareto grafigi, energiya bahosi. 4–6 betlik hisobotda aniq ssenariy (masalan, A1 dagi SHV tizimi) uchun konfiguratsiya tanlang va «GPU bo'lsa nima o'zgaradi» bo'limini yozing. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Tibbiyot: ko'krak saratoni skrining MLP — ONNX eksport va ORT darajalari (oson)

Soha: Tibbiyot (onkologiya)

Klinika laboratoriyasi 30 ta biopsiya belgisiga ko'ra xavfli/xavfsiz o'smani tasniflovchi modelni analizator qurilmasidagi CPU'ga joylashtirmoqchi. Qurilma ishlab chiqaruvchisi faqat ONNX formatini qabul qiladi va bitta namuna uchun p99 kechikish 5 ms dan oshmasligini talab qiladi. Sizdan FP32 etalonni o'qitish, ONNX'ga to'g'ri eksport qilish, ORT optimallashtirish darajalarini o'lchash va qaysi daraja qurilma talabiga mos kelishini asoslash so'raladi. Aniqlik birinchi o'rinda: recall 0,95 dan pastga tushmasligi kerak.

| Parametr | Qiymat |
|---|---|
| Model | MLP 30-64-32-2 (ReLU, Dropout 0,2) |
| Ma'lumot | sklearn load_breast_cancer (569 namuna, 30 belgi), StandardScaler |
| Epoxa | 15, Adam lr 1e-3, paket 32 |
| Opset | 13 va 17 (solishtiriladi) |
| Batch | 1, 8, 64 |
| SLO | p99 <= 5 ms (batch 1), recall >= 0,95 |
| ORT darajalari | DISABLE_ALL, BASIC, EXTENDED, ALL |

Qadamlar:

1. Laboratoriya s1 kabi MLP'ni o'qiting (15 epoxa), test to'plamida aniqlik va recall'ni, PyTorch eager batch 1 p50/p95/p99 kechikishini o'lchang.
2. Modelni opset 13 va opset 17 bilan ONNX'ga eksport qiling (dinamik batch o'qi), onnx.checker bilan tekshiring, tugunlar sonini, operatorlar ro'yxatini va fayl hajmini ikkala opset uchun jadvalga yozing.
3. PyTorch va ORT chiqishlarini 64 namunada np.allclose (rtol 1e-3) bilan solishtiring; Dropout eval rejimida grafikdan yo'qolganini ko'rsating.
4. Laboratoriya s2 kabi ORT 4 ta optimallashtirish darajasini batch 1 da o'lchang (takror 500), har daraja uchun p50/p95/p99 va optimallashtirilgan grafikdagi tugunlar sonini jadvalga yozing.
5. Batch 1, 8, 64 uchun throughput va Little qonuni bo'yicha N ni hisoblang; SLO p99 <= 5 ms bajariladimi, qaysi darajada eng katta zaxira qolishini ko'rsating.
6. Xulosa: qurilmaga qaysi opset va daraja tavsiya etiladi, recall saqlanganmi, GPU bo'lsa nima o'zgarardi (2-3 jumla).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 2-variant. Agro: tuproq sensorlari bo'yicha sug'orish tavsiyasi — chekka qurilma uchun dinamik INT8 (oson)

Soha: Agro

Fermer xo'jaligi dalalarida 20 ta ko'rsatkichni (namlik, harorat, EC, pH, azot va boshqalar) o'lchaydigan avtonom sensor stansiyalari bor. Stansiya ichidagi kichik ARM protsessor va 2 MB flesh xotira uchun 4 sinfli sug'orish tavsiyasi modeli kerak: model fayli 100 KB dan kichik bo'lishi, batareyada ishlashi kerak. Siz MLP'ni o'qitib, PyTorch dinamik INT8 kvantlashini qo'llaysiz, hajm va aniqlikni solishtirasiz hamda qnnpack dvigatelida (ARM uchun aynan shu dvigatel ishlatiladi) tezlikni o'lchaysiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 20-256-128-4 |
| Ma'lumot | make_classification(n_samples=6000, n_features=20, n_informative=12, n_classes=4, random_state=7) |
| Epoxa | 8, Adam lr 2e-3 |
| Hajm cheklovi | <= 100 KB (state_dict) |
| Batch | 1 va 128 |
| Kvantlash dvigateli | qnnpack (ARM muqobili) |
| Kutilgan aniqlik | >= 85 % |

Qadamlar:

1. Sintetik tuproq ma'lumotini yarating (random_state=7), 75/25 ga bo'ling, MLP'ni 8 epoxa o'qiting va FP32 aniqligini yozing.
2. Laboratoriya s3 kabi quantize_dynamic({nn.Linear}, qint8) qo'llang; kvantlangan model tuzilmasini chop eting va DynamicQuantizedLinear qatlamlar sonini ayting.
3. Birinchi qatlam vazni uchun simmetrik kvantlash formulasini qo'lda tekshiring (s = max|w|/127): qayta tiklash xatosi (max, o'rtacha), noyob qiymatlar soni.
4. FP32 va INT8 uchun hajm (KB), aniqlik, p50 kechikish batch 1 va batch 128 ni jadvalga yozing; hajm 100 KB chegarasiga sig'adimi, tekshiring.
5. Qatlam kengligini 256 dan 512 ga oshirib takrorlang: hajm nisbati 4x ga yaqinlashadimi, aniqlik o'sadimi, chegara buziladimi.
6. Xulosa: stansiyaga qaysi konfiguratsiya joylanadi, VNNI yo'q CPU'da tezlik nega oshmagani, ARM NEON'da nimani kutish mumkin.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 3-variant. Transport: yo'l belgilarini tanish CNN — statik INT8 va uch kalibrlash usuli (o'rta)

Soha: Transport

Avtomobil ichidagi yordamchi tizim kameradan kesib olingan kichik belgilar tasvirini tasniflaydi; ishlab chiqaruvchi bort kompyuterida faqat INT8 inferensga ruxsat beradi (issiqlik va quvvat byudjeti tufayli). Siz digits to'plamini belgilar surrogati sifatida ishlatib, kengroq CNN o'qitasiz, ONNX Runtime'da statik kvantlashni MinMax, Percentile va Entropy usullari bilan bajarasiz va noto'g'ri kalibrlash (boshqa yoritilganlik — pikselga doimiy siljish) qanchalik zarar qilishini ko'rsatasiz. Talab: p95 kechikish 2 ms dan kam, aniqlik yo'qotish 0,5 p.p. dan oshmasin.

| Parametr | Qiymat |
|---|---|
| Model | CNN Conv(1->16)-Conv(16->32)-FC 128-10 |
| Ma'lumot | sklearn digits 8x8, 70/30 bo'linish, random_state=3 |
| Kalibrlash to'plami | 256 namuna |
| Kalibrlash usullari | MinMax, Percentile, Entropy + noto'g'ri (pikselga +0,5 siljish) |
| SLO | p95 <= 2 ms (batch 1), aniqlik farqi <= 0,5 p.p. |
| Format | QDQ, per-channel, QUInt8 faollashuv / QInt8 vazn |

Qadamlar:

1. Laboratoriya s1/s4 asosida CNN'ni 4 epoxa o'qiting, ONNX'ga eksport qiling, quant_pre_process dan o'tkazing; FP32 aniqligi va p50/p95 ni yozing.
2. 256 namunali real kalibrlash to'plami bilan quantize_static ni MinMax, Percentile va Entropy usullarida bajaring (QDQ, per-channel).
3. Noto'g'ri kalibrlash: xuddi shu 256 namunaga +0,5 doimiy siljish qo'shib (yoritilganlik o'zgarishi) MinMax bilan kvantlang.
4. Har 5 konfiguratsiya (FP32 + 4 INT8) uchun hajm, aniqlik, p50/p95 batch 1, throughput batch 256 va Q/DQ tugunlar sonini jadvalga yig'ing.
5. Kalibrlash to'plami hajmini 32 va 1000 ga o'zgartirib Percentile uchun aniqlikni qayta o'lchang; aniqlik-hajm grafigini chizing.
6. Xulosa: qaysi usul SLO ga mos, noto'g'ri kalibrlash qancha p.p. yo'qotdi, bort kompyuterida (VNNI/NEON) INT8 tezlanishini qanday tekshirish kerak.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 4-variant. Telekom: tarmoq trafigi anomaliyasi uchun autoenkoder — ONNX va yuqori throughput (o'rta)

Soha: Telekom

Mobil operatorning yadro tarmog'ida har soniyada 40 000 ta oqim yozuvi (flow record) hosil bo'ladi; ularni 16 belgili autoenkoder bilan tekshirib, rekonstruksiya xatosi katta bo'lganlarini anomaliya deb belgilash kerak. Model GPU'siz 2 yadroli virtual mashinada ishlaydi, shuning uchun throughput asosiy mezon: nechta nusxa kerakligini Little qonuni bilan hisoblaysiz. Siz autoenkoderni sintetik trafikda o'qitasiz, ONNX'ga eksport qilasiz, paket o'lchami va oqimlar sonini o'lchaysiz va dinamik INT8 ta'sirini tekshirasiz.

| Parametr | Qiymat |
|---|---|
| Model | Autoenkoder 16-32-8-32-16 (MSE) |
| Ma'lumot | sintetik: normal oqimlar make_blobs(1 markaz, 16 belgi, std 1,0) 20000 ta + 2 % anomaliya (std 4,0) |
| Anomaliya chegarasi | normal xatolarning 99-foizili |
| Batch | 1, 64, 512, 2048 |
| Oqimlar | intra_op 1 va 2 |
| Talab | 40 000 yozuv/s, p99 (batch) <= 50 ms |
| Metrika | precision/recall anomaliya bo'yicha |

Qadamlar:

1. Sintetik trafikni yarating, autoenkoderni faqat normal yozuvlarda 10 epoxa o'qiting; test to'plamida rekonstruksiya xatosi va 99-foizil chegara bo'yicha precision/recall ni hisoblang.
2. ONNX'ga eksport qiling (dinamik batch), onnx.checker, PyTorch bilan np.allclose mosligi; tugunlar soni va hajm.
3. Laboratoriya s2 kabi ORT 4 darajasini batch 1 da o'lchang, keyin batch 1/64/512/2048 uchun p50/p99 va throughput (yozuv/s) jadvalini tuzing.
4. intra_op 1 va 2 oqimni batch 2048 da solishtiring, tezlanish koeffitsientini yozing.
5. ORT quantize_dynamic bilan INT8 variant yarating: precision/recall o'zgaradimi, throughput o'zgaradimi (VNNI yo'q — izohlang).
6. Little qonuni: 40 000 yozuv/s uchun har batch o'lchamida nechta nusxa kerak; p99 <= 50 ms sharti bilan optimal batch'ni tanlang.
7. Xulosa va grafik: batch - throughput, batch - p99; tavsiya etilgan konfiguratsiya.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 5-variant. Energetika: soatlik elektr iste'moli prognozi — regressiya modelida kvantlash xatosi (oson)

Soha: Energetika

Taqsimot kompaniyasi 24 soatlik oyna va ob-havo belgilaridan keyingi soat iste'molini bashorat qiluvchi modelni 1200 ta transformator podstansiyasidagi kontrollerga joylashtirmoqchi. Kontrollerlar eski x86 CPU'li, xotira 256 MB; har 5 daqiqada bitta prognoz kerak, ya'ni tezlik emas, hajm va aniqlik muhim. Siz MLP regressorni o'qitasiz, ONNX'ga eksport qilib ORT'da tekshirasiz, PyTorch va ORT dinamik kvantlashning MAE/MAPE ga ta'sirini o'lchaysiz va tasniflashdan farqli ravishda regressiyada kvantlash xatosi qanday namoyon bo'lishini tushuntirasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 30-128-64-1 (regressiya, MSE) |
| Ma'lumot | make_regression(n_samples=8000, n_features=30, n_informative=20, noise=8, random_state=11), nishon MinMax [0,1] |
| Epoxa | 20, Adam lr 1e-3 |
| Metrika | MAE, MAPE, R^2 |
| Batch | 1 va 256 |
| Cheklov | hajm <= 60 KB, MAE o'sishi <= 2 % |
| Opset | 17 |

Qadamlar:

1. Sintetik iste'mol ma'lumotini yarating (random_state=11), nishonni [0,1] ga normallang, MLP'ni 20 epoxa o'qiting; test MAE, MAPE, R^2 ni yozing.
2. ONNX eksport (opset 17, dinamik batch), onnx.checker; ORT va PyTorch chiqishlarini 256 namunada maksimal farq bilan solishtiring.
3. ORT ALL darajasida batch 1 va 256 uchun p50 kechikish va throughput o'lchang (laboratoriya s2 funksiyasi bilan).
4. PyTorch quantize_dynamic va ORT quantize_dynamic bilan ikki INT8 variant yarating; hajm, MAE, MAPE, R^2, p50 ni FP32 bilan bitta jadvalga yig'ing.
5. Kvantlash xatosini tahlil qiling: FP32 va INT8 bashoratlari farqining gistogrammasini chizing, eng katta farq qaysi diapazonda ekanini ayting.
6. Xulosa: kontrollerga qaysi variant mos, hajm chegarasi bajarildimi, regressiyada «aniqlik o'zgarmadi» deyish nega noto'g'ri bo'lishi mumkin.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 6-variant. Moliya: kredit skoring — tasodifiy o'rmondan MLP distillyatsiyasi va ONNX (qiyin)

Soha: Moliya (bank krediti)

Bank kredit skoringini sklearn RandomForest bilan bajaradi, lekin yangi onlayn xizmat har bir so'rovga 1 ms dan kam kechikish talab qiladi va faqat ONNX Runtime'ni qo'llab-quvvatlaydi; sandbox'da skl2onnx yo'q. Siz daraxt ansamblini «o'qituvchi» sifatida olib, uning ehtimolliklarini MLP «o'quvchi»ga distillyatsiya qilasiz, MLP'ni ONNX'ga eksport qilasiz va sklearn predict_proba bilan ORT kechikishini solishtirasiz. Keyin o'quvchini INT8 ga kvantlab, AUC va «tasdiqlangan arizalar ulushi» o'zgarishini tekshirasiz — bank uchun AUC 0,01 dan ko'p pasaymasligi shart.

| Parametr | Qiymat |
|---|---|
| O'qituvchi | RandomForest 200 daraxt, max_depth 10 |
| O'quvchi | MLP 24-128-64-2 (soft-label KL yo'qotish, T=2) |
| Ma'lumot | make_classification(n_samples=20000, n_features=24, n_informative=14, weights=[0.8,0.2], random_state=21) |
| Metrika | AUC, aniqlik, tasdiqlash chegarasi 0,5 da tasdiqlanganlar ulushi |
| SLO | p99 <= 1 ms (batch 1) |
| Batch | 1, 32, 1024 |
| Kvantlash | ORT dinamik + statik Percentile (512 namuna) |

Qadamlar:

1. Sintetik kredit ma'lumotini yarating, RandomForest'ni o'qiting, test AUC va bitta so'rov uchun predict_proba p50/p99 kechikishini (300 takror) o'lchang.
2. O'qituvchining ehtimolliklari bilan MLP'ni distillyatsiya qiling (KL + CE aralash, 15 epoxa); o'quvchi AUC'si o'qituvchidan qancha farq qilishini yozing. Taqqoslash uchun oddiy yorliqlar bilan o'qitilgan MLP ham qo'shing.
3. O'quvchini ONNX'ga eksport qiling (laboratoriya s1), ORT 4 daraja va batch 1/32/1024 uchun kechikish va throughput jadvalini tuzing; sklearn RF bilan tezlik nisbatini hisoblang.
4. ORT dinamik va statik (Percentile, 512 namuna) INT8 kvantlash: hajm, AUC, p50, tasdiqlanganlar ulushi; FP32 bilan farq.
5. Pareto tahlili (p50 - AUC) va grafik; RF, MLP FP32, MLP INT8 nuqtalarini bitta grafikda ko'rsating.
6. Tasdiqlangan arizalar ulushi kvantlashdan keyin o'zgargan bo'lsa, bu nechta mijozga ta'sir qiladi (20 000 ta ariza misolida) — hisoblang va izohlang.
7. Xulosa: distillyatsiya qancha aniqlik yo'qotdi va qancha tezlik berdi, skl2onnx bo'lsa qanday bo'lardi, bank uchun tavsiya.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); O'qituvchi-o'quvchi taqqoslash jadvali (AUC, kechikish, hajm)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 7-variant. Ta'lim: talaba javoblari ketma-ketligi uchun kichik transformer bloki — ONNX, fusion va MatMul kvantlash (qiyin)

Soha: Ta'lim

Adaptiv test platformasi talabaning oxirgi 16 ta javobi (savol identifikatori va to'g'ri/noto'g'ri) asosida keyingi savolga javob ehtimolini bashorat qiladi. Model — bitta self-attention bloki va ikki Linear qatlam; platformaning inferens serveri GPU'siz, bir vaqtda 1500 ta talaba ishlaydi. Siz transformer blokini sintetik ketma-ketliklarda o'qitasiz, ONNX'ga eksport qilasiz, ORT EXTENDED/ALL darajalarida Attention/LayerNorm birlashuvi (fusion) bo'lgan-bo'lmaganini tugunlar bo'yicha tekshirasiz, MatMul og'ir modelda dinamik INT8 kvantlashni baholaysiz va p99 10 ms SLO uchun nusxalar sonini hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | Embedding(200, 32) + MultiheadAttention(32, 4 bosh) + LayerNorm + FFN 32-64-32 + Linear 32-1 |
| Ma'lumot | sintetik: 20000 ketma-ketlik, uzunlik 16, savol ID 0..199, javob yorlig'i talaba qobiliyati va savol qiyinligidan logistik model bilan |
| Epoxa | 6, Adam lr 1e-3, BCE |
| Batch | 1, 16, 128 |
| SLO | p99 <= 10 ms, 1500 so'rov/s |
| Opset | 17 |
| Kvantlash | PyTorch quantize_dynamic (Linear), ORT quantize_dynamic (MatMul) |

Qadamlar:

1. Sintetik javob ketma-ketliklarini yarating (qobiliyat ~ N(0,1), qiyinlik ~ N(0,1), p = sigmoid(qobiliyat - qiyinlik)), modelni 6 epoxa o'qiting, test AUC ni yozing.
2. ONNX'ga eksport qiling (dinamik batch, opset 17); onnx.checker; operatorlar ro'yxatini chop eting (MatMul, Softmax, LayerNormalization yoki uning qismlari bormi).
3. ORT 4 darajasida optimallashtirilgan grafikni faylga saqlab tugunlarni sanang; EXTENDED/ALL da qaysi birlashgan operatorlar (fused) paydo bo'lganini jadvalga yozing.
4. Batch 1/16/128 uchun p50/p99 va throughput; intra_op 1 va 2 oqim taqqosi.
5. PyTorch va ORT dinamik INT8: hajm, AUC, p50 batch 1, throughput batch 128; embedding qatlami kvantlandimi, tekshiring.
6. Little qonuni: 1500 so'rov/s va p99 bo'yicha nusxalar soni FP32 va INT8 uchun; Pareto grafigi (p50 - AUC).
7. Xulosa: fusion qancha tezlik berdi, MatMul og'ir model uchun dinamik kvantlash tezisi tasdiqlandimi, GPU/VNNI bo'lsa nimalar o'zgaradi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Operatorlar/fusion jadvali (4 daraja)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 8-variant. Sanoat: podshipnik vibratsiyasi bo'yicha nosozlik tasnifi — 1D CNN va sezgir qatlam tahlili (o'rta)

Soha: Sanoat (mashinasozlik)

Zavod nasos stansiyasidagi 300 ta dvigatelning vibratsiya signallari (256 nuqtalik oynalar) 4 sinfga (soz, ichki halqa, tashqi halqa, sharcha nuqsoni) tasniflanadi. Model sexdagi sanoat kompyuteriga joylanadi, u yerda xotira kam va INT8 ma'qul, lekin nosozlikni o'tkazib yuborish qimmat: har sinf bo'yicha recall 0,9 dan pastga tushmasin. Siz 1D CNN'ni sintetik vibratsiya signallarida o'qitasiz, ONNX statik kvantlashni bajarasiz va nodes_to_exclude bilan qaysi Conv qatlami kvantlashga eng sezgir ekanini aniqlab, aralash (INT8 + 1 FP32 qatlam) variantni taklif qilasiz.

| Parametr | Qiymat |
|---|---|
| Model | 1D CNN: Conv1d(1->8,k7)-Pool-Conv1d(8->16,k5)-Pool-Conv1d(16->32,k3)-GAP-Linear 32-4 |
| Ma'lumot | sintetik vibratsiya: 256 nuqta, asosiy sinusoida 50 Hz + sinfga xos zarba chastotasi (0/107/162/141 Hz) + Gauss shovqin std 0,3; 4000 namuna |
| Epoxa | 8, Adam lr 2e-3 |
| Kalibrlash | MinMax va Percentile, 200 namuna, per-channel |
| Metrika | aniqlik, sinf bo'yicha recall |
| Cheklov | recall >= 0,9 har sinfda, hajm minimal |
| Batch | 1 va 128 |

Qadamlar:

1. Sintetik vibratsiya generatorini yozing (4 sinf, zarba chastotalari), 1D CNN'ni o'qiting, test aniqligi va sinflar bo'yicha recall jadvali.
2. ONNX eksport (dinamik batch), quant_pre_process; FP32 ORT p50 batch 1 va throughput batch 128.
3. Laboratoriya s4 kabi statik kvantlash (MinMax, Percentile, QDQ per-channel) va dinamik kvantlash; hajm, aniqlik, recall, p50, Q/DQ tugunlar soni jadvali.
4. nodes_to_exclude bilan uchta Conv va Gemm tugunini navbatma-navbat FP32 da qoldirib 4 ta aralash variant yarating; har biri uchun aniqlik va eng past recall'ni o'lchang.
5. Eng sezgir qatlamni aniqlang va «INT8 + 1 FP32 qatlam» variantini to'liq INT8 va FP32 bilan Pareto (hajm - eng past recall) grafigida ko'rsating.
6. Xulosa: sanoat kompyuteriga qaysi variant, recall talabi bajarildimi, nega birinchi qatlam ko'pincha sezgir bo'ladi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Sezgir qatlam jadvali (4 aralash variant)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 9-variant. Kiberxavfsizlik: login hodisalari bo'yicha hujum aniqlash — keng MLP va 4x hajm chegarasi (oson)

Soha: Xavfsizlik

Korporativ SIEM tizimi har soniyada 25 000 ta autentifikatsiya hodisasini (40 belgi: vaqt, geolokatsiya, qurilma, xatolar soni va h.k.) hujum/normal deb tasniflaydi. Tasniflagich SIEM agentining ichida ishlaydi, agent paketi o'lchami qat'iy cheklangan, shuning uchun model hajmi kamida 3,5 marta qisqarishi kerak. Siz keng MLP'ni sintetik hodisalarda o'qitasiz, PyTorch dinamik INT8 kvantlash bilan hajm, aniqlik va tezlikni o'lchaysiz hamda batch 1 da p99 1 ms SLO ni tekshirasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 40-512-512-2 |
| Ma'lumot | make_classification(n_samples=30000, n_features=40, n_informative=25, weights=[0.95,0.05], flip_y=0.01, random_state=5) |
| Epoxa | 5, Adam lr 1e-3, paket 128 |
| Hajm talabi | >= 3,5x qisqarish |
| SLO | p99 <= 1 ms (batch 1) |
| Batch | 1, 256 |
| Metrika | aniqlik, hujum sinfi uchun recall va F1 |

Qadamlar:

1. Sintetik hodisalarni yarating (muvozanatsiz, 5 % hujum), MLP'ni o'qiting; aniqlik, recall va F1 (hujum sinfi) ni yozing.
2. Laboratoriya s3 kabi quantize_dynamic qo'llang (qnnpack), kvantlangan tuzilmani chop eting; 2-qatlam (512x512) vazni uchun masshtab s, qayta tiklash xatosi va noyob qiymatlar sonini hisoblang.
3. FP32 va INT8: hajm KB, aniqlik, recall, F1, p50/p99 batch 1 va p50 batch 256 jadvali; hajm nisbati 3,5x dan katta-kichikligini tekshiring.
4. Per-channel masshtab bilan qo'lda kvantlab (har qator uchun alohida s) qayta tiklash xatosini per-tensor bilan solishtiring.
5. ONNX'ga eksport qilib ORT ALL da batch 1 p99 ni o'lchang; SLO 1 ms uchun PyTorch INT8, ORT FP32 qaysi biri mos — jadvalda ko'rsating.
6. Xulosa: agentga qaysi format, recall saqlanganmi, VNNI yo'q CPU'da tezlik natijasi va serverda nimani kutish kerak.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 10-variant. Chakana savdo: mahsulot tavsiyasi uchun embedding + MLP — Gather kvantlash va katta paket (o'rta)

Soha: Chakana savdo

Onlayn do'kon 5000 ta mahsulot va 20 000 ta foydalanuvchi uchun «sotib oladi/olmaydi» ehtimolini hisoblaydi; tungi qayta hisoblashda 100 million juftlik baholanadi, kunduzi esa bitta foydalanuvchi uchun 200 ta nomzodni 20 ms ichida saralash kerak. Model — ikkita embedding va MLP; eng katta hajm embeddinglarda. Siz modelni sintetik o'zaro ta'sirlar (interactions) ustida o'qitasiz, ONNX'ga eksport qilasiz, katta paketlarda throughput'ni, dinamik va statik kvantlashda embeddinglar (Gather) kvantlanadimi-yo'qmi tekshirasiz va tungi hisob uchun vaqtni baholaysiz.

| Parametr | Qiymat |
|---|---|
| Model | Embedding(20000, 16) + Embedding(5000, 16) + MLP 32-64-1 |
| Ma'lumot | sintetik: foydalanuvchi va mahsulot yashirin vektorlari N(0,1), yorliq = sigmoid(skalyar ko'paytma + shovqin) > 0,5; 300 000 juftlik |
| Epoxa | 3, Adam lr 3e-3, paket 512 |
| Batch (onlayn) | 200 nomzod bitta so'rovda |
| Batch (tungi) | 4096, 16384 |
| SLO | p95 <= 20 ms (200 nomzod), tungi 100 mln juftlik <= 2 soat 1 nusxada |
| Metrika | AUC |

Qadamlar:

1. Sintetik o'zaro ta'sirlarni yarating, modelni 3 epoxa o'qiting, test AUC ni yozing; parametrlar sonini embedding va MLP bo'yicha ajrating.
2. ONNX'ga eksport qiling (ikki kirish: user_id, item_id, dinamik batch), onnx.checker; operatorlar ichida Gather borligini ko'rsating; hajm.
3. ORT ALL: batch 200 uchun p50/p95 (SLO tekshiruvi), batch 4096 va 16384 uchun throughput (juftlik/s); 100 mln juftlik uchun vaqtni hisoblang.
4. ORT quantize_dynamic va quantize_static (MinMax, 1024 juftlik kalibrlash): hajm, AUC, throughput; Gather/embedding vaznlari kvantlanganmi — grafik tugunlarini tekshiring.
5. Agar embedding kvantlanmagan bo'lsa, embedding jadvallarini qo'lda INT8 ga (per-qator masshtab) kvantlab qayta tiklang va AUC ni qayta o'lchang; hajm nisbatini hisoblang.
6. Pareto (hajm - AUC) va (throughput - AUC) grafiklari; onlayn va tungi rejim uchun alohida tavsiya bering.
7. Xulosa: hajm qayerda, qaysi kvantlash foyda berdi, 1 nusxada tungi hisob ulgurmasa nechta nusxa kerak.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 11-variant. Logistika: qo'l skanerida yetkazib berish vaqti prognozi — energiya bahosi va batareya umri (o'rta)

Soha: Logistika

Kuryerlik kompaniyasi 4000 ta batareyali qo'l skaneriga yetkazib berish muddatini (daqiqa) bashorat qiluvchi modelni o'rnatmoqda; har skaner kuniga 1500 ta prognoz qiladi va bir zaryadda kamida 12 soat ishlashi kerak. Shuning uchun asosiy mezon — inferens boshiga energiya (uJ). Siz MLP regressorni o'qitasiz, ONNX FP32 va INT8 dinamik variantlarini 1 va 2 oqimda, batch 1 va 32 da laboratoriya s6 quvvat modeli bilan baholaysiz, lekin quvvat parametrlarini mobil protsessorga moslaysiz (P_bosh 0,4 Vt, P_to'liq 2,5 Vt) va kunlik energiya sarfini mAh ga aylantirasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 18-96-48-1 (regressiya) |
| Ma'lumot | make_regression(n_samples=10000, n_features=18, n_informative=12, noise=5, random_state=13); nishon 10..180 daqiqaga masshtablangan |
| Quvvat modeli | P_bosh 0,4 Vt, P_to'liq 2,5 Vt (2 yadro) |
| Batareya | 3000 mAh, 3,8 V |
| Batch | 1 va 32 |
| Oqimlar | 1 va 2 |
| Kunlik yuklama | 1500 prognoz/skaner |
| Metrika | MAE (daqiqa) |

Qadamlar:

1. Sintetik ma'lumotni yarating, MLP'ni 15 epoxa o'qiting, test MAE ni daqiqada yozing.
2. ONNX eksport + quant_pre_process + ORT quantize_dynamic; FP32 va INT8 uchun hajm va MAE.
3. Laboratoriya s6 energiya funksiyasini mobil quvvat parametrlari bilan ishlating: FP32/INT8 x 1/2 oqim x batch 1/32 uchun latency, CPU bandlik, P, uJ_tolik, uJ_dinamik, inferens/s/Vt jadvali.
4. Kunlik energiya: 1500 prognoz x uJ_tolik ni J va mAh (3,8 V) ga aylantiring; batareyaning necha foizi inferensga ketishini har konfiguratsiya uchun hisoblang.
5. Ustunli grafik (uJ/inferens, log o'q) va eng tejamkor konfiguratsiyani belgilang; 2 oqim batch 1 da nega energiya ko'payganini tushuntiring.
6. Xulosa: skanerga tavsiya, 12 soat talabi bajariladimi, ARM qurilmada RAPL o'rniga qanday o'lchash mumkin (batareya hisoblagichi).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Energiya jadvali (8 konfiguratsiya) va kunlik mAh hisobi

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 12-variant. Shahar: chorraha trafik xaritasi bo'yicha tiqilinch bashorati — kichik CNN, oqimlar va Little qonuni (oson)

Soha: Shahar

Shahar transport markazi 600 ta chorrahadan har 2 soniyada 6x6 katakli band-bo'shlik xaritasini oladi va 3 sinfli tiqilinch darajasini bashorat qiladi; markaziy server 2 yadroli virtual mashina, GPU yo'q. Umumiy yuklama 3000 so'rov/s ga yetishi mumkin. Siz kichik 2D CNN'ni sintetik xaritalarda o'qitasiz, ONNX'ga eksport qilasiz, ORT darajalari, paket o'lchami va oqimlar sonini o'lchaysiz hamda Little qonuni bilan nechta nusxa kerakligini p99 bo'yicha hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | CNN Conv(1->8,3x3)-ReLU-Conv(8->16,3x3)-ReLU-GAP-Linear 16-3 |
| Ma'lumot | sintetik 6x6 xarita: sinf bo'yicha bandlik ehtimoli 0,2/0,5/0,8 + fazoviy klaster; 9000 namuna |
| Epoxa | 5 |
| Batch | 1, 32, 128, 512 |
| Oqimlar | intra_op 1 va 2 |
| Talab | 3000 so'rov/s, p99 <= 3 ms (batch 1) |
| ORT darajalari | 4 ta |

Qadamlar:

1. Sintetik 6x6 trafik xaritalari generatorini yozing (3 sinf), CNN'ni o'qiting, aniqlikni yozing.
2. ONNX eksport (dinamik batch, opset 17), onnx.checker, PyTorch bilan moslik; tugunlar soni (BatchNorm ishlatgan bo'lsangiz birlashganini ko'rsating).
3. Laboratoriya s2 kabi 4 ta ORT darajasini batch 1 da o'lchang (takror 500) va tugunlar sonini sanang; PyTorch eager bilan tezlanish.
4. Batch 1/32/128/512 uchun p50/p99, throughput va little_N jadvali; 1 va 2 oqim batch 512 da taqqos.
5. Little qonuni: 3000 so'rov/s uchun batch 1 (p99 bo'yicha) va eng yaxshi batch uchun nusxalar soni; p99 <= 3 ms sharti qaysi batch'gacha bajariladi.
6. Grafik (batch - throughput, batch - p99) va xulosa: markaziy serverga nechta nusxa, GPU bo'lsa qanday o'zgaradi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 13-variant. Ekologiya: havo sifati indeksi tasnifi — INT8 kvantlash va magnitudali siyraklashtirish (pruning) taqqosi (o'rta)

Soha: Ekologiya

Shahar ekologiya xizmati 150 ta ko'cha sensor qutisiga havo sifati indeksini 5 sinfga ajratuvchi modelni joylashtirmoqda; qutilar 4G orqali yangilanadi va trafik qimmat, shuning uchun model fayli iloji boricha kichik bo'lishi kerak. Sizdan ikki yo'lni solishtirish so'raladi: INT8 kvantlash va vaznlarni magnituda bo'yicha siyraklashtirish (30/60/90 %), hamda ularning kombinatsiyasi. Siz MLP'ni sintetik sensor ma'lumotida o'qitasiz, torch.nn.utils.prune bilan siyraklashtirib qayta o'qitasiz (fine-tune), ONNX'ga eksport qilasiz va hajm (siqilgan .onnx.gz ham), aniqlik, kechikishni Pareto grafigida taqqoslaysiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 13-256-128-5 |
| Ma'lumot | make_classification(n_samples=12000, n_features=13, n_informative=9, n_classes=5, random_state=17) |
| Siyraklashtirish | L1 unstructured 30 %, 60 %, 90 % + 2 epoxa fine-tune |
| Kvantlash | PyTorch dinamik INT8, ORT dinamik INT8 |
| Hajm o'lchovi | .onnx va gzip siqilgan .onnx.gz (KB) |
| Batch | 1 va 256 |
| Cheklov | aniqlik yo'qotish <= 1 p.p. |

Qadamlar:

1. Sintetik havo sifati ma'lumotini yarating, MLP'ni 10 epoxa o'qiting, FP32 aniqligi, hajm va ORT p50 batch 1.
2. torch.nn.utils.prune.l1_unstructured bilan 30/60/90 % siyraklashtiring (prune.remove bilan niqobni qotiring), har birini 2 epoxa fine-tune qiling; nol vaznlar ulushini tekshiring.
3. Har siyrak modelni ONNX'ga eksport qiling; .onnx hajmi o'zgarmaganini ko'rsating va gzip bilan siqilgan hajmni o'lchang (siyraklik siqilishda namoyon bo'ladi).
4. ORT quantize_dynamic ni FP32 va 60 % siyrak modelga qo'llang; jami 8 konfiguratsiya uchun hajm (.onnx, .gz), aniqlik, p50 batch 1, throughput batch 256 jadvali.
5. Pareto (gz hajm - aniqlik) grafigi; 4G orqali 150 qutiga yangilash trafigini (MB) har konfiguratsiya uchun hisoblang.
6. Xulosa: siyraklashtirish CPU'da tezlik beradimi (nega yo'q), qaysi kombinatsiya 1 p.p. chegarasida eng kichik, strukturaviy siyraklashtirish qachon kerak.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); 8 konfiguratsiyali jadval (hajm .onnx/.gz, aniqlik, tezlik)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 14-variant. Sport: akselerometr bo'yicha harakat turini tanish — 1D CNN + GRU, statik kvantlash per-tensor vs per-channel (qiyin)

Soha: Sport

Fitnes bilaguzuk ishlab chiqaruvchisi 3 o'qli akselerometrning 2 soniyalik oynasi (64 qadam x 3 kanal) bo'yicha 6 ta harakat turini (yurish, yugurish, zinadan chiqish, o'tirish, turish, velosiped) aniqlaydi. Bilaguzuk protsessori faqat INT8 ni tez bajaradi va rekurrent qatlamlar kvantlashda nozik. Siz 1D CNN + GRU modelini sintetik signallarda o'qitasiz, ONNX'ga eksport qilasiz (GRU operatori), ORT statik kvantlashni per-tensor va per-channel rejimlarida, MinMax va Percentile usullarida bajarasiz, GRU tugunini kvantlashdan chiqarib ko'rasiz va 6 sinf bo'yicha chalkashlik matritsasi orqali qaysi harakatlar INT8 da ko'proq zarar ko'rishini aniqlaysiz.

| Parametr | Qiymat |
|---|---|
| Model | Conv1d(3->16,k5)-ReLU-Pool-GRU(16->32)-Linear 32-6 |
| Ma'lumot | sintetik: 64 qadam x 3 o'q, har sinf uchun chastota/amplituda/ofset naqshi + shovqin std 0,25; 6000 namuna |
| Epoxa | 10, Adam lr 2e-3 |
| Kalibrlash | MinMax va Percentile, 300 namuna |
| Rejimlar | per-tensor, per-channel, GRU FP32 da (nodes_to_exclude) |
| Batch | 1 va 64 |
| Cheklov | har sinf recall >= 0,85 |

Qadamlar:

1. Sintetik akselerometr generatorini yozing (6 sinf), modelni o'qiting, aniqlik va chalkashlik matritsasini chop eting.
2. ONNX eksport (dinamik batch, opset 17), onnx.checker; operatorlar ro'yxatida GRU borligini va ORT bilan moslikni ko'rsating; GRU eksportida yuzaga kelgan muammolarni (bo'lsa) hujjatlang.
3. quant_pre_process va ORT dinamik kvantlash; keyin statik kvantlash 4 variant: per-tensor MinMax, per-channel MinMax, per-channel Percentile, per-channel Percentile + GRU FP32 da.
4. Har variant uchun hajm, aniqlik, sinflar bo'yicha recall, p50 batch 1, throughput batch 64, Q/DQ tugunlar soni jadvali.
5. FP32 va eng yomon INT8 variant chalkashlik matritsalarini yonma-yon chizing; qaysi juft sinflar aralashib ketganini va nima uchun (o'xshash chastota) tushuntiring.
6. Pareto (hajm - eng past recall) grafigi; bilaguzukka tavsiya va «recall >= 0,85» talabi bajarilganligini tekshiring.
7. Xulosa: per-tensor va per-channel farqi qancha, GRU ni FP32 da qoldirish qancha hajm/aniqlik savdosini beradi, QAT qachon kerak.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Chalkashlik matritsalari (FP32 va INT8)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 15-variant. Media: sharhlar kayfiyati tasnifi — so'zlar sumkasi MLP va dinamik kvantlash (oson)

Soha: Media

Yangiliklar portali kuniga 2 million foydalanuvchi sharhini ijobiy/salbiy/neytral deb belgilaydi; model 1000 o'lchamli so'zlar sumkasi (bag-of-words) vektorini oladi va keng Linear qatlamlardan iborat, ya'ni dinamik kvantlash uchun ideal nomzod. Portal serverlarida GPU yo'q, xizmat 2 yadroli konteynerlarda ishlaydi. Siz sintetik siyrak matn belgilarini yaratasiz, MLP'ni o'qitasiz, PyTorch va ORT dinamik INT8 kvantlashini qo'llab hajm, aniqlik va tezlikni o'lchaysiz hamda kunlik 2 million sharhni qayta ishlash uchun nechta konteyner-soat kerakligini hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 1000-512-128-3 |
| Ma'lumot | make_classification(n_samples=20000, n_features=1000, n_informative=60, n_classes=3, random_state=23), belgilar >= 0 qilib siyraklashtirilgan (80 % nol) |
| Epoxa | 4, Adam lr 1e-3, paket 128 |
| Batch | 1, 64, 512 |
| Kunlik yuklama | 2 000 000 sharh |
| Kvantlash | PyTorch quantize_dynamic, ORT quantize_dynamic |
| Metrika | makro F1 |

Qadamlar:

1. Sintetik so'zlar sumkasi ma'lumotini yarating (manfiy qiymatlarni nolga kesib siyraklashtiring), MLP'ni o'qiting; aniqlik va makro F1.
2. Laboratoriya s3 kabi PyTorch dinamik kvantlash: tuzilma, 1-qatlam (1000x512) masshtab va qayta tiklash xatosi, hajm nisbati (4x ga qanchalik yaqin).
3. ONNX eksport + ORT quantize_dynamic; FP32 PyTorch, INT8 PyTorch, FP32 ORT, INT8 ORT — 4 konfiguratsiya uchun hajm, makro F1, p50 batch 1, throughput batch 64 va 512.
4. Har konfiguratsiya uchun 2 mln sharhni qayta ishlash vaqtini (soat) va 2 yadroli konteyner-soatlarni hisoblang; batch 512 da ishlash qancha tejaydi.
5. Grafik: hajm - throughput va Pareto (p50 - F1).
6. Xulosa: portalga qaysi konfiguratsiya, bu CPU'da INT8 tezlik bermagani sababi, VNNI serverda kutilgan tezlanish.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 16-variant. Tibbiyot: EKG aritmiya detektori — 1D CNN, uch mezonli Pareto va implant hajm chegarasi (o'rta)

Soha: Tibbiyot (kardiologiya)

Kiyiladigan kardiomonitor 1 soniyalik EKG segmentini (250 nuqta) normal/ekstrasistola/fibrillyatsiya deb tasniflaydi; monitor ichidagi mikrokontroller uchun model 50 KB dan oshmasligi, bitta segment 10 ms ichida qayta ishlanishi va xavfli sinflar uchun recall 0,95 dan kam bo'lmasligi kerak. Siz sintetik EKG segmentlarini yaratib 1D CNN'ni o'qitasiz, ONNX eksport, ORT FP32, dinamik INT8 va statik INT8 (ikki kalibrlash usuli) variantlarini tayyorlaysiz va uch mezonli (hajm, kechikish, recall) Pareto tahlilini o'tkazib, hajmni nuqta kattaligi bilan ko'rsatuvchi grafik chizasiz.

| Parametr | Qiymat |
|---|---|
| Model | Conv1d(1->8,k9)-Pool4-Conv1d(8->16,k7)-Pool4-Conv1d(16->16,k5)-GAP-Linear 16-3 |
| Ma'lumot | sintetik EKG: 250 nuqta, Gauss QRS + P/T to'lqinlar; ekstrasistola = erta keng QRS, fibrillyatsiya = tartibsiz RR + P yo'q; 6000 segment |
| Epoxa | 8 |
| Cheklovlar | hajm <= 50 KB, p99 <= 10 ms, recall (2 xavfli sinf) >= 0,95 |
| Kalibrlash | MinMax, Percentile, 256 segment |
| Batch | 1 (asosiy), 64 |
| Pareto | 3 mezon: hajm, p50, eng past recall |

Qadamlar:

1. Sintetik EKG generatorini yozing (3 sinf), modelni o'qiting, aniqlik va sinflar bo'yicha recall.
2. ONNX eksport, onnx.checker, PyTorch bilan moslik; hajm va tugunlar soni.
3. ORT 4 daraja batch 1 (p50/p99), batch 64 throughput; SLO p99 <= 10 ms tekshiruvi.
4. Dinamik va statik (MinMax, Percentile, per-channel) INT8; FP32 bilan birga 4 konfiguratsiya uchun hajm, recall (har sinf), p50, Q/DQ tugunlar jadvali.
5. Uch mezonli Pareto: laboratoriya s5 pareto funksiyasini (hajm, p50, eng past recall) uchligiga kengaytiring; grafikda x = p50, y = recall, nuqta kattaligi = hajm.
6. Hajm 50 KB dan oshsa kanallarni 8/16 dan 4/8 ga kamaytirib qayta o'qiting va jadvalga qo'shing; chegara ichidagi eng yaxshi recall qaysi.
7. Xulosa: monitorga tavsiya, aniqlik yo'qotish qaysi sinfda, kalibrlash to'plamida xavfli sinflar yetarli bo'lmasa nima bo'ladi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Uch mezonli Pareto grafigi

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 17-variant. Aviatsiya: dvigatel sensorlari bo'yicha qoldiq resurs (RUL) prognozi — ONNX, kvantlash va energiya (o'rta)

Soha: Aviatsiya

Texnik xizmat kompaniyasi dvigatelning 21 ta sensor ko'rsatkichidan qoldiq ish sikllari sonini (RUL) bashorat qiladi; model parvoz paytida bort ma'lumot yig'uvchi blokida (passiv sovutish, quvvat byudjeti 3 Vt) har soniyada ishlaydi va parvoz tugagach natija yuboriladi. Talab: RMSE 15 sikldan oshmasin, inferens energiyasi minimal bo'lsin. Siz sintetik degradatsiya ma'lumotini yaratib MLP regressorni o'qitasiz, ONNX FP32 va INT8 variantlarini tayyorlaysiz, laboratoriya s6 quvvat modelini bort bloki parametrlariga moslab uJ/inferens va 10 soatlik parvoz uchun Wh ni hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 21-128-64-1 (regressiya, Huber yo'qotish) |
| Ma'lumot | sintetik: 300 dvigatel x 200 sikl; sensorlar = asosiy qiymat + degradatsiya egri chizig'i (RUL kamaygani sari eksponensial siljish) + shovqin |
| Metrika | RMSE (sikl), MAE |
| Quvvat modeli | P_bosh 0,8 Vt, P_to'liq 3,0 Vt |
| Parvoz | 10 soat, 1 inferens/s |
| Batch | 1 va 60 (1 daqiqalik to'plam) |
| Oqimlar | 1 va 2 |

Qadamlar:

1. Sintetik degradatsiya generatorini yozing (RUL = 200 - sikl), dvigatellar bo'yicha 80/20 bo'ling (bir dvigatel ikkala to'plamga tushmasin), MLP'ni o'qiting, RMSE va MAE.
2. ONNX eksport, moslik, ORT ALL p50 batch 1; ORT quantize_dynamic va quantize_static (MinMax, 500 namuna) INT8 variantlari uchun RMSE va hajm.
3. Laboratoriya s6 energiya funksiyasini bort bloki quvvat parametrlari bilan FP32/INT8 statik x 1/2 oqim x batch 1/60 uchun ishlating; uJ_tolik, uJ_dinamik, inferens/s/Vt jadvali.
4. 10 soatlik parvoz uchun energiya (Wh) va 3 Vt byudjetdagi o'rtacha quvvat ulushini hisoblang: har soniyada bitta inferens (batch 1) va har daqiqada 60 talik to'plam (batch 60) rejimlarini solishtiring.
5. Ustunli grafik (uJ/inferens) va RMSE - energiya nuqtali grafigi; Pareto nuqtalarini belgilang.
6. Xulosa: bort blokiga qaysi rejim va format, batch 60 rejimida natija kechikishi (1 daqiqa) qabul qilinadimi, VNNI/ARM da nimani kutish mumkin.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Energiya jadvali va parvoz uchun Wh hisobi

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 18-variant. Fintech: tranzaksiya firibgarligini aniqlash — kvantlashning recall va chegaraga ta'siri, Entropy kalibrlash (qiyin)

Soha: Fintech (to'lovlar)

To'lov tizimi har soniyada 8000 tranzaksiyani real vaqtda tekshiradi; firibgarlik ulushi 1 % va model natijasi ehtimollik chegarasi orqali «bloklash/o'tkazish» qaroriga aylanadi. Infratuzilma jamoasi INT8 ga o'tishni taklif qilmoqda, lekin risk bo'limi «aniqlik o'zgarmadi» degan hisobotga ishonmaydi: muvozanatsiz ma'lumotda aniqlik 99 % bo'lib qolaveradi, recall esa tushishi mumkin. Siz MLP'ni o'qitasiz, ORT dinamik va statik (MinMax, Percentile, Entropy) kvantlashni bajarasiz, har variant uchun PR-egri, recall@precision 0,9 va optimal chegarani hisoblaysiz, hamda kvantlashdan keyin chegarani qayta kalibrlash kerakligini ko'rsatasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 28-128-64-1 (BCE, pos_weight) |
| Ma'lumot | make_classification(n_samples=60000, n_features=28, n_informative=16, weights=[0.99,0.01], flip_y=0.002, random_state=29) |
| Epoxa | 6, paket 256 |
| Kalibrlash | MinMax, Percentile, Entropy; 1000 namuna (tasodifiy) va 1000 namuna (50 % firibgarlik bilan boyitilgan) |
| Metrika | AUC-PR, recall@precision=0,9, F1 optimal chegarada |
| SLO | 8000 tranz/s, p99 <= 5 ms |
| Batch | 1 va 256 |

Qadamlar:

1. Muvozanatsiz sintetik tranzaksiyalarni yarating, MLP'ni pos_weight bilan o'qiting; test aniqlik (nega yuqori), AUC-PR, recall@precision 0,9, optimal chegara.
2. ONNX eksport, quant_pre_process, ORT dinamik kvantlash; statik kvantlash 6 variant: 3 usul x 2 kalibrlash to'plami (tasodifiy / boyitilgan).
3. Har variant uchun FP32 chegarasida recall/precision, keyin shu variant uchun qayta tanlangan optimal chegarada recall/precision; farqni jadvalga yozing.
4. PR-egri chiziqlarini bitta grafikda chizing (FP32 + 7 INT8); AUC-PR farqlari.
5. Laboratoriya s2 kabi batch 1 p99 va batch 256 throughput; Little qonuni bilan 8000 tranz/s uchun nusxalar soni.
6. Pareto (p50 - recall@precision 0,9) va xulosa: qaysi kalibrlash to'plami yaxshiroq va nega, chegarani qayta kalibrlash qancha recall qaytardi, risk bo'limiga yakuniy tavsiya.
7. Bootstrap (500 qayta tanlash) bilan FP32 va eng yaxshi INT8 recall farqining 95 % ishonch intervalini hisoblang — farq statistik ahamiyatlimi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); PR-egri grafigi va chegara jadvali

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 19-variant. Oziq-ovqat sanoati: konveyerda nuqsonli mahsulotni ajratish — ikki sinfli CNN, ORT va 500 dona/s (oson)

Soha: Oziq-ovqat sanoati

Qandolat fabrikasi konveyeridan soniyasiga 500 ta mahsulot o'tadi; har birining 8x8 kulrang issiqlik xaritasi bo'yicha «soz/nuqsonli» qarori 4 ms ichida chiqishi kerak, aks holda ajratuvchi mexanizm ulgurmaydi. Siz digits to'plamini ikki sinfga aylantirasiz (masalan 0,6,8,9 raqamlari «nuqsonli» surrogati), kengroq CNN o'qitasiz, ONNX eksport qilasiz, ORT darajalari va oqimlarini o'lchaysiz, paketlash orqali throughput'ni oshirasiz va Little qonuni bilan 500 dona/s uchun nusxalar sonini hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | CNN Conv(1->32)-BN-ReLU-Pool-Conv(32->64)-BN-ReLU-Pool-FC 256-64-2 |
| Ma'lumot | sklearn digits -> 2 sinf (0,6,8,9 = nuqsonli), stratifikatsiya, random_state=31 |
| Epoxa | 4 |
| SLO | p99 <= 4 ms (batch 1), 500 dona/s |
| Batch | 1, 4, 16, 64 |
| Oqimlar | 1 va 2 |
| Metrika | aniqlik, nuqsonli sinf recall |

Qadamlar:

1. Digits'ni ikki sinfga aylantiring, CNN'ni o'qiting; aniqlik va nuqsonli sinf recall'ini yozing; parametrlar soni.
2. ONNX eksport (dinamik batch, opset 17), onnx.checker, BatchNorm birlashganini tugunlar ro'yxatida ko'rsating; PyTorch bilan moslik.
3. Laboratoriya s2 kabi ORT 4 ta darajasini batch 1 da (takror 500) o'lchang, tugunlar soni; PyTorch eager bilan tezlanish.
4. Batch 1/4/16/64 uchun p50/p99 va throughput; 1 va 2 oqim batch 64 da.
5. Little qonuni: 500 dona/s uchun batch 1 da nechta nusxa; batch 4 da paket to'lish kutishi (4/500 s) qo'shilganda p99 SLO ga sig'adimi — hisoblang.
6. Grafik va xulosa: konveyer kompyuteriga konfiguratsiya tavsiyasi; kanallar 32/64 bo'lgani uchun model lab CNN'idan necha marta sekin.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 20-variant. Konchilik: NIR spektr bo'yicha ruda tarkibi regressiyasi — to'liq optimallashtirish zanjiri va ishonch intervallari (qiyin)

Soha: Konchilik

Boyitish fabrikasida konveyer ustidagi spektrometr har 0,5 soniyada 200 kanalli NIR spektr beradi, model ruda tarkibidagi foydali komponent foizini bashorat qiladi va natija flotatsiya reagentini boshqaradi. Model fabrikadagi sanoat kompyuteriga (CPU, GPU yo'q) joylanadi; talablar: RMSE 0,8 % dan oshmasin, p99 20 ms, 6 ta spektrometr. Siz laboratoriyaning 1-6 qadamlarini to'liq takrorlaysiz: etalon va eksport, ORT darajalari va paket, PyTorch dinamik kvantlash, ORT dinamik va statik kvantlash (3 usul), yakuniy jadval va Pareto, energiya bahosi; qo'shimcha ravishda kechikish o'lchovlariga bootstrap ishonch intervallari qo'shasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 200-256-64-1 (regressiya) va 1D CNN Conv1d(1->8,k11)-Pool-Conv1d(8->16,k7)-GAP-Linear (ikki arxitektura) |
| Ma'lumot | sintetik NIR: 200 kanal, 4 ta Gauss yutilish cho'qqisi, amplitudasi tarkib foizi bilan chiziqli + bazaviy chiziq siljishi + shovqin; 8000 spektr, tarkib 2..40 % |
| Metrika | RMSE, MAE (foiz birligida) |
| SLO | p99 <= 20 ms, 12 spektr/s jami |
| Kalibrlash | MinMax, Percentile, Entropy; 400 spektr |
| Quvvat modeli | P_bosh 6 Vt, P_to'liq 20 Vt |
| Statistika | bootstrap 1000 qayta tanlash, 95 % II (p50, p99) |
| Batch | 1, 6, 64 |

Qadamlar:

1. Sintetik NIR generatorini yozing, ikkala arxitekturani o'qiting, RMSE/MAE; ikkala modelni ONNX'ga eksport qiling va moslikni tekshiring (s1).
2. Ikkala model uchun ORT 4 daraja batch 1, batch 1/6/64 throughput, 1 va 2 oqim (s2); har o'lchovni 5 marta takrorlab p50/p99 uchun bootstrap 95 % ishonch intervalini hisoblang.
3. MLP uchun PyTorch dinamik kvantlash (s3): formula tekshiruvi, hajm, RMSE, tezlik.
4. Ikkala model uchun ORT dinamik va statik kvantlash 3 usul bilan (s4): hajm, RMSE, p50, Q/DQ tugunlar; kalibrlash to'plamini faqat past tarkibli (2..10 %) spektrlardan olib qayta kvantlang — yuqori tarkibda xato qanday o'sadi.
5. Yakuniy jadval (kamida 10 konfiguratsiya) va Pareto (p50 - RMSE), grafikda ishonch intervallarini xato chiziqlari sifatida ko'rsating (s5).
6. Energiya bahosi (s6) sanoat kompyuteri quvvat parametrlari bilan: FP32/INT8, 1/2 oqim, batch 1/6; 6 spektrometr uchun yillik kVt-soat va narx (0,08 USD/kVt-soat).
7. 4-6 betlik hisobot: fabrika uchun konfiguratsiya tavsiyasi, statistik ahamiyatli farqlar qaysilar, «GPU bo'lsa nima o'zgaradi» bo'limi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Yakuniy jadval (>= 10 konfiguratsiya) ishonch intervallari bilan; 4-6 betlik hisobot (PDF)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)
