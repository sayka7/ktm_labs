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

## Qo'shimcha topshiriqlar (10 ta, jami 100 ball)

| № | Topshiriq | Daraja | Ball |
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

## Mustaqil bajarish uchun variantlar

20 ta variant — `VARIANTS.md` faylida (jurnal raqami bo'yicha tanlanadi).
