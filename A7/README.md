# A7. Yakuniy kompleks loyihani shakllantirish: ma'lumot -> model -> API -> monitoring -> adolatlilik

Bu virtual laboratoriyada siz tibbiy tasvir triaji loyihasining to'liq ML mahsulot ketma-ketligini mini-loyiha sifatida boshidan oxirigacha o'tasiz: sxema tekshiruvi va bemor bo'yicha sizishsiz bo'lish, model reestri va promote darvozasi, ONNX eksport, FastAPI xizmati metrikalari va audit jurnali, PSI/KS drift ogohlantirishi, fairlearn bilan kesimlar hisoboti va yakunda avtomatik tuziladigan loyiha pasporti. Real labdan farqi: sandbox'da rentgen tasvirlari, GPU va Docker yo'q — har tasvir o'rniga undan olingan 6 ta sintetik belgi, DenseNet o'rniga kichik MLP, uvicorn o'rniga TestClient ishlatiladi; lekin muhandislik qarorlari (sizish, chegara, darvoza, drift qoidasi, ko'lam cheklash) aynan haqiqiy loyihadagidek.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, fastapi, httpx. (pandera, mlflow, prometheus_client, docker YO'Q — ularning vazifasi qadamlarda oddiy kod bilan bajariladi.)

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/A7

## Bosqichlar

### 1-qadam. Ma'lumot sxemasi validatsiyasi va bemor bo'yicha bo'lish

Sintetik «bemor» manifesti yaratiladi (har bemorda 1–3 tasvir, 6 ta tasvir belgisi, yosh, jins, topilma). Pandera uslubidagi sxema tekshiruvi toza ma'lumotda o'tadi, ataylab buzilgan partiyada esa har xatoni aniq nomlaydi. So'ng `GroupShuffleSplit` bilan BEMOR bo'yicha bo'linadi va sizish `assert` bilan tekshiriladi; taqqoslash uchun tasodifiy (tasvir bo'yicha) bo'lishda nechta bemor sizishi hisoblanadi. O'zgartirib ko'ring: `test_size`, buzilgan qiymatlar, `n_bemor`.

Fayl: `s1_1_qadam_malumot_sxemasi_validatsiyasi_va.py` — ishga tushirish: `python3 s1_1_qadam_malumot_sxemasi_validatsiyasi_va.py`

Kutilgan natija: Sxema toza ma'lumotda O'TDI, buzilgan partiyada 6 ta xato satri; bemor bo'yicha bo'lishda sizish 0, tasodifiy bo'lishda ~35% bemor sizadi; split × sex va yosh guruhi × split jadvallari.

Tajriba qilib ko'ring:

- `SXEMA` ga yangi qoida qo'shing: `finding=1` bo'lgan tasvirlarda `opacity` manfiy bo'lmasin — necha qator buziladi?
- `test_size` ni 0.2 va 0.5 qilib, yosh guruhi × split jadvalidagi 0-18 guruhi hajmini kuzating.
- Bo'lishni `patient_id` o'rniga `image_id` bo'yicha qiling va `assert` nima deyishini ko'ring.

### 2-qadam. Model o'qitish, eksperiment jurnali, model reestri va promote darvozasi

Uchta eksperiment (logreg, 2 belgili zaif baseline, HistGradientBoosting) o'qitiladi, chegara validatsiyada sezgirlik >= 0.90 bo'ladigan qilib tanlanadi. Har model /tmp/triage/registry ichiga pickle + SHA-256 xesh bilan JSON reestrga «staging» sifatida yoziladi. `promote` darvozasi SLO (AUC >= 0.75, recall >= 0.90, jins bo'yicha TPR farqi <= 0.05) tekshiradi va joriy production'dan yomon bo'lsa almashtirmaydi. O'zgartirib ko'ring: SLO qiymatlari, yangi eksperiment, xeshni buzish.

Fayl: `s2_2_qadam_model_oqitish_eksperiment_jurnal.py` — ishga tushirish: `python3 s2_2_qadam_model_oqitish_eksperiment_jurnal.py`

Kutilgan natija: Jadval: v1 logreg -> production, v2 -> rejected (auc < 0.75), v3 hgb -> staging (v1 yaxshiroq); reestr holati va butunlik tekshiruvi MOS.

Tajriba qilib ko'ring:

- `SLO['recall_min']` ni 0.95 qiling — chegara va aniqlik qanday o'zgaradi, qaysi modellar darvozadan o'tadi?
- To'rtinchi eksperiment sifatida `RandomForestClassifier` qo'shing va reestrda nechanchi bo'lib production bo'lishini tekshiring.
- Saqlangan pickle faylga bir bayt qo'shib (`open(..., 'ab').write(b'x')`) butunlik tekshiruvi BUZILGAN deyishini ko'ring.

### 3-qadam. PyTorch MLP -> ONNX eksport va onnxruntime bilan inferens tekshiruvi

Normalizatsiyasi grafga «tikilgan» kichik MLP 30 epoxa CPU'da o'qitiladi, dinamik batch bilan ONNX'ga eksport qilinadi, `onnx.checker` va PyTorch–ONNX parity (maks. farq < 1e-4) tekshiriladi, ikkala artefakt xeshi chiqariladi. Keyin PyTorch eager va ONNX Runtime kechikishi batch 1 va 64 uchun p50/p99 bilan taqqoslanadi. O'zgartirib ko'ring: qatlam o'lchamlari, `opset_version`, batch hajmi.

Fayl: `s3_3_qadam_pytorch_mlp_onnx_eksport_va_onnx.py` — ishga tushirish: `python3 s3_3_qadam_pytorch_mlp_onnx_eksport_va_onnx.py`

Kutilgan natija: Test AUC ~0.84; parity farqi ~1e-7 (MOS); ONNX Runtime batch 1 p50 ~10 µs, PyTorch ~40 µs; triage.onnx ~4 KB.

Tajriba qilib ko'ring:

- MLP'ni 6-128-128-1 qiling: AUC o'zgaradimi, ONNX fayl hajmi va p50 qanchaga oshadi?
- `sess_opts = ort.SessionOptions(); sess_opts.intra_op_num_threads = 1` bilan sessiya yarating va 2 oqim bilan solishtiring.
- Eksportda `dynamic_axes` ni olib tashlang va batch 64 bilan `sess.run` chaqirganda qanday xato chiqishini ko'ring.

### 4-qadam. FastAPI xizmati: /predict, /healthz, Prometheus metrikalari va audit jurnali

Xizmat kodi haqiqiy loyihadagidek yoziladi: API kalit, pydantic kontrakt, kirish nazorati, ONNX Runtime inferens, Prometheus matn formatidagi Counter/Histogram (o'z amalga oshirishimiz), audit jurnali (kirishning xeshi, jins/yosh sarlavhadan). TestClient bilan 300 so'rov yuboriladi — to'liq va server ichidagi kechikish p50/p99 taqqoslanadi, /metrics chiqishi va jurnal o'qiladi. O'zgartirib ko'ring: chegara, gistogramma chelaklari, noto'g'ri so'rov turlari.

Fayl: `s4_4_qadam_fastapi_xizmati_predict_healthz_.py` — ishga tushirish: `python3 s4_4_qadam_fastapi_xizmati_predict_healthz_.py`

Kutilgan natija: /healthz ok; 401/400/422 holatlar; 300 so'rov p50 ~4 ms (TestClient) vs ~0.1 ms (server ichida); /metrics da counter va 2 histogram; jurnalda 300 yozuv.

Tajriba qilib ko'ring:

- `/predict` ga `X-Patient-Age` < 19 bo'lsa 422 va «ko'lamdan tashqari» xabari qaytaradigan qoida qo'shing; metrikada `status="out_of_scope"` ko'rinsin.
- `/metrics` dan p99 ni gistogramma chelaklaridan (Prometheus `histogram_quantile` mantig'i bilan) qo'lda hisoblang va `LAT.vals` dagi haqiqiy p99 bilan solishtiring.
- Jurnalga `patient_id` ham yoza boshlang — bu shaxsiy ma'lumotni minimallashtirish tamoyilini qanday buzadi? Muqobil: tuzli xesh.

### 5-qadam. Monitoring: PSI/KS bilan drift aniqlash va ogohlantirish mantiqi

Validatsiya to'plami mos yozuv bo'ladi. 8 kunlik ishlab chiqarish oqimi simulyatsiya qilinadi: 4-kundan «yangi apparat» (opaklik kontrasti −30% va siljish, yorqinlik oshishi) kuchayib boradi. Har kun uchun belgilar va model ballari PSI, KS-test p-qiymati, HIGH ulushi hisoblanib, darajali qoida (OK/WARN/ALERT) ishlaydi; sezgirlik yorliq bilan qayta hisoblanib, driftning haqiqiy zarari ko'rsatiladi. O'zgartirib ko'ring: siljish kuchi, qoidalar chegarasi, chelaklar soni.

Fayl: `s5_5_qadam_monitoring_psi_ks_bilan_drift_an.py` — ishga tushirish: `python3 s5_5_qadam_monitoring_psi_ks_bilan_drift_an.py`

Kutilgan natija: 1–4 kun OK, 5-kundan ALERT (intensity/opacity PSI > 0.3), 8-kunda ball PSI ~0.23, KS p ~1e-14, sezgirlik 0.86 -> 0.68; kun × belgi PSI jadvali va grafik.

Tajriba qilib ko'ring:

- Siljishni faqat `edge_density` ga bering (model unga kam bog'liq) — belgi PSI ALERT beradi, lekin sezgirlik tushmaydi: bu «yolg'on ogohlantirish»mi? Qoidani qanday to'g'rilaysiz?
- Mos yozuvni har 3 kunda yangilab turadigan «sirpanuvchi» rejim qiling va sekin driftni u sezmay qolishini ko'rsating.
- `psi` da `bins=20` va `bins=5` bilan 8-kun qiymatlarini solishtiring; oyna hajmi `n=60` bo'lsa PSI qanchalik shovqinli bo'ladi?

### 6-qadam. Adolatlilik: fairlearn MetricFrame hisoboti va yumshatish

Test to'plamida jins va yosh guruhlari bo'yicha n, TPR, FPR, tanlov ulushi va AUC `MetricFrame` bilan chiqariladi, TPR farqi SLO (<= 0.05) bilan solishtiriladi, bootstrap bilan 95% ishonch oralig'i hisoblanadi. Ikki yumshatish yo'li taqqoslanadi: (A) `ThresholdOptimizer` guruhga xos chegaralar, (B) ko'lamni 19+ bilan cheklash va chegarani qayta tanlash. O'zgartirib ko'ring: cheklov turi, chegara, bolalar shovqin koeffitsienti generator ichida.

Fayl: `s6_6_qadam_adolatlilik_fairlearn_metricfram.py` — ishga tushirish: `python3 s6_6_qadam_adolatlilik_fairlearn_metricfram.py`

Kutilgan natija: Jins bo'yicha farq ~0.01 (OK), yosh bo'yicha ~0.18 (BUZILDI, 0-18 TPR ~0.70, keng ishonch oralig'i); A: farq ~0.02, lekin sezgirlik 0.77; B: farq ~0.03, sezgirlik 0.90.

Tajriba qilib ko'ring:

- `constraints="equalized_odds"` va `"demographic_parity"` bilan A yo'lini qayta bajaring — sezgirlik va aniqlikka ta'sirini jadvalga yozing.
- Generatorda bolalar shovqinini 2.2 dan 1.0 ga tushiring — yosh bo'yicha farq SLO'ga sig'adimi? Demak muammo modeldami yoki ma'lumotdami?
- Bootstrap'ni tasvir emas, BEMOR bo'yicha qiling (bemor indekslarini qayta tanlab) va ishonch oralig'i kengayishini kuzating.

### 7-qadam. Loyiha pasporti va model kartasi avtomatik generatsiyasi, rubrika bo'yicha o'z-o'zini baholash

Butun ketma-ketlik qisqa shaklda qayta bajariladi va barcha o'lchovlar (sizish, AUC, sezgirlik, chegara, parity, p50/p99, PSI, TPR farqlari, ko'lamdan tashqari guruh) bitta `O` lug'atiga yig'iladi. Shundan Markdown pasport + model kartasi (Mitchell va boshq. formati) va 10 mezonli rubrika jadvali avtomatik tuziladi: avtomatik tekshiriladigan mezonlar SLO bo'yicha to'liq/yarim ball oladi, qolganlari «qo'lda» belgilanadi. O'zgartirib ko'ring: SLO qiymatlari, model o'lchami, drift kuchi — hujjat o'zi qayta yoziladi.

Fayl: `s7_7_qadam_loyiha_pasporti_va_model_kartasi.py` — ishga tushirish: `python3 s7_7_qadam_loyiha_pasporti_va_model_kartasi.py`

Kutilgan natija: Markdown hujjat stdout'da va /tmp/triage/docs/model_card.md da; test AUC ~0.85, sezgirlik ~0.91, TPR farqlari SLO ichida, drift ALERT ishlaydi; jami ~84/100 (qo'lda mezonlar yarim ball).

Tajriba qilib ko'ring:

- `SLO['tpr_gap']` ni 0.03 qiling — qaysi mezon yarim ballga tushadi va hujjatda qanday aks etadi?
- Rubrikaga 11-mezon «Takrorlanuvchanlik» qo'shing: kodni ikki marta bajarib `data_sha` va `test_auc` bir xil chiqsa to'liq ball.
- Model kartasiga «Kalibrlash» bo'limi qo'shing: 10 chelakli reliability jadvali (o'rtacha ball vs haqiqiy ulush) va Brier ko'rsatkichi.

## Qo'shimcha topshiriqlar (10 ta, jami 100 ball)

| № | Topshiriq | Daraja | Ball |
|---|---|---|---|
| 1 | **Sxemaga 3 yangi qoida** — 1-qadam `SXEMA` ga kamida 3 ta yangi mantiqiy qoida qo'shing (masalan: bir bemorda jins o'zgarmasligi, bir bemorda yosh farqi <= 1, `edge_density` va `texture` orasidagi korrelyatsiya chegarasi). Toza va buzilgan partiyada tekshiruv natijalarini jadval qilib (qoida, buzilgan qatorlar soni) hisobotda keltiring. Virtual laboratoriyada bajariladi. | oson | 5 |
| 2 | **Chegara–sezgirlik egri chizig'i** — 2-qadam production modelida chegarani 0.05 dan 0.95 gacha 0.05 qadam bilan o'zgartirib, validatsiyada sezgirlik, aniqlik va HIGH ulushini jadval va grafikda ko'rsating. Sezgirlik 0.85, 0.90, 0.95 uchun tanlangan chegaralarni va har birida radiolog ko'radigan «yolg'on HIGH» sonini (1000 tasvirga) hisoblang. Virtual laboratoriyada bajariladi. | oson | 5 |
| 3 | **API kontrakt testlari** — 4-qadam xizmati uchun kamida 8 ta kontrakt testi yozing (to'g'ri so'rov, kalitsiz, noto'g'ri kalit, 5 va 7 belgi, NaN, diapazondan tashqari, yosh < 19 ko'lamdan tashqari). Har test kutilgan HTTP kodi va javob maydonlarini tekshirsin; natijani «test nomi — kutilgan — olingan — PASS/FAIL» jadvali qilib bering va /metrics da holatlar hisoblagichi testlardan keyin qanday ko'rinishini qo'shing. Virtual laboratoriyada bajariladi. | oson | 5 |
| 4 | **Reestr rollback va versiyalar taqqoslash** — 2-qadam reestriga `rollback(version)` funksiyasini qo'shing: joriy production'ni «archived» qilib, ko'rsatilgan eski versiyani qaytaradi va sababini jurnalga yozadi. So'ng yangi, ataylab yomonroq model (masalan, `C=0.001`) yarating, uni promote qilishga urinib, darvoza rad etishini, keyin qo'lda majburlab production qilib, rollback bilan qaytarishni ko'rsating. Hisobot: reestr JSON'ning har bosqichdagi holati, versiyalar metrika jadvali, xeshlar. Virtual laboratoriyada bajariladi. | o'rta | 10 |
| 5 | **ONNX optimallashtirish va aniqlik variantlari** — 3-qadam modelini FP32 ONNX bilan birga onnxruntime grafik optimallashtirishi (`GraphOptimizationLevel.ORT_ENABLE_ALL`) va dinamik INT8 kvantlash (`onnxruntime.quantization.quantize_dynamic`) variantlarida tayyorlang. Jadval: variant, fayl hajmi, p50/p99 batch 1 va 64, test AUC, PyTorch bilan maks. farq. Qaysi variant SLO'ni buzmay eng tez ekanini asoslang. Virtual laboratoriyada bajariladi. | o'rta | 10 |
| 6 | **Drift qoidalarini sozlash: yolg'on va o'tkazib yuborilgan ogohlantirishlar** — 5-qadam simulyatsiyasini 30 kunga cho'zing va uch xil ssenariy kiriting: (a) sekin chiziqli drift, (b) bir kunlik keskin sakrash va qaytish, (c) faqat model ta'sir qilmaydigan belgida siljish. Uch qoida to'plami (faqat PSI ball; PSI ball + belgi; PSI + KS + HIGH ulushi) uchun yolg'on ogohlantirish va o'tkazib yuborish (sezgirlik < 0.85 bo'lgan kun ALERT bo'lmagan) sonini hisoblang. Hisobot: ssenariy × qoida jadvali va tavsiya etilgan qoida. Virtual laboratoriyada bajariladi. | o'rta | 10 |
| 7 | **Bemor bo'yicha bootstrap va kesimlar ishonch oralig'i** — 6-qadamdagi bootstrap'ni bemor (klaster) bo'yicha qayta yozing va jins hamda yosh guruhlari uchun TPR, FPR va AUC ning 95% ishonch oraliqlarini hisoblang. TPR farqi uchun ham ishonch oralig'i tuzing va «farq SLO'dan katta» xulosasi statistik jihatdan ishonchli yoki yo'qligini har kesim uchun ayting. Hisobot: jadval + qisqa talqin (qaysi guruhlarda namunalar kamligi xulosani noaniq qiladi). Virtual laboratoriyada bajariladi. | o'rta | 10 |
| 8 | **Monitoring paneli: metrikalardan SLO hisoboti** — 4-qadam xizmatiga 2000 so'rov yuboring (ularning 20% ida 5-qadamdagi «yangi apparat» siljishi bo'lsin), so'ng /metrics matnini va audit jurnalini o'qib, Grafana panelining matnli muqobilini tuzing: p50/p95/p99 (gistogramma chelaklaridan interpolyatsiya bilan), so'rovlar holat bo'yicha, HIGH ulushi 100 so'rovlik oynalarda, ball PSI oynalar bo'yicha, jins bo'yicha HIGH ulushi. Har ko'rsatkich yoniga SLO va holat (OK/BUZILDI) qo'ying, kamida bitta ogohlantirish qoidasi ishlashini ko'rsating. Virtual laboratoriyada bajariladi. | qiyin | 15 |
| 9 | **Qayta o'qitish quvuri: drift -> retrain -> darvoza -> promote** — 5-qadam ALERT bergandan keyin ishga tushadigan avtomatik quvur yozing: siljigan oqimning «yorliqlangan» qismini train'ga qo'shib qayta o'qitish, 2-qadam reestriga yangi versiya sifatida yozish, SLO darvozasi (shu jumladan 6-qadamdagi TPR farqi) va eski vs yangi modelni siljigan va siljimagan ma'lumotda taqqoslash (AUC, sezgirlik, PSI). Yangi model faqat ikkala to'plamda ham yomonlashmasa promote bo'lsin. Hisobot: quvur bosqichlari jurnali, taqqoslash jadvali va rollback sharti. Virtual laboratoriyada bajariladi. | qiyin | 15 |
| 10 | **O'z mavzungiz uchun to'liq mini-loyiha va pasport** — Transport (bekatda kutish vaqti) yoki sanoat monitoringi (sensorlar bo'yicha nosozlik) mavzusini tanlab, 1–7-qadamlar tuzilmasini sintetik ma'lumot bilan takrorlang: o'z sxemangiz va guruh identifikatoringiz (bekat/jihoz), kamida 3 eksperimentli reestr, ONNX + FastAPI xizmati, drift simulyatsiyasi, kamida bitta sezgir atribut bo'yicha adolatlilik tahlili va 7-qadamdagi kabi avtomatik pasport + rubrika. Hisobotda model kartasi, SLO jadvali, ADR ko'rinishida kamida 2 ta qaror (masalan, chegara tanlovi, ko'lam cheklash) bo'lsin. Virtual laboratoriyada bajariladi. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.
