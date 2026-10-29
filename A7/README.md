# A7. Yakuniy kompleks loyihani shakllantirish: ma'lumot -> model -> API -> monitoring -> adolatlilik

📄 Bosma qo'llanma (bosqichlar, kod, mashqlar, 20 variant): `A7_laboratoriya_qollanma.pdf`
🦉 Bosqichlar Suqrot usulida: har `sN_*.py` — karkas (TODO larni o'zingiz yozasiz), savollar shu faylda, ishoralar `ISHORALAR.md` da.
Bu virtual laboratoriyada siz tibbiy tasvir triaji loyihasining to'liq ML mahsulot ketma-ketligini mini-loyiha sifatida boshidan oxirigacha o'tasiz: sxema tekshiruvi va bemor bo'yicha sizishsiz bo'lish, model reestri va promote darvozasi, ONNX eksport, FastAPI xizmati metrikalari va audit jurnali, PSI/KS drift ogohlantirishi, fairlearn bilan kesimlar hisoboti va yakunda avtomatik tuziladigan loyiha pasporti. Real labdan farqi: sandbox'da rentgen tasvirlari, GPU va Docker yo'q — har tasvir o'rniga undan olingan 6 ta sintetik belgi, DenseNet o'rniga kichik MLP, uvicorn o'rniga TestClient ishlatiladi; lekin muhandislik qarorlari (sizish, chegara, darvoza, drift qoidasi, ko'lam cheklash) aynan haqiqiy loyihadagidek.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, fastapi, httpx. (pandera, mlflow, prometheus_client, docker YO'Q — ularning vazifasi qadamlarda oddiy kod bilan bajariladi.)

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/A7

## Bosqichlar

### 1-qadam. Ma'lumot sxemasi validatsiyasi va bemor bo'yicha bo'lish

Sintetik «bemor» manifesti yaratiladi (har bemorda 1–3 tasvir, 6 ta tasvir belgisi, yosh, jins, topilma). Pandera uslubidagi sxema tekshiruvi toza ma'lumotda o'tadi, ataylab buzilgan partiyada esa har xatoni aniq nomlaydi. So'ng `GroupShuffleSplit` bilan BEMOR bo'yicha bo'linadi va sizish `assert` bilan tekshiriladi; taqqoslash uchun tasodifiy (tasvir bo'yicha) bo'lishda nechta bemor sizishi hisoblanadi. O'zgartirib ko'ring: `test_size`, buzilgan qiymatlar, `n_bemor`.

Fayl: `s1_1_qadam_malumot_sxemasi_validatsiyasi_va.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s1_1_qadam_malumot_sxemasi_validatsiyasi_va.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Bitta bemorning uchta tasviridan biri train'ga, ikkinchisi testga tushsa, test AUC nimani o'lchay boshlaydi — umumlashtirishnimi yoki bemorni «tanib olish»nimi?
2. Sxema tekshiruvchisi har bir ustun uchun qaysi savollarga javob berishi kerak, va nega xato matnida ustun nomi va buzilgan qiymatlar soni bo'lishi muhim?
3. Buzilgan partiyada aynan 6 ta xato satri chiqishi uchun qaysi ustunlarni qanday buzish kerak? 'm' (kichik harf) qaysi qoidaga tushadi, 412 va -1 yosh-chi?
4. GroupShuffleSplit oddiy train_test_split'dan qaysi argumenti bilan farq qiladi va u qaytargan indekslarni df'ga qanday yozasiz?
5. Sizish yo'qligini qanday isbotlaysiz: qaysi ikki to'plam kesishmasi bo'sh bo'lishi shart?
6. Tasvir bo'yicha tasodifiy 70/30 bo'lishda bemorlarning taxminan qancha qismi siziydi deb kutasiz (expect ~35%)? Nega 1 tasvirli bemor hech qachon sizmaydi?
7. Vakillik jadvalida 0-18 yosh guruhi har split'da yetarli bormi? Bu keyingi adolatlilik tahlili uchun nima uchun muhim?

Kutilgan natija: Sxema toza ma'lumotda O'TDI, buzilgan partiyada 6 ta xato satri; bemor bo'yicha bo'lishda sizish 0, tasodifiy bo'lishda ~35% bemor sizadi; split × sex va yosh guruhi × split jadvallari.

Tajriba qilib ko'ring:

- `SXEMA` ga yangi qoida qo'shing: `finding=1` bo'lgan tasvirlarda `opacity` manfiy bo'lmasin — necha qator buziladi?
- `test_size` ni 0.2 va 0.5 qilib, yosh guruhi × split jadvalidagi 0-18 guruhi hajmini kuzating.
- Bo'lishni `patient_id` o'rniga `image_id` bo'yicha qiling va `assert` nima deyishini ko'ring.

### 2-qadam. Model o'qitish, eksperiment jurnali, model reestri va promote darvozasi

Uchta eksperiment (logreg, 2 belgili zaif baseline, HistGradientBoosting) o'qitiladi, chegara validatsiyada sezgirlik >= 0.90 bo'ladigan qilib tanlanadi. Har model /tmp/triage/registry ichiga pickle + SHA-256 xesh bilan JSON reestrga «staging» sifatida yoziladi. `promote` darvozasi SLO (AUC >= 0.75, recall >= 0.90, jins bo'yicha TPR farqi <= 0.05) tekshiradi va joriy production'dan yomon bo'lsa almashtirmaydi. O'zgartirib ko'ring: SLO qiymatlari, yangi eksperiment, xeshni buzish.

Fayl: `s2_2_qadam_model_oqitish_eksperiment_jurnal.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s2_2_qadam_model_oqitish_eksperiment_jurnal.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Triaj tizimida qaysi xato qimmatroq: kasal bemorni «ROUTINE» deb o'tkazib yuborishmi yoki sog'lomni «HIGH» qilishmi? Bu sizning chegara tanlash qoidangizga qanday ta'sir qiladi?
2. Chegarani nega testda emas, validatsiyada tanlash kerak? Agar testda tanlasangiz test metrikasi nimani ko'rsatmay qoladi?
3. Jins bo'yicha TPR farqini hisoblash uchun y va pred'ni qanday ikki qismga ajratasiz va qaysi metrikani har qismda olasiz?
4. Reestrga model faylining o'zi emas, uning SHA-256 xeshi ham nega yoziladi? Kimdir .pkl faylni almashtirsa, buni qanday aniqlaysiz?
5. promote darvozasi qaysi tartibda qaror qilishi kerak: avval SLO, keyin joriy production bilan solishtirishmi yoki aksincha? v3 (hgb) nega production bo'lmasligi kutilmoqda?
6. Kutilgan natija: v1 production, v2 rejected (auc < 0.75), v3 staging. Sizda v2 boshqa sabab bilan rad etilsa, bu nimani anglatadi?

Kutilgan natija: Jadval: v1 logreg -> production, v2 -> rejected (auc < 0.75), v3 hgb -> staging (v1 yaxshiroq); reestr holati va butunlik tekshiruvi MOS.

Tajriba qilib ko'ring:

- `SLO['recall_min']` ni 0.95 qiling — chegara va aniqlik qanday o'zgaradi, qaysi modellar darvozadan o'tadi?
- To'rtinchi eksperiment sifatida `RandomForestClassifier` qo'shing va reestrda nechanchi bo'lib production bo'lishini tekshiring.
- Saqlangan pickle faylga bir bayt qo'shib (`open(..., 'ab').write(b'x')`) butunlik tekshiruvi BUZILGAN deyishini ko'ring.

### 3-qadam. PyTorch MLP -> ONNX eksport va onnxruntime bilan inferens tekshiruvi

Normalizatsiyasi grafga «tikilgan» kichik MLP 30 epoxa CPU'da o'qitiladi, dinamik batch bilan ONNX'ga eksport qilinadi, `onnx.checker` va PyTorch–ONNX parity (maks. farq < 1e-4) tekshiriladi, ikkala artefakt xeshi chiqariladi. Keyin PyTorch eager va ONNX Runtime kechikishi batch 1 va 64 uchun p50/p99 bilan taqqoslanadi. O'zgartirib ko'ring: qatlam o'lchamlari, `opset_version`, batch hajmi.

Fayl: `s3_3_qadam_pytorch_mlp_onnx_eksport_va_onnx.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s3_3_qadam_pytorch_mlp_onnx_eksport_va_onnx.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Normalizatsiya (mu, sd) model ichida emas, serverda alohida qilinsa, ONNX artefaktini boshqa jamoaga berganda qanday xavf paydo bo'ladi?
2. Model chiqishi nega logit (sigmoid'siz) va loss nega BCEWithLogitsLoss? Sinflar nomutanosib bo'lsa pos_weight'ni qanday tanlaysiz?
3. Eksportda dummy kirish shakli (1, 6). Agar dynamic_axes bermasangiz, server 64 tali batch yuborganda nima bo'ladi?
4. Parity tekshiruvi uchun qaysi ikki massivni solishtirasiz va nega chegara 1e-4 — nega aynan 0 emas?
5. Kechikishni o'lchashda birinchi chaqiruvlarni nega tashlab yuborish kerak va nega o'rtacha emas, p50/p99 olinadi?
6. Kutilgan natija: ONNX Runtime batch 1 da ~10 µs, PyTorch eager ~40 µs. Batch 64 da farq kamayadimi yoki ortadimi va nega?

Kutilgan natija: Test AUC ~0.84; parity farqi ~1e-7 (MOS); ONNX Runtime batch 1 p50 ~10 µs, PyTorch ~40 µs; triage.onnx ~4 KB.

Tajriba qilib ko'ring:

- MLP'ni 6-128-128-1 qiling: AUC o'zgaradimi, ONNX fayl hajmi va p50 qanchaga oshadi?
- `sess_opts = ort.SessionOptions(); sess_opts.intra_op_num_threads = 1` bilan sessiya yarating va 2 oqim bilan solishtiring.
- Eksportda `dynamic_axes` ni olib tashlang va batch 64 bilan `sess.run` chaqirganda qanday xato chiqishini ko'ring.

### 4-qadam. FastAPI xizmati: /predict, /healthz, Prometheus metrikalari va audit jurnali

Xizmat kodi haqiqiy loyihadagidek yoziladi: API kalit, pydantic kontrakt, kirish nazorati, ONNX Runtime inferens, Prometheus matn formatidagi Counter/Histogram (o'z amalga oshirishimiz), audit jurnali (kirishning xeshi, jins/yosh sarlavhadan). TestClient bilan 300 so'rov yuboriladi — to'liq va server ichidagi kechikish p50/p99 taqqoslanadi, /metrics chiqishi va jurnal o'qiladi. O'zgartirib ko'ring: chegara, gistogramma chelaklari, noto'g'ri so'rov turlari.

Fayl: `s4_4_qadam_fastapi_xizmati_predict_healthz_.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s4_4_qadam_fastapi_xizmati_predict_healthz_.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Xizmat qaysi uch holatda bashorat qaytarmasligi kerak (API kalit, kirish shakli, qiymat diapazoni) va har biriga qaysi HTTP kodi mos?
2. Pydantic'da ro'yxat uzunligini aniq 6 ga qanday cheklaysiz, va bu tekshiruvni o'zingiz qo'lda yozishdan nimasi yaxshi?
3. Prometheus gistogrammasidagi chelaklar (bucket) kumulyativmi? x=0.0003 kuzatilsa, qaysi le chelaklar oshishi kerak?
4. Counter.expose() chiqishi 'triage_requests_total{status="ok"} 300' ko'rinishida bo'lishi uchun yorliqlarni qanday saqlab, qanday formatlaysiz?
5. Audit jurnaliga nega bemor belgilari emas, ularning xeshi yoziladi? Jins/yosh sarlavhadan olinadi — bu keyin qaysi tahlilga kerak bo'ladi?
6. Kutilgan natija: TestClient bilan p50 ~4 ms, server ichida ~0.1 ms. Bu 40 baravar farq nimadan iborat va qaysi biri foydalanuvchi SLO'siga kiradi?
7. Yuk sinovidan oldin nega jurnal fayli va metrikalarni tozalash kerak? Aks holda check satridagi 300 soni nimaga aylanadi?

Kutilgan natija: /healthz ok; 401/400/422 holatlar; 300 so'rov p50 ~4 ms (TestClient) vs ~0.1 ms (server ichida); /metrics da counter va 2 histogram; jurnalda 300 yozuv.

Tajriba qilib ko'ring:

- `/predict` ga `X-Patient-Age` < 19 bo'lsa 422 va «ko'lamdan tashqari» xabari qaytaradigan qoida qo'shing; metrikada `status="out_of_scope"` ko'rinsin.
- `/metrics` dan p99 ni gistogramma chelaklaridan (Prometheus `histogram_quantile` mantig'i bilan) qo'lda hisoblang va `LAT.vals` dagi haqiqiy p99 bilan solishtiring.
- Jurnalga `patient_id` ham yoza boshlang — bu shaxsiy ma'lumotni minimallashtirish tamoyilini qanday buzadi? Muqobil: tuzli xesh.

### 5-qadam. Monitoring: PSI/KS bilan drift aniqlash va ogohlantirish mantiqi

Validatsiya to'plami mos yozuv bo'ladi. 8 kunlik ishlab chiqarish oqimi simulyatsiya qilinadi: 4-kundan «yangi apparat» (opaklik kontrasti −30% va siljish, yorqinlik oshishi) kuchayib boradi. Har kun uchun belgilar va model ballari PSI, KS-test p-qiymati, HIGH ulushi hisoblanib, darajali qoida (OK/WARN/ALERT) ishlaydi; sezgirlik yorliq bilan qayta hisoblanib, driftning haqiqiy zarari ko'rsatiladi. O'zgartirib ko'ring: siljish kuchi, qoidalar chegarasi, chelaklar soni.

Fayl: `s5_5_qadam_monitoring_psi_ks_bilan_drift_an.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s5_5_qadam_monitoring_psi_ks_bilan_drift_an.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Ishlab chiqarishda haqiqiy yorliq (finding) haftalar kechikib keladi. Unda yorliqsiz, faqat kirish va ball taqsimotlariga qarab nimani kuzatish mumkin?
2. PSI formulasida chelaklar chegaralari nega joriy oqimdan emas, mos yozuv kvantillaridan olinadi? Chetki chelaklarni nega ±inf qilish kerak?
3. PSI va KS-test bir xil narsani o'lchaydimi? Nega n=250 da KS p-qiymati juda kichik (~1e-14) bo'lishi mumkin, PSI esa 0.23 atrofida?
4. Holatni OK/WARN/ALERT ga ajratishda qoidalarni qanday tartibda tekshirasiz, va KS yoki HIGH ulushi signali holatni qanday «bir pog'ona» oshirishi kerak?
5. Qaysi belgilar siljishini kutasiz (yangi_apparat funksiyasiga qarang) va qaysi kundan ALERT boshlanishi kerak?
6. Sezgirlik 0.86 dan 0.68 ga tushsa, bu SLO (0.90) bilan qanday bog'liq? PSI ogohlantirishi yorliqdan qancha oldin kelgan bo'lardi?

Kutilgan natija: 1–4 kun OK, 5-kundan ALERT (intensity/opacity PSI > 0.3), 8-kunda ball PSI ~0.23, KS p ~1e-14, sezgirlik 0.86 -> 0.68; kun × belgi PSI jadvali va grafik.

Tajriba qilib ko'ring:

- Siljishni faqat `edge_density` ga bering (model unga kam bog'liq) — belgi PSI ALERT beradi, lekin sezgirlik tushmaydi: bu «yolg'on ogohlantirish»mi? Qoidani qanday to'g'rilaysiz?
- Mos yozuvni har 3 kunda yangilab turadigan «sirpanuvchi» rejim qiling va sekin driftni u sezmay qolishini ko'rsating.
- `psi` da `bins=20` va `bins=5` bilan 8-kun qiymatlarini solishtiring; oyna hajmi `n=60` bo'lsa PSI qanchalik shovqinli bo'ladi?

### 6-qadam. Adolatlilik: fairlearn MetricFrame hisoboti va yumshatish

Test to'plamida jins va yosh guruhlari bo'yicha n, TPR, FPR, tanlov ulushi va AUC `MetricFrame` bilan chiqariladi, TPR farqi SLO (<= 0.05) bilan solishtiriladi, bootstrap bilan 95% ishonch oralig'i hisoblanadi. Ikki yumshatish yo'li taqqoslanadi: (A) `ThresholdOptimizer` guruhga xos chegaralar, (B) ko'lamni 19+ bilan cheklash va chegarani qayta tanlash. O'zgartirib ko'ring: cheklov turi, chegara, bolalar shovqin koeffitsienti generator ichida.

Fayl: `s6_6_qadam_adolatlilik_fairlearn_metricfram.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s6_6_qadam_adolatlilik_fairlearn_metricfram.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Umumiy AUC yaxshi bo'lsa ham model qaysidir guruhga zarar berishi mumkinmi? Triajda qaysi guruh metrikasi eng muhim — TPR, FPR yoki tanlov ulushi?
2. MetricFrame'ga qaysi uchta narsani berasiz va TPR farqini (max-min) qaysi metod bilan olasiz?
3. 0-18 guruhida musbat namunalar soni kam. TPR ~0.70 nuqtaviy bahosiga qanchalik ishonish mumkin va buni qanday miqdorlashtirasiz?
4. ThresholdOptimizer allaqachon o'qitilgan modelni qayta o'qitmasligi uchun qaysi parametr kerak, va cheklov sifatida nima beriladi?
5. B yo'lida 19+ bemorlarga cheklagach chegarani nega qayta tanlash kerak? Qaysi yo'nalishda qidirasiz — eng katta yoki eng kichik chegara?
6. Kutilgan natija: A farqni ~0.02 ga tushiradi, lekin sezgirlik 0.77; B farq ~0.03, sezgirlik 0.90. Qaysi biri klinik jihatdan halolroq va nega?

Kutilgan natija: Jins bo'yicha farq ~0.01 (OK), yosh bo'yicha ~0.18 (BUZILDI, 0-18 TPR ~0.70, keng ishonch oralig'i); A: farq ~0.02, lekin sezgirlik 0.77; B: farq ~0.03, sezgirlik 0.90.

Tajriba qilib ko'ring:

- `constraints="equalized_odds"` va `"demographic_parity"` bilan A yo'lini qayta bajaring — sezgirlik va aniqlikka ta'sirini jadvalga yozing.
- Generatorda bolalar shovqinini 2.2 dan 1.0 ga tushiring — yosh bo'yicha farq SLO'ga sig'adimi? Demak muammo modeldami yoki ma'lumotdami?
- Bootstrap'ni tasvir emas, BEMOR bo'yicha qiling (bemor indekslarini qayta tanlab) va ishonch oralig'i kengayishini kuzating.

### 7-qadam. Loyiha pasporti va model kartasi avtomatik generatsiyasi, rubrika bo'yicha o'z-o'zini baholash

Butun ketma-ketlik qisqa shaklda qayta bajariladi va barcha o'lchovlar (sizish, AUC, sezgirlik, chegara, parity, p50/p99, PSI, TPR farqlari, ko'lamdan tashqari guruh) bitta `O` lug'atiga yig'iladi. Shundan Markdown pasport + model kartasi (Mitchell va boshq. formati) va 10 mezonli rubrika jadvali avtomatik tuziladi: avtomatik tekshiriladigan mezonlar SLO bo'yicha to'liq/yarim ball oladi, qolganlari «qo'lda» belgilanadi. O'zgartirib ko'ring: SLO qiymatlari, model o'lchami, drift kuchi — hujjat o'zi qayta yoziladi.

Fayl: `s7_7_qadam_loyiha_pasporti_va_model_kartasi.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s7_7_qadam_loyiha_pasporti_va_model_kartasi.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Model kartasidagi raqam qo'lda yozilsa va keyin model qayta o'qitilsa nima bo'ladi? Hujjatni «koddan avtomatik» qilishning himoyadagi afzalligi nimada?
2. Oldingi 6 qadamdan qaysi o'lchovlar pasportga kirishi kerak? Ularni ma'lumot, model, unumdorlik, monitoring, adolatlilik bo'limlariga qanday ajratasiz?
3. 6-qadam qarori (ko'lam 19+) bu yerda qayerda qo'llanishi kerak — o'qitishdami, baholashdami yoki ikkalasidami? 0–18 guruhini butunlay unutish to'g'rimi?
4. Drift ogohlantirishi «ishlaydi» deb qachon aytish mumkin — faqat drift oqimida PSI yuqori bo'lsami yoki oddiy oqimda past bo'lishi ham shartmi?
5. Rubrikada avtomatik tekshirib bo'lmaydigan mezonlar (Docker, Git, himoya) qanday ball olishi kerak va nega ularni to'liq ball bilan belgilash noto'g'ri?
6. Markdown jadval satrini ro'yxatdan qanday hosil qilasiz va check satri '**Jami:' chiqishi uchun oxirgi qatorda nima bo'lishi kerak?
7. Kutilgan natija: test AUC ~0.85, sezgirlik ~0.91, jami ~84/100. SLO qiymatini (masalan recall 0.95) o'zgartirsangiz hujjatning qaysi qismlari o'zi o'zgaradi?

Kutilgan natija: Markdown hujjat stdout'da va /tmp/triage/docs/model_card.md da; test AUC ~0.85, sezgirlik ~0.91, TPR farqlari SLO ichida, drift ALERT ishlaydi; jami ~84/100 (qo'lda mezonlar yarim ball).

Tajriba qilib ko'ring:

- `SLO['tpr_gap']` ni 0.03 qiling — qaysi mezon yarim ballga tushadi va hujjatda qanday aks etadi?
- Rubrikaga 11-mezon «Takrorlanuvchanlik» qo'shing: kodni ikki marta bajarib `data_sha` va `test_auc` bir xil chiqsa to'liq ball.
- Model kartasiga «Kalibrlash» bo'limi qo'shing: 10 chelakli reliability jadvali (o'rtacha ball vs haqiqiy ulush) va Brier ko'rsatkichi.

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
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

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Kredit skoring: mikroqarz defolt xavfi (oson)

Soha: Moliya (mikromoliya)

Mikromoliya tashkiloti 12 oylik iste'mol qarzlari uchun defolt (90+ kun kechikish) ehtimolini baholaydigan xizmat yaratmoqda. Bitta mijozda 1–4 ta qarz tarixi bor, shuning uchun bo'lish mijoz bo'yicha bo'lishi shart. Model kredit menejeriga «qo'shimcha tekshiruv» bayrog'ini beradi, rad etish qarori emas; jins va yosh bo'yicha kamsitish qonunan taqiqlangan. Yangi kreditlar oqimida inflyatsiya tufayli daromad/qarz nisbati siljiydi — monitoring buni yorliq kelguncha (90 kun) sezishi kerak.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 2500 mijoz x 1–4 qarz; belgilar: oylik daromad, qarz/daromad nisbati, kredit tarixi oyi, oldingi kechikishlar soni, ish staji, qarz miqdori; yashirin omil z = to'lov qobiliyati |
| Guruhlash kaliti | customer_id (GroupShuffleSplit, test 0.3) |
| Sezgir atribut | jins (M/F, 48/52) va yosh guruhi (18-25, 26-40, 41-60, 61+) |
| Model | logreg (baseline), logreg faqat 2 belgi (zaif), HistGradientBoosting; ONNX uchun MLP 6-32-16-1 |
| SLO | AUC >= 0.78, recall(defolt) >= 0.80, TPR farqi (jins) <= 0.05, API p99 <= 20 ms |
| Drift ssenariysi | 10 kunlik oqim, 5-kundan daromad −15% va qarz/daromad nisbati +0.12 (inflyatsiya), kuch chiziqli o'sadi |
| Adolatlilik kesimi | jins va yosh guruhi; 18-25 guruhida kredit tarixi qisqa (shovqin x1.8) |
| Promote qoidasi | SLO bajarilsa va val AUC joriy production'dan kamida 0.005 yuqori bo'lsa |

Qadamlar:

1. Lab s1 kabi generator yozing (customer_id, loan_id, age, sex, 6 belgi, default). Sxemaga 8 qoida: loan_id regex ^L\d{7}$, age 18–85, daromad > 0, nisbat 0–3, default {0,1}. Buzilgan partiyada (200 qator, 6 xato turi) har xatoni nomlang. customer_id bo'yicha bo'ling, sizish assert, tasodifiy bo'lishda sizgan mijozlar ulushini hisoblang; split x jins jadvali.
2. Uch eksperimentni o'qiting, validatsiyada recall >= 0.80 beradigan eng katta chegarani tanlang. Reestrga (JSON + SHA-256) staging yozing; promote darvozasi SLO va «AUC +0.005» qoidasini tekshirsin. Jadval: versiya, chegara, valAUC, valRecall, TPR farqi, stage, sabab.
3. MLP 6-32-16-1 ni 30 epoxa o'qiting, normalizatsiyani grafga tiking, ONNX (opset 17, dinamik batch) eksport qiling; onnx.checker va parity (maks. farq < 1e-4); PyTorch vs ONNX Runtime p50/p99 batch 1 va 64.
4. FastAPI: /predict (X-API-Key, pydantic 6 belgi, diapazon nazorati), /healthz, /metrics (Counter + 2 Histogram), audit jurnalida kirish xeshi, jins va yosh guruhi. TestClient bilan 300 so'rov: to'liq va server ichidagi p50/p99; 401/400/422 holatlar.
5. Drift: validatsiya mos yozuv, 10 kun oqim, 5-kundan inflyatsiya siljishi. Har kun belgi PSI, ball PSI, KS p, bayroq ulushi, recall; OK/WARN/ALERT qoidasi (PSI 0.1/0.2, KS p < 0.01, bayroq ulushi ±25%). Birinchi ALERT kunini va eng ko'p siljigan belgini toping.
6. fairlearn MetricFrame: jins va yosh guruhi bo'yicha n, TPR, FPR, tanlov ulushi, AUC; bootstrap 95% IO. Yumshatish A: ThresholdOptimizer (true_positive_rate_parity), B: 18-25 guruhi uchun «qo'lda ko'rib chiqish» yo'li (ko'lam cheklash). Taqqoslash jadvali.
7. Barcha o'lchovlarni bitta lug'atga yig'ib Markdown model kartasi + loyiha pasporti + 10 mezonli rubrika jadvalini avtomatik tuzing; SLO bo'yicha to'liq/yarim ball.

Topshiriladi: Kod papkasi: s1_data.py … s7_passport.py (har biri mustaqil ishlaydi); registry.json, drift_log.json, predictions.jsonl; Avtomatik tuzilgan model_card.md (model kartasi + pasport + rubrika); Hisobot PDF: jadvallar (eksperiment, kechikish, PSI kun x belgi, MetricFrame), 2 grafik, xulosa (>= 200 so'z), ADR: nega B yoki A yumshatish tanlandi

Baholash: Sxema validatsiyasi va mijoz bo'yicha sizishsiz bo'lish — 10, Eksperimentlar, reestr, promote darvozasi to'g'ri ishlaydi — 15, ONNX eksport, parity va kechikish o'lchovlari — 10, FastAPI xizmati: kontrakt, metrikalar, audit — 15, Drift monitoring va ogohlantirish mantiqi — 15, Adolatlilik hisoboti va yumshatish taqqoslash — 15, Model kartasi, pasport va hisobot sifati — 20 (jami 100)

## 2-variant. Telekom: abonent ketishi (churn) bashorati (oson)

Soha: Telekommunikatsiya

Mobil operator keyingi 30 kunda raqamini boshqa operatorga ko'chirish ehtimoli yuqori abonentlarni aniqlab, ularga saqlab qolish taklifi yubormoqchi. Har abonentda 1–3 oylik hisob-kitob yozuvi bor — bo'lish abonent bo'yicha. Taklif byudjeti cheklangan: bayroq ulushi 25% dan oshmasin, lekin ketayotganlarning kamida 75% ushlansin. Tarif yangilanganda foydalanish belgilari siljiydi; shahar va qishloq abonentlari o'rtasida xizmat sifati farq qilmasligi kerak.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 3000 abonent x 1–3 oy; belgilar: oylik daqiqa, internet GB, qo'ng'iroq markaziga murojaatlar, to'lov kechikishi kuni, tarif narxi, shartnoma yoshi oyi; churn ~18% |
| Guruhlash kaliti | subscriber_id |
| Sezgir atribut | hudud turi (shahar 60% / qishloq 40%); qishloqda internet belgisi shovqinliroq (x1.6) |
| Model | logreg, HistGradientBoosting, RandomForest(200); ONNX MLP 6-24-1 |
| SLO | AUC >= 0.80, recall >= 0.75, bayroq ulushi <= 0.25, TPR farqi (hudud) <= 0.06, p99 <= 15 ms |
| Drift ssenariysi | 8 kun, 4-kundan yangi tarif: internet GB +40%, daqiqa −20%; 7-kunda qo'ng'iroq markazi murojaatlari keskin x2 (bir kunlik sakrash) |
| Adolatlilik kesimi | hudud turi va shartnoma yoshi guruhi (0-6, 7-24, 25+ oy) |
| Promote qoidasi | SLO + bayroq ulushi cheklovi; joriy production'dan val AUC past bo'lsa staging'da qoladi |

Qadamlar:

1. Generator (subscriber_id, month, region, 6 belgi, churn). Sxema: GB 0–500, daqiqa 0–20000, kechikish 0–90, tarif narxi {29000, 49000, 79000, 119000} so'm, churn {0,1}. Buzilgan partiya: 7 xato turi. subscriber_id bo'yicha bo'lish, sizish assert, split x hudud jadvali.
2. Uch eksperiment; chegara: recall >= 0.75 VA bayroq ulushi <= 0.25 ikkalasini qanoatlantiradigan chegara oralig'ini toping (yo'q bo'lsa sababini yozing). Reestr JSON + SHA-256, promote darvozasi; jadval.
3. MLP 6-24-1 -> ONNX, parity, p50/p99 batch 1/64; ONNX fayl hajmi va xeshi.
4. FastAPI /predict, /healthz, /metrics; javobda offer = RETAIN / NONE; audit jurnalida hudud va shartnoma yoshi; 300 so'rov p50/p99; 401/400/422.
5. Drift: 8 kun, tarif siljishi + 7-kun sakrashi. Belgi PSI jadvali, ball PSI, KS, bayroq ulushi; sakrash va asta siljish uchun qoida qanday farq qilishini izohlang; grafik.
6. MetricFrame hudud va shartnoma yoshi bo'yicha; bootstrap IO. Yumshatish A: ThresholdOptimizer, B: qishloq uchun internet belgisini olib tashlab qayta o'qitish. Taqqoslash.
7. Model kartasi + pasport + rubrika avtomatik generatsiya.

Topshiriladi: Kod papkasi (7 bosqich); registry.json, drift_log.json, predictions.jsonl, model.onnx; model_card.md avtomatik; Hisobot PDF: jadvallar, PSI grafigi, byudjet–recall kelishuvi tahlili, xulosa (>= 200 so'z)

Baholash: Sxema va abonent bo'yicha bo'lish — 10, Eksperimentlar, ikki shartli chegara, reestr va promote — 15, ONNX eksport va kechikish — 10, FastAPI xizmati va metrikalar — 15, Drift: siljish va sakrashni farqlash — 15, Adolatlilik va yumshatish — 15, Model kartasi va hisobot — 20 (jami 100)

## 3-variant. Ta'lim: talabaning o'qishni tashlab ketish xavfi (oson)

Soha: Ta'lim (oliy ta'lim)

Universitet dekanati birinchi va ikkinchi kurs talabalari orasida o'qishni tashlab ketish xavfi yuqori bo'lganlarni semestr o'rtasida aniqlab, tyutor biriktirmoqchi. Har talabada 1–4 semestr yozuvi bor — bo'lish talaba bo'yicha. Model «yordam kerak» ro'yxatini tuzadi, hech qanday jazo choralari uchun ishlatilmaydi. Ta'lim tili va jins bo'yicha teng xizmat talab qilinadi; yangi o'quv yilida baholash tizimi o'zgarsa (5 ballikdan 100 ballikka) belgilar siljiydi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 2000 talaba x 1–4 semestr; belgilar: davomat %, o'rtacha ball, topshirilmagan ishlar soni, LMS'ga kirishlar/hafta, qarzdor fanlar, kutubxona olishlari; dropout ~14% |
| Guruhlash kaliti | student_id |
| Sezgir atribut | jins (M/F) va ta'lim tili (o'zbek 70% / rus 20% / ingliz 10%) |
| Model | logreg, DecisionTree(depth 4, tushuntiriladigan), HistGradientBoosting; ONNX MLP 6-16-1 |
| SLO | AUC >= 0.77, recall >= 0.85, TPR farqi (til) <= 0.07, p99 <= 15 ms |
| Drift ssenariysi | 6 hafta, 4-haftadan o'rtacha ball shkalasi 5 -> 100 ballikka o'tadi (masshtab siljishi), LMS kirishlar +30% |
| Adolatlilik kesimi | ta'lim tili va jins; ingliz guruhi kichik (n kam, IO keng) |
| Promote qoidasi | SLO bajarilsa; tushuntiriladigan model (daraxt) AUC farqi <= 0.02 bo'lsa unga ustunlik beriladi |

Qadamlar:

1. Generator, sxema (davomat 0–100, ball 2–5, LMS 0–200, qarzdor 0–8, student_id regex ^S\d{6}$), buzilgan partiya (6 xato), student_id bo'yicha bo'lish, sizish taqqoslash, split x til jadvali.
2. Uch eksperiment, chegara recall >= 0.85, reestr + SHA-256, promote darvozasi «tushuntiriladigan modelga ustunlik» qoidasi bilan; jadval va asoslash.
3. MLP 6-16-1 -> ONNX; parity, kechikish, xeshlar.
4. FastAPI: /predict javobida risk_level (LOW/MEDIUM/HIGH, ikki chegara) va tavsiya matni; /metrics, audit jurnali (student_id EMAS — tuzli xesh); 300 so'rov.
5. Drift: 6 hafta, ball shkalasi siljishi — PSI ALERT va recall keskin tushishi; sxema tekshiruvi (max=5) buni API darajasida 400 bilan ushlashini ko'rsating; ikki himoya qatlamini taqqoslang.
6. MetricFrame til va jins; bootstrap; A: ThresholdOptimizer, B: ingliz guruhi uchun natija «tyutor qo'lda ko'radi» rejimi. Taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json, predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, grafiklar, «ikki himoya qatlami» tahlili, xulosa (>= 200 so'z)

Baholash: Sxema va talaba bo'yicha bo'lish — 10, Eksperimentlar, reestr, promote (tushuntirish qoidasi) — 15, ONNX va kechikish — 10, FastAPI, uch darajali javob, metrikalar, audit — 15, Drift va sxema himoyasi taqqoslash — 15, Adolatlilik va yumshatish — 15, Model kartasi va hisobot — 20 (jami 100)

## 4-variant. Ekologiya: ichimlik suvi namunasi ifloslanganligi (oson)

Soha: Ekologiya (suv sifati)

Viloyat sanitariya laboratoriyasi quduq va suv tarmog'i nuqtalaridan olinadigan namunalar uchun «bakteriologik ifloslanish ehtimoli» xizmatini yaratmoqda — ijobiy natija qimmat laborator ekish tahliliga yuboriladi. Bir nuqtadan yiliga 2–6 namuna olinadi, bo'lish nuqta bo'yicha. Sezgirlik 0.92 dan past bo'lmasin (o'tkazib yuborish xavfli), lekin ekish laboratoriyasi quvvati cheklangan. Yomg'irli mavsumda loyqalik siljiydi; shahar va qishloq nuqtalari teng xizmat olishi kerak.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 1200 nuqta x 2–6 namuna; belgilar: loyqalik NTU, pH, qoldiq xlor mg/l, harorat, elektr o'tkazuvchanlik, nitrat mg/l; ifloslanish ~12% |
| Guruhlash kaliti | site_id |
| Sezgir atribut | nuqta turi (shahar tarmog'i 55% / qishloq qudug'i 45%) |
| Model | logreg, GaussianNB, HistGradientBoosting; ONNX MLP 6-32-1 |
| SLO | recall >= 0.92, AUC >= 0.82, TPR farqi (nuqta turi) <= 0.05, p99 <= 15 ms |
| Drift ssenariysi | 12 kun, 6-kundan yomg'ir mavsumi: loyqalik x1.8, qoldiq xlor −0.2; 10-kunda qishloq nuqtalarida xlorlash boshlanadi (qisman qaytish) |
| Adolatlilik kesimi | nuqta turi va mavsum (quruq/yomg'irli) |
| Promote qoidasi | SLO bajarilsa va val recall joriy production'dan kam bo'lmasa (recall-birinchi qoida) |

Qadamlar:

1. Generator, sxema (NTU 0–100, pH 5–10, xlor 0–5, harorat 0–40, nitrat 0–100, site_id regex ^W\d{5}$), buzilgan partiya 6 xato, site_id bo'yicha bo'lish, sizish taqqoslash, split x nuqta turi jadvali.
2. Uch eksperiment, chegara recall >= 0.92, reestr + xesh, promote «recall-birinchi» qoidasi; jadval; GaussianNB nega zaif ekanini izohlang.
3. MLP 6-32-1 -> ONNX, parity, p50/p99, xeshlar.
4. FastAPI /predict (javob: action = CULTURE_TEST / OK), /healthz, /metrics, audit (nuqta turi, mavsum); 300 so'rov; 401/400/422.
5. Drift 12 kun: yomg'ir siljishi va qisman qaytish; PSI kun x belgi, ball PSI, KS, bayroq ulushi; qaytishdan keyin WARN'ga tushadimi?
6. MetricFrame nuqta turi va mavsum bo'yicha; bootstrap; A: ThresholdOptimizer, B: qishloq quduqlari uchun alohida chegara qo'lda; taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json, predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, PSI va bayroq ulushi grafigi, xulosa (>= 200 so'z)

Baholash: Sxema va nuqta bo'yicha bo'lish — 10, Eksperimentlar, reestr, recall-birinchi promote — 15, ONNX va kechikish — 10, FastAPI va metrikalar — 15, Drift (siljish va qaytish) — 15, Adolatlilik — 15, Model kartasi va hisobot — 20 (jami 100)

## 5-variant. Chakana savdo: do'kon-mahsulot haftalik talab prognozi (oson)

Soha: Chakana savdo

Supermarketlar tarmog'i 40 ta do'konda 30 ta tez buziladigan mahsulot uchun keyingi hafta sotuvini bashorat qiluvchi regressiya xizmatini yaratmoqda. Bitta do'kon-mahsulot juftligida 20 haftalik tarix bor — bo'lish do'kon bo'yicha (yangi do'kon ochilganda model ishlashi kerak). SLO MAPE bilan beriladi; bayram haftalarida talab sakraydi, yangi do'konlar ochilganda taqsimot siljiydi. Markaz va chekka do'konlar uchun xatolik teng bo'lishi talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 40 do'kon x 30 mahsulot x 20 hafta = 24000 qator; belgilar: o'tgan hafta sotuvi, 4 haftalik o'rtacha, narx, chegirma %, bayram bayrog'i, do'kon maydoni m2; maqsad: keyingi hafta sotuvi (dona) |
| Guruhlash kaliti | store_id (GroupShuffleSplit, test 0.25) |
| Sezgir atribut | do'kon joylashuvi (markaz 40% / chekka 60%) |
| Model | Ridge, HistGradientBoostingRegressor, RandomForestRegressor; ONNX MLP 6-32-16-1 (regressiya) |
| SLO | MAPE <= 18%, MAE joylashuv bo'yicha farqi <= 15% nisbiy, p99 <= 15 ms, batch 64 p99 <= 2 ms (ONNX) |
| Drift ssenariysi | 10 hafta oqim, 5-haftadan 8 ta yangi kichik do'kon (maydon −40%, sotuv −35%); 8-hafta bayram sakrashi x1.6 |
| Adolatlilik kesimi | joylashuv va do'kon hajmi (kichik/o'rta/katta) |
| Promote qoidasi | val MAPE SLO ichida va joriy production MAPE dan kamida 0.5 foiz bandi yaxshi |

Qadamlar:

1. Generator (store_id, sku, week, 6 belgi, sales). Sxema: sotuv >= 0, chegirma 0–70, narx > 0, bayram {0,1}, (store_id, sku, week) unikal. Buzilgan partiya 6 xato. store_id bo'yicha bo'lish; tasodifiy bo'lishda nechta do'kon ikki to'plamga tushishini hisoblang.
2. Uch regressiya eksperimenti; metrikalar MAE, MAPE, R2; reestr + xesh; promote «MAPE −0.5 bandi» qoidasi; jadval.
3. Regressiya MLP -> ONNX (chiqish log1p masshtabida, grafda expm1 bilan qaytariladi); parity, p50/p99 batch 1 va 64.
4. FastAPI /predict javobi: forecast, lower, upper (±MAPE), /metrics (Histogram forecast_units), audit (do'kon joylashuvi); 300 so'rov.
5. Drift 10 hafta: belgi PSI, prognoz PSI, KS; MAPE haftalik; bayram sakrashini «kutilgan hodisa» sifatida qoidadan istisno qiling (holiday kalendari) va yolg'on ALERT kamayishini ko'rsating.
6. MetricFrame regressiya uchun: guruh bo'yicha MAE, MAPE, n (metrics lug'atiga mean_absolute_error); bootstrap IO; yumshatish: guruh bo'yicha vazn (sample_weight) bilan qayta o'qitish vs alohida model kichik do'konlar uchun.
7. Model kartasi, pasport, rubrika (regressiya metrikalari bilan).

Topshiriladi: Kod papkasi; registry.json, drift_log.json, predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, haftalik MAPE va PSI grafigi, bayram istisnosi tahlili, xulosa (>= 200 so'z)

Baholash: Sxema va do'kon bo'yicha bo'lish — 10, Regressiya eksperimentlari, reestr, promote — 15, ONNX (log masshtab) va kechikish — 10, FastAPI prognoz oralig'i va metrikalar — 15, Drift va bayram istisnosi — 15, Guruhlar bo'yicha xatolik adolati — 15, Model kartasi va hisobot — 20 (jami 100)

## 6-variant. Sport: futbolchi jarohat xavfi (haftalik yuklama) (oson)

Soha: Sport tibbiyoti

Futbol klubi akademiyasi va asosiy jamoasi o'yinchilari uchun keyingi 7 kunda muskul jarohati xavfini baholaydigan xizmat qurmoqda; xavf yuqori bo'lsa murabbiy mashg'ulot yuklamasini kamaytiradi. Har o'yinchida 10–30 haftalik GPS va so'rovnoma yozuvlari bor — bo'lish o'yinchi bo'yicha. Jarohatlar kam (6%), shuning uchun sezgirlik muhim, lekin har haftada jamoaning 30% dan ko'pini «dam oldirish» mumkin emas. Yangi GPS jiletlari o'rnatilganda masofa belgilari siljiydi; o'smir (U-17) va kattalar guruhlari teng himoyalanishi kerak.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 150 o'yinchi x 10–30 hafta = ~3000 qator; belgilar: haftalik masofa km, yuqori tezlikdagi masofa m, ACWR (o'tkir/surunkali nisbat), uyqu sifati 1-10, charchoq 1-10, o'tgan jarohatlar soni; jarohat ~6% |
| Guruhlash kaliti | player_id |
| Sezgir atribut | yosh guruhi (U-17 30% / kattalar 70%) va jins (erkaklar 65% / ayollar jamoasi 35%) |
| Model | logreg (class_weight=balanced), HistGradientBoosting, kNN(15); ONNX MLP 6-16-8-1 |
| SLO | recall >= 0.80, bayroq ulushi <= 0.30, AUC >= 0.75, TPR farqi (yosh) <= 0.08, p99 <= 15 ms |
| Drift ssenariysi | 8 hafta, 4-haftadan yangi GPS jilet: masofa +8%, yuqori tezlik masofasi +25% (kalibrovka) |
| Adolatlilik kesimi | yosh guruhi va jins; U-17 da so'rovnoma shovqini x2 |
| Promote qoidasi | SLO + bayroq ulushi; nomutanosiblik tufayli AUC o'rniga PR-AUC bilan taqqoslanadi |

Qadamlar:

1. Generator, sxema (masofa 0–60 km, ACWR 0.3–2.5, uyqu va charchoq 1–10, player_id regex ^P\d{4}$, bir o'yinchida jins o'zgarmasligi — qo'shimcha mantiqiy qoida), buzilgan partiya, player_id bo'yicha bo'lish, sizish taqqoslash.
2. Uch eksperiment (class_weight), chegara recall >= 0.80 va bayroq <= 0.30; reestr + xesh; promote PR-AUC bilan; jadval; nomutanosiblikda AUC va PR-AUC farqini izohlang.
3. MLP -> ONNX, parity, kechikish, xeshlar.
4. FastAPI /predict (javob: risk, recommendation = REDUCE_LOAD / NORMAL), /metrics, audit (yosh guruhi, jins); 300 so'rov.
5. Drift 8 hafta: GPS kalibrovka siljishi; PSI kun x belgi; ACWR nisbat belgisi siljimasligini (nisbat!) va shu sababli model ta'siri kichikligini ko'rsating — bu «belgi drifti, lekin zarar kam» holatimi?
6. MetricFrame yosh va jins bo'yicha; bootstrap (o'yinchi bo'yicha!); A: ThresholdOptimizer, B: U-17 uchun shifokor ko'rigi majburiy rejimi; taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json, predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, PR egri chizig'i, PSI grafigi, xulosa (>= 200 so'z)

Baholash: Sxema (mantiqiy qoida bilan) va o'yinchi bo'yicha bo'lish — 10, Nomutanosib eksperimentlar, reestr, PR-AUC promote — 15, ONNX va kechikish — 10, FastAPI va metrikalar — 15, Drift: nisbat belgisi tahlili — 15, Adolatlilik, o'yinchi bo'yicha bootstrap — 15, Model kartasi va hisobot — 20 (jami 100)

## 7-variant. Energetika: bino energiya samaradorligi sinfi (oson)

Soha: Energetika (binolar)

Shahar hokimligi ko'p qavatli uylar uchun «yuqori isitish sarfi» (normadan 30% ko'p) ehtimolini baholaydigan xizmat buyurdi — natija energiya auditi navbatini tuzish uchun ishlatiladi. Har binoda 1–3 isitish mavsumi yozuvi bor, bo'lish bino bo'yicha. Qadimgi (1970 gacha) va yangi binolar uchun sezgirlik teng bo'lishi kerak; aqlli hisoblagichlar o'rnatilgach, o'lchov aniqligi oshib belgilar taqsimoti siljiydi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 1800 bino x 1–3 mavsum; belgilar: umumiy maydon m2, qavatlar, deraza/devor nisbati, izolyatsiya indeksi 0–1, o'rtacha tashqi harorat, kvartiralar soni; yuqori sarf ~28% |
| Guruhlash kaliti | building_id |
| Sezgir atribut | qurilgan davri (1970 gacha 35% / 1970–2000 40% / 2000 dan keyin 25%) va tuman (markaz/chekka) |
| Model | logreg, HistGradientBoosting, ExtraTrees(300); ONNX MLP 6-32-1 |
| SLO | AUC >= 0.80, recall >= 0.85, TPR farqi (davr) <= 0.06, p99 <= 15 ms |
| Drift ssenariysi | 9 oy, 5-oydan aqlli hisoblagich: belgilar shovqini −50% (taqsimot torayadi), o'rtacha o'zgarmaydi |
| Adolatlilik kesimi | qurilgan davri va tuman |
| Promote qoidasi | SLO bajarilsa va Brier ko'rsatkichi joriy production'dan yomon bo'lmasa (kalibrovka qoidasi) |

Qadamlar:

1. Generator, sxema (maydon 300–30000, qavat 2–25, nisbat 0.05–0.6, izolyatsiya 0–1, building_id regex ^B\d{5}$), buzilgan partiya 6 xato, building_id bo'yicha bo'lish, sizish taqqoslash, split x davr jadvali.
2. Uch eksperiment; chegara recall >= 0.85; reestr + xesh; promote darvozasi Brier kalibrovka qoidasi bilan; jadval; kalibrovka egri chizig'i (5 chelak).
3. MLP -> ONNX, parity, kechikish, xeshlar.
4. FastAPI /predict (javob: audit_priority), /healthz, /metrics, audit jurnali (davr, tuman); 300 so'rov p50/p99.
5. Drift 9 oy: taqsimot torayishi (o'rtacha o'zgarmaydi) — PSI va KS qanday javob beradi? Recall o'zgarishini hisoblang va «o'rtacha emas, shakl drifti» xulosasini chiqaring.
6. MetricFrame davr va tuman; bootstrap; A: ThresholdOptimizer, B: davr bo'yicha alohida kalibrovka (CalibratedClassifierCV guruh bo'yicha); taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json, predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, kalibrovka va PSI grafiklari, xulosa (>= 200 so'z)

Baholash: Sxema va bino bo'yicha bo'lish — 10, Eksperimentlar, reestr, kalibrovka promote — 15, ONNX va kechikish — 10, FastAPI va metrikalar — 15, Shakl drifti tahlili — 15, Adolatlilik va guruh kalibrovkasi — 15, Model kartasi va hisobot — 20 (jami 100)

## 8-variant. HR: ishga qabul uchun rezyume skrining (o'rta)

Soha: Inson resurslari

IT kompaniyasi junior dasturchi vakansiyasiga kelgan arizalarni «suhbatga taklif qilish» ehtimoli bo'yicha saralaydigan yordamchi tizim qurmoqda. Bir nomzod bir necha vakansiyaga (1–3) ariza beradi — bo'lish nomzod bo'yicha. Tarixiy yorliqlar HR xodimlari qarorlaridan olingan va ularda jins va yosh bo'yicha tarixiy noxolislik bor (sintetik generatorda ataylab kiritiladi). Tizim talabi: jins bo'yicha tanlov ulushi farqi 0.05 dan oshmasin (demografik paritet), yosh bo'yicha TPR farqi 0.06 dan oshmasin. Tashqi auditor model kartasini talab qiladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 2500 nomzod x 1–3 ariza; belgilar: tajriba yili, test ball 0–100, GitHub faolligi, ta'lim darajasi kodi, til sertifikati, oldingi ish joylari soni; yorliqda jins bo'yicha −0.4 logit tarixiy noxolislik |
| Guruhlash kaliti | candidate_id |
| Sezgir atribut | jins (M 62% / F 38%) va yosh guruhi (18-25, 26-35, 36-45, 46+) |
| Model | logreg, HistGradientBoosting, logreg sezgir atributga korrelyatsion belgilarsiz (proxy olib tashlangan); ONNX MLP 6-32-1 |
| SLO | AUC >= 0.78, recall >= 0.80, tanlov ulushi farqi (jins) <= 0.05, TPR farqi (yosh) <= 0.06, p99 <= 15 ms |
| Drift ssenariysi | 8 hafta, 4-haftadan universitet bitiruvi: 18-25 guruhi ulushi 25% -> 55%, test ball o'rtachasi −6 |
| Adolatlilik kesimi | jins (demographic_parity) va yosh (TPR); kesishma jins x yosh jadvali |
| Promote qoidasi | barcha SLO; adolatlilik SLO buzilsa AUC qanchalik yuqori bo'lmasin rad etiladi (qattiq darvoza) |

Qadamlar:

1. Generator (yorliqda ataylab noxolislik), sxema (ball 0–100, tajriba 0–30, candidate_id regex), buzilgan partiya, candidate_id bo'yicha bo'lish, sizish taqqoslash; dastlabki ma'lumotda jins bo'yicha ijobiy yorliq ulushini jadvalda ko'rsating (noxolislik dalili).
2. Uch eksperiment, shu jumladan proxy belgilarsiz model; chegara recall >= 0.80; reestr + xesh; qattiq adolatlilik darvozasi: tanlov ulushi farqini promote ichida hisoblang; jadval — qaysi model rad etildi va nima uchun.
3. MLP -> ONNX, parity, kechikish; ONNX kirishida sezgir atribut yo'qligini kontrakt bilan kafolatlang.
4. FastAPI /predict (javob: shortlist ehtimoli + «inson qaror qabul qiladi» eslatmasi), /metrics, audit jurnali (jins, yosh guruhi faqat statistik hisobot uchun, alohida fayl); 300 so'rov.
5. Drift 8 hafta: yosh taqsimoti siljishi — ball PSI ALERT, lekin AUC deyarli o'zgarmaydi (populyatsiya drifti vs kontsept drifti); tanlov ulushi farqini har hafta monitoring qiling — adolatlilik drifti ham ogohlantirsin.
6. MetricFrame jins (selection_rate, demographic_parity_difference) va yosh (TPR); kesishma jadvali; A: ThresholdOptimizer(demographic_parity), B: ExponentiatedGradient(DemographicParity) in-processing; C: proxy belgilarsiz model; uchlik taqqoslash: AUC, recall, farqlar.
7. Model kartasi auditor uchun: mo'ljallangan foydalanish, noxolislik manbai, yumshatish, cheklovlar; pasport; rubrika.

Topshiriladi: Kod papkasi; registry.json (rad etilgan versiyalar sabablari bilan), drift_log.json (adolatlilik drifti ustuni bilan), predictions.jsonl; model_card.md auditor formati; Hisobot PDF: noxolislik dalili, uchlik yumshatish taqqoslash, kesishma jadvali, ADR, xulosa (>= 250 so'z)

Baholash: Sxema, nomzod bo'yicha bo'lish, noxolislik dalili — 10, Eksperimentlar va qattiq adolatlilik darvozasi — 15, ONNX va kontrakt kafolati — 10, FastAPI, metrikalar, ma'lumot minimallashtirish — 15, Populyatsiya drifti va adolatlilik drifti monitoringi — 15, Uch yumshatish usuli taqqoslash — 15, Auditor model kartasi va hisobot — 20 (jami 100)

## 9-variant. Energetika: podstansiya soatlik yuklama prognozi (o'rta)

Soha: Energetika (elektr tarmog'i)

Hududiy elektr tarmoqlari 60 ta podstansiya uchun keyingi soat yuklamasini (MVt) bashorat qiluvchi regressiya xizmatini yaratmoqda; prognoz dispetcherga generatsiya rejasini tuzishga yordam beradi. Har podstansiyada 30 kunlik soatlik tarix bor — bo'lish podstansiya bo'yicha (yangi podstansiya ulanganda ishlashi uchun). SLO MAPE va alohida «pik soat» xatoligi bilan beriladi. Yoz issig'i boshlanganda konditsioner yuklamasi taqsimotni siljitadi; shahar va qishloq podstansiyalari uchun xatolik teng bo'lishi kerak.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 60 podstansiya x 30 kun x 24 soat = 43200 qator; belgilar: oldingi soat yuklama, 24 soat oldingi yuklama, haftalik o'rtacha, tashqi harorat, soat (sin/cos), dam olish kuni; maqsad: keyingi soat MVt |
| Guruhlash kaliti | substation_id (test 0.25) |
| Sezgir atribut | podstansiya turi (shahar 45% / qishloq 55%); qishloqda o'lchov shovqini x1.5 |
| Model | Ridge, HistGradientBoostingRegressor, MLPRegressor(64,32); ONNX MLP 7-64-32-1 |
| SLO | MAPE <= 6%, pik soat (18–22) MAPE <= 9%, MAE farqi (tur) <= 20% nisbiy, batch 64 p99 <= 1.5 ms (ONNX), API p99 <= 15 ms |
| Drift ssenariysi | 14 kun, 7-kundan issiq to'lqin: harorat +9 °C, pik yuklama +22%; 12-kunda 5 qishloq podstansiyada quyosh panellari — kunduzgi yuklama −15% |
| Adolatlilik kesimi | podstansiya turi va kun turi (ish/dam) |
| Promote qoidasi | umumiy MAPE va pik MAPE ikkalasi SLO ichida; joriy production'dan pik MAPE yomonlashsa rad etiladi |

Qadamlar:

1. Generator (vaqt qatori, soat sin/cos), sxema (yuklama 0–500 MVt, harorat −25…45, soat 0–23, (substation_id, ts) unikal, ketma-ket soatlar orasida bo'shliq yo'q — mantiqiy qoida), buzilgan partiya, substation_id bo'yicha bo'lish; vaqt bo'yicha sizish haqida ham izoh (oxirgi 5 kun test).
2. Uch regressiya eksperimenti; metrikalar MAE, MAPE, pik MAPE; reestr + xesh; promote ikki MAPE qoidasi; jadval.
3. MLP 7-64-32-1 -> ONNX; parity; batch 1 va 64 p50/p99 (dispetcher 60 podstansiyani bir so'rovda yuboradi — batch 64 muhim).
4. FastAPI /predict_batch (ro'yxat kirish, 1–64), /healthz, /metrics (Histogram batch_size), audit (podstansiya turi, soat); 300 so'rov (har biri batch 60).
5. Drift 14 kun: issiq to'lqin va quyosh panellari; belgi PSI, prognoz PSI, qoldiq (residual) taqsimoti KS; soatlik MAPE; «harorat drifti modelga kiritilgan belgi, demak kutilgan» — qoidani harorat belgisi uchun yumshating va asoslang.
6. MetricFrame tur va kun turi bo'yicha MAE/MAPE; bootstrap; yumshatish: tur bo'yicha sample_weight vs alohida qishloq modeli; taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json, predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, soatlik MAPE grafigi, PSI grafigi, xulosa (>= 250 so'z)

Baholash: Vaqt qatori sxemasi va podstansiya bo'yicha bo'lish — 10, Regressiya eksperimentlari, pik MAPE, reestr — 15, ONNX batch inferens — 10, Batch API va metrikalar — 15, Drift: kutilgan vs kutilmagan siljish — 15, Guruhlar bo'yicha xatolik adolati — 15, Model kartasi va hisobot — 20 (jami 100)

## 10-variant. Agro: barg tasviri belgilaridan kasallik aniqlash (o'rta)

Soha: Qishloq xo'jaligi

Agro-maslahat xizmati fermerlar telefonda suratga olgan g'o'za barglaridan zang kasalligi ehtimolini qaytaradigan API qurmoqda. Sandbox'da tasvir o'rniga undan olingan 6 ta belgi (yashil ulushi, dog' maydoni, dog' soni, rang dispersiyasi, qirra zichligi, yorqinlik) sintetik yaratiladi. Bir daladan 2–8 surat keladi — bo'lish dala bo'yicha. Yangi smartfon modeli chiqqanda kamera ranglari siljiydi. Kichik dehqon xo'jaliklari (eski telefon, shovqinli surat) yirik xo'jaliklar bilan teng sifatli xizmat olishi kerak.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 800 dala x 2–8 surat = ~4000 tasvir; 6 belgi; kasallik ~22%; kichik xo'jalik suratlarida shovqin x1.7 |
| Guruhlash kaliti | field_id |
| Sezgir atribut | xo'jalik turi (kichik dehqon 55% / fermer 30% / klaster 15%) va viloyat (3 ta) |
| Model | logreg, SVC(rbf, probability=True), HistGradientBoosting; ONNX MLP 6-32-16-1 |
| SLO | recall >= 0.88, AUC >= 0.82, TPR farqi (xo'jalik turi) <= 0.07, p99 <= 20 ms |
| Drift ssenariysi | 10 kun, 5-kundan yangi smartfon: yorqinlik +0.8, rang dispersiyasi −25%; 8-kunda kechki suratlar ulushi oshadi (yorqinlik −0.5) |
| Adolatlilik kesimi | xo'jalik turi va viloyat |
| Promote qoidasi | SLO + kichik xo'jalik kesimida recall >= 0.85 (kesim darvozasi) |

Qadamlar:

1. Generator, sxema (image_id regex ^F\d{4}_\d{3}\.jpg$, yashil ulushi 0–1, dog' maydoni 0–1, dog' soni 0–200, label {0,1}), buzilgan partiya 7 xato, field_id bo'yicha bo'lish, sizish taqqoslash, split x xo'jalik turi.
2. Uch eksperiment; chegara recall >= 0.88; reestr + xesh; promote kesim darvozasi (kichik xo'jalik recall) bilan; jadval; SVC o'qitish vaqtini ham yozing.
3. MLP -> ONNX, parity, kechikish; haqiqiy loyihada MobileNet bo'lishi va telefonda ONNX Runtime Mobile ishlatilishi haqida izoh.
4. FastAPI /predict (javob: disease_prob, action = TREAT / MONITOR / OK, ikki chegara), /metrics, audit (xo'jalik turi, viloyat, telefon modeli sarlavhasi); 300 so'rov.
5. Drift 10 kun: kamera siljishi va kechki suratlar — ikki yo'nalishli yorqinlik siljishi PSI'da qanday ko'rinadi; sabab topish uchun telefon modeli bo'yicha kesimli PSI hisoblang.
6. MetricFrame xo'jalik turi va viloyat; bootstrap dala bo'yicha; A: ThresholdOptimizer, B: shovqinli suratlar uchun «qayta suratga oling» javobi (sifat darvozasi); taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json, predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, telefon modeli bo'yicha PSI, grafiklar, xulosa (>= 250 so'z)

Baholash: Sxema va dala bo'yicha bo'lish — 10, Eksperimentlar, kesim darvozasi, reestr — 15, ONNX va kechikish — 10, FastAPI uch darajali javob va metrikalar — 15, Kesimli drift tahlili — 15, Adolatlilik va sifat darvozasi — 15, Model kartasi va hisobot — 20 (jami 100)

## 11-variant. Transport: chorrahada tirbandlik darajasi bashorati (o'rta)

Soha: Transport (shahar harakati)

Shahar transport markazi 120 ta chorraha uchun keyingi 15 daqiqada «og'ir tirbandlik» (o'rtacha tezlik 10 km/soat dan past) ehtimolini bashorat qiluvchi xizmat qurmoqda; natija svetofor rejimini almashtirish uchun ishlatiladi. Har chorrahada 14 kunlik 15 daqiqalik oynalar bor — bo'lish chorraha bo'yicha. Yo'l ta'miri boshlanganda ayrim chorrahalarda oqim keskin o'zgaradi. Markaziy va chekka tumanlar chorrahalari uchun sifat teng bo'lishi talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 120 chorraha x 14 kun x 96 oyna = ~161000 qator (10% namuna olib ishlang — 16000); belgilar: oldingi oyna hajmi (mashina/soat), tezlik, kun soati sin/cos, dam olish, yomg'ir, yaqin tadbir bayrog'i; og'ir tirbandlik ~20% |
| Guruhlash kaliti | intersection_id |
| Sezgir atribut | tuman turi (markaz 35% / chekka 65%) |
| Model | logreg, HistGradientBoosting, kNN(25); ONNX MLP 7-32-1 |
| SLO | AUC >= 0.85, recall >= 0.80, aniqlik >= 0.60, TPR farqi (tuman) <= 0.05, p99 <= 10 ms (real vaqt) |
| Drift ssenariysi | 10 kun, 4-kundan 15 chorrahada yo'l ta'miri: hajm −40%, tezlik −30% (lokal drift); 8-kunda yomg'ir (global drift) |
| Adolatlilik kesimi | tuman turi va kun soati guruhi (tun/ertalab pik/kunduz/kechki pik) |
| Promote qoidasi | SLO + p99 <= 10 ms (kechikish darvozasi — kechikish ham reestrga yoziladi) |

Qadamlar:

1. Generator (vaqt oynalari), sxema (hajm 0–4000, tezlik 0–90, (intersection_id, ts) unikal, tezlik > 0 bo'lsa hajm >= 0 — mantiqiy), buzilgan partiya, intersection_id bo'yicha bo'lish, sizish taqqoslash.
2. Uch eksperiment; chegara recall >= 0.80 va aniqlik >= 0.60 oralig'i; reestr + xesh + kechikish (ONNX p99 batch 1); promote kechikish darvozasi bilan; kNN nega kechikish bo'yicha rad etilishini ko'rsating.
3. MLP -> ONNX, parity, kechikish batch 1 va 120 (barcha chorrahalar bir so'rovda).
4. FastAPI /predict va /predict_batch, /metrics (Histogram batch_size va latency), audit (tuman, soat guruhi); 300 so'rov; p99 SLO tekshiruvi.
5. Drift 10 kun: lokal (15 chorraha) vs global (yomg'ir) drift — umumiy PSI lokal driftni sezmasligini, chorraha bo'yicha kesimli PSI esa sezishini ko'rsating; qoida: «kamida 10% guruhda PSI >= 0.2 bo'lsa WARN».
6. MetricFrame tuman va soat guruhi; bootstrap chorraha bo'yicha; A: ThresholdOptimizer, B: tuman bo'yicha alohida chegara qo'lda (ikki chegara); taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json (kechikish ustuni bilan), drift_log.json (kesimli PSI), predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, lokal vs global drift grafigi, xulosa (>= 250 so'z)

Baholash: Vaqt oynasi sxemasi va chorraha bo'yicha bo'lish — 10, Eksperimentlar va kechikish darvozasi — 15, ONNX batch inferens — 10, Real vaqt API va metrikalar — 15, Lokal vs global drift — 15, Adolatlilik — 15, Model kartasi va hisobot — 20 (jami 100)

## 12-variant. Sug'urta: avtosug'urta da'vo xavfi va narxlash yordamchisi (o'rta)

Soha: Sug'urta

Sug'urta kompaniyasi avto polislar uchun keyingi yilda da'vo bo'lish ehtimolini baholaydigan modelni anderrayterga yordamchi sifatida joylamoqchi. Bir polis egasida 1–3 ta avtomobil polisi bor — bo'lish polis egasi bo'yicha. Qonunga ko'ra jins narxlashda ishlatilmaydi, yosh esa ishlatiladi, lekin 18-24 yosh guruhida yolg'on ijobiy ulushi (FPR) boshqa guruhlardan 0.08 dan ko'p farq qilmasin. Yangi elektromobillar bozorga kirganda belgilar siljiydi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 2200 polis egasi x 1–3 polis; belgilar: haydovchilik staji, yillik yurish km, avto yoshi, dvigatel quvvati, oldingi da'volar soni, hudud xavf indeksi; da'vo ~15% |
| Guruhlash kaliti | policyholder_id |
| Sezgir atribut | jins (modelga kirmaydi, faqat auditda) va yosh guruhi (18-24, 25-40, 41-60, 61+) |
| Model | logreg, HistGradientBoosting, GradientBoosting(100); ONNX MLP 6-32-16-1 |
| SLO | AUC >= 0.76, recall >= 0.70, FPR farqi (yosh) <= 0.08, tanlov ulushi farqi (jins) <= 0.04, p99 <= 20 ms |
| Drift ssenariysi | 12 oy, 6-oydan elektromobillar: dvigatel quvvati belgisi +60%, avto yoshi −3; 9-oyda da'vo ulushi 15% -> 21% (kontsept drifti, yorliq 3 oy kechikadi) |
| Adolatlilik kesimi | yosh guruhi (FPR) va jins (tanlov ulushi — proxy tekshiruvi) |
| Promote qoidasi | SLO + jins proxy testi: model bashorati bilan jins orasidagi AUC <= 0.60 bo'lishi shart |

Qadamlar:

1. Generator (jins ma'lumotda bor, belgilarga kirmaydi), sxema (staj 0–60, km 0–100000, quvvat 40–600 kVt, da'vo {0,1}, policy_id regex ^INS\d{7}$), buzilgan partiya, policyholder_id bo'yicha bo'lish, sizish taqqoslash.
2. Uch eksperiment; chegara recall >= 0.70; reestr + xesh; promote darvozasida jins proxy testi (bashoratdan jinsni bashorat qilish AUC); jadval.
3. MLP -> ONNX, parity, kechikish, xeshlar.
4. FastAPI /predict (javob: risk_score, tier = A/B/C), /metrics, audit (yosh guruhi; jins ALOHIDA audit faylida faqat hisobot uchun); 300 so'rov.
5. Drift 12 oy: belgi drifti (6-oy) va kontsept drifti (9-oy, yorliq kechikadi) — PSI birinchi driftni ko'radi, ikkinchisini ko'rmaydi; kechikkan yorliq kelganda AUC tushishini ko'rsating va «yorliqsiz monitoring chegarasi» haqida xulosa.
6. MetricFrame yosh (FPR, TPR) va jins (selection_rate); bootstrap; A: ThresholdOptimizer(false_positive_rate_parity), B: 18-24 uchun anderrayter qo'lda ko'rishi; taqqoslash.
7. Model kartasi (regulyator uchun bo'lim: qaysi atributlar ishlatilmagan va buni qanday isbotlash mumkin), pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json (proxy test natijasi bilan), drift_log.json, predictions.jsonl, audit_sens.jsonl; model_card.md regulyator bo'limi bilan; Hisobot PDF: jadvallar, belgi vs kontsept drifti grafigi, xulosa (>= 250 so'z)

Baholash: Sxema va polis egasi bo'yicha bo'lish — 10, Eksperimentlar va proxy darvozasi — 15, ONNX va kechikish — 10, FastAPI, ajratilgan sezgir audit — 15, Belgi vs kontsept drifti — 15, FPR adolati va yumshatish — 15, Regulyator model kartasi va hisobot — 20 (jami 100)

## 13-variant. Logistika: yetkazib berish kechikishi bashorati (o'rta)

Soha: Logistika

Marketpleys yetkazib berish buyurtmalarining 24 soatdan ko'p kechikish ehtimolini bashorat qilib, mijozga oldindan xabar bermoqchi va kuryer kompaniyasiga ustuvorlik bermoqchi. Bir marshrutda (shahar juftligi + kuryer kompaniyasi) yuzlab buyurtma bor — bo'lish marshrut bo'yicha, aks holda model marshrutni «yodlab» oladi. Qish mavsumi va bayram oldi talab o'sishi kechikishni oshiradi. Viloyatlarga yetkazish uchun bashorat sifati shaharlardagidek bo'lishi kerak.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 300 marshrut x 20–60 buyurtma = ~12000 qator; belgilar: masofa km, posilka og'irligi, kuryer yuklamasi %, hafta kuni, ob-havo indeksi, ombor qayta ishlash soati; kechikish ~17% |
| Guruhlash kaliti | route_id |
| Sezgir atribut | manzil turi (poytaxt 40% / viloyat markazi 35% / tuman 25%) |
| Model | logreg, HistGradientBoosting, RandomForest(300, min_samples_leaf 20); ONNX MLP 6-32-1 |
| SLO | AUC >= 0.80, recall >= 0.75, TPR farqi (manzil) <= 0.06, batch 256 p99 <= 5 ms (ONNX, tungi paketli hisob), API p99 <= 15 ms |
| Drift ssenariysi | 12 hafta, 6-haftadan qish: ob-havo indeksi +1.2, qayta ishlash soati +30%; 11-haftada bayram: kuryer yuklamasi +45%, kechikish ulushi 17% -> 30% |
| Adolatlilik kesimi | manzil turi va posilka og'irligi guruhi (yengil/o'rta/og'ir) |
| Promote qoidasi | SLO + batch 256 kechikish darvozasi; tuman kesimida recall >= 0.70 |

Qadamlar:

1. Generator, sxema (masofa 1–1500, og'irlik 0.1–50, yuklama 0–150, order_id regex ^ORD\d{8}$ unikal, ombor soati 0–72), buzilgan partiya, route_id bo'yicha bo'lish; tasodifiy bo'lishda marshrut «yodlash» ta'sirini AUC farqi bilan ko'rsating (marshrut bo'yicha vs tasodifiy test AUC).
2. Uch eksperiment; chegara recall >= 0.75; reestr + xesh + batch 256 kechikish; promote kesim va kechikish darvozasi; jadval.
3. MLP -> ONNX; parity; batch 1 va 256 p50/p99; fayl hajmi.
4. FastAPI /predict va /predict_batch (<= 256), /metrics, audit (manzil turi); 300 so'rov; tungi paket uchun o'tkazuvchanlik (buyurtma/s) hisoblang.
5. Drift 12 hafta: qish (asta) va bayram (sakrash); PSI, KS, bayroq ulushi; yorliq kechikishi 2 hafta — haftalik recall «kechikib» keladi, buni jadvalda alohida ustun qilib ko'rsating.
6. MetricFrame manzil va og'irlik guruhi; bootstrap marshrut bo'yicha; A: ThresholdOptimizer, B: tuman manzillari uchun alohida chegara; taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json, predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, «yodlash» dalili, PSI grafigi, xulosa (>= 250 so'z)

Baholash: Sxema, marshrut bo'yicha bo'lish, yodlash dalili — 10, Eksperimentlar, kesim va kechikish darvozasi — 15, ONNX katta batch — 10, Batch API va o'tkazuvchanlik — 15, Drift va kechikkan yorliq — 15, Adolatlilik — 15, Model kartasi va hisobot — 20 (jami 100)

## 14-variant. Ekologiya: havo ifloslanishi xavfli darajasi (PM2.5) (o'rta)

Soha: Ekologiya (atmosfera)

Shahar ekologiya boshqarmasi 25 ta monitoring stansiyasi ma'lumotidan keyingi 6 soatda PM2.5 «zararli» darajadan (75 mkg/m3) oshish ehtimolini bashorat qilib, aholiga SMS ogohlantirish yubormoqchi. Har stansiyada 60 kunlik soatlik yozuv bor — bo'lish stansiya bo'yicha. Yolg'on ogohlantirish ishonchni yo'qotadi (aniqlik >= 0.65), o'tkazib yuborish esa sog'liqqa zarar (recall >= 0.85). Sensorlar vaqt o'tishi bilan kalibrovkadan chiqadi (sekin drift). Sanoat va turar-joy tumanlari stansiyalari uchun sifat teng bo'lishi kerak.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 25 stansiya x 60 kun x 24 soat = 36000 qator; belgilar: joriy PM2.5, 6 soat oldingi PM2.5, shamol tezligi, namlik, harorat inversiyasi bayrog'i, kun soati sin; zararli ~13% |
| Guruhlash kaliti | station_id |
| Sezgir atribut | tuman turi (sanoat 30% / turar-joy 50% / shahar atrofi 20%) |
| Model | logreg, HistGradientBoosting, AdaBoost(100); ONNX MLP 6-32-16-1 |
| SLO | recall >= 0.85, aniqlik >= 0.65, AUC >= 0.88, TPR farqi (tuman) <= 0.06, p99 <= 15 ms |
| Drift ssenariysi | 20 kun, sensor kalibrovka drifti: joriy PM2.5 belgisi har kun +1.5% (sekin, kumulyativ 30%); 15-kunda 5 stansiya qayta kalibrovka (qisman qaytish) |
| Adolatlilik kesimi | tuman turi va mavsum (kun/tun) |
| Promote qoidasi | SLO + aniqlik; bir vaqtda recall va aniqlik bajariladigan chegara mavjud bo'lmasa model rad etiladi |

Qadamlar:

1. Generator, sxema (PM2.5 0–1000, shamol 0–30, namlik 0–100, (station_id, ts) unikal, 6 soat oldingi qiymat joriy jadvaldagi qiymat bilan mos — mantiqiy), buzilgan partiya, station_id bo'yicha bo'lish, sizish taqqoslash.
2. Uch eksperiment; recall va aniqlik ikkalasini qanoatlantiradigan chegara oralig'i; reestr + xesh; promote; jadval; oraliq bo'sh bo'lgan modelni ko'rsating.
3. MLP -> ONNX, parity, kechikish, xeshlar.
4. FastAPI /predict (javob: alert = SMS / NONE, confidence), /metrics (Counter alert turi bo'yicha), audit (tuman, stansiya xeshi); 300 so'rov.
5. Drift 20 kun sekin kumulyativ: kunlik PSI (kechagi kun bilan) deyarli hech narsani ko'rmaydi, mos yozuv bilan PSI 12-kunda WARN, 16-kunda ALERT; «sirpanuvchi mos yozuv tuzog'i» ni ikkala usul grafigi bilan ko'rsating.
6. MetricFrame tuman va kun/tun; bootstrap stansiya bo'yicha; A: ThresholdOptimizer, B: sanoat tumanida qo'shimcha sensor (belgi) bilan qayta o'qitish taqlidi; taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json (ikki PSI usuli), predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, sirpanuvchi vs qat'iy mos yozuv grafigi, xulosa (>= 250 so'z)

Baholash: Sxema va stansiya bo'yicha bo'lish — 10, Ikki shartli chegara va promote — 15, ONNX va kechikish — 10, FastAPI va metrikalar — 15, Sekin drift va mos yozuv tuzog'i — 15, Adolatlilik — 15, Model kartasi va hisobot — 20 (jami 100)

## 15-variant. Shahar xizmatlari: fuqaro murojaatlari ustuvorligi (o'rta)

Soha: Shahar boshqaruvi (elektron murojaat)

Shahar hokimligining murojaatlar portali kelib tushgan arizalar ichidan 48 soat ichida hal qilinishi shart bo'lgan «shoshilinch» (suv uzilishi, yo'l o'pirilishi, gaz hidi) murojaatlarni avtomatik ajratmoqchi. Sandbox'da matn o'rniga undan olingan 6 ta belgi (kalit so'z ballari, kategoriya kodi, takror murojaat soni, geolokatsiya zichligi, soat, foto bor-yo'qligi) sintetik yaratiladi. Bir fuqaro bir nechta murojaat yuboradi — bo'lish fuqaro bo'yicha. Kam daromadli mahallalardan kelgan murojaatlar bir xil sifatda ko'rib chiqilishi kerak. Yangi kategoriya (masalan, issiqlik tarmog'i avariyasi) paydo bo'lganda taqsimot siljiydi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 3000 fuqaro x 1–5 murojaat = ~9000 qator; 6 belgi; shoshilinch ~11%; kam daromadli mahallalarda foto kam (belgi shovqini) |
| Guruhlash kaliti | citizen_id |
| Sezgir atribut | mahalla daromad darajasi (past 30% / o'rta 50% / yuqori 20%) va murojaat tili (o'zbek/rus) |
| Model | logreg, MultinomialNB (kalit so'z ballari uchun), HistGradientBoosting; ONNX MLP 6-32-1 |
| SLO | recall >= 0.90, AUC >= 0.85, TPR farqi (daromad) <= 0.05, tanlov ulushi farqi (til) <= 0.04, p99 <= 15 ms |
| Drift ssenariysi | 10 kun, 6-kundan yangi kategoriya kodi (sxemada yo'q!) murojaatlarning 15% ida; 8-kunda sovuq: takror murojaatlar x2 |
| Adolatlilik kesimi | mahalla daromad darajasi va til |
| Promote qoidasi | SLO; qo'shimcha: past daromadli kesimda recall >= 0.88 |

Qadamlar:

1. Generator, sxema (kategoriya kodi isin {1..12}, kalit so'z balli 0–1, takror 0–20, zichlik 0–1, foto {0,1}, ticket_id regex), buzilgan partiya, citizen_id bo'yicha bo'lish, sizish taqqoslash, split x daromad jadvali.
2. Uch eksperiment; chegara recall >= 0.90; reestr + xesh; promote kesim qoidasi bilan; jadval.
3. MLP -> ONNX, parity, kechikish, xeshlar.
4. FastAPI /predict (javob: priority = URGENT / NORMAL, sla_hours), /metrics, audit (daromad darajasi, til; citizen_id tuzli xesh); 300 so'rov; noma'lum kategoriya kodi 422 qaytarsin.
5. Drift 10 kun: yangi kategoriya — sxema/API 422 bilan rad etadi, demak modelgacha yetmaydi; rad etilgan ulushni metrikada (status=schema_reject) kuzatib, u 10% dan oshsa ALERT beradigan qoida qo'shing; takror murojaat sakrashi uchun PSI.
6. MetricFrame daromad (TPR) va til (selection_rate); bootstrap; A: ThresholdOptimizer, B: foto yo'q murojaatlar uchun alohida chegara; taqqoslash.
7. Model kartasi, pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json (schema_reject ulushi bilan), predictions.jsonl; model_card.md; Hisobot PDF: jadvallar, rad etish ulushi grafigi, xulosa (>= 250 so'z)

Baholash: Sxema va fuqaro bo'yicha bo'lish — 10, Eksperimentlar va kesim promote — 15, ONNX va kechikish — 10, FastAPI, 422 himoyasi, metrikalar — 15, Sxema rad etish drifti — 15, Adolatlilik — 15, Model kartasi va hisobot — 20 (jami 100)

## 16-variant. Moliya: karta tranzaksiyalarida firibgarlik aniqlash (qiyin)

Soha: Moliya (to'lov xavfsizligi)

To'lov tizimi real vaqtda karta tranzaksiyalarini firibgarlik ehtimoli bo'yicha baholab, shubhalilarni SMS-tasdiqqa yuboradi. Firibgarlik juda kam (1.5%), shuning uchun SLO «FPR <= 1% da recall >= 0.80» ko'rinishida; har soat 300 ming tranzaksiya — p99 kechikishi 5 ms dan oshmasin. Bir kartada o'nlab tranzaksiya bor — bo'lish karta bo'yicha, aks holda model kartani yodlaydi. Firibgarlar taktikani o'zgartiradi (kontsept drifti): yangi sxema paydo bo'lganda eski belgilar ishlamay qoladi. Yoshi katta mijozlar uchun yolg'on bloklash ulushi boshqalardan ko'p bo'lmasin.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 5000 karta x 10–40 tranzaksiya = ~125000 qator; belgilar: summa (log), o'rtacha summadan og'ish, oxirgi 1 soatdagi tranzaksiya soni, yangi savdogar bayrog'i, geografik masofa km, tungi vaqt; firibgarlik 1.5% |
| Guruhlash kaliti | card_id (test 0.3) |
| Sezgir atribut | mijoz yosh guruhi (18-30, 31-50, 51-65, 66+) va hudud (poytaxt/viloyat) |
| Model | logreg(class_weight), HistGradientBoosting(class_weight), IsolationForest (nazoratsiz baseline); ONNX MLP 6-32-16-1 pos_weight bilan |
| SLO | recall >= 0.80 @ FPR <= 0.01, PR-AUC >= 0.45, FPR farqi (yosh) <= 0.005 absolyut, ONNX batch 1 p99 <= 0.2 ms, API p99 <= 5 ms |
| Drift ssenariysi | 14 kun, 6-kundan yangi sxema: firibgar tranzaksiyalar kichik summali va kunduzgi bo'ladi (summa belgisi ishlamaydi), firibgarlik ulushi 1.5% -> 2.5%; belgi PSI kichik, lekin recall 0.82 -> 0.55 |
| Adolatlilik kesimi | yosh guruhi (FPR — yolg'on bloklash) va hudud |
| Promote qoidasi | SLO @ FPR 1% + kechikish darvozasi; yangi versiya joriy production'dan PR-AUC bo'yicha 0.02 yaxshi bo'lsa almashtiriladi |

Qadamlar:

1. Generator (nomutanosib, karta bo'yicha), sxema (summa > 0, soatlik son 0–50, masofa 0–20000, tx_id regex ^TX\d{10}$ unikal, bir kartada hudud o'zgarmasligi — mantiqiy), buzilgan partiya 7 xato, card_id bo'yicha bo'lish; tasodifiy bo'lishda sizish ulushi va PR-AUC farqi (yodlash dalili).
2. Uch eksperiment (IsolationForest nazoratsiz baseline sifatida); chegara: FPR <= 1% bo'ladigan eng kichik chegara, so'ng recall o'lchanadi; reestr + xesh + ONNX p99; promote darvozasi; jadval; AUC va PR-AUC farqini izohlang.
3. MLP pos_weight bilan -> ONNX; parity; batch 1 p50/p99 (SLO 0.2 ms) va batch 512; intra_op_num_threads=1 va 2 taqqoslash.
4. FastAPI /score (javob: fraud_prob, action = BLOCK / VERIFY_SMS / ALLOW, uch chegara), /metrics (Counter action bo'yicha, Histogram latency 0.1–10 ms chelaklari), audit (summa EMAS — xesh; yosh guruhi); 300 so'rov p99 <= 5 ms tekshiruvi.
5. Drift 14 kun kontsept drifti: belgi PSI deyarli o'zgarmaydi, ball PSI biroz, lekin VERIFY_SMS ulushi va (kechikkan) recall keskin o'zgaradi — «belgi monitoringi kontsept driftini ko'rmaydi» xulosasi; qo'shimcha proksi-signal: mijoz SMS rad etish ulushi (sintetik) bilan ogohlantirish.
6. MetricFrame yosh (FPR absolyut farq!) va hudud; bootstrap karta bo'yicha; A: ThresholdOptimizer(false_positive_rate_parity), B: 66+ guruhi uchun yuqori chegara + qo'ng'iroq tasdig'i; taqqoslash: recall yo'qotish vs adolat.
7. Model kartasi, pasport (kechikish byudjeti jadvali bilan), rubrika.

Topshiriladi: Kod papkasi; registry.json, drift_log.json (proksi-signal ustuni), predictions.jsonl; model_card.md; Hisobot PDF: PR egri chizig'i, kechikish byudjeti, kontsept drifti grafigi, kesimlar, xulosa (>= 300 so'z)

Baholash: Nomutanosib sxema, karta bo'yicha bo'lish, yodlash dalili — 10, FPR-chegara, PR-AUC, reestr, promote — 15, ONNX 0.2 ms darvozasi va oqimlar — 10, Uch harakatli API va 5 ms p99 — 15, Kontsept drifti va proksi-signal — 15, FPR adolati — 15, Model kartasi va hisobot — 20 (jami 100)

## 17-variant. Sanoat: liniyada mahsulot nuqsonini sensorlardan aniqlash (qiyin)

Soha: Sanoat (sifat nazorati)

Kabel zavodi ekstruziya liniyasida har mahsulot partiyasi uchun 6 ta sensor (harorat, bosim, tezlik, vibratsiya, diametr og'ishi, tok) o'rtacha qiymatlaridan nuqson ehtimolini bashorat qilib, nuqsonli partiyani laboratoriyaga yuboradi. Bir smena ichida sensorlar o'zaro bog'liq — bo'lish smena bo'yicha (yangi smenada model ishlashi uchun). Sensorlar eskirishi bilan bosim va vibratsiya o'lchovlari sekin siljiydi, bir sensor to'satdan buzilishi ham mumkin (NaN yoki doimiy qiymat). Kunduzgi va tungi smenalar uchun sifat teng bo'lishi kerak; ikki liniya (eski va yangi) bor.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 400 smena x 30–60 partiya = ~18000 qator; 6 sensor belgisi; nuqson ~8%; tungi smenada vibratsiya shovqini x1.5; 2 liniya (eski liniyada diametr og'ishi kattaroq) |
| Guruhlash kaliti | shift_id |
| Sezgir atribut | smena (kunduz 55% / tun 45%) va liniya (eski/yangi) |
| Model | logreg, HistGradientBoosting, OneClassSVM (faqat normal partiyalarda o'qitilgan nazoratsiz baseline); ONNX MLP 6-32-16-1 |
| SLO | recall >= 0.90, aniqlik >= 0.50, AUC >= 0.90, TPR farqi (smena) <= 0.05, API p99 <= 10 ms, uzilishsiz: sensor NaN bo'lsa xizmat 400 emas, «degraded» javob |
| Drift ssenariysi | 15 kun, 1-kundan bosim +0.5%/kun va vibratsiya +1%/kun (sekin); 9-kunda vibratsiya sensori buziladi: doimiy 0 qiymat (dispersiya nolga tushadi); 12-kunda almashtiriladi |
| Adolatlilik kesimi | smena va liniya; kesishma smena x liniya |
| Promote qoidasi | SLO + sensor buzilish testi: har bir belgini NaN qilib yuborganda model (imputer bilan) recall >= 0.80 saqlasa — «mustahkamlik» darvozasi |

Qadamlar:

1. Generator, sxema (harorat 150–260, bosim 50–400 bar, tezlik 5–120 m/min, vibratsiya 0–20, diametr og'ishi −1…1 mm, batch_id regex ^L[12]-S\d{4}-\d{3}$, bir smenada liniya o'zgarmasligi), buzilgan partiya 7 xato (shu jumladan doimiy qiymat — dispersiya 0 qoidasi), shift_id bo'yicha bo'lish, sizish taqqoslash.
2. Uch eksperiment; chegara recall >= 0.90 va aniqlik >= 0.50; reestr + xesh; promote mustahkamlik darvozasi (har belgini NaN qilib imputer bilan recall); jadval 6 ustun (har sensor yo'qligida recall).
3. MLP (kirishda NaN -> o'rtacha almashtirish grafga tikiladi: torch.where(isnan)) -> ONNX; parity NaN li kirishda ham; kechikish.
4. FastAPI /predict (javob: defect_prob, status = OK / DEGRADED (sensor yo'q) / REJECT), /metrics (Counter status), audit (smena, liniya, yo'q sensor ro'yxati); 300 so'rov, 30 tasi NaN li.
5. Drift 15 kun: sekin drift (PSI 8-kundan WARN), sensor buzilishi (dispersiya 0 — PSI juda katta, KS p ~0) va almashtirish; «buzilgan sensor drifti» ni «jarayon drifti» dan ajratuvchi qoida (dispersiya nolga yaqin bo'lsa — SENSOR_FAULT, drift emas).
6. MetricFrame smena va liniya, kesishma; bootstrap smena bo'yicha; A: ThresholdOptimizer, B: tungi smena uchun vibratsiya belgisini silliqlash (3 partiya o'rtachasi) bilan qayta o'qitish; taqqoslash.
7. Model kartasi, pasport (sensor buzilishi runbook bo'limi), rubrika.

Topshiriladi: Kod papkasi; registry.json (mustahkamlik jadvali bilan), drift_log.json (SENSOR_FAULT holati), predictions.jsonl; model_card.md + runbook bo'limi; Hisobot PDF: jadvallar, drift va sensor buzilish grafigi, kesishma jadvali, xulosa (>= 300 so'z)

Baholash: Sxema (dispersiya qoidasi), smena bo'yicha bo'lish — 10, Eksperimentlar va mustahkamlik darvozasi — 15, NaN-bardosh ONNX graf — 10, DEGRADED rejimli API — 15, Sensor buzilishi vs jarayon drifti — 15, Adolatlilik va kesishma — 15, Model kartasi, runbook, hisobot — 20 (jami 100)

## 18-variant. Tibbiyot: tez yordam chaqiruvlari triaji (qiyin)

Soha: Tibbiyot (shoshilinch yordam)

103 xizmati dispetcherlik markazi chaqiruv paytida operator kiritgan 6 ta strukturaviy belgi (yosh, hushsizlik, nafas qiyinligi balli, ko'krak og'rig'i, surunkali kasallik soni, chaqiruv davomiyligi) asosida «hayot uchun xavfli» ehtimolini hisoblab, brigadani ustuvor yuboradi. Bir bemor yiliga bir necha marta chaqiradi — bo'lish bemor bo'yicha. O'tkazib yuborish qimmat: recall >= 0.95; lekin barcha chaqiruvlarni shoshilinch deb belgilash mumkin emas (bayroq ulushi <= 0.40). Bolalar (0-12) va keksalar (75+) uchun model ishonchli bo'lishi shart, aks holda ko'lamdan chiqariladi. Qish epidemiyasida belgilar taqsimoti siljiydi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 6000 bemor x 1–4 chaqiruv = ~15000 qator; 6 belgi; xavfli ~14%; 0-12 va 75+ guruhlarida belgilar shovqinliroq (x1.8), ularning ulushi 10% va 15% |
| Guruhlash kaliti | patient_id |
| Sezgir atribut | yosh guruhi (0-12, 13-40, 41-74, 75+) va jins; qo'shimcha: hudud (shahar/qishloq — brigada yetib borish vaqti) |
| Model | logreg, HistGradientBoosting (monotonic_cst: nafas balli va hushsizlik bo'yicha monoton), ekspert qoidasi baseline (hushsizlik yoki nafas balli >= 7 -> xavfli); ONNX MLP 6-32-16-1 |
| SLO | recall >= 0.95, bayroq ulushi <= 0.40, AUC >= 0.85, TPR farqi (yosh) <= 0.04, har guruhda TPR >= 0.92, p99 <= 10 ms |
| Drift ssenariysi | 12 kun, 5-kundan gripp epidemiyasi: nafas balli +1.5, chaqiruvlar soni x1.8, xavfli ulushi 14% -> 19%; 10-kunda yangi so'rovnoma shakli: davomiyligi belgisi birliklari daqiqa -> soniya (60x siljish — sxema tutishi kerak) |
| Adolatlilik kesimi | yosh guruhi (TPR, har guruh alohida chegara bilan) va hudud |
| Promote qoidasi | SLO + ekspert qoidasidan recall bo'yicha yomon bo'lmasligi + monotonlik tekshiruvi (nafas balli oshganda ehtimol kamaymasin) |

Qadamlar:

1. Generator, sxema (yosh 0–105, nafas balli 0–10, hushsizlik {0,1}, surunkali 0–10, davomiylik 10–900 soniya, call_id regex, bir bemorda jins o'zgarmasligi, yosh farqi <= 1), buzilgan partiya 8 xato, patient_id bo'yicha bo'lish, sizish taqqoslash, yosh guruhi x split jadvali.
2. Uch eksperiment (ekspert qoidasi baseline bilan); chegara recall >= 0.95 va bayroq <= 0.40; reestr + xesh; promote: SLO + ekspert bilan taqqoslash + monotonlik testi (100 sintetik nuqtada); jadval.
3. MLP -> ONNX; parity; kechikish; monotonlik ONNX chiqishida ham tekshiriladi.
4. FastAPI /triage (javob: priority = P1 / P2 / P3, uch chegara, eta_minutes hudud bo'yicha), /metrics, audit (yosh guruhi, hudud, kirish xeshi); 300 so'rov; davomiylik 900 dan katta bo'lsa 422.
5. Drift 12 kun: epidemiya (kutilgan, lekin xavfli ulushi oshadi) va birlik xatosi (sxema 422 ushlaydi, rad etish ulushi ALERT); PSI, KS, P1 ulushi, kechikkan recall; brigada yuklamasi (P1 soni x 40 daqiqa) hisoblab resurs ogohlantirishi.
6. MetricFrame yosh (har guruhda TPR >= 0.92 shartini tekshiring) va hudud; bootstrap bemor bo'yicha; A: ThresholdOptimizer(true_positive_rate_parity), B: 0-12 va 75+ uchun «operator majburiy qo'shimcha savollar» rejimi (ko'lam cheklash), C: har guruh uchun alohida chegara qo'lda tanlash; taqqoslash — bayroq ulushi 0.40 dan oshmasligi sharti bilan.
7. Model kartasi (klinik cheklovlar, «tashxis emas, navbat» eslatmasi), pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json (monotonlik va ekspert taqqoslash bilan), drift_log.json (resurs ogohlantirishi), predictions.jsonl; model_card.md klinik bo'lim bilan; Hisobot PDF: jadvallar, guruh bo'yicha TPR IO grafigi, PSI, ADR (A/B/C tanlovi), xulosa (>= 300 so'z)

Baholash: Sxema (mantiqiy qoidalar), bemor bo'yicha bo'lish — 10, Eksperimentlar, ekspert baseline, monotonlik darvozasi — 15, ONNX va monotonlik tekshiruvi — 10, Uch ustuvorlikli API — 15, Epidemiya drifti va birlik xatosi himoyasi — 15, Har guruhda TPR >= 0.92 adolati, uch yumshatish — 15, Klinik model kartasi va hisobot — 20 (jami 100)

## 19-variant. Kiberxavfsizlik: tarmoq oqimlarida anomaliya va hujum aniqlash (qiyin)

Soha: Kiberxavfsizlik

Korporativ SOC markazi tarmoq oqimi (NetFlow) yozuvlaridan hujum ehtimolini baholab, analitik navbatiga qo'yadigan xizmat qurmoqda. Bir xost (qurilma) minglab oqim yaratadi — bo'lish xost bo'yicha, aks holda model xost IP sini yodlaydi. Hujumlar kam (3%) va turlari o'zgaradi (yangi skaner vositasi — kontsept drifti). SOC analitiklari kuniga 200 tadan ortiq ogohlantirishni ko'ra olmaydi — bu bayroq byudjeti. Bo'limlar (IT, moliya, ishlab chiqarish) uchun yolg'on ogohlantirish ulushi teng bo'lishi kerak, aks holda bir bo'lim doimiy «shubhali» bo'lib qoladi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 600 xost x 50–200 oqim = ~75000 qator; belgilar: baytlar (log), paketlar, davomiylik, noyob portlar soni (1 daqiqa), muvaffaqiyatsiz ulanishlar, tashqi IP bayrog'i; hujum 3% (skanerlash, brute-force, eksfiltratsiya) |
| Guruhlash kaliti | host_id |
| Sezgir atribut | bo'lim (IT 25% / moliya 20% / ishlab chiqarish 35% / boshqa 20%); IT bo'limi oqimlari tabiatan «shubhaliroq» (ko'p port) |
| Model | IsolationForest (nazoratsiz), logreg(class_weight), HistGradientBoosting; ikki bosqichli: IsolationForest ball + nazoratli model; ONNX MLP 6-32-16-1 |
| SLO | recall >= 0.85, kuniga ogohlantirish <= 200 (25000 oqimda bayroq ulushi <= 0.8%), PR-AUC >= 0.50, FPR farqi (bo'lim) <= 0.003 absolyut, ONNX batch 1024 p99 <= 3 ms |
| Drift ssenariysi | 14 kun, 6-kundan yangi skaner: noyob portlar belgisi hujumlarda pasayadi (sekin skaner), hujumlar 3% -> 5%; 10-kunda IT bo'limida yangi monitoring agenti — normal oqimlarda tashqi IP ulushi +30% (yolg'on ogohlantirish to'lqini) |
| Adolatlilik kesimi | bo'lim (FPR va ogohlantirish soni) va vaqt (ish soati / tun) |
| Promote qoidasi | SLO + ogohlantirish byudjeti darvozasi: validatsiyada kunlik ogohlantirish soni 200 dan oshsa chegara qayta tanlanadi, baribir recall < 0.85 bo'lsa rad etiladi |

Qadamlar:

1. Generator (xost bo'yicha, 3 hujum turi), sxema (baytlar >= 0, portlar 0–65535, davomiylik 0–3600, flow_id regex unikal, bir xostda bo'lim o'zgarmasligi), buzilgan partiya, host_id bo'yicha bo'lish; tasodifiy bo'lishda PR-AUC «shishib ketishi» (yodlash) dalili.
2. Uch eksperiment (IsolationForest ballini belgi sifatida ikkinchi bosqichga berish); chegara: kunlik byudjet 200 ga mos bayroq ulushi; recall; reestr + xesh; promote byudjet darvozasi; jadval hujum turi bo'yicha recall.
3. MLP -> ONNX; parity; batch 1 va 1024 p50/p99 (oqimlar paketli keladi); o'tkazuvchanlik oqim/s.
4. FastAPI /score_batch (<= 1024 oqim), javob har oqimga: score, alert {0,1}; /metrics (Counter alert bo'lim bo'yicha, Histogram batch_size), audit (bo'lim, xost xeshi, kunlik alert hisoblagichi); 300 so'rov x 256 oqim.
5. Drift 14 kun: kontsept drifti (sekin skaner) — belgi PSI kichik, hujum turi bo'yicha recall tushadi; IT agenti — bo'lim kesimida FPR va alert soni sakraydi (byudjet buziladi); ikki holat uchun ikki xil ogohlantirish (MODEL_DRIFT vs ALERT_BUDGET) va runbook harakati.
6. MetricFrame bo'lim (FPR absolyut, alert soni) va vaqt; bootstrap xost bo'yicha; A: ThresholdOptimizer(false_positive_rate_parity), B: bo'lim bo'yicha alohida IsolationForest mos yozuvi (kontekstual anomaliya) bilan qayta o'qitish; taqqoslash: recall va byudjet.
7. Model kartasi, pasport (hujum turlari bo'yicha qamrov jadvali), rubrika.

Topshiriladi: Kod papkasi; registry.json (hujum turi bo'yicha recall), drift_log.json (ikki xil holat), predictions.jsonl; model_card.md; Hisobot PDF: PR egri chizig'i, byudjet grafigi, bo'lim bo'yicha FPR, drift turlari, xulosa (>= 300 so'z)

Baholash: Sxema, xost bo'yicha bo'lish, yodlash dalili — 10, Ikki bosqichli model, byudjet darvozasi, reestr — 15, ONNX katta batch va o'tkazuvchanlik — 10, Batch API va bo'lim metrikalari — 15, MODEL_DRIFT vs ALERT_BUDGET — 15, Bo'limlar bo'yicha FPR adolati — 15, Model kartasi va hisobot — 20 (jami 100)

## 20-variant. Media: yangiliklar ilovasida kontent tavsiyasi (CTR) (qiyin)

Soha: Media (tavsiya tizimi)

Yangiliklar ilovasi har foydalanuvchiga ko'rsatiladigan maqolalar ro'yxatini «bosish ehtimoli» (CTR) bo'yicha tartiblaydi. Bir foydalanuvchida o'nlab ko'rsatuv (impression) bor — bo'lish foydalanuvchi bo'yicha, aks holda model foydalanuvchini yodlaydi. Tahririyat talabi: yosh guruhlari va til auditoriyasi (o'zbek/rus) uchun mahalliy yangiliklar ko'rsatilish ulushi (exposure) 0.05 dan ko'p farq qilmasin; shuningdek «mashhurlik halqasi» (ko'p bosilgan maqola yana ko'p ko'rsatiladi) monitoring qilinsin. Yangi kontent turi (qisqa video) paydo bo'lganda taqsimot siljiydi; p99 juda qattiq, chunki ro'yxatga 50 maqola bir so'rovda baholanadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot generatori | 4000 foydalanuvchi x 20–60 ko'rsatuv = ~160000 qator (30% namuna — 48000); belgilar: maqola yoshi soat, maqola kategoriyasi kodi, foydalanuvchi kategoriya qiziqishi (0–1), sarlavha uzunligi, mashhurlik (oxirgi soat bosishlar), kun soati; bosish ~6% |
| Guruhlash kaliti | user_id |
| Sezgir atribut | yosh guruhi (18-24, 25-44, 45+) va til (o'zbek 65% / rus 35%) |
| Model | logreg, HistGradientBoosting, FM uslubidagi MLP (kategoriya embedding 8 o'lchov) — PyTorch; ONNX MLP batch 50 |
| SLO | AUC >= 0.72, LogLoss <= 0.21, mahalliy kontent exposure farqi (til) <= 0.05, top-10 ichida har kategoriya >= 1 (xilma-xillik), ONNX batch 50 p99 <= 1 ms, API p99 <= 10 ms |
| Drift ssenariysi | 14 kun, 6-kundan qisqa video kontent (yangi kategoriya kodi, maqola yoshi juda kichik) ko'rsatuvlarning 25% i; 10-kunda mashhurlik halqasi: top-5 maqola mashhurligi x3, bosish ulushi 6% -> 9% (o'z-o'zini kuchaytirish) |
| Adolatlilik kesimi | til (mahalliy kontent exposure — selection_rate analogi) va yosh guruhi (AUC farqi) |
| Promote qoidasi | SLO + xilma-xillik va exposure darvozasi; AUC yuqori bo'lsa ham exposure farqi > 0.05 bo'lsa rad etiladi |

Qadamlar:

1. Generator (foydalanuvchi x maqola ko'rsatuvlari, bosish 6%), sxema (maqola yoshi 0–168 soat, kategoriya isin {1..8}, qiziqish 0–1, impression_id regex unikal, (user_id, article_id, ts) unikal), buzilgan partiya, user_id bo'yicha bo'lish; tasodifiy bo'lishda AUC shishishi dalili.
2. Uch eksperiment (shu jumladan PyTorch embedding model); metrikalar AUC, LogLoss, top-10 xilma-xillik, til bo'yicha mahalliy exposure; reestr + xesh; promote exposure darvozasi; jadval.
3. Embedding MLP -> ONNX (kategoriya kodi int64 kirish + 5 float belgi; Gather operatori); parity; batch 50 p50/p99; fayl hajmi.
4. FastAPI /rank (kirish: foydalanuvchi konteksti + 50 nomzod maqola; javob: tartiblangan ro'yxat + xilma-xillik qayta tartiblash), /metrics (Histogram ro'yxat uzunligi, Counter kategoriya exposure), audit (til, yosh guruhi, top-10 kategoriya ro'yxati); 300 so'rov p99 <= 10 ms.
5. Drift 14 kun: yangi kontent turi (sxema yangi kodni rad etadi — 422 ulushi ALERT, so'ng sxema yangilanib model qayta o'qitiladi: ikki bosqichli runbook) va mashhurlik halqasi (mashhurlik belgisi PSI, top-5 exposure ulushi monitoring); «halqa» ogohlantirish qoidasi: top-5 exposure > 40%.
6. MetricFrame til (mahalliy exposure, selection_rate) va yosh (AUC); bootstrap foydalanuvchi bo'yicha; A: ThresholdOptimizer o'rniga qayta tartiblash (re-ranking) bilan exposure tenglash, B: mashhurlik belgisini log-siqish (damping) bilan qayta o'qitish; taqqoslash: AUC, exposure farqi, xilma-xillik.
7. Model kartasi (tahririyat siyosati bo'limi), pasport, rubrika.

Topshiriladi: Kod papkasi; registry.json (exposure va xilma-xillik bilan), drift_log.json (halqa ogohlantirishi), predictions.jsonl; model_card.md tahririyat bo'limi bilan; Hisobot PDF: jadvallar, exposure va top-5 ulushi grafigi, re-ranking taqqoslash, xulosa (>= 300 so'z)

Baholash: Sxema, foydalanuvchi bo'yicha bo'lish, shishish dalili — 10, Embedding model, exposure darvozasi, reestr — 15, Aralash kirishli ONNX (int64 + float) — 10, /rank API va exposure metrikalari — 15, Yangi kontent va mashhurlik halqasi drifti — 15, Exposure adolati va re-ranking — 15, Model kartasi va hisobot — 20 (jami 100)
