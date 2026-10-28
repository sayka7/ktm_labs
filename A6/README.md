# A6. Sun'iy intellekt tizimlarining xavfsizligi va ishonchliligini tahlil qilish

Bu virtual laboratoriyada talaba sklearn digits (8x8 rasm) ustida kichik PyTorch MLP'ga FGSM va PGD hujumlarini qo'lda yozadi, aniqlikning epsilon'ga bog'liqlik egri chizig'ini quradi, qisqa adversarial o'qitish bilan mustahkamlikni taqqoslaydi, Grad-CAM o'rniga gradient-saliency xaritasini chizadi, sintetik «kredit» ma'lumotida fairlearn MetricFrame va ThresholdOptimizer bilan adolatlilikni o'lchaydi va tuzatadi, shap TreeExplainer bilan daraxt modelini tushuntiradi hamda 20 bandli etik tekshiruv ro'yxatini avtomatik baholaydi. Real labdan farqi: CIFAR-10/ResNet-18 va Adult o'rniga kichik, tarmoqsiz yuklanadigan ma'lumotlar; torchattacks va grad-cam kutubxonalari yo'q — hujumlar va saliency qo'lda yoziladi (bu o'rganish uchun foydaliroq).

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/A6

## Bosqichlar

### 1-qadam. Bazaviy model va FGSM: aniqlikning epsilon'ga bog'liqligi

Digits ma'lumoti [0,1] piksel fazosiga keltiriladi (1 kulrang daraja = 1/16), MLP 300 qadam Adam bilan o'qitiladi (test ~97%). FGSM qo'lda yoziladi: `x + eps*sign(grad)`, [0,1] ga qirqish. 7 ta epsilon uchun aniqlik jadvali va mustahkamlik egri chizig'i chiziladi; shu eps'dagi tasodifiy ±eps shovqin bilan taqqoslanadi. Talaba model chuqurligini, epsilon to'rini va o'qitish qadamlarini o'zgartirib ko'radi.

Fayl: `s1_1_qadam_bazaviy_model_va_fgsm_aniqliknin.py` — ishga tushirish: `python3 s1_1_qadam_bazaviy_model_va_fgsm_aniqliknin.py`

Kutilgan natija: eps=0.05 da aniqlik ~0.77, eps=0.1 da ~0.3, eps=0.15 da ~0.1; tasodifiy shovqin eps=0.15 da ~0.93 qoldiradi. Grafik: keskin tushuvchi egri chiziq.

Tajriba qilib ko'ring:

- `train(..., steps=50)` — kam o'qitilgan model hujumga ko'proq yoki kamroq chidamlimi?
- Modelga `nn.Dropout(0.3)` qo'shing va eval rejimini unutib (`model.train()`), FGSM aniqligini solishtiring — «soxta mustahkamlik».
- Yo'naltirilgan (targeted) FGSM yozing: gradient ishorasini teskari olib, maqsad sinf yo'qotishini kamaytiring.

### 2-qadam. PGD hujumini qo'lda yozish va FGSM bilan taqqoslash

PGD: tasodifiy boshlang'ich nuqta, K qadam `alpha*sign(grad)`, har qadimdan keyin eps-sharga proyeksiya va [0,1] qirqish. FGSM, PGD-10 (tasodifiy boshlanish bilan/siz) va PGD-40 bir xil epsilon to'rida solishtiriladi, har hujum uchun L_inf chegarasi `assert` bilan tekshiriladi. Talaba alpha, qadamlar soni va boshlang'ich nuqtaning ta'sirini ko'radi.

Fayl: `s2_2_qadam_pgd_hujumini_qolda_yozish_va_fgs.py` — ishga tushirish: `python3 s2_2_qadam_pgd_hujumini_qolda_yozish_va_fgs.py`

Kutilgan natija: PGD har eps'da FGSM dan past yoki teng aniqlik beradi (eps=0.15: FGSM ~0.11, PGD ~0.05; eps=0.2: PGD ~0.00). Assert o'tadi. 4 egri chiziqli grafik.

Tajriba qilib ko'ring:

- `alpha = eps` (juda katta qadam) qilib PGD kuchsizlanishini kuzating.
- Proyeksiya satrini olib tashlang — assert nega yiqiladi?
- L_2 normali PGD yozing: gradientni normalashtirib, buzilishni eps radiusli sharga proyeksiya qiling.

### 3-qadam. Adversarial o'qitish (qisqa) va mustahkamlik murozasi

Madry uslubi: har qadamda avval PGD-5 (eps=0.1) bilan adversarial o'quv paketi yasaladi, model aynan unda o'qitiladi. Toza va adversarial modellar FGSM/PGD-10 ostida 5 ta eps uchun taqqoslanadi (4 egri chiziq), o'qitish vaqti nisbati chiqadi. Talaba EPS_TRAIN, k va qadamlar sonini o'zgartirib, murozani o'rganadi.

Fayl: `s3_3_qadam_adversarial_oqitish_qisqa_va_mus.py` — ishga tushirish: `python3 s3_3_qadam_adversarial_oqitish_qisqa_va_mus.py`

Kutilgan natija: Adversarial o'qitish ~6x qimmat; PGD-10 eps=0.1 ostida aniqlik ~0.27 dan ~0.83 ga ko'tariladi; eps=0.2 da himoya zaiflashadi (~0.2).

Tajriba qilib ko'ring:

- EPS_TRAIN=0.05 va 0.2 bilan qayta o'qiting: himoya radiusi va toza aniqlik qanday o'zgaradi?
- k=1 (FGSM adversarial training) — «katastrofik overfitting» kuzatiladimi (FGSM'ga chidamli, PGD'ga zaif)?
- Toza va adversarial yo'qotishni 50/50 aralashtirib (TRADES-ga o'xshash) murozani yumshating.

### 4-qadam. Gradient-saliency xaritasi: toza va hujum ostidagi raqamlar

MLP uchun Grad-CAM o'rniga gradient-saliency (|d logit/d x|, 8x8) hisoblanadi. 8 ta test raqami uchun toza rasm, uning saliency xaritasi, PGD (eps=0.15) bilan buzilgan rasm va uning xaritasi 4 qatorli panelda chiziladi; toza/adv xaritalar korrelyatsiyasi va entropiyasi jadvalda. Talaba tushuntirishning hujum ostida «ko'chishini» ko'radi.

Fayl: `s4_4_qadam_gradient_saliency_xaritasi_toza_.py` — ishga tushirish: `python3 s4_4_qadam_gradient_saliency_xaritasi_toza_.py`

Kutilgan natija: Adversarial rasmlar ko'zga toza bilan deyarli bir xil, lekin bashorat noto'g'ri; saliency korrelyatsiyasi ~0 atrofida (xarita ko'chgan). Panel rasmi.

Tajriba qilib ko'ring:

- SmoothGrad yozing: 20 ta shovqinli nusxa bo'yicha saliency o'rtachasi — xarita silliqroqmi?
- Adversarial o'qitilgan model (3-qadam) uchun xaritalarni solishtiring: korrelyatsiya yuqoriroq chiqadimi?
- Eng muhim 8 pikselni 0 qilib (deletion) bashorat ehtimoli qanchaga tushishini o'lchang — sodiqlik metrikasi.

### 5-qadam. Adolatlilik: MetricFrame va ThresholdOptimizer

Sintetik kredit ma'lumotida sezgir atribut A, u bilan bog'liq vakil belgi (income) va tarixiy tarafkashlik (A=1 guruhda ijobiylarning 25% noto'g'ri rad etilgan) modellashtiriladi. Logistik regressiya uchun MetricFrame guruhlar kesimida aniqlik, tanlov ulushi, TPR, FPR beradi; DP va EO farqlari hisoblanadi. To'rt variant taqqoslanadi: A bilan, A siz (unawareness), ThresholdOptimizer DP va EO cheklovlari bilan.

Fayl: `s5_5_qadam_adolatlilik_metricframe_va_thres.py` — ishga tushirish: `python3 s5_5_qadam_adolatlilik_metricframe_va_thres.py`

Kutilgan natija: Baza: DP farqi ~0.25, EO ~0.27; A olib tashlanganda DP ~0.13 (proxy orqali saqlanadi); ThresholdOptimizer(DP) DP ni ~0.01 ga, (EO) EO ni ~0.02 ga tushiradi, aniqlik 1-3 bandga pasayadi.

Tajriba qilib ko'ring:

- Tarafkashlik ulushini 0.25 dan 0 ga tushiring — bazaviy farq qoladimi (haqiqiy bazaviy ulush farqi tufayli)?
- `fairlearn.reductions.ExponentiatedGradient` + `EqualizedOdds(difference_bound=0.02)` bilan in-processing qo'shing va jadvalga kiriting.
- `income` ni belgilar ro'yxatidan ham olib tashlang: aniqlik va DP farqi qanday o'zgaradi?

### 6-qadam. SHAP TreeExplainer: global muhimlik, lokal izoh, additivlik

breast_cancer (569x30) ustida GradientBoosting o'qitiladi, TreeExplainer log-odds fazosida SHAP qiymatlarini beradi. Global o'rtacha |SHAP| top-10, bitta namuna uchun top-6 hissalar, additivlik tekshiruvi (bazaviy + sum(SHAP) = decision_function, farq ~1e-15) va 1% shovqinda tushuntirish barqarorligi chiqadi; beeswarm grafigi chiziladi.

Fayl: `s6_6_qadam_shap_treeexplainer_global_muhiml.py` — ishga tushirish: `python3 s6_6_qadam_shap_treeexplainer_global_muhiml.py`

Kutilgan natija: Aniqlik ~0.94; top belgilar: worst perimeter, worst concave points, mean concave points; additivlik farqi ~1e-15; barqarorlik ~0.99.

Tajriba qilib ko'ring:

- RandomForestClassifier bilan takrorlang — TreeExplainer ehtimollik fazosida qaytaradi; additivlikni qayta tekshiring.
- Korrelyatsiyalangan belgilar (worst radius / worst perimeter / worst area) hissalari qanday taqsimlangan? Bittasini olib tashlab qayta o'qiting.
- `shap.KernelExplainer` bilan 20 namunada solishtiring va vaqt farqini o'lchang.

### 7-qadam. Etik tekshiruv ro'yxatini avtomatik baholash

20 bandli ro'yxat (ISO/IEC 42001, 23894, EU AI Act, NIST AI RMF) skript bilan baholanadi: texnik bandlar (7, 8, 9, 11, 20) skript ichida tez o'lchangan metrikalardan (toza/FGSM aniqlik, DP/EO farqi, urug' va versiyalar) chegaralar bo'yicha avtomatik, tashkiliy bandlar talaba to'ldiradigan DALIL lug'atidan olinadi; dalilsiz «ha» avtomatik «qisman»ga tushiriladi. Umumiy ball /100 va xulosa chiqadi.

Fayl: `s7_7_qadam_etik_tekshiruv_royxatini_avtomat.py` — ishga tushirish: `python3 s7_7_qadam_etik_tekshiruv_royxatini_avtomat.py`

Kutilgan natija: 20 qatorli jadval, umumiy ball ~60/100, bajarilmagan bandlar ro'yxati (masalan 9, 12, 14, 18) va xulosa.

Tajriba qilib ko'ring:

- DALIL lug'atini o'z tizimingiz uchun to'ldiring va 80+ ballga chiqing — qaysi bandlar texnik ish talab qiladi?
- 9-band chegarasini qondirish uchun 3-qadamdagi adversarial o'qitishni skriptga qo'shing.
- Xavf toifasini (minimal/cheklangan/yuqori) kiritish parametri qiling va yuqori xavfda 12, 14, 18-bandlar «yo'q» bo'lsa umumiy xulosani «taqiqlanadi» deb chiqaring.

## Qo'shimcha topshiriqlar (10 ta, jami 100 ball)

| № | Topshiriq | Daraja | Ball |
|---|---|---|---|
| 1 | **Epsilon to'rini kengaytirish** — 1-qadam kodida epsilon to'rini 0 dan 0.3 gacha 0.01 qadam bilan kengaytiring, FGSM aniqligini har nuqtada o'lchang va aniqlik 50% dan pastga tushadigan eng kichik epsilon'ni (eps_50) toping. Grafik va eps_50 qiymati (kulrang darajalarda ham) hisobotga kiritiladi. | oson | 5 |
| 2 | **Guruhlar kesimida metrikalar** — 5-qadam MetricFrame'iga `false_negative_rate` va `precision` metrikalarini qo'shing, to'rt variant uchun guruhlar jadvalini qayta chiqaring va «kredit» konteksti uchun qaysi metrika farqi eng muhim ekanini 3-4 jumla bilan asoslang. | oson | 5 |
| 3 | **SHAP lokal izohlar** — 6-qadam kodida 3 ta namuna (bitta to'g'ri «xavfsiz», bitta to'g'ri «xavfli», bitta noto'g'ri bashorat) uchun top-5 SHAP hissalarini chiqaring, har biri uchun additivlikni tekshiring va noto'g'ri bashoratda qaysi belgilar modelni chalg'itganini izohlang. | oson | 5 |
| 4 | **Adversarial o'qitish radiusi** — 3-qadam kodini EPS_TRAIN = 0.05, 0.1, 0.2 uchun ishga tushiring va har model uchun toza aniqlik hamda PGD-10 aniqligini eps = 0.05, 0.1, 0.15, 0.2 da o'lchang. 3x4 jadval va 3 egri chiziqli grafik tuzing; o'qitish radiusi kattalashganda toza aniqlik va kichik eps'dagi mustahkamlik qanday o'zgarishini tushuntiring. | o'rta | 10 |
| 5 | **Saliency asosida adversarial detektor** — 4-qadam saliency funksiyasidan foydalanib, 200 toza va 200 PGD (eps=0.15) test namunasi uchun xarita entropiyasi va eng katta 5 piksel ulushini hisoblang. Shu ikki belgi bo'yicha oddiy chegara-detektor (yoki logistik regressiya) yasang, ROC-AUC ni (sklearn.metrics.roc_auc_score) o'lchang va ROC egri chizig'ini chizing. | o'rta | 10 |
| 6 | **In-processing tuzatish** — 5-qadamga fairlearn.reductions.ExponentiatedGradient (EqualizedOdds, difference_bound=0.02) va GridSearch variantlarini qo'shing. Olti usul uchun aniqlik, DP farqi, EO farqi jadvalini tuzing, aniqlik-EO murozasini nuqtali grafikda ko'rsating va post- vs in-processing afzallik/kamchiliklarini (inferensda A kerakmi?) yozing. | o'rta | 10 |
| 7 | **Vakil belgilarni aniqlash** — 5-qadam ma'lumotida har belgining sezgir atribut A bilan korrelyatsiyasini va A ni belgilar orqali bashorat qiluvchi yordamchi modelning AUC qiymatini hisoblang (proxy kuchi). So'ng 6-qadam SHAP yondashuvini 5-qadam modeliga (GradientBoosting) qo'llab, A olib tashlanganda ham A ta'siri qaysi belgi orqali o'tayotganini SHAP hissalari bilan ko'rsating. | o'rta | 10 |
| 8 | **Qora quti hujum: ko'chuvchanlik** — Bir xil ma'lumotda ikki xil arxitekturali model (2-qadam MLP va kengroq/chuqurroq MLP yoki sklearn MLPClassifier) o'qiting. Birinchi modelga qarshi yasalgan FGSM va PGD namunalarini ikkinchisiga berib, ko'chuvchanlik (transfer) aniqligini 4 ta eps uchun o'lchang; oq quti natijasi bilan jadval va grafikda solishtiring va qora quti tahdid modeli uchun xulosa yozing. | qiyin | 15 |
| 9 | **Mos kelmaslik teoremasi amalda** — 5-qadam generatorida bazaviy ulushlar farqini boshqaradigan parametr kiriting (A=1 uchun income siljishi 0, 0.4, 0.8, 1.2). Har holat uchun ThresholdOptimizer(DP) va ThresholdOptimizer(EO) ni qo'llab, DP farqi, EO farqi va guruhlar bo'yicha kalibrlash (bashorat ehtimoli bo'yicha haqiqiy ijobiy ulush, 5 ta bin) ni hisoblang. Bazaviy ulushlar farqi oshganda uchala mezonni bir vaqtda qondirib bo'lmasligini jadval va grafik bilan ko'rsating. | qiyin | 15 |
| 10 | **To'liq ishonchlilik hisoboti generatori** — 7-qadam skriptini kengaytiring: u 1-6 qadam funksiyalarini chaqirib toza/FGSM/PGD aniqligini (3 eps), adversarial o'qitilgan modelni, DP/EO farqlarini (baza va tuzatilgan), SHAP top-5 va saliency korrelyatsiyasini o'lchasin, so'ng 20 bandni dalillar bilan to'ldirib Markdown formatida model kartasi + tekshiruv ro'yxati hisobotini print qilsin. Xavf toifasi (minimal/yuqori) kirish parametri bo'lsin; yuqori xavfda majburiy bandlar bajarilmasa «joriy etish mumkin emas» xulosasi chiqsin. Umumiy ish vaqti 60 s dan oshmasin. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.
