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

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
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

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Mikrokredit: hudud bo'yicha adolatlilik va raqamli imzo rasmlariga FGSM (oson)

Soha: Moliya (mikrokredit)

Mikromoliya tashkiloti arizachilarning skanerlangan raqamli kodlarini (digits) avtomatik o'qiydi va alohida model bilan kredit berish qarorini chiqaradi. Sezgir atribut — hudud (qishloq = 1, shahar = 0); tarixiy ma'lumotda qishloq arizachilarining to'lovga qodir qismi 20 % hollarda noto'g'ri rad etilgan. Sizdan raqam o'qiydigan modelga FGSM/PGD hujumi, qisqa adversarial o'qitish, saliency, kredit modelida DP mezoni bo'yicha tuzatish, SHAP tushuntirish va etik ro'yxatni to'ldirish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf, [0,1] piksel |
| Epsilon to'ri | 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12 |
| PGD | 7 qadam, alpha = 2.5*eps/7, tasodifiy boshlanish |
| EPS_TRAIN | 0.06 |
| Sezgir atribut | hudud (A=1 qishloq, ulushi 0.45) |
| Tarafkashlik | A=1 ijobiylarining 20 % noto'g'ri rad (yorliq flip) |
| Vakil belgi | oylik daromad = N(4.8 - 0.6*A, 1) |
| Adolatlilik mezoni | demographic parity |

Qadamlar:

1. s1 kodini variant epsilon to'ri bilan ishga tushiring; FGSM aniqlik jadvali (eps, eps*16 daraja, aniqlik) va tasodifiy shovqin bilan taqqoslash eps=0.08 da; aniqlik 50 % dan tushadigan eps_50 ni toping.
2. s2 kodida PGD-7 va PGD-20 ni FGSM bilan shu to'rda solishtiring; L_inf assert o'tsin; 3 egri chiziqli grafik.
3. s3 adversarial o'qitishni EPS_TRAIN=0.06, k=5 bilan bajaring; toza vs adversarial model jadvali (eps = 0, 0.04, 0.08, 0.12) va o'qitish vaqti nisbati.
4. s4 saliency panelini 8 ta raqam uchun eps=0.08 PGD bilan chizing; o'rtacha korrelyatsiya va entropiyani yozing.
5. s5 generatorini variant parametrlariga (ulush 0.45, flip 0.2, daromad siljishi 0.6) moslang; baza, unawareness, ThresholdOptimizer(DP) va (EO) uchun aniqlik, DP farqi, EO farqi jadvali; DP nega tanlanganini izohlang.
6. s6 SHAP'ni kredit ma'lumotida o'qitilgan GradientBoosting (A'siz belgilar) uchun bajaring: global top-3, bitta rad etilgan arizachi uchun lokal izoh, additivlik tekshiruvi.
7. s7 ro'yxatini mikrokredit tizimi uchun to'ldiring (xavf toifasi: yuqori), umumiy ball va bajarilmagan majburiy bandlar.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 2-variant. Ishga qabul: rezyume skoringi — jins bo'yicha EO va hujjat rasmlariga PGD (oson)

Soha: Bandlik (ishga qabul)

HR-platforma nomzodlar yuklagan hujjatlardagi raqamli maydonlarni (digits) o'qiydi va alohida skoring modeli suhbatga taklif qilish qarorini chiqaradi. Sezgir atribut — jins (A=1 ayollar, ulushi 0.35); tarixiy ma'lumotda ayol nomzodlarning muvaffaqiyatli qismi 15 % hollarda rad etilgan, qo'shimcha ravishda vakil belgi sifatida 'ish tajribasi yillari' A bilan bog'liq. Sizdan hujjat o'quvchi modelga hujum tahlili va skoring modelida equalized odds bo'yicha tuzatish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, faqat 0-4 sinflar (5 sinf) |
| Epsilon to'ri | 0, 0.03, 0.06, 0.1, 0.15, 0.2 |
| PGD | 10 va 30 qadam, alpha = 2*eps/qadam |
| EPS_TRAIN | 0.1 |
| Sezgir atribut | jins (A=1 ulushi 0.35) |
| Tarafkashlik | A=1 ijobiylarining 15 % flip |
| Vakil belgi | tajriba = N(6 - 1.2*A, 2) |
| Adolatlilik mezoni | equalized odds |

Qadamlar:

1. s1 kodini 0-4 sinflarga cheklang (maska bilan), chiqish qatlami 5 ta; FGSM aniqlik jadvali variant to'rida; 5 sinfli masala 10 sinfliga qaraganda chidamliroqmi — izohlang.
2. s2 PGD-10 va PGD-30 (alpha = 2*eps/qadam) vs FGSM; L_inf assert; grafik.
3. s3 adversarial o'qitish EPS_TRAIN=0.1; toza/adversarial jadval; muroza qiymatlari.
4. s4 saliency: eps=0.1 PGD; 8 ta namuna paneli, korrelyatsiya o'rtachasi.
5. s5 generatori: jins ulushi 0.35, flip 0.15, tajriba belgisi A bilan bog'liq; MetricFrame'ga false_negative_rate qo'shing; 4 usul jadvali; EO nega ishga qabul uchun mos — 3 jumla.
6. s6 SHAP (GradientBoosting, skoring ma'lumoti): tajriba belgisi A ta'sirini qanday o'tkazayotganini SHAP bilan ko'rsating; additivlik.
7. s7 ro'yxati: ishga qabul = yuqori xavf (AI Act III ilova); ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 3-variant. Tibbiy triaj: shoshilinch yordam ustuvorligi — yosh guruhi va o'lchov tarafkashligi (oson)

Soha: Tibbiyot (triaj)

Shoshilinch yordam bo'limi bemorlarni 'shoshilinch / kutishi mumkin' sinfiga ajratuvchi model va analizator ko'rsatkichlarini rasmdan o'qiydigan OCR moduli ishlatadi. Sezgir atribut — yosh guruhi (A=1: 65 yoshdan katta, ulushi 0.3); tarafkashlik yorliqda emas, o'lchovda: keksa bemorlarda asosiy belgi (simptom bali) ko'proq shovqin bilan yozilgan. Sizdan noto'g'ri 'kutishi mumkin' (FN) zarari og'ir bo'lgani uchun TPR tengligi mezonini tekshirish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.01, 0.03, 0.05, 0.08, 0.12 |
| PGD | 10 qadam, tasodifiy boshlanish bilan va boshlanishsiz |
| EPS_TRAIN | 0.05 |
| Sezgir atribut | yosh guruhi (A=1 ulushi 0.30) |
| Tarafkashlik | o'lchov shovqini: simptom bali A=1 da N(0, 1.8), A=0 da N(0, 0.6) |
| Adolatlilik mezoni | true positive rate parity (fairlearn: true_positive_rate_parity) |
| SHAP ma'lumoti | breast_cancer, GradientBoosting |

Qadamlar:

1. s1 FGSM jadvali variant to'rida; kichik eps (0.01, 0.03) da aniqlik pasayishini kulrang darajalarda izohlang.
2. s2 PGD-10 tasodifiy boshlanish bilan/siz vs FGSM; assert; grafik; boshlanish ta'siri 2 jumla.
3. s3 adversarial o'qitish EPS_TRAIN=0.05; jadval; eps=0.12 da himoya zaiflashishini ko'rsating.
4. s4 saliency eps=0.08; korrelyatsiya va entropiya; entropiya detektor belgisi bo'la oladimi — izoh.
5. s5 generatorini o'zgartiring: yorliq flip yo'q, A=1 guruhda simptom bali shovqini kattaroq; baza, unawareness, ThresholdOptimizer(true_positive_rate_parity) va (equalized_odds) uchun aniqlik, TPR farqi, FNR farqi jadvali.
6. s6 SHAP'ni breast_cancer GradientBoosting uchun bajaring: top-10, bitta 'xavfli' namuna lokal izohi, additivlik, 1 % shovqin barqarorligi.
7. s7 ro'yxati triaj tizimi uchun (yuqori xavf, 12 va 18-bandlar majburiy): ball, xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 4-variant. Ta'lim: stipendiya tanlovi — maktab turi bo'yicha DP va imtihon varaqlari OCR hujumi (oson)

Soha: Ta'lim

Universitet stipendiya komissiyasi imtihon varaqlaridagi raqamlarni OCR bilan o'qiydi va alohida model stipendiyaga tavsiya qiladi. Sezgir atribut — maktab turi (A=1 qishloq maktabi, ulushi 0.5); tarixiy qarorlarda qishloq maktabi bitiruvchilarining munosib qismi 30 % hollarda rad etilgan, vakil belgi — 'ingliz tili sertifikati bali'. Sizdan OCR modelining mustahkamligini va stipendiya modelida DP tuzatishini baholash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf, model MLP 64-96-10 (bitta yashirin qatlam) |
| Epsilon to'ri | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3 |
| PGD | 10 va 40 qadam |
| EPS_TRAIN | 0.15 |
| Sezgir atribut | maktab turi (A=1 ulushi 0.50) |
| Tarafkashlik | A=1 ijobiylarining 30 % flip |
| Vakil belgi | ingliz tili bali = N(60 - 12*A, 15) |
| Adolatlilik mezoni | demographic parity |

Qadamlar:

1. s1 modelini bitta yashirin qatlamli (64-96-10) qiling; FGSM jadvali variant to'rida; chuqurroq lab modeli bilan eps=0.1 da solishtiring.
2. s2 PGD-10 va PGD-40 vs FGSM; assert; grafik.
3. s3 EPS_TRAIN=0.15 bilan adversarial o'qitish; toza aniqlik qancha tushdi; eps=0.05 da himoya.
4. s4 saliency eps=0.15; panel; korrelyatsiya.
5. s5 generatori: ulush 0.5, flip 0.3, ingliz tili bali vakil belgi; 4 usul jadvali; MetricFrame'ga precision qo'shing; DP tanlovini 3 jumla bilan asoslang.
6. s6 SHAP (stipendiya ma'lumoti, GradientBoosting): ingliz tili bali hissasi guruhlar bo'yicha o'rtacha qanday farq qiladi; additivlik.
7. s7 ro'yxati ta'lim tizimi uchun (yuqori xavf: ta'limga kirish); ball va tavsiyalar.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 5-variant. Avto sug'urta: narxlash skoringi — yosh guruhi va haqiqiy bazaviy farq (oson)

Soha: Sug'urta

Sug'urta kompaniyasi avto polis uchun 'yuqori xavfli haydovchi' bashoratini qiladi va haydovchilik guvohnomasi raqamlarini rasmdan o'qiydi. Sezgir atribut — yosh (A=1: 25 yoshgacha, ulushi 0.25). Bu variantda yorliq flip yo'q: guruhlar orasida haqiqiy bazaviy ulush farqi bor (yoshlarda hodisalar ko'proq), shuning uchun DP va EO bir vaqtda qondirilmaydi. Sizdan hujum tahlili bilan birga qaysi mezon sug'urta uchun to'g'ri ekanini raqamlar bilan asoslash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.02, 0.05, 0.08, 0.1, 0.15 |
| PGD | 10 qadam, alpha = 3*eps/10 |
| EPS_TRAIN | 0.08 |
| Sezgir atribut | yosh guruhi (A=1 ulushi 0.25) |
| Tarafkashlik | flip 0; bazaviy farq: xavf bali = 0.9*staj + 0.7*A + shovqin |
| Adolatlilik mezoni | equalized odds vs demographic parity (taqqoslash) |
| SHAP ma'lumoti | sug'urta sintetik, GradientBoosting |

Qadamlar:

1. s1 FGSM jadvali; tasodifiy shovqin taqqoslash eps=0.1.
2. s2 PGD-10 (alpha = 3*eps/10) vs FGSM; alpha kattaligining ta'siri haqida izoh; assert; grafik.
3. s3 EPS_TRAIN=0.08 adversarial o'qitish; jadval; vaqt nisbati.
4. s4 saliency eps=0.1; panel; korrelyatsiya.
5. s5 generatorini o'zgartiring: flip yo'q, A to'g'ridan-to'g'ri xavfga ta'sir qiladi (haqiqiy bazaviy farq); 4 usul jadvali; ThresholdOptimizer(DP) aniqlikni qancha tushiradi va EO nega sug'urta uchun ma'qulroq — asoslang.
6. s6 SHAP (GradientBoosting, sug'urta ma'lumoti, A'siz): staj belgisining hissasi; bitta yosh haydovchi uchun lokal izoh; additivlik.
7. s7 ro'yxati (sug'urta narxlash: yuqori xavf AI Act bo'yicha); ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 6-variant. Reklama: ish e'lonlarini ko'rsatish — jins bo'yicha tanlov ulushi va banner raqamlariga hujum (oson)

Soha: Reklama (marketing)

Reklama platformasi foydalanuvchiga yuqori maoshli ish e'lonini ko'rsatish-ko'rsatmaslikni bashorat qiladi va bannerlardagi raqamli kodlarni rasmdan tekshiradi. Sezgir atribut — jins (A=1 ulushi 0.5); tarixiy bosish (click) yorlig'ida ayollar uchun ijobiy natijalar 25 % kam qayd etilgan, vakil belgi — 'ko'rilgan toifalar' soni. Sizdan tanlov ulushi (selection rate) farqini o'lchash va DP tuzatishi reklama samaradorligiga ta'sirini baholash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, juft/toq (2 sinf) |
| Epsilon to'ri | 0, 0.05, 0.1, 0.15, 0.2, 0.3 |
| PGD | 10 qadam |
| EPS_TRAIN | 0.1 |
| Sezgir atribut | jins (A=1 ulushi 0.50) |
| Tarafkashlik | A=1 ijobiylarining 25 % flip |
| Vakil belgi | toifalar soni = Poisson(7 - 2*A) |
| Adolatlilik mezoni | demographic parity (selection rate) |

Qadamlar:

1. s1 modelini 2 sinfli (juft/toq) qiling; FGSM jadvali; 2 sinfli masalada tasodifiy taxmin 50 % ekanini hisobga olib eps_50 o'rniga eps_75 ni toping.
2. s2 PGD-10 vs FGSM; assert; grafik.
3. s3 EPS_TRAIN=0.1 adversarial o'qitish; jadval; muroza.
4. s4 saliency eps=0.15, 8 ta namuna; korrelyatsiya.
5. s5 generatori: ulush 0.5, flip 0.25, Poisson vakil belgi; MetricFrame: selection_rate, TPR, FPR; 4 usul; DP tuzatishdan keyin umumiy tanlov ulushi va aniqlik qanday o'zgardi.
6. s6 SHAP (reklama ma'lumoti, GradientBoosting): toifalar soni belgisi hissasi; beeswarm; additivlik.
7. s7 ro'yxati (reklama: cheklangan xavf, lekin bandlikka ta'sir bo'lsa yuqori); ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 7-variant. Telekom: chegirma taklifi modeli — til/hudud bo'yicha adolatlilik va SIM-karta raqamlari OCR (oson)

Soha: Telekom

Mobil operator ketish ehtimoli yuqori abonentlarga chegirma taklif qiladi; SIM-karta raqamlari rasmdan o'qiladi. Sezgir atribut — xizmat tili/hudud (A=1 ulushi 0.4); tarixda A=1 guruhdagi 'saqlab qolinadigan' abonentlarning 18 % noto'g'ri belgilangan, vakil belgi — 'oylik to'lov'. Sizdan OCR mustahkamligi va chegirma modelida EO tuzatishini baholash hamda wine ma'lumotida SHAP tushuntirish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf, o'qitish 150 qadam (kam o'qitilgan model) |
| Epsilon to'ri | 0, 0.02, 0.05, 0.1, 0.15, 0.2 |
| PGD | 10 qadam |
| EPS_TRAIN | 0.1, k = 3 |
| Sezgir atribut | til/hudud (A=1 ulushi 0.40) |
| Tarafkashlik | A=1 ijobiylarining 18 % flip |
| Vakil belgi | oylik to'lov = N(5.2 - 0.9*A, 1.2) |
| SHAP ma'lumoti | sklearn wine (3 sinf), GradientBoosting |

Qadamlar:

1. s1 modelini 150 qadam o'qiting; FGSM jadvali; 300 qadamli model bilan eps=0.1 da solishtiring (kam o'qitilgan model chidamliroqmi?).
2. s2 PGD-10 vs FGSM; assert; grafik.
3. s3 EPS_TRAIN=0.1, k=3 adversarial o'qitish; k=5 bilan vaqt va mustahkamlikni solishtiring.
4. s4 saliency eps=0.15; panel; korrelyatsiya.
5. s5 generatori: ulush 0.4, flip 0.18, oylik to'lov vakil belgi; 4 usul jadvali; EO tanlovini asoslang.
6. s6 SHAP'ni wine (3 sinf) GradientBoosting uchun bajaring: sinf bo'yicha SHAP massivini to'g'ri tanlang, top-5, additivlik ko'p sinfli holatda.
7. s7 ro'yxati (telekom chegirma: cheklangan xavf); ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 8-variant. Jamoat xavfsizligi: tekshiruv ustuvorligi modeli — hudud bo'yicha FPR tengligi va fikr-mulohaza halqasi (o'rta)

Soha: Xavfsizlik (jamoat tartibi)

Shahar xavfsizlik xizmati qaysi murojaatlarni birinchi tekshirishni bashorat qiluvchi model va hujjat raqamlarini o'qiydigan OCR ishlatadi. Sezgir atribut — hudud (A=1 ulushi 0.35); tarafkashlik fikr-mulohaza halqasi ko'rinishida: A=1 hududda tekshiruvlar ko'proq o'tkazilgani uchun 'ijobiy' yorliqlar 1.6 baravar ko'p qayd etilgan (haqiqiy xavf teng). Noto'g'ri tekshiruv (FP) zarari og'ir, shuning uchun FPR tengligi asosiy mezon.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.02, 0.05, 0.1, 0.15 |
| PGD | 10 va 40 qadam; qo'shimcha L_2 PGD (eps_2 = 1.0, 2.0) |
| EPS_TRAIN | 0.1 |
| Sezgir atribut | hudud (A=1 ulushi 0.35) |
| Tarafkashlik | A=1 guruhda haqiqiy salbiylarning 20 % 'ijobiy' deb yozilgan (ortiqcha nazorat) |
| Adolatlilik mezoni | false positive rate parity |
| SHAP ma'lumoti | xavfsizlik sintetik, GradientBoosting |

Qadamlar:

1. s1 FGSM jadvali variant to'rida.
2. s2 PGD-10, PGD-40 vs FGSM (L_inf) va qo'shimcha L_2 PGD (gradientni normalashtirib, eps_2 radiusli sharga proyeksiya) eps_2 = 1.0, 2.0; ikkala tahdid modelini bitta jadvalda.
3. s3 adversarial o'qitish EPS_TRAIN=0.1; L_inf'ga o'qitilgan model L_2 hujumga qanchalik chidamli — jadval.
4. s4 saliency eps=0.1; panel; korrelyatsiya va entropiya.
5. s5 generatorini o'zgartiring: A=1 guruhda salbiylarning 20 % ijobiy deb flip (ortiqcha nazorat); MetricFrame: FPR, FNR, selection_rate; baza, unawareness, ThresholdOptimizer(false_positive_rate_parity), (equalized_odds); FPR farqi va aniqlik jadvali.
6. Fikr-mulohaza halqasi simulyatsiyasi: tuzatilmagan model bashoratlarini 3 'yil' davomida yangi yorliq sifatida qayta o'qiting va FPR farqi qanday o'sishini grafikda ko'rsating; tuzatilgan model bilan takrorlang.
7. s6 SHAP (sintetik, GradientBoosting): A olib tashlanganda ham qaysi belgi hudud ta'sirini o'tkazadi; s7 ro'yxati (yuqori xavf, huquqni muhofaza qilish — AI Act III ilova): ball, majburiy bandlar.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 9-variant. Uy-joy ijarasi: ijarachini tanlash platformasi — oilaviy holat va vakil belgilar (o'rta)

Soha: Uy-joy (ijara platformasi)

Onlayn ijara platformasi uy egalariga ijarachi arizasini 'tavsiya / tavsiya emas' deb baholaydi va pasport raqamlarini rasmdan o'qiydi. Sezgir atribut — bolalar bilan oila (A=1 ulushi 0.3); tarixiy rad etishlarda A=1 guruhning to'lovga qodir qismi 22 % flip qilingan, ikkita vakil belgi bor: 'oila a'zolari soni' va 'kerakli xonalar soni'. Sizdan vakil belgilar kuchini o'lchash (A ni bashorat qiluvchi yordamchi model AUC) va DP tuzatish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.03, 0.06, 0.1, 0.15, 0.2 |
| PGD | 10 qadam, alpha = 2.5*eps/10 |
| EPS_TRAIN | 0.08 |
| Sezgir atribut | bolalar bilan oila (A=1 ulushi 0.30) |
| Tarafkashlik | A=1 ijobiylarining 22 % flip |
| Vakil belgilar | oila a'zolari = Poisson(2 + 2*A), xonalar = 1 + a'zolar/2 + shovqin |
| Adolatlilik mezoni | demographic parity; proxy kuchi: A ni belgilar orqali bashorat qiluvchi LogisticRegression AUC |

Qadamlar:

1. s1 FGSM jadvali; s2 PGD-10 vs FGSM; assert; grafik (ikkala qadam bitta skriptda).
2. s3 adversarial o'qitish EPS_TRAIN=0.08; jadval; muroza.
3. s4 saliency eps=0.1; panel; SmoothGrad (10 shovqinli nusxa) bilan korrelyatsiya qanday o'zgaradi.
4. s5 generatori: ikki vakil belgi; har belgining A bilan korrelyatsiyasi va A ni bashorat qiluvchi yordamchi model AUC (proxy kuchi) jadvali.
5. 4 usul jadvali (baza, unawareness, ThresholdOptimizer DP va EO); vakil belgilarni ham olib tashlagan 5-usulni qo'shing: aniqlik va DP farqi.
6. s6 SHAP (ijara ma'lumoti, GradientBoosting, A'siz): oila a'zolari va xonalar hissasi A=0/A=1 guruhlarda o'rtacha; additivlik.
7. s7 ro'yxati (uy-joyga kirish: yuqori xavf); ball va xulosa — vakil belgilar bilan nima qilish kerak.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 10-variant. Universitetga qabul: kirish ballari modeli — maktab tili va test balining vakilligi (o'rta)

Soha: Ta'lim (qabul)

Universitet qabul komissiyasi abituriyentning birinchi kursni muvaffaqiyatli tugatish ehtimolini bashorat qiladi va test varaqlaridagi raqamlarni OCR bilan o'qiydi. Sezgir atribut — maktab ta'lim tili (A=1 ulushi 0.4); tarafkashlik ikki qatlamli: test bali A=1 uchun o'rtacha 8 ballga past o'lchangan (o'lchov tarafkashligi), yorliq flip 10 %. Sizdan EO tuzatish, in-processing (ExponentiatedGradient) qo'shish va OCR modeliga qora quti hujumni baholash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf; ikkinchi model sklearn MLPClassifier (qora quti) |
| Epsilon to'ri | 0, 0.05, 0.1, 0.15, 0.2 |
| PGD | 10 qadam |
| EPS_TRAIN | 0.1 |
| Sezgir atribut | maktab tili (A=1 ulushi 0.40) |
| Tarafkashlik | test bali = haqiqiy - 8*A + N(0, 5); flip 10 % |
| Adolatlilik mezoni | equalized odds; in-processing ExponentiatedGradient(EqualizedOdds, difference_bound=0.02) |
| SHAP ma'lumoti | qabul sintetik, GradientBoosting |

Qadamlar:

1. s1 FGSM va s2 PGD-10 jadvallari; assert; grafik.
2. Ko'chuvchanlik: PyTorch MLP uchun yasalgan FGSM/PGD namunalarini sklearn MLPClassifier (hidden 100, 64 belgi) ga bering; 4 eps uchun transfer aniqligi jadvali va oq quti bilan taqqoslash.
3. s3 adversarial o'qitish EPS_TRAIN=0.1; adversarial model uchun transfer hujum qanchalik ishlaydi.
4. s4 saliency eps=0.1; panel; korrelyatsiya.
5. s5 generatori: o'lchov tarafkashligi (test bali - 8*A) + flip 0.1; baza, unawareness, ThresholdOptimizer(EO), ExponentiatedGradient(EO, 0.02) — 4 usul; aniqlik-EO muroza nuqtali grafigi.
6. s6 SHAP (GradientBoosting, qabul ma'lumoti): test bali hissasi A guruhlari bo'yicha; additivlik; shovqin barqarorligi.
7. s7 ro'yxati (ta'limga kirish: yuqori xavf); ball va xulosa — o'lchov tarafkashligini qaysi bosqichda tuzatish kerak (ma'lumot yoki model).

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 11-variant. Sog'liq sug'urtasi: xarajat prognozi — surunkali kasallik guruhi va regressiya SHAP (o'rta)

Soha: Sug'urta (tibbiy)

Tibbiy sug'urta kompaniyasi mijozning yillik xarajatini bashorat qiladi va retsept raqamlarini rasmdan o'qiydi. Sezgir atribut — surunkali kasallik mavjudligi (A=1 ulushi 0.2); kompaniya 'yuqori xarajat' sinfini chegara bilan aniqlaydi va A=1 guruhda tarixiy yorliqlar 20 % ortiqcha 'yuqori' deb belgilangan. Sizdan tasniflash adolatliligi bilan birga diabetes regressiya ma'lumotida SHAP tushuntirish (TreeExplainer, GradientBoostingRegressor) talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.02, 0.04, 0.08, 0.12, 0.16 |
| PGD | 10 qadam; 20 qadam |
| EPS_TRAIN | 0.08 |
| Sezgir atribut | surunkali kasallik (A=1 ulushi 0.20) |
| Tarafkashlik | A=1 salbiylarining 20 % 'yuqori xarajat' deb flip |
| Adolatlilik mezoni | equalized odds, FPR farqiga alohida e'tibor |
| SHAP ma'lumoti | sklearn diabetes (regressiya), GradientBoostingRegressor |

Qadamlar:

1. s1 FGSM va s2 PGD-10/PGD-20 jadvallari; assert; grafik.
2. s3 adversarial o'qitish EPS_TRAIN=0.08; jadval; k=1 (FGSM-AT) bilan 'katastrofik overfitting' bor-yo'qligini tekshiring.
3. s4 saliency eps=0.08; panel; eng muhim 8 pikselni 0 qilib (deletion) bashorat ehtimoli qanchaga tushishini o'lchang.
4. s5 generatori: A=1 salbiylarining 20 % ijobiy flip; MetricFrame: FPR, FNR, selection_rate; baza, unawareness, ThresholdOptimizer(EO), (DP); FPR farqi jadvali.
5. s6 SHAP'ni diabetes GradientBoostingRegressor uchun bajaring: expected_value + sum(SHAP) = predict (additivlik), top-5 belgilar, bitta bemor uchun lokal izoh, dependence plot (bmi).
6. Barqarorlik: 1 % va 5 % shovqinda top-5 mosligi; regressiya SHAP'da qiymatlar birligi (xarajat birligida) haqida izoh.
7. s7 ro'yxati (tibbiy sug'urta: yuqori xavf); ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 12-variant. E-tijorat: to'lov firibgarligi detektori — mamlakat/hudud bo'yicha FPR va nomutanosib sinflar (o'rta)

Soha: E-tijorat (to'lovlar)

To'lov shlyuzi tranzaksiyani 'firibgarlik' deb bloklash qarorini chiqaradi va karta raqamlarini rasmdan o'qiydi. Sezgir atribut — hudud (A=1 ulushi 0.3); firibgarlik 4 % (nomutanosib), A=1 hududda tarixiy bloklash qoidalari tufayli salbiylarning 12 % 'firibgarlik' deb yozilgan. Halol mijozni bloklash (FP) biznesga zarar, shuning uchun FPR tengligi asosiy; sizdan nomutanosib sinflarda MetricFrame bilan to'g'ri o'lchash va chegara tanlash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, faqat 1 vs 7 (2 sinf, chalkash juftlik) |
| Epsilon to'ri | 0, 0.03, 0.06, 0.1, 0.15, 0.2 |
| PGD | 10 qadam; yo'naltirilgan (targeted) FGSM 1 -> 7 |
| EPS_TRAIN | 0.1 |
| Sezgir atribut | hudud (A=1 ulushi 0.30) |
| Tarafkashlik | ijobiy ulush 4 %; A=1 salbiylarining 12 % flip |
| Adolatlilik mezoni | false positive rate parity |
| SHAP ma'lumoti | firibgarlik sintetik, GradientBoosting (class_weight o'rniga sample_weight) |

Qadamlar:

1. s1 modelini 1 vs 7 ikki sinfli qiling; FGSM jadvali; yo'naltirilgan FGSM (1 -> 7, gradientni maqsad sinf yo'qotishi bo'yicha teskari) muvaffaqiyat ulushi eps bo'yicha.
2. s2 PGD-10 vs FGSM; assert; grafik.
3. s3 adversarial o'qitish EPS_TRAIN=0.1; jadval.
4. s4 saliency eps=0.1 uchun 1 va 7 rasmlari; qaysi piksellar hal qiluvchi; korrelyatsiya.
5. s5 generatori: ijobiy ulush 4 % (chegara bilan), A=1 salbiylarining 12 % flip; LogisticRegression(class_weight='balanced'); MetricFrame: FPR, TPR, precision, selection_rate; baza, unawareness, ThresholdOptimizer(false_positive_rate_parity), (equalized_odds).
6. Nomutanosiblik ta'siri: umumiy aniqlik 96 % bo'lsa ham FPR farqi katta bo'lishi mumkinligini ko'rsating; balanced accuracy qo'shing; s6 SHAP (GradientBoosting, sample_weight) top-5 va additivlik.
7. s7 ro'yxati (to'lov bloklash: yuqori xavf — moliyaviy xizmatlarga kirish); ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 13-variant. Ijtimoiy himoya: subsidiya ustuvorligi — oila kattaligi/hudud va kalibrlash (o'rta)

Soha: Davlat xizmatlari

Ijtimoiy himoya agentligi oilalarni subsidiya uchun 'ustuvor' deb belgilaydi va arizadagi raqamlarni OCR bilan o'qiydi. Sezgir atribut — hudud (A=1 ulushi 0.55, ko'pchilik guruh); tarixiy ro'yxatlarda A=1 hududning muhtoj oilalari 28 % hollarda ro'yxatga kiritilmagan, vakil belgi — 'kommunal to'lov qarzi'. Agentlik ehtimollik bahosiga tayanadi, shuning uchun guruhlar bo'yicha kalibrlash ham tekshiriladi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3 |
| PGD | 10 qadam |
| EPS_TRAIN | 0.1 |
| Sezgir atribut | hudud (A=1 ulushi 0.55) |
| Tarafkashlik | A=1 ijobiylarining 28 % flip; vakil belgi: kommunal qarz = N(2 + 1.1*A, 1) |
| Adolatlilik mezoni | demographic parity + guruhlar bo'yicha kalibrlash (5 bin) |
| SHAP ma'lumoti | subsidiya sintetik, GradientBoosting |

Qadamlar:

1. s1 FGSM va s2 PGD-10 jadvallari; assert; grafik.
2. s3 adversarial o'qitish EPS_TRAIN=0.1; jadval; muroza.
3. s4 saliency eps=0.15; panel; korrelyatsiya va entropiya; adversarial o'qitilgan model uchun xaritalar korrelyatsiyasi yuqoriroqmi.
4. s5 generatori: ulush 0.55, flip 0.28, kommunal qarz vakil belgi; 4 usul jadvali (aniqlik, DP, EO).
5. Kalibrlash: har usul uchun predict_proba bo'yicha 5 bin, har binda haqiqiy ijobiy ulush A=0/A=1 uchun; kalibrlash egri chiziqlari; ThresholdOptimizer kalibrlashni buzadimi.
6. s6 SHAP (GradientBoosting, A'siz): kommunal qarz hissasi; bitta rad etilgan oila uchun lokal izoh (fuqaroga tushunarli tilda yozib bering); additivlik.
7. s7 ro'yxati (davlat xizmatlariga kirish: yuqori xavf; 13, 19-bandlar muhim); ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 14-variant. Taksi platformasi: haydovchi reytingi va buyurtma taqsimoti — jins/yosh kesishmasi (o'rta)

Soha: Transport (platforma)

Taksi platformasi haydovchiga 'premium buyurtma' berish qarorini modellashtiradi va avtomobil raqamlarini rasmdan o'qiydi. Ikki sezgir atribut: jins (A1, ulushi 0.2) va yosh guruhi 50+ (A2, ulushi 0.25); tarafkashlik faqat kesishmada — A1=1 va A2=1 bo'lgan haydovchilarning munosib qismi 35 % flip. Sizdan MetricFrame'da kesishgan guruhlar (4 ta) bo'yicha tahlil, ThresholdOptimizer'ni birlashgan atribut bilan qo'llash va avtomobil raqami OCR mustahkamligi talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.04, 0.08, 0.12, 0.16, 0.2 |
| PGD | 10 qadam, alpha = 2.5*eps/10 |
| EPS_TRAIN | 0.12 |
| Sezgir atributlar | jins A1 (ulushi 0.20), yosh 50+ A2 (ulushi 0.25) — kesishma 4 guruh |
| Tarafkashlik | faqat A1=1 va A2=1 kesishmada ijobiylarning 35 % flip |
| Vakil belgi | o'rtacha reyting = N(4.6 - 0.15*A1 - 0.1*A2, 0.2) |
| Adolatlilik mezoni | equalized odds (kesishgan guruhlar) |

Qadamlar:

1. s1 FGSM va s2 PGD-10 jadvallari; assert; grafik.
2. s3 adversarial o'qitish EPS_TRAIN=0.12; jadval; muroza.
3. s4 saliency eps=0.12; panel; korrelyatsiya.
4. s5 generatorini ikki atributli qiling; MetricFrame'ni sensitive_features sifatida DataFrame [A1, A2] bilan chaqiring — 4 kesishgan guruh uchun aniqlik, TPR, FPR, selection_rate; faqat A1 yoki faqat A2 bo'yicha tahlil kesishma muammosini yashirishini ko'rsating.
5. ThresholdOptimizer(EO) ni birlashgan atribut (A1*2 + A2) bilan qo'llang; 4 usul jadvali; guruhlar soni oshganda har guruhda namuna kamayishi va chegaralar barqarorligi haqida izoh (bootstrap 5 marta).
6. s6 SHAP (GradientBoosting, A'siz): reyting belgisi hissasi 4 guruhda; additivlik.
7. s7 ro'yxati (platforma ishchilari: yuqori xavf — bandlik); ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 15-variant. Kontent moderatsiyasi: izohlarni bloklash — til bo'yicha FPR va matn belgilariga hujum (o'rta)

Soha: Media (moderatsiya)

Ijtimoiy tarmoq izohlarni 'zararli' deb avtomatik yashiradi; model so'z-sumka belgilarida ishlaydi, rasmli kodlar uchun OCR bor. Sezgir atribut — izoh tili (A=1 ulushi 0.45); tarafkashlik: A=1 tilidagi zararsiz izohlarning 15 % tarixan noto'g'ri bloklangan, vakil belgi — 'noodatiy belgilar ulushi'. Bu variantda adversarial hujum jadval (so'z-sumka) modeliga ham qo'llanadi: belgilar fazosida L_inf buzilish va chegaralangan to'liq sonli (so'z qo'shish) hujum.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf (OCR moduli) |
| Matn ma'lumoti | sintetik so'z-sumka 6000 x 200 (hisoblagichlar 0-5), 2 sinf |
| Epsilon to'ri (rasm) | 0, 0.05, 0.1, 0.15, 0.2 |
| Epsilon (matn, standartlangan fazo) | 0, 0.1, 0.25, 0.5, 1.0; to'liq sonli hujum: 1, 3, 5 so'z qo'shish |
| EPS_TRAIN | 0.1 (rasm) |
| Sezgir atribut | izoh tili (A=1 ulushi 0.45) |
| Tarafkashlik | A=1 salbiylarining 15 % flip (ortiqcha bloklash) |
| Adolatlilik mezoni | false positive rate parity |

Qadamlar:

1. s1/s2: digits OCR uchun FGSM va PGD-10 jadvali; assert; grafik.
2. Matn modeli: so'z-sumka generatori + MLP 200-64-2; standartlangan fazoda FGSM/PGD (eps ro'yxati) va to'liq sonli hujum (gradient bo'yicha eng ta'sirli k ta so'z hisoblagichini +1) uchun aniqlik jadvali; qaysi tahdid modeli realistik — izoh.
3. s3 adversarial o'qitish (rasm, EPS_TRAIN=0.1) va matn modeli uchun to'liq sonli hujumga qarshi augmentatsiya; jadval.
4. s4 saliency: rasm uchun panel; matn uchun eng ta'sirli 10 so'z indeksini (|gradient|) chiqaring.
5. s5 generatori: til ulushi 0.45, salbiylarning 15 % flip, noodatiy belgilar ulushi vakil belgi; MetricFrame FPR/TPR; baza, unawareness, ThresholdOptimizer(false_positive_rate_parity), (EO).
6. s6 SHAP (GradientBoosting, moderatsiya sintetik ma'lumoti): noodatiy belgilar hissasi; additivlik; 1 % shovqin barqarorligi.
7. s7 ro'yxati (so'z erkinligi, DSA talablari — cheklangan/yuqori xavf); 13 va 19-bandlar (foydalanuvchi xabardorligi, shikoyat) majburiy; ball va xulosa.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (hujumlar L_inf chegarasida, assert o'tadi) — 30, O'lchovlar va jadvallar (mustahkamlik, adolatlilik, SHAP) — 25, Tahlil va xulosa (mezon tanlovi asoslangan) — 25, Hisobot va tekshiruv ro'yxati sifati — 20 (jami 100)

## 16-variant. Bank: qarz undirish ustuvorligi — kalibrlash, DP va EO mos kelmasligi amalda (qiyin)

Soha: Moliya (bank)

Bank muddati o'tgan qarzdorlar orasidan 'qattiq undirish' choralarini qo'llanadiganlarni bashorat qiladi; sezgir atribut — hudud (A=1 ulushi 0.4). Generatorda A=1 uchun daromad siljishi parametri bor (0, 0.5, 1.0, 1.5): bazaviy ulushlar farqi oshganda DP, EO va kalibrlashni bir vaqtda qondirib bo'lmasligini ko'rsatish kerak. Qo'shimcha: bank OCR moduliga qora quti (transfer) hujum va in-processing tuzatish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf; qora quti: kengroq MLP 64-256-256-10 |
| Epsilon to'ri | 0, 0.05, 0.1, 0.15, 0.2 |
| PGD | 10 va 40 qadam |
| EPS_TRAIN | 0.1 |
| Sezgir atribut | hudud (A=1 ulushi 0.40) |
| Daromad siljishi (A=1) | 0, 0.5, 1.0, 1.5 |
| Tarafkashlik | flip 10 % (doimiy) |
| Mezonlar | DP, EO, kalibrlash (5 bin), ExponentiatedGradient(EqualizedOdds, 0.02) |

Qadamlar:

1. s1/s2: FGSM, PGD-10, PGD-40 jadvali; assert; ko'chuvchanlik: lab MLP'ga qarshi yasalgan namunalarni kengroq MLP'ga bering, 4 eps uchun transfer aniqligi.
2. s3 adversarial o'qitish EPS_TRAIN=0.1; adversarial modelga transfer hujum aniqligi; s4 saliency korrelyatsiyasi toza va adversarial modellar uchun.
3. s5 generatoriga siljish parametri kiriting; har siljish (0, 0.5, 1.0, 1.5) uchun bazaviy ulushlar P(y=1|A) jadvali.
4. Har siljish uchun baza, ThresholdOptimizer(DP), ThresholdOptimizer(EO), ExponentiatedGradient(EO) — DP farqi, EO farqi, aniqlik va guruhlar bo'yicha kalibrlash xatosi (5 bin, |haqiqiy ulush - bashorat ehtimoli| o'rtachasi); 4 x 4 jadval.
5. Grafik: x — siljish, y — DP farqi / EO farqi / kalibrlash xatosi, har usul alohida; siljish oshganda uchala mezonni birga qondirib bo'lmasligini (Kleinberg-Chouldechova) raqamlar bilan ko'rsating.
6. s6 SHAP (GradientBoosting, siljish 1.0 holati): daromad hissasi guruhlar bo'yicha; additivlik; barqarorlik.
7. s7 ro'yxati (qarz undirish: yuqori xavf) + bank uchun 1 sahifalik tavsiya: qaysi mezon, nega, qanday monitoring; ball.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadvallar va grafiklar — 25, Chuqur tahlil (murozalar, mos kelmaslik, cheklovlar) — 30, Hisobot, model kartasi va tekshiruv ro'yxati sifati — 20 (jami 100)

## 17-variant. Kasalxona: qayta yotqizish xavfi — yosh va jins kesishmasi, ko'chuvchan hujum va detektor (qiyin)

Soha: Tibbiyot (klinika)

Kasalxona 30 kun ichida qayta yotqizish xavfini bashorat qiladi; ikki sezgir atribut (yosh 65+, jins). Tibbiy ko'rsatkichlar OCR orqali rasmdan olinadi va hujumchi OCR modelining arxitekturasini bilmaydi (qora quti). Sizdan transfer hujum, saliency asosida adversarial detektor (ROC-AUC), kesishgan guruhlar adolatliligi (EO), breast_cancer SHAP va yuqori xavfli tibbiy tizim uchun to'liq tekshiruv ro'yxati talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf; qora quti: sklearn MLPClassifier(hidden 128,64) va o'zgacha PyTorch MLP 64-64-64-10 |
| Epsilon to'ri | 0, 0.05, 0.1, 0.15, 0.2 |
| PGD | 10 qadam; EPS_TRAIN 0.1 |
| Detektor | saliency entropiyasi + top-5 piksel ulushi, 200 toza + 200 PGD namuna, LogisticRegression, ROC-AUC |
| Sezgir atributlar | yosh 65+ (0.3), jins (0.5) — 4 kesishgan guruh |
| Tarafkashlik | keksa ayollar guruhida ijobiylarning 25 % flip; o'lchov shovqini keksalarda 1.5x |
| Adolatlilik mezoni | equalized odds (kesishma) |
| SHAP ma'lumoti | breast_cancer, GradientBoosting |

Qadamlar:

1. s1/s2 oq quti jadvali; so'ng ikki qora quti modelga transfer aniqligi 4 eps uchun; qaysi arxitektura juftligi ko'proq ko'chadi — izoh.
2. s3 adversarial o'qitish EPS_TRAIN=0.1; adversarial model uchun transfer; jadval va grafik (6 egri chiziq).
3. s4 saliency asosida detektor: 200 toza + 200 PGD (eps=0.15) namuna uchun entropiya va top-5 ulush; LogisticRegression detektor; ROC-AUC va ROC egri chizig'i; eps=0.05 namunalarda AUC qanday tushadi.
4. s5 generatori ikki atributli (yosh, jins), kesishmada flip 0.25 va o'lchov shovqini; MetricFrame 4 guruh (TPR, FPR, FNR); baza, unawareness, ThresholdOptimizer(EO) birlashgan atribut bilan, ExponentiatedGradient(EO).
5. Kalibrlash 4 guruh bo'yicha (5 bin) va FNR farqi — klinik zarar nuqtai nazaridan qaysi metrika birinchi; jadval + grafik.
6. s6 SHAP breast_cancer: top-10, 3 ta namuna (to'g'ri xavfsiz, to'g'ri xavfli, noto'g'ri) lokal izohlari; additivlik; korrelyatsiyalangan belgilar (worst radius/perimeter/area) hissalari taqsimoti.
7. s7 ro'yxati tibbiy tizim uchun (yuqori xavf, MDR/AI Act): 12, 14, 18-bandlar majburiy, xavf toifasi kiritish parametri, bajarilmasa 'joriy etish mumkin emas' xulosasi; ball.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadvallar va grafiklar — 25, Chuqur tahlil (murozalar, mos kelmaslik, cheklovlar) — 30, Hisobot, model kartasi va tekshiruv ro'yxati sifati — 20 (jami 100)

## 18-variant. HR: rezyume saralash tizimi — to'liq ishonchlilik hisoboti va model kartasi generatori (qiyin)

Soha: Bandlik (HR tizimi)

Yirik kompaniya rezyumelarni avtomatik saralaydi (jins va yosh — sezgir atributlar) va skanerlangan hujjatlardagi raqamlarni OCR bilan o'qiydi. Yuridik bo'lim audit oldidan Markdown formatida model kartasi + 20 bandli tekshiruv ro'yxatini avtomatik ishlab chiqaruvchi skript talab qiladi: u 1-6 qadam funksiyalarini chaqirib barcha metrikalarni o'lchasin, xavf toifasi parametriga qarab yakuniy xulosa chiqarsin va 60 s ichida ishlasin.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.05, 0.1, 0.2 |
| PGD | 10 qadam |
| EPS_TRAIN | 0.1, k = 3 (vaqt chegarasi uchun) |
| Sezgir atributlar | jins (A1, 0.4) va yosh 45+ (A2, 0.3) |
| Tarafkashlik | A1=1 ijobiylarining 20 % flip; A2=1 uchun 'oxirgi ish davomiyligi' belgisi siljigan |
| Adolatlilik mezoni | equalized odds, DP qo'shimcha |
| Chiqish | Markdown hisobot: model kartasi, metrikalar jadvali, 20 band, xulosa; xavf toifasi = minimal/yuqori; vaqt chegarasi 60 s |

Qadamlar:

1. 1-4 qadam funksiyalarini bitta modulga yig'ing (train, fgsm, pgd, adv_train, saliency); toza/FGSM/PGD aniqligini 3 eps uchun, adversarial modelni (k=3) va saliency korrelyatsiyasini o'lchaydigan funksiya yozing; vaqtini o'lchang.
2. s5 generatorini ikki atributli HR ssenariysiga moslang; baza va ThresholdOptimizer(EO) uchun DP/EO farqlari va kesishgan guruhlar jadvali funksiya ko'rinishida.
3. s6 SHAP (GradientBoosting, HR ma'lumoti): top-5 belgilar va additivlik farqini qaytaruvchi funksiya.
4. s7 ro'yxatini kengaytiring: texnik bandlar (7, 8, 9, 11, 20) o'lchovlardan avtomatik, 2 ta yangi band qo'shing (kesishgan guruhlar tahlili; adversarial detektor mavjudligi) — jami 22 band, ball 100 ga normallashtirilsin.
5. Markdown generator: model kartasi (maqsad, ma'lumot, metrikalar, cheklovlar), metrikalar jadvali, tekshiruv ro'yxati jadvali, bajarilmagan bandlar, xulosa; xavf toifasi 'yuqori' bo'lsa 12, 14, 18-bandlar bajarilmaganda 'joriy etish mumkin emas'.
6. Skriptni ikki rejimda ishga tushiring (minimal, yuqori) va ikkala hisobotni saqlang; umumiy vaqt 60 s dan oshmasligini ko'rsating (vaqt jadvali: qaysi qadam qancha).
7. Xulosa: HR tizimi uchun qaysi bandlar texnik ish talab qiladi, 80+ ballga chiqish rejasi; hisobotning auditor uchun yetarliligi haqida tanqidiy baho.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadvallar va grafiklar — 25, Chuqur tahlil (murozalar, mos kelmaslik, cheklovlar) — 30, Hisobot, model kartasi va tekshiruv ro'yxati sifati — 20 (jami 100)

## 19-variant. Mobil mikromoliya: telefon ma'lumotlari skoringi — vakil belgilarni SHAP bilan fosh qilish (qiyin)

Soha: Moliya (fintech)

Fintech ilovasi telefon foydalanish ma'lumotlaridan (qo'ng'iroqlar, ilovalar, joylashuv) mikroqarz berishni hal qiladi; sezgir atribut A (hudud, 0.4) belgilar ro'yxatida yo'q, lekin 6 ta belgidan 3 tasi A bilan turlicha kuchda bog'liq. Sizdan vakil belgilar kuchini o'lchash (korrelyatsiya, yordamchi model AUC), SHAP orqali A ta'siri qaysi belgi orqali o'tishini ko'rsatish, vakil belgilarni bosqichma-bosqich olib tashlab aniqlik-adolatlilik egri chizig'ini qurish va OCR moduliga L_2 hamda L_inf hujumlarni taqqoslash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | L_inf: 0, 0.05, 0.1, 0.15; L_2: 0, 0.5, 1.0, 2.0 |
| PGD | 10 qadam (L_inf va L_2 versiyalari); EPS_TRAIN 0.1 (L_inf) |
| Sezgir atribut | hudud (A=1 ulushi 0.40), belgilarda yo'q |
| Vakil belgilar | tungi qo'ng'iroqlar (korr 0.6), ilovalar soni (korr 0.35), zaryad sikli (korr 0.15) + 3 ta neytral belgi |
| Tarafkashlik | A=1 ijobiylarining 20 % flip |
| Adolatlilik mezoni | demographic parity |
| SHAP | GradientBoosting, TreeExplainer, guruhlar bo'yicha o'rtacha hissa |

Qadamlar:

1. s1/s2: L_inf FGSM/PGD jadvali va L_2 PGD (gradient normalashtirilgan, eps_2 radiusli sharga proyeksiya) jadvali; L_2 va L_inf buzilishlarining ko'rinishini panelda solishtiring.
2. s3 adversarial o'qitish (L_inf 0.1); L_2 hujumga chidamlilik; s4 saliency korrelyatsiyasi.
3. Generator: 6 belgi, uchtasi A bilan berilgan korrelyatsiyada (Cholesky yoki aralashtirish bilan); har belgining A bilan korrelyatsiyasi va A ni bashorat qiluvchi yordamchi model AUC jadvali.
4. s5: baza (A'siz, 6 belgi), ThresholdOptimizer(DP), ExponentiatedGradient(DP); so'ng vakil belgilarni kuchi bo'yicha birin-ketin olib tashlab (3 bosqich) aniqlik va DP farqi egri chizig'ini chizing.
5. s6 SHAP (GradientBoosting, 6 belgi): har belgi uchun o'rtacha SHAP hissasi A=0 va A=1 guruhlarida — farq eng katta belgi A ta'sirini o'tkazuvchi ekanini ko'rsating; additivlik; barqarorlik.
6. Taqqoslash: vakil belgini olib tashlash vs post-processing vs in-processing — aniqlik, DP farqi, inferensda A kerakmi, tushuntirish murakkabligi; jadval.
7. s7 ro'yxati (mikromoliya: yuqori xavf), 5-band (proxy aniqlangan) uchun o'z o'lchovlaringizni dalil qiling; ball va fintech uchun tavsiya.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadvallar va grafiklar — 25, Chuqur tahlil (murozalar, mos kelmaslik, cheklovlar) — 30, Hisobot, model kartasi va tekshiruv ro'yxati sifati — 20 (jami 100)

## 20-variant. Jinoyat sodir etish xavfini baholash: COMPAS uslubidagi sintetik tizim — kalibrlash vs xatolik tengligi (qiyin)

Soha: Huquqni muhofaza qilish

Sud-huquq tizimi uchun takroriy huquqbuzarlik xavfini baholovchi vositaning sintetik nusxasi tahlil qilinadi: sezgir atribut — guruh (A=1 ulushi 0.45), bazaviy ulushlar haqiqatan farq qiladi (0.45 vs 0.30) va tarixiy hibs yozuvlari A=1 da 1.4 baravar ko'p (nazorat tarafkashligi). Sizdan guruhlar bo'yicha kalibrlangan, lekin FPR/FNR tengsiz model (COMPAS bahsi) ni takrorlash, uchta mezonni birga qondirib bo'lmasligini isbotlash, qarorlar uchun tushuntirish berish va hujjat OCR'iga adversarial o'qitish murozasini EPS_TRAIN bo'yicha o'lchash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Rasm ma'lumoti | sklearn digits, 10 sinf |
| Epsilon to'ri | 0, 0.05, 0.1, 0.15, 0.2 (PGD 10 qadam) |
| EPS_TRAIN | 0.05, 0.1, 0.2 (uchta model) |
| Sezgir atribut | guruh (A=1 ulushi 0.45) |
| Bazaviy ulushlar | P(y=1|A=0)=0.30, P(y=1|A=1)=0.45 |
| Tarafkashlik | A=1 da 'oldingi hibslar' belgisi 1.4x ko'paytirilgan (nazorat); flip 0 |
| Mezonlar | kalibrlash (10 bin, Brier guruhlar bo'yicha), FPR farqi, FNR farqi, DP, EO |
| SHAP | GradientBoosting, lokal izohlar 3 shaxs uchun |

Qadamlar:

1. s1/s2 FGSM va PGD-10 jadvali; s3 ni EPS_TRAIN = 0.05, 0.1, 0.2 uchun uch marta bajaring: toza aniqlik va PGD aniqligi 4 eps da — 3 x 4 jadval va grafik; radius oshganda muroza.
2. s4 saliency uchta adversarial model uchun: korrelyatsiya o'rtachasi EPS_TRAIN bilan qanday o'zgaradi.
3. Generator: bazaviy ulushlar 0.30/0.45, 'oldingi hibslar' belgisi A=1 da 1.4x; GradientBoosting va LogisticRegression o'qiting; guruhlar bo'yicha kalibrlash (10 bin) va Brier — model kalibrlanganini ko'rsating.
4. MetricFrame: FPR, FNR, selection_rate, TPR guruhlar bo'yicha — kalibrlangan modelda FPR A=1 da sezilarli yuqori ekanini (COMPAS effekti) ko'rsating; nazorat tarafkashligini olib tashlab (1.4x -> 1.0x) farq qancha qoladi.
5. ThresholdOptimizer(EO) va (DP) qo'llang va kalibrlash buzilishini o'lchang; 'kalibrlash + EO + DP' uchala shart bazaviy ulushlar farqlanganda mos kelmasligini jadval va grafikda ko'rsating (nazariy formula bilan solishtiring: FPR/FNR va bazaviy ulush bog'liqligi).
6. s6 SHAP: 3 shaxs (yuqori xavf A=1, yuqori xavf A=0, chegaradagi holat) uchun lokal izohlar; 'oldingi hibslar' hissasi; additivlik; izohni sudya uchun tushunarli tilda yozing.
7. s7 ro'yxati (huquqni muhofaza qilish: yuqori xavf, ba'zi qo'llanishlar taqiqlangan — AI Act 5-modda) : qaysi bandlar texnik jihatdan umuman qondirib bo'lmaydi; umumiy ball va 'joriy etish mumkin emas' asoslari.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (aniqlik vs epsilon, guruhlar kesimida metrikalar, SHAP top-belgilar); Grafik(lar): mustahkamlik egri chizig'i, saliency paneli, beeswarm; To'ldirilgan etik tekshiruv ro'yxati (20 band, dalillar bilan); Xulosa (>= 150 so'z): xavf, adolatlilik mezoni tanlovi va tavsiyalar

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadvallar va grafiklar — 25, Chuqur tahlil (murozalar, mos kelmaslik, cheklovlar) — 30, Hisobot, model kartasi va tekshiruv ro'yxati sifati — 20 (jami 100)
