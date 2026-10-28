# MT6. Sun'iy intellektga asoslangan kompyuter tizimlarining xavfsizligi, ishonchliligi va etik tamoyillarini xalqaro standartlar (ISO/IEC 42001, ISO/IEC 23894, EU AI Act) asosida tahlil qilish va takomillashtirish

📄 Bosma qo'llanma (bosqichlar, kod, mashqlar, 20 variant): `MT6_laboratoriya_qollanma.pdf`
Bu mustaqil ishda talaba AI boshqaruvi (governance) vositalarini kod sifatida quradi: ISO/IEC 23894 uslubidagi xavf reyestri, EU AI Act xavf toifasini aniqlovchi qoida dvigateli, kesishgan guruhlar bo'yicha fairness audit va yumshatish, sensor ma'lumotidagi mustahkamlik testlari, tushuntiriluvchanlik tahlili, ISO/IEC 42001 asosidagi gap-tahlil hisoboti va byudjet ostida optimal takomillashtirish rejasi. Real labdan farqi: ma'lumotlar sintetik, standart talablari qisqa parafraz ko'rinishida (rasmiy matn o'rnini bosmaydi), narx va ballar shartli; natijalar o'quv maqsadidagi taxminiy baholardir, yuridik xulosa emas.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, numpy, pandas, scipy, scikit-learn, matplotlib, fairlearn, shap, psutil, pyyaml, tabulate.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/MT6

## Bosqichlar

### 1-qadam. AI tizimi xavf reyestri generatori (ISO/IEC 23894 uslubi): ehtimollik x ta'sir, qoldiq xavf, Monte Karlo

Kredit skoring tizimi profili (shaxsiy ma'lumot, avtomatik qaror, inson nazorati yo'q, tashqi API, drift) asosida 12 ta xavfdan iborat katalog baholanadi: profil modifikatorlari ehtimollik (L) va ta'sirni (I) o'zgartiradi, xavf = L x I, daraja (past/o'rta/yuqori/kritik). Nazorat choralari qoldiq xavfni hisoblaydi; xavf ishtahasi (8) bo'yicha ishlov qarori chiqadi (qabul, kamaytirish, eskalatsiya). Ekspert bahosidagi +-1 noaniqlik Monte Karlo bilan baholanadi, ikki issiqlik xaritasi (oldin/keyin) chiziladi.

Fayl: `s1_1_qadam_ai_tizimi_xavf_reyestri_generato.py` — ishga tushirish: `python3 s1_1_qadam_ai_tizimi_xavf_reyestri_generato.py`

Kutilgan natija: Jami xavf ~126 dan ~75 ga tushadi (~40%); choralari yo'q R08, R02, R10 eskalatsiyaga chiqadi; R08 uchun P(qoldiq > 8) ~0,8. Ikki 5x5 issiqlik xaritasi.

Tajriba qilib ko'ring:

- PROFILE da `inson_nazorati=True` qiling — qaysi xavflar darajasi o'zgaradi?
- R08 va R10 uchun o'z choralaringizni CONTROLS ga qo'shing va qoldiq xavfni ishtahaga tushiring.
- APPETITE ni 6 va 12 qilib ishlov qarorlari sonini taqqoslang.

### 2-qadam. EU AI Act xavf toifasini aniqlovchi qoida dvigateli (15 ssenariy, avtomatik testlar bilan)

Regulation (EU) 2024/1689 mantig'i qoidalar ketma-ketligi sifatida yoziladi: 5-modda taqiqlari (a–h bandlar, 5(1)(f) dagi tibbiy/xavfsizlik istisnosi bilan), 6(1)+I ilova, 6(2)+III ilova (8 soha), 6(3) istisnosi (profillash bo'lsa qo'llanmaydi), 50-modda shaffoflik majburiyatlari va minimal xavf; GPAI modellari (53, 51/55-moddalar) alohida belgilanadi. 15 ssenariy tasniflanadi, kutilgan javob bilan solishtiriladi, kredit skoringi uchun majburiyatlar ro'yxati va asosiy muddatlar chiqadi. Bu o'quv soddalashtirishi, yuridik xulosa emas.

Fayl: `s2_2_qadam_eu_ai_act_xavf_toifasini_aniqlov.py` — ishga tushirish: `python3 s2_2_qadam_eu_ai_act_xavf_toifasini_aniqlov.py`

Kutilgan natija: 15/15 ssenariy kutilgan toifaga mos; kredit skoringi — YUQORI XAVF, 12 majburiyat (27-modda FRIA bilan); chatbot ortidagi katta model — qo'shimcha GPAI tizimli xavf belgisi.

Tajriba qilib ko'ring:

- O'z tashkilotingizdagi 3 ta AI tizimini SCEN ga qo'shing va kutilgan toifani oldindan yozib, dvigatel bilan solishtiring.
- 6(3) istisnosiga «inson baholashini tayyorlovchi» holatni qo'shing.
- Har toifa uchun 99-modda jarima chegaralarini chiqaradigan funksiya yozing.

### 3-qadam. Fairness audit: kesishgan guruhlar (jins x yosh), 4/5 qoidasi va yumshatish usullari

Ishga qabul (III-4 soha) sintetik ma'lumotida tarixiy qarorlar «ish tanaffusi» (jins vakili) va >45 yosh bo'yicha tarafkash. Fairlearn MetricFrame ikki sezgir atribut bilan 6 kesishgan guruh uchun tanlov ulushi, TPR, FPR beradi; disparate impact (eng kam/eng ko'p tanlov) 4/5 qoidasi bilan tekshiriladi. To'rt variant taqqoslanadi: bazaviy, vakil belgilarsiz, Kamiran–Calders qayta tortish (qo'lda) va ExponentiatedGradient (DemographicParity, kesishgan guruh bo'yicha).

Fayl: `s3_3_qadam_fairness_audit_kesishgan_guruhla.py` — ishga tushirish: `python3 s3_3_qadam_fairness_audit_kesishgan_guruhla.py`

Kutilgan natija: Bazaviy DI ~0,5 (4/5 qoidasidan o'tmaydi); vakil belgilarni olib tashlash DI ni yaxshilamaydi, qayta tortish ~0,7, ExpGrad ~0,77 ga ko'taradi, aniqlik 1–4 bandga pasayadi.

Tajriba qilib ko'ring:

- ExpGrad `difference_bound` ni 0.01 qilib 4/5 qoidasidan o'tishga harakat qiling — aniqlik qancha pasayadi?
- `EqualizedOdds` cheklovini qo'shib EO_diff ni solishtiring.
- Guruhlar bo'yicha DI uchun 200 marta bootstrap qilib 95% ishonch oralig'ini hisoblang.

### 4-qadam. Mustahkamlik testi: shovqin, sensor uzilishi, kalibrlash drifti va adversarial hujum

Zavod uskunasi holatini 6 sensor bo'yicha tasniflovchi uch model (LogReg, RandomForest, PyTorch MLP) ISO/IEC TR 24029-1 ruhidagi empirik testlardan o'tadi: Gauss shovqini, har bir sensorning uzilishi (o'rtacha qiymat bilan to'ldirish), harorat sensorining +2/5/10 C drifti va FGSM (MLP va LR uchun oq quti — LR gradienti analitik, RF uchun MLP'dan ko'chirilgan hujum). Natijalar oldindan belgilangan qabul mezonlari bilan OK/XATO jadvaliga keltiriladi.

Fayl: `s4_4_qadam_mustahkamlik_testi_shovqin_senso.py` — ishga tushirish: `python3 s4_4_qadam_mustahkamlik_testi_shovqin_senso.py`

Kutilgan natija: Toza aniqlik ~0,95–0,96; harorat sensori uzilishi ~11–13 p.p. pasayish beradi (mezon 10 — XATO); +10 C drift RF ni ~0,65 ga tushiradi; FGSM eps=0,2 da MLP mezondan o'tmaydi, ko'chirilgan hujum RF ga kuchsizroq.

Tajriba qilib ko'ring:

- Sensor uzilishiga chidamlilik uchun o'qitishda tasodifiy sensorni 0 qilib (dropout augmentatsiya) qayta o'qiting.
- Harorat driftini aniqlovchi oddiy monitor (o'rtacha siljish > 2 sigma) yozing.
- MLP ni FGSM eps=0.1 bilan adversarial o'qitib mezon jadvalini qayta chiqaring.

### 5-qadam. Tushuntiriluvchanlik: permutatsion muhimlik vs SHAP, barqarorlik, surrogat daraxt, kontrfaktual

Mikrokredit sintetik ma'lumotida (7 belgi, shundan bittasi ataylab ta'sirsiz) GradientBoosting o'qitiladi. Global muhimlik ikki usulda (permutatsion va SHAP TreeExplainer) hisoblanib, Kendall tau bilan solishtiriladi; 5 bootstrap qayta o'qitishda SHAP top-3 barqarorligi tekshiriladi; chuqurligi 3 bo'lgan surrogat daraxtning sodiqligi (fidelity) va qoidalari chiqadi. Rad etilgan ariza uchun «tushuntirish xati» (eng salbiy 3 omil) va bitta omilli kontrfaktual izohlar yaratiladi.

Fayl: `s5_5_qadam_tushuntiriluvchanlik_permutatsio.py` — ishga tushirish: `python3 s5_5_qadam_tushuntiriluvchanlik_permutatsio.py`

Kutilgan natija: Aniqlik ~0,81; ikki usul top-4 da kelishadi (tau ~0,6), ta'sirsiz belgi ikkalasida ham eng past; bootstrap top-3 to'plami ~100% barqaror; surrogat sodiqligi ~0,89.

Tajriba qilib ko'ring:

- Kontrfaktual qidiruvga 2 omilni birga o'zgartirishni qo'shing va eng kichik o'zgarishni toping.
- Surrogat daraxt chuqurligini 2..6 qilib sodiqlik egri chizig'ini chizing.
- `yosh` belgisini olib tashlab, model va izohlar qanday o'zgarishini yozing.

### 6-qadam. Muvofiqlik tekshiruv ro'yxati va gap-tahlil hisoboti (ISO/IEC 42001, 23894, EU AI Act)

22 talab (ISO/IEC 42001 4–10 bandlari va A.2–A.10 nazorat sohalari, ISO/IEC 23894 xavf jarayoni, EU AI Act 12, 14, 15, 17, 73-moddalari) qisqa parafraz ko'rinishida beriladi. Tashkilotning joriy yetukligi (0–5) va dalili CURRENT lug'atidan olinadi; dalilsiz yuqori baho avtomatik pasaytiriladi. Xavf toifasiga qarab maqsadli yetuklik belgilanadi, gap va ustuvorlik (gap x vazn) hisoblanib, Markdown hisobot (qamrov, top-8 bo'shliq, xulosa) va standartlar orasidagi bog'lanishlar chiqadi.

Fayl: `s6_6_qadam_muvofiqlik_tekshiruv_royxati_va_.py` — ishga tushirish: `python3 s6_6_qadam_muvofiqlik_tekshiruv_royxati_va_.py`

Kutilgan natija: Umumiy muvofiqlik indeksi ~42/100; eng ustuvor bo'shliqlar — ta'sirni baholash (A.5) va jiddiy hodisalar tartibi (73-modda); 7 ta majburiy talab bajarilmagan -> «joriy etishdan oldin yopish SHART».

Tajriba qilib ko'ring:

- CURRENT ni o'z tashkilotingiz (yoki tanish korxona) uchun to'ldiring va indeksni qayta hisoblang.
- RISK_CLASS = "minimal" qiling — maqsadlar va xulosa qanday o'zgaradi?
- Hisobotni faylga emas, `print` orqali to'liq Markdown qilib 4-bo'lim (tavsiyalar) qo'shing.

### 7-qadam. Takomillashtirish rejasi: byudjet ostida optimal choralar, bog'liqliklar bilan Gantt va PDCA

13 chora (narx, muddat, qaysi gap va xavflarni yopishi, bog'liqliklar) orasidan 250 mln so'm (shartli) byudjetda eng katta qiymat beradigan to'plam to'liq qidiruv (8192 kombinatsiya) bilan topiladi va ochko'z usul bilan taqqoslanadi. Tanlangan choralar 2 jamoa va bog'liqliklarni hisobga olib haftalarga joylashtiriladi (ro'yxatli rejalashtirish), PDCA bosqichiga ajratiladi; Markdown yo'l xaritasi, vaznli gap va xavf «oldin/keyin», KPI ro'yxati va Gantt diagrammasi chiqadi.

Fayl: `s7_7_qadam_takomillashtirish_rejasi_byudjet.py` — ishga tushirish: `python3 s7_7_qadam_takomillashtirish_rejasi_byudjet.py`

Kutilgan natija: Optimal to'plam qiymati ~94,5, ochko'z usul ~77,5 (bog'liqlik tufayli M2 ni o'tkazib yuboradi); reja ~14 hafta, vaznli gap ~80% yopiladi, R01 va R07 ishtahadan yuqori qoladi.

Tajriba qilib ko'ring:

- BUDGET ni 150 va 350 qilib qiymat-byudjet egri chizig'ini chizing.
- TEAM=1 va 3 bilan umumiy muddatni solishtiring.
- R07 ni ishtahaga tushirish majburiy cheklov sifatida qo'shing (M11 doim tanlansin) va optimumni qayta toping.

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
|---|---|---|---|
| 1 | **O'z tizimingiz xavf reyestri** — 1-qadam PROFILE va CATALOG ni o'zingiz tanlagan AI tizimi (masalan, universitet qabul tizimi) uchun moslang: kamida 12 xavf, har biriga kamida bitta chora. Reyestr jadvali, ikki issiqlik xaritasi va ishtahadan yuqori qolgan xavflar ro'yxatini hisobotga kiriting. | oson | 5 |
| 2 | **Ssenariylarni tasniflash** — 2-qadam dvigateliga o'zbekistonlik tashkilotlar uchun real ko'rinishdagi 8 ta yangi ssenariy qo'shing (har toifadan kamida bittasi), kutilgan toifani oldindan yozing va dvigatel natijasi bilan solishtiring. Mos kelmagan holatlar uchun qoidani tuzating va sababini 2–3 jumlada izohlang. | oson | 5 |
| 3 | **Gap-tahlilni to'ldirish** — 6-qadam CURRENT lug'atini tanlangan tashkilot uchun dalillar bilan to'ldiring, RISK_CLASS ni 2-qadam natijasiga moslang va Markdown hisobotni chiqaring. Indeks qiymati va top-5 bo'shliqni izohlang. | oson | 5 |
| 4 | **Fairness: mezonlar to'qnashuvi** — 3-qadamda ExponentiatedGradient ni DemographicParity va EqualizedOdds bilan, difference_bound = 0.01, 0.03, 0.05 uchun o'qiting. 6 model uchun aniqlik, DI, EO_diff jadvalini va aniqlik–DI nuqtali grafigini tuzing; ishga qabul konteksti uchun qaysi mezon to'g'riroq ekanini asoslang. | o'rta | 10 |
| 5 | **Mustahkamlikni oshirish** — 4-qadam modellarini (a) sensor-dropout augmentatsiyasi, (b) shovqinli augmentatsiya bilan qayta o'qiting. Qabul mezonlari jadvalini uch holat (bazaviy, a, b) uchun takrorlang va toza aniqlik bilan muroza haqida xulosa yozing. | o'rta | 10 |
| 6 | **Tushuntirish sodiqligi** — 5-qadamda SHAP bo'yicha eng muhim k = 1..5 belgini test to'plamida o'rtacha qiymat bilan almashtirib (deletion), aniqlik pasayishini o'lchang; xuddi shuni permutatsion reyting va tasodifiy tartib bilan takrorlang. Uchta egri chiziq va qaysi usul sodiqroq ekanini ko'rsating. | o'rta | 10 |
| 7 | **Xavf reyestri va reja integratsiyasi** — 1 va 7-qadamlarni birlashtiring: 7-qadamdagi RISK lug'ati 1-qadam reyestridan avtomatik olinsin, tanlangan choralar esa reyestrdagi qoldiq xavfni yangilasin. «Oldin/keyin» issiqlik xaritalarini va qoldiq xavflar yig'indisini byudjet 150/250/350 uchun taqqoslang. | o'rta | 10 |
| 8 | **To'liq muvofiqlik paketi generatori** — 2, 3, 4, 6-qadam funksiyalarini bitta skriptga birlashtiring: dvigatel toifani aniqlasin, fairness va mustahkamlik testlari o'tkazilsin, natijalar avtomatik ravishda CURRENT dagi tegishli bandlarga (A.7, E15, A5) yetuklik bahosi sifatida yozilsin va yakuniy Markdown gap-hisobot chiqsin. Umumiy ish vaqti 60 s dan oshmasin. | qiyin | 15 |
| 9 | **Ko'p mezonli rejalashtirish** — 7-qadam optimizatsiyasini ikki mezonli qiling: qiymat va umumiy muddat (hafta). Byudjet 100–400 oralig'ida Pareto-optimal rejalarni toping, Pareto frontini chizing va «tez, lekin qimmat» hamda «arzon, lekin uzoq» rejalarni taqqoslab tavsiya bering. Bog'liqliklar va jamoa cheklovi saqlansin. | qiyin | 15 |
| 10 | **Model kartasi va FRIA qoralamasi** — 1–7 qadam natijalaridan foydalanib, kredit skoring tizimi uchun avtomatik Markdown hujjat yarating: model kartasi (maqsad, ma'lumot, aniqlik, fairness, mustahkamlik, tushuntirish usuli, cheklovlar) va asosiy huquqlarga ta'sirni baholash (FRIA) qoralamasi (ta'sir ko'radigan guruhlar, xavflar, inson nazorati, shikoyat tartibi). Hujjatdagi har raqam skript ichida hisoblangan bo'lsin. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Bank: iste'mol krediti skoringi — to'liq muvofiqlik tahlili (oson)

Soha: Moliya (bank)

Bank yangi kredit skoring modelini joriy etmoqchi. Regulyator va xalqaro hamkorlar ISO/IEC 42001 asosida boshqaruv tizimi va EU AI Act yuqori xavf talablariga moslikni so'ramoqda. Xavf reyestri, toifa, fairness (jins, hudud), mustahkamlik va gap-tahlil asosida takomillashtirish rejasi kerak.

| Parametr | Qiymat |
|---|---|
| AI tizimi | kredit skoring (GradientBoosting) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-5) |
| Sezgir atributlar | jins, hudud |
| Mustahkamlik ma'lumoti | sintetik ariza belgilari + shovqin |
| Byudjet (shartli, mln so'm) | 250 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 11 |

Qadamlar:

1. s1 xavf reyestrini «kredit skoring (GradientBoosting)» uchun moslang: kamida 11 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-5)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (jins, hudud) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
5. s7 rejasini 250 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 2-variant. Ishga qabul platformasi: rezyume saralash (oson)

Soha: Bandlik (HR)

HR-texnologiya startapi rezyumelarni avtomatik saralaydi. Mijoz kompaniyalar EU bozorida ishlaydi. Yosh va jins bo'yicha kesishgan fairness audit, 4/5 qoidasi, vakil belgilar (bitirgan yil, ish tanaffusi) tahlili va inson nazorati choralari talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | rezyume skoringi (LogReg) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-4) |
| Sezgir atributlar | jins x yosh |
| Mustahkamlik ma'lumoti | rezyume belgilari, yorliq tarafkashligi |
| Byudjet (shartli, mln so'm) | 200 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 12 |

Qadamlar:

1. s1 xavf reyestrini «rezyume skoringi (LogReg)» uchun moslang: kamida 12 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-4)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (jins x yosh) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
5. s7 rejasini 200 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 3-variant. Universitet: kirish imtihoni insholarini avtomatik baholash (oson)

Soha: Ta'lim

Universitet insholarni AI bilan dastlabki baholaydi. Talabalar ona tili (o'zbek/rus maktabi) bo'yicha adolatsizlikdan shikoyat qilmoqda. Toifa aniqlash, fairness audit, tushuntirish (qaysi belgilar ballga ta'sir qiladi) va apellyatsiya tartibi rejasi kerak.

| Parametr | Qiymat |
|---|---|
| AI tizimi | insho baholash (regressiya -> o'tdi/o'tmadi) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-3) |
| Sezgir atributlar | maktab tili, jins |
| Mustahkamlik ma'lumoti | sintetik insho belgilari |
| Byudjet (shartli, mln so'm) | 180 |
| Xavf ishtahasi (L x I) | 6 |
| Reyestrdagi xavflar | 13 |

Qadamlar:

1. s1 xavf reyestrini «insho baholash (regressiya -> o'tdi/o'tmadi)» uchun moslang: kamida 13 xavf, profil modifikatorlari, choralar va ishtaha = 6; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-3)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (maktab tili, jins) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
5. s7 rejasini 180 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 4-variant. Shahar: jamoat transportida yo'lovchi oqimini bashorat qilish (oson)

Soha: Shahar (transport)

Shahar hokimligi avtobuslar jadvalini AI bashorati asosida o'zgartiradi. Shaxsiy ma'lumot ishlatilmaydi, lekin sensor (turniket) uzilishi va drift muammosi bor. Toifani (minimal bo'lishi kutiladi) asoslash, mustahkamlik testi va yengil boshqaruv rejasi talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | oqim bashorati (RandomForest) |
| Kutilgan EU AI Act toifasi | minimal |
| Sezgir atributlar | tuman (hudud) |
| Mustahkamlik ma'lumoti | turniket sensorlari, uzilish va drift |
| Byudjet (shartli, mln so'm) | 100 |
| Xavf ishtahasi (L x I) | 10 |
| Reyestrdagi xavflar | 14 |

Qadamlar:

1. s1 xavf reyestrini «oqim bashorati (RandomForest)» uchun moslang: kamida 14 xavf, profil modifikatorlari, choralar va ishtaha = 10; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (minimal) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (tuman (hudud)) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
5. s7 rejasini 100 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 5-variant. Bank sayti: mijozlar chatboti (oson)

Soha: Moliya (mijozlarga xizmat)

Bank saytida LLM asosidagi chatbot ishlaydi. 50-modda shaffoflik majburiyatlari, javob xatolari va maxfiylik xavflari baholanishi kerak. Xavf reyestrida uchinchi tomon modeli (API) xavfi va hodisalar tartibi alohida ko'rib chiqiladi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | LLM chatbot (tashqi API) |
| Kutilgan EU AI Act toifasi | cheklangan (50-modda) |
| Sezgir atributlar | til (o'zbek/rus) |
| Mustahkamlik ma'lumoti | savol matni belgilari + shovqin |
| Byudjet (shartli, mln so'm) | 120 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 10 |

Qadamlar:

1. s1 xavf reyestrini «LLM chatbot (tashqi API)» uchun moslang: kamida 10 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (cheklangan (50-modda)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (til (o'zbek/rus)) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
5. s7 rejasini 120 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 6-variant. Zavod: uskuna nosozligini bashorat qilish (predictive maintenance) (oson)

Soha: Sanoat

Zavodda 8 sensorli uskuna holati AI bilan kuzatiladi. Noto'g'ri «normal» bashorati avariyaga olib kelishi mumkin. Mustahkamlik (sensor uzilishi, harorat drifti, FGSM), xavf reyestri va monitoring choralariga asoslangan reja kerak; toifa mahsulot xavfsizligiga bog'liqligini tekshiring.

| Parametr | Qiymat |
|---|---|
| AI tizimi | holat tasnifi (MLP) |
| Kutilgan EU AI Act toifasi | minimal yoki yuqori (I ilova, mashina xavfsizlik komponenti bo'lsa) |
| Sezgir atributlar | smena (kunduzgi/tungi) |
| Mustahkamlik ma'lumoti | 8 sensor, uzilish/drift |
| Byudjet (shartli, mln so'm) | 150 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 11 |

Qadamlar:

1. s1 xavf reyestrini «holat tasnifi (MLP)» uchun moslang: kamida 11 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (minimal yoki yuqori (I ilova, mashina xavfsizlik komponenti bo'lsa)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (smena (kunduzgi/tungi)) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
5. s7 rejasini 150 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 7-variant. Media: deepfake va sintetik kontent generatori (oson)

Soha: Media

Media kompaniya reklama uchun sintetik video va ovoz yaratadi. 50(2) va 50(4) belgilash majburiyatlari, suiiste'mol xavfi va reputatsion xavflar baholanadi. Muvofiqlik ro'yxati va belgilash (watermark) choralarini o'z ichiga olgan reja talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | generativ model (tashqi GPAI) |
| Kutilgan EU AI Act toifasi | cheklangan (50-modda) + GPAI provayder majburiyatlari |
| Sezgir atributlar | — |
| Mustahkamlik ma'lumoti | sintetik kontent detektori belgilari |
| Byudjet (shartli, mln so'm) | 120 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 12 |

Qadamlar:

1. s1 xavf reyestrini «generativ model (tashqi GPAI)» uchun moslang: kamida 12 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (cheklangan (50-modda) + GPAI provayder majburiyatlari) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (—) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
5. s7 rejasini 120 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 8-variant. Sug'urta: hayot va sog'liq sug'urtasi narxlash (o'rta)

Soha: Sug'urta

Sug'urta kompaniyasi hayot va sog'liq polisi narxini AI bilan belgilaydi (III-5(c)). Yosh bo'yicha haqiqiy xavf farqi va noqonuniy kamsitishni ajratish, FRIA (27-modda), tushuntirish xati va kontrfaktual izohlar talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | narxlash (GradientBoosting) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-5) |
| Sezgir atributlar | yosh x jins |
| Mustahkamlik ma'lumoti | bemor yozuvlari (sintetik) |
| Byudjet (shartli, mln so'm) | 220 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 13 |

Qadamlar:

1. s1 xavf reyestrini «narxlash (GradientBoosting)» uchun moslang: kamida 13 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-5)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (yosh x jins) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «bemor yozuvlari (sintetik)» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 220 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 9-variant. Kasalxona: shoshilinch bo'limda triaj ustuvorligi (o'rta)

Soha: Tibbiyot

Shoshilinch bo'limda bemor ustuvorligini AI taklif qiladi (III-5(d)). Xato past baho hayotga xavf. Kesishgan fairness (yosh, jins), sensor (hayotiy ko'rsatkichlar) uzilishiga chidamlilik, inson nazorati va jiddiy hodisalar (73-modda) tartibi bo'yicha reja kerak.

| Parametr | Qiymat |
|---|---|
| AI tizimi | triaj tasnifi (LogReg + RF) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-5) |
| Sezgir atributlar | yosh x jins |
| Mustahkamlik ma'lumoti | hayotiy ko'rsatkichlar sensorlari |
| Byudjet (shartli, mln so'm) | 260 |
| Xavf ishtahasi (L x I) | 6 |
| Reyestrdagi xavflar | 14 |

Qadamlar:

1. s1 xavf reyestrini «triaj tasnifi (LogReg + RF)» uchun moslang: kamida 14 xavf, profil modifikatorlari, choralar va ishtaha = 6; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-5)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (yosh x jins) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «hayotiy ko'rsatkichlar sensorlari» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 260 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 10-variant. Chegara nazorati: hujjat tekshiruvi va xavf profillash (o'rta)

Soha: Migratsiya

Chegara punktida hujjatlar va yo'lovchi ma'lumotlari asosida qo'shimcha tekshiruv taklif qilinadi (III-7). Fuqarolik bo'yicha FPR tengligi, profillash xavfi, kiberxavfsizlik va log yuritish talablari alohida tahlil qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | xavf profillash (GradientBoosting) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-7) |
| Sezgir atributlar | fuqarolik, jins |
| Mustahkamlik ma'lumoti | hujjat belgilari + adversarial |
| Byudjet (shartli, mln so'm) | 240 |
| Xavf ishtahasi (L x I) | 6 |
| Reyestrdagi xavflar | 10 |

Qadamlar:

1. s1 xavf reyestrini «xavf profillash (GradientBoosting)» uchun moslang: kamida 10 xavf, profil modifikatorlari, choralar va ishtaha = 6; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-7)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (fuqarolik, jins) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «hujjat belgilari + adversarial» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 240 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 11-variant. Elektr tarmog'i: yuklamani avtomatik boshqarish (o'rta)

Soha: Energetika

Hududiy elektr tarmog'i yuklamani AI bilan taqsimlaydi (III-2, kritik infratuzilma). Asosiy xavflar — mustahkamlik, kiberhujum, drift va uchinchi tomon SCADA integratsiyasi. Xavf reyestri, mustahkamlik testlari va 14-modda bo'yicha inson nazorati rejasi kerak.

| Parametr | Qiymat |
|---|---|
| AI tizimi | yuklama bashorati va boshqaruv (MLP) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-2) |
| Sezgir atributlar | hudud (shahar/qishloq) |
| Mustahkamlik ma'lumoti | sensor telemetriya, drift, FGSM |
| Byudjet (shartli, mln so'm) | 300 |
| Xavf ishtahasi (L x I) | 6 |
| Reyestrdagi xavflar | 11 |

Qadamlar:

1. s1 xavf reyestrini «yuklama bashorati va boshqaruv (MLP)» uchun moslang: kamida 11 xavf, profil modifikatorlari, choralar va ishtaha = 6; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-2)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (hudud (shahar/qishloq)) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «sensor telemetriya, drift, FGSM» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 300 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 12-variant. Davlat: ijtimoiy nafaqa tayinlash ustuvorligi (o'rta)

Soha: Davlat xizmatlari

Ijtimoiy himoya agentligi nafaqa arizalarini AI bilan saralaydi (III-5(a)). Oila kattaligi va hudud bo'yicha adolatlilik, rad etilgan fuqaroga tushuntirish (86-modda) va shikoyat tartibi talab qilinadi; davlat organi sifatida FRIA majburiy.

| Parametr | Qiymat |
|---|---|
| AI tizimi | ariza ustuvorligi (LogReg) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-5) |
| Sezgir atributlar | hudud x oila kattaligi |
| Mustahkamlik ma'lumoti | ariza belgilari (sintetik) |
| Byudjet (shartli, mln so'm) | 200 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 12 |

Qadamlar:

1. s1 xavf reyestrini «ariza ustuvorligi (LogReg)» uchun moslang: kamida 12 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-5)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (hudud x oila kattaligi) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «ariza belgilari (sintetik)» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 200 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 13-variant. Telekom: abonent ketishini (churn) bashorat qilish (o'rta)

Soha: Telekom

Operator ketish ehtimoli yuqori abonentlarga chegirma taklif qiladi. Toifa minimal bo'lishi kutiladi, lekin hudud va til bo'yicha taklif ulushi farqi reputatsion xavf. Fairness audit, SHAP tushuntirish va yengil gap-tahlil (ixtiyoriy kodeks, 95-modda) talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | churn modeli (GradientBoosting) |
| Kutilgan EU AI Act toifasi | minimal |
| Sezgir atributlar | hudud, til |
| Mustahkamlik ma'lumoti | abonent faolligi loglari |
| Byudjet (shartli, mln so'm) | 90 |
| Xavf ishtahasi (L x I) | 10 |
| Reyestrdagi xavflar | 13 |

Qadamlar:

1. s1 xavf reyestrini «churn modeli (GradientBoosting)» uchun moslang: kamida 13 xavf, profil modifikatorlari, choralar va ishtaha = 10; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (minimal) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (hudud, til) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «abonent faolligi loglari» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 90 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 14-variant. Maktab: o'quvchi diqqatini kamera orqali kuzatish loyihasi (o'rta)

Soha: Ta'lim (maktab)

Xususiy maktab darsda o'quvchilar hissiyoti va diqqatini kamera bilan aniqlashni taklif qilmoqda. Dvigatel 5(1)(f) taqiqini ko'rsatishi, loyiha qanday o'zgartirilsa (masalan, anonim sinf darajasidagi statistika) ruxsat etilishi tahlil qilinishi va muqobil reja tuzilishi kerak.

| Parametr | Qiymat |
|---|---|
| AI tizimi | hissiyot aniqlash (kamera) |
| Kutilgan EU AI Act toifasi | taqiqlangan (5(1)(f)) |
| Sezgir atributlar | yosh, jins |
| Mustahkamlik ma'lumoti | kamera kadrlaridan belgilar (sintetik) |
| Byudjet (shartli, mln so'm) | 80 |
| Xavf ishtahasi (L x I) | 4 |
| Reyestrdagi xavflar | 14 |

Qadamlar:

1. s1 xavf reyestrini «hissiyot aniqlash (kamera)» uchun moslang: kamida 14 xavf, profil modifikatorlari, choralar va ishtaha = 4; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (taqiqlangan (5(1)(f))) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (yosh, jins) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «kamera kadrlaridan belgilar (sintetik)» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 80 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 15-variant. Politsiya: jinoyatchilik xaritasi va patrul rejalashtirish (o'rta)

Soha: Huquqni muhofaza qilish

Ichki ishlar organi hudud bo'yicha patrul ustuvorligini AI bilan belgilaydi. Shaxs profillashi (5(1)(d)) va hudud darajasidagi bashorat (III-6) farqini qoida dvigatelida aniq ajratish, fikr-mulohaza halqasi xavfi va FPR tengligi tahlil qilinishi kerak.

| Parametr | Qiymat |
|---|---|
| AI tizimi | hudud xavf bashorati (RF) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-6), shaxs profillashi bo'lsa taqiqlangan |
| Sezgir atributlar | hudud |
| Mustahkamlik ma'lumoti | hodisa statistikasi (sintetik), drift |
| Byudjet (shartli, mln so'm) | 220 |
| Xavf ishtahasi (L x I) | 6 |
| Reyestrdagi xavflar | 10 |

Qadamlar:

1. s1 xavf reyestrini «hudud xavf bashorati (RF)» uchun moslang: kamida 10 xavf, profil modifikatorlari, choralar va ishtaha = 6; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-6), shaxs profillashi bo'lsa taqiqlangan) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (hudud) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «hodisa statistikasi (sintetik), drift» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 220 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin».

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 16-variant. Fintech: telefon ma'lumotlari asosida mikrokredit — to'liq audit paketi (qiyin)

Soha: Moliya (fintech)

Mobil ilova telefon metama'lumotlari (qurilma narxi, ilovalar soni) asosida kredit beradi. Bu belgilar daromad va hudud vakili bo'lishi mumkin. Vakil belgilarni SHAP bilan fosh qilish, fairness + mustahkamlik + tushuntirishni bitta skriptda birlashtirib, avtomatik gap-hisobot generatori yaratish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | skoring (GradientBoosting) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-5) |
| Sezgir atributlar | hudud x jins |
| Mustahkamlik ma'lumoti | telefon metama'lumotlari + adversarial |
| Byudjet (shartli, mln so'm) | 260 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 11 |

Qadamlar:

1. s1 xavf reyestrini «skoring (GradientBoosting)» uchun moslang: kamida 11 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-5)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (hudud x jins) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «telefon metama'lumotlari + adversarial» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 260 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin»; ssenariydagi qo'shimcha talabni (avtomatik hisobot / Pareto / model kartasi + FRIA) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 17-variant. Klinika: rentgen tasvirlarida AI yordamchi (tibbiy qurilma) (qiyin)

Soha: Tibbiyot (diagnostika)

Klinika rentgen tasvirlarida patologiyani belgilovchi AI ni joriy etmoqda; u MDR bo'yicha tibbiy qurilma (I ilova). Mustahkamlik (shovqin, kontrast drifti, FGSM) sinovi digits ma'lumotida simulyatsiya qilinadi; texnik hujjat (11-modda), bozordan keyingi monitoring (72) va model kartasi generatori talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | tasvir tasnifi (kichik CNN/MLP, digits simulyatsiya) |
| Kutilgan EU AI Act toifasi | yuqori xavf (6(1) + I ilova) |
| Sezgir atributlar | yosh guruhi |
| Mustahkamlik ma'lumoti | sklearn digits + shovqin/FGSM |
| Byudjet (shartli, mln so'm) | 300 |
| Xavf ishtahasi (L x I) | 6 |
| Reyestrdagi xavflar | 12 |

Qadamlar:

1. s1 xavf reyestrini «tasvir tasnifi (kichik CNN/MLP, digits simulyatsiya)» uchun moslang: kamida 12 xavf, profil modifikatorlari, choralar va ishtaha = 6; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (6(1) + I ilova)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (yosh guruhi) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «sklearn digits + shovqin/FGSM» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 300 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin»; ssenariydagi qo'shimcha talabni (avtomatik hisobot / Pareto / model kartasi + FRIA) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 18-variant. Taksi platformasi: haydovchi reytingi va buyurtma taqsimoti (qiyin)

Soha: Transport (platforma)

Platforma haydovchilarga buyurtmani reyting bo'yicha taqsimlaydi; bu ish sharoitiga ta'sir (III-4: ishchilarga vazifa taqsimoti). Jins va yosh kesishmasi, ko'p mezonli takomillashtirish rejasi (qiymat va muddat bo'yicha Pareto) talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | taqsimot modeli (LogReg) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-4) |
| Sezgir atributlar | jins x yosh |
| Mustahkamlik ma'lumoti | buyurtma loglari (sintetik) |
| Byudjet (shartli, mln so'm) | 200 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 13 |

Qadamlar:

1. s1 xavf reyestrini «taqsimot modeli (LogReg)» uchun moslang: kamida 13 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-4)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (jins x yosh) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «buyurtma loglari (sintetik)» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 200 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin»; ssenariydagi qo'shimcha talabni (avtomatik hisobot / Pareto / model kartasi + FRIA) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 19-variant. Katta til modeli provayderi: GPAI muvofiqlik rejasi (qiyin)

Soha: IT (AI provayder)

Kompaniya umumiy maqsadli katta til modelini EU bozoriga chiqarmoqda; o'qitish hisob-kitobi 1e25 FLOP dan oshishi mumkin. 53 va 55-moddalar majburiyatlari (texnik hujjat, mualliflik huquqi siyosati, tizimli xavf baholash, insidentlar), ISO/IEC 42001 gap-tahlili va byudjetli reja talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | GPAI model (simulyatsiya) |
| Kutilgan EU AI Act toifasi | GPAI + tizimli xavf (51/55-moddalar) |
| Sezgir atributlar | til |
| Mustahkamlik ma'lumoti | matn sinovlari (sintetik, injection to'plami) |
| Byudjet (shartli, mln so'm) | 400 |
| Xavf ishtahasi (L x I) | 8 |
| Reyestrdagi xavflar | 14 |

Qadamlar:

1. s1 xavf reyestrini «GPAI model (simulyatsiya)» uchun moslang: kamida 14 xavf, profil modifikatorlari, choralar va ishtaha = 8; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (GPAI + tizimli xavf (51/55-moddalar)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (til) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «matn sinovlari (sintetik, injection to'plami)» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 400 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin»; ssenariydagi qo'shimcha talabni (avtomatik hisobot / Pareto / model kartasi + FRIA) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)

## 20-variant. Sud: qamoqdan ozod qilishda takroriy jinoyat xavfini baholash (qiyin)

Soha: Adliya

Sudyaga takroriy jinoyat xavfi bahosi taqdim etiladi (III-8 va III-6 bilan chegaradosh). COMPAS uslubidagi sintetik tizimda kalibrlash va xatolik tengligi mos kelmasligini ko'rsatish, model kartasi + FRIA qoralamasini avtomatik yaratish va «joriy etish mumkin/mumkin emas» xulosasi talab qilinadi.

| Parametr | Qiymat |
|---|---|
| AI tizimi | xavf skori (LogReg) |
| Kutilgan EU AI Act toifasi | yuqori xavf (III-8/III-6) |
| Sezgir atributlar | irq o'rniga sintetik guruh, yosh |
| Mustahkamlik ma'lumoti | ish yozuvlari (sintetik) |
| Byudjet (shartli, mln so'm) | 280 |
| Xavf ishtahasi (L x I) | 4 |
| Reyestrdagi xavflar | 10 |

Qadamlar:

1. s1 xavf reyestrini «xavf skori (LogReg)» uchun moslang: kamida 10 xavf, profil modifikatorlari, choralar va ishtaha = 4; ikki issiqlik xaritasi va Monte Karlo natijasi.
2. s2 dvigateliga o'z tizimingiz va 3 ta o'xshash ssenariyni qo'shing; kutilgan toifa (yuqori xavf (III-8/III-6)) bilan solishtiring va majburiyatlar ro'yxatini chiqaring.
3. s3 fairness auditini sintetik ma'lumotda (irq o'rniga sintetik guruh, yosh) bajaring: guruhlar jadvali, DI (4/5 qoidasi), kamida 2 yumshatish usuli.
4. s4 mustahkamlik testlarini «ish yozuvlari (sintetik)» uchun moslang; qabul mezonlarini o'zingiz asoslab belgilang va OK/XATO jadvalini chiqaring.
5. s5 bo'yicha permutatsion muhimlik va SHAP ni solishtiring, bitta salbiy qaror uchun tushuntirish xati va kontrfaktual yozing.
6. s6 gap-tahlilini tashkilotingiz holati bo'yicha (CURRENT) to'ldiring va Markdown hisobotni chiqaring.
7. s7 rejasini 280 mln so'm byudjet bilan tuzing: optimal va ochko'z tanlov, Gantt, gap va xavf «oldin/keyin»; ssenariydagi qo'shimcha talabni (avtomatik hisobot / Pareto / model kartasi + FRIA) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Xavf reyestri va gap-tahlil hisoboti (Markdown); Natijalar jadvallari (toifa, fairness, mustahkamlik, muhimlik); Grafiklar (issiqlik xaritasi, muroza, Gantt); Xulosa va takomillashtirish takliflari (>= 200 so'z)

Baholash: To'g'ri ishlaydigan kod — 25, Standartlarga (ISO/IEC 42001, 23894, EU AI Act) to'g'ri tayanish — 25, O'lchovlar va jadvallar — 20, Takliflar va reja sifati — 20, Hisobot rasmiylashtirilishi — 10 (jami 100)
