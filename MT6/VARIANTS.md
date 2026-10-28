# MT6. Mustaqil bajarish uchun variantlar (20)

Variant raqami = talabaning jurnal raqami (21 va undan katta bo'lsa, 20 ga bo'lgandagi qoldiq; qoldiq 0 bo'lsa 20-variant). Mustaqil ishning s1–s7 qadamlari kodi asos qilib olinadi: tizim profili, xavf katalogi, ssenariylar, sintetik ma'lumot generatori, qabul mezonlari, talablar holati va choralar ro'yxati o'z variantingizga almashtiriladi. Topshiriq .py fayllar + hisobot (PDF yoki Markdown) ko'rinishida; standart talablari parafraz qilinadi, natijalar o'quv maqsadidagi taxminiy baho, yuridik xulosa emas.

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
