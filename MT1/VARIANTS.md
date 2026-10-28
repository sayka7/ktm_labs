# MT1. Mustaqil bajarish uchun variantlar (20)

Variant raqami = jurnal (guruh ro'yxati) raqamingiz; 20 dan katta bo'lsa 20 ga bo'lgandagi qoldiq (0 bo'lsa 20-variant). Asos sifatida MT1 mustaqil ishining s1–s7 qadamlari kodini oling va uni variantdagi soha, ma'lumot va parametrlarga moslang — barcha ma'lumotlar oflayn (sklearn yoki sintetik generator), GPU shart emas, hamma narsa 2 yadroli CPU da bajariladi; GPU/TensorRT/OpenVINO/Jetson qismlari simulyatsiya yoki izoh sifatida beriladi. Topshiriq: .py fayl(lar), natijalar jadvali, grafik(lar) va xulosa bitta PDF hisobotda; raqamlar taxminiy ekanini va qaysi farazlarga tayanganingizni ko'rsating.

## 1-variant. Media: musiqa striming xizmatida qo'shiq tavsiyasi (oson)

Soha: Media (musiqa)

Mahalliy musiqa striming xizmati foydalanuvchilarga «Siz uchun» pleylistini taklif qilmoqchi. Tinglovlar soni (implicit feedback) bor, aniq reyting yo'q. Sizdan mahsulot hayot siklini kichik modelda bosib o'tish: sintetik tinglov matritsasi, mashhurlik va item-kNN modellari, precision@k va coverage, A/B test rejasi va monitoring qoidalarini taklif qilish so'raladi. Yangi qo'shiqlar har hafta chiqadi — sovuq start muhim.

| Parametr | Qiymat |
|---|---|
| Foydalanuvchi/qo'shiq | 800 / 500 |
| Signal | tinglov soni (Puasson), >=3 — «yoqdi» |
| Modellar | mashhurlik, item-kNN K=20 |
| Metrika | precision@10, coverage |
| A/B | CTR 5%, MDE 10% |
| Drift | har hafta 30 yangi qo'shiq |

Qadamlar:

1. Lab s1 kabi hayot sikli jadvalini musiqa xizmatiga moslab tuzing va sintetik tinglov matritsasini (800x500, Zipf 1,0) yarating; siyraklik va uzun dumni o'lchang.
2. Vaqt bo'yicha bo'lish (oxirgi 2 hafta — test) qiling; mashhurlik modelini bazaviy sifatida baholang.
3. Lab s2 kabi item-kNN (K=20) quring; precision@10, recall@10 va coverage ni jadvalga yozing.
4. Lab s3 kabi CTR 5%, MDE 10% uchun kerakli tanlanmani va tajriba davomiyligini (kuniga 5000 seans) hisoblang.
5. Lab s6 kabi har hafta 30 yangi qo'shiq qo'shilganda sovuq start ulushi va PSI ni 8 hafta kuzating.
6. Xulosa: qaysi bosqichda eng katta xavf bor va qanday monitoring qoidasi tavsiya etiladi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 2-variant. Ta'lim: onlayn kurs platformasida kurs tavsiyasi (oson)

Soha: Ta'lim

Onlayn ta'lim platformasi talabaga keyingi kursni tavsiya qilmoqchi. Talabalar kursni 1–5 baholaydi va tugatish holati ham ma'lum. Maqsad — tugatish ulushini oshirish. Sizdan bias-model va matrix factorization ni qurish, RMSE va «tugatish bo'yicha» precision@5 ni o'lchash, A/B test bilan qaror qabul qilish rejasini tuzish so'raladi.

| Parametr | Qiymat |
|---|---|
| Talaba/kurs | 1000 / 150 |
| Reyting | 1–5, zichlik ~5% |
| Modellar | bias-model, MF k=6 |
| Metrika | RMSE, precision@5 |
| A/B metrika | kursni tugatish ulushi 18% |
| MDE | 8% nisbiy |

Qadamlar:

1. Lab s1 kabi 1000x150 sintetik reyting matritsasini yarating va EDA bajaring (siyraklik, faol talabalar ulushi).
2. Vaqt bo'yicha bo'lib, global o'rtacha va bias-model RMSE sini hisoblang.
3. Lab s2 kabi MF (k=6) o'qiting, RMSE va precision@5 ni bias-model bilan solishtiring.
4. Lab s3 kabi tugatish ulushi 18%, MDE 8% uchun tanlanma va quvvat hisobini bajaring.
5. Model kartasini (lab s7) kurs tavsiyasi uchun YAML ko'rinishida tayyorlang.
6. Xulosa: offline va onlayn metrikalar qanday bog'langan, qaysi biri qaror uchun asosiy.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 3-variant. Transport: taksi xizmatida soatlik talab va haydovchi joylashuvi (o'rta)

Soha: Transport

Shahar taksi xizmati 4 ta tumanda soatlik talabni prognoz qilib, haydovchilarni oldindan joylashtirmoqchi. Talab kunlik va haftalik mavsumiylikka, yomg'ir va futbol o'yinlariga bog'liq. Sizdan sintetik talab generatori, uchta prognoz modeli, MAPE bo'yicha taqqos va prognoz asosida haydovchi taqsimoti rejasi so'raladi.

| Parametr | Qiymat |
|---|---|
| Tumanlar | 4 |
| Tarix | 8 hafta soatlik |
| Modellar | mavsumiy naiv, Ridge, gradient boosting |
| Metrika | MAE, MAPE |
| Haydovchilar | jami 300 |
| Tadbirlar | haftada 1 futbol o'yini |

Qadamlar:

1. Lab s4 kabi 4 tuman uchun sintetik soatlik talab yarating (har tumanga o'z profili, futbol o'yini effekti +60%).
2. Uch modelni oxirgi hafta bo'yicha MAE/MAPE bilan har tuman uchun solishtiring.
3. Prognoz bo'yicha 300 haydovchini tumanlarga proporsional taqsimlang va qondirilmagan talabni hisoblang.
4. Prognozsiz (teng taqsimot) va ideal prognoz bilan solishtiring.
5. Lab s6 kabi yangi tuman ochilganda (talab profili o'zgarganda) MAPE driftini 4 hafta kuzating.
6. Xulosa: prognoz aniqligining biznes qiymati (qondirilmagan safarlar soni bo'yicha).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 4-variant. Chakana savdo: supermarket tarmog'ida mahsulot talabi va zaxira (o'rta)

Soha: Chakana savdo

Supermarket tarmog'i 200 ta mahsulot uchun kunlik talabni prognoz qilib, zaxira siyosatini avtomatlashtirmoqchi. Bayramlar va aksiyalar talabni keskin oshiradi. Sizdan ABC tahlili, talab prognozi, ROP/EOQ hisobi va (s, Q) simulyatsiyasi bilan servis darajasini baholash so'raladi.

| Parametr | Qiymat |
|---|---|
| SKU | 200 |
| Tarix | 180 kun |
| Aksiya effekti | +80% (yiliga 6 marta) |
| Yetkazish | 3–7 kun (tasodifiy) |
| Servis darajasi | 0,95 |
| Saqlash xarajati | 20% / yil |

Qadamlar:

1. Lab s5 kabi 200 SKU uchun 180 kunlik talab (aksiya kunlari bilan) yarating va ABC tahlilini bajaring.
2. A sinf uchun lab s4 usulida kunlik talab prognozini (Ridge vs boosting) MAPE bilan baholang.
3. Prognoz xatosi sigma'si bo'yicha xavfsizlik zaxirasi, ROP va EOQ ni hisoblang.
4. (s, Q) siyosatini tasodifiy yetkazish muddati bilan simulyatsiya qiling: fill rate, tanqislik kunlari.
5. Aksiya kunlari uchun alohida zaxira qoidasini qo'shib, natijani solishtiring.
6. Xulosa: prognoz yaxshilanishi zaxira xarajatini qanchaga kamaytiradi (taxminiy).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 5-variant. Media: yangiliklar portalida maqola tavsiyasi va bosish ulushi (oson)

Soha: Media (yangiliklar)

Yangiliklar portali har sahifada «Sizga ham qiziq» blokini ko'rsatadi. Maqolalar umri qisqa (1–2 kun), shuning uchun mashhurlik tez o'zgaradi. Sizdan mashhurlik va shaxsiy tavsiya siyosatlarini A/B test simulyatsiyasida solishtirish, peeking xavfini ko'rsatish va tajriba muddatini asoslash so'raladi.

| Parametr | Qiymat |
|---|---|
| Foydalanuvchi | 600 |
| Maqolalar | 300 (har kuni 40 yangi) |
| Siyosatlar | mashhurlik vs shaxsiy (shovqin 1,2) |
| CTR | ~3% |
| Seanslar | 10 000 va 50 000 |
| Oraliq qarash | 10 marta |

Qadamlar:

1. Lab s3 simulyatorini yangiliklar uchun moslang: maqola «yoshi» bilan mashhurlik pasayadi.
2. Ikki siyosatning haqiqiy CTR farqini simulyatorda hisoblang.
3. 10 000 va 50 000 seansli A/B test o'tkazing: z-test, 95% CI, qaror.
4. A/A testda 10 marta oraliq qarash bilan soxta g'alaba ulushini o'lchang.
5. Yangilik effekti bilan kunlik o'sish grafigini chizing.
6. Xulosa: minimal tajriba muddati va to'xtatish qoidasi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 6-variant. Elektron tijorat: marketplace tavsiya tizimining to'liq hayot sikli (qiyin)

Soha: Elektron tijorat

Yirik marketplace mahsulot sahifasida «Bu bilan birga xarid qilishadi» va bosh sahifada shaxsiy tavsiya ko'rsatadi. Katalog katta, mashhurlik siljishi kuchli, sotuvchilar xilma-xillikni talab qiladi. Sizdan item-kNN, MF va gibrid modellarni offline baholash, offline/onlayn metrikalar mosligini simulyatsiyada tekshirish va kanareyka relizini rejalash so'raladi.

| Parametr | Qiymat |
|---|---|
| Foydalanuvchi/mahsulot | 1500 / 800 |
| Modellar | item-kNN, MF k=16, gibrid alfa 0–1,5 |
| Metrika | precision@10, coverage, novelty |
| A/B | konversiya 2,5% |
| Kanareyka | 1% → 10% → 50% → 100% |
| Qo'riqchi | p95 < 80 ms, xato < 0,1% |

Qadamlar:

1. Lab s1–s2 kabi 1500x800 sintetik xarid/reyting ma'lumotini yarating va vaqt bo'yicha bo'ling.
2. Item-kNN, MF va 5 xil alfa li gibrid uchun precision@10, coverage va novelty jadvalini tuzing.
3. Lab s3 simulyatorida har siyosatning haqiqiy konversiyasini hisoblang; offline va onlayn metrikalar o'rtasidagi Spearman korrelyatsiyasini toping.
4. Konversiya 2,5%, MDE 5% uchun tanlanma va tajriba davomiyligini hisoblang.
5. Lab s7 kabi FastAPI xizmatida 4 bosqichli kanareykani qo'riqchi metrikalar bilan o'tkazing.
6. Lab s6 kabi mavsumiy drift (bayram savdosi) ssenariysida qayta o'qitish qoidasini sinang.
7. Xulosa: sotuvchilar adolati (coverage) va konversiya o'rtasidagi murosa bo'yicha tavsiya.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)

## 7-variant. Logistika: kuryer xizmatida yetkazish vaqti prognozi va SLA monitoringi (qiyin)

Soha: Logistika

Kuryer xizmati mijozga yetkazish vaqtini (ETA) ko'rsatadi; ETA xatosi mijoz ishonchini pasaytiradi. Yangi tuman qo'shilishi va qishki ob-havo ma'lumot taqsimotini o'zgartiradi. Sizdan ETA modelini qurish, kvantil prognoz bilan SLA ni baholash, drift detektorlarini taqqoslash va avtomatik qayta o'qitish siyosatini ishlab chiqish so'raladi.

| Parametr | Qiymat |
|---|---|
| Buyurtmalar | 20 000 (12 hafta) |
| Belgilar | masofa, soat, ob-havo, tuman, kuryer tajribasi |
| Modellar | Ridge, boosting, kvantil 0,9 |
| SLA | 90% buyurtma ETA+10 daqiqa ichida |
| Drift | 9-haftadan qor (+25% vaqt) |
| Detektorlar | PSI, KS, MAE +10% |

Qadamlar:

1. Sintetik buyurtmalar generatorini yarating (lab s4 usulida) va ETA ning MAE/MAPE sini uch model uchun o'lchang.
2. Kvantil (0,9) boosting bilan SLA bajarilish ulushini hisoblang.
3. Lab s6 kabi 12 hafta monitoring paneli: PSI (ob-havo, masofa), KS (bashorat), haftalik MAE.
4. Uch detektorning drift aniqlash kechikishi va soxta ogohlantirishlar sonini jadvalda solishtiring.
5. Qayta o'qitish siyosatlari (signal bo'yicha / har 4 hafta / hech qachon) uchun o'rtacha MAE ni hisoblang.
6. Xulosa va model kartasi (YAML).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)

## 8-variant. Bank: kredit karta tranzaksiyalarida firibgarlik modeli hayot sikli (o'rta)

Soha: Moliya

Bank firibgarlikni aniqlash modelini ishga tushirmoqchi. Firibgarlik kam uchraydi (0,5%), noto'g'ri bloklash mijozni yo'qotadi. Sizdan sintetik tranzaksiyalar, model, chegara tanlovi (xarajat bo'yicha), «qo'riqchi» metrikalari bilan kanareyka va firibgarlar xulqi o'zgarishi (drift) monitoringini rejalash so'raladi.

| Parametr | Qiymat |
|---|---|
| Tranzaksiyalar | 50 000 |
| Firibgarlik ulushi | 0,5% |
| Model | gradient boosting |
| Xarajat | FN 500$, FP 5$ |
| Metrika | PR-AUC, recall@FPR 1% |
| Drift | 6-haftadan yangi firibgarlik sxemasi |

Qadamlar:

1. make_classification (weights 0,995) yoki o'z generatoringiz bilan tranzaksiyalarni yarating va EDA bajaring.
2. Modelni vaqt bo'yicha bo'lib o'qiting; PR-AUC va recall@FPR 1% ni hisoblang.
3. Xarajat funksiyasi bo'yicha optimal chegarani toping.
4. Lab s6 kabi 10 haftalik monitoring: belgilar PSI si va haftalik recall.
5. Lab s7 kabi FastAPI xizmati va kanareyka (FP ulushi qo'riqchi metrikasi).
6. Xulosa: hayot siklidagi eng katta xavf va nazorat choralari.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 9-variant. Sport: fitnes ilovasida mashg'ulot tavsiyasi (oson)

Soha: Sport

Fitnes ilovasi foydalanuvchiga kunlik mashg'ulot tavsiya qiladi; foydalanuvchi mashg'ulotni 1–5 baholaydi. Yangi foydalanuvchilar ko'p, ular tarixga ega emas. Sizdan bias-model va item-kNN ni qurish, sovuq start foydalanuvchilar uchun alohida metrika berish va oddiy monitoring qoidasini taklif qilish so'raladi.

| Parametr | Qiymat |
|---|---|
| Foydalanuvchi/mashg'ulot | 500 / 120 |
| Reyting | 1–5 |
| Modellar | bias-model, item-kNN K=15 |
| Sovuq start | <5 reyting |
| Metrika | RMSE, precision@5 |

Qadamlar:

1. Lab s1 kabi 500x120 sintetik matritsa yarating; sovuq start foydalanuvchilar ulushini toping.
2. Bias-model va item-kNN RMSE sini umumiy va sovuq start guruhida alohida hisoblang.
3. precision@5 ni ikkala guruh uchun bering.
4. Sovuq start uchun mashhurlik siyosatiga o'tish qoidasini sinab ko'ring.
5. Xulosa: sovuq start muammosi hayot siklining qaysi bosqichida hal qilinishi kerak.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 10-variant. Energetika: elektr tarmog'ida soatlik yuklama prognozi (o'rta)

Soha: Energetika

Hududiy elektr tarmoqlari kompaniyasi soatlik yuklamani prognoz qilib, generatsiyani rejalashtirmoqchi. Yuklama haroratga, dam olish kunlariga va mavsumga bog'liq. Sizdan sintetik yuklama, uch model, MAPE bo'yicha taqqos va prognoz xatosining zaxira quvvatga ta'sirini baholash so'raladi.

| Parametr | Qiymat |
|---|---|
| Tarix | 12 hafta soatlik |
| Belgilar | soat, kun, harorat, bayram |
| Modellar | mavsumiy naiv, Ridge, boosting |
| Metrika | MAPE, p95 xato |
| Zaxira | prognoz + 2 sigma |
| Drift | issiq to'lqin (+8 °C) |

Qadamlar:

1. Lab s4 kabi haroratga bog'liq sintetik yuklama yarating (konditsioner effekti 28 °C dan yuqori).
2. Uch modelni MAPE va p95 xato bilan solishtiring.
3. Prognoz xatosi asosida zarur zaxira quvvatni hisoblang.
4. Issiq to'lqin haftasida drift: MAPE qanchaga oshadi, PSI (harorat) qanday?
5. Qayta o'qitishdan keyingi natija.
6. Xulosa: prognoz aniqligining iqtisodiy ta'siri (zaxira MW bo'yicha).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 11-variant. Sog'liqni saqlash: poliklinikada qabul navbati va shifokorlar jadvali (o'rta)

Soha: Tibbiyot

Poliklinika har kuni qabulga keladigan bemorlar sonini prognoz qilib, shifokorlar jadvalini tuzmoqchi. Gripp mavsumida talab keskin oshadi. Sizdan talab prognozi, shifokorlar sonini rejalash va prognoz xatosining kutish vaqtiga ta'sirini navbat simulyatsiyasida baholash so'raladi.

| Parametr | Qiymat |
|---|---|
| Tarix | 26 hafta kunlik |
| Mavsum | gripp +50% (8 hafta) |
| Modellar | naiv, Ridge, boosting |
| Shifokor | kuniga 25 bemor |
| Maqsad | kutish < 30 daqiqa |
| Metrika | MAPE, kutish vaqti |

Qadamlar:

1. Sintetik kunlik qabul ma'lumotini (hafta kuni + gripp mavsumi) yarating.
2. Uch modelni oxirgi 4 hafta bo'yicha MAPE bilan solishtiring.
3. Prognoz bo'yicha kunlik shifokorlar sonini hisoblang.
4. Oddiy navbat simulyatsiyasida kutish vaqtini prognozli va prognozsiz rejada solishtiring.
5. Gripp mavsumi boshlanishini monitoring qoidasi (MAPE, PSI) bilan qanchalik tez sezish mumkin.
6. Xulosa va model kartasi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 12-variant. Kitob do'koni: onlayn kitob tavsiyasi va katalog qamrovi (oson)

Soha: Chakana savdo (kitoblar)

Onlayn kitob do'koni mijozlarga kitob tavsiya qiladi. Bestsellerlar ko'p sotiladi, lekin noshirlar kam sotiladigan kitoblar ham ko'rinishini xohlaydi. Sizdan mashhurlik va MF modellari, precision@10 va coverage bo'yicha taqqos va oddiy gibrid modelni sinash so'raladi.

| Parametr | Qiymat |
|---|---|
| Foydalanuvchi/kitob | 700 / 600 |
| Zipf | 1,1 |
| Modellar | mashhurlik, MF k=8, gibrid |
| Metrika | precision@10, coverage |
| Gibrid alfa | 0; 0,5; 1,0 |

Qadamlar:

1. Lab s1 kabi 700x600 sintetik reyting matritsasi (Zipf 1,1) yarating.
2. Mashhurlik va MF modellarini quring.
3. Gibridni uch alfa bilan sinang.
4. precision@10 va coverage jadvalini va murosa grafigini chizing.
5. Xulosa: noshirlar talabi (coverage) va sotuv (precision) o'rtasida tanlov.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 13-variant. Agro: fermer xo'jaligida hosil va o'g'it talabi prognozi (o'rta)

Soha: Agro

Agroservis kompaniyasi 50 ta fermer xo'jaligiga o'g'it yetkazadi va mavsumiy talabni oldindan bilishi kerak. Talab ekin turiga, ob-havoga va oldingi yilgi hosilga bog'liq. Sizdan sintetik talab, prognoz modeli, ombor zaxirasi rejasi va ob-havo anomaliyasidagi drift tahlili so'raladi.

| Parametr | Qiymat |
|---|---|
| Xo'jaliklar | 50 |
| Tarix | 3 mavsum haftalik |
| Mahsulotlar | 5 turdagi o'g'it |
| Modellar | Ridge, boosting |
| Yetkazish | 10–21 kun |
| Drift | qurg'oqchilik mavsumi |

Qadamlar:

1. Sintetik haftalik talab (ekin, yog'ingarchilik) generatorini yarating.
2. Ikki modelni oxirgi mavsum bo'yicha MAPE bilan baholang.
3. Lab s5 kabi 5 mahsulot uchun ROP va xavfsizlik zaxirasini hisoblang (uzoq yetkazish muddati bilan).
4. Qurg'oqchilik ssenariysida drift va fill rate pasayishini o'lchang.
5. Qayta o'qitish va zaxirani qayta hisoblash qoidasini taklif qiling.
6. Xulosa: hayot sikli bosqichlari bo'yicha xavflar jadvali.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 14-variant. Telekom: mobil operatorda tarif tavsiyasi va abonent ketishi (churn) (qiyin)

Soha: Telekom

Mobil operator abonentlarga mos tarifni tavsiya qilib, ketish (churn) ulushini kamaytirmoqchi. Tavsiya ham daromadga, ham mijoz qoniqishiga ta'sir qiladi. Sizdan churn modelini qurish, tavsiya siyosatini A/B test simulyatsiyasida baholash (ikki metrika: churn va ARPU), quvvat hisobi va drift monitoringini ishlab chiqish so'raladi.

| Parametr | Qiymat |
|---|---|
| Abonentlar | 30 000 |
| Churn | oyiga 2,5% |
| Model | gradient boosting + uplift (T-learner) |
| A/B metrikalar | churn, ARPU |
| MDE | churn bo'yicha 10% nisbiy |
| Drift | raqobatchi aksiyasi |

Qadamlar:

1. Sintetik abonent ma'lumoti (trafik, to'lov, shikoyatlar) yarating va churn modelini ROC-AUC bilan baholang.
2. T-learner bilan «tarif taklifi» ta'sirini (uplift) bashorat qiling.
3. Lab s3 kabi A/B simulyatsiyasi: churn va ARPU bo'yicha ikki metrika, ko'p taqqoslash tuzatmasi.
4. Kerakli tanlanma va tajriba muddatini hisoblang.
5. Raqobatchi aksiyasi driftida monitoring paneli (PSI, haftalik AUC).
6. Lab s7 kabi model kartasi va kanareyka rejasi.
7. Xulosa: daromad va mijoz manfaati o'rtasidagi muvozanat.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)

## 15-variant. Kino teatr: chiptalar talabi va seanslar jadvali (oson)

Soha: Ko'ngilochar

Kinoteatr tarmog'i har film uchun seanslar sonini talab prognozi asosida belgilamoqchi. Premyera haftasida talab yuqori, keyin pasayadi. Sizdan sintetik chipta sotuvi, oddiy prognoz modeli va zallar to'lishini baholash so'raladi.

| Parametr | Qiymat |
|---|---|
| Filmlar | 30 |
| Tarix | 10 hafta kunlik |
| Premyera effekti | 1-hafta x3, keyin -30%/hafta |
| Model | Ridge |
| Zal | 120 o'rin |

Qadamlar:

1. Sintetik kunlik chipta sotuvi (premyera egri chizig'i, dam olish kunlari) yarating.
2. Ridge modeli bilan keyingi hafta talabini prognoz qiling, MAPE ni hisoblang.
3. Prognoz bo'yicha seanslar sonini belgilang va o'rtacha zal to'lishini hisoblang.
4. Prognozsiz (teng seans) reja bilan solishtiring.
5. Xulosa: prognozning daromadga ta'siri.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 16-variant. Sanoat: zavodda ehtiyot qismlar ombori va texnik xizmat (o'rta)

Soha: Sanoat

Zavod 150 turdagi ehtiyot qismni saqlaydi; ba'zilari kam ishlatiladi, lekin yo'qligi ishlab chiqarishni to'xtatadi. Sizdan ABC/XYZ tahlili, kam talabli qismlar uchun zaxira siyosati va ombor joylashuvini optimallashtirish so'raladi.

| Parametr | Qiymat |
|---|---|
| Qismlar | 150 |
| Tarix | 365 kun |
| Talab turi | ko'p qismlarda vaqti-vaqti bilan (Croston) |
| To'xtash narxi | soatiga 2000$ |
| Yetkazish | 14 kun |
| Joylar | 150 (10x15) |

Qadamlar:

1. Sintetik talab (ko'p nolli, vaqti-vaqti bilan) generatorini yarating; ABC va XYZ (o'zgaruvchanlik) tahlilini bajaring.
2. Doimiy va vaqti-vaqti bilan talab uchun ROP ni hisoblang (Croston yoki Puasson).
3. (s, Q) simulyatsiyasida tanqislik va saqlash xarajatini o'lchang.
4. Lab s5 kabi slotting: tez ishlatiladigan qismlar ustaxonaga yaqin.
5. To'xtash xarajatini hisobga olib servis darajasini optimallashtiring.
6. Xulosa va tavsiyalar jadvali.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 17-variant. Turizm: mehmonxona bandligi prognozi va dinamik narx (o'rta)

Soha: Turizm

Shahar mehmonxonasi xona narxini kunlik bandlik prognozi asosida o'zgartirmoqchi. Bayramlar, konferensiyalar va mavsum talabni o'zgartiradi. Sizdan prognoz modeli, Uber surge'iga o'xshash dinamik narx qoidasi va tushum/bandlik bo'yicha taqqos so'raladi.

| Parametr | Qiymat |
|---|---|
| Xonalar | 120 |
| Tarix | 2 yil kunlik |
| Elastiklik | -0,8 |
| Narx | 60–180$ |
| Modellar | naiv, boosting |
| Tadbirlar | oyiga 2 konferensiya |

Qadamlar:

1. Sintetik kunlik bandlik talabi (mavsum, hafta kuni, tadbirlar) yarating.
2. Ikki modelni oxirgi 60 kun bo'yicha MAPE bilan solishtiring.
3. Lab s4 kabi narx koeffitsienti qoidasini prognozga bog'lang.
4. Tushum, bandlik va qondirilmagan talabni 3 ssenariyda (qat'iy narx, naiv, boosting) hisoblang.
5. Narx koeffitsientining yuqori chegarasini optimallashtiring.
6. Xulosa: mijozga adolatli narx va tushum murosasi.

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar va jadval — 25, Tahlil va xulosa — 30, Hisobot sifati — 20 (jami 100)

## 18-variant. Shahar: velosiped ijarasi stansiyalarida talab va qayta taqsimlash (qiyin)

Soha: Shahar transporti

Shahar velosiped ijarasi tizimida 20 ta stansiya bor; ertalab velosipedlar markazga oqib ketadi va chekka stansiyalar bo'shab qoladi. Sizdan stansiya darajasidagi soatlik talab prognozi, qayta taqsimlash yuk mashinasi rejasini optimallashtirish (tayinlash masalasi), bajarilmagan ijaralar soni bo'yicha baholash va drift (yangi metro bekati) monitoringi so'raladi.

| Parametr | Qiymat |
|---|---|
| Stansiyalar | 20 (har biri 15 joy) |
| Tarix | 6 hafta soatlik |
| Modellar | boosting (stansiya belgisi bilan) |
| Qayta taqsimlash | kuniga 2 marta, 40 velosiped |
| Optimallash | linear_sum_assignment |
| Drift | 4-haftadan yangi metro bekati |

Qadamlar:

1. Har stansiya uchun olish/qaytarish oqimlarini sintetik yarating (ertalabki va kechki oqim).
2. Soatlik sof oqimni boosting bilan prognoz qiling, MAE ni stansiyalar bo'yicha bering.
3. Prognoz asosida qayta taqsimlashni tayinlash masalasi sifatida yeching (masofa x velosiped).
4. Bajarilmagan ijaralar sonini qayta taqsimlashsiz, oddiy va optimal rejada solishtiring.
5. Metro bekati ochilgandan keyingi drift: PSI, MAE, qayta o'qitish.
6. Xulosa va hayot sikli hisoboti (lab s7 uslubida).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)

## 19-variant. Oziq-ovqat yetkazish: restoranlar reytingi va tavsiya (oson)

Soha: Oziq-ovqat xizmati

Oziq-ovqat yetkazish ilovasi foydalanuvchiga restoran tavsiya qiladi. Restoranlarni baholash 1–5, lekin yetkazish vaqti ham muhim. Sizdan bias-model va MF, RMSE va precision@5, hamda yetkazish vaqtini hisobga olgan oddiy qayta saralash qoidasi so'raladi.

| Parametr | Qiymat |
|---|---|
| Foydalanuvchi/restoran | 600 / 150 |
| Reyting | 1–5 |
| Modellar | bias-model, MF k=6 |
| Qayta saralash | baho - 0,02 x yetkazish daqiqasi |
| Metrika | RMSE, precision@5 |

Qadamlar:

1. Lab s1 kabi 600x150 sintetik reyting va restoranlarning yetkazish vaqtlarini yarating.
2. Bias-model va MF ni RMSE bilan solishtiring.
3. precision@5 ni qayta saralashsiz va qayta saralash bilan hisoblang.
4. O'rtacha yetkazish vaqti tavsiyalarda qanday o'zgarganini ko'rsating.
5. Xulosa: bir nechta maqsadli tavsiya (sifat va tezlik).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 20-variant. Aviakompaniya: chipta talabi, narx va qo'shimcha xizmat tavsiyasi (qiyin)

Soha: Aviatsiya

Aviakompaniya reyslar bo'yicha chipta talabini prognoz qilib, narx darajalarini boshqaradi va yo'lovchiga qo'shimcha xizmat (bagaj, joy tanlash) tavsiya qiladi. Sizdan uch qismli tizim — talab prognozi, dinamik narx, xizmat tavsiyasi — uchun to'liq hayot siklini qurish, A/B rejasi, monitoring va yagona API xizmatini joylash so'raladi.

| Parametr | Qiymat |
|---|---|
| Reyslar | 40 (90 kun oldin sotuv) |
| Talab modeli | boosting (kunlar qolgani, mavsum) |
| Narx | 4 daraja (50–250$) |
| Tavsiya | item-kNN, 6 xizmat |
| A/B | qo'shimcha xizmat konversiyasi 12% |
| Kanareyka | 5% → 25% → 100% |

Qadamlar:

1. Reyslar bo'yicha bron qilish egri chizig'i (kunlar qolgani bo'yicha) va yo'lovchi-xizmat matritsasini sintetik yarating.
2. Talab prognozini MAPE bilan baholang va narx darajasini prognozga bog'lang (lab s4).
3. Xizmat tavsiyasi uchun item-kNN va mashhurlik modelini precision@3 bilan solishtiring.
4. Lab s3 kabi konversiya 12% uchun A/B tajriba va quvvat hisobini bajaring.
5. Lab s6 kabi bayram mavsumi driftida monitoring paneli.
6. Lab s7 kabi /price va /upsell endpointli xizmat, kanareyka va model kartalari.
7. Xulosa: uch modelning o'zaro ta'siri (narx tavsiya konversiyasiga ta'sir qiladimi).

Topshiriladi: Kod (.py, laboratoriya qadamlari asosida moslashtirilgan); Natijalar jadvali (CSV yoki hisobotdagi jadval); Grafik(lar) (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 25, O'lchovlar, statistik asoslash — 30, Tahlil, murosa va tavsiyalar — 30, Hisobot sifati — 15 (jami 100)
