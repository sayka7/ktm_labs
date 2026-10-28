# MT2. Mustaqil bajarish uchun variantlar (20)

Variant raqami = jurnal (guruh ro'yxati) raqamingiz; 20 dan katta bo'lsa 20 ga bo'lgandagi qoldiq (0 bo'lsa 20-variant). Asos sifatida MT2 mustaqil ishining s1–s7 qadamlari kodini oling va uni variantdagi soha, ma'lumot va parametrlarga moslang — barcha ma'lumotlar oflayn (sklearn yoki sintetik generator), GPU shart emas, hamma narsa 2 yadroli CPU da bajariladi; GPU/TensorRT/OpenVINO/Jetson qismlari simulyatsiya yoki izoh sifatida beriladi. Topshiriq: .py fayl(lar), natijalar jadvali, grafik(lar) va xulosa bitta PDF hisobotda; raqamlar taxminiy ekanini va qaysi farazlarga tayanganingizni ko'rsating.

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
