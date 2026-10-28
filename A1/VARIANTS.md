# A1. Mustaqil bajarish uchun variantlar (20)

Variant raqami = guruh jurnalidagi tartib raqamingiz (21 va undan katta bo'lsa 20 ga bo'lgandagi qoldiq, qoldiq 0 bo'lsa 20-variant). Asos sifatida A1 laboratoriyasining s1–s7 bosqichlari kodini oling va undagi KAMERALAR/FPS, MODELLAR, PLATFORMALAR, BLOKLAR, VARIANTLAR jadvallarini o'z variantingiz parametrlari bilan almashtiring; har qadamda qaysi laboratoriya bosqichiga mos kelishi qavsda (s1…s7) ko'rsatilgan. Barcha hisoblar CPU'da bajariladi (GPU/tezlatkich ko'rsatkichlari rasmiy tavsiflardan jadval sifatida kiritiladi), ma'lumotlar sintetik. Topshiriq: .py/.ipynb fayllar, jadval va grafiklar hamda PDF hisobot fan saytiga yuklanadi; himoyada tanlov asosini raqamlar bilan tushuntirasiz.

## 1-variant. Shifoxona KT kesimlarida o'pka tugunlarini segmentatsiyalash serveri (o'rta)

Soha: Tibbiyot

Viloyat ko'p tarmoqli shifoxonasida 4 ta KT apparat ishlaydi, har biri soatiga 12 ta tekshiruv o'tkazadi va har tekshiruvda o'rtacha 320 ta 512x512 kesim hosil bo'ladi. Radiolog hisobot yozguncha segmentatsiya natijasi butun tekshiruv uchun 90 s ichida tayyor bo'lishi kerak. Bemor ma'lumotlari shifoxona tarmog'idan chiqmasligi shart, shuning uchun bulut variantlari ko'rilmaydi. Tizim har KT xonasida (edge) yoki server xonasida (markaz) joylashishi mumkin; apparat byudjeti 12 000 $, quvvat chegarasi 400 Vt. Ertalabki pik soatlarda yuklama o'rtachadan 3 baravar yuqori.

| Parametr | Qiymat |
|---|---|
| Yuklama | 4 KT x 12 tekshiruv/soat x 320 kesim (pik x3) |
| Modellar (GFLOP/kesim, Dice) | U-Net-lite 256^2: 9, 0,82; U-Net 512^2: 62, 0,86; nnU-Net 2.5D 512^2: 130, 0,89 |
| SLO | butun tekshiruv <= 90 s, inferens ulushi <= 60 s |
| Platformalar | RTX A4000, L4, T4, Xeon 16 yadro (AMX), AGX Orin 64GB |
| Cheklovlar | byudjet 12 000 $, quvvat 400 Vt, ma'lumot ichkarida |
| Formatlar | FP16, INT8 |
| TCO | 5 yil, elektr 0,06 $/kVt*soat, KT soni 1-12 |

Qadamlar:

1. (s1) KAMERALAR/FPS o'rniga kesim/s hisoblang (o'rtacha va pik x3); 3 model x 2 kirish o'lchami (256, 512) uchun GFLOP, vazn MB (FP16/INT8), TFLOP/s talabi va kerakli cho'qqi TOPS (samaradorlik 0,4) jadvali; DICOM kirish oqimini (512^2 x 2 bayt/kesim) MB/s da hisoblang.
2. (s2) 5 platforma jadvali (cho'qqi FP16 TFLOP/s, INT8 TOPS, Vt, narx, FP16/INT8 qo'llab-quvvatlash); zaxira koeffitsiyenti pik yuklamaga nisbatan, bir kesim energiyasi (J); 6 mezonli og'irlikli matritsa (tekshiruv vaqti, joylashuv/maxfiylik, quvvat, narx, format, ekotizim) va TANLOV.
3. (s3) 90 s byudjetni bloklarga bo'ling: PACS'dan yuklash, resampling, inferens (320 kesim, batch 8/16/32), 3D post-processing, hisobot yozish; lognormal Monte-Karlo (10 000 tekshiruv) bilan p50/p95/p99 va byudjetdan oshgan tekshiruvlar ulushini toping; batch ta'siri grafigi.
4. (s4) Roofline: har platforma tizza nuqtasi; U-Net 512^2 FP16/INT8 batch 1/8/32 uchun AI (vaznlar + faollashuvlar ~180 MB/kesim) va rejim (H/X); 2.5D model uchun ham joylashtiring.
5. (s5) torch'da MiniUNet (3 ta conv enkoder stride 2 + 2 ta ConvTranspose2d dekoder + 1x1 chiqish) yozing; hook formulalari va FlopCounterMode (ConvTranspose uchun ham) 128^2/256^2 kirishda; batch 1/8 CPU vaqti va GFLOP/s.
6. (s6) Co-design grid: 3 model x 2 imgsz x 2 format x 3 batch x 5 platforma; cheklovlar zaxira >= 1,5 (pik), inferens <= 60 s/tekshiruv, quvvat, narx; Pareto narx-Dice; har platforma uchun eng yaxshi variant.
7. (s7) TCO 5 yil: edge (har KT xonasida AGX Orin) vs markaz (L4 server) vs markaz (A4000 ish stansiyasi), KT soni 1-12; tenglik nuqtasi va $/tekshiruv ko'rsatkichi.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z); Strukturaviy sxema matni: PACS -> ishlov -> model -> RIS interfeyslari (DICOM, HL7) o'tkazuvchanligi bilan

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar) — 25, Hisob-kitob va simulyatsiya to'g'riligi — 25, Grafiklar (Roofline, byudjet, co-design, TCO) — 15, Tahlil, tanlov asosi va xavflar — 25, Hisobot sifati — 10 (jami 100)

## 2-variant. Paxta dalasida begona o'tlarni dron bortida aniqlash (oson)

Soha: Agro

Fermer xo'jaligi 300 gektar paxta maydonini kuzatish uchun dron sotib olmoqda. Dron 30 m balandlikda uchadi, bort kamerasi 1080p 15 kadr/s beradi va begona o'tlar (4 sinf) shu yerning o'zida aniqlanib, GPS bilan belgilanishi kerak, chunki dalada aloqa yo'q. Bort hisoblagichining quvvati 10 Vt dan oshmasligi, massasi 150 g dan kam va narxi 500 $ dan arzon bo'lishi kerak; muhit harorati yozda +45 C gacha. Bitta parvoz 25 daqiqa, kechikish 150 ms dan oshsa belgilangan koordinata 1 m dan ko'proq siljiydi.

| Parametr | Qiymat |
|---|---|
| Yuklama | 1 kamera x 15 kadr/s (sinov: 30 kadr/s) |
| Modellar | yolov8n (8,7 GFLOP), yolo11n (6,5), yolov8s (28,6) @ 640; imgsz 320/480/640 |
| SLO | kadr <= 150 ms uchdan-uchga |
| Platformalar | Orin Nano 8GB, Hailo-8L (RPi 5 bilan), Coral Edge TPU, RK3588, Orin NX |
| Cheklovlar | <= 10 Vt, <= 150 g, <= 500 $, +45 C |
| Energiya | batareya 60 Vt*soat, parvoz 25 daqiqa |
| TCO | 3 yil, 1 dron vs 3 dron, ta'mir 15 %/yil |

Qadamlar:

1. (s1) KAMERALAR=1, FPS=15 va 30 uchun 3 model x 3 imgsz jadvali: GFLOP, vazn INT8 MB, TFLOP/s va kerakli cho'qqi TOPS (samaradorlik 0,35); bir parvozdagi jami kadrlar va GB hisobini qo'shing.
2. (s2) 5 platforma (TOPS dense, samaradorlik, Vt, narx, massa g, dekoder, INT8/FP16, harorat): zaxira, inferens_J va parvoz davomida hisoblagich iste'moli (Vt*soat) ustunlari; 6 mezonli matritsa (zaxira, quvvat, massa, narx, harorat, ekotizim) va TANLOV; massa va quvvat og'irliklarini almashtirib tanlov barqarorligini tekshiring.
3. (s3) 150 ms byudjet: kamera olish + ISP, oldindan ishlov, inferens, NMS + GPS bog'lash; lognormal simulyatsiya 20 000 kadr, p95/p99; batch 1/2/4 kutish vaqti 30 kadr/s da.
4. (s4) Roofline 4 platforma uchun (Orin Nano 68 GB/s, Hailo-8L va Coral host DRAM taxminiy 20-30 GB/s, RK3588 25 GB/s): yolov8n INT8 batch 1/4 AI va rejim; nima uchun kichik model batch 1 da xotira bilan cheklanishini yozing.
5. (s5) MiniYolo'ni 4 sinf uchun 320x320 va 480x480 kirishda hook va FlopCounterMode bilan hisoblang; batch 1/4 CPU vaqti; FLOP imgsz^2 proporsionalligini jadvalda tasdiqlang.
6. (s6) Co-design grid: 3 model x 3 imgsz x INT8/FP16 x batch 1/2/4 x 5 platforma; cheklovlar zaxira >= 1,3, inferens <= 60 ms, <= 10 Vt, <= 500 $, <= 150 g; Pareto narx-mAP va eng yengil yaroqli variant.
7. (s7) TCO 3 yil: 1 dron bortida hisoblash vs tasvirlarni uyga olib kelib ish stansiyasida (RTX 3060, 1500 $) tungi ishlov; 1/2/3 dron uchun jadval; bort variantining asosiy afzalligi (kechikish) TCO'da yo'qligini izohlang.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar, hisob-kitob va jadvallar — 25, Tahlil va asoslangan tanlov — 25, Hisobot sifati (grafik, sxema, til) — 20 (jami 100)

## 3-variant. Shahar chorrahalarida transport oqimini hisoblash va qoidabuzarlikni aniqlash (o'rta)

Soha: Transport

Shahar yo'l harakati boshqarmasi 12 ta chorrahaning har birida 4 tadan kamera (jami 48 ta, 1080p 25 kadr/s) o'rnatgan. Har kadrda transport turlari (6 sinf) aniqlanib, qizil chiroqda o'tish va to'xtash chizig'i buzilishi 100 ms ichida qayd etilishi kerak (svetofor kontrolleriga moslashuvchan signal uchun). Har chorrahada shkafda 35 Vt quvvat va -30...+50 C harorat; chorrahadan boshqarmaga 20 Mbit/s optika bor. Byudjet chorraha boshiga 3000 $. Arxitektura tanlovi: har chorrahada edge modul yoki boshqarmada markaziy GPU server.

| Parametr | Qiymat |
|---|---|
| Yuklama | 48 kamera x 25 kadr/s (har chorrahada 4 x 25 = 100 inferens/s) |
| Modellar | yolov8n/s/m @ 640 (8,7/28,6/78,9 GFLOP), imgsz 416/640/800 |
| SLO | p99 <= 100 ms (tarmoq 20 Mbit/s markaz uchun +30 ms) |
| Platformalar | Orin NX, AGX Orin, Hailo-8 (x2), L4 server (markaz), T4 server (markaz) |
| Cheklovlar | shkaf <= 35 Vt, -30...+50 C, <= 3000 $/chorraha |
| Tarmoq | chorraha -> markaz 20 Mbit/s, kamera oqimi 4 Mbit/s |
| TCO | 3 yil, 12 chorraha, elektr 0,07 $/kVt*soat, optika ijarasi 600 $/yil |

Qadamlar:

1. (s1) Bitta chorraha (100 inferens/s) va butun shahar (1200 inferens/s) uchun 3 model x 3 imgsz jadvali: GFLOP, TFLOP/s, kerakli cho'qqi TOPS (samaradorlik 0,45); 4 kamera x 4 Mbit/s oqim 20 Mbit/s kanalga sig'adimi - markaz varianti uchun tekshiring.
2. (s2) 5 platforma jadvali: edge uchun chorraha yuklamasi, markaz uchun 1200 inferens/s talab bilan zaxira; dekodlash yetarliligi (4 va 48 oqim 1080p25); 7 mezonli matritsa (zaxira, kechikish, harorat, quvvat, dekod, narx, ekotizim); edge va markaz uchun alohida TANLOV.
3. (s3) 100 ms byudjet ikki arxitekturada: edge (olish+dekod, ishlov, inferens, kuzatuv+qoida mantiqi) va markaz (+tarmoq 30 ms median, sigma 0,5); 20 000 kadr Monte-Karlo, p99 va oshgan kadrlar ulushi; batch 1/4/8 ta'siri; ikki panelli grafik.
4. (s4) Roofline: Orin NX, AGX Orin, L4, T4 (xotira 102/205/300/320 GB/s); yolov8s INT8 batch 1/4/8 AI va rejim; markaz serverida batch 8 bilan nima o'zgaradi.
5. (s5) MiniYolo 6 sinf, kirish 128/256/416; hook vs FlopCounterMode; batch 1/8 CPU vaqti va GFLOP/s; birinchi conv stride 1 bo'lsa FLOP necha baravar oshishini o'lchang.
6. (s6) Co-design grid 3 model x 3 imgsz x 2 format x 3 batch x 5 platforma; edge cheklovlari (zaxira >= 1,5, inferens <= 40 ms, <= 35 Vt, <= 3000 $, harorat) va markaz cheklovlari (zaxira >= 1,5 jami yuklama uchun, inferens <= 30 ms) alohida; Pareto narx-mAP ikkala holatda.
7. (s7) TCO 3 yil: edge Orin NX har chorrahada vs markaz L4 server(lar) vs markaz T4 server(lar); chorraha soni 1-30 bo'yicha tenglik nuqtasi; tarmoq ijarasi 0 va 600 $/yil holatlarini solishtiring.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z); Strukturaviy sxema: kamera -> edge/markaz -> svetofor kontrolleri interfeyslari (RTSP, MQTT/NTCIP) va o'tkazuvchanlik

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar) — 25, Hisob-kitob va simulyatsiya to'g'riligi — 25, Grafiklar (Roofline, byudjet, co-design, TCO) — 15, Tahlil, tanlov asosi va xavflar — 25, Hisobot sifati — 10 (jami 100)

## 4-variant. 5G baza stansiyalari KPI oqimida anomaliyani real vaqtda aniqlash (oson)

Soha: Telekom

Mobil operator 2 400 ta hujayra (sektor) uchun har sekundda 48 ta KPI (trafik, PRB bandligi, RSRP, uzilishlar va h.k.) oladi va har hujayra uchun 1 Hz chastotada anomaliya ehtimolini hisoblamoqchi. Model - kichik MLP yoki 1D-CNN (oxirgi 60 s oynasi); tasvir yo'q, shuning uchun yuklama kamera emas, hujayra x chastota bilan hisoblanadi. SLO: anomaliya 2 s ichida aniqlanishi kerak; tizim operatorning mavjud x86 serverlarida (GPU'siz) yoki kichik GPU'da ishlashi mumkin. Keyingi yilda hujayralar soni 2 baravar oshadi. Ma'lumotlar sklearn make_classification bilan sintetik hosil qilinadi.

| Parametr | Qiymat |
|---|---|
| Yuklama | 2 400 hujayra x 1 Hz (kelajak: 4 800 x 1 Hz, sinov 10 Hz) |
| Modellar | MLP 48-128-64-1 (~0,04 MFLOP), 1D-CNN 60x48 (3 conv, ~3 MFLOP), kichik Transformer 60x48 (~25 MFLOP) |
| SLO | p99 <= 2 s uchdan-uchga, inferens <= 50 ms partiya uchun |
| Platformalar | x86 8 yadro (AVX2), Xeon 16 yadro (AMX), T4, Orin Nano |
| Byudjet | 4 000 $, mavjud server xonasi |
| Partiya (batch) | 1, 64, 2400 (butun davr) |

Qadamlar:

1. (s1) KAMERALAR*FPS o'rniga HUJAYRALAR x CHASTOTA: 3 model uchun MFLOP/inferens, vazn MB (FP32/FP16/INT8), GFLOP/s talab (2 400 va 4 800 hujayra, 1 va 10 Hz), kerakli cho'qqi (samaradorlik 0,2, chunki kichik modellar past samarali); KPI oqimi (48 x 4 bayt x hujayra x Hz) kbayt/s.
2. (s2) 4 platforma (FP32 TFLOP/s, INT8 TOPS, Vt, narx, ekotizim): zaxira koeffitsiyenti 1D-CNN uchun, bir inferens energiyasi (mJ); 5 mezonli matritsa (zaxira, narx, quvvat, mavjud infratuzilma bilan moslik, ekotizim) va TANLOV; GPU bu yuklama uchun nima uchun ortiqcha ekanini raqam bilan ko'rsating.
3. (s3) 2 000 ms byudjet: Kafka/kolleksiya (median 300 ms, sigma 0,6), oyna yig'ish, inferens partiyada, ogohlantirish yozish; 20 000 davr Monte-Karlo; partiya 1/64/2400 uchun kutish vaqti (1 Hz davr ichida) va p99.
4. (s4) Roofline: MLP va 1D-CNN uchun AI (vazn + oyna baytlari) batch 1/64/2400 - hammasi xotira bilan cheklangan ekanini x86 (70 GB/s) va T4 (320 GB/s) uchun ko'rsating; qaysi batch'dan boshlab H rejimga o'tadi.
5. (s5) torch'da MLP va 1D-CNN (Conv1d) yozing, Linear/Conv1d hook formulalari va FlopCounterMode; batch 1/64/2400 CPU vaqti - bir inferens vaqti batch bilan necha baravar kamayadi.
6. (s6) Co-design grid: 3 model x 3 batch x FP32/INT8 x 4 platforma; cheklovlar zaxira >= 2 (4 800 hujayra), inferens <= 50 ms, narx <= 4 000 $; Pareto narx-sifat (sifat proxysi: MLP 0,90, 1D-CNN 0,94, Transformer 0,96 AUC).
7. (s7) TCO 3 yil: mavjud x86 server (faqat elektr+xizmat) vs yangi T4 server vs bulut GPU (0,35 $/soat); hujayralar 2 400 -> 9 600 o'sishida qaysi variant qachon arzon.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar, hisob-kitob va jadvallar — 25, Tahlil va asoslangan tanlov — 25, Hisobot sifati (grafik, sxema, til) — 20 (jami 100)

## 5-variant. Elektr tarmog'i PMU o'lchovlarida nosozlikni aniqlash va tasniflash (o'rta)

Soha: Energetika

Hududiy elektr tarmog'ida 400 ta PMU (fazor o'lchagich) 50 Hz chastotada 12 kanalli (3 faza kuchlanish/tok, chastota, burchak) ma'lumot yuboradi. 1D-CNN model 200 ms oynani (10 namuna) tahlil qilib, 8 turdagi nosozlikni (qisqa tutashuv, uzilish, tebranish...) aniqlaydi. Himoya avtomatikasi uchun uchdan-uchga kechikish 100 ms dan oshmasligi kerak, ya'ni nosozlik oyna tugaganidan 100 ms ichida tasniflanadi. Tizim podstansiyalarda (har birida 20 PMU, sanoat shkafi, -40...+55 C, 20 Vt) yoki dispetcherlik markazida joylashishi mumkin; markazga 2 Mbit/s radioreleyli kanal, kechikishi 15-40 ms.

| Parametr | Qiymat |
|---|---|
| Yuklama | 400 PMU x 50 Hz (20 PMU/podstansiya, 20 podstansiya); oyna 10 namuna x 12 kanal |
| Modellar | 1D-CNN-S (0,8 MFLOP), 1D-CNN-M (6 MFLOP), TCN (40 MFLOP), aniqlik 0,91/0,95/0,97 |
| SLO | p99 <= 100 ms (himoya), markaz uchun kanal 15-40 ms |
| Platformalar | Orin Nano, Orin NX, Xilinx Kria K26 (1,4 TOPS INT8, 10 Vt), x86 sanoat PC (4 yadro), T4 server (markaz) |
| Cheklovlar | podstansiya <= 20 Vt, -40...+55 C, <= 1 500 $/podstansiya |
| Kanal | 2 Mbit/s, PMU oqimi 12 x 4 bayt x 50 Hz |
| TCO | 5 yil, elektr 0,05 $/kVt*soat, 20 podstansiya |

Qadamlar:

1. (s1) Oyna/s hisobi (har PMU 50 Hz da har namuna uchun yangi oyna: 400 x 50 = 20 000 inferens/s; podstansiya 1 000 inferens/s); 3 model uchun GFLOP/s, kerakli cho'qqi TOPS (samaradorlik 0,25); PMU oqimi kbit/s va 20 PMU uchun 2 Mbit/s kanal yetarliligi.
2. (s2) 5 platforma: zaxira podstansiya (1 000 inf/s) va markaz (20 000 inf/s) uchun; inferens energiyasi (mJ); 7 mezonli matritsa (zaxira, kechikish, harorat, quvvat, narx, sertifikatlash/IEC 61850 ekotizimi, format); edge va markaz uchun TANLOV.
3. (s3) 100 ms byudjet: PMU -> konsentrator (10 ms), oyna yig'ish, inferens, rele buyrug'i (GOOSE 4 ms); markaz uchun +radiorele (median 25 ms, sigma 0,5); 20 000 oyna Monte-Karlo p99; partiya 1/20/400 ta'siri va qaysi arxitektura SLO'ni bajaradi.
4. (s4) Roofline: 1D-CNN-M va TCN batch 1/20/400 uchun AI; Orin Nano (68 GB/s), Kria (19 GB/s), x86 (40 GB/s), T4 (320 GB/s) tizza nuqtalari; rejim H/X.
5. (s5) torch'da 1D-CNN-M (Conv1d x3 + Linear) yozing; hook (2*K*Cin*Cout*Lout) va FlopCounterMode; batch 1/20/400 CPU vaqti; bir oyna uchun mks va GFLOP/s.
6. (s6) Co-design: 3 model x 3 batch x FP32/FP16/INT8 x 5 platforma; cheklovlar zaxira >= 2, inferens+kutish <= 40 ms, quvvat, narx, harorat; Pareto narx-aniqlik; Kria INT8 nima uchun chegarada ekanini tushuntiring.
7. (s7) TCO 5 yil: edge (20 podstansiyada Orin Nano/Kria) vs markaz T4 server + 20 kanal; podstansiya soni 1-40 tenglik nuqtasi; markaz arxitekturasidagi yagona nosozlik nuqtasi va kanal uzilishi xavfini xulosada baholang.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z); Strukturaviy sxema: PMU -> konsentrator -> model -> rele (IEC 61850 GOOSE/MMS) interfeyslari

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar) — 25, Hisob-kitob va simulyatsiya to'g'riligi — 25, Grafiklar (Roofline, byudjet, co-design, TCO) — 15, Tahlil, tanlov asosi va xavflar — 25, Hisobot sifati — 10 (jami 100)

## 6-variant. To'lov tizimida kartali tranzaksiyalar firibgarligini real vaqtda skoring qilish (o'rta)

Soha: Moliya

Milliy to'lov tizimi o'rtacha 3 000 tranzaksiya/s (bayram pikida 12 000/s) qayta ishlaydi. Har tranzaksiya 64 ta belgidan iborat vektor bo'lib, firibgarlik skoringi avtorizatsiya zanjirida sinxron bajariladi: p99 kechikish 20 ms dan oshsa bank tomoni taym-aut beradi. Uchta model nomzod: gradient-busting o'rnini bosuvchi MLP, kengroq MLP va 32 so'nggi tranzaksiya ketma-ketligi uchun GRU. Ma'lumotlar markazida ikki zona (faol-faol) bo'lishi shart, ya'ni har zona to'liq yuklamani ko'tarishi kerak. Byudjet 40 000 $ (2 zona), har server 300 Vt chegara; GPU'siz x86 variant ham jiddiy ko'rib chiqiladi. Ma'lumotlar sklearn make_classification (imbalance 0,2 %) bilan sintetik.

| Parametr | Qiymat |
|---|---|
| Yuklama | 3 000 tr/s o'rtacha, 12 000 tr/s pik; 2 zona, har biri 100 % |
| Modellar | MLP 64-256-64-1 (0,07 MFLOP, AUC 0,93), MLP 64-1024-512-1 (1,2 MFLOP, 0,95), GRU 32x64 h=128 (5 MFLOP, 0,97) |
| SLO | p99 <= 20 ms, inferens <= 5 ms |
| Platformalar | x86 32 yadro (AVX-512), Xeon AMX 32 yadro, T4, L4, A10 |
| Cheklovlar | <= 40 000 $ (2 zona), <= 300 Vt/server, faol-faol |
| Partiya | 1, 16, 128 (dinamik batching 2 ms oyna) |
| TCO | 3 yil, elektr 0,10 $/kVt*soat, bulut muqobili 0,7 $/soat (L4) |

Qadamlar:

1. (s1) 3 model uchun MFLOP, vazn MB, GFLOP/s talab (o'rtacha va pik), kerakli cho'qqi (samaradorlik 0,25 kichik partiya uchun, 0,5 katta partiya uchun); belgi oqimi (64 x 4 bayt x tr/s) MB/s va 32 tranzaksiyali ketma-ketlikni xotirada saqlash hajmi (10 M karta uchun).
2. (s2) 5 platforma (FP32 TFLOP/s, INT8 TOPS, Vt, narx): zaxira pik yuklamaga va bir zona uchun, tranzaksiya energiyasi (mJ); 6 mezonli matritsa (zaxira, kechikish, narx, quvvat, ekotizim/Triton, ishonchlilik); TANLOV va 'GPU shartmi' savoliga raqamli javob.
3. (s3) 20 ms byudjet: tarmoq kirish (2 ms), belgilarni Redis'dan olish (median 3 ms, sigma 0,6), dinamik batching kutish (0-2 ms), inferens, qaror va javob; 50 000 tranzaksiya Monte-Karlo p99; batch 1/16/128 ta'siri; 12 000 tr/s pikda kutish qanday kamayishini ko'rsating.
4. (s4) Roofline: MLP va GRU batch 1/16/128 AI; x86 (120 GB/s), T4 (320), L4 (300), A10 (600) tizza nuqtalari; GRU ketma-ket qadamlari nima uchun past AI berishini yozing.
5. (s5) torch'da 3 modelni (nn.Linear, nn.GRU) yozing; Linear hook va GRU uchun formula (3 x 2 x (in+h) x h x T) ni FlopCounterMode bilan solishtiring; batch 1/16/128 CPU vaqti va p50 mks/tranzaksiya.
6. (s6) Co-design grid: 3 model x 3 batch x FP32/FP16/INT8 x 5 platforma; cheklovlar zaxira >= 1,5 (pik, 1 zona), inferens+kutish <= 5 ms, narx 2 zona uchun, quvvat; Pareto narx-AUC; faol-faol uchun eng arzon ikki server.
7. (s7) TCO 3 yil: 2 x x86 server vs 2 x L4 server vs bulut L4 (2 zona, 24/7) vs bulut avtomasshtab (o'rtacha 30 % yuk); tr/s 3 000 -> 30 000 o'sishida tenglik nuqtasi.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar) — 25, Hisob-kitob va simulyatsiya to'g'riligi — 25, Grafiklar (Roofline, byudjet, co-design, TCO) — 15, Tahlil, tanlov asosi va xavflar — 25, Hisobot sifati — 10 (jami 100)

## 7-variant. Onlayn imtihonda avtomatik proktoring: yuz va qarash yo'nalishini kuzatish (o'rta)

Soha: Ta'lim

Universitet masofaviy imtihonlarida bir vaqtda 1 500 ta talaba veb-kamera orqali kuzatiladi. Har talaba oqimidan 2 kadr/s olinib, yuz aniqlash, bir nechta odam borligi, qarash yo'nalishi va telefon aniqlash bajariladi; shubhali hodisa 3 s ichida nazoratchiga ko'rsatilishi kerak. Imtihon mavsumi yiliga 6 hafta, qolgan vaqtda tizim bo'sh turadi - shuning uchun bulut ijarasi va o'z serveri jiddiy solishtiriladi. Talaba brauzerida (WebAssembly, ~0,5 TFLOP/s) qisman ishlov berish ham variant. Byudjet 25 000 $, universitet server xonasida 2 kVt bo'sh quvvat bor.

| Parametr | Qiymat |
|---|---|
| Yuklama | 1 500 talaba x 2 kadr/s = 3 000 inferens/s (pik 2 500 talaba) |
| Modellar | SCRFD-0.5G (0,5 GFLOP) + MobileFaceNet (0,45) + yolov8n telefon @ 320 (2,2); og'ir: SCRFD-2.5G + ArcFace-R50 (12,6) + yolov8s @ 416 |
| SLO | hodisa <= 3 s, inferens zanjiri <= 500 ms |
| Platformalar | T4 server, L4 server, A10 server, Xeon 32 yadro, talaba brauzeri (WASM 0,5 TFLOP/s, 0 $) |
| Cheklovlar | <= 25 000 $, <= 2 kVt, yiliga 6 hafta faol |
| Tarmoq | talaba oqimi 0,5 Mbit/s (2 kadr/s JPEG) yoki faqat hodisalar (brauzer varianti) |
| TCO | 3 yil, bulut T4 0,35 $/soat, L4 0,7 $/soat, faqat 6 hafta/yil |

Qadamlar:

1. (s1) 2 ta model zanjiri (yengil/og'ir) uchun bir kadr GFLOP yig'indisi, 3 000 va 5 000 inferens/s uchun TFLOP/s va kerakli cho'qqi TOPS (samaradorlik 0,4); kiruvchi JPEG oqimi 1 500 x 0,5 Mbit/s va 1 Gbit/s port yetarliligi; brauzer varianti uchun talaba kompyuteriga tushadigan yuklama (GFLOP/s).
2. (s2) 5 platforma jadvali (brauzer satri: 1 500 x 0,5 TFLOP/s taqsimlangan, lekin ishonchsiz); zaxira, kadr energiyasi; 7 mezonli matritsa (zaxira, kechikish, narx, quvvat, maxfiylik, ishonchlilik, ekotizim); TANLOV ikki ssenariyda: 'hammasi serverda' va 'brauzer + server tasdiqlash'.
3. (s3) 3 000 ms byudjet: yuklash (median 400 ms, sigma 0,7 - uy interneti), navbat, zanjir inferensi (3 model ketma-ket), qoida mantiqi, nazoratchi UI; 20 000 kadr Monte-Karlo; batch 1/8/32 (3 000 inferens/s da kutish kichik) ta'siri.
4. (s4) Roofline: T4/L4/A10/Xeon tizza nuqtalari; SCRFD, MobileFaceNet, ArcFace-R50 FP16 batch 1/32 AI va rejim; kichik modellar uchun batch'ning rolini raqam bilan ko'rsating.
5. (s5) torch'da MiniFace (4 conv + global pool + Linear 128 embedding) 112x112 va MiniYolo 320x320; hook va FlopCounterMode; batch 1/32 CPU vaqti.
6. (s6) Co-design: 2 zanjir x imgsz (yolov8 uchun 320/416) x FP16/INT8 x batch 1/8/32 x 4 server platformasi; cheklovlar zaxira >= 1,7 (pik), zanjir inferensi <= 500 ms, narx, quvvat; Pareto narx-sifat (sifat proxysi: yengil 0,88, og'ir 0,95).
7. (s7) TCO 3 yil: o'z L4 server(lar)i (yil bo'yi elektr, 6 hafta foydali) vs bulut L4 faqat 6 hafta vs gibrid (brauzer + 1 server); talaba soni 500-5 000 uchun tenglik nuqtasi va $/imtihon.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar) — 25, Hisob-kitob va simulyatsiya to'g'riligi — 25, Grafiklar (Roofline, byudjet, co-design, TCO) — 15, Tahlil, tanlov asosi va xavflar — 25, Hisobot sifati — 10 (jami 100)

## 8-variant. Shisha idish zavodida yuqori tezlikli nuqson nazorati (4 liniya) (qiyin)

Soha: Sanoat

Zavodda 4 ta ishlab chiqarish liniyasi har biri sekundiga 20 ta idish chiqaradi; har idish 3 ta kameradan (2048x1536, monoxrom) suratga olinadi, ya'ni jami 240 tasvir/s. Nuqsonli idish 30 ms ichida rad etuvchi pnevmatik klapanga signal berilmasa konveyerdan o'tib ketadi. Model - segmentatsiya + tasniflash (nuqson 7 sinf), tasvirni 1024x768 yoki 512x384 ga tushirish mumkin, lekin mayda yoriqlar 512 da yo'qoladi (sifat proxysi beriladi). Sex harorati +50 C, chang, har liniya shkafida 150 Vt; byudjet 4 liniya uchun 30 000 $; to'xtash 1 soati 8 000 $ zarar, shuning uchun zaxira va ishonchlilik muhim.

| Parametr | Qiymat |
|---|---|
| Yuklama | 4 liniya x 20 idish/s x 3 kamera = 240 tasvir/s (2048x1536 mono) |
| Modellar (GFLOP @1024x768, sifat) | SegFormer-B0: 25, 0,90; U-Net-R18: 70, 0,94; U-Net-R34: 110, 0,96; 512x384 da FLOP/4, sifat -0,05 |
| SLO | p99 <= 30 ms uchdan-uchga, inferens <= 15 ms |
| Platformalar | Orin NX, AGX Orin, RTX A4000 (sanoat PC), L4, Hailo-8 x4, Xeon 16 AMX |
| Cheklovlar | +50 C, <= 150 Vt/liniya, <= 30 000 $, zaxira >= 2 (ishonchlilik) |
| Kirish oqimi | 2048x1536x1 bayt x 60 tasvir/s har liniya (GigE Vision) |
| TCO | 5 yil, elektr 0,09 $/kVt*soat, to'xtash zarari 8 000 $/soat, buzilish ehtimoli yiliga 5 % |

Qadamlar:

1. (s1) Liniya (60 tasvir/s) va zavod (240/s) uchun 3 model x 2 o'lcham jadvali: GFLOP, vazn FP16/INT8, TFLOP/s, kerakli cho'qqi (samaradorlik 0,45); GigE Vision oqimi (MB/s va Gbit/s) har liniya uchun va 10 GbE zarurligini aniqlang; xom kadr xotira yozuvi GB/s.
2. (s2) 6 platforma: zaxira liniya va zavod yuklamasiga, inferens_J, GigE dekod talab qilmasligini hisobga oling; 8 mezonli matritsa (zaxira >= 2, kechikish, harorat, quvvat, narx, ishonchlilik/MTBF, format, ekotizim); liniya-boshi (edge) va zavod-markaz TANLOVI.
3. (s3) 30 ms byudjet: kamera ekspozitsiya+uzatish (5 ms), oldindan ishlov, inferens, qaror+klapan (2 ms); lognormal Monte-Karlo 50 000 tasvir; batch 1/3/6 (3 kamera bir idish) ta'siri; p99 <= 30 ms ta'minlaydigan maksimal batch'ni toping; ikkita grafik.
4. (s4) Roofline: 6 platforma tizza nuqtasi; U-Net-R18 FP16/INT8 batch 1/3/6 AI (faollashuvlar 1024x768 uchun ~400 MB); kichik o'lcham 512x384 da AI qanday o'zgaradi; GEMM va elementlararo amal nuqtalari.
5. (s5) torch'da MiniSeg (enkoder 4 conv stride 2, dekoder 2 ConvTranspose2d, 7 sinf) 256x192 va 512x384 kirish (1 kanal); hook va FlopCounterMode; batch 1/3 CPU vaqti; ConvTranspose FLOP formulasini alohida tekshiring.
6. (s6) Co-design grid: 3 model x 2 o'lcham x 2 format x 3 batch x 6 platforma; cheklovlar zaxira >= 2, inferens+kutish <= 15 ms, quvvat, narx (4 liniya), harorat; Pareto narx-sifat; har cheklovni navbat bilan olib tashlab yaroqli soni jadvali (eng 'qimmat' cheklov).
7. (s7) TCO 5 yil: edge (har liniyada Orin NX/AGX) vs markaz L4 server (zaxira server bilan N+1) vs A4000 sanoat PC har liniyada; to'xtash zararini (buzilish 5 %/yil x tiklash 4 soat) qo'shing; liniya soni 1-12 tenglik nuqtasi; sezgirlik: elektr 0,05-0,2 $.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z); Strukturaviy sxema: GigE Vision kamera -> frame grabber -> model -> PLC (PROFINET/EtherCAT) interfeyslari va kechikish byudjeti jadvali; 3 ta xavf va kamaytirish chorasi (skript matn sifatida chop etsin)

Baholash: Kod: parametrlangan, qayta ishlatiladigan funksiyalar — 25, Hisob, simulyatsiya va sezgirlik tahlili to'g'riligi — 20, Grafiklar va Pareto/tenglik nuqtasi tahlili — 20, Muhandislik xulosasi, strukturaviy sxema, xavflar — 25, Hisobot sifati — 10 (jami 100)

## 9-variant. Aeroport bagaj rentgen tasvirlarida taqiqlangan buyumlarni aniqlash (o'rta)

Soha: Xavfsizlik

Xalqaro aeroportda 6 ta rentgen skaner (ikki energiyali, 1024x768 tasvir) ishlaydi, har skanerdan tasma tezligi 0,25 m/s bo'lgani uchun o'rtacha 1,5 bagaj/s chiqadi, pik 2,5 bagaj/s. Har bagaj uchun 2 proyeksiya (yuqori va yon) tahlil qilinadi, 12 sinf taqiqlangan buyum aniqlanadi. Operator ekranida natija 1 s ichida ko'rinishi kerak; skanerlar terminalning turli qavatlarida; xavfsizlik talablari tufayli bulut mumkin emas, lekin terminal server xonasi bor. Byudjet 20 000 $, skaner shkafida 80 Vt; sifat kritik - mAP 0,80 dan past model qabul qilinmaydi.

| Parametr | Qiymat |
|---|---|
| Yuklama | 6 skaner x 1,5 bagaj/s x 2 proyeksiya = 18 inferens/s (pik 30/s) |
| Modellar | yolov8m (78,9 GFLOP, mAP 0,84), yolov8l (165,2, 0,87), RT-DETR-R50 (136, 0,88) @ 640; imgsz 640/1024 |
| SLO | p99 <= 1 000 ms, inferens <= 300 ms |
| Platformalar | Orin NX, AGX Orin, RTX A2000 (70 Vt, 153 INT8 TOPS, 650 $), T4 server, L4 server |
| Cheklovlar | <= 20 000 $, <= 80 Vt skanerda, mAP >= 0,80, bulut yo'q |
| Formatlar | FP16, INT8 (INT8 mAP -1,5 punkt; RT-DETR INT8 -3) |
| TCO | 5 yil, 6 -> 12 skaner kengayish, elektr 0,08 $/kVt*soat |

Qadamlar:

1. (s1) 18 va 30 inferens/s uchun 3 model x 2 imgsz jadvali: GFLOP, vazn, TFLOP/s, kerakli cho'qqi (samaradorlik 0,4); 1024x768x2 bayt x 2 proyeksiya tasvir oqimi MB/s har skaner va server uchun.
2. (s2) 5 platforma: zaxira pikka, inferens_J; 7 mezonli matritsa (zaxira, kechikish, sifat >= 0,80 ni qo'llab-quvvatlovchi model sig'imi, quvvat, narx, xavfsizlik/joylashuv, ekotizim); skaner-boshi va markaz TANLOVI.
3. (s3) 1 000 ms byudjet: skaner -> tasvir shakllantirish (median 150 ms), uzatish, oldindan ishlov, inferens (2 proyeksiya), operator UI render (80 ms); 20 000 bagaj Monte-Karlo; batch 1/2/4 (2 proyeksiya birga) ta'siri va p99.
4. (s4) Roofline: 5 platforma tizza nuqtasi; yolov8l va RT-DETR FP16/INT8 batch 1/2/4 AI va rejim; RT-DETR'ning attention qismi uchun alohida nuqta (AI past) qo'shing.
5. (s5) MiniYolo 12 sinf, 1 kanal (rentgen) kirish 256x192 va 512x384; hook va FlopCounterMode; batch 1/2 CPU vaqti; imgsz^2 proporsionallik.
6. (s6) Co-design: 3 model x 2 imgsz x 2 format x 3 batch x 5 platforma; cheklovlar zaxira >= 1,5, inferens <= 300 ms, quvvat, narx (6 skaner), mAP >= 0,80 (INT8 jarima bilan); Pareto narx-mAP; INT8 jarimasini -3 punktga oshirganda top-5 qanday o'zgaradi.
7. (s7) TCO 5 yil: har skanerda Orin NX/A2000 vs markaz T4/L4 server; 6 va 12 skaner uchun jadval, 1-16 skaner tenglik nuqtasi; markaz varianti uchun zaxira (N+1) serverni qo'shib qayta hisoblang.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar) — 25, Hisob-kitob va simulyatsiya to'g'riligi — 25, Grafiklar (Roofline, byudjet, co-design, TCO) — 15, Tahlil, tanlov asosi va xavflar — 25, Hisobot sifati — 10 (jami 100)

## 10-variant. Kassasiz do'kon: xaridorlarni kuzatish va javondan olingan tovarni aniqlash (qiyin)

Soha: Chakana savdo

Chakana tarmoq 400 m2 maydonli kassasiz do'kon ochmoqda: shiftda 48 ta kamera (1080p 15 kadr/s), javonlarda 600 ta og'irlik sensori (10 Hz). Har kadrda odamlar aniqlanadi, poza (17 nuqta) baholanadi va qayta identifikatsiya (ReID) embedding olinadi; sensor hodisasi bilan birlashtirilib 'kim nima oldi' hal qilinadi. Xaridor chiqishida chek 2 s ichida shakllanishi kerak, lekin kuzatuv zanjiri har kadrda 300 ms dan oshmasligi lozim, aks holda traektoriyalar uziladi. Do'kon orqa xonasida 1,5 kVt va 2 ta server o'rni bor; byudjet 45 000 $; tarmoq 10 GbE. Savdo kuni 16 soat, tungi vaqtda tizim o'chadi.

| Parametr | Qiymat |
|---|---|
| Yuklama | 48 kamera x 15 kadr/s = 720 kadr/s; har kadrda o'rtacha 3 odam -> poza+ReID 2 160 inferens/s; 600 sensor x 10 Hz |
| Modellar | yolov8s odam (28,6 GFLOP @640); RTMPose-m (1,9 GFLOP/odam); OSNet ReID (1,0 GFLOP/odam); og'ir: yolov8m + RTMPose-l (4,5) |
| SLO | zanjir p99 <= 300 ms/kadr, chek <= 2 s |
| Platformalar | 2 x L4 server, 2 x T4 server, 1 x A100 server, 4 x AGX Orin, 2 x RTX 4090 ish stansiyasi |
| Cheklovlar | <= 45 000 $, <= 1 500 Vt, 2 server o'rni, 10 GbE |
| Dekodlash | 48 x 1080p15 oqim (NVDEC 1080p30 soni: L4 - 8 muvozanat, T4 - 7, A100 - 10, AGX - 14, 4090 - 2 x NVDEC ~ 14) |
| TCO | 3 yil, 16 soat/kun, elektr 0,11 $/kVt*soat, 1 -> 10 do'kon |

Qadamlar:

1. (s1) Uch bosqichli zanjir (detektor + poza + ReID) uchun bir kadr jami GFLOP (3 odam bilan) va 720 kadr/s uchun TFLOP/s; yengil va og'ir zanjir; kerakli cho'qqi (samaradorlik 0,4); 48 oqim x 4 Mbit/s tarmoq, xom kadr GB/s, 600 sensor oqimi kbit/s.
2. (s2) 5 konfiguratsiya (server to'plamlari) jadvali: jami TOPS, dekoder sig'imi 48 oqim uchun, Vt, narx; zaxira, kadr energiyasi; 8 mezonli matritsa (zaxira, kechikish, dekod, quvvat, narx, o'rin soni, ishonchlilik N+1, ekotizim/DeepStream); TANLOV.
3. (s3) 300 ms byudjet: dekod, detektor, poza (odam soni bilan o'sadi: 1-8 odam), ReID, kuzatuv + sensor birlashtirish (median 20 ms, sigma 0,5); odam soni Poisson(3) tasodifiy; 20 000 kadr Monte-Karlo p99; batch 1/8/16 ta'siri; 'juma kechqurun' ssenariysida (6 odam/kadr) byudjet buziladimi.
4. (s4) Roofline: L4, T4, A100, AGX Orin, 4090 tizza nuqtalari; yolov8s, RTMPose-m, OSNet FP16/INT8 batch 1/8/32 AI va rejim; kichik ReID modelini katta batch'da yig'ish nima uchun zarurligini ko'rsating.
5. (s5) torch'da MiniYolo (1 sinf), MiniPose (4 conv + 17 kanal heatmap 64x48) va MiniReID (4 conv + GAP + Linear 128) uchun hook va FlopCounterMode; batch 1/8/32 CPU vaqti; zanjir jami ms/kadr 3 odam uchun.
6. (s6) Co-design grid: 2 zanjir x imgsz (detektor 512/640) x 2 format x batch 1/8/16 x 5 konfiguratsiya; cheklovlar zaxira >= 1,5, zanjir <= 150 ms inferens, dekod, quvvat, narx; Pareto narx-sifat (proxy: yengil 0,90, og'ir 0,95; INT8 -0,01); har platforma uchun eng yaxshi.
7. (s7) TCO 3 yil: do'kon ichida 2 x L4 vs 4 x AGX Orin vs 2 x 4090 vs markaziy A100 (bir necha do'kon uchun, do'kon -> markaz 1 Gbit/s ijarasi 2 400 $/yil); do'konlar soni 1-10; 16 soat/kun ish rejimini energiyada hisobga oling; tenglik nuqtasi.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z); Strukturaviy sxema: kamera/sensor -> dekod -> zanjir -> hodisalar ombori -> chek xizmati interfeyslari (RTSP, MQTT, gRPC) va o'tkazuvchanlik; Sezgirlik: odam/kadr 2-6 oralig'ida zaxira va p99 grafigi

Baholash: Kod: parametrlangan, qayta ishlatiladigan funksiyalar — 25, Hisob, simulyatsiya va sezgirlik tahlili to'g'riligi — 20, Grafiklar va Pareto/tenglik nuqtasi tahlili — 20, Muhandislik xulosasi, strukturaviy sxema, xavflar — 25, Hisobot sifati — 10 (jami 100)

## 11-variant. Omborda paketlarni saralash: shtrix-kod va manzil yorlig'ini o'qish (oson)

Soha: Logistika

Pochta-logistika kompaniyasining saralash markazida 12 ta saralash liniyasi bor; har liniyadan sekundiga 4 ta paket o'tadi (1080p kamera, paket o'tganda 1 kadr). Har paket uchun yorliq aniqlanadi (detektor), so'ng shtrix-kod va manzil matni o'qiladi (OCR). Saralash qurilmasiga yo'nalish 400 ms ichida yuborilmasa paket 'noma'lum' tarmog'iga ketadi va qo'lda qayta ishlanadi (har biri 0,5 $ xarajat). Ombor isitilmaydi (-10...+40 C), har liniyada 40 Vt quvvat, markazda kichik server xonasi; byudjet 15 000 $.

| Parametr | Qiymat |
|---|---|
| Yuklama | 12 liniya x 4 paket/s = 48 inferens zanjiri/s (pik 80/s) |
| Modellar | yolov8n yorliq @ 640 (8,7 GFLOP) + CRNN OCR 32x320 (1,2 GFLOP/qator, 3 qator); og'ir: yolov8s + SVTR-small OCR (3,5/qator) |
| SLO | p99 <= 400 ms, inferens zanjiri <= 150 ms |
| Platformalar | Orin Nano, Orin NX, Hailo-8 + x86, RK3588, T4 server (markaz) |
| Cheklovlar | <= 40 Vt liniyada, -10...+40 C, <= 15 000 $ |
| Xato narxi | noto'g'ri yo'naltirilgan paket 0,5 $; o'qish aniqligi yengil 0,96, og'ir 0,985 |
| TCO | 3 yil, 12 -> 24 liniya, elektr 0,07 $/kVt*soat |

Qadamlar:

1. (s1) 48 va 80 zanjir/s uchun 2 zanjir (detektor + 3 OCR qatori) GFLOP, TFLOP/s va kerakli cho'qqi (samaradorlik 0,4); OCR uchun 3 qator x 1,2 GFLOP hisobini alohida ko'rsating; bir paket tasviri (JPEG 300 KB) oqimi MB/s.
2. (s2) 5 platforma jadvali: zaxira liniya (4/s) va markaz (48/s) uchun, inferens_J; 6 mezonli matritsa (zaxira, kechikish, harorat, quvvat, narx, ekotizim) va TANLOV; xato narxi (0,5 $ x xato ulushi x paketlar/yil) ni og'ir model foydasiga qo'shimcha ustun qilib hisoblang.
3. (s3) 400 ms byudjet: suratga olish trigger (10 ms), uzatish, detektor, OCR (3 qator ketma-ket yoki batch 3), saralash buyrug'i (PLC 20 ms); 20 000 paket Monte-Karlo p99; OCR qatorlarini batch qilish kechikishni qanday kamaytiradi.
4. (s4) Roofline: Orin Nano/NX, RK3588, T4 tizza nuqtalari; yolov8n INT8 batch 1/4 va CRNN FP16 batch 3/12 AI va rejim.
5. (s5) MiniYolo (1 sinf yorliq) 256x256 va MiniCRNN (3 conv + GRU 64 + Linear 37 belgi) 32x160 kirish; hook va FlopCounterMode (GRU formulasini qo'lda qo'shing); batch 1/8 CPU vaqti.
6. (s6) Co-design: 2 zanjir x imgsz 480/640 x FP16/INT8 x batch 1/4/8 x 5 platforma; cheklovlar zaxira >= 1,5, inferens <= 150 ms, quvvat, narx (12 liniya), harorat; Pareto narx-aniqlik.
7. (s7) TCO 3 yil: har liniyada Orin Nano vs markaz T4 server; 12 va 24 liniya; xato xarajatini (paketlar/yil x xato ulushi x 0,5 $) TCO'ga qo'shib qaysi model-platforma eng arzon ekanini toping.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar, hisob-kitob va jadvallar — 25, Tahlil va asoslangan tanlov — 25, Hisobot sifati (grafik, sxema, til) — 20 (jami 100)

## 12-variant. Shahar avtoturargohlari: kirish-chiqishda davlat raqamini tanish (ANPR) (oson)

Soha: Shahar

Shahar hokimligi 40 ta ko'p qavatli va ochiq avtoturargohni yagona to'lov tizimiga ulamoqchi. Har avtoturargohda 2 ta shlagbaum kamerasi (1080p) bor; avtomobil kelganda 2 kadr/s tahlil qilinadi va raqam 1 s ichida tanilib shlagbaum ochilishi kerak, aks holda navbat hosil bo'ladi. Avtoturargoh soatiga pikda 300 avtomobil qabul qiladi. Shlagbaum shkafida 15 Vt, harorat -25...+45 C; markaziy ma'lumot markazi mavjud, avtoturargohlar 4G/optika bilan ulangan (kechikish 30-150 ms). Byudjet avtoturargoh boshiga 1 200 $; 2 yilda avtoturargohlar soni 100 taga yetadi.

| Parametr | Qiymat |
|---|---|
| Yuklama | 40 x 2 kamera x 2 kadr/s = 160 inferens/s (faqat harakat bo'lganda ~30 %) |
| Modellar | yolov8n raqam @ 416 (3,7 GFLOP) + CRNN 32x128 (0,5 GFLOP); og'ir: yolov8s @ 640 + SVTR (2,0) |
| SLO | p95 <= 1 000 ms (shlagbaum), inferens <= 200 ms |
| Platformalar | RK3588 (6 TOPS), Orin Nano, Hailo-8L + RPi 5, x86 mini PC (4 yadro), T4 server (markaz) |
| Cheklovlar | <= 15 Vt, -25...+45 C, <= 1 200 $/avtoturargoh |
| Tarmoq | 4G (median 80 ms, sigma 0,7) yoki optika (median 20 ms) |
| TCO | 3 yil, 40 -> 100 avtoturargoh, 4G ijarasi 120 $/yil |

Qadamlar:

1. (s1) Bitta avtoturargoh (4 inferens/s) va markaz (160/s, pik 100 %) uchun 2 zanjir GFLOP, TFLOP/s, kerakli cho'qqi (samaradorlik 0,35); kadr JPEG (200 KB) oqimi 4G kanal (10 Mbit/s) ga sig'ishini tekshiring.
2. (s2) 5 platforma: zaxira avtoturargoh va markaz yuklamasiga, inferens_J; 6 mezonli matritsa (zaxira, kechikish, harorat, quvvat, narx, ekotizim); edge va markaz uchun TANLOV; 4G uzilganda edge ishlashda davom etishini mezon sifatida qo'shing.
3. (s3) 1 000 ms byudjet: harakat trigger, kadr olish, detektor, OCR, to'lov bazasi so'rovi (median 60 ms), shlagbaum rele (50 ms); markaz uchun +4G; 20 000 avtomobil Monte-Karlo p95/p99; 4G sigma 0,7 bo'lganda markaz varianti SLO'ni bajaradimi.
4. (s4) Roofline: RK3588 (25 GB/s), Orin Nano (68), x86 mini PC (40), T4 (320); yolov8n INT8 batch 1 va CRNN batch 1/8 AI va rejim.
5. (s5) MiniYolo (1 sinf) 256x256 va MiniCRNN 32x128 (3 conv + GRU 48 + Linear 40); hook va FlopCounterMode; batch 1/8 CPU vaqti; bir avtomobil uchun zanjir ms.
6. (s6) Co-design: 2 zanjir x imgsz 416/640 x FP16/INT8 x batch 1/2/4 x 5 platforma; cheklovlar zaxira >= 1,5, inferens <= 200 ms, quvvat, narx, harorat; Pareto narx-aniqlik (yengil 0,95, og'ir 0,985).
7. (s7) TCO 3 yil: har avtoturargohda RK3588/Orin Nano vs markaz T4 + 4G ijarasi; 40 va 100 avtoturargoh; tenglik nuqtasi 1-120 oralig'ida; xulosada 4G ishonchliligi xavfini yozing.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar, hisob-kitob va jadvallar — 25, Tahlil va asoslangan tanlov — 25, Hisobot sifati (grafik, sxema, til) — 20 (jami 100)

## 13-variant. O'rmon xo'jaligida yong'in tutunini quyosh batareyali minora kameralaridan aniqlash (oson)

Soha: Ekologiya

Tog'li o'rmon xo'jaligi 10 ta kuzatuv minorasiga 4K panoramali kamera o'rnatmoqda; kamera 360 gradusni 6 ta 1280x720 bo'lakka bo'lib har 5 s da bir marta tahlil qiladi (0,2 kadr/s x 6 bo'lak). Minorada elektr yo'q: quyosh paneli 100 Vt + akkumulyator 1,2 kVt*soat, hisoblagichga o'rtacha 8 Vt ajratiladi (qish kunlari 4 Vt). Aloqa - LoRa/4G kam tezlikli (0,3 Mbit/s), shuning uchun video markazga yuborilmaydi, faqat ogohlantirish va kichik rasm. Tutun 60 s ichida aniqlanishi kerak; yolg'on ogohlantirish narxi yuqori (vertolyot chiqadi). Byudjet minora boshiga 900 $.

| Parametr | Qiymat |
|---|---|
| Yuklama | 10 minora x 6 bo'lak x 0,2 kadr/s = 12 inferens/s (minora: 1,2/s) |
| Modellar | yolov8n tutun @ 640 (8,7 GFLOP, F1 0,86), yolov8s @ 640 (28,6, 0,91), yolov8s @ 1280 (114, 0,94) |
| SLO | aniqlash <= 60 s, bo'lak zanjiri <= 5 s |
| Platformalar | Orin Nano (7/15 Vt rejim), RK3588, Hailo-8L + RPi 5, Coral Edge TPU + RPi 4, x86 markaz serveri (faqat solishtirish) |
| Cheklovlar | <= 8 Vt o'rtacha (qish 4 Vt), <= 900 $, -30...+40 C |
| Energiya | quyosh 100 Vt panel, 1,2 kVt*soat akkumulyator, qishda 3 kun quyoshsiz |
| TCO | 5 yil, akkumulyator almashtirish 3 yilda, 10 -> 30 minora |

Qadamlar:

1. (s1) Minora (1,2 inferens/s) va butun tarmoq uchun 3 konfiguratsiya GFLOP, TFLOP/s, kerakli cho'qqi (samaradorlik 0,35); kunlik energiya (Vt*soat) = quvvat x 24 soat va 3 quyoshsiz kunda akkumulyator yetarliligini hisoblang.
2. (s2) 4 edge platforma + markaz satri: zaxira, inferens_J, kunlik Vt*soat; 6 mezonli matritsa (zaxira, energiya byudjeti, harorat, narx, ekotizim, oflayn ishlash); TANLOV; quyosh paneli 60 Vt bo'lsa tanlov o'zgaradimi.
3. (s3) 5 000 ms byudjet bo'lak uchun: kamera burilish+ekspozitsiya (median 800 ms), olish, inferens, vaqt bo'yicha tasdiqlash (3 ketma-ket kadr), LoRa yuborish (median 1 500 ms, sigma 0,6); 10 000 sikl Monte-Karlo; 60 s aniqlash talabiga 3-kadr tasdiq sig'adimi.
4. (s4) Roofline: Orin Nano 7 Vt rejimda (cho'qqi 40 %), RK3588, Coral, Hailo-8L; yolov8n/s INT8 batch 1/6 (6 bo'lak birga) AI va rejim.
5. (s5) MiniYolo (1 sinf tutun) 320x180 va 640x360 kirish; hook va FlopCounterMode; batch 1/6 CPU vaqti; 1 Vt*soat energiyada nechta inferens bajarilishini CPU o'lchovidan (TDP taxminiy 15 Vt) baholang.
6. (s6) Co-design: 3 konfiguratsiya x FP16/INT8 x batch 1/6 x 4 platforma x 2 quvvat rejimi (yoz/qish); cheklovlar zaxira >= 1,5, inferens <= 2 s, o'rtacha quvvat, narx; Pareto narx-F1 va energiya-F1.
7. (s7) TCO 5 yil: minora edge (platforma + panel + akkumulyator) vs 'rasmlarni 4G orqali markazga yuborish' (4G ijarasi 240 $/yil + markaz server); 10 va 30 minora; tenglik nuqtasi yo'qligini yoki borligini ko'rsating.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar, hisob-kitob va jadvallar — 25, Tahlil va asoslangan tanlov — 25, Hisobot sifati (grafik, sxema, til) — 20 (jami 100)

## 14-variant. Futbol stadionida o'yinchilar va to'pni ko'p kamerali real vaqt kuzatuvi (qiyin)

Soha: Sport

Professional klub stadionida 8 ta statsionar 4K kamera (3840x2160, 50 kadr/s) o'yin maydonini qoplaydi. Har kadrda 22 o'yinchi + hakamlar + to'p aniqlanadi, poza baholanadi va murabbiy planshetiga 3D traektoriyalar 150 ms ichida yetkaziladi (taktik tahlil va ofsayd yordami). To'p kichik (4K da ~12 piksel), shuning uchun kirish o'lchamini 1280 dan pastga tushirish sifatni keskin pasaytiradi. Stadionda texnik xona (3 kVt, 42U shkaf) bor; o'yin haftasiga 1-2 marta 2 soat, qolgan vaqt tizim bo'sh. Byudjet 80 000 $; bulut variantida 8 x 4K oqimni yuborish uchun 400 Mbit/s kanal kerak.

| Parametr | Qiymat |
|---|---|
| Yuklama | 8 kamera x 50 kadr/s = 400 kadr/s 4K; har kadrda ~25 obyekt -> poza 10 000 inferens/s |
| Modellar | yolov8m @ 1280 (316 GFLOP, mAP 0,78 to'p), yolov8l @ 1280 (661, 0,84), yolov8x @ 1920 (2 320, 0,88); RTMPose-m 1,9 GFLOP/odam |
| SLO | p99 <= 150 ms, inferens <= 60 ms |
| Platformalar | 2 x A100 server, 4 x L4 server, 2 x RTX 4090 ish stansiyasi, 1 x H100 PCIe (1 513 INT8 TOPS dense, 350 Vt, 25 000 $), 8 x AGX Orin (kamera boshida) |
| Cheklovlar | <= 80 000 $, <= 3 000 Vt, 4K dekod 8 x 2160p50 |
| Dekodlash | 4K50 oqim = 4 x 1080p50 = 6,7 x 1080p30; NVDEC soni: A100 - 5 (1080p30 ~ 10), L4 - 4 (~ 8), 4090 - 2 (~ 14), H100 - 7, AGX - 14 |
| TCO | 3 yil, 100 soat/yil foydalanish, bulut A100 3 $/soat + 400 Mbit/s kanal 1 500 $/oy |

Qadamlar:

1. (s1) 400 kadr/s uchun 3 detektor konfiguratsiyasi + poza (25 obyekt) GFLOP va TFLOP/s; kerakli cho'qqi (samaradorlik 0,5 katta batch'da); 4K kadr xom hajmi (24,9 MB) x 400 = GB/s xotira yozuvi; 8 x 50 Mbit/s HEVC tarmoq.
2. (s2) 5 konfiguratsiya jadvali: jami TOPS, dekod sig'imi (4K50 oqimlar), Vt, narx; zaxira, kadr energiyasi; 8 mezonli matritsa (zaxira, kechikish, dekod, quvvat, narx, sifat to'p, ishonchlilik, ekotizim); TANLOV va kamera-boshi (AGX) varianti nima uchun yiqilishini raqam bilan ko'rsating.
3. (s3) 150 ms byudjet: 4K dekod (median 12 ms), tile'larga bo'lish, detektor, poza (25 obyekt batch), ko'p kamerali triangulyatsiya (median 15 ms, sigma 0,4), planshetga Wi-Fi (median 20 ms, sigma 0,6); 20 000 kadr Monte-Karlo; batch 1/4/8 (8 kamera sinxron kadr) ta'siri; p99 <= 150 ms maksimal batch.
4. (s4) Roofline: A100, L4, 4090, H100 (3 350 GB/s), AGX Orin tizza nuqtalari; yolov8l @1280 FP16/INT8 batch 1/8 va RTMPose batch 25/200 AI va rejim; 4090 (1 008 GB/s) va L4 (300 GB/s) farqi.
5. (s5) MiniYolo 2 sinf (odam, to'p) 256x144 / 512x288 / 1024x576 kirish; hook va FlopCounterMode; batch 1/8 CPU vaqti; to'p uchun kichik obyekt oqibatida imgsz pasaytirib bo'lmasligini FLOP o'sishi bilan birga tahlil qiling.
6. (s6) Co-design grid: 3 detektor x 2 format x batch 1/4/8 x 5 konfiguratsiya (+ poza yuklamasi qo'shiladi); cheklovlar zaxira >= 1,3, inferens+kutish <= 60 ms, dekod, quvvat, narx; Pareto narx-mAP(to'p); har cheklovni olib tashlab yaroqli soni jadvali.
7. (s7) TCO 3 yil: o'z serveri (100 soat/yil, lekin 24/7 'idle' 15 % quvvat) vs bulut A100 soatbay + 400 Mbit/s kanal vs 2 x 4090 ish stansiyasi; foydalanish soatlari 50-1 000/yil bo'yicha tenglik nuqtasi grafigi.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z); Strukturaviy sxema: 4K kamera (SDI/IP) -> dekod -> detektor/poza -> triangulyatsiya -> planshet (WebSocket) interfeyslari va o'tkazuvchanlik; 3 ta xavf (dekod chegarasi, to'p sifati, Wi-Fi) va kamaytirish choralari

Baholash: Kod: parametrlangan, qayta ishlatiladigan funksiyalar — 25, Hisob, simulyatsiya va sezgirlik tahlili to'g'riligi — 20, Grafiklar va Pareto/tenglik nuqtasi tahlili — 20, Muhandislik xulosasi, strukturaviy sxema, xavflar — 25, Hisobot sifati — 10 (jami 100)

## 15-variant. Telekanallar uchun jonli efirda avtomatik subtitr (nutqni tanish) serveri (o'rta)

Soha: Media

Milliy teleradiokompaniya 24 ta telekanal uchun jonli efirda o'zbek va rus tillarida avtomatik subtitr yaratmoqchi. Audio 16 kHz mono, 10 s bo'laklarga bo'linib Whisper oilasidagi modelga beriladi; subtitr efirdan 3 s dan ko'p kechikmasligi kerak (qonun talabi). Yuklama tasvir emas, audio soniya bilan hisoblanadi: 24 kanal x 1 real vaqt = 24 audio-soniya/s, ya'ni har 10 s da 24 ta bo'lak. Yangiliklar vaqtida sifat (WER) ustuvor, tungi takrorlarda tejamkor model yetarli. Server xonasi mavjud, 1,5 kVt bo'sh; byudjet 30 000 $; bulut (ma'lumot chiqib ketishi) ruxsat etilmaydi.

| Parametr | Qiymat |
|---|---|
| Yuklama | 24 kanal x real vaqt; bo'lak 10 s -> 2,4 bo'lak/s (30 s bo'lak: 0,8/s) |
| Modellar (GFLOP / 30 s audio, WER) | whisper-tiny: 60, 0,28; whisper-base: 120, 0,22; whisper-small: 400, 0,15; bo'lak 10 s uchun FLOP/3 |
| SLO | subtitr kechikishi <= 3 s (bo'lak 10 s bo'lsa: inferens <= 2 s) |
| Platformalar | Xeon 32 yadro (AMX), T4 server, L4 server, A10 server, AGX Orin x3 |
| Cheklovlar | <= 30 000 $, <= 1 500 Vt, bulut yo'q |
| Formatlar | FP16, INT8 (WER +0,01), batch 1/8/24 (24 kanal birga) |
| TCO | 5 yil, 24 -> 48 kanal, elektr 0,08 $/kVt*soat |

Qadamlar:

1. (s1) KAMERALAR*FPS o'rniga kanal x bo'lak chastotasi: 3 model x 2 bo'lak uzunligi (10/30 s) uchun GFLOP/bo'lak, TFLOP/s talab, kerakli cho'qqi (samaradorlik 0,3 - avtoregressiv dekoder past samarali); vazn FP16/INT8 MB; audio oqimi (16 kHz x 2 bayt x 24) kB/s.
2. (s2) 5 platforma: zaxira 24 va 48 kanal uchun, bo'lak energiyasi (J); 6 mezonli matritsa (zaxira, kechikish, narx, quvvat, WER sig'imi, ekotizim/CTranslate2); TANLOV; AGX Orin x3 varianti nima uchun quvvatda yutadi, lekin ekotizimda yutqazadi.
3. (s3) 3 000 ms byudjet: bo'lak to'planishi (10 s bo'lak uchun kechikish bo'lak uzunligiga teng - buni alohida ko'rsating: shuning uchun 'sirpanuvchi oyna' 2 s qadam), VAD, enkoder, dekoder (token soni 20-80, uniform), subtitr formatlash (SRT/WebVTT), efirga qo'shish (200 ms); 10 000 bo'lak Monte-Karlo; batch 1/8/24 kutish.
4. (s4) Roofline: Xeon (300 GB/s), T4, L4, A10, AGX Orin; enkoder (AI yuqori) va dekoder qadami (AI past, vazn o'qish ustun) nuqtalarini batch 1/8/24 uchun alohida joylashtiring; qaysi qism xotira bilan cheklangan.
5. (s5) torch'da MiniWhisper (Conv1d x2 + 2 qatlam TransformerEncoderLayer d=128 + Linear) 80x300 kirish; hook (Conv1d, Linear) va FlopCounterMode (attention FLOP'ini qo'lda qo'shing); batch 1/8/24 CPU vaqti.
6. (s6) Co-design: 3 model x 2 bo'lak x 2 format x 3 batch x 5 platforma; cheklovlar zaxira >= 1,5 (48 kanal kelajak), inferens <= 2 s, narx, quvvat; Pareto narx-WER; kun/tun rejimi uchun ikki xil tanlov.
7. (s7) TCO 5 yil: L4 server vs Xeon AMX server vs 3 x AGX Orin; 24 va 48 kanal; elektr 0,08 va 0,2 $ sezgirlik; $/kanal/oy.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar) — 25, Hisob-kitob va simulyatsiya to'g'riligi — 25, Grafiklar (Roofline, byudjet, co-design, TCO) — 15, Tahlil, tanlov asosi va xavflar — 25, Hisobot sifati — 10 (jami 100)

## 16-variant. Muzey va ko'rgazmalarda vizual qidiruv kiosklari: eksponatni rasmdan tanish (oson)

Soha: Turizm

Yirik tarix muzeyi 15 ta sensorli kiosk va mobil ilova orqali 'rasmga ol - eksponat haqida o'qi' xizmatini ishga tushirmoqda: foydalanuvchi suratidan embedding (ResNet/EfficientNet) olinadi va 12 000 eksponatli bazadan eng yaqin 5 tasi topiladi. Kiosklar pikda har 2 s da 1 so'rov beradi (jami 7,5 so'rov/s), mobil ilova mavsumda yana 10 so'rov/s qo'shadi. Javob 1,5 s ichida ko'rinishi kerak, aks holda tashrif buyuruvchi ketadi. Muzeyda kichik server xonasi (500 Vt) bor; kioskning o'zida ham hisoblash mumkin (har kioskda 20 Vt). Byudjet 8 000 $. Baza sintetik: 12 000 x 512 o'lchamli normallashtirilgan vektorlar.

| Parametr | Qiymat |
|---|---|
| Yuklama | 15 kiosk x 0,5 so'rov/s + mobil 10 so'rov/s = 17,5 so'rov/s (pik x2) |
| Modellar | MobileNetV3-L @224 (0,44 GFLOP, top-1 proxy 0,85), EfficientNet-B0 (0,78, 0,89), ResNet50 (8,2, 0,92); qidiruv 12 000 x 512 skalyar ko'paytma (12,3 MFLOP) |
| SLO | p95 <= 1 500 ms, inferens+qidiruv <= 300 ms |
| Platformalar | RK3588 (kioskda), Orin Nano (kioskda), x86 mini PC 6 yadro (server), T4 server, Apple M2 Mac mini (15,8 TOPS NPU) |
| Cheklovlar | <= 8 000 $, kiosk <= 20 Vt, server <= 500 Vt |
| Tarmoq | muzey Wi-Fi (median 40 ms, sigma 0,8), surat 300 KB |
| TCO | 3 yil, 15 -> 40 kiosk, elektr 0,09 $/kVt*soat |

Qadamlar:

1. (s1) 17,5 va 35 so'rov/s uchun 3 model + qidiruv GFLOP/so'rov, TFLOP/s, kerakli cho'qqi (samaradorlik 0,3); baza hajmi (12 000 x 512 x FP32/FP16/INT8) MB va surat oqimi MB/s.
2. (s2) 5 platforma: zaxira 'kioskda' (0,5 so'rov/s) va 'serverda' (35/s) uchun, so'rov energiyasi (J); 6 mezonli matritsa (zaxira, kechikish, narx, quvvat, ekotizim, boshqaruv qulayligi - bitta server vs 15 qurilma); TANLOV.
3. (s3) 1 500 ms byudjet: surat olish+siqish (median 200 ms), Wi-Fi uzatish, oldindan ishlov, embedding, qidiruv, kontent sahifasini yuklash (median 150 ms, sigma 0,5); 20 000 so'rov Monte-Karlo p95; batch 1/4/8 - past so'rov tezligida batch kutish nima uchun zararli.
4. (s4) Roofline: 5 platforma tizza nuqtasi; MobileNetV3 va ResNet50 FP16 batch 1/8 AI; qidiruv (matritsa-vektor, AI ~0,5 FLOP/bayt) xotira bilan cheklanganini ko'rsating va baza INT8 bo'lsa qanday tezlashadi.
5. (s5) torch'da MiniEmbed (5 conv + GAP + Linear 512) 224x224 va 160x160; hook va FlopCounterMode; batch 1/8 CPU vaqti; qidiruvni torch.mm bilan 12 000 x 512 da o'lchab ms qo'shing.
6. (s6) Co-design: 3 model x imgsz 160/224 x FP16/INT8 x batch 1/4 x 5 platforma; cheklovlar zaxira >= 1,5 (pik), inferens+qidiruv <= 300 ms, narx (15 kiosk yoki 1 server), quvvat; Pareto narx-top-1.
7. (s7) TCO 3 yil: har kioskda RK3588/Orin Nano vs markaz server (mini PC yoki T4) + Wi-Fi; 15 va 40 kiosk; tenglik nuqtasi; mobil ilova uchun baribir server kerakligini xulosada hisobga oling.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar, hisob-kitob va jadvallar — 25, Tahlil va asoslangan tanlov — 25, Hisobot sifati (grafik, sxema, til) — 20 (jami 100)

## 17-variant. Qurilish kompaniyasida dron tasvirlaridan beton yoriqlarini tungi ommaviy tahlil qilish (o'rta)

Soha: Qurilish

Yirik qurilish kompaniyasi 30 ta obyektda har kuni dron bilan beton konstruksiyalarni suratga oladi: kuniga jami 24 000 ta 4000x3000 tasvir. Tasvirlar kechqurun ofisga keladi va ertalab 7:00 gacha (8 soat oynada) yoriq segmentatsiyasi va o'lchami hisoboti tayyor bo'lishi kerak; demak, bu real vaqt emas, o'tkazuvchanlik (throughput) va narx masalasi. Tasvir 1024x768 bo'laklarga (tile) bo'linadi: bir tasvirdan 16 ta bo'lak. Mavsumda (bahor-kuz) yuklama 2 baravar oshadi, qishda 3 baravar kamayadi. Ofisda 1 kVt, byudjet 20 000 $; bulut GPU (L4 0,7 $/soat, A100 3 $/soat) ruxsat etilgan, chunki tasvirlar maxfiy emas.

| Parametr | Qiymat |
|---|---|
| Yuklama | 24 000 tasvir/kun x 16 bo'lak = 384 000 bo'lak / 8 soat = 13,3 bo'lak/s (mavsum: 26,7/s) |
| Modellar (GFLOP @1024x768, IoU) | SegFormer-B0: 25, 0,72; U-Net-R18: 70, 0,78; DeepLabV3-R50: 180, 0,82 |
| SLO | butun partiya <= 8 soat (p99 emas, o'tkazuvchanlik) |
| Platformalar | RTX 4090 ish stansiyasi, L4 server, T4 server, Xeon 16 AMX, bulut L4/A100 soatbay |
| Cheklovlar | <= 20 000 $ (yoki bulut), <= 1 000 Vt, 8 soat oyna |
| Formatlar | FP16, INT8 (IoU -0,01), batch 8/32/64 |
| TCO | 3 yil, mavsumiylik (6 oy x2, 3 oy x1, 3 oy x1/3), elektr 0,1 $/kVt*soat |

Qadamlar:

1. (s1) KAMERALAR*FPS o'rniga bo'lak/s (o'rtacha, mavsum, qish) uchun 3 model GFLOP, TFLOP/s, kerakli cho'qqi (samaradorlik 0,55 - katta batch); kunlik tasvir hajmi (JPEG 6 MB x 24 000) GB va diskdan o'qish MB/s.
2. (s2) 5 platforma (bulut satri: soatbay narx, 0 $ apparat): zaxira 8 soat oynaga nisbatan (mavsumda ham), bo'lak energiyasi; 6 mezonli matritsa (zaxira, narx, quvvat, mavsumiy moslashuvchanlik, ekotizim, ma'lumot ko'chirish vaqti); TANLOV.
3. (s3) 8 soatlik 'byudjet' bo'laklar uchun o'tkazuvchanlik simulyatsiyasi: tasvirni o'qish+tile (median 40 ms), inferens batch 8/32/64, hisobot (vektorlash, o'lcham), yozish; 384 000 bo'lakni navbat modeli bilan simulyatsiya qilib jami vaqt p50/p95 (10 marta takror) va 8 soatga sig'ish ehtimoli; batch kattalashganda jami vaqt qanday kamayadi.
4. (s4) Roofline: 4090, L4, T4, Xeon, A100 tizza nuqtalari; U-Net-R18 va DeepLabV3 FP16/INT8 batch 8/64 AI; katta batch'da hamma H rejimda bo'lishini ko'rsating.
5. (s5) torch'da MiniSeg (4 conv enkoder + 2 ConvTranspose2d + 1 sinf) 256x192 va 512x384; hook va FlopCounterMode; batch 1/8/32 CPU vaqti va GFLOP/s; batch bilan bir bo'lak vaqti qanchaga kamayadi.
6. (s6) Co-design: 3 model x 2 format x 3 batch x 5 platforma x 3 mavsum yuklamasi; cheklovlar zaxira >= 1,2, 8 soat oyna, narx, quvvat; Pareto narx-IoU; mavsumda bulutga 'to'lib ketish' (burst) gibrid variantini qo'shing.
7. (s7) TCO 3 yil: o'z 4090/L4 (yil bo'yi) vs bulut L4 soatbay (faqat ishlagan soatlar, mavsumiylik bilan) vs gibrid (T4 o'zimizniki + mavsumda bulut); kunlik tasvir 10 000-60 000 oralig'ida tenglik nuqtasi.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar) — 25, Hisob-kitob va simulyatsiya to'g'riligi — 25, Grafiklar (Roofline, byudjet, co-design, TCO) — 15, Tahlil, tanlov asosi va xavflar — 25, Hisobot sifati — 10 (jami 100)

## 18-variant. Yer osti ko'mir konida metan sensorlari va konveyer kameralarini birgalikda tahlil qilish (qiyin)

Soha: Ko'mir / kon sanoati

Yer osti ko'mir konida 3 ta lava (qazish uchastkasi) bor; har lavada 400 ta metan/CO/harorat sensori (10 Hz, 6 kanal) va 6 ta konveyer tasmasi kamerasi (720p, 10 kadr/s) o'rnatilgan. Sensor oynasi (5 s) 1D-CNN bilan portlash xavfi darajasiga tasniflanadi, kameralar tasma yirtilishi va begona jism (metall) ni aniqlaydi. Metan ogohlantirishi 500 ms ichida, konveyer to'xtatish buyrug'i 200 ms ichida berilishi kerak. Yer ostidagi qurilmalar portlashdan himoyalangan (Ex) korpusda, passiv sovutish, 15 Vt chegara, -10...+40 C; yer ustiga 100 Mbit/s optika (kechikish 5 ms), lekin optika uzilganda tizim yer ostida mustaqil ishlashi shart. Byudjet 3 lava uchun 25 000 $ (Ex korpus har qurilmaga +800 $). To'xtash 1 soati 15 000 $ zarar.

| Parametr | Qiymat |
|---|---|
| Yuklama | sensor: 3 x 400 x 10 Hz oyna = 12 000 inferens/s (oyna 50x6); kamera: 3 x 6 x 10 = 180 kadr/s |
| Modellar | 1D-CNN 50x6 (2 MFLOP, F1 0,93), TCN (15 MFLOP, 0,96); yolov8n @ 640 (8,7 GFLOP, mAP 0,80), yolov8s (28,6, 0,86) |
| SLO | metan <= 500 ms p99, konveyer to'xtatish <= 200 ms p99 |
| Platformalar | Orin Nano (Ex), Orin NX (Ex), Hailo-8 + ARM (Ex), RK3588 (Ex), yer usti: T4 server, L4 server |
| Cheklovlar | <= 15 Vt passiv (cho'qqi 70 %), Ex korpus +800 $, <= 25 000 $, oflayn ishlash shart |
| Quvvat rejimi | Orin NX 10/15/25 Vt -> 40/70/100 % cho'qqi; Orin Nano 7/15 Vt -> 50/100 % |
| TCO | 5 yil, elektr 0,06 $/kVt*soat, Ex sertifikat 5 yilda yangilanadi (+500 $), to'xtash 15 000 $/soat |

Qadamlar:

1. (s1) Ikki yuklama turi uchun alohida hisob: sensor oynalari (lava 4 000 inferens/s) va kameralar (lava 60 kadr/s): GFLOP, TFLOP/s, kerakli cho'qqi (sensor samaradorlik 0,25, kamera 0,45); jami lava talabi; sensor oqimi (6 x 4 bayt x 10 Hz x 400) kbit/s va 6 x 2 Mbit/s kamera oqimi 100 Mbit/s optika uchun.
2. (s2) 6 platforma: Ex quvvat rejimida (15 Vt, passiv -> cho'qqi 70 %) haqiqiy TOPS, zaxira lava talabiga, inferens_J; 8 mezonli matritsa (zaxira, kechikish, Ex/harorat, quvvat, narx + Ex korpus, oflayn ishlash, dekod 6 oqim, ekotizim); yer osti TANLOVI va yer usti server ('ikkinchi fikr' va arxiv) roli.
3. (s3) Ikki byudjet: 500 ms (sensor: konsentrator RS-485 median 60 ms, oyna, inferens, ogohlantirish sirena) va 200 ms (kamera: dekod, inferens, kuzatuv, PLC to'xtatish 20 ms); 20 000 hodisa Monte-Karlo har biri uchun p99; sensor partiyasi 1/400/4000 (bitta lava bir yo'la) ta'siri; optika uzilgan rejimda (yer osti mustaqil) byudjet bajariladimi.
4. (s4) Roofline: Orin Nano/NX (15 Vt rejim cho'qqisi), RK3588, Hailo-8, T4, L4; 1D-CNN batch 1/400/4000 va yolov8n INT8 batch 1/6 AI va rejim; sensor modeli nima uchun faqat katta partiyada H rejimga chiqadi.
5. (s5) torch'da 1D-CNN (Conv1d x3 + Linear, 50x6) va MiniYolo 2 sinf 320x180; hook va FlopCounterMode; batch 1/400 (sensor) va 1/6 (kamera) CPU vaqti; ikkala yuklamani bitta jarayonda ketma-ket bajarib lava uchun ms/sikl.
6. (s6) Co-design grid: (2 sensor modeli x 3 partiya) x (2 detektor x imgsz 416/640 x 2 format x batch 1/6) x 4 Ex platforma x 2 quvvat rejimi; cheklovlar zaxira >= 2 (xavfsizlik), ikkala kechikish, 15 Vt, narx (3 lava + Ex), oflayn; Pareto narx-(F1+mAP)/2; har cheklovni olib tashlab yaroqli soni.
7. (s7) TCO 5 yil: har lavada Ex edge (Orin NX / RK3588) + yer usti arxiv server vs 'hammasi yer ustida' (T4/L4 + optika, oflayn talab buziladi - jarima sifatida yiliga 2 soat to'xtash 15 000 $/soat); lava soni 1-8 tenglik nuqtasi; Ex sertifikat va to'xtash zararining ulushi.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z); Strukturaviy sxema: sensor konsentrator (RS-485/Modbus) + kamera (RTSP) -> Ex edge -> sirena/PLC va yer usti server (optika) interfeyslari, oflayn rejim tavsifi; Quvvat rejimi tahlili: 10/15/25 Vt rejimlarida zaxira va yaroqli konfiguratsiyalar soni jadvali

Baholash: Kod: parametrlangan, qayta ishlatiladigan funksiyalar — 25, Hisob, simulyatsiya va sezgirlik tahlili to'g'riligi — 20, Grafiklar va Pareto/tenglik nuqtasi tahlili — 20, Muhandislik xulosasi, strukturaviy sxema, xavflar — 25, Hisobot sifati — 10 (jami 100)

## 19-variant. Bankomat tarmog'ida yuz orqali tasdiqlash va shubhali xatti-harakatni aniqlash (oson)

Soha: Bank

Tijorat banki 500 ta bankomatga kamera orqali 'yuz bilan tasdiqlash' va 'niqob/shlem/ikkinchi odam' kabi shubhali holatlarni aniqlashni joriy qilmoqda. Bankomat o'rtacha har 90 s da bir mijozga xizmat ko'rsatadi; tasdiqlash uchun 5 kadr, shubhali holat uchun sessiya davomida 1 kadr/s tahlil qilinadi. Tasdiqlash 800 ms ichida bo'lishi kerak, aks holda mijoz PIN rejimiga o'tadi. Bankomat ichida 10 Vt bo'sh quvvat va sanoat PC (2 yadro) bor; markaziy ma'lumot markazi 2 ta zonada; bankomat -> markaz kanali 2 Mbit/s (median 40 ms). Biometrik ma'lumot bankdan chiqmasligi shart; byudjet bankomat boshiga 300 $ yoki markazda 30 000 $.

| Parametr | Qiymat |
|---|---|
| Yuklama | 500 bankomat x (5 kadr/90 s + 1 kadr/s x 30 s sessiya/90 s) ~ 0,39 inferens/s har biri = 195 inferens/s (pik 2x) |
| Modellar | SCRFD-0.5G (0,5 GFLOP) + MobileFaceNet (0,45, TAR 0,97) + yolov8n shubha @ 320 (2,2); og'ir: SCRFD-2.5G + ArcFace-R50 (12,6, TAR 0,995) |
| SLO | tasdiqlash p95 <= 800 ms, inferens <= 150 ms |
| Platformalar | mavjud bankomat PC (2 yadro, ~0,05 TFLOP/s, 0 $), Hailo-8L M.2 (bankomatga), Orin Nano, T4 server (markaz), L4 server (markaz) |
| Cheklovlar | <= 10 Vt bankomatda, <= 300 $/bankomat yoki <= 30 000 $ markaz, 2 zona |
| Tarmoq | 2 Mbit/s, median 40 ms, sigma 0,5; kadr JPEG 150 KB |
| TCO | 5 yil, 500 -> 1 200 bankomat, elektr 0,1 $/kVt*soat |

Qadamlar:

1. (s1) Bankomat (0,39 inferens/s) va markaz (195/s, pik 390/s) uchun 2 zanjir GFLOP, TFLOP/s, kerakli cho'qqi (samaradorlik 0,3); 5 kadr x 150 KB ni 2 Mbit/s kanaldan yuborish vaqti (ms) - markaz varianti uchun byudjetga ta'sirini hisoblang.
2. (s2) 5 platforma: zaxira bankomat va markaz uchun (mavjud PC yengil zanjirni ko'taradimi?), inferens_J; 6 mezonli matritsa (zaxira, kechikish, narx, quvvat, biometrik maxfiylik, ekotizim); TANLOV ikki arxitektura uchun.
3. (s3) 800 ms byudjet: kamera (5 kadr yig'ish 200 ms), yuz aniqlash + embedding, markaz uchun +uzatish (5 x 150 KB / 2 Mbit/s) va kanal kechikishi, bazadan solishtirish (median 30 ms), javob; 20 000 sessiya Monte-Karlo p95; qaysi arxitektura SLO'ni bajaradi; batch 1/8/32 markazda.
4. (s4) Roofline: bankomat PC (20 GB/s), Hailo-8L, Orin Nano, T4, L4; MobileFaceNet va ArcFace-R50 FP16/INT8 batch 1/32 AI va rejim.
5. (s5) torch'da MiniFace (4 conv + GAP + Linear 128) 112x112 va MiniYolo 320x320 (3 sinf shubha); hook va FlopCounterMode; torch.set_num_threads(2) bilan batch 1/5 CPU vaqti - mavjud bankomat PC'ga o'xshash muhitda zanjir ms.
6. (s6) Co-design: 2 zanjir x FP16/INT8 x batch 1/5/32 x 5 platforma; cheklovlar zaxira >= 1,5 (pik), inferens <= 150 ms, narx (500 bankomat yoki markaz 2 zona), quvvat; Pareto narx-TAR.
7. (s7) TCO 5 yil: har bankomatda Hailo-8L (300 $) vs mavjud PC (0 $, yengil zanjir) vs markaz T4/L4 2 zona + kanal; 500 va 1 200 bankomat; tenglik nuqtasi; xulosada biometrik maxfiylik va kanal uzilishini hisobga oling.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar, hisob-kitob va jadvallar — 25, Tahlil va asoslangan tanlov — 25, Hisobot sifati (grafik, sxema, til) — 20 (jami 100)

## 20-variant. Davlat xizmatlari markazida hujjatlarni raqamlashtirish: skan -> tasnif -> OCR -> maydon ajratish (qiyin)

Soha: Davlat xizmatlari

Respublika davlat xizmatlari agentligi 200 ta markazda arizalar, pasport nusxalari va ma'lumotnomalarni skaner qiladi: ish vaqtida jami 250 sahifa/min (pik 600/min), A4 300 dpi (2480x3508). Har sahifa uchun 4 bosqichli zanjir: hujjat turi tasnifi (12 sinf), matn bloklarini aniqlash, OCR (o'rtacha 40 qator), kalit maydonlarni ajratish (kichik Transformer). Operator skan qilgach 2 s ichida to'ldirilgan shaklni ko'rishi kerak (p95); aks holda navbat o'sadi. Shaxsiy ma'lumotlar - faqat davlat ma'lumot markazi (2 zona, 10 kVt) yoki markazning o'zida ishlov. Byudjet 120 000 $; har markazda 300 Vt va 1 server o'rni bor; markaz -> DPM kanali 20 Mbit/s (median 25 ms). Model zanjiri bosqichlari turli xil AI'ga ega, shuning uchun bir platforma hammasiga mos kelmasligi mumkin.

| Parametr | Qiymat |
|---|---|
| Yuklama | 250 sahifa/min = 4,2 sahifa/s (pik 10/s); har sahifa: 1 tasnif + 1 detektor + 40 OCR qator + 1 maydon modeli |
| Modellar (GFLOP) | tasnif EfficientNet-B0 @224: 0,78; detektor DBNet-R18 @1024: 48; OCR SVTR-tiny 32x320: 0,7/qator (og'ir SVTR-small 3,5); maydon LayoutLM-lite 512 token: 22 |
| SLO | p95 <= 2 000 ms sahifa, inferens zanjiri <= 800 ms |
| Platformalar | markazda: Orin NX, RTX A2000 PC, Xeon 8 AMX; DPM: 2 x L4 server, 2 x A10 server, 4 x T4 server |
| Cheklovlar | <= 120 000 $, markazda <= 300 Vt, DPM 2 zona, ma'lumot davlat tarmog'ida |
| Tarmoq | 20 Mbit/s, skan JPEG 1,5 MB (markazdan DPM ga) |
| TCO | 5 yil, 200 -> 400 markaz, elektr 0,08 $/kVt*soat, kanal 300 $/yil/markaz |

Qadamlar:

1. (s1) Bir sahifa zanjiri jami GFLOP (40 OCR qator bilan) ikki konfiguratsiyada (tiny/small OCR); 4,2 va 10 sahifa/s uchun TFLOP/s, kerakli cho'qqi (bosqichlarga alohida samaradorlik: CNN 0,45, OCR kichik batch 0,2, Transformer 0,3); markaz (1,25 sahifa/min) uchun ham; 1,5 MB skan x sahifa/s ni 20 Mbit/s kanalga nisbatan tekshiring.
2. (s2) 6 platforma/konfiguratsiya: zaxira markaz va DPM yuklamasiga, sahifa energiyasi; 8 mezonli matritsa (zaxira, kechikish, narx, quvvat, maxfiylik/joylashuv, 2 zona ishonchlilik, ekotizim, boshqaruv 200 qurilma vs 1 klaster); 'markazda' va 'DPM' TANLOVI hamda gibrid (markazda tasnif+detektor, DPM'da OCR+maydon) varianti.
3. (s3) 2 000 ms byudjet: skaner (median 600 ms, sigma 0,2), uzatish (DPM uchun 1,5 MB / 20 Mbit/s + 25 ms), tasnif, detektor, OCR (40 qator: batch 1 ketma-ket vs batch 40), maydon modeli, shakl render (100 ms); 20 000 sahifa Monte-Karlo p95/p99 uch arxitekturada; OCR batch'lash kechikishni necha baravar kamaytiradi.
4. (s4) Roofline: Orin NX, A2000, Xeon, L4, A10, T4; 4 bosqich uchun alohida AI (DBNet yuqori, OCR qator past, LayoutLM o'rtacha) batch 1/40 va rejim; qaysi bosqich qaysi platformada xotira bilan cheklangan - gibrid taqsimot asosi.
5. (s5) torch'da 3 kichik model: MiniCls (4 conv + GAP + Linear 12) 224x224, MiniOCR (3 conv + GRU 64 + Linear 100) 32x160, MiniLayout (2 qatlam TransformerEncoderLayer d=128, 128 token); hook va FlopCounterMode (attention va GRU qo'lda); batch 1/8/40 CPU vaqti; zanjir ms/sahifa.
6. (s6) Co-design grid: 2 OCR x OCR batch 1/8/40 x FP16/INT8 x 3 joylashuv (markaz/DPM/gibrid) x 6 platforma; cheklovlar zaxira >= 1,5 (pik), zanjir inferens <= 800 ms, narx (200 markaz yoki DPM), quvvat, maxfiylik; Pareto narx-aniqlik (tiny 0,96, small 0,985); har cheklovni olib tashlab yaroqli soni.
7. (s7) TCO 5 yil: har markazda Orin NX/A2000 vs DPM 2 zona L4/A10 + 200 kanal vs gibrid; 200 va 400 markaz; markaz soni 50-500 tenglik nuqtasi; sezgirlik: kanal 100-600 $/yil, elektr 0,05-0,2 $.

Topshiriladi: Kod (.py yoki .ipynb), har qadam alohida funksiya/bo'lim; Natijalar jadvallari (CSV yoki markdown); Grafiklar (PNG): kechikish byudjeti, Roofline, co-design, TCO; Hisobot (PDF): talablar matritsasi, tanlov asosi, strukturaviy sxema matni, xulosa (>= 150 so'z); Strukturaviy sxema: skaner -> markaz ishlov -> DPM zanjiri -> shakl xizmati interfeyslari (TWAIN, HTTPS/gRPC, navbat) va o'tkazuvchanlik jadvali; 3 ta xavf (kanal, OCR sifati, 2 zona sinxron) va kamaytirish choralari

Baholash: Kod: parametrlangan, qayta ishlatiladigan funksiyalar — 25, Hisob, simulyatsiya va sezgirlik tahlili to'g'riligi — 20, Grafiklar va Pareto/tenglik nuqtasi tahlili — 20, Muhandislik xulosasi, strukturaviy sxema, xavflar — 25, Hisobot sifati — 10 (jami 100)
