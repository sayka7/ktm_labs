# A1. Sun'iy intellektga asoslangan kompyuter tizimlari arxitekturasini tahlil qilish va loyihalash

Bu virtual laboratoriyada siz SHV (shaxsiy himoya vositalari) aniqlash tizimi uchun hisoblash platformasini raqamlar bilan tanlaysiz: talablardan FLOP/TFLOP/s yuklamasini hisoblaysiz, platformalarni talablar matritsasi bo'yicha baholaysiz, kechikish byudjetini simulyatsiya qilasiz, Roofline tahlilini quriysiz, kichik CNN'ning FLOP'ini torch bilan sanaysiz va Hardware–Software Co-design izlanish fazosini grid search bilan ko'rib chiqasiz. Real labdan farqi: tarmoq yo'qligi sababli ultralytics YOLO vaznlari yuklanmaydi — rasmiy hujjatlardagi GFLOP/parametr qiymatlari jadval sifatida kiritilgan; GPU yo'q, shuning uchun o'lchovlar 2 yadroli CPU'da kichik modelda bajariladi.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/A1

## Bosqichlar

### 1-qadam. Talablardan yuklama hisobi: GFLOP, bayt, TFLOP/s

Ssenariy talablari (16 kamera × 10 kadr/s) YOLOv8 variantlari uchun bir inferens GFLOP, vazn hajmi (FP16/INT8) va jami TFLOP/s talabiga aylantiriladi; samaradorlik koeffitsiyenti orqali kerakli cho'qqi TOPS topiladi. Kutilgan natija: yolov8s @ 640 uchun 28,6 GFLOP, 4,58 TFLOP/s haqiqiy, ~10 TOPS cho'qqi. Kodda KAMERALAR, FPS, SAMARADORLIK va MODELLAR jadvalini o'zgartirib, talab qanday o'sishini kuzating.

Fayl: `s1_1_qadam_talablardan_yuklama_hisobi_gflop.py` — ishga tushirish: `python3 s1_1_qadam_talablardan_yuklama_hisobi_gflop.py`

Kutilgan natija: 12 qatorli jadval (4 model × 3 imgsz), etalon yolov8s@640: 4,58 TFLOP/s, cho'qqi 10,2 TOPS; tarmoq 64 Mbit/s, xom kadrlar ~1 GB/s.

Tajriba qilib ko'ring:

- KAMERALAR=64 va FPS=15 qilib, qaysi modellar 50 TOPS (Orin NX) chegarasiga sig'ishini aniqlang.
- SAMARADORLIK ni 0.3 va 0.6 qilib, kerakli cho'qqi TOPS qanday o'zgarishini jadvalga yozing.
- Yangi model qo'shing (masalan, yolov8x: 68,2 M parametr, 257,8 GFLOP) va uning vazn hajmini INT4 (0,5 bayt) uchun hisoblang.

### 2-qadam. Platformalar jadvali va talablar matritsasi bo'yicha tanlash skripti

Olti platforma (Orin NX, AGX Orin, A100, Coral Edge TPU, x86 CPU, Hailo-8) uchun haqiqiy unumdorlik, zaxira koeffitsiyenti, dekodlash yetarliligi va bir inferensga energiya hisoblanadi; so'ng 7 mezonli og'irlikli talablar matritsasi bo'yicha ball beriladi. Kutilgan natija: Orin NX ~96 %, Coral va CPU zaxira < 1 bo'lgani uchun yaroqsiz. Og'irliklarni va baholash funksiyalarini o'zgartirib, tanlov qanchalik barqarorligini tekshiring.

Fayl: `s2_2_qadam_platformalar_jadvali_va_talablar.py` — ishga tushirish: `python3 s2_2_qadam_platformalar_jadvali_va_talablar.py`

Kutilgan natija: Ikki jadval: texnik ko'rsatkichlar (zaxira_koef: Orin NX ~4,9, A100 ~68, Coral 0,44) va og'irlikli ballar; TANLOV: Jetson Orin NX.

Tajriba qilib ko'ring:

- Narx og'irligini 3 dan 5 ga, haroratnikini 4 dan 1 ga o'zgartiring — tanlov o'zgaradimi?
- Talabni yolov8m @ 640 (12,6 TFLOP/s) ga oshiring va qaysi platformalar yaroqli qolishini ko'ring.
- Hailo-8 uchun apparat dekoder yo'qligini hisobga olib, CPU dekodlash quvvatini (+20 Vt) qo'shing va inferens_J ni qayta hisoblang.

### 3-qadam. Kechikish byudjeti va Monte-Karlo simulyatsiyasi

200 ms umumiy byudjet bloklar (tarmoq+dekodlash, oldindan ishlov, inferens, keyingi ishlov) orasida taqsimlanadi; har blok kechikishi lognormal taqsimot bilan 20 000 marta simulyatsiya qilinib p50/p95/p99 topiladi. Paket (batch) hajmi inferensni tejasa-da, kutish vaqtini oshirishi ko'rsatiladi. Kutilgan natija: batch=1 da p95 ~105 ms, batch=32 da p95 200 ms dan oshadi. Grafikda byudjet vs simulyatsiya ustunlari va batch egri chizig'i chiqadi.

Fayl: `s3_3_qadam_kechikish_byudjeti_va_monte_karl.py` — ishga tushirish: `python3 s3_3_qadam_kechikish_byudjeti_va_monte_karl.py`

Kutilgan natija: Bloklar jadvali (p50/p95/p99), jami p95 ~105 ms, batch jadvali: 16 da ~150 ms, 32 da ~240 ms; 2 panelli grafik.

Tajriba qilib ko'ring:

- Tarmoq blokining sigma qiymatini 0.35 dan 0.8 ga oshiring (Wi-Fi kamera) — p99 qanday o'zgaradi?
- Byudjetni 50 ms ga tushiring va qaysi bloklar qayta loyihalanishi kerakligini aniqlang.
- Inferens medianini Orin NX o'rniga AGX Orin (3 baravar tez) uchun o'zgartirib, ruxsat etilgan maksimal batch'ni toping.

### 4-qadam. Arifmetik intensivlik va Roofline modeli

Har platforma uchun tizza nuqtasi (P_peak / B_mem) hisoblanadi; YOLOv8s (FP16/INT8, batch 1/8), GEMM va GELU yuklamalari uchun AI = FLOP/bayt topilib, erishiladigan unumdorlik P = min(P_peak, AI·B_mem) va rejim (hisoblash yoki xotira bilan cheklangan) aniqlanadi. Kutilgan natija: YOLOv8s FP16 batch 1 Orin NX'da xotira bilan (X), A100'da hisoblash bilan (H) cheklangan; log-log Roofline grafigi chiziladi.

Fayl: `s4_4_qadam_arifmetik_intensivlik_va_rooflin.py` — ishga tushirish: `python3 s4_4_qadam_arifmetik_intensivlik_va_rooflin.py`

Kutilgan natija: Tizza nuqtalari (A100 INT8 ~322, Orin NX ~490 FLOP/bayt), yuklama×platforma jadvali H/X belgilari bilan, Roofline grafigi.

Tajriba qilib ko'ring:

- AKTIV_BAYT_FP16 ni 2 baravar oshiring (katta kirish) va Orin NX'da qaysi batch'dan boshlab rejim H ga o'tishini toping.
- Yangi platforma qo'shing (masalan, H100 SXM: 1979 TOPS INT8, 3350 GB/s) va tizza nuqtasini solishtiring.
- GEMM uchun n=256 va n=4096 nuqtalarini qo'shib, kichik matritsalar nima uchun xotira bilan cheklanishini tushuntiring.

### 5-qadam. Kichik CNN uchun FLOP va parametrlarni torch bilan hisoblash

MiniYolo nomli kichik detektor tarmog'i uchun forward hook'lar bilan har qatlam FLOP'i formuladan (2·Kh·Kw·Cin·Cout·Hout·Wout) hisoblanadi va torch FlopCounterMode bilan 0 % farq tasdiqlanadi. So'ng kirish o'lchami (64/128/256) va batch (1/8) uchun haqiqiy CPU vaqti o'lchanib erishilgan GFLOP/s topiladi. Kutilgan natija: 128×128 da ~41,6 MFLOP, 61 K parametr; FLOP imgsz² ga proporsional; batch paketlash bir inferens vaqtini kamaytiradi.

Fayl: `s5_5_qadam_kichik_cnn_uchun_flop_va_paramet.py` — ishga tushirish: `python3 s5_5_qadam_kichik_cnn_uchun_flop_va_paramet.py`

Kutilgan natija: Qatlamlar jadvali (5 ta Conv2d), jami 41,58 MFLOP hook va FlopCounterMode bir xil; o'lchov jadvali 12–40 GFLOP/s oralig'ida.

Tajriba qilib ko'ring:

- Tarmoqqa nn.Linear bosh (head) qo'shib, linear_hook ishlashini va FLOP ulushini tekshiring.
- Birinchi conv'ning stride'ini 2 dan 1 ga o'zgartiring — FLOP necha baravar oshadi, vaqt-chi?
- torch.set_num_threads(1) bilan qayta o'lchab, 2 ipning tezlanishini batch 1 va 8 uchun solishtiring.

### 6-qadam. Hardware–Software Co-design: izlanish fazosini grid search bilan ko'rib chiqish

Dasturiy qarorlar (3 model × 3 imgsz × 2 format × 3 batch) va apparat qarorlari (6 platforma) birgalikda — 270 ta konfiguratsiya — zaxira, kechikish, quvvat, narx, harorat va format cheklovlari bo'yicha baholanadi. Yaroqli konfiguratsiyalar orasidan mAP bo'yicha eng yaxshilari, har platforma uchun eng yaxshi variant va narx–sifat Pareto chegarasi topiladi; grafikda yaroqli (rangli) va yaroqsiz (kulrang) nuqtalar ko'rsatiladi. Cheklov qiymatlarini o'zgartirib, izlanish fazosi qanday torayishini kuzating.

Fayl: `s6_6_qadam_hardware_software_co_design_izla.py` — ishga tushirish: `python3 s6_6_qadam_hardware_software_co_design_izla.py`

Kutilgan natija: 270 konfiguratsiya, ~118 yaroqli; top-10 jadval (yolov8m AGX Orin / Orin NX INT8), Pareto ro'yxati (Hailo-8 → Orin NX → AGX Orin), scatter grafik.

Tajriba qilib ko'ring:

- INFERENS_BYUDJET_MS ni 60 dan 20 ga tushiring — batch 8 variantlari yaroqli qoladimi?
- MAKS_NARX=5000 va harorat cheklovini olib tashlab (sanoat korpusi), A100 yaroqli bo'ladimi, Pareto o'zgaradimi?
- mAP proxy formulasini o'zgartiring (masalan, INT8 uchun −3 punkt) va eng yaxshi konfiguratsiya qanday siljishini yozing.

### 7-qadam. Edge va markazlashgan arxitekturaning 3 yillik TCO taqqoslovi

Uch variant (har sexda Orin NX; markaziy A100 server; markaziy L4 server) uchun apparat, elektr energiyasi (PUE bilan), tarmoq va texnik xizmat xarajatlari 3 yilga hisoblanadi; sexlar soni 1–24 oralig'ida tenglik nuqtasi qidiriladi va grafik chiziladi. Kutilgan natija: 16 kamera uchun Edge ~1,5 ming $, A100 ~26 ming $; 24 sexgacha markaz Edge'dan arzonlashmaydi. Elektr narxi, tarmoq ijara va server sig'imini o'zgartirib qaytadan hisoblang.

Fayl: `s7_7_qadam_edge_va_markazlashgan_arxitektur.py` — ishga tushirish: `python3 s7_7_qadam_edge_va_markazlashgan_arxitektur.py`

Kutilgan natija: 1/2/4/8 sex uchun TCO jadvallari ($/kamera/oy bilan), tenglik nuqtasi xabari, 3 chiziqli TCO grafigi.

Tajriba qilib ko'ring:

- NARX_KVT_SOAT ni 0.25 $ (Yevropa) qilib energiya ulushini solishtiring.
- Markaz uchun tarmoq xarajatini 0 qilib (mavjud optika), tenglik nuqtasi paydo bo'ladimi?
- Edge variantiga 3 yilda bir marta modul almashtirish (buzilish 10 %) xarajatini qo'shing.

## Qo'shimcha topshiriqlar (10 ta, jami 100 ball)

| № | Topshiriq | Daraja | Ball |
|---|---|---|---|
| 1 | **Yuklama hisobini kengaytirish** — 1-qadam skriptiga yolov8x (68,2 M parametr, 257,8 GFLOP @640) va yolo11s (9,4 M, 21,5 GFLOP) qatorlarini qo'shing, KAMERALAR=32 uchun jadvalni qayta hosil qiling va qaysi modellar 50 TOPS dense cho'qqiga (samaradorlik 45 %) sig'ishini ro'yxat qilib hisobot bering. | oson | 5 |
| 2 | **Og'irliklar sezgirligi** — 2-qadamda talablar matritsasining og'irliklarini uch xil ssenariyda (narx ustuvor, kechikish ustuvor, energiya ustuvor) o'zgartiring; har ssenariy uchun birinchi uchta platformani jadvalga yozing va tanlov qaysi og'irliklarda o'zgarishini aniqlang. | oson | 5 |
| 3 | **Kechikish byudjetini qayta taqsimlash** — 3-qadamda umumiy byudjetni 100 ms ga tushiring, bloklar byudjetini o'zingiz qayta taqsimlang va simulyatsiyada p95 ≤ 100 ms bo'lishi uchun qaysi blok medianini necha foizga kamaytirish kerakligini raqam bilan ko'rsating (grafik bilan). | oson | 5 |
| 4 | **Roofline'ga yangi platformalar** — 4-qadam Roofline tahliliga kamida ikkita yangi platforma (masalan, H100 SXM, Jetson Orin Nano, Apple M-seriya NPU) rasmiy tavsiflari bilan qo'shing, tizza nuqtalarini hisoblang va YOLOv8s INT8 batch 1 har platformada qaysi rejimda ishlashini jadval + grafik bilan asoslang; manbalarni ko'rsating. | o'rta | 10 |
| 5 | **Modelni o'lchash: nazariya va amaliyot** — 5-qadamdagi MiniYolo'ni kengaytiring (kamida 8 conv qatlam, 256×256 kirish), hook va FlopCounterMode natijalarini solishtiring, so'ng torch.set_num_threads(1) va (2) bilan batch 1/4/16 uchun ms va GFLOP/s jadvalini tuzing; erishilgan GFLOP/s nazariy CPU cho'qqisining necha foizi ekanini baholang. | o'rta | 10 |
| 6 | **Co-design cheklovlar tahlili** — 6-qadam grid search'ida har bir cheklovni navbat bilan olib tashlab (zaxira, kechikish, quvvat, narx, harorat) yaroqli konfiguratsiyalar soni qanday o'zgarishini jadvalga yozing; qaysi cheklov eng «qimmat» ekanini va uni yumshatish uchun qanday muhandislik yechimi borligini tushuntiring. | o'rta | 10 |
| 7 | **TCO sezgirlik tahlili** — 7-qadamda elektr narxi (0,05–0,30 $/kVt·soat), tarmoq ijarasi (0–3000 $/yil) va server narxi bo'yicha sezgirlik tahlili o'tkazing: har parametr uchun tenglik nuqtasi (sexlar soni) qanday siljishini grafikda ko'rsating va 5 yillik muddat uchun xulosani qayta tekshiring. | o'rta | 10 |
| 8 | **Ikkinchi ssenariy: 64 kamera, markazlashgan ishlov** — 1–4 va 6-qadamlar skriptlarini 64 kamera × 15 kadr/s, kechikish ≤ 500 ms, markaziy server xonasi ssenariysi uchun qayta sozlang: yuklama, platforma tanlovi (A100/L4/AGX Orin klaster), kechikish byudjeti (tarmoq 100 ms), Roofline va Co-design natijalarini bitta hisobotda keltiring va Edge ssenariysi bilan solishtiring. | qiyin | 15 |
| 9 | **Energiya byudjeti va termal cheklov modeli** — 2- va 6-qadamlarga quvvat rejimi modelini qo'shing: Orin NX 10/15/25 Vt rejimlarida cho'qqi unumdorlik mos ravishda 40/70/100 % deb olinsin, +40 °C muhitda passiv sovutishda 15 Vt rejimiga tushadi. Har rejim uchun zaxira koeffitsiyenti, yaroqli konfiguratsiyalar soni va bir inferens energiyasini (J) hisoblab, faol sovutish qarorini raqamlar bilan asoslang. | qiyin | 15 |
| 10 | **Strukturaviy sxema va interfeyslar generatori** — Tanlangan konfiguratsiya (6-qadam natijasi) asosida strukturaviy sxemani avtomatik hosil qiluvchi skript yozing: bloklar ro'yxati, har interfeys uchun protokol, o'tkazuvchanlik (Mbit/s yoki MB/s — 1-qadam formulalaridan) va kechikish byudjeti (3-qadam natijasidan) jadval ko'rinishida chiqsin; 3 ta xavf va ularning kamaytirish choralarini ham skript matn sifatida chop etsin. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.

## Mustaqil bajarish uchun variantlar

20 ta variant — `VARIANTS.md` faylida (jurnal raqami bo'yicha tanlanadi).
