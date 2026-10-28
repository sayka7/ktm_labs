# A5. Taqsimlangan sun'iy intellekt tizimlarini ishlab chiqish

Bu virtual laboratoriyada talaba taqsimlangan o'qitish va parallel inferensning asosiy mexanizmlarini CPU sandbox'da bajaradi: torch.distributed (gloo backend) bilan 2 jarayonli DDP, kuchli/kuchsiz masshtablash o'lchovi, halqa all-reduce algoritmini qo'lda yozish, gradient to'plash ekvivalentligi, multiprocessing Pool bilan parallel inferens, Amdal/Gustafson qonunlari va GPU klasteri uchun aloqa/hisob kalkulyatori. Real labdan farqi: GPU va NCCL yo'q — 2 ta CPU yadro «2 ta GPU» rolini o'ynaydi, modellar kichik MLP, ma'lumot sintetik yoki sklearn digits; mantiq (process group, DistributedSampler, DDP o'rami, all-reduce) esa haqiqiy klasterdagi bilan bir xil.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/A5

## Bosqichlar

### 1-qadam. DDP: gloo backend bilan 2 jarayonli taqsimlangan o'qitish

torch.multiprocessing orqali 2 ta jarayon (rank 0 va 1) ishga tushiriladi, har biri jarayonlar guruhiga qo'shiladi, DistributedSampler ma'lumotni kesishmaydigan ikki qismga bo'ladi, DDP o'rami gradientlarni har qadamda all-reduce qiladi. Oxirida parametrlar barcha ranklarda bir xil ekani (max farq 0) va jami ko'rilgan namunalar soni tekshiriladi. Talaba `EPOCHS`, `BS`, LR masshtablash qoidasini (`0.05 * world`) va `set_epoch` chaqiruvini o'zgartirib ko'rishi mumkin.

Fayl: `s1_1_qadam_ddp_gloo_backend_bilan_2_jarayon.py` — ishga tushirish: `python3 s1_1_qadam_ddp_gloo_backend_bilan_2_jarayon.py`

Kutilgan natija: Ikkala rank o'z 4096 namunasini ko'radi, jami 8192; aniqlik ~0.95; ranklar orasidagi parametr farqi 0.00e+00 (DDP sinxron); o'tkazuvchanlik bir necha o'n ming namuna/s.

Tajriba qilib ko'ring:

- `sampler.set_epoch(ep)` satrini izohga olib, ikki epoxada bir xil aralashtirish bo'lishini tekshiring (yo'qotish qanday o'zgaradi?).
- `DDP(make_model())` o'rniga oddiy `make_model()` qo'ying — parametr farqi endi nolga teng bo'ladimi? Nega?
- `lr=0.05 * world` ni `lr=0.05` qilib, yakuniy yo'qotishni solishtiring (chiziqli LR qoidasi).

### 2-qadam. Masshtablash: 1 vs 2 jarayon, S(N) va E(N)

Bir xil MLP uchun to'rt konfiguratsiya o'lchanadi: 1 jarayon (1 yadro) — baza T(1); 2 jarayonli DDP kuchli rejimda (global paket o'zgarmaydi, har rankka yarmi); kuchsiz rejimda (har rankda to'liq paket, global 2 baravar); va taqqoslash uchun 1 jarayon 2 yadro bilan (intra-op parallellik). Tezlashish S(N) va samaradorlik E(N) jadvalda chiqadi. Talaba `HID`, `GB`, `STEPS` ni o'zgartirib, model kattalashganda samaradorlik qanday o'zgarishini ko'radi.

Fayl: `s2_2_qadam_masshtablash_1_vs_2_jarayon_s_n_.py` — ishga tushirish: `python3 s2_2_qadam_masshtablash_1_vs_2_jarayon_s_n_.py`

Kutilgan natija: Baza ~4-5 s; DDP kuchli rejimda S(2)≈1.6-1.8 (E≈80-90%), kuchsiz rejimda E≈90%; intra-op 2 yadro ham ~1.9x. Qadam 15-25 s davom etadi.

Tajriba qilib ko'ring:

- `HID` ni 1024 dan 256 ga tushiring: kichik modelda aloqa ulushi oshib, E(2) pasayishini kuzating.
- `GB` ni 64 qiling — kuchli masshtablashda har rankka 32 namuna tegadi; samaradorlik nima bo'ladi?
- DDP o'rnida `bucket_cap_mb=1` parametri bilan sinab ko'ring (ko'proq, kichikroq all-reduce).

### 3-qadam. Halqa all-reduce ni NumPy'da qo'lda yozish va aloqa hajmi

N ta ishchining gradient vektorlari uchun halqa all-reduce (N-1 reduce-scatter + N-1 all-gather qadami) simulyatsiya qilinadi, natija oddiy yig'indi bilan solishtiriladi va har ishchi yuborgan baytlar `2(N-1)/N * S` formulasiga mos kelishi ko'rsatiladi. So'ng ResNet-50 gradientlari (102 MB) uchun NVLink, InfiniBand va 10 GbE tarmoqlarda aloqa vaqti hisoblanadi.

Fayl: `s3_3_qadam_halqa_all_reduce_ni_numpyda_qold.py` — ishga tushirish: `python3 s3_3_qadam_halqa_all_reduce_ni_numpyda_qold.py`

Kutilgan natija: N=2,4,8 uchun «natija to'g'ri=True» va yuborilgan baytlar formulaga aynan teng; 10 GbE da 8 GPU uchun ~143 ms, NVLink'da 0.3 ms.

Tajriba qilib ko'ring:

- `ring_allreduce` ni o'rtacha (SUM/N) qaytaradigan qilib o'zgartiring — DDP aynan shuni qiladi.
- Parametr-server sxemasini yozing (hamma 0-ishchiga yuboradi, u qaytaradi) va 0-ishchi yuboradigan hajmni N bo'yicha solishtiring.
- S_bytes ni BF16 (2 bayt) uchun qayta hisoblang.

### 4-qadam. Gradient to'plash va ma'lumotlar parallelligi ekvivalentligi

Bir xil 128 namunali global paket uchun uch usul taqqoslanadi: bitta katta paket; 4 ta mikro-paketda gradient to'plash (yo'qotish 1/K ga ko'paytiriladi); 2 «ishchi»da 64 tadan hisoblab gradientlarni o'rtachalash (DDP all-reduce'ning qo'lda nusxasi). Farqlar 1e-7 dan kichik chiqishi, 1/K ga bo'lmaslik esa gradientni K baravar kattalashtirishi ko'rsatiladi.

Fayl: `s4_4_qadam_gradient_toplash_va_malumotlar_p.py` — ishga tushirish: `python3 s4_4_qadam_gradient_toplash_va_malumotlar_p.py`

Kutilgan natija: max|gA-gB| va max|gA-gC| ~1e-8; xato variantda farq ~0.3 va norma nisbati 4.00.

Tajriba qilib ko'ring:

- Modelga `nn.BatchNorm1d(64)` qo'shing va ekvivalentlik buzilishini ko'ring.
- `lossf = nn.CrossEntropyLoss(reduction='sum')` qilib, to'plashda 1/K kerak bo'lmay qolishini tekshiring.
- K=8, 2 ishchi x 4 mikro-paket kombinatsiyasini yozing (haqiqiy DDP+accumulation).

### 5-qadam. Parallel inferens: sharding, ishchilar puli, natijalarni yig'ish

Digits ustida o'qitilgan MLP og'irliklari NumPy'ga eksport qilinadi, ~360 ming so'rov 2048 talik bo'laklarga bo'linib multiprocessing Pool'ga tarqatiladi (scatter), natijalar tartib raqami bo'yicha yig'iladi (gather). 1 va 2 ishchi uchun o'tkazuvchanlik, S(N), E(N) va bo'lak kechikishining p50/p99 qiymatlari jadvalda chiqadi.

Fayl: `s5_5_qadam_parallel_inferens_sharding_ishch.py` — ishga tushirish: `python3 s5_5_qadam_parallel_inferens_sharding_ishch.py`

Kutilgan natija: 1 ishchi ~0.5-0.6 s, 2 ishchi ~0.3 s; S(2)≈1.7-1.9, E≈85-95%; aniqlik ikkala holatda bir xil (0.993).

Tajriba qilib ko'ring:

- `SHARD` ni 128 va 16384 qilib, IPC xarajati va p99 kechikishga ta'sirini o'lchang.
- Ishchilar sonini 4 qiling (2 yadroda) — samaradorlik nega tushadi?
- `res.sort(...)` satrini olib tashlang va `pool.imap_unordered` bilan aniqlik hisobida nima buzilishini ko'ring.

### 6-qadam. Amdal va Gustafson qonunlari grafigi

Parallel ulush p=0.90/0.95/0.99 uchun Amdal (kuchli masshtablash) va Gustafson (kuchsiz) tezlashish egri chiziqlari N=1..256 oralig'ida hisoblanadi va log-log grafikda chiziladi; har N uchun E(N) jadvali va N→∞ chegarasi chiqadi. Talaba o'z o'lchovlaridan (2-qadam) p ni baholab, grafikka qo'shishi mumkin.

Fayl: `s6_6_qadam_amdal_va_gustafson_qonunlari_gra.py` — ishga tushirish: `python3 s6_6_qadam_amdal_va_gustafson_qonunlari_gra.py`

Kutilgan natija: Jadval: p=0.90 da S(256)≈9.7 (chegara 10x), p=0.99 da 72; Gustafson deyarli chiziqli. Ikki panelli grafik.

Tajriba qilib ko'ring:

- 2-qadamdagi E(2) dan p ni toping: p = 2(1 - 1/S(2)) va Amdal egri chizig'iga nuqta sifatida qo'shing.
- Aloqa vaqtini N bilan o'suvchi qilib (masalan, p ulushi N oshganda kamayadi) egri chiziqning pasayishini modellashtiring.

### 7-qadam. Aloqa/hisob nisbati kalkulyatori (GPU klasteri uchun baho)

Analitik model: T_step = T_comp + (1-overlap)·T_comm, bunda T_comp FLOPs/(cho'qqi·MFU), T_comm — halqa all-reduce hajmi/tarmoq o'tkazuvchanligi. ResNet-18/50, ViT-B, GPT-2 1.5B uchun 4 xil tarmoqda r = T_comm/T_comp va E(N) hisoblanadi, gradient to'plashning ta'siri ko'rsatiladi, ResNet-50 uchun E(N) grafigi chiziladi. Talaba o'z klasteri parametrlarini (PEAK, MFU, BATCH, BYTES_PER_GRAD) kiritadi.

Fayl: `s7_7_qadam_aloqa_hisob_nisbati_kalkulyatori.py` — ishga tushirish: `python3 s7_7_qadam_aloqa_hisob_nisbati_kalkulyatori.py`

Kutilgan natija: NVLink'da E≈99%, 10 GbE da ResNet-50 uchun E≈26% (accum=8 bilan 74%); katta model (ViT-B, GPT-2) yaxshiroq masshtablanadi.

Tajriba qilib ko'ring:

- BYTES_PER_GRAD=4 (FP32) va 1 (FP8/siqish) uchun jadvalni qayta hisoblang.
- OVERLAP=0 va 0.9 — DDP chelak overlap'i qanchalik muhim?
- O'zingizning model (parametr soni, GFLOP) va tarmog'ingizni qo'shib, E(8) ni baholang.

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
|---|---|---|---|
| 1 | **Epoxa va paket ta'siri** — 1-qadam kodida EPOCHS=4 va BS=32, 64, 128 uchun DDP ni ishga tushiring. Har variant uchun o'tkazuvchanlik (namuna/s), oxirgi yo'qotish va aniqlikni jadvalga yozing; paket kattalashganda o'tkazuvchanlik nega oshishini 2-3 jumla bilan tushuntiring. | oson | 5 |
| 2 | **Aloqa hajmi jadvali** — 3-qadam formulasidan foydalanib ResNet-18 (11.7 mln), ViT-B/16 (86 mln) va 7B parametrli model uchun N=2, 8, 64 GPU'da har GPU yuboradigan hajmni (FP32 va BF16) va 100 Gbit/s InfiniBand'dagi vaqtni hisoblang. Natijani 3x3 jadval ko'rinishida chiqaring va 7B model uchun DDP nega yaroqsiz ekanini izohlang. | oson | 5 |
| 3 | **Amdal bo'yicha p ni baholash** — 2-qadamdagi o'lchovingizdan (kuchli rejim) S(2) ni olib, Amdal formulasidan ketma-ket ulush (1-p) ni toping. Shu p bilan S(4), S(8), S(64) ni hisoblang va 6-qadam grafigiga o'lchangan nuqtani qo'shing. Hisobotda p qiymati va maksimal nazariy tezlashish keltiriladi. | oson | 5 |
| 4 | **Kuchli vs kuchsiz masshtablash egri chizig'i** — 2-qadam kodini global paket GB=64, 128, 256, 512 uchun tsiklda ishga tushiring va har GB uchun kuchli hamda kuchsiz rejimda E(2) ni o'lchang. Ikki egri chiziqni bitta matplotlib grafikda chizing (x — GB, y — E(2) %). Qaysi paketdan boshlab kuchli masshtablash «to'yinmay» qolishini aniqlang. | o'rta | 10 |
| 5 | **Parametr-server vs halqa** — 3-qadamdagi NumPy simulyatsiyasiga parametr-server (PS) sxemasini qo'shing: barcha ishchilar gradientni 0-ishchiga yuboradi, u yig'ib qaytaradi. N=2, 4, 8, 16, 32 uchun eng band tugun yuboradigan+qabul qiladigan hajmni ikkala sxemada hisoblang, natijani jadval va grafikda ko'rsating va PS sxemasi qaysi N dan boshlab 10 GbE da 100 ms dan oshishini toping. | o'rta | 10 |
| 6 | **Gradient to'plash bilan DDP** — 4-qadam g'oyasini 1-qadamdagi haqiqiy DDP kodiga ko'chiring: `model.no_sync()` konteksti yordamida K=4 mikro-paketdan keyin bitta all-reduce bajaring. K=1 va K=4 uchun 2 jarayonli o'qitish vaqtini va yakuniy yo'qotishni solishtiring; all-reduce chaqiruvlari soni K baravar kamayganini tushuntiring. | o'rta | 10 |
| 7 | **Inferens kechikish profili** — 5-qadam kodida bo'lak hajmini SHARD = 64, 256, 1024, 4096, 16384 qilib, 1 va 2 ishchi uchun o'tkazuvchanlik hamda p50/p99 kechikishni o'lchang. Ikki o'qli grafik (o'tkazuvchanlik va p99) chizing va onlayn xizmat uchun (p99 < 5 ms sharti) optimal bo'lak hajmini asoslang. | o'rta | 10 |
| 8 | **Qo'lda DDP: all_reduce bilan** — DDP o'ramisiz 2 jarayonli o'qitish yozing: 1-qadam modelida har `backward()` dan keyin har parametr gradienti uchun `dist.all_reduce(p.grad)` va `/ world_size` bajaring, boshida esa 0-rank parametrlarini `dist.broadcast` bilan tarqating. Natija DDP varianti bilan bir xil parametrlarga (farq < 1e-6) olib kelishini ko'rsating va ikkala variant vaqtini solishtiring (DDP chelak/overlap afzalligi). | qiyin | 15 |
| 9 | **Sekin ishchi (straggler) simulyatsiyasi** — 1-qadam DDP kodida rank 1 ga har qadamda `time.sleep(0.002)` qo'shib sun'iy sekinlashtiring. 2 jarayon vaqti qanday o'zgaradi va nega rank 0 ham sekinlashadi (sinxron SGD)? Keyin 3 ta turli kechikish (0, 1, 3 ms) uchun E(2) jadvalini tuzing va sekin ishchi muammosini kamaytiradigan ikkita yondashuvni (masalan, asinxron SGD, elastik o'qitish) taqqoslab yozing. | qiyin | 15 |
| 10 | **Klaster konfiguratsiyasi tanlovi** — 7-qadam kalkulyatorini kengaytiring: berilgan model (parametr soni va GFLOP talaba tanlaydi), byudjet (GPU soni ≤ 32) va uchta tarmoq varianti uchun qadam vaqtini, E(N) ni va «1 epoxa narxi»ni (GPU-soat) hisoblaydigan funksiya yozing. Eng kam GPU-soat beradigan (N, tarmoq, accum) kombinatsiyasini toping, natijani issiqlik xaritasi (N x accum) ko'rinishida chizing va tanlovni asoslang. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Telekom: abonent segmentlari tasnifi — 2 jarayonli DDP va paket ta'siri (oson)

Soha: Telekom (mobil operator)

Mobil operator abonentlarning kunlik chaqiruv va trafik yozuvlaridan (CDR) 6 ta tarif segmentini bashorat qiluvchi MLP'ni 2 GPU'li serverda o'qitmoqchi. Sandbox'da GPU yo'q, shuning uchun 2 CPU yadro va gloo backend bilan DDP mantig'i aynan shu tarzda tekshiriladi. Talab: har rank ma'lumotning kesishmaydigan yarmini ko'rishi, parametrlar sinxron bo'lishi (farq 0), o'tkazuvchanlik uchta paket hajmi uchun o'lchanishi va 10 Gb/s tarmoqda aloqa vaqti baholanishi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik CDR: make_classification(n=6000, 40 belgi, n_informative=20, 6 sinf, random_state=7) |
| Model | MLP 40-192-192-6 (taxminan 46 ming parametr) |
| Paket (har rank) | 32, 64, 128 |
| Epoxa | 3 |
| LR qoidasi | 0.03 * world_size, momentum 0.9 |
| Tarmoq (kalkulyator) | 10 Gb/s |
| Ishchilar | 2 jarayon, gloo, 1 yadro/jarayon |

Qadamlar:

1. Laboratoriya s1 kodini make_classification ma'lumotiga moslang (X ni float32, standartlang). BS=32, 64, 128 va EPOCHS=3 uchun DDP'ni ishga tushiring; har holat uchun jami ko'rilgan namunalar (36000 bo'lishi kerak), namuna/s, oxirgi yo'qotish, aniqlik va ranklar orasidagi parametr farqini jadvalga yozing.
2. s2 kabi HID=192, GB=128, STEPS=60 bilan 4 konfiguratsiyani (1 jarayon; DDP kuchli; DDP kuchsiz; 1 jarayon 2 yadro) o'lchang, S(2) va E(2) ni hisoblang.
3. s3 ring all-reduce funksiyasini N=2, 4, 8 uchun tekshiring (natija to'g'ri=True), so'ng o'z modelingiz gradienti (46 ming parametr, FP32) va ResNet-18 (11.7 mln) uchun 1, 10, 100 Gb/s tarmoqlarda bitta all-reduce vaqtini N=2, 8, 32 da hisoblang.
4. s4 kabi 96 namunali global paket uchun 1x96, 3x32 (to'plash, 1/K bilan va 1/K siz) va 2 ishchi x 48 gradientlarini solishtiring; max farqlarni chiqaring.
5. s5 kabi o'qitilgan modelni NumPy'ga eksport qilib, 300 ming sintetik CDR so'rovini SHARD=1024 bilan 1 va 2 ishchida bajaring; so'rov/s, S(2), E(2), p50/p99 bo'lak kechikishini jadvalga kiriting.
6. 2-qadamdagi kuchli rejim S(2) dan Amdal bo'yicha p ni baholang, S(8) va S(32) hamda chegarani hisoblang; s6 grafigiga o'lchangan nuqtani qo'shing.
7. Xulosa: paket kattalashganda o'tkazuvchanlik nega oshadi, 10 Gb/s tarmoq shu model uchun to'siq bo'ladimi, 2 GPU serverga qaysi paketni tavsiya qilasiz.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 2-variant. Agro: tuproq namligi sensorlari — kuchli va kuchsiz masshtablash (oson)

Soha: Agro

Agroklaster 8 ta tuproq sensorining 24 soatlik o'lchovlari (192 belgi) bo'yicha dalaning sug'orish holatini 4 sinfga (shoshilinch, bugun, 2 kundan keyin, kerak emas) ajratuvchi MLP'ni fermer xo'jaligi serverida (2 ta GPU, tugunlar orasida 1 Gb/s tarmoq) o'qitmoqchi. Sizdan: CPU sandbox'da kuchli va kuchsiz masshtablashni ikki model o'lchami uchun o'lchash va 1 Gb/s tarmoqda aloqa vaqtini baholab, qaysi rejim maqsadga muvofiqligini aytish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik sensor matritsasi: 8000 x 192 (sinus kunlik sikl + shovqin), 4 sinf |
| Model | MLP 192-HID-HID-4, HID = 256 va 768 |
| Global paket | 128 va 256 |
| STEPS | 50 |
| Tarmoq | 1 Gb/s (tugunlar orasida), NVLink muqobili yo'q |
| Gradient formati | FP32 va BF16 |

Qadamlar:

1. Sensor generatorini yozing: har sensor uchun kunlik sinus + tasodifiy shovqin, sinf namlik o'rtachasi bo'yicha; s1 DDP kodini shu ma'lumotga moslang, EPOCHS=2, BS=64; parametr farqi 0 ekanini ko'rsating.
2. s2 kodini HID=256 va HID=768, GB=128 va 256 uchun (4 kombinatsiya) ishga tushiring; har biri uchun T(1), kuchli va kuchsiz DDP vaqtini, S(2), E(2) ni jadvalga yozing.
3. E(2) ni HID bo'yicha ikki rejimda ustunli diagrammada chizing; kichik modelda kuchli rejim samaradorligi nega past ekanini izohlang.
4. s3 formulasidan HID=768 modeli gradient hajmini (parametrlar sonini kod bilan hisoblang) FP32 va BF16 da toping va 1 Gb/s tarmoqda N=2, 4, 8 uchun all-reduce vaqtini hisoblang; ring simulyatsiyasini N=4 da tekshiring.
5. s4 kabi gradient to'plash K=2 va K=4 ekvivalentligini 128 namunali paketda tekshiring (farq < 1e-6).
6. s6 Gustafson formulasi bilan kuchsiz rejim uchun S(N) ni N=2..64 da chizing, o'lchangan E(2) dan p ni baholang.
7. Xulosa: fermer xo'jaligi klasteri uchun kuchsiz masshtablash + BF16 + gradient to'plash kombinatsiyasini raqamlar bilan asoslang.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 3-variant. Moliya: firibgarlik detektori gradientlari uchun halqa all-reduce va tarmoq tanlovi (oson)

Soha: Moliya (to'lov tizimi)

To'lov tizimi uchta nomzod modelni ko'rib chiqmoqda: kichik MLP (2.1 mln parametr), jadval-transformer (45 mln) va matnli izohlar uchun 7 mlrd parametrli til modeli. Infratuzilma bo'limi 1, 10 va 100 Gb/s tarmoq variantlarini taklif qiladi. Sizdan har model va tarmoq uchun bitta all-reduce vaqtini hisoblash, NumPy ring all-reduce simulyatsiyasini tekshirish va qaysi model DDP bilan umuman masshtablanmasligini ko'rsatish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Modellar | MLP 2.1 mln; TabTransformer 45 mln; LLM 7 mlrd parametr |
| Tarmoqlar | 1, 10, 100 Gb/s |
| N (GPU) | 2, 4, 8, 16, 64 |
| Gradient formati | FP32 (4 bayt) va BF16 (2 bayt) |
| Simulyatsiya vektori | L = 2400 element, N = 2, 3, 6 |
| Hisob vaqti (taqqoslash) | T_comp = 120 ms/qadam |

Qadamlar:

1. s3 ring_allreduce funksiyasini L=2400 va N=2, 3, 6 (N ga bo'linmaydigan uzunlik!) uchun ishga tushiring; natija to'g'ri=True va yuborilgan baytlar 2(N-1)/N*S formulasiga mos ekanini ko'rsating.
2. Funksiyani SUM/N (o'rtacha) qaytaradigan qilib o'zgartiring va DDP shu rejimda ishlashini izohlang.
3. Uchta model x uchta tarmoq x N=2,4,8,16,64 uchun har GPU yuboradigan hajm (MB) va aloqa vaqti (ms) jadvalini FP32 da chiqaring; BF16 uchun takrorlang.
4. T_comp=120 ms deb olib, r = T_comm/T_comp va E(N) = T_comp/(T_comp+T_comm) ni hisoblang; E(8) < 50% bo'lgan kombinatsiyalarni belgilang.
5. s1 DDP kodini make_classification(n=5000, 30 belgi, 2 sinf, weights=[0.97, 0.03]) firibgarlik ma'lumotiga moslang va 2 jarayonda parametr sinxronligini tekshiring (DDP mantig'i kichik modelda).
6. Xulosa: 7 mlrd model uchun DDP nega yaroqsiz (xotira va 2S hajm), FSDP/ZeRO qanday yordam beradi, 45 mln model uchun minimal tarmoq qaysi.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 4-variant. Tibbiyot: EKG segmentlari tasnifi — gradient to'plash ekvivalentligi (oson)

Soha: Tibbiyot (kardiologiya)

Kardiologiya markazi 256 nuqtali EKG segmentlarini 3 sinfga (normal, aritmiya, artefakt) ajratuvchi modelni o'qitadi. Klinika serverida xotira kam, shuning uchun 192 namunali global paketni bir yo'la sig'dirib bo'lmaydi — gradient to'plash yoki 2 GPU'da ma'lumotlar parallelligi kerak. Sizdan to'plash, DDP o'rtachalash va bitta katta paket bir xil gradient berishini isbotlash, xato variantlarni (1/K siz, BatchNorm bilan) ko'rsatish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik EKG: 256 nuqta, sinus + QRS impuls + shovqin, 3 sinf, n=3000 |
| Model | MLP 256-96-3 (BatchNorm bilan va BatchNorm'siz) |
| Global paket | 192 |
| K (mikro-paket) | 3 va 6 |
| Ishchilar (qo'lda DDP) | 2 x 96 va 4 x 48 |
| Loss reduction | mean va sum |

Qadamlar:

1. EKG generatorini yozing (har sinf uchun boshqa impuls shakli); s4 kodini 192 namunali paketga moslang: (A) 1x192, (B) K=3 to'plash 1/K bilan, (C) 2 ishchi x 96 o'rtacha; max farqlarni chiqaring (< 1e-6).
2. K=6 va 4 ishchi x 48 variantlarini qo'shing; (B') 1/K siz to'plashda ||gE||/||gA|| nisbati K ga teng ekanini ko'rsating.
3. Modelga nn.BatchNorm1d(96) qo'shing va ekvivalentlik buzilishini raqam bilan ko'rsating; nega buzilishini izohlang (SyncBatchNorm haqida 2 jumla).
4. reduction='sum' bilan to'plashda 1/K kerak emasligini tekshiring va LR qanday o'zgarishi kerakligini yozing.
5. s1 DDP kodini shu EKG ma'lumotiga moslang (BS=48, EPOCHS=3) va 2 jarayonda parametr farqi 0 ekanini ko'rsating; namuna/s ni o'lchang.
6. s3 formulasi bilan model gradienti (parametr sonini hisoblang) uchun 1 Gb/s klinika tarmog'ida all-reduce vaqtini N=2 da toping va to'plash K=6 aloqa chastotasini qanday kamaytirishini jadvalda ko'rsating.
7. Xulosa: xotira cheklangan klinika serverida qaysi usulni (to'plash yoki DDP) tavsiya qilasiz va nima uchun.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 5-variant. Chakana savdo: mahsulot toifasini aniqlash inferensi — ishchilar puli (oson)

Soha: Chakana savdo

Onlayn do'kon har kecha 500 ming yangi mahsulot kartochkasini 12 toifaga avtomatik ajratadi. Hozir bu bitta jarayonda 40 daqiqa olmoqda; rahbariyat 2 va 4 yadroli serverda parallel inferens qanchalik tezlashtirishini bilmoqchi. Sizdan multiprocessing Pool bilan sharding, natijalarni tartib bilan yig'ish va bo'lak hajmining o'tkazuvchanlik hamda kechikishga ta'sirini o'lchash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | make_classification(n=5000, 50 belgi, 12 sinf, n_informative=30, random_state=3), 100 marta takrorlab 500 ming so'rov |
| Model | MLP 50-128-12, Adam, 200 qadam |
| SHARD | 512, 2048, 8192 |
| Ishchilar | 1, 2 (va 4 — 2 yadroda sinov) |
| SLO | p99 bo'lak kechikishi < 8 ms |
| Yig'ish | tartiblangan map va imap_unordered |

Qadamlar:

1. s5 kodini make_classification ma'lumotiga moslang: modelni o'qiting (aniqlikni yozing), og'irliklarni NumPy'ga eksport qiling, 500 ming so'rovni tayyorlang.
2. SHARD=2048 bilan 1 va 2 ishchi uchun so'rov/s, S(2), E(2), p50/p99 bo'lak kechikishini o'lchang; aniqlik ikkala holatda bir xil ekanini tekshiring.
3. SHARD=512 va 8192 uchun takrorlang; 3x2 jadval tuzing va ikki o'qli grafik (so'rov/s va p99) chizing; SLO p99 < 8 ms ni qondiruvchi eng katta bo'lakni tanlang.
4. 4 ishchini 2 yadroda ishga tushiring — E(4) nega 50% atrofida yoki past bo'lishini izohlang.
5. pool.imap_unordered bilan res.sort'siz yig'ing va aniqlik qanday buzilishini ko'rsating; so'ng idx bo'yicha tiklang.
6. s6 Amdal formulasi bilan o'lchangan S(2) dan p ni baholang va 8 yadroli server uchun S(8) ni bashorat qiling; 40 daqiqa qancha bo'lishini hisoblang.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 6-variant. Ta'lim: o'quvchi natijalarini bashorat qiluvchi klaster — Amdal va Gustafson (oson)

Soha: Ta'lim

Respublika ta'lim portali 2 mln o'quvchi yozuvi bo'yicha o'zlashtirish xavfini (3 daraja) bashorat qiluvchi modelni o'qitadi va IT bo'limi 4, 16 yoki 64 GPU sotib olish masalasini hal qilmoqda. Sizdan CPU'da 2 jarayonli o'lchov o'tkazib, ketma-ket ulushni baholash va Amdal hamda Gustafson qonunlari asosida har uchala klaster hajmi uchun kutilayotgan tezlashishni hisoblab berish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik o'quvchi yozuvlari: 12000 x 24 belgi (davomat, baholar, ijtimoiy ko'rsatkichlar), 3 sinf |
| Model | MLP 24-256-256-3 |
| p (nazariy) | 0.80, 0.92, 0.98 |
| N | 1, 2, 4, 8, 16, 32, 64, 128, 512 |
| Global paket (kuchli) | 192 |
| STEPS | 60 |

Qadamlar:

1. s1 DDP kodini o'quvchi ma'lumotiga moslang (EPOCHS=2, BS=96) va 2 jarayonda to'g'ri ishlashini (parametr farqi 0, jami 24000 namuna) ko'rsating.
2. s2 kodi bilan HID=256, GB=192, STEPS=60 uchun kuchli va kuchsiz rejimda S(2), E(2) ni o'lchang.
3. Kuchli rejim S(2) dan Amdal bo'yicha ketma-ket ulushni toping: 1-p = (2/S(2) - 1); p qiymatini yozing.
4. s6 kodini p=0.80, 0.92, 0.98 va o'lchangan p uchun N=1..512 oralig'ida ishga tushiring; S(N), E(N) jadvali va log-log grafik (o'lchangan nuqta bilan).
5. Gustafson formulasi bilan kuchsiz rejim uchun S(4), S(16), S(64) ni hisoblang; ikki qonun farqini bitta grafikda ko'rsating.
6. Xulosa: 4, 16, 64 GPU uchun kutilayotgan E(N) jadvali; Amdal bo'yicha 64 GPU sotib olish o'zini oqlaydimi, kuchsiz rejimda-chi.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 7-variant. Energetika: elektr yuklama prognozi — aloqa/hisob kalkulyatori (V100 klasteri) (oson)

Soha: Energetika

Energiya kompaniyasi 3 ta vaqt qatori modelini (kichik LSTM, TCN, Transformer) 8 ta V100 GPU'li klasterda o'qitmoqchi; tugunlar orasidagi tarmoq 1, 10 yoki 100 Gb/s bo'lishi mumkin. Sizdan s7 analitik modeli bilan har model va tarmoq uchun T_comp, T_comm, r va E(8) ni hisoblash, gradient to'plash ta'sirini ko'rsatish va byudjet doirasida tarmoq tanlash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Modellar | LSTM-kichik 2.3 mln / 0.5 GFLOP; TCN 9 mln / 1.2 GFLOP; Transformer 110 mln / 22 GFLOP (bir namuna forward) |
| GPU | V100 FP16 cho'qqi 125 TFLOPS, MFU 0.35 |
| Tarmoqlar | 1, 10, 100 Gb/s |
| Paket/GPU | 64 |
| Gradient | BF16 (2 bayt) |
| Overlap | 0.5 |
| Accum | 1, 2, 4, 8 |

Qadamlar:

1. s7 kodidagi MODELS, NETS, PEAK, MFU, BATCH ni variant qiymatlariga almashtiring va N=8 uchun 3x3 jadvalni (T_comm, r, E) chiqaring.
2. Transformer uchun accum=1, 2, 4, 8 da 10 Gb/s tarmoqda E(8) ni hisoblang va to'plash qaysi qiymatdan boshlab E > 80% berishini toping.
3. Har tarmoq uchun N=1..64 da E(N) egri chiziqlarini TCN modeli uchun chizing (log2 o'q).
4. BYTES_PER_GRAD=4 (FP32) va OVERLAP=0 holatlarini qo'shib, eng yomon va eng yaxshi holat jadvalini tuzing.
5. s3 ring simulyatsiyasini N=8 da tekshiring va LSTM-kichik gradient hajmi uchun 1 Gb/s da haqiqiy all-reduce vaqtini formula bilan solishtiring.
6. Xulosa: 10 Gb/s tarmoq qaysi modellar uchun yetarli, 100 Gb/s qachon zarur, kichik LSTM uchun 8 GPU umuman kerakmi.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 8-variant. Transport: yo'l tirbandligi bashorati — sekin ishchi (straggler) ta'siri (o'rta)

Soha: Transport

Shahar transport markazi 300 ta yo'l sensoridan kelayotgan oqim bo'yicha 15 daqiqalik tirbandlik darajasini (5 sinf) bashorat qiluvchi modelni 2 GPU'da o'qitadi, lekin GPU'lardan biri boshqa xizmat bilan birga ishlaydi va vaqti-vaqti bilan sekinlashadi. Sizdan sinxron SGD'da bitta sekin ishchi butun o'qitishni qanday tutib qolishini o'lchash va kechikish darajasiga qarab E(2) jadvalini tuzish, so'ng yumshatish usullarini taqqoslash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik sensor oqimi 10000 x 300, 5 sinf (soat va hafta kuni bo'yicha naqsh) |
| Model | MLP 300-384-384-5 |
| Sun'iy kechikish (rank 1) | 0, 1, 2, 4 ms har qadamda |
| BS (har rank) | 64 |
| EPOCHS | 2 |
| Qadam vaqti (baza) | o'lchanadi |
| Tarmoq (kalkulyator) | 10 Gb/s |

Qadamlar:

1. s1 DDP kodini transport ma'lumotiga moslang va kechikishsiz baza vaqtini hamda bitta qadam o'rtacha vaqtini (ms) o'lchang.
2. rank 1 ga har qadamda time.sleep(d) qo'shing (d = 1, 2, 4 ms); har d uchun ikkala rank vaqtini, namuna/s va E(2) = T_baza/T_d ni jadvalga yozing. Rank 0 nega ham sekinlashadi — all-reduce to'siq (barrier) tushunchasi bilan izohlang.
3. Kechikishning qadam vaqtiga nisbati (d / T_qadam) bo'yicha E(2) grafigini chizing va chiziqli model E = 1/(1 + d/T_qadam) bilan solishtiring.
4. s2 kodi bilan kechikishsiz kuchli/kuchsiz S(2) ni o'lchang va straggler ta'sirini shu jadvalga qo'shimcha ustun qiling.
5. s3 formulasi bilan model gradienti hajmini va 10 Gb/s da aloqa vaqtini hisoblang; aloqa vaqti bilan 4 ms kechikishni taqqoslang.
6. s6 Amdal modeliga straggler'ni ketma-ket ulush sifatida qo'shing va N=8 da kutilayotgan E(8) ni baholang.
7. Xulosa: asinxron SGD, zaxira ishchilar (backup workers) va elastik o'qitishning afzallik/kamchiliklarini jadvalda taqqoslang va transport markazi uchun tanlov bering.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 9-variant. Logistika: ombor buyurtmalarini marshrutlash inferensi — bo'lak hajmi va p99 (o'rta)

Soha: Logistika

Katta ombor har soatda 1.2 mln buyurtma qatorini 16 ta yig'ish zonasiga taqsimlovchi modelni ishlatadi; onlayn rejimda har bo'lak uchun p99 kechikish 4 ms dan oshmasligi kerak, kechki qayta ishlashda esa maksimal o'tkazuvchanlik muhim. Sizdan bo'lak hajmi bo'yicha o'tkazuvchanlik va p50/p99 profilini 1 va 2 ishchida o'lchash, ikki rejim uchun turli optimal bo'lakni asoslash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | make_classification(n=6000, 36 belgi, 16 sinf, n_informative=24, random_state=11), 200 marta takror (1.2 mln so'rov) |
| Model | MLP 36-160-16 |
| SHARD | 64, 256, 1024, 4096, 16384 |
| Ishchilar | 1, 2 |
| SLO onlayn | p99 bo'lak < 4 ms |
| Maxsus | imap chunksize ta'siri |

Qadamlar:

1. s5 kodini moslang: modelni o'qiting, 1.2 mln so'rov tayyorlang; SHARD ro'yxati bo'yicha 1 va 2 ishchida so'rov/s, S(2), E(2), p50, p99 ni 5x2 jadvalga yozing.
2. Ikki o'qli grafik: x — SHARD (log), chap o'q — so'rov/s, o'ng o'q — p99 ms; 1 va 2 ishchi egri chiziqlari.
3. Onlayn rejim (p99 < 4 ms) va kechki rejim (maks. so'rov/s) uchun optimal SHARD ni tanlang va asoslang.
4. pool.map o'rniga pool.imap(..., chunksize=c) bilan c=1, 4, 16 uchun kichik SHARD=64 da IPC xarajati qanday kamayishini o'lchang.
5. 4 ishchi 2 yadroda: E(4) ni o'lchang va kontekst almashinuvi ta'sirini izohlang.
6. s6 Amdal bilan eng yaxshi S(2) dan p ni baholang va 8 yadroli server uchun soatlik 1.2 mln buyurtma necha daqiqada bajarilishini bashorat qiling.
7. Xulosa: ikki rejim uchun konfiguratsiya jadvali (SHARD, ishchilar, kutilgan p99, so'rov/s) va dinamik paketlash haqida tavsiya.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 10-variant. Sanoat: dastgoh nosozligini bashorat — masshtablash egri chizig'i (paket va model o'lchami) (o'rta)

Soha: Sanoat

Mashinasozlik zavodi 120 ta vibratsiya belgisidan dastgoh nosozligini 7 sinfga ajratuvchi modelni o'qitadi va 2 GPU'li stansiyada kuchli masshtablash qachon to'yinishini bilmoqchi. Sizdan global paket va model kengligi bo'yicha E(2) egri chiziqlarini ikki rejimda o'lchash, to'yinish nuqtasini aniqlash va bucket_cap_mb ta'sirini tekshirish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik vibratsiya spektrlari 8000 x 120, 7 sinf |
| Model | MLP 120-HID-HID-7, HID = 256 va 1024 |
| Global paket | 32, 64, 128, 256, 512 |
| STEPS | 40 |
| DDP sozlamasi | bucket_cap_mb = 1 va 25 (standart) |
| Tarmoq (kalkulyator) | 100 Gb/s |

Qadamlar:

1. s2 kodini tsiklga o'rab, HID=256 uchun GB ro'yxati bo'yicha T(1), kuchli va kuchsiz DDP vaqtini o'lchang; E(2) ni ikki rejimda hisoblang (jami vaqt 90 s dan oshsa STEPS ni 25 ga tushiring).
2. HID=1024 uchun takrorlang (kerak bo'lsa GB ro'yxatini 64..512 qiling).
3. Bitta grafikda 4 egri chiziq: (HID, rejim) bo'yicha E(2) vs GB; kuchli rejim qaysi GB dan boshlab to'yinishini (E > 85%) har HID uchun aniqlang.
4. DDP(model, bucket_cap_mb=1) bilan HID=1024, GB=256 ni qayta o'lchang va standart bilan solishtiring; chelak soni ko'payganda nima o'zgaradi.
5. s1 kodini vibratsiya ma'lumotiga moslang (EPOCHS=2) va parametr sinxronligi hamda aniqlikni ko'rsating.
6. s3 formulasi bilan HID=1024 model gradienti uchun 100 Gb/s da all-reduce vaqtini va s7 modeli bilan r nisbatini baholang; o'lchangan E(2) bilan mos keladimi.
7. Xulosa: zavod stansiyasi uchun GB va rejim tavsiyasi, to'yinish nuqtasi va chiziqli LR qoidasi haqida eslatma.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 11-variant. Video-kuzatuv: kadr embedding tasnifi — parametr-server va halqa taqqoslovi (o'rta)

Soha: Xavfsizlik

Xavfsizlik kompaniyasi 2048 o'lchamli kadr embedding'laridan 9 ta hodisa turini aniqlovchi modelni ko'p GPU'li klasterda o'qitadi; arxitektura guruhi parametr-server (PS) va halqa all-reduce sxemalari orasida tanlov qilmoqda, tugunlar orasida 10 Gb/s tarmoq bor. Sizdan NumPy'da ikkala sxemani yozish, eng band tugun trafigini N bo'yicha solishtirish va PS qaysi N dan boshlab 100 ms chegarasidan oshishini topish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik embedding 6000 x 2048 (9 ta markaz + shovqin), 9 sinf |
| Model (gradient hajmi) | embedding tasniflagich 2048-512-9 (taxminan 1.05 mln) va ResNet-50 23 mln |
| N | 2, 4, 8, 16, 32, 64 |
| Tarmoq | 10 Gb/s |
| Chegara | 100 ms / all-reduce |
| Simulyatsiya | L = 4096, N = 2, 4, 8 |

Qadamlar:

1. s3 ring_allreduce ni L=4096, N=2,4,8 da tekshiring; so'ng parametr-server sxemasini yozing (hamma 0-ishchiga yuboradi, u yig'ib hammaga qaytaradi) va natija bir xil ekanini ko'rsating.
2. Har ikki sxema uchun eng band tugun yuboradigan + qabul qiladigan baytlarni N ro'yxati bo'yicha jadvalga yozing (S = model hajmi FP32, ikkala model uchun).
3. 10 Gb/s da vaqtni hisoblang; PS sxemasi ResNet-50 uchun qaysi N dan 100 ms dan oshadi; halqa uchun chegara 2S/BW ekanini grafikda ko'rsating (log-log).
4. s1 DDP kodini embedding ma'lumotiga moslang (BS=64, EPOCHS=2) va parametr sinxronligini tekshiring; DDP ichida qaysi sxema ishlatilishini izohlang.
5. s2 bilan HID=512 uchun S(2), E(2) ni o'lchang.
6. s7 kalkulyatoriga PS sxemasini qo'shing (T_comm_PS = 2(N-1)*S/BW server uchun) va ResNet-50, N=8 da E(8) ni ikki sxemada solishtiring.
7. Xulosa: qaysi N gacha PS maqbul, halqa all-reduce'ning kamchiligi (kechikish, straggler) va kompaniya uchun tavsiya.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 12-variant. Media: yangiliklar matnini tasniflash — no_sync bilan gradient to'plash DDP'da (o'rta)

Soha: Media

Yangiliklar agentligi 3000 o'lchamli so'z-sumka vektorlaridan 8 ta rubrikani aniqlovchi modelni 2 GPU'da o'qitadi; tarmoq sekin bo'lgani uchun har qadamda all-reduce qilish qimmat. Sizdan haqiqiy DDP kodida model.no_sync() konteksti yordamida K mikro-paketdan keyin bitta all-reduce bajarish, K bo'yicha vaqt va yakuniy yo'qotishni o'lchash hamda all-reduce soni K baravar kamayganini ko'rsatish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik so'z-sumka: 6000 x 3000 (siyrak, 8 rubrika markazi), float32 |
| Model | MLP 3000-256-8 (taxminan 770 ming parametr) |
| K (to'plash) | 1, 2, 4, 8 |
| Mikro-paket (har rank) | 32 |
| EPOCHS | 2 |
| Tarmoq (kalkulyator) | 1 Gb/s |
| Tekshiruv | all-reduce soni = qadamlar / K |

Qadamlar:

1. s1 DDP kodini so'z-sumka ma'lumotiga moslang (BS=32, EPOCHS=2) va K=1 baza vaqtini, namuna/s va yakuniy yo'qotishni o'lchang.
2. O'qitish siklini qayta yozing: K-1 mikro-paket model.no_sync() ichida backward, K-chisi oddiy backward + opt.step(); yo'qotishni 1/K ga ko'paytiring. K=2, 4, 8 uchun vaqt, namuna/s, yakuniy yo'qotish jadvali.
3. All-reduce chaqiruvlari sonini hisoblang (DDP hook yoki hisoblagich bilan) va K baravar kamayganini ko'rsating; chiziqli LR qoidasi global paket K*32*2 uchun qanday bo'lishi kerakligini yozing.
4. s4 kabi K=4 to'plash + 2 ishchi kombinatsiyasining bitta 256 namunali paket bilan gradient ekvivalentligini tekshiring (farq < 1e-6).
5. s3 formulasi bilan 770 ming parametr gradienti uchun 1 Gb/s da bitta all-reduce vaqtini hisoblang va K bo'yicha qadamga to'g'ri keladigan aloqa vaqtini jadvalga qo'shing.
6. s6 Amdal modelida aloqa ulushini K ga bo'lib, N=8 uchun kutilgan E(8) ni K=1 va K=8 da baholang.
7. Xulosa: K ning ortishi qaysi nuqtadan boshlab yaqinlashuvga zarar bera boshlaydi (yo'qotish jadvali), agentlik uchun tavsiya etilgan K.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 13-variant. Aqlli shahar: kamera oqimlari uchun klaster kalkulyatori (1/10/100 Gb/s) (o'rta)

Soha: Shahar

Shahar boshqaruv markazi 400 ta kameradan kelayotgan kadrlarni tahlil qiluvchi uchta modelni (kichik YOLO, BERT-base matn uchun, ViT-L) 16 GPU'li A100 klasterida o'qitishni rejalashtiradi; tarmoq uchun 1, 10 va 100 Gb/s variantlari va overlap darajasi noma'lum. Sizdan s7 kalkulyatorini kengaytirib, N=4, 8, 16 da E(N) ni hisoblash, overlap va gradient formatining ta'sirini ko'rsatish va har model uchun minimal yetarli tarmoqni tanlash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Modellar | YOLO-kichik 7 mln / 8 GFLOP; BERT-base 110 mln / 11 GFLOP (seq 128); ViT-L 304 mln / 61 GFLOP |
| GPU | A100 BF16 312 TFLOPS, MFU 0.4 |
| Tarmoqlar | 1, 10, 100 Gb/s |
| N | 4, 8, 16 |
| Overlap | 0.3 va 0.7 |
| Gradient | FP32, BF16 |
| Paket/GPU | 32 |

Qadamlar:

1. s7 kodidagi MODELS, NETS, BATCH, OVERLAP ni variant qiymatlariga almashtiring; N=8, overlap=0.5 uchun 3x3 jadval (T_comm, r, E).
2. Har model uchun N=4, 8, 16 x 3 tarmoq E(N) jadvalini tuzing va E > 80% shartini qondiruvchi minimal tarmoqni belgilang.
3. Overlap 0.3 va 0.7 uchun ViT-L, 10 Gb/s da E(16) ni solishtiring; DDP chelaklari overlap'ni qanday ta'minlashini izohlang.
4. BYTES_PER_GRAD=4 va 2 uchun BERT-base natijasini solishtiring; gradient siqish 1 Gb/s da yetarli bo'ladimi.
5. Accum=1,2,4,8 uchun YOLO-kichik, 1 Gb/s da E(16) ni hisoblang va grafik chizing.
6. s3 ring simulyatsiyasini N=16 da tekshiring va YOLO-kichik hajmi uchun har GPU yuboradigan baytlar 2S ga yaqinlashishini ko'rsating.
7. Xulosa: har model uchun (tarmoq, accum, gradient formati) tavsiyasi va umumiy klaster uchun bitta tarmoq tanlansa qaysi biri.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 14-variant. Ekologiya: sun'iy yo'ldosh tasvirlarini tasniflash — gradient siqish (FP32/BF16/FP8) (o'rta)

Soha: Ekologiya

Ekologik monitoring markazi 512 o'lchamli sun'iy yo'ldosh tasvir belgilaridan 6 ta yer qoplami turini aniqlovchi modelni 4 ta viloyat serverida (tugunlar orasida 1 Gb/s) o'qitmoqchi; aloqa to'siq bo'lgani uchun gradientlarni BF16 yoki FP8 ga siqish ko'rib chiqilmoqda. Sizdan NumPy ring all-reduce'da past aniqlikli formatlarning hajm va xatolik ta'sirini o'lchash, aloqa vaqtini hisoblash va DDP'da xatolik to'planishini baholash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik tasvir belgilari 6000 x 512, 6 sinf |
| Model | MLP 512-384-6 (taxminan 199 ming parametr) |
| Gradient formati | FP32, BF16 (torch.bfloat16 orqali), FP8-taqlid (int8 kvantlash, masshtab bilan) |
| Tarmoq | 1 Gb/s |
| N | 2, 4, 8, 16 |
| Xatolik o'lchovi | nisbiy L2 xato, kosinus o'xshashlik |

Qadamlar:

1. s3 ring_allreduce ga 'dtype' parametri qo'shing: yuborishdan oldin bo'lakni BF16 (torch orqali) yoki int8 (max-abs masshtab) ga aylantirib, qabulda qaytaring; N=4, L=8192 da uch format uchun yuborilgan baytlar va natijaning aniq yig'indiga nisbiy xatosini jadvalga yozing.
2. Formula bilan 199 ming parametr modeli va ResNet-50 (25.6 mln) uchun 1 Gb/s da N=2,4,8,16 all-reduce vaqtini uch formatda hisoblang.
3. s4 gradient ekvivalentligi kodida 2 ishchi gradientini BF16 va int8 ga siqib o'rtachalang; max farq va kosinus o'xshashlikni FP32 bilan solishtiring.
4. s1 DDP kodini tasvir belgilariga moslang va DDP'ga BF16 siqish hook'i (register_comm_hook, default_hooks.bf16_compress_hook) qo'shing; vaqt va yakuniy aniqlikni hook'siz holat bilan solishtiring.
5. s7 kalkulyatorida BYTES_PER_GRAD = 4, 2, 1 uchun ResNet-50, 1 Gb/s, N=4 da E(4) ni hisoblang.
6. Xulosa: qaysi format aniqlik yo'qotmasdan aloqa vaqtini yetarlicha kamaytiradi, xatolik-fikr (error feedback) nima uchun kerak, viloyat klasteri uchun tavsiya.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 15-variant. Sport: harakat sensorlari bo'yicha mashq turini aniqlash — DDP'ni qo'lda yozish (o'rta)

Soha: Sport

Sport texnologiyalari startapi bilak sensorlaridan (6 o'q x 50 nuqta = 300 belgi) 10 ta mashq turini aniqlovchi modelni o'qitadi va jamoa DDP o'rami ichida nima bo'layotganini tushunishi kerak. Sizdan DDP'siz 2 jarayonli o'qitishni qo'lda yozish (broadcast + har gradient uchun all_reduce), natijasi DDP bilan bir xil ekanini isbotlash va vaqt farqini (chelak/overlap afzalligi) o'lchash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik IMU signallari 5000 x 300 (10 mashq uchun turli chastotali sinuslar), 10 sinf |
| Model | MLP 300-256-128-10 |
| BS (har rank) | 64 |
| EPOCHS | 2 |
| Taqqoslash | DDP o'rami vs qo'lda broadcast + all_reduce(p.grad)/world |
| Tekshiruv | parametr farqi < 1e-6, bir xil urug' va sampler |

Qadamlar:

1. s1 DDP kodini IMU ma'lumotiga moslang va DDP bilan yakuniy parametrlarni (flat vektor) faylga saqlang; vaqtni o'lchang.
2. Qo'lda variant yozing: DDP o'ramisiz, boshida har parametr uchun dist.broadcast(p.data, src=0); har backward'dan keyin har p.grad uchun dist.all_reduce(SUM) va /world; sampler va urug' bir xil bo'lsin.
3. Ikkala variant yakuniy parametrlarini solishtiring (max farq < 1e-6) va vaqtni jadvalga yozing; DDP tezroq bo'lsa — chelaklar va overlap tufayli ekanini izohlang.
4. Qo'lda variantda gradientlarni bitta flat vektorga yig'ib (torch.cat) bitta all_reduce qiling; vaqt qanday o'zgaradi — all-reduce soni va kechikish (latency) rolini tushuntiring.
5. s3 formulasi bilan model gradienti hajmini va 10 Gb/s da all-reduce vaqtini hisoblang; parametrlar soni bo'yicha chaqiruvlar soni (qatlam soni) bilan solishtiring.
6. s2 kodi bilan S(2), E(2) ni DDP va qo'lda (flat) variant uchun o'lchang va jadvalga kiriting.
7. Xulosa: DDP nimani avtomatlashtiradi, qo'lda yozilgan variant qachon foydali (maxsus siqish, gradient klipping), startap uchun tavsiya.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (DDP sinxron, tekshiruvlar o'tadi) — 30, O'lchovlar va jadval (to'liq, takrorlanuvchan) — 25, Tahlil va xulosa (formulalar bilan asoslangan) — 25, Hisobot sifati (grafik, tuzilma, til) — 20 (jami 100)

## 16-variant. Bank: to'lov skoring modeli — eng arzon klaster konfiguratsiyasi (GPU-soat) (qiyin)

Soha: Moliya (bank)

Bankning ML platformasi 180 mln parametrli jadval-transformer skoring modelini 40 mln yozuvda 5 epoxa o'qitadi; byudjet 32 GPU'gacha, tarmoq variantlari 10, 25, 100 Gb/s, GPU-soat narxi tarmoqqa qarab farq qiladi. Sizdan s7 kalkulyatorini kengaytirib, (N, tarmoq, accum) kombinatsiyalari uchun qadam vaqti, E(N), epoxa davomiyligi va umumiy narxni hisoblash, eng arzon va eng tez variantlarni topish hamda natijani issiqlik xaritasida ko'rsatish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Model | TabTransformer 180 mln parametr, 6 GFLOP/namuna forward |
| Ma'lumot | 40 mln yozuv, 5 epoxa |
| GPU | A100, 312 TFLOPS BF16, MFU 0.38; narx 2.1 $/GPU-soat |
| Tarmoq va qo'shimcha narx | 10 Gb/s +0 %, 25 Gb/s +8 %, 100 Gb/s +20 % |
| N | 1, 2, 4, 8, 16, 32 |
| Accum | 1, 2, 4, 8, 16 |
| Paket/GPU | 256 |
| Cheklov | devor vaqti <= 12 soat |

Qadamlar:

1. s7 estimate funksiyasini kengaytiring: qadam vaqti, 1 epoxa vaqti = (yozuvlar / (N*paket*accum)) * T_step, umumiy devor vaqti (5 epoxa), GPU-soat va narx (tarmoq ustamasi bilan).
2. N x tarmoq x accum (6 x 3 x 5 = 90 kombinatsiya) uchun jadval tuzing; devor vaqti 12 soatdan oshgan variantlarni chiqarib tashlang.
3. Eng arzon va eng tez kombinatsiyani toping; narx va vaqt bo'yicha Pareto chegarasini nuqtali grafikda chizing.
4. 10 Gb/s tarmoq uchun N x accum issiqlik xaritasi (rang — narx, $) va E(N) uchun ikkinchi xarita.
5. s3 formulasi bilan 180 mln parametr gradienti (BF16) uchun har tarmoqda all-reduce vaqtini hisoblang va kalkulyator T_comm bilan mos ekanini tekshiring; s6 Amdal bilan N=32 da nazariy chegarani solishtiring.
6. Sezgirlik tahlili: MFU 0.3 va 0.5, overlap 0.3 va 0.8 uchun eng arzon variant o'zgaradimi — jadval.
7. Xulosa (bank rahbariyati uchun 1 sahifa): tavsiya etilgan N, tarmoq, accum; nega katta accum bepul emas (yaqinlashuv, global paket); cheklovlar.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadval va grafiklar — 25, Chuqur tahlil (analitik model vs o'lchov, cheklovlar) — 30, Hisobot sifati va tavsiya asosliligi — 20 (jami 100)

## 17-variant. Genomika: variant patogenligi tasnifi — DDP va gradient to'plash kombinatsiyasi (qiyin)

Soha: Genomika (biotibbiyot)

Genomika laboratoriyasi 1024 o'lchamli ketma-ketlik belgilari bo'yicha mutatsiya patogenligini 4 sinfga ajratadi; 2 GPU'da xotira faqat 32 namunali mikro-paketga yetadi, lekin tajribalar 512 global paket talab qiladi. Sizdan haqiqiy DDP + no_sync bilan 2 ishchi x K mikro-paket sxemasini yozish, uning bitta 512 paket bilan gradient ekvivalentligini isbotlash, all-reduce sonini, vaqtni va yaqinlashuvni o'lchash hamda BatchNorm/SyncBatchNorm masalasini tahlil qilish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik genomik belgilar 8192 x 1024 (4 sinf, siyrak motivlar), float32 |
| Model | MLP 1024-512-128-4 (BatchNorm'li va BatchNorm'siz) |
| Mikro-paket (har rank) | 32 |
| Global paket | 512 (2 ishchi x K=8) |
| K | 1, 2, 4, 8 |
| EPOCHS | 2 |
| LR qoidasi | 0.02 * (global paket / 64) |
| Tekshiruv | max|g_512 - g_ddp_accum| < 1e-5 |

Qadamlar:

1. s1 DDP kodini genomik ma'lumotga moslang; BS=32 va K=1 baza vaqtini o'lchang.
2. DDP + no_sync bilan K=2, 4, 8 to'plashni yozing (oxirgi mikro-paketda sinxron); har K uchun vaqt, all-reduce soni, yakuniy yo'qotish va aniqlik jadvali; LR ni qoidaga ko'ra masshtablang.
3. Ekvivalentlik isboti: bir xil 512 namuna uchun (a) bitta jarayon 1x512 gradienti, (b) 2 ishchi x 8 x 32 DDP+accum gradienti (opt.step oldidan p.grad ni yig'ib oling) — max farq < 1e-5 ekanini ko'rsating.
4. Modelga BatchNorm1d qo'shing: ekvivalentlik buzilishini o'lchang; nn.SyncBatchNorm.convert_sync_batchnorm qo'llang va farq qanday o'zgarishini (to'plash ichida baribir qoladi) izohlang.
5. s3 formulasi bilan model gradienti hajmi (parametrlar sonini hisoblang) uchun 10 Gb/s da all-reduce vaqtini hisoblang; K bo'yicha epoxa uchun jami aloqa vaqtini jadvalga qo'shing.
6. s6 Amdal va s7 kalkulyatori bilan N=8 GPU, 10 Gb/s da K=1 va K=8 uchun E(8) ni baholang; o'lchangan 2 jarayon natijasi bilan taqqoslang.
7. Xulosa: 2 GPU xotira cheklovida 512 global paketga erishish rejasi, yaqinlashuv uchun LR/warmup eslatmasi, BatchNorm o'rniga LayerNorm tavsiyasi.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadval va grafiklar — 25, Chuqur tahlil (analitik model vs o'lchov, cheklovlar) — 30, Hisobot sifati va tavsiya asosliligi — 20 (jami 100)

## 18-variant. 5G tarmoq: anomaliya detektori — straggler va asinxron SGD simulyatsiyasi (qiyin)

Soha: Telekom (5G infratuzilma)

5G operatori 64 ta bazaviy stansiya serverida tarmoq anomaliyalarini aniqlovchi modelni taqsimlangan tarzda o'qitadi; serverlar bir xil emas, ba'zilari 2-3 marta sekin. Sizdan sinxron SGD, zaxira ishchili sinxron SGD (eng sekin k tasini kutmaslik) va asinxron (eskirgan gradientli) SGD'ni NumPy'da simulyatsiya qilish, bir xil hisoblash byudjetida yaqinlashuv va devor vaqtini solishtirish hamda 2 jarayonli haqiqiy DDP'da straggler o'lchovini olish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik KPI oqimi 20000 x 48 (2 sinf: normal/anomaliya, 5 % anomaliya) |
| Model (simulyatsiya) | logistik regressiya 48 og'irlik, NumPy'da qo'lda gradient |
| Ishchilar (simulyatsiya) | N = 16, qadam vaqti log-normal (sigma = 0.3 va 0.8) |
| Sxemalar | sinxron; sinxron + 2 zaxira (eng sekin 2 tasi kutilmaydi); asinxron (eskirish tau <= 4) |
| Haqiqiy DDP | 2 jarayon, rank 1 kechikishi 0/2/5 ms |
| Byudjet | 1600 gradient hisobi |

Qadamlar:

1. KPI generatorini yozing va NumPy'da mini-paket gradientli logistik regressiyani amalga oshiring; 1 ishchi baza yaqinlashuv egri chizig'ini (yo'qotish vs gradient soni) oling.
2. Sinxron SGD simulyatsiyasi: har qadamda 16 ishchi vaqti log-normal taqsimotdan olinadi, qadam vaqti = maksimum; devor vaqti va yo'qotish egri chizig'i (sigma=0.3 va 0.8).
3. Zaxira ishchilar: har qadamda eng tez 14 ta gradient o'rtachalanadi; devor vaqti va yaqinlashuvni solishtiring.
4. Asinxron SGD: har ishchi o'z tezligida tugatganda eskirgan parametr bo'yicha hisoblangan gradient qo'llanadi (eskirish tau ni yozib boring); yo'qotish va o'rtacha tau jadvali; LR ni tau ga qarab kamaytirish ta'sirini sinang.
5. Haqiqiy o'lchov: s1 DDP kodini KPI ma'lumotiga moslang, rank 1 ga 0/2/5 ms sleep qo'shib E(2) jadvalini tuzing va simulyatsiya modeli E = T_qadam/(T_qadam + d) bilan solishtiring.
6. Uch sxema uchun 'bir xil devor vaqtida erishilgan yo'qotish' grafigini chizing (sigma=0.8 holatida) va s6 Amdal modeliga straggler ulushini qo'shib N=64 uchun E(64) ni baholang.
7. Xulosa: operator uchun qaysi sxema (yoki kombinatsiya) tavsiya etiladi, asinxron rejimning narxi (eskirgan gradient, yaqinlashuv kafolati), amaliy vositalar (torch elastic, Horovod) haqida izoh.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadval va grafiklar — 25, Chuqur tahlil (analitik model vs o'lchov, cheklovlar) — 30, Hisobot sifati va tavsiya asosliligi — 20 (jami 100)

## 19-variant. Aviatsiya: dvigatel telemetriyasi — ierarxik all-reduce va alfa-beta kechikish modeli (qiyin)

Soha: Aviatsiya

Aviakompaniya dvigatel telemetriyasidan qolgan resursni bashorat qiluvchi 95 mln parametrli modelni 4 tugun x 8 GPU klasterida o'qitadi: tugun ichida NVLink (600 GB/s), tugunlar orasida 10 yoki 100 Gb/s. Oddiy halqa 32 ta GPU'ni bitta halqaga tizadi va sekin bog'lamlar hammani tutib qoladi. Sizdan alfa-beta (kechikish + o'tkazuvchanlik) aloqa modelini yozish, yassi halqa va ierarxik (tugun ichida reduce-scatter/all-gather, tugunlar orasida halqa) sxemalarni NumPy'da simulyatsiya qilib vaqtini taqqoslash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Model | 95 mln parametr (BF16 gradient = 190 MB), 14 GFLOP/namuna |
| Topologiya | 4 tugun x 8 GPU; tugun ichida 600 GB/s, alfa 5 mks; tugunlar orasida 10 va 100 Gb/s, alfa 20 mks |
| Aloqa modeli | T = alfa * qadamlar + baytlar / BW (har bosqich uchun) |
| Sxemalar | yassi halqa (N=32); ierarxik 2 bosqich (8 ichida, 4 orasida) |
| Chelak hajmi | 1, 25, 100 MB (qadamlar soniga ta'sir) |
| Simulyatsiya | L = 4096, 32 ishchi |
| GPU | A100, MFU 0.4, paket 64 |

Qadamlar:

1. s3 ring_allreduce asosida ierarxik all-reduce yozing: 1) har tugun ichida 8 ishchi reduce-scatter, 2) tugunlar orasida 4 ta bo'lak egasi halqa all-reduce, 3) tugun ichida all-gather; L=4096, 32 ishchi uchun natija to'g'riligini tekshiring va har ishchi yuborgan baytlarni bog'lam turi bo'yicha (ichki/tashqi) ajratib hisoblang.
2. Alfa-beta modelini amalga oshiring: har qadam vaqti = alfa + baytlar/BW; yassi halqa (31+31 qadam, eng sekin bog'lam tezligi bilan) va ierarxik sxema uchun 190 MB gradientda jami vaqtni 10 va 100 Gb/s uchun hisoblang.
3. Chelak hajmi 1, 25, 100 MB uchun all-reduce soni va alfa ulushini hisoblang; kichik chelakda kechikish (alfa) hukmron bo'lishini grafikda ko'rsating.
4. s7 kalkulyatoriga ierarxik T_comm ni qo'shing va N=32 uchun E(32) ni yassi va ierarxik sxemalarda, ikki tarmoqda solishtiring; overlap 0.5.
5. s1 DDP kodini telemetriya sintetik ma'lumotiga (6000 x 200, regressiya o'rniga 5 sinf) moslang va 2 jarayonda to'g'ri ishlashini ko'rsating; bucket_cap_mb=1 va 25 uchun vaqt farqini o'lchang (alfa ta'sirining CPU muqobili).
6. s6 Amdal bilan aloqa ulushidan N=32 va N=128 uchun chegarani baholang.
7. Xulosa: aviakompaniya klasteri uchun sxema, chelak hajmi va tarmoq tavsiyasi; NCCL ierarxik algoritmlarni qanday tanlashi haqida izoh.

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadval va grafiklar — 25, Chuqur tahlil (analitik model vs o'lchov, cheklovlar) — 30, Hisobot sifati va tavsiya asosliligi — 20 (jami 100)

## 20-variant. E-hukumat: murojaatlarni tasniflash — o'qitish va inferens uchun to'liq masshtablash hisoboti (qiyin)

Soha: Davlat xizmatlari

Davlat xizmatlari portali kuniga 2 mln fuqaro murojaatini 24 ta idoraga yo'naltiradi; modelni 2 GPU'li serverda o'qitish va 8 yadroli serverda inferens qilish rejalashtirilgan, byudjet qat'iy. Sizdan laboratoriyaning barcha bosqichlarini bitta ssenariyga birlashtirish: DDP o'lchovi, masshtablash jadvali, gradient to'plash, parallel inferens profili, Amdal bahosi va kalkulyator — va yakuniy texnik hisobot (o'qitish SLA 4 soat, inferens p99 < 6 ms) tayyorlash talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Ma'lumot | sintetik murojaat vektorlari 12000 x 768 (24 idora markazi + shovqin), 24 sinf |
| Model | MLP 768-512-256-24 (taxminan 532 ming parametr) |
| O'qitish | 2 jarayon DDP, BS 64/128, EPOCHS 3, K = 1 va 4 |
| Inferens | 2 mln so'rov, SHARD 256/1024/4096, 1 va 2 ishchi, SLO p99 < 6 ms |
| Tarmoq (kalkulyator) | 1, 10, 100 Gb/s |
| Real klaster bahosi | N = 2..16, ResNet-50 o'rniga shu model + BERT-base 110 mln |
| Hisobot | Markdown, 2 jadval + 3 grafik |

Qadamlar:

1. s1 DDP: murojaat ma'lumotida BS=64 va 128, EPOCHS=3 uchun namuna/s, yo'qotish, aniqlik, parametr sinxronligi jadvali.
2. s2 masshtablash: HID=512 (ikki yashirin qatlam), GB=256 uchun 4 konfiguratsiya, S(2), E(2); kuchli rejim S(2) dan Amdal p ni toping.
3. s4 + DDP no_sync: K=4 to'plash bilan vaqt va all-reduce soni; 1x256 bilan gradient ekvivalentligini tekshiring.
4. s5 inferens: 2 mln so'rovni SHARD=256, 1024, 4096 da 1 va 2 ishchida; so'rov/s, p50/p99 jadvali; SLO p99 < 6 ms ni qondiruvchi eng katta SHARD; 8 yadro uchun S(8) ni Amdal bilan bashorat qiling va kunlik 2 mln so'rov necha daqiqa olishini hisoblang.
5. s3 + s7: 532 ming va 110 mln parametr uchun 1/10/100 Gb/s da all-reduce vaqti, N=2..16 da E(N) egri chiziqlari (kalkulyator) va ring simulyatsiyasi tekshiruvi.
6. Barcha natijalarni Markdown hisobotga yig'ing: o'qitish SLA 4 soat (12000 yozuvdan 2 mln ga ekstrapolyatsiya qilib, epoxa vaqtini namuna/s dan hisoblang) bajariladimi, inferens SLO bajariladimi, qaysi konfiguratsiya tavsiya etiladi.
7. Xulosa: ikki alohida tavsiya (o'qitish va inferens), xavflar (straggler, tarmoq, xotira) va keyingi qadamlar (profiler bilan tekshirish).

Topshiriladi: Kod (.py fayllar, har qadam alohida); Natijalar jadvali (konfiguratsiya, vaqt, namuna/s, S(N), E(N)); Grafik(lar) (matplotlib, PNG); Xulosa (>= 150 so'z): o'lchovlar talqini va klaster uchun tavsiya

Baholash: To'g'ri ishlaydigan kod (barcha tekshiruvlar o'tadi) — 25, O'lchovlar, jadval va grafiklar — 25, Chuqur tahlil (analitik model vs o'lchov, cheklovlar) — 30, Hisobot sifati va tavsiya asosliligi — 20 (jami 100)
