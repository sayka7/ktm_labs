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

## Qo'shimcha topshiriqlar (10 ta, jami 100 ball)

| № | Topshiriq | Daraja | Ball |
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
