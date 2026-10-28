# A4. Edge AI va Cloud AI platformalarida sun'iy intellekt tizimlarini joylashtirish

Bu virtual laboratoriyada siz inferens modelini xizmat sifatida joylashtirish zanjirini jarayon ichida bajarasiz: FastAPI xizmati (/healthz, /readyz, /predict, /metrics) `TestClient` bilan sinaladi va yuklama ostida o'lchanadi, Dockerfile va Kubernetes manifestlari (Deployment/Service/HPA/ConfigMap) `pyyaml` bilan tahlil qilinib muhandislik qoidalariga tekshiriladi, HPA avtomasshtablash navbat modeli bilan simulyatsiya qilinadi, Cloud/Edge/Gibrid arxitekturalar TCO va kechikish kalkulyatori bilan solishtiriladi. Real labdan farqi: Docker/minikube/Jetson/Coral yo'q — konteyner va klaster o'rniga jarayon ichidagi xizmat va simulyatsiya; model — A3 dagi digits MLP (ONNX).

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/A4

## Bosqichlar

### 1-qadam. FastAPI inferens xizmati: probe'lar, predict, metrikalar

Kichik MLP ONNX'ga eksport qilinadi va FastAPI xizmatiga o'raladi: `lifespan` ichida model yuklanadi va isitiladi, `/healthz` (liveness) va `/readyz` (readiness) ajratilgan, `/predict` JSON so'rovni qabul qilib top-3 va vaqt taqsimotini (preprocess/inference) qaytaradi, `/metrics` Prometheus matn formatida so'rovlar, xatolar, kechikish kvantillarini beradi. TestClient model yuklanmasdan oldin (`/readyz` → 503) va keyin (`with` blok) chaqiriladi.

Fayl: `s1_1_qadam_fastapi_inferens_xizmati_probela.py` — ishga tushirish: `python3 s1_1_qadam_fastapi_inferens_xizmati_probela.py`

Kutilgan natija: Yuklanmasdan oldin /readyz 503, /healthz 200; keyin 200; 5 namunadan 4–5 tasi to'g'ri, 200 namunada ~90 % aniqlik; noto'g'ri so'rovlar 422; /metrics matni.

Tajriba qilib ko'ring:

- `/predict` ga `batch` maydonli variant qo'shing (bir so'rovda N ta rasm).
- `ORT_THREADS` ni `os.environ` orqali 1 ga qo'ying — timing qanday o'zgaradi?
- `/metrics` ga xotira (`psutil.Process().memory_info().rss`) gauge'ini qo'shing.

### 2-qadam. Yuklama sinovi: ketma-ket va parallel so'rovlar, Little qonuni, paketlash

`hey -c N` o'rniga `ThreadPoolExecutor` bilan 1, 2, 4, 8, 16 parallel mijoz simulyatsiya qilinadi: throughput, p50/p95/p99 va Little qonuni (N = X·R) tekshiriladi; to'liq kechikish ichidagi sof inferens ulushi ko'rsatiladi; 64 ta alohida so'rov va bitta paketli so'rov solishtiriladi; p95 ≤ 50 ms SLO tekshiriladi. Grafik: throughput va kechikish.

Fayl: `s2_2_qadam_yuklama_sinovi_ketma_ket_va_para.py` — ishga tushirish: `python3 s2_2_qadam_yuklama_sinovi_ketma_ket_va_para.py`

Kutilgan natija: c=1: p50 ~2–3 ms, ~400 so'rov/s (sof inferens 0,1 ms — qolgani HTTP/JSON); throughput 2 yadroda to'yinadi, p99 mijozlar bilan chiziqli o'sadi; paket 20–30x tezroq.

Tajriba qilib ko'ring:

- SLO ni 10 ms ga tushiring — nechta mijozda buziladi?
- `so.intra_op_num_threads = 2` qiling va parallel natijalarni solishtiring (oqimlar raqobati).
- Har so'rovga 5 ms sun'iy ishlov (`time.sleep`) qo'shib ResNet-50 xizmatini taqlid qiling va Little qonunini qayta tekshiring.

### 3-qadam. Dockerfile tahlili va tasvir hajmini baholash

Docker bo'lmagani uchun Dockerfile matni dasturiy tahlil qilinadi: ko'p bosqichli qurish, qotirilgan versiya, yengil asos, root bo'lmagan USER, pip/apt kesh, HEALTHCHECK, EXPOSE, exec CMD, qatlamlar tartibi, ENV sozlamalar — 11 qoida. Yaxshi (slim + ONNX Runtime) va yomon (pytorch:latest, root) variantlar solishtiriladi; tasvir hajmi, tortish vaqti va 20 tugunli klasterga tarqatish hajmi baholanadi.

Fayl: `s3_3_qadam_dockerfile_tahlili_va_tasvir_haj.py` — ishga tushirish: `python3 s3_3_qadam_dockerfile_tahlili_va_tasvir_haj.py`

Kutilgan natija: Yaxshi variant 11/11, yomon 1/11; hajm bahosi ~270 MB vs ~8 GB (30x); 100 Mbit/s da tortish 22 s vs 11 daqiqa.

Tajriba qilib ko'ring:

- O'zingizning Dockerfile'ingizni `YAXSHI` o'rniga qo'yib tekshiring.
- Yangi qoida qo'shing: `COPY . .` taqiqlangan (.dockerignore'siz sirlar kiradi).
- GPU varianti uchun `nvidia/cuda:*-devel` asosini qo'shib hajm farqini ko'ring (devel ~2x runtime).

### 4-qadam. Kubernetes manifestlari: pyyaml bilan validatsiya va 18 qoida

Deployment/ConfigMap/Service/HPA YAML'i `yaml.safe_load_all` bilan o'qilib 18 qoida bo'yicha tekshiriladi: replicas ≥ 2, selector/label mosligi, requests/limits, readiness va liveness farqi va vaqtlari, runAsNonRoot, RollingUpdate maxUnavailable 0, Service port/selector mosligi, HPA min/max/behavior/target, Prometheus annotatsiyalari. Tipik xatolar bilan yozilgan ikkinchi manifest 2/18 oladi. HPA chegarasi millicore'da hisoblanadi.

Fayl: `s4_4_qadam_kubernetes_manifestlari_pyyaml_b.py` — ishga tushirish: `python3 s4_4_qadam_kubernetes_manifestlari_pyyaml_b.py`

Kutilgan natija: To'g'ri manifest 18/18 PASS, xatoli 2/18; requests 1500m × 60 % = 900m chegara; 2 nusxa 95 % da → 4 nusxa.

Tajriba qilib ko'ring:

- GPU varianti uchun `limits: {nvidia.com/gpu: 1}` qo'shing va «GPU Pod'da HPA CPU bo'yicha ishlamaydi» qoidasini yozing.
- PodDisruptionBudget va Ingress ob'ektlarini qo'shib, ularga 2 tadan qoida yozing.
- `kubectl apply --dry-run=client` nima tekshiradi, sizning skriptingiz nima tekshiradi — farqni 3 jumlada yozing.

### 5-qadam. HPA avtomasshtablash simulyatsiyasi (navbat modeli)

Minikube o'rniga diskret-vaqtli simulyatsiya: so'rov oqimi λ(t) (40 → 340 → 40 so'rov/s), har nusxa sig'imi μ = 1000/14 ms, CPU foizi = λ/(R·μ), HPA har 15 s da `ceil(R·cpu/nishon)` hisoblaydi, scaleUp/scaleDown stabilizatsiya oynalari va Pod start vaqti (20 s) hisobga olinadi, navbat va p95 kechikish Little qonuni bilan baholanadi. 7 ta konfiguratsiya (nishon 40/60/80 %, minReplicas 4, sekin start 60 s, maxReplicas 3, oynasiz) SLO buzilish vaqti va pod-soniya narxi bo'yicha solishtiriladi.

Fayl: `s5_5_qadam_hpa_avtomasshtablash_simulyatsiy.py` — ishga tushirish: `python3 s5_5_qadam_hpa_avtomasshtablash_simulyatsiy.py`

Kutilgan natija: Asosiy konfiguratsiyada reaksiya ~65 s, SLO ~200 s buziladi; minReplicas 4 SLO buzilishini 4x kamaytiradi, narx +23 %; maxReplicas 3 — eng yomon.

Tajriba qilib ko'ring:

- `yuklama(t)` ni sinusoidal (kunlik sikl) qiling — HPA qanday kuzatadi?
- So'rov/s bo'yicha masshtablash (KEDA) qo'shing: istalgan = ceil(λ/(μ·0,6)) — CPU bilan solishtiring.
- Navbat chegarasi qo'shing (masalan 500): ortiqcha so'rovlar 503 bilan rad etilsin va rad etilganlar sonini hisoblang.

### 6-qadam. Cloud vs Edge vs Gibrid: kechikish, trafik, energiya va TCO kalkulyatori

A1 SHV ssenariysi (16 kamera, 160 inferens/s, 3 yil) uchun uch arxitektura hisoblanadi: uchdan-uchga kechikish (RTT + navbat + inferens), tarmoq trafigi, mJ/inferens (PUE bilan), TCO (GPU soati, egress, qo'shimcha xizmatlar; Jetson narxi, energiya, texnik xizmat, zaxira). TCO 1/2/3/5 yil, sezgirlik tahlili (kameralar x2/x8, bulut narxi −30 %, hodisa bo'yicha oqim, xizmat narxi, RTT) va kesishish nuqtasi. Jetson/TPU yo'q — bu simulyatsiya, raqamlar o'zgartiriladigan parametrlar.

Fayl: `s6_6_qadam_cloud_vs_edge_vs_gibrid_kechikis.py` — ishga tushirish: `python3 s6_6_qadam_cloud_vs_edge_vs_gibrid_kechikis.py`

Kutilgan natija: Cloud TCO Edge'dan 30–50 marta qimmat, kechikish ~6 marta katta; bulut narxi −30 % ham nisbatni saqlaydi; grafik: TCO yillar bo'yicha va sezgirlik ustunlari.

Tajriba qilib ko'ring:

- Yuklama notekis (kuniga 2 soat cho'qqi) ssenariysini qo'shing: bulutda GPU faqat ishlaganda to'lansin — kim yutadi?
- Muhandis vaqti (OTA, flot boshqaruvi) xarajatini Edge'ga 1000 qurilma uchun qo'shing.
- Ma'lumotlar markazi PUE 1,1 va 1,5 bilan energiya ustunini qayta hisoblang.

## Qo'shimcha topshiriqlar (10 ta, jami 100 ball)

| № | Topshiriq | Daraja | Ball |
|---|---|---|---|
| 1 | **Probe'lar ssenariysi** — 1-qadam xizmatiga `/readyz` ni vaqtincha 503 ga o'tkazuvchi (model qayta yuklanmoqda) `/admin/reload` endpointi qo'shing. TestClient bilan ketma-ketlikni ko'rsating: ready → reload → 503 → ready. Har holatda K8s nima qilishini (trafik, qayta ishga tushirish) 1 jumlada izohlang. | oson | 5 |
| 2 | **Dockerfile qoidalarini kengaytirish** — 3-qadam tahlilchisiga 3 ta yangi qoida qo'shing (masalan: `COPY . .` taqiqi, `ADD` o'rniga `COPY`, `WORKDIR` mavjudligi, `pip` versiyasi qotirilgan `==`). Ikkala Dockerfile'ni qayta baholang va qoidalar jadvalini topshiring. | oson | 5 |
| 3 | **TCO parametrlarini yangilash** — 6-qadam kalkulyatoridagi GPU soati, egress va Jetson narxini bugungi ochiq narxlar bilan (manba va sana ko'rsatib) almashtiring, 3 yillik TCO jadvalini qayta hisoblang va nisbat qanday o'zgarganini 2–3 jumlada tushuntiring. | oson | 5 |
| 4 | **Kechikish byudjeti dekompozitsiyasi** — 2-qadamda to'liq so'rov kechikishini qismlarga ajrating: JSON parse, pydantic validatsiya, oldindan ishlov, inferens, javob serializatsiyasi (har qismga `perf_counter` qo'ying). c=1 va c=8 uchun qismlar ulushini ustunli diagramma bilan ko'rsating va eng katta qismni kamaytirish usulini (masalan, `orjson`, raw bytes kirish) sinab natija bering. | o'rta | 10 |
| 5 | **Manifest validatori kengaytmasi** — 4-qadam tekshiruvchisiga PodDisruptionBudget, Ingress (TLS) va NetworkPolicy ob'ektlarini qo'shing, ularga kamida 6 ta yangi qoida yozing (masalan, PDB minAvailable ≥ 1, Ingress TLS secret mavjud, NetworkPolicy faqat 8000-portga ruxsat). Jami 24+ qoidali hisobot va «to'g'ri» manifestni 100 % PASS holatiga keltiring. | o'rta | 10 |
| 6 | **KEDA uslubida so'rov/s bo'yicha masshtablash** — 5-qadam simulyatsiyasiga ikkinchi masshtablash siyosati qo'shing: istalgan = ceil(λ/(μ·0,6)) (so'rov/s bo'yicha, CPU emas). CPU-HPA bilan bir xil yuklama profilida SLO buzilish vaqti, maksimal navbat va pod-soniya narxini solishtiring; GPU xizmatlari uchun qaysi biri to'g'ri ekanini asoslang. | o'rta | 10 |
| 7 | **Rolling update simulyatsiyasi** — 5-qadam modeliga yangilanish (rollout) qo'shing: t=400 s da `maxSurge=1, maxUnavailable=0` bilan nusxalar birma-bir almashtiriladi (yangi Pod 20 s da tayyor, keyin eski o'chadi). `maxUnavailable=1` varianti bilan solishtiring: yangilanish davomida sig'im, navbat va SLO buzilishi grafigi; GPU'lar soni cheklangan (2 GPU) holatda qaysi strategiya ishlashini tushuntiring. | o'rta | 10 |
| 8 | **Chekka qurilma simulyatori va gibrid mezon** — 1-qadam xizmatining «Edge» nusxasini yarating: ORT 1 oqim, har so'rovga ARM CPU oldindan ishlovini taqlid qiluvchi 15 ms kechikish, 2 Vt quvvat modeli. «Cloud» nusxasi: 2 oqim + 80 ms RTT. Gibrid marshrutlash qoidasini yozing (ishonch < chegara bo'lsa bulutga yuboriladi) va chegarani 0,5–0,99 oralig'ida o'zgartirib, bulutga ketgan so'rovlar ulushi, o'rtacha kechikish, umumiy aniqlik va trafikni grafikda ko'rsating; optimal chegarani asoslang. | qiyin | 15 |
| 9 | **Xizmatga nosozlik kiritish (chaos) sinovi** — 2-qadam yuklama sinoviga nosozlik ssenariylari qo'shing: (a) 10 % so'rovlarda sun'iy 500 ms kechikish, (b) 5 % so'rovlarda 500 xato, (c) t=5 s da model qayta yuklash (readyz 503, 2 s). Har ssenariy uchun p50/p95/p99, xatolar ulushi va `/metrics` dagi ko'rsatkichlarni jadvalga yig'ing; mijoz tomonida timeout + 1 marta qayta urinish (retry) strategiyasini qo'shib natijani solishtiring. 4–5 betlik hisobot. | qiyin | 15 |
| 10 | **To'liq joylashtirish hisoboti o'z ssenariyingiz uchun** — A1 da tanlagan tizimingiz uchun 1–6 qadamlarni to'liq takrorlang: o'z ONNX modelingiz bilan xizmat (probe'lar, metrikalar), yuklama sinovi (kamida 5 parallellik darajasi), o'z Dockerfile va K8s manifestlaringizni skriptlar bilan 100 % PASS holatiga keltirish, o'z yuklama profilingiz bilan HPA simulyatsiyasi (kamida 3 konfiguratsiya), Cloud/Edge/Gibrid TCO 1/3/5 yil va sezgirlik tahlili. 6–8 betlik hisobot: yakuniy arxitektura tavsiyasi va A1 strukturaviy sxemasining yangilangan varianti bilan. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.
