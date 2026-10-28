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

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
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

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Tibbiyot: rentgen tasvirlarini triaj qilish xizmati — probe'lar, yuklama va kasalxona ichidagi Edge (o'rta)

Soha: Tibbiyot (radiologiya)

Viloyat kasalxonasi rentgen tasvirlarini «shoshilinch/odatiy» deb saralaydigan xizmatni ishga tushirmoqda; tasvirlar surrogati — digits 8x8 CNN. Ertalab 8:00-10:00 da yuklama 5 so'rov/s dan 60 so'rov/s ga ko'tariladi, qolgan vaqtda 5 so'rov/s. Bemor ma'lumotlari binodan chiqmasligi kerak, shuning uchun bulut varianti faqat anonimlashtirilgan tasvirlar bilan ko'rib chiqiladi. Siz FastAPI xizmatini yozib TestClient bilan sinaysiz, yuklama sinovi o'tkazasiz, Dockerfile va K8s manifestlarini tekshirasiz, HPA ni ertalabki profil bilan simulyatsiya qilasiz va kasalxona serveri (Edge) bilan bulutni TCO va maxfiylik bo'yicha solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | kichik CNN (A3 s1) -> ONNX, 2 sinf (shoshilinch = raqamlar 1,4,7) |
| Xizmat turi | tasvir tasnifi, /predict 64 piksel |
| SLO | p95 <= 100 ms, mavjudlik 99,9 % |
| Yuklama profili | 5 -> 60 -> 5 so'rov/s (ertalabki cho'qqi 2 soat, o'sish 10 daqiqa) |
| Xizmat vaqti (sim) | 25 ms |
| HPA | min 2, max 5, nishon 55 %, start 25 s |
| TCO | 3 yil, kasalxona serveri 2500 USD + 40 Vt vs bulut GPU 0,50 USD/soat, RTT 60 ms |

Qadamlar:

1. Laboratoriya s1 kabi xizmatni yozing: /healthz, /readyz, /predict (tasvir -> sinf + ishonch + timing), /metrics; readyz yuklanishdan oldin 503 ekanini TestClient bilan ko'rsating; 200 tasvirda aniqlik.
2. Laboratoriya s2 kabi yuklama sinovi: ketma-ket 300 so'rov (sof inferens ulushi), parallel 1/2/4/8/16 mijoz jadvali (so'rov/s, p50/p95/p99, little_N), SLO p95 <= 100 ms tekshiruvi.
3. Dockerfile tahlili (s3): o'z Dockerfile'ingizni yozing (slim + ORT, root'siz, HEALTHCHECK /readyz) va 11 qoidadan o'tkazing; tasvir hajmi bahosi va kasalxona 100 Mbit/s tarmog'ida tortish vaqti.
4. K8s manifestlari (s4): Deployment/ConfigMap/Service/HPA (min 2, max 5, nishon 55 %) 18 qoidadan 100 % o'tsin; HPA chegarasini millicore'da hisoblang.
5. HPA simulyatsiyasi (s5): yuklama(t) ni ertalabki profilga o'zgartiring (T = 3 soat, 10 daqiqada 5 -> 60), xizmat vaqti 25 ms; asosiy + 3 konfiguratsiya (nishon 40 %, min 3, start 60 s) uchun SLO buzilish vaqti va pod-soniya.
6. TCO (s6): kasalxona serveri (Edge) vs bulut vs gibrid (faqat anonim shubhali tasvirlar bulutga) 1/3/5 yil; kechikish (RTT 60 ms) va maxfiylik ustunlari; xulosa va tavsiya.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Dockerfile va K8s YAML (PASS natijasi bilan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 2-variant. Chakana savdo: mahsulot tavsiya xizmati — top-5, Qora juma yuklamasi va bulut tanlovi (o'rta)

Soha: Chakana savdo

Onlayn do'kon har sahifa ochilganda foydalanuvchiga 5 ta tavsiya ko'rsatadi; model — foydalanuvchi va mahsulot embeddinglari ustidagi MLP (A3 kabi, sintetik). Odatiy yuklama 150 so'rov/s, Qora juma kuni 20 daqiqa ichida 3000 so'rov/s ga chiqadi va 6 soat davom etadi. Har so'rov 200 nomzodni baholaydi, tavsiya 80 ms ichida kelmasa sahifa tavsiyasiz ochiladi (daromad yo'qoladi). Siz /recommend endpointini yozasiz, yuklama sinovida nomzodlar sonining kechikishga ta'sirini o'lchaysiz, manifestlarni tekshirasiz, HPA ni Qora juma profilida simulyatsiya qilasiz va faqat bulut variantlari (doimiy nusxa vs avtomasshtab vs oldindan rejalashtirilgan masshtab) TCO sini solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | Embedding(20000,16)+Embedding(5000,16)+MLP 32-64-1 -> ONNX |
| Xizmat turi | tavsiya, /recommend {user_id, item_ids[200]} -> top-5 |
| SLO | p95 <= 80 ms |
| Yuklama profili | 150 -> 3000 so'rov/s (20 daqiqada), 6 soat plato, keyin 150 |
| Xizmat vaqti (sim) | 6 ms (200 nomzod) |
| HPA | min 3, max 40, nishon 65 %, start 30 s |
| Bulut narxi | vCPU-soat 0,045 USD, nusxa 2 vCPU |

Qadamlar:

1. A3 dagi embedding+MLP modelini (sintetik) o'qitib ONNX'ga eksport qiling; FastAPI xizmati: /readyz, /recommend (200 nomzodni paketda baholab top-5 qaytaradi), /metrics (so'rovlar, p95, nomzodlar soni gistogrammasi).
2. Yuklama sinovi: nomzodlar soni 20/50/100/200/500 uchun ketma-ket p50/p95; keyin 200 nomzod bilan parallel 1/2/4/8/16 mijoz jadvali va SLO tekshiruvi.
3. Dockerfile (s3) 11 qoida va K8s manifest (s4) 18 qoida — o'z fayllaringiz 100 % PASS; HPA max 40 uchun klaster sig'imi (vCPU) hisobi.
4. HPA simulyatsiyasi (s5): Qora juma profili (T = 8 soat, 1 s qadam yoki 5 s qadam), xizmat vaqti 6 ms, min 3 / max 40; SLO buzilish vaqti, maksimal navbat, pod-soniya.
5. Uch siyosat taqqosi: (a) CPU-HPA, (b) oldindan rejalashtirilgan masshtab (t = -30 daqiqada 30 nusxaga), (c) doimiy 40 nusxa; har biri uchun SLO buzilgan soniyalar va vCPU-soat narxi.
6. TCO (s6) ni bulut ichida qayta yozing: 3 siyosat uchun yillik narx (oddiy kunlar + 5 ta aksiya kuni), yo'qotilgan tavsiyalar soni (SLO buzilgan vaqt x so'rov/s) va har yo'qotilgan tavsiya 0,02 USD deb daromad yo'qotishini hisoblang; xulosa.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Uch masshtablash siyosati taqqos jadvali

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 3-variant. Sanoat: vibratsiya anomaliyasini baholash xizmati — sex shlyuzida Edge va ARM tasviri (o'rta)

Soha: Sanoat

Zavodda 200 ta dvigatel sensorlari har soniyada bittadan 256 nuqtalik oyna yuboradi (jami 200 so'rov/s, doimiy). Autoenkoder (A3 kabi) rekonstruksiya xatosini hisoblab anomaliya ballini qaytaradi; qaror 500 ms ichida kerak, zavod tarmog'i internetga 10 Mbit/s kanal orqali ulangan va tez-tez uziladi. Siz /score xizmatini yozasiz, yuklamani sinaysiz, ARM shlyuzi uchun Dockerfile (python:3.12-slim arm64, ORT) tahlil qilasiz, K8s o'rniga bitta tugunli k3s manifestini tekshirasiz, HPA simulyatsiyasida sensorlar sonining o'sishini (200 -> 600) modellashtirasiz va Edge shlyuz vs bulut TCO ni oflayn ishlash talabi bilan solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | Autoenkoder 256-64-16-64-256 -> ONNX (anomaliya = MSE > chegara) |
| Xizmat turi | anomaliya balli, /score {signal[256]} -> {score, is_anomaly} |
| SLO | p95 <= 500 ms, oflayn ishlash majburiy |
| Yuklama profili | 200 so'rov/s doimiy; 2-yilda 600 |
| Xizmat vaqti (sim) | 4 ms |
| Shlyuz | ARM 4 yadro, 4 GB (Edge), 450 USD + 12 Vt |
| Bulut | 2 vCPU nusxa 0,09 USD/soat, egress 0,09 USD/GB, RTT 90 ms |
| Trafik | 256 x 4 bayt x 200/s |

Qadamlar:

1. Autoenkoderni sintetik vibratsiyada o'qiting (A3 4-variant usuli), chegarani 99-foizil qilib ONNX'ga eksport qiling; FastAPI /score, /readyz, /metrics (anomaliya ulushi gauge bilan).
2. Yuklama sinovi: parallel 1/2/4/8/16 mijoz; 200 so'rov/s talabini qaysi parallellikda bajarish mumkin; little_N; /score_batch (50 oyna bitta so'rovda) bilan solishtiring.
3. Dockerfile: arm64 uchun slim + onnxruntime (torch'siz) variant va «hamma narsa bilan» variant; 11 qoida va hajm bahosi; 10 Mbit/s kanalda tasvirni shlyuzga yetkazish vaqti.
4. k3s manifesti (bitta tugun): Deployment replicas 2 (ikki jarayon), Service, HPA max 3; 18 qoidadan o'ting, bitta tugunda replicas >= 2 nimani beradi va nimani bermaydi — izohlang.
5. HPA simulyatsiyasi: yuklama 200 -> 600 so'rov/s 1 yil davomida chiziqli o'sadi deb (simulyatsiyada 600 s ga siqilgan), xizmat vaqti 4 ms; shlyuz sig'imi qachon tugaydi, ikkinchi shlyuz qachon kerak.
6. TCO (s6): Edge shlyuz (1+1 zaxira) vs bulut vs gibrid (shlyuzda ball, bulutga faqat anomal oynalar) 1/3/5 yil; trafik GB/oy; oflayn uzilishda (yiliga 40 soat) bulut variantida nechta oyna yo'qoladi; xulosa.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Ikki Dockerfile taqqos jadvali

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 4-variant. Telekom: SMS spam filtri xizmati — doimiy 2000 so'rov/s, paketli endpoint va K8s manifest (oson)

Soha: Telekom

Mobil operator SMS markazidan o'tayotgan xabarlarni spam/oddiy deb belgilaydi: doimiy oqim 2000 xabar/s, kechikish talabi yumshoq (p95 300 ms), lekin xizmat to'xtamasligi kerak. Model — 1000 o'lchamli so'zlar sumkasi MLP (A3 15-variant kabi, sintetik). SMS markazi xabarlarni 100 talik paketlarda yuborishi mumkin. Siz /classify va /classify_batch endpointlarini yozasiz, yuklama sinovida yakka va paketli rejimni solishtirasiz, Dockerfile va K8s manifestlarini qoidalardan o'tkazasiz va paketli rejimda nechta nusxa kerakligini Little qonuni bilan hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 1000-256-2 -> ONNX |
| Xizmat turi | matn tasnifi, /classify {vector[1000]} va /classify_batch [100 ta] |
| SLO | p95 <= 300 ms, mavjudlik 99,95 % |
| Yuklama profili | 2000 xabar/s doimiy |
| Paket | 1 va 100 |
| K8s | replicas 3, HPA min 3 max 8, nishon 60 %, PDB minAvailable 2 |
| Nusxa | 2 vCPU, 1 GiB |

Qadamlar:

1. Sintetik so'zlar sumkasi ma'lumotida MLP'ni o'qitib ONNX'ga eksport qiling; FastAPI xizmati /readyz, /classify, /classify_batch, /metrics (xabarlar soni, spam ulushi).
2. Yuklama sinovi (s2): yakka so'rov parallel 1/2/4/8/16 mijoz; paketli (100 ta) so'rov parallel 1/2/4 mijoz; xabar/s ko'rsatkichini ikkala rejim uchun jadvalga yozing.
3. Little qonuni: 2000 xabar/s uchun yakka va paketli rejimda nusxalar soni (o'lchangan throughput/nusxa asosida); paketli rejimda 100 ta xabar yig'ilish kutishi (100/2000 s) kechikishga qo'shiladi — SLO ga sig'adimi.
4. Dockerfile (s3): o'z variantingiz 11 qoidadan o'tsin; tasvir hajmi va 8 tugunga tarqatish GB.
5. K8s manifest (s4): Deployment replicas 3, PDB (minAvailable 2) qo'shing va unga 2 ta yangi qoida yozing (jami 20); HPA chegarasi millicore'da.
6. Xulosa: operatorga paketli yoki yakka rejim, nechta nusxa, PDB nega kerak (tugun yangilanishida xizmat to'xtamasligi), GPU kerakmi.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); K8s YAML (PDB bilan) va 20 qoidali tekshiruv natijasi

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 5-variant. Energetika: aqlli hisoblagichlar uchun iste'mol prognozi — 15 daqiqalik burst, HPA va batch-job taqqosi (qiyin)

Soha: Energetika

Elektr tarmog'i operatori 40 000 ta aqlli hisoblagichdan har 15 daqiqada ko'rsatkich oladi va har biri uchun keyingi soat prognozini hisoblaydi; so'rovlar 15 daqiqalik oynaning birinchi 2 daqiqasida keladi (burst 330 so'rov/s), qolgan 13 daqiqa deyarli nol. Model — 24 soatlik oyna MLP regressori (A3 5-variant kabi). Siz /forecast xizmatini yozasiz, yuklama sinovi o'tkazasiz, HPA ni burst profilida simulyatsiya qilasiz (HPA bunday profilda yomon ishlaydi), muqobil sifatida har 15 daqiqada ishga tushadigan batch Job (K8s CronJob) rejimini modellashtirasiz va ikkala rejim uchun narx va kechikishni solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 30-128-64-1 -> ONNX (regressiya) |
| Xizmat turi | vaqt qatori prognozi, /forecast {window[24], features[6]} |
| SLO | barcha 40 000 prognoz oyna boshidan 5 daqiqa ichida tayyor |
| Yuklama profili | har 15 daqiqa: 2 daqiqa 330 so'rov/s, 13 daqiqa 2 so'rov/s |
| Xizmat vaqti (sim) | 8 ms |
| HPA | min 1, max 10, nishon 60 %, start 20 s, scaleDown 300 s |
| CronJob | 4 parallel pod, 2 daqiqa ishlaydi |
| Narx | vCPU-soat 0,045 USD |

Qadamlar:

1. MLP regressorni sintetik ma'lumotda o'qitib ONNX'ga eksport qiling; FastAPI /forecast, /forecast_batch (1000 hisoblagich), /readyz, /metrics; TestClient bilan MAE tekshiruvi.
2. Yuklama sinovi: yakka parallel 1/2/4/8/16; paketli 1000 talik so'rov; 40 000 prognoz uchun ikkala rejimda vaqt.
3. K8s manifestlari: Deployment + HPA (18 qoida) va qo'shimcha CronJob manifesti (schedule */15, parallelism 4, activeDeadlineSeconds 600) — CronJob uchun 4 ta yangi qoida yozing.
4. HPA simulyatsiyasi (s5): yuklama(t) ni 15 daqiqalik davriy burst qiling (T = 1 soat), xizmat vaqti 8 ms; SLO «5 daqiqada hammasi» ni navbat orqali tekshiring; scaleDown oynasi 300 s va 60 s ni solishtiring.
5. Batch rejimi simulyatsiyasi: har 15 daqiqada 4 pod 20 s da ishga tushib 40 000 prognozni o'lchangan throughput bilan bajaradi; tugash vaqti, pod-soniya.
6. Narx taqqosi: HPA (min 1 va min 4) vs CronJob vs doimiy 6 nusxa — kunlik vCPU-soat va USD; qaysi rejim SLO ni arzonroq bajaradi.
7. Xulosa: davriy burst yuklamada reaktiv HPA nega yomon, KEDA cron-trigger nima beradi, hisoblagichlar 400 000 bo'lsa nima o'zgaradi.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); HPA vs CronJob taqqos jadvali va grafigi

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 6-variant. Fintech: tranzaksiya firibgarligi skoring xizmati — qattiq p99, ikki mintaqa va kanareyka relizi (qiyin)

Soha: Fintech (to'lovlar)

To'lov tizimi har tranzaksiyani o'tkazishdan oldin firibgarlik ballini so'raydi: p99 20 ms dan oshsa tranzaksiya tekshiruvsiz o'tkaziladi (xavf), xizmat ikki mintaqada ishlaydi va yangi model versiyasi avval 10 % trafikka (kanareyka) chiqariladi. Yuklama 500 so'rov/s, oyning oxirgi kunida 5000 so'rov/s. Model — A3 18-variantdagi MLP. Siz xizmatga model versiyasi yorlig'ini va /metrics da versiya bo'yicha hisoblagichlarni qo'shasiz, yuklama sinovida p99 ni alohida kuzatasiz, ikki Deployment (stable/canary) manifestini tekshirasiz, HPA ni oy oxiri profilida simulyatsiya qilasiz va ikki mintaqali bulut TCO sini hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 28-128-64-1 -> ONNX, versiyalar v1 (FP32) va v2 (INT8 statik) |
| Xizmat turi | anomaliya/skoring, /score {features[28]} -> {score, decision, model_version} |
| SLO | p99 <= 20 ms, xatolar < 0,1 % |
| Yuklama profili | 500 so'rov/s, oy oxiri 5000 so'rov/s (30 daqiqada o'sish, 4 soat plato) |
| Xizmat vaqti (sim) | 3 ms |
| Kanareyka | stable 90 % / canary 10 % (Service selector + replicas nisbati 9:1) |
| HPA | min 4, max 30, nishon 50 %, start 20 s |
| Mintaqalar | 2, har birida to'liq nusxa, RTT 25 ms yaqin mintaqa |

Qadamlar:

1. Ikki ONNX versiya tayyorlang (FP32 va INT8); FastAPI xizmati MODEL_VERSION env bilan ishga tushadi, /score javobida versiya, /metrics da infer_requests_total{version=...} va p99.
2. Yuklama sinovi: parallel 1/2/4/8/16/32 mijoz, p99 ni alohida ustun qilib SLO 20 ms qaysi parallellikda buziladi; v1 va v2 ni solishtiring.
3. K8s: ikkita Deployment (score-stable replicas 9, score-canary replicas 1, umumiy label app=score + version label), bitta Service, HPA faqat stable'ga; 18 qoidaga kanareyka uchun 3 ta yangi qoida qo'shing (versiya label, canary replicas <= 10 %, canary'da alohida readiness).
4. Kanareyka tahlili: TestClient bilan 1000 so'rovni 9:1 nisbatda yo'naltirib (tasodifiy marshrutlash), versiya bo'yicha p99 va qaror ulushini /metrics dan oling; v2 qarorlari v1 dan qancha farq qiladi (tranzaksiya soni).
5. HPA simulyatsiyasi (s5): oy oxiri profili, xizmat vaqti 3 ms, min 4/max 30; SLO p99 ni «p95_ms» o'rniga navbat + 2x xizmat vaqti bilan baholang; min 4 va min 10 taqqosi.
6. TCO (s6) ikki mintaqa uchun: bulut nusxalar (oddiy kun va oy oxiri), mintaqalararo trafik, 3 yil; bitta mintaqa + zaxira bilan solishtiring; mintaqa uzilsa (yiliga 4 soat) yo'qotilgan tranzaksiyalar.
7. Xulosa: kanareyka qachon 100 % ga o'tkaziladi (mezon), p99 uchun HPA yetarlimi, GPU kerakmi (yo'q — asoslang).

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Kanareyka manifestlari va versiya bo'yicha metrikalar jadvali

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 7-variant. Agro: issiqxona sug'orish tavsiyasi — Raspberry Pi shlyuzi simulyatsiyasi va 5 yillik TCO (oson)

Soha: Agro

Agrofirma 50 ta issiqxonada har 5 daqiqada sensorlar (A3 2-variant: 20 belgi, 4 sinf tavsiya) bo'yicha sug'orish qarorini oladi; internet qishloqda beqaror (kuniga 1-2 soat uziladi), qaror kechiksa ekin zarar ko'radi. Siz /advise xizmatini yozasiz va uni ikki rejimda sinaysiz: «Edge» (1 oqim + har so'rovga 12 ms sun'iy ishlov — Raspberry Pi taqlidi) va «Cloud» (2 oqim + 100 ms RTT taqlidi). Yuklama kichik (jami 0,17 so'rov/s), shuning uchun masshtablash emas, ishonchlilik va narx asosiy. Dockerfile arm64 uchun tekshiriladi, HPA simulyatsiyasi o'rniga uzilishlar simulyatsiyasi, TCO 5 yil.

| Parametr | Qiymat |
|---|---|
| Model | MLP 20-256-128-4 -> ONNX |
| Xizmat turi | tavsiya, /advise {sensors[20]} -> {action, confidence} |
| SLO | qaror 5 daqiqa ichida, oflayn ishlash |
| Yuklama | 50 issiqxona x 1 so'rov / 5 daqiqa |
| Edge taqlidi | ORT 1 oqim + 12 ms sleep |
| Cloud taqlidi | ORT 2 oqim + 100 ms sleep (RTT) |
| Uzilishlar | kuniga 90 daqiqa, tasodifiy |
| TCO | Raspberry Pi 4 (90 USD + 5 Vt) x 50 + zaxira 5 vs bulut 1 nusxa 0,045 USD/soat + 4G modemlar |

Qadamlar:

1. MLP'ni o'qitib ONNX'ga eksport qiling; FastAPI /advise, /readyz, /metrics; EDGE_MODE env bilan 12 ms yoki 100 ms sun'iy kechikish qo'shiladi.
2. Ikkala rejimda ketma-ket 200 so'rov p50/p95 va parallel 1/2/4 mijoz; sof inferens ulushi; Little qonuni tekshiruvi.
3. Dockerfile (s3): python:3.12-slim arm64 + ORT varianti 11 qoida; tasvir hajmi va 4G (5 Mbit/s) orqali 50 qurilmaga yangilash vaqti va trafik.
4. Uzilishlar simulyatsiyasi (s5 o'rniga): 7 kunlik vaqt o'qi (1 daqiqa qadam), kuniga 90 daqiqa tasodifiy uzilish; Cloud rejimida uzilish paytidagi so'rovlar yo'qoladi yoki navbatga tushadi (10 daqiqagacha); Edge rejimida yo'qolmaydi; yo'qolgan/kechikkan qarorlar sonini hisoblang.
5. TCO (s6) 1/3/5 yil: Edge (50+5 Pi, elektr, yiliga 10 % almashtirish) vs Cloud (1 nusxa + 50 modem x 3 USD/oy) vs gibrid (Pi'da qaror, bulutda monitoring); sezgirlik: issiqxona 200 ta, uzilish kuniga 4 soat.
6. Xulosa: agrofirmaga tavsiya, uzilishlar narxini (zarar ko'rgan qarorlar) TCO ga qanday qo'shish mumkin, Jetson kerakmi (yo'q — asoslang).

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Uzilishlar simulyatsiyasi jadvali (yo'qolgan qarorlar)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 8-variant. Transport: avtobus bortida yo'lovchilarni sanash — Jetson taqlidi, 4G RTT va 300 avtobus TCO (o'rta)

Soha: Transport

Shahar avtobus parki 300 ta avtobusning eshik kamerasidan yo'lovchilar sonini hisoblaydi; kadr surrogati — digits CNN (sinf = yo'lovchilar soni 0..9). Har eshik ochilganda 2 soniya davomida 5 kadr/s, kuniga avtobus boshiga 400 to'xtash. Bulut varianti 4G orqali ishlaydi (RTT 120 ms, tunnel va chekka hududlarda uzilish), Edge varianti — bortdagi kichik kompyuter (Jetson Nano taqlidi: 1 oqim + 8 ms). Siz /count xizmatini yozasiz, ikkala rejimni sinaysiz, bulut uchun K8s manifest va HPA ni ertalabki/kechki cho'qqi profilida tekshirasiz va 300 avtobus uchun 3 yillik TCO va trafikni hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | kichik CNN (digits) -> ONNX, 10 sinf |
| Xizmat turi | tasvir, /count {pixels[64]} -> {count, confidence} |
| SLO | p95 <= 400 ms (bulut), oflayn ishlash istalgan |
| Yuklama profili | cho'qqi 07-09 va 17-19: 300 avtobus x 10 kadr / 30 s = 100 so'rov/s; kunduzi 30; tunda 2 |
| Xizmat vaqti (sim) | 5 ms |
| HPA | min 2, max 8, nishon 60 % |
| Edge | bort kompyuter 180 USD + 8 Vt, 1 oqim + 8 ms |
| Trafik | kadr 3 KB, 4G 1,5 USD/GB |

Qadamlar:

1. CNN'ni o'qitib ONNX'ga eksport qiling; FastAPI /count, /readyz, /metrics; EDGE_MODE (1 oqim + 8 ms) va CLOUD_MODE (2 oqim + 120 ms RTT taqlidi).
2. Yuklama sinovi ikkala rejimda: parallel 1/2/4/8/16; cloud rejimida RTT tufayli little_N qanday o'sadi; SLO 400 ms tekshiruvi.
3. K8s manifest (s4) bulut xizmati uchun 18 qoida; HPA chegarasi; Dockerfile (s3) 11 qoida.
4. HPA simulyatsiyasi (s5): 24 soatlik profil (tun 2, kunduz 30, cho'qqi 100 so'rov/s, 30 daqiqalik o'tish), 1 daqiqa qadam; SLO buzilish, pod-soniya; min 2 vs min 4.
5. Trafik va TCO (s6): bulut — kuniga 300 x 400 x 10 kadr x 3 KB egress + 4G narxi + nusxalar; Edge — 300 bort kompyuter + 10 % zaxira + quvvat; gibrid — bortda sanash, bulutga faqat soatlik yig'indi; 1/3/5 yil.
6. Xulosa: avtobus parkiga tavsiya, 4G trafik ulushi TCO da qancha, uzilishlar bo'lsa bulut variantida ma'lumot yo'qolishi, Jetson haqiqiy qurilmada nimani o'lchash kerak (tegrastats).

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 9-variant. Ta'lim: insholarni avtomatik baholash xizmati — imtihon kuni, sekin start va oldindan masshtablash (o'rta)

Soha: Ta'lim

Milliy test markazi imtihon kuni 2 soat ichida 50 000 talabaning insholarini (matn belgilari MLP, A3 15-variant kabi 1000 o'lchamli vektor) 5 ballik shkalada baholaydi; talabalar inshoni topshirgach 30 soniya ichida dastlabki ball ko'rsatilishi kerak. Yuklama imtihon boshlanishidan 90 daqiqa o'tgach keskin oshadi (ko'pchilik oxirgi 30 daqiqada topshiradi): 5 so'rov/s dan 40 so'rov/s gacha. Model tasviri katta (til resurslari bilan 1,2 GB), shuning uchun Pod start 90 s. Siz /grade xizmatini yozasiz, yuklama sinovi o'tkazasiz, HPA ni sekin start bilan simulyatsiya qilasiz va oldindan masshtablash (imtihondan 15 daqiqa oldin) hamda tasvirni kichraytirish (start 20 s) variantlarini solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 1000-512-128-5 -> ONNX |
| Xizmat turi | matn baholash, /grade {vector[1000]} -> {score 1..5, confidence} |
| SLO | p95 <= 30 s (navbat bilan), ball yo'qolmasligi |
| Yuklama profili | 0-90 daqiqa 5 so'rov/s, 90-120 daqiqa 40 so'rov/s |
| Xizmat vaqti (sim) | 120 ms (matn ishlovi bilan) |
| HPA | min 2, max 12, nishon 60 %, start 90 s (katta tasvir) yoki 20 s (slim) |
| Tasvir | 1,2 GB vs 280 MB |
| Narx | vCPU-soat 0,045 USD |

Qadamlar:

1. MLP'ni o'qitib ONNX'ga eksport qiling; FastAPI /grade, /readyz (model + «til resursi» yuklash taqlidi 3 s), /metrics; TestClient bilan readyz 503 -> 200 ketma-ketligi.
2. Yuklama sinovi: parallel 1/2/4/8/16 mijoz, har so'rovga 120 ms sun'iy ishlov qo'shilgan; Little qonuni bilan N; bitta nusxa sig'imi (so'rov/s).
3. Dockerfile (s3): ikki variant — til resurslari tasvir ichida (1,2 GB) va alohida volume'dan yuklanadi (280 MB); 11 qoida va tortish vaqti 1 Gbit/s da; start vaqtiga ta'siri.
4. K8s manifest (s4) 18 qoida; readinessProbe initialDelay/failureThreshold ni 90 s startga moslang va buni qoida sifatida tekshiring.
5. HPA simulyatsiyasi (s5): imtihon profili (T = 2 soat, 1 s qadam), xizmat vaqti 120 ms; 4 konfiguratsiya: start 90 s, start 20 s, oldindan masshtab (t = 75 daqiqada 10 nusxa), min 10; SLO 30 s buzilgan vaqt, maksimal navbat, pod-soniya.
6. Narx va xulosa: har konfiguratsiya uchun imtihon kuni vCPU-soat va USD; kechikkan ballar soni; test markaziga tavsiya (oldindan masshtab + slim tasvir), GPU kerakmi.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); 4 konfiguratsiyali HPA taqqos jadvali

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 10-variant. Xavfsizlik: turniketda yuz embeddinglarini solishtirish — oflayn Edge va 40 turniket (oson)

Soha: Xavfsizlik

Korxona 40 ta turniketda xodimlarni yuz embeddingi (128 o'lchamli vektor, sintetik make_blobs bilan 500 xodim) bo'yicha tekshiradi: /verify so'rovi embeddingni oladi, bazadagi eng yaqin xodimni kosinus o'xshashlik bilan topadi va chegara bo'yicha «ruxsat/rad» qaytaradi. Ertalab 08:00-09:00 da 40 turniketdan jami 20 so'rov/s, qolgan vaqt 1 so'rov/s; javob 300 ms ichida bo'lmasa navbat hosil bo'ladi; turniket internet uzilganda ham ishlashi shart. Siz xizmatni yozasiz, yuklamani sinaysiz, Edge (turniket kontrolleri) va Cloud rejimlarini solishtirasiz, K8s manifest va HPA ni ertalabki profil bilan tekshirasiz va TCO hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | kosinus o'xshashlik + chegara 0,8 (sklearn make_blobs: 500 markaz, 128 o'lcham, std 0,15), ONNX o'rniga numpy matritsa |
| Xizmat turi | tasvir/embedding, /verify {embedding[128]} -> {employee_id, similarity, allowed} |
| SLO | p95 <= 300 ms, oflayn ishlash majburiy |
| Yuklama profili | 08:00-09:00 20 so'rov/s, boshqa vaqt 1 so'rov/s |
| Xizmat vaqti (sim) | 2 ms (Edge 1 oqim + 20 ms ARM taqlidi) |
| HPA | min 1, max 3, nishon 50 % |
| Edge | turniket kontrolleri 120 USD + 4 Vt x 40 |
| Baza yangilanishi | kuniga 1 marta, 500 x 128 x 4 bayt |

Qadamlar:

1. 500 xodim embedding bazasini make_blobs bilan yarating, test so'rovlari = markaz + shovqin (xodim) va tasodifiy vektor (begona); FastAPI /verify, /readyz, /metrics (ruxsat/rad hisoblagichlari); TestClient bilan FAR/FRR ni chegara 0,7/0,8/0,9 da o'lchang.
2. Yuklama sinovi: parallel 1/2/4/8 mijoz Edge va Cloud taqlidida (Cloud: +80 ms RTT); SLO 300 ms; Little qonuni.
3. Dockerfile (s3) arm64 slim variant 11 qoida; baza faylini volume orqali ulash (ENV + ConfigMap) — 12 omil qoidasi.
4. K8s manifest (s4) bulut varianti uchun 18 qoida; HPA min 1 bo'lsa qaysi qoida buziladi va nega bu xizmat uchun qabul qilinadi/qilinmaydi.
5. HPA simulyatsiyasi (s5): 24 soat, 1 daqiqa qadam, ertalabki cho'qqi; Cloud rejimida SLO buzilish; Edge rejimida har turniket o'z yukini (0,5 so'rov/s) ko'taradi — masshtablash kerak emasligini ko'rsating.
6. TCO (s6) 1/3/5 yil: Edge 40 kontroller + bulut sinxronizatsiya vs Cloud + 40 yupqa mijoz; internet uzilishida (yiliga 30 soat) Cloud rejimida nechta xodim kira olmaydi; xulosa va maxfiylik ustuni (embeddinglar binodan chiqadimi).

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); FAR/FRR jadvali (3 chegara)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 11-variant. Logistika: ombor konveyerida paket nuqsonini aniqlash — 8 liniya, PDB va rolling update simulyatsiyasi (qiyin)

Soha: Logistika

Saralash ombori 8 ta konveyer liniyasida soniyasiga 20 tadan paketning tasvirini (digits surrogati, 2 sinf) tekshiradi — jami 160 so'rov/s, 24/7, qaror 50 ms ichida kerak, aks holda paket qayta aylanadi (liniya unumdorligi tushadi). Xizmat ombor ichidagi 3 tugunli k3s klasterida ishlaydi; model haftada bir yangilanadi va yangilanish paytida liniyalar to'xtamasligi shart. Siz xizmatni yozasiz, yuklamani sinaysiz, Deployment + PDB + HPA manifestlarini tekshirasiz, HPA simulyatsiyasiga rolling update (maxSurge 1 / maxUnavailable 0 vs maxUnavailable 1) va tugun nosozligini (3 tugundan biri 10 daqiqaga o'chadi) qo'shasiz, qayta aylangan paketlar sonini hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | kichik CNN (digits -> 2 sinf) -> ONNX |
| Xizmat turi | tasvir, /inspect {pixels[64]} -> {defect, confidence} |
| SLO | p95 <= 50 ms, qayta aylangan paketlar < 0,5 % |
| Yuklama profili | 160 so'rov/s doimiy |
| Xizmat vaqti (sim) | 6 ms |
| K8s | 3 tugun, replicas 4, HPA min 4 max 9, PDB minAvailable 3, podAntiAffinity |
| Rollout | maxSurge 1 / maxUnavailable 0 vs maxUnavailable 1, yangi Pod 25 s |
| Nosozlik | t=600 s da 1 tugun 10 daqiqaga o'chadi |

Qadamlar:

1. CNN'ni o'qitib ONNX'ga eksport qiling; FastAPI /inspect, /readyz, /metrics (nuqson ulushi, p95); TestClient bilan 200 paketda aniqlik.
2. Yuklama sinovi (s2): parallel 1/2/4/8/16; bitta nusxa sig'imi; Little qonuni bilan 160 so'rov/s uchun nusxalar soni p95 <= 50 ms sharti bilan.
3. K8s manifestlari (s4): Deployment (replicas 4, RollingUpdate), PDB, HPA, Service; podAntiAffinity (tugun bo'yicha tarqatish) va PDB uchun 4 ta yangi qoida — jami 22 qoida 100 % PASS; Dockerfile 11 qoida.
4. HPA simulyatsiyasi (s5) doimiy 160 so'rov/s da: asosiy holat SLO ichida ekanini tasdiqlang (bazaviy chiziq).
5. Rolling update simulyatsiyasi: t=300 s da yangilanish, nusxalar birma-bir almashadi (yangi Pod 25 s), ikki strategiya; sig'im, navbat, p95 grafigi; qayta aylangan paketlar (p95 > 50 ms bo'lgan soniyalar x 160) soni.
6. Tugun nosozligi simulyatsiyasi: t=600 s da 1 tugun o'chadi (uning 1-2 Pod'i yo'qoladi), HPA va PDB qanday javob beradi, qayta ko'tarilish vaqti; antiAffinity bo'lmasa (barcha Pod bitta tugunda) nima bo'lishini solishtiring.
7. Xulosa: omborga tavsiya (replicas, PDB, strategiya), haftalik yangilanish qaysi vaqtda, GPU/Jetson kerakmi.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Rolling update va tugun nosozligi grafiklari; 22 qoidali manifest tekshiruvi

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 12-variant. Shahar: chorraha svetofor kontrollerlari uchun tiqilinch bashorati — 400 chorraha, Edge kontroller vs markaz (oson)

Soha: Shahar

Shahar transport markazi 400 ta chorrahadan har 2 soniyada 6x6 trafik xaritasini oladi (A3 12-variant CNN) va tiqilinch darajasini qaytaradi; natija svetofor fazasini sozlaydi, shuning uchun 500 ms ichida kelishi kerak. Yuklama doimiy 200 so'rov/s, ertalab va kechqurun 1,5 baravar. Markaziy serverga ulanish optik tola, lekin ba'zi chorrahalar radiokanal (RTT 150 ms, paketlar yo'qoladi). Siz /congestion xizmatini yozasiz, yuklamani sinaysiz, Dockerfile va manifestlarni tekshirasiz, HPA ni kunlik profilda simulyatsiya qilasiz va «hamma chorrahada kontroller» vs «markazda K8s» vs «gibrid (radiokanalli 60 chorrahada kontroller)» TCO sini hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | CNN 6x6 -> 3 sinf -> ONNX |
| Xizmat turi | vaqt qatori/tasvir, /congestion {grid[36]} -> {level, probs} |
| SLO | p95 <= 500 ms |
| Yuklama profili | 200 so'rov/s, 07-09 va 17-19 da 300 |
| Xizmat vaqti (sim) | 3 ms |
| HPA | min 2, max 6, nishon 60 % |
| Edge kontroller | 140 USD + 6 Vt, 1 oqim |
| Radiokanal | 60 chorraha, RTT 150 ms, 2 % yo'qotish |

Qadamlar:

1. Sintetik 6x6 xaritalarda CNN'ni o'qitib ONNX'ga eksport qiling; FastAPI /congestion, /readyz, /metrics; TestClient aniqlik.
2. Yuklama sinovi (s2): parallel 1/2/4/8/16; 200 va 300 so'rov/s uchun nusxalar soni; radiokanal taqlidi (150 ms sleep + 2 % so'rovda 503) bilan p95 va xatolar ulushi.
3. Dockerfile (s3) 11 qoida; K8s manifest (s4) 18 qoida; HPA chegarasi millicore'da.
4. HPA simulyatsiyasi (s5): 24 soat, 1 daqiqa qadam, ikki cho'qqi; SLO buzilish, pod-soniya; nishon 60 % vs 40 %.
5. TCO (s6) 1/3/5 yil: 400 kontroller vs markaz (K8s, 3 tugun server 4000 USD har biri + elektr) vs gibrid (60 kontroller + markaz); radiokanalli chorrahalarda yo'qolgan qarorlar soni markaz variantida (2 % x so'rovlar).
6. Xulosa: shaharga tavsiya, radiokanal muammosini Edge qanday hal qiladi, 400 kontrollerni yangilash (OTA) xarajati TCO ga qanday qo'shiladi.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 13-variant. Ekologiya: havo sifati prognozi — juda kam yuklama, scale-to-zero va serverless TCO (o'rta)

Soha: Ekologiya

Ekologiya agentligi 120 ta havo monitoring stansiyasidan daqiqada bitta o'lchov oladi va har biri uchun 6 soatlik prognoz hisoblaydi — jami 2 so'rov/s, tunda 0,5 so'rov/s; natija veb-saytda ko'rsatiladi, 3 soniya kechikish qabul qilinadi. Doimiy ishlaydigan 2 nusxa bunday yuklama uchun isrof. Siz /aqi_forecast xizmatini yozasiz, yuklamani sinaysiz, K8s manifestida minReplicas 1 va scale-to-zero (KEDA/Knative taqlidi: 5 daqiqa so'rovsiz bo'lsa 0 nusxa, sovuq start 15 s) variantlarini simulyatsiya qilasiz va doimiy pod, scale-to-zero hamda serverless funksiya (so'rov boshiga narx) TCO sini solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 13-256-128-5 (AQI sinfi) + regressiya boshi -> ONNX |
| Xizmat turi | vaqt qatori prognozi, /aqi_forecast {features[13]} -> {aqi_class, values[6]} |
| SLO | p95 <= 3 s |
| Yuklama profili | kunduzi 2 so'rov/s, tunda 0,5; har 10 daqiqada 60 s so'rovsiz oraliq |
| Xizmat vaqti (sim) | 10 ms |
| Scale-to-zero | 5 daqiqa so'rovsiz -> 0 nusxa, sovuq start 15 s |
| Serverless narxi | 0,0000002 USD/so'rov + 0,0000167 USD/GB-s (256 MB) |
| Doimiy pod | 0,5 vCPU 0,0225 USD/soat |

Qadamlar:

1. MLP'ni o'qitib ONNX'ga eksport qiling; FastAPI /aqi_forecast, /readyz (sovuq start taqlidi: model yuklash 2 s), /metrics; TestClient bilan birinchi so'rov (sovuq) va keyingi (iliq) kechikishini alohida o'lchang.
2. Yuklama sinovi (s2): parallel 1/2/4 mijoz (yuklama kichik), bitta nusxa sig'imi; 2 so'rov/s uchun CPU bandlik foizi (requests 500m ga nisbatan).
3. K8s manifest (s4): minReplicas 1 varianti 18 qoida (replicas >= 2 qoidasi buziladi — asoslang); Dockerfile (s3) hajmi sovuq startga qanday ta'sir qiladi.
4. Simulyatsiya (s5 o'zgartirilgan): 24 soat, 1 s qadam, 3 siyosat — doimiy 1 nusxa, doimiy 2 nusxa, scale-to-zero (5 daqiqa bo'shliqdan keyin 0, sovuq start 15 s); SLO 3 s buzilgan so'rovlar soni, pod-soniya.
5. TCO (s6) yillik: doimiy 1 va 2 pod, scale-to-zero (pod-soniya bo'yicha), serverless (so'rov soni x narx + GB-s); stansiyalar 1200 ta bo'lsa kesishish nuqtasi qayerda.
6. Xulosa: agentlikka tavsiya, sovuq start SLO ni qancha buzadi, Edge (stansiyada hisoblash) nega bu yerda ma'nosiz.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); 3 siyosat taqqos jadvali (SLO, pod-soniya, USD)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 14-variant. Sport: fitnes ilovasi uchun harakat tasnifi — 1 mln foydalanuvchi, kunlik sinusoidal yuklama va on-device taqqosi (qiyin)

Soha: Sport

Fitnes ilovasi 1 million foydalanuvchining akselerometr oynalarini (A3 14-variant 1D CNN + GRU, 6 sinf) tasniflaydi; mashg'ulot paytida har 2 soniyada bitta so'rov. Bir vaqtda faol foydalanuvchilar kun davomida sinusoidal o'zgaradi: tunda 2000, ertalab 7:00 va kechqurun 19:00 da 60 000 (jami 30 000 so'rov/s cho'qqi). Muqobil — modelni telefonda ishlatish (on-device): server yuklamasi nolga tushadi, lekin batareya sarfi va yangilash murakkabligi paydo bo'ladi. Siz /activity xizmatini yozasiz, yuklamani sinaysiz, HPA ni sinusoidal profil bilan simulyatsiya qilasiz (nishon va oynalarni sozlab), bulut TCO ni hisoblaysiz va on-device variant bilan (telefon energiyasi A3 s6 modeli bilan) solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | Conv1d+GRU -> ONNX (64x3 -> 6 sinf) |
| Xizmat turi | vaqt qatori, /activity {window[64][3]} -> {activity, probs} |
| SLO | p95 <= 200 ms |
| Yuklama profili | sinusoidal: 1000 + 29000 x (1 + sin) / 2 so'rov/s, ikki cho'qqi (12 soat davr), 1 daqiqa qadam, 24 soat |
| Xizmat vaqti (sim) | 5 ms |
| HPA | min 10, max 300, nishon 50/70 %, scaleUp oynasi 0/60 s, scaleDown 300/900 s |
| Bulut narxi | 2 vCPU nusxa 0,09 USD/soat |
| On-device | telefon 0,3 Vt x inferens vaqti, 1 so'rov / 2 s, 1 soat mashg'ulot |

Qadamlar:

1. 1D CNN + GRU modelini sintetik akselerometrda o'qitib ONNX'ga eksport qiling; FastAPI /activity, /activity_batch, /readyz, /metrics; TestClient aniqlik.
2. Yuklama sinovi (s2): parallel 1/2/4/8/16/32; bitta nusxa sig'imi; Little qonuni bilan cho'qqi 30 000 so'rov/s uchun nusxalar soni; paketli so'rov (telefon 10 oynani birga yuborsa) sig'imni qancha oshiradi.
3. K8s manifest (s4) 18 qoida, maxReplicas 300 uchun klaster vCPU hisobi; Dockerfile (s3).
4. HPA simulyatsiyasi (s5): sinusoidal profil, 1 daqiqa qadam (HPA davri 15 s ni mos ravishda moslang), 6 konfiguratsiya (nishon 50/70 %, scaleUp 0/60, scaleDown 300/900); SLO buzilish daqiqalari, maksimal nusxa, pod-soat; HPA sinusoidani qanchalik kechikib kuzatadi (fazaviy siljish daqiqada).
5. Bulut TCO yillik: eng yaxshi HPA konfiguratsiyasi vs doimiy 300 nusxa vs prognozli masshtab (ideal — sig'im = yuklama / 0,7); grafik.
6. On-device variant: A3 s6 energiya modeli bilan telefon uchun uJ/inferens (P_bosh 0,1 Vt, P_to'liq 0,3 Vt), 1 soatlik mashg'ulotda mAh (3,85 V, 4000 mAh batareya) va foiz; model yangilash trafigi (1 mln x hajm); server narxi 0 — lekin qaysi xarajatlar qoladi.
7. Xulosa: ilova uchun gibrid taklif (on-device tasnif + serverda yig'ma tahlil), HPA sozlamalari, GPU bu xizmatga kerakmi.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); 6 konfiguratsiyali HPA jadvali; Bulut vs on-device taqqos jadvali

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 15-variant. Media: yangiliklar sarlavhasini teglash xizmati — viral portlash, navbat chegarasi va 503 (o'rta)

Soha: Media

Yangiliklar agregatori yangi sarlavhalarni 8 ta mavzu tegiga ajratadi (matn vektori MLP, sintetik). Odatiy yuklama 50 so'rov/s, viral voqea paytida 5 daqiqa ichida 30 baravar (1500 so'rov/s) oshadi va 20 daqiqa davom etadi. Teg kechiksa yangilik lentaga «tegsiz» tushadi — bu qabul qilinadi, lekin xizmat butunlay qulab qolishi mumkin emas. Siz /tag xizmatini yozasiz, yuklamani sinaysiz, HPA simulyatsiyasiga navbat chegarasini (ortiqcha so'rovlar 503 bilan rad etiladi) qo'shasiz, scaleUp oynasi va nishonni sozlab rad etilganlar sonini kamaytirasiz va mijoz tomonida qayta urinish (retry) strategiyasining ta'sirini TestClient bilan o'lchaysiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 1000-256-8 -> ONNX |
| Xizmat turi | matn tasnifi, /tag {vector[1000]} -> {tags top-2, probs} |
| SLO | p95 <= 500 ms, rad etilganlar < 5 % (viral vaqtda) |
| Yuklama profili | 50 -> 1500 so'rov/s (5 daqiqada), 20 daqiqa, keyin 50 |
| Xizmat vaqti (sim) | 7 ms |
| Navbat chegarasi | har nusxada 200 so'rov, ortig'i 503 |
| HPA | min 2, max 25, nishon 60 %, scaleUp oynasi 0/30/60 s |
| Retry | mijoz 503 da 1 marta 200 ms dan keyin qayta uradi |

Qadamlar:

1. MLP'ni o'qitib ONNX'ga eksport qiling; FastAPI /tag, /readyz, /metrics (rad etilganlar hisoblagichi); xizmatga navbat chegarasi qo'shing (bir vaqtdagi so'rovlar semafori, ortig'i 503).
2. Yuklama sinovi (s2): parallel 1/2/4/8/16/32 mijoz; semafor 4 bo'lganda 503 ulushi; mijozda retry (1 marta, 200 ms) bilan va retry'siz p95 va muvaffaqiyatli so'rovlar ulushi.
3. K8s manifest (s4) 18 qoida; Dockerfile (s3); HPA chegarasi.
4. HPA simulyatsiyasi (s5) navbat chegarasi bilan: har soniyada navbat > 200 x nusxa bo'lsa ortiqcha so'rovlar rad etiladi; viral profil; scaleUp oynasi 0/30/60 s va nishon 40/60 % uchun rad etilganlar soni, SLO buzilish, pod-soniya.
5. Retry ta'siri simulyatsiyada: rad etilgan so'rovlarning 100 % i 1 soniyadan keyin qaytib keladi — yuklama qanday o'sadi (retry bo'roni), rad etilganlar soni qanday o'zgaradi; eksponensial kechiktirish (1, 2, 4 s) bilan solishtiring.
6. Xulosa: agregatorga tavsiya (navbat chegarasi, HPA sozlamalari, retry siyosati), rad etish nega qulashdan yaxshi, prognozli masshtablash mumkinmi (viral oldindan ma'lum emas).

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Rad etilganlar va retry bo'roni grafiklari

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 16-variant. Tibbiyot: palata EKG monitoringi — 200 bemor, hayot uchun kritik probe'lar, PDB va bedside Edge (qiyin)

Soha: Tibbiyot (reanimatsiya)

Kasalxona 200 ta palata monitoridan EKG oqimini (A3 16-variant 1D CNN, 3 sinf) har soniyada bir segment sifatida tahlil qiladi — 200 so'rov/s, 24/7; fibrillyatsiya aniqlanganda 2 soniya ichida signal berilishi shart, xizmat 1 daqiqadan ortiq to'xtasa hayot xavfi. Siz xizmatni yozasiz (readiness/liveness farqi, graceful shutdown — preStop paytida yangi so'rovlar qabul qilinmaydi, lekin joriylar tugaydi), yuklamani sinaysiz, Deployment + PDB + PriorityClass manifestlarini qoidalardan o'tkazasiz, HPA simulyatsiyasiga Pod nosozliklarini (tasodifiy o'lim, 2 daqiqa qayta tiklanish) va rolling update'ni qo'shasiz, bedside Edge (har monitor o'zi hisoblaydi) bilan markaziy xizmatni ishonchlilik va TCO bo'yicha solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | 1D CNN (250 nuqta -> 3 sinf) -> ONNX |
| Xizmat turi | vaqt qatori/anomaliya, /ecg {segment[250]} -> {class, alarm, confidence} |
| SLO | signal <= 2 s, mavjudlik 99,99 %, yo'qolgan segment < 0,01 % |
| Yuklama profili | 200 so'rov/s doimiy |
| Xizmat vaqti (sim) | 4 ms |
| K8s | replicas 4, PDB minAvailable 3, PriorityClass yuqori, terminationGracePeriodSeconds 30, HPA min 4 max 8 |
| Nosozlik | har 30 daqiqada tasodifiy 1 Pod o'ladi, 60 s da qayta tiklanadi |
| Edge | monitor ichidagi modul 60 USD + 2 Vt x 200 |

Qadamlar:

1. 1D CNN'ni sintetik EKG'da o'qitib ONNX'ga eksport qiling; FastAPI /ecg, /healthz, /readyz, /metrics (alarm hisoblagichi); graceful shutdown: /admin/drain readyz'ni 503 qiladi, joriy so'rovlar tugaydi — TestClient bilan ketma-ketlikni ko'rsating.
2. Yuklama sinovi (s2): parallel 1/2/4/8/16; 200 so'rov/s uchun nusxalar soni p95 va alarm kechikishi 2 s sharti bilan.
3. K8s manifestlari (s4): Deployment (preStop sleep 5, terminationGracePeriodSeconds 30), PDB, PriorityClass, HPA, Service; 5 ta yangi qoida (PDB, PriorityClass, preStop, grace period, antiAffinity) — jami 23 qoida 100 % PASS.
4. Dockerfile (s3) 11 qoida; tasvir hajmi va qayta tiklanish vaqtiga ta'siri.
5. Simulyatsiya (s5 kengaytirilgan): 2 soat, 1 s qadam, doimiy 200 so'rov/s; tasodifiy Pod o'limi har 30 daqiqada (60 s tiklanish); replicas 2/3/4 va PDB bor/yo'q (yangilanish paytida) uchun yo'qolgan segmentlar soni va eng uzun uzilish; rolling update t=3600 s.
6. TCO va ishonchlilik (s6): markaziy xizmat (3 tugun + zaxira) vs bedside Edge (200 modul) vs gibrid (bedside aniqlash + markazda arxiv va ikkinchi fikr); mavjudlik hisobi (ketma-ket/parallel komponentlar), 1/3/5 yil.
7. Xulosa: kasalxonaga tavsiya, nima uchun hayot uchun kritik xizmatda Edge + markaz dublikati kerak, qaysi K8s mexanizmlari mavjudlikni qanchaga oshiradi.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); 23 qoidali manifest tekshiruvi; Nosozlik simulyatsiyasi jadvali (yo'qolgan segmentlar)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 17-variant. Bank: chat-bot niyat tasnifi xizmati — ish vaqti profili, versiya bo'yicha metrikalar va kanareyka 10 % (o'rta)

Soha: Bank xizmatlari

Bank mijozlarga xizmat ko'rsatish chat-boti har xabarni 20 ta niyatdan biriga ajratadi (matn vektori MLP, sintetik); yuklama 9:00-18:00 da 120 so'rov/s, tushlikda 180, tunda 10. Javob 150 ms ichida bo'lmasa bot «bir oz kuting» deydi va mijoz qoniqishi tushadi. Yangi model versiyasi har 2 haftada chiqadi va avval 10 % trafikda sinaladi; versiyalar bo'yicha aniqlik va kechikish alohida kuzatilishi kerak. Siz xizmatga versiya yorlig'ini qo'shasiz, yuklamani sinaysiz, stable/canary manifestlarini tekshirasiz, HPA ni ish kuni profilida simulyatsiya qilasiz va «ish vaqti bo'yicha rejalashtirilgan masshtab» (CronHPA taqlidi) bilan solishtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 1000-256-20 -> ONNX, v1 va v2 (v2 = boshqa seed bilan o'qitilgan) |
| Xizmat turi | matn tasnifi, /intent {vector[1000]} -> {intent, confidence, model_version} |
| SLO | p95 <= 150 ms |
| Yuklama profili | tun 10, 9:00-18:00 120, 13:00-14:00 180 so'rov/s; o'tishlar 15 daqiqa |
| Xizmat vaqti (sim) | 6 ms |
| Kanareyka | stable 9 nusxa / canary 1 nusxa |
| HPA | min 2, max 12, nishon 60 % |
| CronHPA | 8:45 da min 6, 18:15 da min 2 |

Qadamlar:

1. Ikki versiya MLP'ni o'qitib (seed 0 va 1) ONNX'ga eksport qiling; FastAPI MODEL_VERSION env; /intent, /readyz, /metrics da versiya yorlig'i bilan so'rovlar, xatolar, p95; TestClient bilan har versiya aniqligi.
2. Yuklama sinovi (s2): parallel 1/2/4/8/16; SLO 150 ms; bitta nusxa sig'imi; tushlik 180 so'rov/s uchun nusxalar soni.
3. K8s manifestlari (s4): intent-stable (9) va intent-canary (1), umumiy Service, HPA stable'ga; 18 + 3 kanareyka qoidasi; Dockerfile (s3).
4. Kanareyka tahlili: TestClient bilan 2000 so'rovni 9:1 marshrutlab versiya bo'yicha aniqlik, p95 va «ishonch < 0,5» ulushini /metrics dan oling; v2 ni 100 % ga chiqarish mezonini (aniqlik >= v1 - 0,5 p.p., p95 <= v1 x 1,1) tekshiring.
5. HPA simulyatsiyasi (s5): 24 soat, 1 daqiqa qadam, ish kuni profili; CPU-HPA vs CronHPA (min 6 ish vaqtida) vs ikkalasi birga; SLO buzilish daqiqalari, pod-soat.
6. Xulosa: bankka tavsiya (CronHPA + HPA), kanareyka natijasi, hafta oxiri profili uchun nima o'zgaradi, yillik vCPU-soat narxi (0,045 USD).

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Versiya bo'yicha metrikalar jadvali

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 18-variant. Konchilik: konveyer tasmasi anomaliyasi — sun'iy yo'ldosh kanali, majburiy Edge va gibrid sinxronizatsiya (oson)

Soha: Konchilik

Uzoq hududdagi kon konveyer tasmasining 40 ta sensor tugunidan (vibratsiya + harorat, A3 8-variant 1D CNN 4 sinf) har 2 soniyada ma'lumot keladi — 20 so'rov/s. Kon internetga faqat sun'iy yo'ldosh orqali ulangan: 2 Mbit/s, RTT 600 ms, oyiga 20 GB trafik limiti, 1 GB ortig'i 8 USD. Nosozlik aniqlansa konveyer 1 soniyada to'xtatilishi kerak — bulut varianti RTT tufayli imkonsiz. Siz /belt xizmatini yozasiz, Edge (kon serveri) va bulut (RTT taqlidi) rejimlarini sinaysiz, trafikni hisoblaysiz, bitta tugunli k3s manifestini tekshirasiz va gibrid variantda bulutga nima yuborish mumkinligini (soatlik yig'indi, anomal segmentlar) trafik limiti ichida hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | 1D CNN (256 nuqta -> 4 sinf) -> ONNX |
| Xizmat turi | anomaliya, /belt {signal[256], temp} -> {class, stop} |
| SLO | to'xtatish qarori <= 1 s, oflayn ishlash majburiy |
| Yuklama profili | 20 so'rov/s doimiy |
| Xizmat vaqti (sim) | 5 ms |
| Kanal | 2 Mbit/s, RTT 600 ms, 20 GB/oy limit, 8 USD/GB ortig'i |
| Segment hajmi | 256 x 4 bayt + sarlavha ~ 1,1 KB |
| Edge server | sanoat kompyuteri 1200 USD + 35 Vt, zaxira 1 |

Qadamlar:

1. 1D CNN'ni o'qitib ONNX'ga eksport qiling; FastAPI /belt, /readyz, /metrics (to'xtatish hisoblagichi); TestClient aniqlik va recall (nosozlik sinflari).
2. Yuklama sinovi (s2): Edge rejimi parallel 1/2/4/8; bulut taqlidi (600 ms sleep) bilan p95 — SLO 1 s qaysi parallellikda buziladi; Little qonuni bulut rejimida (little_N katta — nega).
3. Trafik hisobi: barcha segmentlar bulutga yuborilsa oyiga GB va ortiqcha to'lov; gibrid — faqat anomal segmentlar (2 %) + soatlik yig'indi (1 KB) — oyiga GB; model yangilash (hajm) trafigi.
4. Dockerfile (s3) 11 qoida; k3s manifesti (s4) bitta tugun, replicas 2, hostPath model volume, 18 qoida.
5. Simulyatsiya (s5 o'rniga): 7 kun, 1 daqiqa qadam, sun'iy yo'ldosh kanali kuniga 2 soat uziladi (ob-havo); bulut rejimida qancha so'rov javobsiz qoladi va necha marta konveyer «xavfsizlik rejimida» to'xtaydi; Edge rejimida 0.
6. TCO (s6) 1/3/5 yil: Edge (1+1 server) vs bulut (nusxa + trafik ortig'i) vs gibrid; xulosa: konga tavsiya va bulutga nima uchun faqat arxiv/qayta o'qitish ma'lumotlari ketadi.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Trafik hisobi jadvali (to'liq / gibrid)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 19-variant. Oziq-ovqat: restoranlar tarmog'ida talab prognozi — 300 filial, kuniga 3 burst va 300 ta Edge tasvirini tarqatish (oson)

Soha: Oziq-ovqat

Tez ovqatlanish tarmog'i 300 filialda har kuni 3 marta (06:00, 11:00, 16:00) keyingi 4 soat uchun taomlar talabini prognoz qiladi (vaqt qatori MLP, sintetik: hafta kuni, soat, ob-havo, aksiya). Har burst 5 daqiqa davom etadi: 300 filial x 40 taom = 12 000 so'rov, ya'ni 40 so'rov/s; qolgan vaqt 0,2 so'rov/s. Tarmoq xizmatni markaziy bulutda yoki har filialdagi kassa kompyuterida (Edge) ishlatishi mumkin. Siz /demand xizmatini yozasiz, yuklamani sinaysiz, Dockerfile tahlilida slim va to'liq variantlarni 300 filialga tarqatish hajmi va vaqti bo'yicha solishtirasiz, K8s manifestini tekshirasiz, HPA ni 3 burst profilida simulyatsiya qilasiz va TCO hisoblaysiz.

| Parametr | Qiymat |
|---|---|
| Model | MLP 12-128-64-1 (regressiya) -> ONNX |
| Xizmat turi | vaqt qatori prognozi, /demand {features[12]} -> {qty}; /demand_batch (40 taom) |
| SLO | burst 5 daqiqada tugashi, p95 <= 1 s |
| Yuklama profili | 3 x kuniga 40 so'rov/s (5 daqiqa), boshqa vaqt 0,2 |
| Xizmat vaqti (sim) | 5 ms |
| HPA | min 1, max 4, nishon 60 %, start 20 s |
| Tasvir | slim+ORT (~270 MB) vs pytorch (~8 GB) |
| Filial kanali | 20 Mbit/s |

Qadamlar:

1. Sintetik talab ma'lumotida MLP'ni o'qitib ONNX'ga eksport qiling; FastAPI /demand, /demand_batch, /readyz, /metrics; TestClient MAE.
2. Yuklama sinovi (s2): yakka parallel 1/2/4/8; paketli (40 taom) so'rov; 12 000 prognozni ikkala rejimda bajarish vaqti.
3. Dockerfile (s3): slim va to'liq variantlar 11 qoida; 300 filialga tarqatish: jami GB, 20 Mbit/s kanalda har filial uchun tortish vaqti, registr trafigi; haftalik model yangilanishi bilan yillik trafik.
4. K8s manifest (s4) bulut varianti 18 qoida (minReplicas 1 ni asoslang yoki 2 qiling); Dockerfile HEALTHCHECK /readyz.
5. HPA simulyatsiyasi (s5): 24 soat, 1 daqiqa qadam (HPA davri 15 s ni moslang), 3 burst; min 1 vs min 2 vs CronHPA (burstdan 3 daqiqa oldin 3 nusxa); SLO buzilish, pod-soat.
6. TCO (s6) 1/3/5 yil: bulut (HPA) vs Edge (kassa kompyuteri allaqachon bor — faqat dastur + yangilash xarajati) vs gibrid; xulosa: tarmoqqa tavsiya va 300 filialga tasvir tarqatish nega slim bo'lishi shart.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Tasvir tarqatish jadvali (300 filial)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)

## 20-variant. Davlat xizmatlari: hujjat raqamlarini tanish — kuniga 1 mln hujjat, onlayn xizmat vs batch job TCO (oson)

Soha: Davlat boshqaruvi

Davlat arxivi skanerlangan hujjatlardagi raqamli maydonlarni (digits CNN, 10 sinf) tanib bazaga yozadi: kuniga 1 million hujjat, har birida 12 ta raqam — 12 mln inferens; bundan tashqari, fuqarolar portalidan kuniga 20 000 ta onlayn so'rov (javob 2 s ichida) keladi. Siz /ocr_digit va /ocr_batch (bir hujjat 12 raqam) endpointlarini yozasiz, yuklamani sinab bitta nusxa sig'imini o'lchaysiz, kunlik arxivni onlayn xizmat orqali (doimiy nusxalar) yoki tungi batch Job (K8s Job, parallelism N) orqali qayta ishlash vaqtini va narxini hisoblaysiz, manifestlarni tekshirasiz va Edge (skaner stansiyasida tanish) variantini TCO ga qo'shasiz.

| Parametr | Qiymat |
|---|---|
| Model | kichik CNN (digits) -> ONNX, 10 sinf |
| Xizmat turi | tasvir, /ocr_digit {pixels[64]} va /ocr_batch [12 ta] |
| SLO | onlayn p95 <= 2 s; arxiv 1 mln hujjat 8 soatlik tun oynasida |
| Yuklama profili | onlayn 0,25 so'rov/s (kunduzi 0,6), arxiv 12 mln inferens/kun |
| Xizmat vaqti (sim) | 2 ms / raqam |
| K8s Job | parallelism 2/4/8, har pod 2 vCPU |
| Narx | vCPU-soat 0,045 USD |
| Skaner stansiyalari | 20 ta, har birida kompyuter bor |

Qadamlar:

1. CNN'ni o'qitib ONNX'ga eksport qiling; FastAPI /ocr_digit, /ocr_batch (12 raqam, paket inferens), /readyz, /metrics; TestClient bilan 200 hujjatda raqam bo'yicha va hujjat bo'yicha (12 tasi ham to'g'ri) aniqlik.
2. Yuklama sinovi (s2): yakka va paketli so'rovlar parallel 1/2/4/8/16; bitta nusxa sig'imi (hujjat/s); 1 mln hujjat uchun 1 nusxada soatlar.
3. K8s manifestlari (s4): onlayn Deployment + HPA (18 qoida) va alohida Job manifesti (parallelism, completions, backoffLimit, activeDeadlineSeconds) — Job uchun 4 ta qoida; Dockerfile (s3).
4. Batch hisobi: parallelism 2/4/8 uchun tungi oynada tugash vaqti (o'lchangan throughput asosida) va vCPU-soat; 8 soat oynasiga sig'adigan minimal parallelism.
5. HPA simulyatsiyasi (s5): arxiv oqimi onlayn xizmatga kunduzi 10 soat davomida tekis yuborilsa (12 mln / 36 000 s = 333 inferens/s) va onlayn so'rovlar bilan aralashsa — onlayn p95 SLO buziladimi; ajratilgan Job bilan solishtiring.
6. TCO (s6) yillik: doimiy nusxalar (kunduzi arxiv + onlayn) vs onlayn kichik + tungi Job vs Edge (20 skaner stansiyasida tanish, bulutga faqat natija); xulosa: arxivga tavsiya va onlayn/batch ajratish nega to'g'ri.

Topshiriladi: Kod (.py: xizmat + yuklama sinovi + tekshiruvchi skriptlar); Natijalar jadvali (yuklama, HPA, TCO); Grafik (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Batch Job hisobi jadvali (parallelism 2/4/8)

Baholash: To'g'ri ishlaydigan kod — 30, O'lchovlar va jadval — 25, Tahlil va xulosa — 25, Hisobot sifati — 20 (jami 100)
