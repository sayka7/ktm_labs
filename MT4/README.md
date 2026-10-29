# MT4. Docker va Kubernetes platformalaridan foydalanib sun'iy intellekt modelini joylashtirish (Deployment) va AI xizmatlarini boshqarish bo'yicha amaliy loyiha

🦉 Bosqichlar Suqrot usulida: har `sN_*.py` — karkas (TODO larni o'zingiz yozasiz), savollar shu faylda, ishoralar `ISHORALAR.md` da.
Bu mustaqil ishda siz bitta AI xizmatini (tumor-api — o'sma belgilarini tasniflovchi model) boshidan oxirigacha joylashtirish loyihasini bajarasiz: FastAPI xizmati va avtomatik testlar, qatlamli konfiguratsiya va sog'liq tekshiruvlari, Dockerfile va docker-compose generatsiyasi va validatsiyasi, dev/prod Kubernetes manifestlari (Deployment, Service, HPA, ConfigMap, Secret, Ingress, PDB, NetworkPolicy), rolling update va canary, HPA va Cluster Autoscaler yuklama simulyatsiyasi hamda SLO asosidagi operatsion hisobot. Sandbox'da Docker va Kubernetes yo'q: xizmat jarayon ichida TestClient bilan ishlaydi, fayllar matn sifatida yaratilib Python'da tekshiriladi, klaster xulq-atvori esa simulyatsiya qilinadi. Vaqt, narx va yuklama raqamlari taxminiy.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, onnx, onnxruntime, fairlearn, shap, numba, psutil, pyyaml, fastapi, httpx.

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/MT4

## Bosqichlar

### 1-qadam. AI xizmati: FastAPI, model reestri, API kalit, probe'lar va avtomatik testlar

Loyiha — «tumor-api»: sklearn breast_cancer (30 belgi) bo'yicha o'sma turini tasniflovchi xizmat. Ikki model versiyasi (v1 — logistik regressiya, v2 — tasodifiy o'rmon) o'qitilib, SHA-256 xeshi bilan reestrga (/tmp/models) yoziladi. Xizmat `lifespan` ichida modelni butunlik tekshiruvi va isitish bilan yuklaydi; /startupz, /livez, /readyz alohida, /v1/predict va /v1/predict/batch API kalit (x-api-key) talab qiladi, har javobga x-request-id va x-model-version sarlavhalari qo'shiladi, /metrics Prometheus matn formatida. Oxirida CI'dagi kabi 11 ta avtomatik test TestClient bilan bajariladi. MODEL_VERSION, MAX_BATCH ni o'zgartirib ko'ring.

Fayl: `s1_1_qadam_ai_xizmati_fastapi_model_reestri.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s1_1_qadam_ai_xizmati_fastapi_model_reestri.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Model reestriga faqat model faylini yozish yetarlimi? Xizmat yuklash paytida fayl buzilmagan va aynan kerakli versiya ekanini qanday bilib oladi?
2. Nega model har so'rovda emas, `lifespan` ichida bir marta yuklanadi? Yuklagandan keyin bitta «bo'sh» bashorat (isitish) nima uchun kerak?
3. /startupz, /livez va /readyz probe'lari Kubernetes uchun nimasi bilan farq qiladi? Qaysi biri 503 qaytarishi mumkin va qachon?
4. Har javobga x-request-id ni qanday qo'shasiz: mijoz yuborgan ID bo'lsa nima qilasiz, bo'lmasa-chi? Bu qayerda — har endpoint ichidami yoki bitta joyda?
5. 29 ta belgi yuborilganda 422 qaytishi uchun tekshiruvni qayerga qo'yasiz? API kalitsiz so'rov 401, katta paket esa 413 bo'lishi uchun-chi?
6. Prometheus matn formatida counter va gauge qanday yoziladi? Testda predictions_total nega aynan 133 bo'lishi kerak?
7. Yakunda nimani kutasiz: v1 aniqligi ~0,96, v2 ~0,94, testlar 11/11 PASS va ~300–400 so'rov/s? FAIL chiqqan test qaysi TODO ga ishora qiladi?

Kutilgan natija: v1 aniqligi ~0,96, v2 ~0,94; 11/11 PASS; ketma-ket ~300-400 so'rov/s; /metrics da predictions_total 133 va model_info.

Tajriba qilib ko'ring:

- MODEL_VERSION ni os.environ orqali 'v2' qiling — /version va x-model-version sarlavhasi o'zgarishini tekshiring.
- Model faylini yozib bo'lgach bir baytini o'zgartiring: xizmat ishga tushmasligi (butunlik tekshiruvi) kerak.
- /metrics ga model bo'yicha so'rovlar sonini (label bilan) qo'shing.

### 2-qadam. Konfiguratsiya qatlamlari, chuqur readiness, probe'lar va to'g'ri to'xtash

(a) Sozlamalar uch qatlamdan yig'iladi: standart < ConfigMap (YAML) < muhit o'zgaruvchilari (TUMOR_*); har kalit sxema bo'yicha tur va diapazonga tekshiriladi, sir maskalanadi, xato bo'lsa xizmat ishga tushmaydi (fail fast); konfiguratsiya xeshi Deployment annotatsiyasi uchun hisoblanadi. (b) /readyz model fayli, uning xeshi, disk, xotira, o'z-o'zini sinash va drain holatini tekshiradi; /livez esa bog'liqliklarni tekshirmaydi. (c) Kubelet probe'lari simulyatsiyasi: model 45 s yuklanganda noto'g'ri liveness yoki qisqa startupProbe CrashLoopBackOff'ga olib kelishi ko'rsatiladi. (d) preStop va terminationGracePeriod ning yo'qolgan so'rovlarga ta'siri.

Fayl: `s2_2_qadam_konfiguratsiya_qatlamlari_chuqur.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s2_2_qadam_konfiguratsiya_qatlamlari_chuqur.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Bitta kalit uch joyda (standart, ConfigMap, muhit o'zgaruvchisi) berilgan bo'lsa, qaysi biri g'olib bo'lishi kerak va nega aynan shu tartib?
2. Muhit o'zgaruvchilari doim satr bo'lib keladi. "128" ni int ga o'tkazish va "ikki" ni xato deb topish uchun SXEMA dan qanday foydalanasiz?
3. Nega xato konfiguratsiyada xizmat umuman ishga tushmasligi (fail fast) yaxshiroq? Konfiguratsiya xeshini Deployment annotatsiyasiga qo'yish nima beradi?
4. Readiness va liveness probe'lari nimani tekshirishi kerak? Model fayli buzilganda /livez ham 503 qaytarsa nima bo'ladi?
5. Model 45 s yuklansa, liveness 10 s kechikish bilan 5 s periodda 3 marta tekshirilsa nima bo'ladi? startupProbe 2 s × 15 bilan-chi?
6. preStop = 0 bo'lsa, nega Pod to'xtatilganda so'rovlar yo'qoladi? Endpoint yangilanishi 3 s kechiksa va 100 so'rov/s kelsa, taxminan nechta?

Kutilgan natija: To'g'ri konfiguratsiyada OK va config-hash, xatoda 3 ta xato; buzilgan faylda readyz 503, livez 200; 2 ta XATO konfiguratsiyada 6 restart va hech qachon tayyor emas, to'g'rilarida 45 s; preStop=0 da ~300 so'rov yo'qoladi.

Tajriba qilib ko'ring:

- Sxemaga 'redis_url' (redis:// bilan boshlanishi shart) kalitini qo'shing va xato holatni sinang.
- Model yuklanishini 90 s qiling — startupProbe parametrlarini qanday tanlaysiz?
- Endpoint yangilanish kechikishini 3 s dan 8 s ga oshiring: preStop qancha bo'lishi kerak?

### 3-qadam. Dockerfile va docker-compose: generatsiya, validatsiya va qatlam keshi

Loyiha parametrlaridan ko'p bosqichli (multi-stage) Dockerfile generatsiya qilinadi: builder bosqichida wheel'lar, yakuniy obrazda faqat runtime, root bo'lmagan foydalanuvchi, HEALTHCHECK va exec-shakldagi CMD. 13 qoidali tahlilchi uni ataylab yomon yozilgan Dockerfile bilan solishtiradi. Qatlam keshi xeshlar zanjiri bilan simulyatsiya qilinadi: faqat kod o'zgarganda nechta qatlam qayta qurilishi va yomon tartibda pip ham qayta ishlashi ko'rsatiladi. So'ng API + Redis + Prometheus uchun docker-compose.yml pyyaml bilan yaratiladi va tekshiriladi: obraz teglari, port to'qnashuvi, depends_on + healthcheck, hajm va tarmoqlar e'loni, sirlar, ishga tushish tartibi (topologik saralash) va sikl.

Fayl: `s3_3_qadam_dockerfile_va_docker_compose_gen.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s3_3_qadam_dockerfile_va_docker_compose_gen.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Nega Dockerfile ikki bosqichli (builder va yakuniy) qilinadi? Yakuniy obrazga builder'dan nima o'tadi va nima o'tmaydi?
2. Xavfsizlik nuqtai nazaridan yakuniy obrazda qaysi uch narsa bo'lishi shart: kim nomidan ishlaydi, sirlar qayerda, sog'liq qanday tekshiriladi?
3. 13 ta qoidani matndan tekshirish uchun Dockerfile ni qanday bo'laklarga ajratasiz? Masalan, «ENV da sir yo'q» ni qanday regex bilan topasiz?
4. Docker qatlam keshi qanday ishlaydi: qachon qatlam qayta quriladi? Nega requirements.txt ni koddan oldin alohida COPY qilish kerak?
5. Faqat app/main.py o'zgarganda yaxshi Dockerfile da nechta qatlam qayta qurilishini kutasiz? requirements o'zgarganda-chi? Yomon Dockerfile da pip nega qayta ishlaydi?
6. Compose faylda xizmatlar ishga tushish tartibi depends_on dan qanday chiqariladi va sikl bo'lsa buni qanday aniqlaysiz?
7. Buzilgan compose'da (prometheus 8000 porti, redis healthcheck'siz, API_KEY environment'da, redis → prometheus bog'liqligi) qaysi 4 xato chiqishi kerak?

Kutilgan natija: Yaxshi Dockerfile 13/13, yomoni 0/13; kod o'zgarganda 1 qatlam (~3 s), requirements o'zgarganda 5 qatlam (~200 s, taxminiy); compose OK, tartib redis -> api -> prometheus; buzilgan compose'da 4 xato.

Tajriba qilib ko'ring:

- Dockerfile'ga .dockerignore generatsiyasini va uni tekshiruvchi qoidani qo'shing (.git, __pycache__, *.ipynb).
- Compose'ga GPU ishlatadigan 'trainer' xizmatini (deploy.resources.reservations.devices) qo'shing va validatorni kengaytiring.
- Yomon Dockerfile'ni bosqichma-bosqich tuzatib, har tuzatishdan keyin ballni yozib boring.

### 4-qadam. Kubernetes manifestlari: generator, dev/prod overlay va 18 qoidali tekshiruv

Python generatori bitta parametrlar lug'atidan 9 ta obyekt yaratadi: Namespace, ConfigMap, Secret, Deployment (startup/readiness/liveness probe, preStop, securityContext, topologySpread, checksum/config annotatsiyasi, NAS'dagi PVC), Service, HPA (autoscaling/v2), PodDisruptionBudget, Ingress (TLS) va NetworkPolicy. Kustomize'ga o'xshash chuqur birlashtirish bilan dev va prod overlay'lari olinadi, natija multi-hujjat YAML'ga yozilib qayta o'qiladi. Tekshiruvchi obyekt qoidalari bilan birga obyektlararo moslikni ham ko'radi: Service selector, targetPort, Ingress backend, HPA target, PDB, configMapKeyRef kalitlari. Ataylab buzilgan manifestda tipik xatolar ushlanadi.

Fayl: `s4_4_qadam_kubernetes_manifestlari_generato.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s4_4_qadam_kubernetes_manifestlari_generato.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. dev va prod muhitlari bir-biridan nimasi bilan farq qiladi va nega har biri uchun alohida YAML yozmasdan, bitta asos + overlay ishlatiladi?
2. Deployment da Pod shabloni label'lari, selector va Service selector qanday bog'langan? Qaysi biri mos kelmasa trafik Pod'larga yetmaydi?
3. ConfigMap o'zgarganda Pod'lar avtomatik qayta yaratilishi uchun nima qilinadi? Konteynerga sozlama va sir qanday yo'llar bilan beriladi?
4. 1 nusxali dev muhitida PodDisruptionBudget nega zararli? Uni yaratish shartini qanday yozasiz va prod da minAvailable qancha bo'lishi kerak?
5. Obyektlararo tekshiruvlarni qulay yozish uchun hujjatlarni qanday indekslaysiz? Ingress backend porti Service portiga mosligini qanday tekshirasiz?
6. CPU "500m" va "1", xotira "512Mi" va "1Gi" ni qanday solishtirasiz? Buzilgan manifestda requests.cpu = "2" qaysi qoidani yiqitadi?
7. Natijani tekshiring: dev va prod 18/18 PASS, buzilgan manifest 11/18 chiqdimi? YAML'ga yozib qayta o'qish nega foydali?

Kutilgan natija: dev 8 obyekt (1 nusxada PDB yo'q), prod 9 obyekt; dev va prod 18/18 PASS, buzilgan 11/18; Ingress YAML ko'rsatiladi.

Tajriba qilib ko'ring:

- 'staging' overlay qo'shing (replicas 2, boshqa host) va uchala muhitni bitta jadvalda solishtiring.
- HPA'ga ikkinchi metrika (Pods turidagi http_requests_per_second) qo'shing va validatorga uning tekshiruvini yozing.
- Generatsiya qilingan YAML'ni faylga yozib, real klasterda `kubectl apply --dry-run=server -f` bilan tekshiring (agar klaster bo'lsa).

### 5-qadam. Versiyani chiqarish: RollingUpdate simulyatsiyasi va haqiqiy so'rovlar bilan canary

(a) Deployment kontrolleri 1 s qadamli modelda simulyatsiya qilinadi: 6 nusxa, har biri 50 so'rov/s, talab 260 so'rov/s, yangi Pod 30 s da tayyor. maxSurge/maxUnavailable kombinatsiyalari, blue/green va Recreate uchun yangilanish davomiyligi, minimal sig'im, yo'qolgan so'rovlar va maksimal Pod soni solishtiriladi. (b) Ikki FastAPI versiyasi TestClient orqali ishga tushadi va vaznli marshrutlovchi (Ingress/Service mesh o'rnida) trafikni 5 -> 25 -> 50 -> 100 % bosqichlarda bo'ladi. Har bosqichda kamida 60 canary namunasi yig'ilib, xato ulushi va p95 kechikish bazaviy versiya bilan solishtiriladi; mezon buzilsa avtomatik orqaga qaytariladi. Sog'lom, nuqsonli va sekin v2 sinab ko'riladi.

Fayl: `s5_5_qadam_versiyani_chiqarish_rollingupdat.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s5_5_qadam_versiyani_chiqarish_rollingupdat.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. 6 nusxa × 50 so'rov/s = 300 so'rov/s sig'im va talab 260. Yangilanish paytida bir vaqtda nechta Pod ishlamay qolsa so'rovlar yo'qola boshlaydi?
2. maxSurge va maxUnavailable Deployment kontrolleriga nimani ruxsat beradi? Har soniyada qaysi tartibda qaror qabul qilinadi?
3. Recreate strategiyasini qanday modellaysiz va unda nima uchun yo'qotish eng katta? Blue/green (surge=6) ning bahosi nimada?
4. Natijalaringiz kutilganga mosmi: maxUnavailable=0 da yo'qotish 0, maxUnavailable=1 da ~1800, Recreate da ~7800 so'rov?
5. Canary'da trafikni w % ehtimol bilan v2 ga qanday yo'naltirasiz? Nega har versiyadan kamida 60 namuna yig'ish kerak?
6. Avtomatik orqaga qaytarish mezoni qanday bo'lishi kerak? Nega p95 uchun nisbiy (1,5×) bilan birga mutlaq (+2 ms) chegara ham qo'yiladi?
7. Uch ssenariyda nimani kutasiz: sog'lom v2 100 % gacha chiqadimi, nuqsonli va sekin v2 qaysi bosqichda to'xtaydi?

Kutilgan natija: maxUnavailable=0 da yo'qotish 0, maxUnavailable=1 da ~1800, Recreate da ~7800 so'rov yo'qoladi; canary: sog'lom v2 100 % gacha chiqadi, nuqsonli va sekin v2 5 % bosqichda to'xtatiladi.

Tajriba qilib ko'ring:

- Canary qaroriga ikki ulush uchun z-test qo'shing (p < 0,05) va qat'iy chegara bilan solishtiring.
- Pod start vaqtini 90 s qiling: qaysi strategiya maqbul bo'ladi?
- v2 xato ulushini 2,5 % qiling — canary uni qaysi bosqichda va necha so'rovdan keyin ushlaydi?

### 6-qadam. HPA (autoscaling/v2) + Cluster Autoscaler + Erlang-C: yuklama simulyatsiyasi

HPA'ning haqiqiy algoritmi modellashtiriladi: desired = ceil(current x metrika / nishon), 10 % tolerantlik, bir nechta metrikada maksimum, scaleUp siyosati (15 s da 100 % yoki 4 Pod) va scaleDown uchun 300 s stabilizatsiya oynasi. Pod'lar tugunlarga joylashadi (tugunda 3 Pod), joy yetmasa Pending bo'ladi va Cluster Autoscaler yangi tugun so'raydi (120 s), ortiqcha tugunni 10 daqiqadan keyin olib tashlaydi. Kechikish M/M/c (Erlang C) bilan baholanadi. 6 soatlik kunlik to'lqin va 10 daqiqalik keskin sakrash (+450 so'rov/s) ostida on-prem (4 tugun) va bulut konfiguratsiyalari SLO buzilish daqiqalari, yo'qolgan so'rovlar, Pod-soat va tugun-soat bo'yicha solishtiriladi.

Fayl: `s6_6_qadam_hpa_autoscaling_v2_cluster_autos.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s6_6_qadam_hpa_autoscaling_v2_cluster_autos.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. HPA formulasi desired = ceil(current × metrika / nishon). Agar 4 Pod 90 % CPU da ishlasa va nishon 70 % bo'lsa, nechta Pod tavsiya qilinadi? Tolerantlik 10 % qachon o'zgarishni to'xtatadi?
2. Nega CPU metrikasi keskin sakrashda yomon ishlaydi, RPS metrikasi esa yaxshiroq? CPU foydalanishni qanday hisoblaysiz va u qayerda to'yinadi?
3. scaleUp va scaleDown siyosatlari nega nosimmetrik? 300 s stabilizatsiya oynasi qanday «tebranish» (flapping) ni oldini oladi?
4. Pod yaratildi, lekin tugunda joy yo'q — u qaysi holatda turadi? Cluster Autoscaler qachon yangi tugun so'raydi va qachon olib tashlaydi?
5. M/M/c navbatida Erlang C nimani bildiradi? Undan p95 javob vaqtini qanday baholaysiz?
6. Natijani tekshiring: Erlang-C da 300 so'rov/s, 7 Pod p95 ~110 ms, 10 Pod ~64 ms chiqdimi? On-prem 4 tugunda sakrashda ~10 daqiqa SLO buzilishi va ~110 ming so'rov yo'qolishi kutiladi — nega aynan shuncha?

Kutilgan natija: On-prem 4 tugunda sakrashda ~10 daqiqa SLO buzilishi va ~110 ming so'rov yo'qoladi; bulut CA bilan ~1,5-5 daqiqa; Erlang-C: 300 so'rov/s da 7 Pod p95 ~110 ms, 10 Pod ~64 ms.

Tajriba qilib ko'ring:

- scaleUp siyosatini 15 s da 2 Pod ga cheklang — SLO buzilishi qanday o'zgaradi?
- Sakrashni prognoz qilib (aksiya jadvali), t=2:55 da minReplicas ni vaqtincha 16 ga ko'taring (KEDA cron kabi).
- Pod start vaqtini 30 s dan 10 s ga tushirsangiz (kichik obraz, model keshlangan) natija qancha yaxshilanadi?

### 7-qadam. Operatsion hisobot: SLO, xato byudjeti, burn-rate ogohlantirishlari va Go/No-Go

30 kunlik daqiqalik so'rov/xato qatori (kunlik to'lqin, 12 % o'sish, ikki insident: keskin va sekin) sintetik yaratiladi. SLO 99,9 % uchun xato byudjeti va uning sarfi, Google SRE Workbook'dagi ko'p oynali burn-rate qoidalari (1 soat/5 daq > 14,4x, 6 soat/30 daq > 6x, 3 kun/6 soat > 1x) hisoblanib, har insidentni aniqlash kechikishi topiladi. Shu qoidalar asosida PrometheusRule (monitoring.coreos.com/v1) YAML generatsiya qilinadi va validatsiya qilinadi. Oxirida kunlik cho'qqi trendi bo'yicha sig'im prognozi va oldingi qadamlar natijalarini jamlovchi Go/No-Go nazorat ro'yxati.

Fayl: `s7_7_qadam_operatsion_hisobot_slo_xato_byud.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s7_7_qadam_operatsion_hisobot_slo_xato_byud.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. SLO 99,9 % bo'lsa, 30 kunlik xato byudjeti nima va u qanday birlikda o'lchanadi? Mavjudlik ~99,92 % bo'lganda byudjetning qancha qismi sarflangan bo'ladi?
2. Burn rate 1x, 6x va 14,4x nimani anglatadi? 14,4x tezlikda 30 kunlik byudjet qancha vaqtda tugaydi?
3. 43200 daqiqalik qatorda 3 kunlik (4320 daq) sirpanuvchi oyna yig'indisini sikl bilan hisoblash juda sekin. Buni bir harakatda qanday qilasiz?
4. Nega har qoida ikki oynani (uzun VA qisqa) talab qiladi? Faqat uzun oyna bo'lsa insident tugagandan keyin nima bo'ladi?
5. Keskin (40 daq, 15 % xato) va sekin (12 soat, 1,2 % xato) insidentlarni qaysi qoidalar ushlashini kutasiz? Aniqlanish kechikishini qanday topasiz?
6. PrometheusRule YAML ni yaratgandan keyin uni qaysi xatolarga tekshirasiz? Nega «page» darajasidagi ogohlantirishda runbook_url majburiy?
7. Sig'im prognozida kunlik cho'qqining chiziqli trendini qanday topasiz va «necha kun qoldi» ni qanday hisoblaysiz? Go/No-Go qarori qachon GO bo'ladi?

Kutilgan natija: Mavjudlik ~99,92 %, byudjet sarfi ~84 %; keskin insident 5 daqiqada, sekin insident faqat 6 soatlik va 3 kunlik qoidalarda aniqlanadi; PrometheusRule VALID (5 qoida); sig'im ~45 kunga yetadi; QAROR: GO.

Tajriba qilib ko'ring:

- SLO ni 99,95 % ga oshiring — qaror va qaysi bandlar o'zgaradi?
- Kechikish SLO'si (p95 < 200 ms) uchun ham burn-rate qoidasini qo'shing (histogram_quantile ifodasi bilan).
- Insidentlar sonini 4 taga oshirib, byudjet tugaydigan kunni va 'muzlatish' (feature freeze) siyosatini hisobotga yozing.

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
|---|---|---|---|
| 1 | **Xizmatga yangi endpoint va testlar** — 1-qadam xizmatiga /v1/explain endpointini qo'shing (v1 logistik regressiya uchun eng katta hissa qo'shgan 5 ta belgi va ularning koeffitsient x qiymati). Kamida 3 ta yangi test yozing (200, 401, 422) va test jadvalini topshiring. | oson | 5 |
| 2 | **Konfiguratsiya sxemasini kengaytirish** — 2-qadam sxemasiga 3 ta yangi kalit qo'shing (masalan, redis_url, cache_ttl_s, cors_origins) va har biri uchun tur/diapazon tekshiruvi yozing. To'g'ri va xato konfiguratsiyalar uchun natijalarni hamda manba (standart/ConfigMap/env) jadvalini ko'rsating. | oson | 5 |
| 3 | **Dockerfile qoidalari va obraz hajmi** — 3-qadam tahlilchisiga 3 ta qoida qo'shing (.dockerignore borligi, LABEL versiyasi, HEALTHCHECK'da curl o'rniga python yoki wget). Paketlar hajmini taxminiy baholab (python:3.12-slim ~ 130 MB, numpy, scikit-learn, fastapi), bir bosqichli va ko'p bosqichli obraz hajmini jadvalda solishtiring. | oson | 5 |
| 4 | **Probe parametrlarini optimallashtirish** — 2-qadam kubelet simulyatorida model yuklanish vaqti 20, 45, 90, 180 s bo'lgan 4 ssenariy uchun startupProbe (period, failureThreshold) ni tanlang: tayyor bo'lish vaqti minimal, restart 0 bo'lsin, osilib qolgan konteyner esa 5 daqiqada qayta ishga tushsin. Natijani jadval va tavsiya ko'rinishida bering. | o'rta | 10 |
| 5 | **Manifest validatorini kengaytirish** — 4-qadam tekshiruvchisiga 5 ta yangi qoida qo'shing: NetworkPolicy podSelector Deployment'ga mos, Ingress TLS secretName mavjud (Secret obyekti), HPA metrikasi uchun requests.cpu bor, imagePullPolicy IfNotPresent/Always, PVC readOnly mount. Har qoida uchun uni buzadigan test manifest yozing va natijalar jadvalini topshiring. | o'rta | 10 |
| 6 | **Rollout strategiyasi tanlovi** — 5-qadam rolling simulyatorida 10 nusxa, Pod start 60 s, talab 80 % sig'im sharoitida maxSurge (0..3) x maxUnavailable (0..3) setkasini hisoblang: davomiylik, yo'qolgan so'rovlar va qo'shimcha Pod-soniya. Pareto-optimal kombinatsiyalarni ajrating va grafikda ko'rsating. | o'rta | 10 |
| 7 | **Canary statistik kuchi** — 5-qadam canary tahliliga ikki ulush uchun z-test qo'shing. v2 xato ulushi 1, 2, 4, 8 % bo'lganda (baza 0,5 %) 5 % bosqichda nuqsonni aniqlash uchun kerakli canary namunalari sonini va umumiy so'rovlar sonini toping; 20 ta urug' bilan takrorlab, aniqlash ehtimolini jadvalda bering. | o'rta | 10 |
| 8 | **HPA va CA sozlash tadqiqoti** — 6-qadam simulyatorida o'z yuklama profilingizni (masalan, ertalabki cho'qqi + 2 ta sakrash) bering va kamida 8 konfiguratsiyani (nishon %, min/max, RPS metrikasi, tugun start vaqti, oldindan zaxira, scaleDown oynasi) SLO buzilishi va narx (Pod-soat x narx + tugun-soat x narx) bo'yicha solishtiring. Pareto-chegara grafigini chizing va tanlovni asoslang. | qiyin | 15 |
| 9 | **To'liq joylashtirish to'plami** — 1-4 qadamlar asosida bitta skript yozing: u xizmat kodini (main.py), requirements.txt, Dockerfile, docker-compose.yml va dev/prod K8s manifestlarini /tmp/deploy papkasiga generatsiya qilsin, barcha validatorlarni ishga tushirsin va umumiy PASS/FAIL hisobotini chiqarsin. Agar real Docker yoki Kubernetes (minikube, kind, k3s) bo'lsa, docker build va kubectl apply natijasini skrinshot bilan ilova qiling; bo'lmasa simulyatsiya natijalarini bering. | qiyin | 15 |
| 10 | **Operatsion hisobot o'z xizmatingiz uchun** — O'z AI xizmatingiz (masalan, tasvir tasnifi, chatbot, tavsiya tizimi) uchun 7-qadamni moslang: SLO'larni (mavjudlik va kechikish) tanlang, 30 kunlik sintetik ma'lumotga 3 ta turli insident kiriting, burn-rate qoidalari va PrometheusRule'ni generatsiya qiling, aniqlash kechikishlarini jadvalda bering, sig'im prognozi va Go/No-Go qarorini raqamlar bilan asoslang (kamida 400 so'z). | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


## 1-variant. Tibbiyot: qandli diabet xavfini baholash API (oson)

Soha: Tibbiyot

Poliklinikalar tarmog'i bemor laboratoriya ko'rsatkichlari bo'yicha diabet xavfini baholovchi ichki API'ni Kubernetes'ga joylashtiradi. Ish vaqtida 30 so'rov/s, kechasi deyarli nol. Ma'lumot shifoxona tarmog'idan chiqmaydi, model har chorakda yangilanadi.

| Parametr | Qiymat |
|---|---|
| Model | sklearn diabetes (regressiya -> xavf darajasi), Ridge v1 / GBR v2 |
| Yuklama profili | kunduz 30, tun 2 so'rov/s |
| SLO | p95 <= 150 ms, 99,5 % |
| Klaster | 3 tugun on-prem |
| Rollout | RollingUpdate maxSurge 1 |
| HPA | min 2, max 6, CPU 60 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sklearn diabetes (regressiya -> xavf darajasi), Ridge v1 / GBR v2): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 2, max 6, CPU 60 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (RollingUpdate maxSurge 1) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 2-variant. Agro: tuproq namligi bo'yicha sug'orish tavsiyasi (oson)

Soha: Qishloq xo'jaligi

Fermer xo'jaliklari IoT sensorlari har 10 daqiqada ma'lumot yuboradi, xizmat sug'orish kerakmi-yo'qligini qaytaradi. Sensorlar soni mavsumda 2000 dan 15000 gacha o'sadi. Xizmat tuman data-markazidagi kichik k3s klasterida ishlaydi.

| Parametr | Qiymat |
|---|---|
| Model | make_regression (8 belgi) -> RandomForest |
| Yuklama profili | 2000 -> 15000 qurilma / 10 daq |
| SLO | p95 <= 500 ms, 99 % |
| Klaster | k3s, 3 tugun |
| Rollout | Recreate vs RollingUpdate |
| HPA | min 1, max 4, CPU 70 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: make_regression (8 belgi) -> RandomForest): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 1, max 4, CPU 70 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (Recreate vs RollingUpdate) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 3-variant. Transport: taksi narxini dinamik hisoblash (oson)

Soha: Transport

Taksi agregatori har buyurtmada narxni ML model bilan hisoblaydi; juma kechqurun yuklama 5 barobar oshadi. Kechikish ilova tajribasiga bevosita ta'sir qiladi. Siz canary bilan yangi narx modelini xavfsiz chiqarasiz.

| Parametr | Qiymat |
|---|---|
| Model | make_regression (12 belgi) -> GradientBoosting |
| Yuklama profili | 80 -> 400 so'rov/s (juma) |
| SLO | p99 <= 100 ms, 99,9 % |
| Klaster | bulut, 4 -> 12 tugun |
| Rollout | canary 5/25/50/100 |
| HPA | min 4, max 30, CPU 65 % + RPS |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: make_regression (12 belgi) -> GradientBoosting): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 4, max 30, CPU 65 % + RPS, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (canary 5/25/50/100) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 4-variant. Telekom: abonent ketishini (churn) bashorat qilish (oson)

Soha: Telekom

Operator CRM tizimi har abonent qo'ng'irog'ida churn ehtimolini so'raydi; call-markaz ish vaqtida 120 so'rov/s. Model har oy qayta o'qitiladi va A/B sinovdan o'tkaziladi. Konfiguratsiya ConfigMap orqali, API kalit Secret orqali boshqariladi.

| Parametr | Qiymat |
|---|---|
| Model | make_classification (20 belgi, nomutanosib 9:1) -> LogisticRegression |
| Yuklama profili | 120 so'rov/s, 8:00-20:00 |
| SLO | p95 <= 200 ms, 99,9 % |
| Klaster | on-prem, 5 tugun |
| Rollout | blue/green |
| HPA | min 3, max 10, CPU 70 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: make_classification (20 belgi, nomutanosib 9:1) -> LogisticRegression): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 3, max 10, CPU 70 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (blue/green) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 5-variant. Energetika: elektr iste'molini soatlik prognoz qilish (oson)

Soha: Energetika

Hududiy elektr tarmog'i kompaniyasi har soat 1200 podstansiya bo'yicha keyingi 24 soat iste'molini prognoz qiladi. Yuklama soat boshida qisqa cho'qqi beradi (paketli so'rovlar). Xizmat batch endpointi bilan ishlaydi.

| Parametr | Qiymat |
|---|---|
| Model | sintetik vaqt qatori (sinus + shovqin) -> Ridge, batch 64 |
| Yuklama profili | soat boshida 1200 so'rov 2 daqiqada |
| SLO | batch p95 <= 2 s |
| Klaster | on-prem, 3 tugun |
| Rollout | RollingUpdate maxUnavailable 0 |
| HPA | min 2, max 8, RPS 20/Pod |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik vaqt qatori (sinus + shovqin) -> Ridge, batch 64): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 2, max 8, RPS 20/Pod, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (RollingUpdate maxUnavailable 0) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 6-variant. Moliya: kredit skoring xizmati (oson)

Soha: Moliya (bank)

Bank onlayn kredit arizalarini 2 soniyada ko'rib chiqishi kerak; skoring modeli regulyator talabiga ko'ra versiyalanadi va har javob model versiyasi bilan jurnallanadi. Xizmat ikki namespace'da (dev, prod) va NetworkPolicy bilan izolyatsiyalangan.

| Parametr | Qiymat |
|---|---|
| Model | make_classification (15 belgi) -> LogisticRegression v1 / RandomForest v2 |
| Yuklama profili | 25 so'rov/s, cho'qqi 90 |
| SLO | p95 <= 300 ms, 99,95 % |
| Klaster | on-prem, 3 control-plane + 4 worker |
| Rollout | canary + qo'lda tasdiq |
| HPA | min 3, max 8, CPU 60 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: make_classification (15 belgi) -> LogisticRegression v1 / RandomForest v2): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 3, max 8, CPU 60 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (canary + qo'lda tasdiq) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 7-variant. Ta'lim: insho baholash yordamchisi (oson)

Soha: Ta'lim

Universitet talabalar inshosini matn belgilari (uzunlik, so'z boyligi, xatolar soni) bo'yicha dastlabki baholovchi xizmatni ishga tushiradi. Sessiya davrida yuklama 10 barobar o'sadi. Xizmat docker-compose bilan sinovdan o'tib, keyin K8s'ga ko'chiriladi.

| Parametr | Qiymat |
|---|---|
| Model | sintetik matn belgilari (10 belgi) -> Ridge |
| Yuklama profili | 5 -> 50 so'rov/s (sessiya) |
| SLO | p95 <= 1 s, 99 % |
| Klaster | docker-compose -> k3s 2 tugun |
| Rollout | RollingUpdate |
| HPA | min 1, max 5, CPU 75 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik matn belgilari (10 belgi) -> Ridge): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 1, max 5, CPU 75 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (RollingUpdate) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 8-variant. Sanoat: nasoslarning nosozligini vibratsiya bo'yicha bashorat qilish (o'rta)

Soha: Sanoat

Kimyo zavodida 600 nasos sensori har 5 s da vibratsiya spektri xususiyatlarini yuboradi; xizmat nosozlik ehtimolini qaytaradi. Zavod tarmog'i internetdan uzilgan, obrazlar ichki registrdan (NAS) olinadi. Yangilanish faqat ta'mirlash oynasida.

| Parametr | Qiymat |
|---|---|
| Model | sintetik sensor (16 belgi) -> IsolationForest + LogisticRegression |
| Yuklama profili | 120 so'rov/s doimiy |
| SLO | p99 <= 250 ms, 99,9 % |
| Klaster | havo bo'shlig'i (air-gap), 3 tugun |
| Rollout | ta'mirlash oynasida Rolling |
| HPA | min 3, max 6, CPU 60 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik sensor (16 belgi) -> IsolationForest + LogisticRegression): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 3, max 6, CPU 60 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (ta'mirlash oynasida Rolling) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (120 so'rov/s doimiy, havo bo'shlig'i (air-gap), 3 tugun) — kamida 4 konfiguratsiya, SLO (p99 <= 250 ms, 99,9 %) buzilishi va Pod/tugun-soat.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 9-variant. Xavfsizlik: kirish tizimida yuz embeddingini solishtirish (o'rta)

Soha: Xavfsizlik

Korxona turniketlarida xodimning yuz embeddingi (128 o'lchamli) bazadagi 5000 ta bilan solishtiriladi. Smena boshida 15 daqiqada 3000 kirish. Xizmat ishdan chiqsa turniket ochilmaydi, shuning uchun PDB va topologiya tarqalishi majburiy.

| Parametr | Qiymat |
|---|---|
| Model | sintetik embedding (128) -> kNN (sklearn NearestNeighbors) |
| Yuklama profili | smena boshida 200 so'rov/s, 15 daq |
| SLO | p99 <= 300 ms, 99,99 % |
| Klaster | on-prem, 3 tugun, 2 zona |
| Rollout | RollingUpdate maxSurge 2 |
| HPA | min 4, max 12, RPS 25/Pod |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik embedding (128) -> kNN (sklearn NearestNeighbors)): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 4, max 12, RPS 25/Pod, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (RollingUpdate maxSurge 2) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (smena boshida 200 so'rov/s, 15 daq, on-prem, 3 tugun, 2 zona) — kamida 4 konfiguratsiya, SLO (p99 <= 300 ms, 99,99 %) buzilishi va Pod/tugun-soat.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 10-variant. Chakana savdo: mahsulot tavsiya API (o'rta)

Soha: Chakana savdo

Onlayn do'kon har sahifa ochilganda tavsiya ro'yxatini so'raydi; aksiya kunlari yuklama 8 barobar o'sadi. Natijalar Redis keshida 60 s saqlanadi. Siz compose'da kesh bilan, K8s'da esa HPA va Cluster Autoscaler bilan sinaysiz.

| Parametr | Qiymat |
|---|---|
| Model | sintetik foydalanuvchi-mahsulot matritsasi -> TruncatedSVD |
| Yuklama profili | 150 -> 1200 so'rov/s (aksiya) |
| SLO | p95 <= 120 ms, 99,9 % |
| Klaster | bulut, 3 -> 15 tugun |
| Rollout | canary 10/50/100 |
| HPA | min 3, max 40, CPU 60 % + RPS |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik foydalanuvchi-mahsulot matritsasi -> TruncatedSVD): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 3, max 40, CPU 60 % + RPS, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (canary 10/50/100) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (150 -> 1200 so'rov/s (aksiya), bulut, 3 -> 15 tugun) — kamida 4 konfiguratsiya, SLO (p95 <= 120 ms, 99,9 %) buzilishi va Pod/tugun-soat.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 11-variant. Logistika: yetkazib berish vaqtini bashorat qilish (ETA) (o'rta)

Soha: Logistika

Kuryerlik xizmati har buyurtmada ETA ni ML bilan hisoblaydi va mijozga ko'rsatadi; tushlik va kechki cho'qqilar. Model har hafta yangilanadi, xato bashorat mijoz shikoyatiga olib keladi — canary tahlilida MAE ham kuzatiladi.

| Parametr | Qiymat |
|---|---|
| Model | make_regression (10 belgi, shovqinli) -> HistGradientBoosting |
| Yuklama profili | 40 -> 250 so'rov/s (2 cho'qqi) |
| SLO | p95 <= 200 ms, 99,9 % |
| Klaster | bulut, 3 -> 8 tugun |
| Rollout | canary + MAE tahlili |
| HPA | min 3, max 15, CPU 65 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: make_regression (10 belgi, shovqinli) -> HistGradientBoosting): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 3, max 15, CPU 65 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (canary + MAE tahlili) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (40 -> 250 so'rov/s (2 cho'qqi), bulut, 3 -> 8 tugun) — kamida 4 konfiguratsiya, SLO (p95 <= 200 ms, 99,9 %) buzilishi va Pod/tugun-soat.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 12-variant. Shahar: murojaatlarni avtomatik tasniflash (311 xizmati) (o'rta)

Soha: Shahar boshqaruvi

Hokimiyat portaliga kelgan fuqarolar murojaatlari (matn belgilari) tegishli bo'limga yo'naltiriladi. Ish kunlari 9:00-11:00 cho'qqi. Xizmat davlat data-markazidagi klasterda, barcha sirlar Secret orqali, tasvirlar root'siz.

| Parametr | Qiymat |
|---|---|
| Model | sintetik matn TF-IDF (500 belgi) -> LinearSVC |
| Yuklama profili | 10 -> 60 so'rov/s |
| SLO | p95 <= 400 ms, 99,5 % |
| Klaster | on-prem, 4 tugun |
| Rollout | RollingUpdate maxSurge 25 % |
| HPA | min 2, max 8, CPU 70 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik matn TF-IDF (500 belgi) -> LinearSVC): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 2, max 8, CPU 70 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (RollingUpdate maxSurge 25 %) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (10 -> 60 so'rov/s, on-prem, 4 tugun) — kamida 4 konfiguratsiya, SLO (p95 <= 400 ms, 99,5 %) buzilishi va Pod/tugun-soat.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 13-variant. Ekologiya: suv sifati anomaliyasini aniqlash (o'rta)

Soha: Ekologiya

Suv ta'minoti korxonasi 80 nazorat nuqtasidan har daqiqada pH, loyqalik, xlor ma'lumotini oladi; anomaliya bo'lsa navbatchiga ogohlantirish. Xizmat kichik edge-klasterda (3 ta mini-PC) ishlaydi, elektr uzilishiga UPS bilan chidamli bo'lishi kerak.

| Parametr | Qiymat |
|---|---|
| Model | sintetik sensor (6 belgi) -> IsolationForest |
| Yuklama profili | 80 so'rov/daq doimiy |
| SLO | p95 <= 1 s, 99,9 % |
| Klaster | edge k3s, 3 mini-PC |
| Rollout | Recreate taqiqlangan, Rolling |
| HPA | min 2, max 3, CPU 80 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik sensor (6 belgi) -> IsolationForest): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 2, max 3, CPU 80 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (Recreate taqiqlangan, Rolling) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (80 so'rov/daq doimiy, edge k3s, 3 mini-PC) — kamida 4 konfiguratsiya, SLO (p95 <= 1 s, 99,9 %) buzilishi va Pod/tugun-soat.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 14-variant. Sport: o'yinchi jarohati xavfini baholash (o'rta)

Soha: Sport

Futbol akademiyasi mashg'ulot yuklamasi (GPS, yurak urishi) bo'yicha o'yinchi jarohati xavfini har kuni hisoblaydi; murabbiylar planshetdan so'raydi. Yuklama kichik, lekin mavjudlik mashg'ulot vaqtida muhim. Xizmat bitta tugunli klasterdan 3 tugunliga ko'chiriladi.

| Parametr | Qiymat |
|---|---|
| Model | make_classification (12 belgi) -> RandomForest |
| Yuklama profili | 2 -> 15 so'rov/s |
| SLO | p95 <= 500 ms, 99,5 % |
| Klaster | 1 -> 3 tugun |
| Rollout | blue/green |
| HPA | min 1, max 3, CPU 70 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: make_classification (12 belgi) -> RandomForest): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 1, max 3, CPU 70 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (blue/green) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (2 -> 15 so'rov/s, 1 -> 3 tugun) — kamida 4 konfiguratsiya, SLO (p95 <= 500 ms, 99,5 %) buzilishi va Pod/tugun-soat.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 15-variant. Media: kontent moderatsiyasi (toksik izohlar) (o'rta)

Soha: Media

Yangiliklar portali izohlarni chop etishdan oldin toksiklikka tekshiradi; mashhur yangilikda bir necha daqiqada yuklama 20 barobar sakraydi. Xizmat bulutda, Cluster Autoscaler tugun qo'shishi 3 daqiqa oladi — oldindan zaxira siyosatini tanlaysiz.

| Parametr | Qiymat |
|---|---|
| Model | sintetik matn belgilari (300) -> LogisticRegression |
| Yuklama profili | 20 -> 400 so'rov/s (10 daq) |
| SLO | p95 <= 250 ms, 99,9 % |
| Klaster | bulut, 2 -> 12 tugun |
| Rollout | canary 5/20/100 |
| HPA | min 2, max 40, RPS 30/Pod |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik matn belgilari (300) -> LogisticRegression): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 2, max 40, RPS 30/Pod, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (canary 5/20/100) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (20 -> 400 so'rov/s (10 daq), bulut, 2 -> 12 tugun) — kamida 4 konfiguratsiya, SLO (p95 <= 250 ms, 99,9 %) buzilishi va Pod/tugun-soat.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan)

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 16-variant. Tibbiyot: tez yordam chaqiruvlarini triaj qilish (qiyin)

Soha: Tibbiyot (shoshilinch yordam)

103 xizmati operatori chaqiruv belgilarini kiritadi, model ustuvorlik darajasini (1-4) taklif qiladi. Xizmat to'xtashi mumkin emas: ikki klaster (asosiy va zaxira), har yangilanish canary orqali, xato byudjeti juda kichik. Burn-rate ogohlantirishlari pager'ga ulanadi.

| Parametr | Qiymat |
|---|---|
| Model | make_classification (4 sinf, 18 belgi) -> GradientBoosting |
| Yuklama profili | 15 -> 60 so'rov/s |
| SLO | p99 <= 300 ms, 99,99 % |
| Klaster | 2 klaster x 3 tugun |
| Rollout | canary 1/10/50/100 |
| HPA | min 4, max 10, CPU 50 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: make_classification (4 sinf, 18 belgi) -> GradientBoosting): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 4, max 10, CPU 50 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (canary 1/10/50/100) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (15 -> 60 so'rov/s, 2 klaster x 3 tugun) — kamida 4 konfiguratsiya, SLO (p99 <= 300 ms, 99,99 %) buzilishi va Pod/tugun-soat.
7. 7-qadam: 30 kunlik sintetik ma'lumotda SLO va xato byudjeti, burn-rate qoidalari, PrometheusRule YAML, sig'im prognozi va Go/No-Go qarori.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Operatsion hisobot: SLO, ogohlantirishlar va Go/No-Go

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 17-variant. Neft-gaz: quduq debitini bashorat qilish (qiyin)

Soha: Neft-gaz

Kompaniya 900 quduq bo'yicha har soat debit prognozini hisoblaydi va dispetcherga ko'rsatadi. Klaster dala ofisida, internet sun'iy yo'ldosh orqali sekin, shuning uchun obrazlar kichik bo'lishi va ichki registrda keshlanishi shart. Dockerfile hajmini optimallashtirasiz.

| Parametr | Qiymat |
|---|---|
| Model | sintetik vaqt qatori (14 belgi) -> HistGradientBoosting |
| Yuklama profili | 900 so'rov / soat boshida |
| SLO | batch p95 <= 3 s, 99,5 % |
| Klaster | dala ofisi, 3 tugun, sekin WAN |
| Rollout | RollingUpdate, obraz oldindan tortiladi |
| HPA | min 2, max 4, CPU 70 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik vaqt qatori (14 belgi) -> HistGradientBoosting): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 2, max 4, CPU 70 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (RollingUpdate, obraz oldindan tortiladi) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (900 so'rov / soat boshida, dala ofisi, 3 tugun, sekin WAN) — kamida 4 konfiguratsiya, SLO (batch p95 <= 3 s, 99,5 %) buzilishi va Pod/tugun-soat.
7. 7-qadam: 30 kunlik sintetik ma'lumotda SLO va xato byudjeti, burn-rate qoidalari, PrometheusRule YAML, sig'im prognozi va Go/No-Go qarori.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Operatsion hisobot: SLO, ogohlantirishlar va Go/No-Go

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 18-variant. Sug'urta: avtomobil zarari da'vosini firibgarlikka tekshirish (qiyin)

Soha: Sug'urta

Sug'urta kompaniyasi har da'voni firibgarlik ehtimoli bo'yicha baholaydi; da'volar ish kunlari tekis keladi, oy oxirida 3 barobar. Regulyator har model versiyasining audit izini talab qiladi: x-model-version, sha256 va konfiguratsiya xeshi jurnallanadi.

| Parametr | Qiymat |
|---|---|
| Model | make_classification (25 belgi, 19:1) -> RandomForest v1 / GBC v2 |
| Yuklama profili | 10 -> 30 so'rov/s (oy oxiri) |
| SLO | p95 <= 800 ms, 99,9 % |
| Klaster | on-prem, 3 control-plane + 3 worker |
| Rollout | canary + audit |
| HPA | min 2, max 6, CPU 65 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: make_classification (25 belgi, 19:1) -> RandomForest v1 / GBC v2): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 2, max 6, CPU 65 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (canary + audit) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (10 -> 30 so'rov/s (oy oxiri), on-prem, 3 control-plane + 3 worker) — kamida 4 konfiguratsiya, SLO (p95 <= 800 ms, 99,9 %) buzilishi va Pod/tugun-soat.
7. 7-qadam: 30 kunlik sintetik ma'lumotda SLO va xato byudjeti, burn-rate qoidalari, PrometheusRule YAML, sig'im prognozi va Go/No-Go qarori.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Operatsion hisobot: SLO, ogohlantirishlar va Go/No-Go

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 19-variant. Davlat xizmatlari: hujjat turini tasniflash (OCR natijasi) (qiyin)

Soha: Davlat xizmatlari

Yagona interaktiv portalda yuklangan skanlar OCR'dan keyin matn belgilari bo'yicha hujjat turiga (pasport, diplom, ma'lumotnoma ...) ajratiladi. Murojaatlar soni yil davomida o'sib bormoqda — sig'im prognozi va HPA maxReplicas rejasini tuzasiz.

| Parametr | Qiymat |
|---|---|
| Model | sintetik matn belgilari (200) -> MultinomialNB / LogisticRegression |
| Yuklama profili | 50 so'rov/s, oyiga +5 % |
| SLO | p95 <= 600 ms, 99,9 % |
| Klaster | davlat DC, 6 tugun |
| Rollout | RollingUpdate + canary |
| HPA | min 3, max 20, CPU 70 % |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik matn belgilari (200) -> MultinomialNB / LogisticRegression): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 3, max 20, CPU 70 %, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (RollingUpdate + canary) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (50 so'rov/s, oyiga +5 %, davlat DC, 6 tugun) — kamida 4 konfiguratsiya, SLO (p95 <= 600 ms, 99,9 %) buzilishi va Pod/tugun-soat.
7. 7-qadam: 30 kunlik sintetik ma'lumotda SLO va xato byudjeti, burn-rate qoidalari, PrometheusRule YAML, sig'im prognozi va Go/No-Go qarori.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Operatsion hisobot: SLO, ogohlantirishlar va Go/No-Go

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)

## 20-variant. Kiberxavfsizlik: login urinishlarida anomaliyani aniqlash (qiyin)

Soha: Kiberxavfsizlik

Korporativ SSO tizimi har login urinishini (IP, vaqt, qurilma belgilari) xavf bali bo'yicha baholaydi; hujum paytida yuklama 50 barobar sakraydi va aynan shu payt xizmat ishlashi shart. Rate-limit, NetworkPolicy va tez masshtablash muhim.

| Parametr | Qiymat |
|---|---|
| Model | sintetik log belgilari (10) -> IsolationForest + LogisticRegression |
| Yuklama profili | 50 -> 2500 so'rov/s (hujum) |
| SLO | p99 <= 150 ms, 99,95 % |
| Klaster | bulut, 4 -> 20 tugun |
| Rollout | canary 5/50/100 |
| HPA | min 4, max 60, RPS 40/Pod |

Qadamlar:

1. 1-qadam kabi FastAPI xizmatini yozing (model: sintetik log belgilari (10) -> IsolationForest + LogisticRegression): probe'lar, API kalit, so'rov ID, batch, /metrics; kamida 10 ta TestClient testi PASS.
2. 2-qadam: konfiguratsiya sxemasi (standart < ConfigMap < env) va chuqur readiness; variant model yuklanish vaqti uchun probe parametrlarini kubelet simulyatorida tekshiring.
3. 3-qadam: ko'p bosqichli Dockerfile va docker-compose (xizmat + kerakli yordamchi xizmatlar) generatsiyasi, 13 qoida va compose validatsiyasi PASS; qatlam keshi tahlili.
4. 4-qadam: dev/prod manifestlari (Deployment, Service, HPA min 4, max 60, RPS 40/Pod, ConfigMap, Secret, Ingress, PDB, NetworkPolicy) — 18 qoida PASS, buzilgan nusxada xatolar ushlansin.
5. 5-qadam: rollout (canary 5/50/100) simulyatsiyasi va TestClient bilan canary — sog'lom va nuqsonli v2 uchun qarorlar jadvali.
6. 6-qadam: HPA + Cluster Autoscaler simulyatsiyasi (50 -> 2500 so'rov/s (hujum), bulut, 4 -> 20 tugun) — kamida 4 konfiguratsiya, SLO (p99 <= 150 ms, 99,95 %) buzilishi va Pod/tugun-soat.
7. 7-qadam: 30 kunlik sintetik ma'lumotda SLO va xato byudjeti, burn-rate qoidalari, PrometheusRule YAML, sig'im prognozi va Go/No-Go qarori.

Topshiriladi: Kod (.py: xizmat, generatorlar, simulyatorlar); Dockerfile, docker-compose.yml va K8s YAML (PASS natijasi bilan); Natijalar jadvallari va grafiklar (PNG); Xulosa (kamida 150 so'z, raqamlarga tayangan); Operatsion hisobot: SLO, ogohlantirishlar va Go/No-Go

Baholash: To'g'ri ishlaydigan kod va testlar — 30, Manifestlar va validatsiya — 25, Simulyatsiya va tahlil — 25, Hisobot sifati — 20 (jami 100)
