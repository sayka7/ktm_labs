# MT4. Mustaqil bajarish uchun variantlar (20)

Variant raqami = jurnal (guruh ro'yxati) raqamingiz; 20 dan katta bo'lsa 20 ga bo'lgandagi qoldiq (0 bo'lsa 20-variant). MT4 laboratoriyasining s1-s7 qadamlari kodini (FastAPI xizmati va testlar, konfiguratsiya va probe'lar, Dockerfile/compose, K8s generatori va validatori, rollout/canary, HPA+CA simulyatsiyasi, operatsion hisobot) asos qilib oling va variantdagi model, yuklama, SLO va klasterga moslang. Docker va Kubernetes talab qilinmaydi — xizmat TestClient bilan, klaster esa simulyatsiya sifatida 2 yadroli CPU da ishlaydi; real klaster bo'lsa natijani qo'shimcha ilova qiling. Topshiriq: .py, Dockerfile, YAML, jadval va grafiklar bitta PDF hisobotda, variant raqami ko'rsatilgan holda.

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
