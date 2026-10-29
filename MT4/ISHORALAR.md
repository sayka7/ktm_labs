# MT4. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. AI xizmati: FastAPI, model reestri, API kalit, probe'lar va avtomatik testlar

1. meta.json ga versiya, aniqlik va pickle baytlarining SHA-256 xeshini (dastlabki 16 belgi) yozing; lifespan da xeshni qayta hisoblab solishtiring.

   ```python
   hashlib.sha256(blob).hexdigest()[:16]
   ```
2. Startup — og'ir ish bir marta. Isitish birinchi foydalanuvchi so'rovi sekin bo'lmasligi uchun. yield dan keyingi kod — SIGTERM da drain (ready=False).
3. livez — jarayon tirikmi (har doim ok); readyz — trafik qabul qila oladimi (model yuklanmagan bo'lsa 503); startupz — ishga tushish tugadimi.
4. @app.middleware("http") — bitta joyda. Kelgan sarlavhani saqlang, aks holda uuid4().hex[:12]. /v1/ yo'llari kechikishini S["lat"] ga yozing.

   ```python
   rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
   ```
5. Pydantic field_validator ValueError chiqarsa FastAPI o'zi 422 beradi. Kalit — Header(None) parametri va HTTPException(401); paket uzunligi > MAX_BATCH → HTTPException(413).
6. `# TYPE nom counter` qatori, keyin `nom qiymat`. Hisoblang: 1 predict + 32 batch + 100 ketma-ket = 133 (401 va 422 so'rovlar hisoblanmaydi).
7. Har test bitta talabni tekshiradi — nomidan qaysi endpoint yoki sarlavha ekanini toping. MODEL_VERSION=v2 bilan ham ishga tushirib ko'ring.

Bajargach o'ylang: Namunadagi middleware va kalit() funksiyasini o'zingiznikiga solishtiring: API kalitni Depends() orqali tekshirish qanday afzallik beradi? Nega xato hisoblagichi (auth_errors_total) foydali?

## 2-qadam. Konfiguratsiya qatlamlari, chuqur readiness, probe'lar va to'g'ri to'xtash

1. Ustuvorlik: standart < ConfigMap < env. Env — eng aniq (Pod darajasida, Secret'dan). Har kalit qayerdan kelganini manba lug'atida saqlang.
2. tur(qiymat) ni try/except (TypeError, ValueError) ichida chaqiring, keyin shart(qiymat) ni tekshiring. Xatoli konfiguratsiyada 3 ta xato chiqishi kerak.

   ```python
   cfg[k] = tur(cfg[k])
   ```
3. Xato sozlama bilan ishlagan xizmat jimgina noto'g'ri natija beradi. Xesh o'zgarsa Pod shabloni o'zgaradi va Pod'lar qayta yaratiladi. json.dumps(..., sort_keys=True) — barqaror xesh uchun.
4. readyz — bog'liqliklar (model, xesh, disk, xotira, drain); livez — faqat jarayon. Buzilgan faylda livez yiqilsa, kubelet keraksiz restart qiladi — bu muammoni hal qilmaydi.
5. Ilova tayyor bo'lgunga qadar threshold marta yiqilsa — restart va backoff (10, 20, 40… 300 s). 2×15 = 30 s < 45 s, demak har safar yiqiladi. 2 xato konfiguratsiyada 6 restart, to'g'rilarida tayyor 45 s.
6. Trafik hali kelayotgan paytda jarayon yopiladi: 100 × (3 − preStop) so'rov. grace < preStop + 0,4 bo'lsa, ishlanayotgan so'rovlar ham uziladi. preStop=0 da ~300 kutiladi.

Bajargach o'ylang: Namunadagi kubelet() modelini o'zingiznikiga solishtiring. Real Kubernetes'da yana qaysi omil (masalan, readiness yiqilganda Service'dan chiqarish) bu modelda hisobga olinmagan?

## 3-qadam. Dockerfile va docker-compose: generatsiya, validatsiya va qatlam keshi

1. Builder'da pip wheel bilan g'ildiraklar yasaladi; yakuniy obrazga faqat /wheels (COPY --from=builder) o'tadi, kompilyator va kesh qolmaydi. Baza — slim, teg qotirilgan.
2. USER (uid 10001, root emas), ENV da KEY/SECRET yo'q (Secret orqali), HEALTHCHECK /readyz ga. CMD — JSON (exec) shaklida, aks holda SIGTERM ilovaga yetmaydi.
3. Bo'sh va # bilan boshlanuvchi qatorlarni tashlang; har qatorning birinchi so'zi — ko'rsatma. Yaxshi fayl 13/13, yomoni 0/13 bo'lishi kerak.

   ```python
   re.search(r"ENV.*(KEY|SECRET|PASSWORD|TOKEN)=", l, re.I)
   ```
4. Har qatlam xeshi = oldingi xesh + ko'rsatma + nusxalangan fayllar mazmuni. Bitta qatlam o'zgarsa, undan keyingilarning hammasi qayta quriladi.

   ```python
   h = hashlib.sha1((h + l + manba).encode()).hexdigest()
   ```
5. Kutilgan: kod — 1 qatlam (~3 s), requirements — 5 qatlam (~200 s). YOMON da COPY . . birinchi turadi.
6. Topologik saralash: har qadamda barcha bog'liqliklari allaqachon tartibda bo'lgan xizmatlarni oling; hech biri tayyor bo'lmasa — sikl. Kutilgan: redis → api → prometheus.
7. Port to'qnashuvi, service_healthy shartida healthcheck yo'qligi, environment'dagi sir va depends_on sikli.

Bajargach o'ylang: Namunadagi tahlil() qoidalarini o'zingiznikiga solishtiring: qaysi qoidani regex bilan yozish ishonchsiz (yolg'on PASS beradi)? Haqiqiy loyihada hadolint qanday afzallik beradi?

## 4-qadam. Kubernetes manifestlari: generator, dev/prod overlay va 18 qoidali tekshiruv

1. Faqat ns, replicas, hpa, host va LOG_LEVEL farq qiladi. Chuqur birlashtirish (deep merge) config ichidagi bitta kalitni almashtiradi, qolganlarini saqlaydi — oddiy dict.update esa butun config ni yo'qotadi.
2. selector.matchLabels ⊆ template.metadata.labels bo'lishi shart; Service.spec.selector ham shu label'larni tanlashi kerak (K08, K09).
3. Pod shabloniga checksum/config annotatsiyasi (config xeshi) qo'ying. Sozlama — envFrom configMapRef, sir — secretKeyRef (ochiq value emas).

   ```python
   hashlib.sha256(json.dumps(p["config"], sort_keys=True).encode()).hexdigest()[:10]
   ```
4. minAvailable ≥ replicas bo'lsa tugun drain'i butunlay bloklanadi. replicas ≥ 2 da yarating, minAvailable = replicas − 1. Natija: dev 8, prod 9 obyekt.
5. {(kind, name): obyekt} lug'ati. Ingress port name yoki number bo'yicha Service.spec.ports dan biriga mos kelishi kerak; targetPort esa konteyner port nomi yoki raqami.

   ```python
   ob = {(h["kind"], h["metadata"]["name"]): h for h in hujjatlar}
   ```
6. mcpu() va mib() yordamchilari birliklarni bir xil o'lchovga keltiradi. requests > limits → K03 FAIL.
7. safe_dump_all → safe_load_all aylanmasi YAML'ga o'tmaydigan Python turlari (masalan, tuple) yo'qligini ko'rsatadi. Buzilganda 7 qoida yiqilishi kerak.

Bajargach o'ylang: Namunadagi tekshir() bilan solishtiring: qaysi qoidalar faqat bitta obyektga, qaysilari bir nechta obyekt orasidagi moslikka tegishli? kubeconform yoki kube-linter bu ishning qaysi qismini bajara olmaydi?

## 5-qadam. Versiyani chiqarish: RollingUpdate simulyatsiyasi va haqiqiy so'rovlar bilan canary

1. Sig'im talabdan kam bo'lsa yo'qotish boshlanadi: 5 × 50 = 250 < 260. Demak bitta Pod ham kam bo'lsa — yo'qotish bor.
2. Avval start tugagan Pod'lar tayyor bo'ladi; keyin eski Pod mavjudlik ≥ N − unav saqlansa o'chiriladi; keyin jami ≤ N + surge bo'lguncha yangi Pod'lar yaratiladi.
3. Recreate: eski Pod'lar birdaniga 0 ga tushadi, yangilari 30 s start oladi. Blue/green: yo'qotish 0, lekin maksimal Pod soni 12 — ikki baravar resurs.
4. Recreate: 30 s × 260 = 7800. maxUnavailable=1: har bosqichda 30 s davomida 10 so'rov/s yetishmaydi (6 bosqich × 30 × 10).
5. rng.random()·100 < w bo'lsa v2. Kam namunada xato ulushi va p95 juda shovqinli — 8 % xatoni 0,5 % dan ishonchli ajratib bo'lmaydi.

   ```python
   ver = "v2" if rng.random() * 100 < w else "v1"
   ```
6. Xato: v2 > v1 + 2 p.p.; kechikish: p95_v2 > max(1,5·p95_v1, p95_v1 + 2 ms). Juda kichik p95 da 1,5× tasodifiy shovqindan ham oshib ketadi.

   ```python
   np.percentile(latlar, 95)
   ```
7. Kutilgan: nuqsonli va sekin v2 5 % bosqichdayoq orqaga qaytariladi. 100 % bosqichda v1 namunasi yo'q — baza oldingi bosqichdan olinadi.

Bajargach o'ylang: Namunadagi rolling() bilan solishtiring: eski Pod'larni o'chirish va yangilarini yaratish tartibini almashtirsangiz natija o'zgaradimi? Argo Rollouts canary tahlilida yana qaysi metrikalarni qo'shardingiz?

## 6-qadam. HPA (autoscaling/v2) + Cluster Autoscaler + Erlang-C: yuklama simulyatsiyasi

1. ceil(4 × 90/70) = 6. |nisbat − 1| ≤ 0,1 bo'lsa joriy son saqlanadi. Bir nechta metrika bo'lsa — tavsiyalar maksimumi olinadi.
2. cpu_util = 100·min(λ/tayyor·XIZMAT, 1) — Pod 100 % dan ortiq ishlay olmaydi, shuning uchun HPA bir qadamda ~1,4× o'sadi. RPS (λ/tayyor) esa haqiqiy talabni ko'radi.
3. scaleUp: 15 s da max(joriy, 4) gacha qo'shish. scaleDown: oxirgi 300 s dagi eng katta tavsiyadan pastga tushmaslik va 60 s da ≤ 10 % kamaytirish.
4. Pending. ceil(pending / 3) tugun kerak (yo'ldagilarni ayiring), tugun_max dan oshmasin; tugun 120 s da tayyor. Ortiqcha tugun 600 s ketma-ket bo'lsa olib tashlanadi.
5. a = λ/μ, ρ = λ/(cμ). C = [aᶜ/c!/(1−ρ)] / [Σₖ<c aᵏ/k! + aᶜ/c!/(1−ρ)]. Kutish p95 ≈ ln(C/0,05)/(cμ − λ) (C ≤ 0,05 bo'lsa 0) + xizmat p95 ≈ 3·20 ms.

   ```python
   a = lam / mu; rho = lam / (c * mu)
   ```
6. 4 tugun × 3 Pod × 50 = 600 so'rov/s, sakrashda talab ~770 so'rov/s — 10 daqiqa davomida ~170 so'rov/s yetishmaydi. Bulut CA bilan buzilish ~1,5–5 daqiqa.

Bajargach o'ylang: Namunadagi sim() bilan solishtiring: siz Pending Pod'larni scaleDown paytida birinchi bo'lib bekor qildingizmi? «Oldindan 6 tugun, min 8» varianti nega eng kam buzilish beradi va uning narxi qanday?

## 7-qadam. Operatsion hisobot: SLO, xato byudjeti, burn-rate ogohlantirishlari va Go/No-Go

1. Byudjet = (1 − SLO) × jami so'rovlar — ruxsat etilgan xatolar soni. Sarf = jami xato / byudjet; ~84 % kutiladi.
2. Burn rate = oynadagi xato ulushi / (1 − SLO). 1x — byudjet aynan 30 kunda tugaydi; 14,4x da 1 soatda byudjetning 2 % i yonadi (30 kun / 14,4 ≈ 2 kun).
3. Kumulyativ yig'indi: oyna yig'indisi = cs[i] − cs[i − oyna]. Boshida 0 qo'shing va indeksni 0 dan pastga tushirmang.

   ```python
   cs = np.cumsum(np.r_[0, sorov])
   ```
4. Uzun oyna — ahamiyatlilik, qisqa oyna — muammo hozir ham davom etyaptimi. Faqat uzun oynada ogohlantirish tuzatilgandan keyin ham soatlab yonib turadi.
5. Insident boshidan keyingi oralikdagi birinchi True indeks (np.flatnonzero). Keskin — 5 daqiqada; sekin — faqat 6 soatlik va 3 kunlik qoidalarda.
6. kind, alert/expr/for/labels/annotations, severity ∈ {page, ticket}, qavslar balansi, 'for' = raqam + s/m/h, takroriy nomlar. Natija: VALID (5 qoida). Tungi chaqiruvda navbatchi nima qilishini bilishi kerak.
7. np.polyfit(kunlar, cho'qqi, 1) → a (o'sish/kun), b. Qoldi = (SIGIM − b)/a − 29; ~45 kun kutiladi. Barcha bandlar True bo'lsa — GO.

   ```python
   a, b = np.polyfit(np.arange(30), cho, 1)
   ```

Bajargach o'ylang: Namunadagi burn() va qoidalarni o'zingiznikiga solishtiring. Agar SLO 99,99 % qilinsa, xuddi shu ikki insident byudjetga qanday ta'sir qiladi va qaror o'zgaradimi?
