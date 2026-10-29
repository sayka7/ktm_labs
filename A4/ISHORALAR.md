# A4. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. FastAPI inferens xizmati: probe'lar, predict, metrikalar

1. Liveness (jarayon tirikmi) va readiness (so'rov qabul qila oladimi) ikki xil savol. /readyz state['ready'] ga qarab 503 qaytarishi kerak.

   ```python
   raise HTTPException(503, "model not loaded")
   ```
2. lifespan ichida yield'dan oldin InferenceSession yarating va np.zeros((1, 64)) bilan bir marta run qiling (isinish).

   ```python
   so = ort.SessionOptions(); so.intra_op_num_threads = ORT_THREADS
   ```
3. O'qitishda X/16 ishlatilgan; ONNX kirishi (batch, 64) — bitta namuna uchun [None] bilan o'lcham qo'shing.
4. time.perf_counter() ni uch marta oling (t0, t1, t2); argsort() oxirgi 3 tasini teskari tartibda oling.

   ```python
   top = logits.argsort()[-3:][::-1]
   ```
5. Har metrika oldidan '# TYPE nom tur' qatori; kvantillar infer_latency_ms{quantile="0.95"} ko'rinishida, np.percentile bilan.
6. Kutilgan: oldin 503, keyin 200; aniqlik ~90 %; noto'g'ri uzunlik va yaroqsiz JSON — 422 (pydantic yoki o'zingizning tekshiruv).

Bajargach o'ylang: Namunada /readyz va /healthz qanday ajratilgan va metrika matni qanday tuzilgan — sizning variantingiz bilan solishtiring. Xatolar hisoblagichini qayerda oshirish to'g'riroq: o'z tekshiruvingizdami yoki pydantic 422 da ham?

## 2-qadam. Yuklama sinovi: ketma-ket va parallel so'rovlar, Little qonuni, paketlash

1. Kvantil — so'rovlarning 95 % (99 %) i shu vaqtdan tezroq bajarilgan chegara. np.percentile ni ms ga o'tkazilgan massivga qo'llang.

   ```python
   t = np.array(t) * 1e3; np.percentile(t, 95)
   ```
2. c=1 da so'rov/s ≈ 1 / o'rtacha kechikish. Javobdagi inference_ms ni to'liq vaqtdan ayirsangiz HTTP, JSON va pydantic ulushi chiqadi.
3. Har bir oqim bitta so'rov yuborib uning vaqtini qaytarsin; ex.map bilan n ta so'rov; throughput = n / wall.

   ```python
   with ThreadPoolExecutor(cc) as ex: tt = list(ex.map(bitta, range(n)))
   ```
4. X — throughput (so'rov/s), R — o'rtacha kechikish (s). Yopiq tizimda N ≈ parallel mijozlar soni bo'lishi kerak (little_N ustuni).
5. HTTP/JSON qo'shimcha xarajati har so'rovga bir marta to'lanadi; matritsa ko'paytmasi paketda samarali. Narxi — paket to'lguncha kutish (kechikish).
6. Kutilgan: throughput tez to'yinadi (2 yadro, GIL), p99 esa mijozlar bilan deyarli chiziqli o'sadi — chunki faqat navbat uzayadi.

Bajargach o'ylang: Namunadagi little_N ustuni va sizning hisobingiz mos keldimi? Real `hey -c N` yoki locust natijalari TestClient + oqimlardan nimasi bilan farq qilishi mumkin?

## 3-qadam. Dockerfile tahlili va tasvir hajmini baholash

1. Faqat oxirgi FROM dan keyingi satrlar tasvirga kiradi. FROM satrlari ro'yxatini oling va oxirgisining indeksidan kesib oling.

   ```python
   froms = [l for l in satrlar if l.upper().startswith("FROM ")]
   ```
2. Tasvir nomi ':latest' bilan tugasa yoki unda ':' umuman bo'lmasa — qotirilmagan. Yengil asos uchun 'slim', 'runtime', 'alpine', 'distroless' so'zlarini qidiring.
3. 'requirements' bor birinchi satr indeksi 'COPY app/' yoki 'COPY .' satri indeksidan kichik bo'lishi kerak; re.match bilan COPY satrini toping.
4. Exec shaklida ilova PID 1 bo'ladi va K8s yuborgan signalni to'g'ridan oladi; tekshiruv: satr 'CMD [' bilan boshlanadimi.
5. MB × 8 = megabit; megabit / 100 = soniya. 20 tugun: MB × 20 / 1024 GB.
6. Kutilgan: yaxshi 11/11, yomon 1/11; ~270 MB vs ~8 GB (~30x); tortish ~22 s vs ~11 daqiqa.

Bajargach o'ylang: Yomon variant qaysi yagona qoidadan o'tdi va nega? Namunadagi qoidalar bilan sizniki farq qilsa, qaysi biri haqiqiy Dockerfile'larda (hadolint kabi) ishonchliroq ishlaydi?

## 4-qadam. Kubernetes manifestlari: pyyaml bilan validatsiya va 18 qoida

1. yaml.safe_load_all generator qaytaradi; bo'sh hujjatlarni tashlab, {kind: ob'ekt} lug'atini tuzing.

   ```python
   obj = {d["kind"]: d for d in yaml.safe_load_all(text) if d}
   ```
2. Millicore: 'm' bilan tugasa sonning o'zi, aks holda × 1000. '2' -> 2000, '1500m' -> 1500.
3. Moslik — lug'atlar tengligi. Xatoli manifestda Pod label 'digits', selector esa 'digits-infer'; targetPort 9000 ham containerPort 8000 ga mos emas.
4. Liveness erta tekshirsa, model yuklanayotgan Pod 'o'lik' deb qayta-qayta o'ldiriladi. Qoida: liveness.initialDelaySeconds >= readiness + 10.
5. Ichma-ich .get(..., {}) zanjirini ishlating — xatoli manifestda ko'p kalit yo'q va KeyError bo'lmasligi kerak (strategy, securityContext, behavior, annotations).
6. Foiz requests.cpu ga nisbatan: 1500 × 60 / 100. Kerakli nusxa = ceil(joriy × joriy_foiz / nishon).

   ```python
   -(-2 * 95 // 60)   # butun sonli ceil
   ```
7. Kutilgan: 18/18 va 2/18; chegara 900m; 2 nusxa 95 % da -> 4 nusxa.

Bajargach o'ylang: Xatoli manifest qaysi 2 qoidadan o'tdi? Sizning tekshiruvlaringiz namunadagidan qayerda qat'iyroq yoki yumshoqroq — kubeconform/kube-linter kabi vositalar yana nimani ushlaydi?

## 5-qadam. HPA avtomasshtablash simulyatsiyasi (navbat modeli)

1. μ = 1000 / 14 ≈ 71 so'rov/s. Kerakli nusxa = ceil(λ / μ / (nishon/100)).
2. navbat = max(0, navbat + λ − sig'im), sig'im = tayyor × μ. CPU foizi = min(λ/sig'im, 1) × 100.

   ```python
   navbat = max(0.0, navbat + lam - sigim)
   ```
3. kutish = navbat / sig'im (s) → ms. p95 ≈ xizmat vaqti + kutish (+ CPU > 90 % da empirik qo'shimcha).
4. Bu tebranishdan (flapping) himoya: qisqa sakrashga keskin javob bermaslik va erta qisqartirmaslik. tavsiyalar ro'yxatida (t, istalgan) juftlarini saqlang.
5. Yangi Pod'larni 'tayyor bo'lish vaqti' t + start_s bilan kutilayotgan ro'yxatiga qo'shing va har qadamda vaqti kelganlarini tayyor ga o'tkazing.

   ```python
   kutilayotgan += [t + start_s] * (xavfsiz - jami)
   ```
6. pod_soniya = har soniyadagi tayyor nusxalar yig'indisi; narx_nisbati = pod_soniya / asosiy pod_soniya.
7. Kutilgan: reaksiya ~65 s, SLO ~200 s buziladi; minReplicas 4 buzilishni ~4x kamaytiradi, narx +23 %; maxReplicas 3 — eng yomon.

Bajargach o'ylang: Namunadagi simulyatsiya bilan reaksiya vaqti va SLO buzilgan soniyalar bir xil chiqdimi? Farq bo'lsa, u qaysi qadam tartibidan (Pod tayyorlanishi, navbat, HPA qarori) kelib chiqqan?

## 6-qadam. Cloud vs Edge vs Gibrid: kechikish, trafik, energiya va TCO kalkulyatori

1. 16 × 10 = 160 inferens/s; soat = yil × 365 × 24. GPU soati va egress — chiziqli; Jetson qurilmasi va korpus — kapital (oldindan).
2. max(zaxira_nusxa, ceil(inf_s / sig'im)). 160 inferens/s da sig'im yetarli bo'lsa ham zaxira GPU'lar to'lanadi.

   ```python
   gpu_n = max(S["zaxira_nusxa"], math.ceil(inf_s / S["gpu_sigim_inf_s"]))
   ```
3. Mbit/s / 8 = MB/s; / 1024 = GB/s; × 3600 × soat. Keyin × egress narxi va (1 + qo'shimcha_foiz/100).
4. Cloud: RTT + navbat + inferens; Edge: faqat qurilmadagi inferens vaqti.
5. W = J/s; (quvvat × qurilmalar) / inf_s × 1000 = mJ/inferens; bulutda PUE ga ko'paytiring.
6. Yangi lug'at: asosiy S ustiga o'zgarishlar yoziladi.

   ```python
   d = hisob({**S, **kw})
   ```
7. Kutilgan: TCO 30–50 marta, kechikish ~6 marta; −30 % narxda ham nisbat deyarli saqlanadi; juda kam kamerada kesishish bo'lishi mumkin.

Bajargach o'ylang: Kesishish nuqtasi natijasi sizni ajablantirdimi? Namunadagi gibrid model taxminlari (GPU ulushi, egress/10) bilan sizniki farq qilsa, qaysi taxmin real hisob-fakturaga yaqinroq?
