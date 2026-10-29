# MT2. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Etalon model va o'lchov metodikasi

1. goal'ga qarang: taymer aniqligi, isinish, takrorlar, foizlar (p50/p99) va CV. Test aniqligi ~0,92 bo'lsa, model yetarli o'qitilgan.
2. perf_counter monoton va nanosekund darajasidagi aniqlikka ega; uning aniqligini time.get_clock_info bilan bilish mumkin.

   ```python
   a = time.perf_counter(); fn(); dt = time.perf_counter() - a
   ```
3. Birinchi chaqiruvda xotira ajratish, kesh/oqimlar havzasi va algoritm tanlash bo'ladi. Sovuq modelda 1-qiymatni, 2–10 va 40–50 oralig'ini alohida solishtiring.
4. p50 — odatdagi holat, p99/p99.9 — eng yomon 1%/0,1% so'rov. Numpy'da foizlar bitta funksiya bilan olinadi.

   ```python
   np.percentile(t, [50, 95, 99, 99.9])
   ```
5. Variatsiya koeffitsienti CV = std / mean. Takrorlar natijasini DataFrame'ga yig'ib, groupby('Rejim') bilan mean va std oling; CV < 1% bo'lsa o'lchov barqaror.
6. Python interpretatori va dispatcher ustamasi kamayadi, freeze konstantalarni grafikka singdiradi. torch.inference_mode() ichida o'lchang.

   ```python
   jit = torch.jit.freeze(torch.jit.trace(model, x1))
   ```

Bajargach o'ylang: Namunada isinish va takrorlar qanday tashkil qilingan — sizning o'lchovingiz bilan p50 va CV qanchaga farq qildi? Agar isinishni olib tashlasangiz, qaysi ustun eng ko'p o'zgaradi?

## 2-qadam. ONNX eksport, grafik optimallashtirish va operatorlar profili

1. torch.onnx.export ga namuna kirish, input_names/output_names, dynamic_axes va opset_version=17 beriladi; so'ng onnx.checker bilan tekshiriladi.

   ```python
   dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}
   ```
2. Har bir tugunning op_type maydoni bor; pandas value_counts bilan sanash qulay. eval'da BN chiziqli almashtirish bo'lib, Conv vazniga qo'shiladi (constant folding).
3. SessionOptions.optimized_model_filepath ga fayl nomi bersangiz, sessiya optimallashtirilgan grafikni saqlaydi; uni onnx.load bilan ochib operatorlarni sanang. Conv+Relu → FusedConv.

   ```python
   so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   ```
4. intra_op_num_threads = 1, 50 ta isinish, 500 ta o'lchov va p50/p99 — 1-qadamdagi metodika.
5. Kichik modelda vaqtning katta qismi hisob emas, balki Python va dispatcher ustamasi. Eager p50 ni ORT ALL p50 ga bo'ling.
6. enable_profiling=True, so'ng end_profiling() JSON fayl yo'lini qaytaradi; cat == 'Node' va nomi '_kernel_time' bilan tugaydigan hodisalarning args.op_name va dur maydonlari kerak. Kutilgan: Conv/FusedConv ~80%.

   ```python
   prof = s.end_profiling(); ev = json.load(open(prof))
   ```

Bajargach o'ylang: Sizning tugunlar soni va profil ulushlaringiz namuna bilan mos keldimi? Agar eng og'ir operatorni 2 marta tezlashtirsangiz, umumiy vaqt Amdahl bo'yicha qanchaga kamayadi?

## 3-qadam. Hisoblash aniqligi formatlari: FP32, FP16/BF16, INT8 dinamik va statik

1. Jadval ustunlari: hajm, aniqlik, FP32 bilan moslik, max|Δlogit|, KL, batch 1 p50, batch 64 inf/s. Aniqlik ~0,92 da tasodifan saqlanib qolishi mumkin, moslik va KL sezgirroq.
2. Har bir initializer'ni numpy massivga aylantirib, astype(np.float16).astype(np.float32) qiling va qaytadan grafikka yozing.

   ```python
   w = numpy_helper.to_array(init)
   init.CopyFrom(numpy_helper.from_array(w2, init.name))
   ```
3. Dinamik — har chaqiruvda; statik — oldindan kalibrlash to'plamida (bu yerda 320 namuna). CalibrationDataReader.get_next() lug'at yoki None qaytaradi.
4. Percentile chetdagi qiymatlarni kesadi (CalibPercentile 99.99). QDQ — QuantizeLinear/DequantizeLinear juftliklari, QOperator — butun sonli operatorlar. quantize_static ning quant_format va calibrate_method parametrlari.

   ```python
   quantize_static("prep.onnx", out, Kalib(kal), quant_format=QuantFormat.QDQ, per_channel=True, ...)
   ```
5. Softmax ehtimolliklari bo'yicha KL(p_ref || p) = mean Σ p_ref·log(p_ref/p); nolga bo'linmaslik uchun 1e-9 qo'shing. Moslik — argmax'lar teng bo'lgan ulush (kutilgan ≥ 99%).
6. Dinamik kvantlash har chaqiruvda masshtab hisoblaydi — kichik Conv modelda bu ustama katta. Throughput = 64 / median(batch 64 vaqti).

Bajargach o'ylang: Sizning jadvalingizdagi moslik va KL qiymatlari namuna bilan qanchalik mos? Qaysi format bilakuzuk uchun, qaysi biri server uchun maqbul deb o'ylaysiz va nega?

## 4-qadam. Paket va oqim tahlili: S(b) modeli va dinamik paketlash simulyatsiyasi

1. S(b) = c0 + c1·b: c0 — har chaqiruvning qat'iy ustamasi, c1 — bitta namuna narxi. Throughput = b / S(b), uning chegarasi 1/c1.
2. Har batch uchun jami ish taxminan teng bo'lsin: masalan max(30, 600 // b) takror, oldidan 10 ta isinish.

   ```python
   c1, c0 = np.polyfit(S.batch, S["S_p50, ms"], 1)
   ```
3. Eksponensial taqsimot, o'rtachasi 1000/lam ms; kelish vaqtlari — intervallarning kumulyativ yig'indisi.

   ```python
   kel = np.cumsum(r.exponential(1000.0 / lam, n))
   ```
4. Boshlanish = max(ishchi bo'shagan payt, navbatdagi so'rov kelgan payt); muddat = kel[i] + max_wait. Chegaragacha kelganlarni np.searchsorted bilan toping, max_b dan oshirmang; tugash = launch + c0 + c1·(paket).
5. Har yuklama x siyosat uchun p99 ni SLO bilan solishtirib, 'ha' bo'lganlar ichida maksimal yuklamani oling.
6. c0/c1 nisbatiga qarang: qat'iy ustama katta bo'lsa, uni ko'p namunaga bo'lish foydali. CPU'da c0 ~0,02 ms — juda kichik.

Bajargach o'ylang: Sizning simulyatsiyangiz namunadagi bilan bir xil «maksimal yuklama» raqamlarini berdimi? Farq bo'lsa — paket jo'natish qoidasida yoki ishchi bo'shash vaqtida nimani boshqacha qildingiz?

## 5-qadam. Parallellik: intra-op oqimlar, sessiyalar va LATENCY/THROUGHPUT rejimlari

1. intra_op_num_threads — bitta operator ichida (kechikish ↓); bir nechta sessiya/oqim — so'rovlar parallel (throughput ↑). OpenVINO'da bu LATENCY va THROUGHPUT hint'lari.
2. Oqimlarni sinxronlash ustamasi hisobdan katta bo'lsa, parallellik foydasiz. SessionOptions'da intra/inter_op_num_threads va execution_mode bor.

   ```python
   so = ort.SessionOptions(); so.intra_op_num_threads = intra
   ```
3. Umumiy hisoblagich + threading.Lock; k-ishchi sessiyalar[k % len(sessiyalar)] ni oladi. ORT run() GIL'ni qo'yib yuboradi, shuning uchun oqimlar haqiqatda parallel ishlaydi.

   ```python
   with ThreadPoolExecutor(ishchi) as ex: list(ex.map(ish, range(ishchi)))
   ```
4. inf/s = (chaqiruvlar soni × batch) / umumiy devor vaqti; kechikish foizlari esa alohida chaqiruv vaqtlaridan.
5. intra × ishchi > yadro bo'lsa oqimlar navbat kutadi va kontekst almashadi. Kutilgan: eng past kechikish — B (intra=2), eng yuqori throughput — D (2 sessiya x 1 oqim, ~2x).
6. Har bir ORT konfiguratsiyasini bitta qatorda ikkita analog bilan pandas DataFrame jadvali sifatida chiqaring; 4-qadamdagi dinamik paketni ham qo'shing.

Bajargach o'ylang: Sizning eng yaxshi konfiguratsiyalaringiz namunadagi bilan mos keldimi? Real serverda 8 yadro bo'lsa, sessiyalar va intra oqimlar sonini qanday tanlardingiz?

## 6-qadam. Energiya samaradorligi: J/inferens, EDP, DVFS va quvvat rejimlari

1. P = P_bosh + (P_max − P_bosh)·bandlik; bandlik = CPU vaqti / (devor vaqti × yadro). CPU vaqti — time.process_time(), devor vaqti — time.perf_counter().

   ```python
   c0, w0 = time.process_time(), time.perf_counter()
   ```
2. To'liq: E = P·devor_vaqti / n_inf; dinamik: (P_max − P_bosh)·CPU_vaqti / yadro / n_inf. inf/J = 1/E, EDP = E × kechikish.
3. Kutilgan: eng kam energiya — INT8 statik, 1 oqim, batch 32 (~500 µJ/inf, taxminiy).
4. P_din ~ f·V², kuchlanish ham f bilan pasayadi; kechikish ~ 1/f, demak ish vaqti uzayadi va statik quvvat uzoqroq sarflanadi. Bandlik = min(yuk × kechikish / 1000, 1).

   ```python
   Pd = P_DYN_MAX * (F / F[-1]) * (V / V[-1]) ** 2
   ```
5. Energiya = bandlik·(P_stat + P_din) + (1 − bandlik)·P_uyqu; SLO dan oshgan chastotalarni np.inf bilan chiqarib tashlab argmin oling. Kutilgan optimum ~1,2 GHz, statik quvvat oshsa yuqoriga siljiydi.
6. Talabni qanoatlantirganlarni filtrlab, inf/J = inf_s / quvvat_W bo'yicha maksimalini tanlang.

Bajargach o'ylang: Namunadagi energiya jadvali bilan sizniki qanday farq qildi? Quvvat modeli faraziy ekanini hisobga olsak, xulosalardan qaysi biri real o'lchovda ham saqlanib qolishi ehtimoli yuqori?

## 7-qadam. Yakuniy Pareto tahlili, ssenariy bo'yicha tanlov va TensorRT/OpenVINO yo'l xaritasi

1. Aniqlik (max), p99 batch 1 (min), throughput batch 32 (max), µJ/inferens (min), hajm (min). Barchasi bir xil metodika bilan o'lchanishi kerak.
2. Batch 1 da isinish + 400 chaqiruvdan p99; batch 32 da k chaqiruv uchun process_time va perf_counter; uJ = P(bandlik)·devor / (k·batch)·1e6.
3. Max mezonlarni manfiy ishora bilan min ga aylantiring; j konfiguratsiya i ni dominatsiya qiladi, agar barcha mezonda ≤ va kamida bittasida < bo'lsa.

   ```python
   (v[j] <= v[i]).all() and (v[j] < v[i]).any()
   ```
4. Bilakuzuk: hajm ≤ 20 KB, aniqlik ≥ FP32 − 1 p.p., energiya min; server: aniqlik ≥ FP32 − 0,5 p.p., throughput max; real vaqt: p99 min. Filtr bo'sh qolsa, buni alohida xabar qiling.
5. O'lchov shovqini (CV), oqimlar soni va CPU turi natijaga ta'sir qiladi; raqamlar taxminiy.
6. trtexec (--fp16/--int8, --dumpProfile), ovc + benchmark_app (-hint latency|throughput), NNCF, nvidia-smi / tegrastats / RAPL.

Bajargach o'ylang: Pareto ro'yxatingiz va ssenariy tanlovlaringiz namunadagi bilan bir xilmi? Agar to'rtinchi ssenariy (masalan, smartfon ilovasi) qo'shilsa, unga qanday cheklov va maqsad funksiyasi berardingiz?
