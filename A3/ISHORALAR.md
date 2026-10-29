# A3. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Etalon: PyTorch FP32 o'qitish va ONNX eksport

1. FP32 modelning test aniqligi (~94 %) va batch 1 dagi p50 kechikishi — barcha kvantlash/optimallashtirish shu bilan solishtiriladi.
2. Gradientni nollash, oldinga o'tish + loss, orqaga tarqalish, qadam. Baholashda model.eval() — BatchNorm o'rtacha statistikadan foydalanadi.

   ```python
   opt.zero_grad(); loss = lossf(model(xb), yb)
   loss.backward(); opt.step()
   ```
3. Birinchi chaqiruvlarda kesh/xotira ajratish sekin; taqsimot dumi (p95) foydalanuvchi sezadigan eng yomon holatni ko'rsatadi.

   ```python
   a = time.perf_counter(); model(x1); t.append(time.perf_counter() - a)
   np.percentile(t, 95)
   ```
4. Kirish shakli 1x1x8x8 ga qotib qoladi. Paket o'qiga 'batch' nomini bering — input va logits uchun.

   ```python
   dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
   ```
5. BN eval rejimida chiziqli amal — uni Conv vazni va biasiga 'birlashtirish' mumkin. Kutilgan: ~10 tugun.
6. Suzuvchi nuqta amallari tartibi farq qiladi; ~1e-6 farq normal. rtol=1e-3, atol=1e-4 bilan np.allclose.

   ```python
   sess.run(None, {"input": x.numpy()})[0]
   ```

Bajargach o'ylang: Sizning o'qitish va o'lchash kodingiz namunadan nimasi bilan farq qiladi? Eksportda qaysi parametr (dynamic_axes, do_constant_folding) tugunlar soniga ta'sir qilganini namunadagi natija bilan solishtiring.

## 2-qadam. ONNX Runtime: optimallashtirish darajalari, paket, oqimlar, Little qonuni

1. Optimallashtirish darajasi va oqimlar — kechikishga; paket o'lchami — asosan throughput'ga; Little qonuni — rejalashtirishga.
2. SessionOptions.optimized_model_filepath; keyin onnx.load(...).graph.node uzunligi. Conv+ReLU kabi birlashtirishlar (fusion) — 7–8 tugun.

   ```python
   so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   so.optimized_model_filepath = 'opt_ALL.onnx'
   ```
3. Kichik modelda vaqtning asosiy qismi Python va dispetcherlik xarajati; ORT grafikni oldindan tuzib, bu xarajatni yo'qotadi.
4. X = b / T; N = X · T = b — sinxron siklda tizimdagi so'rovlar soni paket o'lchamiga teng.

   ```python
   thr = b / (r['mean'] / 1e3)
   ```
5. Sinxronizatsiya xarajati va kichik model — ishni teng bo'lish qiyin; batch 256 da sezilarli, batch 1 da deyarli yo'q.
6. N = λ · R (R sekundda). Yarim nusxa bo'lmaydi — math.ceil.

Bajargach o'ylang: Sizning jadvalingizdagi tugunlar soni va ORT/PyTorch tezlanishi namunadagi bilan mos keldimi? Little qonuni hisobida qaysi kechikishni (p50, p99) olganingiz rejaga qanday ta'sir qildi?

## 3-qadam. PyTorch'da INT8 dinamik kvantlash (Linear)

1. Vaznlar oldindan INT8; faollashuvlar har chaqiruvda kvantlanadi — kalibrlash kerak emas. Vaqt asosan MatMul'da ketadi.
2. torch.ao.quantization.quantize_dynamic — qatlam turlari to'plami va dtype beriladi; chop etilganda DynamicQuantizedLinear ko'rinadi.

   ```python
   quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
   ```
3. s = max|w| / 127; q = round(w/s), clamp(-128, 127); xato ≤ s/2.

   ```python
   q = torch.round(w / s).clamp(-128, 127)
   ```
4. io.BytesIO ga torch.save(state_dict) va tell(). Bias va masshtablar FP32 da qoladi.

   ```python
   b = io.BytesIO(); torch.save(m.state_dict(), b); b.tell()
   ```
5. INT8 tezlanishi VNNI/AMX buyruqlariga bog'liq; bu sandbox CPU'da ular yo'q — A3 dagi 5-xato namunasi.

Bajargach o'ylang: Namuna bilan solishtiring: qo'lda hisoblangan s va tiklash xatosi PyTorch kvantlagan natija bilan qanchalik mos? VNNI'li serverda jadvaldagi qaysi ustun o'zgarishini kutasiz?

## 4-qadam. ONNX Runtime'da kvantlash: dinamik va statik (MinMax, Percentile, shovqin)

1. Statikda faollashuvlar masshtabi oldindan hisoblanadi — buning uchun kalibrlash to'plami (real namunalar) kerak.
2. CalibrationDataReader.get_next() har gal {kirish_nomi: massiv} qaytaradi, oxirida None.

   ```python
   def get_next(self):
       return next(self.it, None)
   ```
3. QDQ — QuantizeLinear/DequantizeLinear juftlari; per-channel — har chiqish kanaliga alohida s; Percentile chetdagi 'outlier' qiymatlarni kesadi.

   ```python
   quantize_static(src, dst, Reader(kalib), quant_format=QuantFormat.QDQ,
       per_channel=True, calibrate_method=CalibrationMethod.Percentile)
   ```
4. s juda katta bo'ladi — real qiymatlar bir necha INT8 pog'onasiga tiqiladi; ~1 p.p. yo'qotish kutiladi (A3 matnidagi 3-xato).
5. onnx.load(f).graph.node dan op_type ni olib, QuantizeLinear/DequantizeLinear/DynamicQuantizeLinear larni sanang.
6. Kichik modelda grafik tuzilmasi, Q/DQ tugunlari va masshtablar ulushi katta; real kalibrlashda ±0,3 p.p.

Bajargach o'ylang: Sizning kalibrlash o'quvchingiz va variantlar jadvali namunadan nimasi bilan farq qiladi? Shovqinli kalibrlash qancha p.p. yo'qotdi va real loyihada bunday xatoni qanday oldini olasiz?

## 5-qadam. Yakuniy jadval, Pareto tahlili va grafik

1. Bitta test to'plami, intra_op 2 oqim, batch 1 da p50 va batch 256 da throughput — hamma uchun bir xil o'lchov funksiyasi.
2. Etalon — PyTorch FP32 p50; tezlanish = p50_etalon / p50_variant.

   ```python
   df["tezlanish"] = (df.p50_ms[0] / df.p50_ms).round(2)
   ```
3. A ikkala ko'rsatkichda ham yomon emas (≤ latency, ≥ aniqlik) va kamida bittasida qat'iy yaxshiroq bo'lishi kerak.
4. Mantiqiy maskalarni & va | bilan birlashtirib, .any() oling; ustun yo'q bo'lsa — Pareto.

   ```python
   ustun = ((df.p50_ms <= r.p50_ms) & (df.aniqlik >= r.aniqlik) & (...)).any()
   ```
5. PyTorch va ORT orasida 4–7 marta farq bor — chiziqli shkalada ORT nuqtalari bir joyga yopishadi. ax.set_xscale('log'), Pareto uchun boshqa rang.
6. Aniqlik bo'yicha boshqalardan qat'iy yuqori bo'lsa, sekin bo'lsa ham Pareto'da qoladi — aniqlik farqlarini tekshiring.

Bajargach o'ylang: Sizning Pareto funksiyangiz namunadagi bilan bir xil to'plamni berdimi? Tibbiy tasniflash va chekka qurilma uchun qaysi konfiguratsiyani tanlardingiz va nega?

## 6-qadam. Energiya bahosi, nusxalar rejasi va GPU/TensorRT yo'l xaritasi

1. Chiziqli model: P = P_bosh + (P_to'liq − P_bosh)·bandlik; bandlikni CPU vaqti / devor vaqti dan olasiz.
2. process_time — jarayonning barcha oqimlari CPU vaqti yig'indisi; 2 yadroda u devor vaqtining 2 baravarigacha yetadi, shuning uchun 2 ga bo'linadi.

   ```python
   bandlik = min(cpu / wall / 2, 1.0)
   ```
3. E = P · wall / n · 1e6 (n = batch·takror). Dinamikda P o'rniga P − P_bosh. TCO uchun to'liq — bo'sh quvvat ham pul turadi.
4. Doimiy chaqiruv xarajati paketga taqsimlanadi. Batch 1 da 2-oqim tezlik bermaydi, lekin bandlikni (ya'ni quvvatni) oshiradi.
5. Little: N = λ·R, nusxa = ceil(N), quvvat ≈ nusxa · P. Kechikish 100 marta katta bo'lsa, nusxalar ham shuncha ko'p.

   ```python
   N = lam * lat / 1e3; math.ceil(N) * P
   ```
6. Energiya/rasm = P_o'rtacha / throughput — throughput oshgani sari energiya kamayadi. trtexec --fp16 / --int8, nvidia-smi --query-gpu=power.draw.

Bajargach o'ylang: Sizning energiya modelingiz namunadagi bilan solishtirganda qanday farq qiladi? P_TOLIK_W ni o'zgartirsangiz, eng tejamkor konfiguratsiya o'zgaradimi yoki faqat mutlaq raqamlar?
