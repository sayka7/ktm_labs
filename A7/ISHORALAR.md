# A7. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Ma'lumot sxemasi validatsiyasi va bemor bo'yicha bo'lish

1. Bir bemor tasvirlari bir-biriga juda o'xshash (bir xil yashirin z omili). Bo'lish birligi tasvir emas, bemor bo'lishi kerak.
2. SXEMA lug'atidagi kalitlar: dtype, min, max, isin, regex, unique. Har biri alohida tekshiruv; ustun yo'q bo'lsa qolganini tekshirib bo'lmaydi.

   ```python
   if "max" in q and (s > q["max"]).any():
       xatolar.append(f"{col}: ... > {q['max']}")
   ```
3. Yosh: max va min — ikki alohida satr; jins va finding — isin; fayl nomi — regex; NaN — bo'sh qiymat. Jami 6.
4. split(X, groups=...) — guruh to'liq bir tomonga tushadi. Natija pozitsion indeks, shuning uchun iloc ishlating.

   ```python
   tr_idx, rest_idx = next(gss.split(df, groups=df["patient_id"]))
   ```
5. Har split'dagi patient_id larni set() qiling va isdisjoint bilan solishtiring; assert xato bo'lsa skript to'xtaydi.
6. Kamida 2 tasvirli bemorlarni hisoblang: ularning tasvirlari ikki tomonga tushish ehtimoli 1 - 0.7^k - 0.3^k.
7. pd.cut bilan guruhlab, pd.crosstab(age_group, split) chiqaring; kichik guruhda metrika ishonchsiz bo'ladi.

Bajargach o'ylang: Namunadagi validate() funksiyasi sizning variantingizdan qaysi tekshiruvlar tartibi bilan farq qiladi? Sizish testini CI'da avtomatik bajariladigan qilib qanday ko'chirasiz?

## 2-qadam. Model o'qitish, eksperiment jurnali, model reestri va promote darvozasi

1. SLO recall >= 0.90 — avval sezgirlik ta'minlanadi, shu shart ichida aniqlikni (precision) iloji boricha saqlash uchun eng katta chegara olinadi.
2. Test faqat yakuniy, «ko'rilmagan» baho uchun. Chegara ham giperparametr.

   ```python
   for t in np.arange(0.95, 0.01, -0.01):
       if recall_score(y, p >= t) >= recall_min: ...
   ```
3. meta.sex.values == 'M' niqobi bilan; TPR = recall. Farqning moduli SLO tpr_gap_max (0.05) bilan solishtiriladi.
4. hashlib.sha256(open(yol,'rb').read()).hexdigest() — fayl baytlaridan; yuklashdan oldin qayta hisoblab solishtiriladi.
5. SLO buzilsa — rejected. SLO o'tsa, lekin joriy production val AUC yuqoriroq — staging qoladi. Bir vaqtda faqat bitta production.
6. 2 belgili model (intensity, edge_density) signalga zaif bog'langan, shuning uchun AUC past; reason ustunini tekshiring.

Bajargach o'ylang: Namunadagi promote() bilan solishtiring: «faqat bitta production» qoidasini qanday ta'minlagansiz? Haqiqiy MLflow Model Registry'da bu qaysi tushunchaga mos keladi?

## 3-qadam. PyTorch MLP -> ONNX eksport va onnxruntime bilan inferens tekshiruvi

1. Training-serving skew: server boshqa mu/sd ishlatishi mumkin. register_buffer — parametr emas, lekin state_dict va ONNX grafiga kiradi.

   ```python
   self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
   ```
2. BCEWithLogits sigmoid+BCE ni sonli barqaror birlashtiradi. pos_weight ≈ manfiylar/musbatlar = (1-p)/p.
3. Graf qat'iy batch=1 bilan qotadi. dynamic_axes={'features': {0: 'batch'}, 'logit': {0: 'batch'}}.
4. model(Xte) (no_grad) va sess.run(None, {'features': ...})[0]; float32 hisoblash tartibi farqi ~1e-7 xato beradi.

   ```python
   farq = float(np.abs(out - ref).max())
   ```
5. Isitish (warm-up): xotira ajratish, kesh. Dum (tail) kechikish SLO uchun muhim — np.percentile(t, 99).
6. Kichik batchda Python/dispatcher yuklamasi ustun; katta batchda haqiqiy matematik ish ulushi oshadi.

Bajargach o'ylang: Sizning eksport va parity kodingiz namunadan nimasi bilan farq qiladi? Opset_version'ni o'zgartirsangiz operatorlar ro'yxati o'zgaradimi — tekshirib ko'ring.

## 4-qadam. FastAPI xizmati: /predict, /healthz, Prometheus metrikalari va audit jurnali

1. 401 — autentifikatsiya; 422 — pydantic kontrakti (FastAPI o'zi qaytaradi); 400 — mantiqiy nazorat (|x|>15, sxema bilan bir xil chegara).
2. Field(min_length=..., max_length=...) — OpenAPI hujjatida ham ko'rinadi.

   ```python
   features: list[float] = Field(min_length=6, max_length=6)
   ```
3. Har chelak «<= ub» sonini saqlaydi: x <= ub bo'lgan BARCHA chelaklar oshadi; +Inf = jami son, yana _sum va _count.
4. Kalit — yorliq qiymatlari kortej (tuple); chiqishda name{label="value"} son. Oldidan # HELP va # TYPE satrlari.
5. Maxfiylik (PHI minimallashtirish) va kuzatuvchanlik; jins/yosh — adolatlilik monitoringi (6-qadam).

   ```python
   hashlib.sha256(x.tobytes()).hexdigest()[:16]
   ```
6. JSON parse, pydantic validatsiya, middleware, HTTP qatlam. Foydalanuvchi to'liq vaqtni his qiladi.
7. Avvalgi namunaviy so'rovlar ham sanaladi: 'ok' 301 bo'lib qoladi.

Bajargach o'ylang: Namunadagi /predict bilan solishtiring: metrikalar qaysi joyda yangilanadi — xatolik holatlarida ham kechikish kuzatiladimi? Bu dashboard'ni qanday buzishi mumkin?

## 5-qadam. Monitoring: PSI/KS bilan drift aniqlash va ogohlantirish mantiqi

1. Ma'lumot drifti: kirish belgilari va model ballari taqsimoti mos yozuvdan (validatsiya) qanchalik farq qiladi.
2. Har chelakda mos yozuvning ~10% i bo'ladi; ±inf siljigan qiymatlarni yo'qotmaydi. Bo'sh chelakda log(0) bo'lmasligi uchun kichik epsilon qo'shing.

   ```python
   e = np.quantile(ref, np.linspace(0, 1, bins + 1)); e[0], e[-1] = -np.inf, np.inf
   ```
3. KS — statistik farq bormi (p-qiymat n ga bog'liq); PSI — farq kattaligi (effekt hajmi). Ikkalasi birga ishonchliroq.

   ```python
   ks_p = ks_2samp(ref_scores, sc).pvalue
   ```
4. Avval PSI (ball yoki eng katta belgi PSI) bo'yicha asosiy daraja, keyin qo'shimcha signallar: OK -> WARN, WARN -> ALERT. Sabablar ro'yxatini saqlang.
5. opacity, intensity, texture o'zgaradi; kuch 4-kunda 1/3, 5-kunda 2/3. Expect: 1–4 kun OK, 5-kundan ALERT.
6. recall_score(cur.finding, sc >= THRESHOLD) ni kunlik hisoblang va ALERT kuni bilan solishtiring.

Bajargach o'ylang: Namunadagi qoidalar mantiqini o'zingiznikidan farqi bormi? QOIDA chegaralarini (0.1/0.2) o'zgartirsangiz «yolg'on signal» va «kech signal» o'rtasida qanday murosa paydo bo'ladi?

## 6-qadam. Adolatlilik: fairlearn MetricFrame hisoboti va yumshatish

1. Equal opportunity: kasal bemorlar har guruhda bir xil ulushda aniqlanishi (TPR) — o'tkazib yuborish eng qimmat xato.
2. metrics lug'ati, y_true, y_pred, sensitive_features; .by_group jadval, .difference() farq.

   ```python
   mf = MetricFrame(metrics={...}, y_true=y, y_pred=pred, sensitive_features=sens)
   gap = float(mf.difference()["tpr"])
   ```
3. Faqat y==1 bo'lgan qatorlardan qaytarib tanlash (bootstrap) qiling, 2.5 va 97.5 persentilni oling; n+ ni ham chiqaring.
4. prefit=True, predict_method='predict_proba', constraints='true_positive_rate_parity'; predict'da ham sensitive_features shart.
5. Ko'lam o'zgardi — sezgirlik >= 0.90 shartini qayta ta'minlash kerak; shart ichida eng katta chegara aniqlikni saqlaydi.
6. A kam ma'lumotli guruhga sun'iy chegara qo'yadi; B esa model kartasida «mo'ljallanmagan foydalanish» deb ochiq aytadi.

Bajargach o'ylang: Sizning hisobot() funksiyangiz namunadagidan nimasi bilan farq qiladi? Bootstrap'ni tasvir emas, bemor bo'yicha qilsangiz ishonch oralig'i qanday o'zgarishini taxmin qiling.

## 7-qadam. Loyiha pasporti va model kartasi avtomatik generatsiyasi, rubrika bo'yicha o'z-o'zini baholash

1. Hujjat va artefakt bir-biridan uzilib qoladi. Barcha o'lchovlarni bitta O lug'atiga yig'ib, hujjatni faqat undan yasang; data_sha va onnx_sha moslikni isbotlaydi.
2. sizish, n_img/n_pat, chegara, AUC/sezgirlik/aniqlik, parity, p50/p99/rps, PSI normal/drift, TPR farqlari, 0–18 guruh TPR.
3. train/val/test ni age>=19 bilan filtrlang, lekin bolalar uchun alohida TPR ni «ko'lamdan tashqari» deb hisobotga chiqaring.

   ```python
   tr, va, te = (df[(df.split == s) & (df.age >= 19)] for s in ("train", "val", "test"))
   ```
4. Ikkala tomon: oddiy oqim (val vs test) PSI < 0.1 — yolg'on signal yo'q; drift oqimida >= 0.2 — signal bor.
5. ok=None -> ball // 2 va holat «qo'lda»; o'qituvchi keyin to'ldiradi. SLO buzilgan avtomatik mezon ham yarim ball.
6. Har satr '| № | mezon | dalil | holat | b/m |' ko'rinishida f-string; jami ballar yig'indisi qalin (**) qilib yoziladi.

   ```python
   md += [f"| {i+1} | {n} | {d} | {h} | {b}/{m} |" for i, (n, d, h, b, m) in enumerate(satrlar)]
   ```
7. Chegara, sezgirlik/aniqlik, rubrikadagi holat va jami ball — barchasi O va SLO dan hisoblanadi.

Bajargach o'ylang: Namunadagi pasport bilan o'zingiznikini solishtiring: qaysi o'lchovni unutdingiz yoki ortiqcha qo'shdingiz? Haqiqiy loyihada bu skriptni CI'da har relizda ishga tushirish nimani kafolatlaydi?
