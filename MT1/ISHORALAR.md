# MT1. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Hayot sikli xaritasi, sintetik reytinglar va bazaviy modellar

1. Muammo ta'rifidan boshlab ma'lumot, modellash, offline baholash, onlayn tajriba, boshqa sohalar, monitoring va joylashgacha boring; har biriga qisqa mazmun yozing.
2. Zichlik = reytinglar soni / (foydalanuvchi × film). Siyraklik = 1 − zichlik.

   ```python
   zichlik = len(df) / (n_u * n_i)
   ```
3. Filmlarni reytinglar soni bo'yicha kamayish tartibida saralab, eng mashhur 10% ining ulushini oling (~40–45% kutiladi).

   ```python
   cnt = df.item.value_counts().values
   ```
4. Tasodifiy bo'lish kelajakdagi reytinglarni o'qitishga kiritadi (ma'lumot sizishi). Oxirgi 60 kunni test qiling va sovuq start yozuvlarini chiqarib tashlang.
5. b_i = Σ(r − μ − b_u) / (n_i + λ); keyin b_u ni xuddi shunday — ALS uslubida bir necha marta navbatma-navbat. λ kam reytingli filmlarda siljishni 0 ga tortadi.

   ```python
   g = tr.assign(res=tr.rating - mu - bu[tr.user]).groupby("item").res
   ```
6. Bashoratni [1; 5] oralig'iga kesishni (np.clip) unutmang; regulyarizatsiyasiz model kam ma'lumotli obyektlarda ortiqcha moslashadi.

Bajargach o'ylang: Sizning bias-model va namunadagi navbatma-navbat yangilash bir xil RMSE berdimi? λ ni 0 dan 10 ga o'zgartirish qaysi foydalanuvchilar/filmlar bashoratiga eng ko'p ta'sir qildi?

## 2-qadam. Tavsiya modellari: item-kNN, matrix factorization va offline metrikalar

1. RMSE — ko'rilgan juftliklarda baho xatosi; precision@10/recall@10 — tavsiya ro'yxatiga «yoqqan» filmlar qanchalik tushgani.
2. Qattiqqo'l va yumshoq baholovchilar farqini olib tashlash kerak (adjusted cosine). Diagonalni 0 qiling va har qatorda faqat top-K qo'shnini qoldiring.

   ```python
   S = (C.T @ C) / np.outer(norm, norm)
   ```
3. Faqat u ko'rgan filmlar: o'rtacha + Σ w·c / Σ w, bu yerda w — i bilan o'xshashlik (manfiylari 0), c — u ning shu filmlardagi farqi.
4. Gradient: e·Q[i] − reg·P[u] va e·P[u] − reg·Q[i]; eski qiymatlardan foydalanish uchun ikkalasini bitta tuple tayinlashda yangilang.

   ```python
   e = Y[j] - (mu + bu[u] + bi[i] + P[u] @ Q[i])
   ```
5. Ko'rilganlarga −inf qo'ying, np.argpartition bilan 10 ta oling; coverage = tavsiya qilingan noyob filmlar / 400; novelty = −log2(mashhurlik ulushi).

   ```python
   s[M[u]] = -np.inf; top = np.argpartition(-s, N)[:N]
   ```
6. Offline testda foydalanuvchi faqat ko'rgan filmini baholaydi, ko'rish esa mashhurlikka bog'liq — bu offline baholashning mashhurlik siljishi.
7. RMSE dan oldin bashoratni [1; 5] ga kesing; egri ko'tarila boshlasa — lr yoki reg ni o'zgartirib ko'ring.

Bajargach o'ylang: Sizning item-kNN va MF natijalaringiz namunadagidan farq qildimi? Mashhurlik siljishini hisobga olmasdan qaysi modelni tanlagan bo'lardingiz va bu nega xavfli?

## 3-qadam. Onlayn tajriba: A/B test, statistik quvvat va peeking xatosi

1. haqiqiy matritsa ma'lum: A — hamma uchun bir xil top-10 mashhur, B — xatoli model bahosi bo'yicha shaxsiy top-10. CTR = p_click(haqiqiy baho) ning 10 film bo'yicha o'rtachasi (~+11% o'sish kutiladi).

   ```python
   np.take_along_axis(haqiqiy, topB, 1)
   ```
2. z = (p_B − p_A) / sqrt(p(1−p)·2/n); p-qiymat ikki tomonlama: 2·(1 − Φ(|z|)). CI uchun esa har guruhning o'z dispersiyasi olinadi.

   ```python
   pval = 2 * (1 - stats.norm.cdf(abs(z)))
   ```
3. Effekt bir xil, faqat standart xato 1/sqrt(n) tezlikda kamayadi — bu statistik quvvat masalasi.
4. n = (z_a + z_b)² · (p0(1−p0) + p1(1−p1)) / (p1 − p0)², p1 = p0·(1 + MDE). ~25 ming kutiladi.

   ```python
   za, zb = stats.norm.ppf(1 - alfa/2), stats.norm.ppf(quvvat)
   ```
5. Har qarash — yangi «imkoniyat». Kumulyativ hisoblar (np.cumsum) bilan har qarashdagi z ni toping va «biror marta |z|>1,96» ulushini oling (~0,25).

   ```python
   fp_peek = np.mean((np.abs(zz) > 1.96).any(1))
   ```
6. Lift·(1 + 0,8·e^(−kun/3)) — 1-kunda bo'rttirilgan, 14-kunga haqiqiy qiymatga yaqinlashadi; faqat oxirida tekshirilgan FP ~0,05 bo'lishi kerak.

Bajargach o'ylang: Sizning z-test va kerakli tanlanma natijalaringiz namunadagiga mos keldimi? Agar mahsulot jamoasi har kuni natijaga qarashni talab qilsa, peeking xatosidan qanday himoyalanardingiz?

## 4-qadam. Uber uslubidagi misol: talab prognozi, dinamik narx va haydovchilar tahlili

1. Kunlik (24 soat) va haftalik (168 soat) mavsumiylik: lag24, lag168, soat/kun, siklik sin/cos; yomg'ir va tadbir — tashqi omillar.

   ```python
   df["lag168"] = df.talab.shift(168)
   ```
2. Vaqt qatorida kelajakni o'tmishdan bashorat qilamiz; naiv model = o'tgan haftaning shu soati (lag168). Uni yengmagan model foydasiz.
3. Chiziqli model 8:00 va 18:00 cho'qqilarini son sifatida tushunolmaydi. pd.get_dummies va reindex(columns=..., fill_value=0).

   ```python
   pd.get_dummies(d[X].astype({"soat": "category"}), columns=["soat"])
   ```
4. Maxrajni np.maximum(y, 1) bilan himoyalang.
5. k = clip(1 + 0,6·(prognoz/taklif − 1), 1, 2,5); talab·k^(−0,7); taklif·(1 + 0,25·(k − 1)); bajarilgan = min(talab, taklif). Qondirilmagan ~4700 → ~2300 kutiladi.
6. KMeans Evklid masofasidan foydalanadi — katta masshtabli belgi ustunlik qiladi. StandardScaler, keyin KMeans(3, n_init=10); bitta segmentda bekor ~20% bo'lishi kerak.

Bajargach o'ylang: Sizning surge qoidangiz naiv va boosting prognozida qanchalik farq qildi? Prognoz xatosi yo'lovchi va haydovchi uchun qanday oqibatlarga olib keladi — namunadagi xulosalar bilan solishtiring.

## 5-qadam. Amazon uslubidagi ombor: ABC tahlili, slotting va zaxira siyosati

1. Qiymat = jami dona × narx; kamayish tartibida saralab np.cumsum ulushini oling. ~17% SKU qiymatning 80% ini berishi kutiladi.

   ```python
   tart = np.argsort(-qiymat); ulush = np.cumsum(qiymat[tart]) / qiymat.sum()
   ```
2. Tez aylanadigan SKU — eng yaqin joyga. Baholash oxirgi 30 kunda: o'rganish va test davrini ajratish.
3. joy[argsort(−tezlik)] = argsort(masofa). Xarajat = tezlik ⊗ masofa (np.outer); natija qator/ustun indekslari.

   ```python
   r_idx, c_idx = linear_sum_assignment(np.outer(tezlik, masofa))
   ```
4. Ikki ketma-ketlikni teskari tartibda juftlash ko'paytmalar yig'indisini minimallashtiradi (qayta tartiblash tengsizligi).
5. ss = z·σ·√L, z = Φ⁻¹(0,95); ROP = d·L + ss; EOQ = √(2·D_yil·S / (h·narx)).

   ```python
   z = stats.norm.ppf(xizmat)
   ```
6. Kelgan yuklarni qo'shish → sotish (min(zaxira, talab)) → zaxira + yo'ldagi ≤ ROP bo'lsa EOQ buyurtma. Formula σ_L ni hisobga olmagani uchun servis rejadan past.

Bajargach o'ylang: Sizning slotting va zaxira natijalaringiz namunadagidan farq qildimi? Yetkazish muddati noaniqligini xavfsizlik zaxirasi formulasiga qanday kiritgan bo'lardingiz?

## 6-qadam. Model monitoringi: drift (PSI, KS), haftalik sifat va qayta o'qitish

1. Yangi 80 film katalogga qo'shiladi (film taqsimoti o'zgaradi) va auditoriyaning yarmi qattiqroq baholaydi (baho taqsimoti va RMSE o'zgaradi).
2. PSI = Σ (c − r)·ln(c / r); bo'sh bin log(0) beradi. <0,1 barqaror, 0,1–0,2 kuzatish, >0,2 drift.

   ```python
   r = np.histogram(ref, bins)[0] / len(ref) + 1e-4
   ```
3. Baholar: np.arange(0.5, 6); filmlar: 20 talik guruhlar np.arange(0, n_i+1, 20). Avval umuman bo'lmagan filmlar birdan paydo bo'ldi.
4. Etalon bashoratlari va joriy hafta bashoratlarini ikki tanlanmali KS testi bilan solishtiring; p < 0,01 — taqsimot o'zgargan.

   ```python
   stats.ks_2samp(ref_pred, pr).pvalue
   ```
5. Yangi hafta + eski ma'lumotning 30% (sample(frac=0.3)); etalon yangilanmasa, PSI har hafta DRIFT deb qichqiraveradi.
6. Drift oldidan, drift davrida (v1) va qayta o'qitishdan keyin (v2) o'rtacha RMSE ni alohida hisoblang; bitta haftalik ma'lumot kichik.

Bajargach o'ylang: Sizning ogohlantirish qoidangiz drift'ni namunadagi bilan bir haftada aniqladimi? Faqat RMSE ga tayanganingizda (PSIsiz) signal qachon kelardi va bu nimani anglatadi?

## 7-qadam. Joylash va hujjat: API xizmat, kanareyka relizi, model kartasi va hayot sikli hisoboti

1. /health — holat va modellar ro'yxati; /recommend/{user_id} — top-n. Diapazondan tashqari id uchun HTTPException(404, ...).

   ```python
   @app.get("/recommend/{user_id}")
   def recommend(user_id: int, n: int = 10): ...
   ```
2. Bir odam doim bitta versiyani ko'rishi kerak (tajriba toza, interfeys sakramaydi). Multiplikativ xesh: (id·2654435761 mod 2³²)/2³² < ulush → v2.

   ```python
   (user_id * 2654435761 % 2**32) / 2**32
   ```
3. fastapi.testclient.TestClient jarayon ichida so'rov yuboradi — tarmoq yo'q, p95 ~4 ms; real tizimda bir necha barobar katta.
4. Xatolar soni (status != 200) va kechikish p95 (np.percentile(lat, 95)); masalan xato 0 va p95 < 50 ms bo'lsa — davom.
5. Model/versiya, maqsad, ma'lumot va bo'lish usuli, offline va onlayn metrikalar (2–3-qadam), cheklovlar, monitoring qoidalari (6-qadam), etika. yaml.safe_dump(allow_unicode=True, sort_keys=False).
6. Barcha bosqichlar qarori «davom» bo'lsa — bajarildi, aks holda to'xtatildi; hisobotni json.dump bilan faylga saqlang.

Bajargach o'ylang: Sizning xizmatingiz va kanareyka mantiqingiz namunadagidan nimasi bilan farq qiladi? Qaysi qo'riqchi metrika qo'shilsa (masalan CTR yoki bo'sh tavsiya ulushi) reliz xavfsizroq bo'lardi?
