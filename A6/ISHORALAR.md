# A6. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Bazaviy model va FGSM: aniqlikning epsilon'ga bog'liqligi

1. data/16 — 1 kulrang daraja = 1/16. eps ning ma'nosi faqat shu fazoda aniq; eps×16 — necha kulrang darajaga teng.
2. Yo'qotishning kirish bo'yicha gradienti; sign(grad) har pikselni roppa-rosa ±eps ga suradi — L_inf sharining burchagi.
3. x ni nusxalab, grafikdan uzib, requires_grad_(True) qiling; torch.autograd.grad(loss, x) bilan oling.

   ```python
   grad, = torch.autograd.grad(loss, x)
   ```
4. .clamp(0, 1) — haqiqiy rasm bo'lib qolishi uchun; .detach() — keyingi hisoblarda eski grafik saqlanmasligi uchun.
5. torch.sign(torch.randn_like(Xte)) — tasodifiy yo'nalish. Hujum xavfi buzilish kattaligida emas, yo'nalishida.
6. Kutilgan: toza test ~0.97; eps=0.05 ~0.77, 0.1 ~0.3, 0.15 ~0.1; tasodifiy shovqin eps=0.15 da ~0.93. O'rtacha L_inf eps ga teng (yoki qirqish tufayli kichik).

Bajargach o'ylang: Namunadagi FGSM bilan sizniki farq qiladimi (masalan, model.eval() yoki clamp joyi)? Egri chiziq nega bunchalik keskin tushadi — chiziqli model gipotezasi (Goodfellow) bilan tushuntiring.

## 2-qadam. PGD hujumini qo'lda yozish va FGSM bilan taqqoslash

1. PGD — kichik alpha qadamli iterativ FGSM; har qadamda gradient yangidan hisoblanadi.
2. L_inf shariga proyeksiya — elementma-element x0−eps va x0+eps orasiga qirqish (torch.max/torch.min yoki clamp), so'ng [0,1].

   ```python
   xa = torch.min(torch.max(xa, x0 - eps), x0 + eps).clamp(0, 1)
   ```
3. eps-shar ichida bir tekis tasodifiy nuqtadan boshlash (torch.empty_like(x).uniform_(-eps, eps)) lokal tekis sohalardan qochishga yordam beradi.
4. Aks holda hisob grafigi qadamdan qadamga o'sib boradi va gradient eski nuqtalarga ham tarqaladi; xa.detach() + alpha * grad.sign().
5. (xa − Xte).abs().max() <= eps + 1e-6 — assert bilan; suzuvchi nuqta xatosi uchun kichik tolerantlik.
6. PGD har eps da FGSM dan past yoki teng; eps=0.15: FGSM ~0.11, PGD ~0.05; eps=0.2: PGD ~0.00. PGD kuchsiz chiqsa — eval()/detach/proyeksiyada xato.

Bajargach o'ylang: PGD-10 va PGD-40 farqi qanchalik katta chiqdi va hisob narxi necha baravar oshdi? Namunadagi alpha = 2.5·eps/steps tanlovi nimaga xizmat qiladi — boshqa alpha bilan sinab ko'rdingizmi?

## 3-qadam. Adversarial o'qitish (qisqa) va mustahkamlik murozasi

1. Madry: min_θ E[ max_{||δ||∞≤eps} L(θ, x+δ, y) ]. Ichki max — PGD hujumi, tashqi min — optimizator qadami.
2. Joriy model bilan: hujum har doim hozirgi zaif joylarni topishi kerak. Hujum paytida m.eval(), qadam paytida m.train().
3. PGD-k har qadamda k ta qo'shimcha forward+backward qiladi: taxminan (k+1) baravar. Vaqtni time.time() bilan o'lchang.
4. new_model() bir xil urug' bilan; baholashda torch.manual_seed(1) har hujumdan oldin; eps=0 da toza Xte.
5. Himoya faqat o'qitilgan chegaragacha kafolatlanadi; undan katta eps da aniqlik yana pasayadi.
6. Kutilgan: ~6x qimmat; PGD-10 eps=0.1 da ~0.27 -> ~0.83; eps=0.2 da ~0.2. Digits'da toza aniqlik deyarli tushmasligi mumkin.

Bajargach o'ylang: Murozani o'z raqamlaringiz bilan tavsiflang: mustahkamlik qancha oshdi, narx qancha? Namuna bilan solishtirganda EPS_TRAIN yoki k ni o'zgartirish natijani qanday siljitdi?

## 4-qadam. Gradient-saliency xaritasi: toza va hujum ostidagi raqamlar

1. Bashorat qilingan sinf logitining kirish x bo'yicha gradienti; ishorasi emas, moduli (|grad|) olinadi.
2. Ilg'or indekslash: model(x)[torch.arange(len(x)), cls]; logitlar yig'indisining x bo'yicha gradienti har namunaga alohida tushadi.

   ```python
   logit = model(x)[torch.arange(len(x)), cls].sum()
   ```
3. p = s / s.sum(1, keepdim=True); H = −Σ p log p (log(0) dan qochish uchun +1e-12). Katta H — tarqoq xarita.
4. np.corrcoef(a, b)[0, 1] — Pearson; ~0 bo'lsa xarita 'ko'chgan', ya'ni tushuntirish butunlay o'zgargan.
5. Biz modelning hozirgi qarorini tushuntiryapmiz; adv rasmda model boshqa sinfni tanlagan.
6. Kutilgan: rasmlar toza bilan deyarli bir xil, bashorat esa noto'g'ri; o'rtacha korrelyatsiya ~0 atrofida.

Bajargach o'ylang: Entropiya adversarial namunani aniqlash belgisi bo'la oladimi — sizning jadvalingizda toza va adv entropiyalari aniq ajraladimi? Namunadagi xulosa bilan solishtiring.

## 5-qadam. Adolatlilik: MetricFrame va ThresholdOptimizer

1. Yorliqlarda: A=1 guruhda haqiqiy ijobiylarning 25 % i 0 deb yozilgan. income A ga bog'liq — vakil (proxy) belgi.
2. DP — tanlov ulushi P(ŷ=1|A) farqi; EO — TPR va FPR farqlarining kattasi. Fairlearn'da demographic_parity_difference / equalized_odds_difference.
3. metrics= lug'at (nom -> funksiya), y_true, y_pred, sensitive_features; natija — mf.by_group.

   ```python
   mf = MetricFrame(metrics={"tanlov": selection_rate}, y_true=yte, y_pred=pred, sensitive_features=Ate)
   ```
4. 'Bilmaslik orqali adolat' proxy belgilar tufayli ishlamaydi — income orqali A haqidagi axborot saqlanadi. feats = ['income', 'history', 'debt'].
5. Oldindan o'qitilgan modelning ehtimolliklari ustidan guruhga xos chegaralar tanlanadi: prefit=True, predict_method='predict_proba'; fit/predict da sensitive_features shart.
6. Kutilgan: baza DP ~0.25, EO ~0.27; A siz DP ~0.13; TO(DP) DP ~0.01, TO(EO) EO ~0.02; aniqlik 1-3 bandga pasayadi.

Bajargach o'ylang: DP va EO ni bir vaqtda nolga tushira oldingizmi? Kredit masalasida qaysi mezon asosliroq — namunadagi xulosa va 7-qadamdagi DALIL[6] bilan solishtiring.

## 6-qadam. SHAP TreeExplainer: global muhimlik, lokal izoh, additivlik

1. Barcha belgi koalitsiyalari bo'yicha o'rtacha marginal hissa; TreeSHAP daraxt tuzilmasidan foydalanib polinomial vaqtda aniq hisoblaydi.
2. Log-odds (decision_function). Ba'zi versiyalarda (n, 30, 2) qaytadi — unda 1-sinfni oling; expected_value ham massiv bo'lishi mumkin.

   ```python
   sv = sv[..., 1] if sv.ndim == 3 else sv
   ```
3. np.abs(sv).mean(0) — ijobiy va salbiy hissalar bir-birini yeyib qo'ymasligi uchun; pd.Series(..., index=X.columns).sort_values().
4. bazaviy qiymat + Σ SHAP = modelning shu namunadagi chiqishi; clf.decision_function(Xte.iloc[[i]]) bilan solishtiring.
5. Har namuna uchun top-5 (np.argsort(-np.abs(...))[:5]) to'plamlarining kesishmasi / 5, keyin o'rtacha.
6. Kutilgan: aniqlik ~0.94; top: worst perimeter, worst concave points, mean concave points; farq ~1e-15; barqarorlik ~0.99.

Bajargach o'ylang: Sizning top-10 ro'yxatingiz namunadagi bilan bir xilmi? SHAP global muhimligi GradientBoosting'ning feature_importances_ dan nimasi bilan farq qiladi va qaysi biri ishonchliroq?

## 7-qadam. Etik tekshiruv ro'yxatini avtomatik baholash

1. Texnik bandlar 7, 8, 9, 11, 20 — metrikalardan; qolganlari (inson nazorati, audit izi, fallback...) — DALIL lug'atidan.
2. Toza va FGSM eps=0.1 aniqligi (1-qadam), DP/EO farqi (5-qadam), urug' va versiyalar. Kichik MLP + FGSM + LogisticRegression yetarli.
3. 7-band: o'lchangan, lekin chegaradan o'tmasa — 'qisman'; 9-band: himoya sinovdan o'tmasa — 'yo'q'.

   ```python
   holat = "ha" if o["eo_farqi"] < 0.05 else "qisman"
   ```
4. Audit tamoyili: da'vo dalil bilan tasdiqlanadi. holat == 'ha' va dalil.strip() bo'sh bo'lsa -> 'qisman'; lug'atda yo'q band -> 'yo'q'.
5. ha=5, qisman=2.5, yo'q=0 (20 × 5 = 100). >= 70 yetarli, < 50 qoniqarsiz, oralig'i — qisman.
6. Kutilgan: ~60/100; bajarilmagan bandlar masalan 9, 12, 14, 18 — AI Act 9, 12, 14-moddalar (xavf boshqaruvi, jurnal, inson nazorati).

Bajargach o'ylang: O'z tizimingiz uchun DALIL lug'atini to'ldirsangiz ball qanday o'zgaradi? Namunadagi baholash qoidalaridan qaysi biri juda yumshoq yoki juda qattiq ko'rinadi?
