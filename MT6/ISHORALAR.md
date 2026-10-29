# MT6. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. AI tizimi xavf reyestri generatori (ISO/IEC 23894 uslubi): ehtimollik x ta'sir, qoldiq xavf, Monte Karlo

1. Ehtimollik L va ta'sir I (1..5). Ko'paytma ehtimoliy 'kutilgan zarar' g'oyasiga yaqin: kam ehtimolli, lekin halokatli xavf ham, tez-tez, lekin kichik xavf ham o'rtacha chiqadi.
2. MODS ro'yxatidagi har qoida: profil kaliti True bo'lsa, shu xavf id siga dL, dI qo'shiladi. Keyin np.clip(L, 1, 5) bilan cheklang.

   ```python
   for key, mid, dL, dI in MODS:
       if mid == rid and PROFILE.get(key): ...
   ```
3. Shu xavfga tegishli barcha choralarning L va I kamayishlarini yig'ib, ayiring va yana 1..5 ga cheklang; qoldiq = rL x rI.
4. Avval boshlang'ich xavf ishtahadan past bo'lsa — qabul; keyin chorasi yo'q bo'lsa — eskalatsiya; so'ng qoldiqqa qarab kamaytirish. Kutilgan: R08, R02, R10 eskalatsiyaga chiqadi.
5. reg.xavf.sum() va reg.qoldiq.sum() ni solishtiring: kutilgan ~126 -> ~75 (~40%).
6. 5000 marta rL va rI ga {-1, 0, +1} dan tasodifiy xato qo'shing, 1..5 ga cheklang va ko'paytma > 8 bo'lgan ulushni oling. Kutilgan R08 uchun ~0,8.

   ```python
   Ls = np.clip(r.rL + rng.integers(-1, 2, 5000), 1, 5)
   ```

Bajargach o'ylang: Sizning reyestringiz namunadagi bilan bir xil ishlov qarorlarini berdimi? Monte Karlo natijasi qaysi xavflar bo'yicha qo'shimcha dalil yoki chora kerakligini ko'rsatadi?

## 2-qadam. EU AI Act xavf toifasini aniqlovchi qoida dvigateli (15 ssenariy, avtomatik testlar bilan)

1. Bir tizim bir nechta shartga mos kelishi mumkin (masalan, real vaqt yuz tanish ham biometrika sohasida). Eng og'ir toifa birinchi tekshiriladi va topilsa darhol qaytariladi.
2. 5(1)(f) bandida tibbiy yoki xavfsizlik maqsadi uchun istisno bor: shu atribut True bo'lsa, taqiq tekshiruvini o'tkazib yuboring (continue) — keyin 50(3) shaffoflik qoidasi ishlaydi.
3. 6(3): tor protsedura vazifasi (masalan, fayllarni formatga keltirish) — istisno, lekin profillash bo'lsa istisno qo'llanmaydi. Istisnoda ham 6(4) hujjatlashtirish va 49(2) ro'yxat shart.

   ```python
   if s.get("tor_protsedura_vazifa") and not s.get("profillash"): ...
   ```
4. Davlat xizmati yoki kredit bo'lsa 27-modda (asosiy huquqlarga ta'sirni baholash, FRIA) qo'shiladi: HIGH_OBLIG (11 ta) + 1 = 12.
5. Toifa — tizimning qo'llanishiga (muloqot -> shaffoflik), GPAI — modelning o'ziga tegishli (53-modda, >1e25 FLOP bo'lsa 51/55). Bitta tizimda ikkalasi ham bo'lishi mumkin.
6. Har ssenariyning kutilgan toifasi bor — cat == exp ni hisoblab, mos sonini chiqaring (kutilgan 15/15). Mos kelmaganning asos ustuni qaysi qoida ishlaganini ko'rsatadi.

   ```python
   df.mos.sum()
   ```

Bajargach o'ylang: Sizning qoida dvigatelingiz namunadagi bilan bir xil asoslarni (moddalarni) berdimi? Real tasnifda qaysi ssenariylar yurist bilan qo'shimcha ko'rib chiqishni talab qiladi?

## 3-qadam. Fairness audit: kesishgan guruhlar (jins x yosh), 4/5 qoidasi va yumshatish usullari

1. Tarixiy qarorlarda ish tanaffusi (jins vakili) va >45 yosh nohaq jazolangan. Model shu yorliqdan o'rganadi, aniqlik ham tarafkash yorliqqa nisbatan o'lchanadi.
2. Har bir atribut bo'yicha farq kichik ko'rinsa ham, masalan, '>45 yoshli ayollar' guruhi eng ko'p zarar ko'rishi mumkin. MetricFrame ga ikki ustunli DataFrame bering.

   ```python
   MetricFrame(metrics={...}, y_true=yte, y_pred=pred, sensitive_features=Ate)
   ```
3. DI = eng past tanlov ulushi / eng yuqori tanlov ulushi; 0,8 dan kichik bo'lsa qoida buziladi. Kutilgan bazaviy DI ~0,5.
4. Yorliqning o'zi tarafkash va boshqa belgilar (tajriba yoshga bog'liq) orqali ma'lumot sizib chiqadi. 'Unawareness' tarafkash yorliqni tuzatmaydi.
5. Kutilgan ulush P(g)*P(y) ni kuzatilgan P(g, y) ga bo'ling; vaznni sample_weight orqali LogisticRegression.fit ga bering.

   ```python
   w[m] = (g == gg).mean() * (ytr == yy).mean() / m.mean()
   ```
6. DemographicParity(difference_bound=0.03) — guruhlar tanlov ulushi farqini cheklaydi. Kesishgan guruh uchun jins + '|' + yosh qatorini bering; predict ga sezgir atribut kerak emas.
7. Kutilgan: unawareness DI ni yaxshilamaydi, qayta tortish ~0,7, ExpGrad ~0,77, aniqlik esa 1-4 bandga pasayadi.

Bajargach o'ylang: Sizning natijalaringiz namunadagi DI qiymatlariga yaqinmi? Real auditda sezgir atributlarni yig'ishning huquqiy asosi va kichik guruhlarning statistik ishonchliligini qanday hal qilasiz?

## 4-qadam. Mustahkamlik testi: shovqin, sensor uzilishi, kalibrlash drifti va adversarial hujum

1. Toza test faqat 'ideal' sharoitni o'lchaydi. Shovqin — sensor shovqini, uzilish — sensor ishdan chiqishi, drift — kalibrlash siljishi, FGSM — ataylab manipulyatsiya.
2. StandardScaler dan keyin har ustun o'rtachasi 0. Ustunni nusxada 0 ga almashtiring va har model uchun eng yomon (minimal) aniqlikni saqlang.

   ```python
   Zm = Zte.copy(); Zm[:, j] = 0.0
   ```
3. Drift fizik birlikda (+5 C) bo'ladi: Xte xom nusxasining 0-ustuniga qo'shib, keyin sc.transform qiling. Kutilgan: +10 C da RF ~0,65 gacha tushadi.
4. Har belgini yo'qotishni oshiruvchi tomonga eps qadar (L_inf) siljitish. MLP uchun torch.autograd.grad bilan kirish bo'yicha gradient oling.

   ```python
   g, = torch.autograd.grad(F.cross_entropy(mlp(x), torch.tensor(yte)), x)
   ```
5. LR: softmax ehtimollaridan haqiqiy sinf uchun 1 ni ayiring va coef_ ga ko'paytiring: (P - onehot) @ W. RF uchun MLP da yasalgan adversarial misollarni ko'chiring (transfer).
6. Pasayish = (toza - buzilgan) x 100 p.p.; chegaralar: shovqin 5, sensor 10, drift 5, FGSM 10. Kutilgan: harorat sensori uzilishi ~11-13 p.p. (XATO), MLP FGSM eps=0,2 da o'tmaydi.

Bajargach o'ylang: Sizning OK/XATO jadvalingiz namunadagidan farq qiladimi? Qaysi model qaysi buzilishga eng sezgir va texnik hujjatda (EU AI Act 15-modda) buni qanday asoslaysiz?

## 5-qadam. Tushuntiriluvchanlik: permutatsion muhimlik vs SHAP, barqarorlik, surrogat daraxt, kontrfaktual

1. Global — model umuman qaysi belgilarga tayanadi; lokal — aynan shu ariza nega rad etildi (EU AI Act 86-modda: qaror sababini bilish huquqi).
2. Permutatsion: belgi qiymatlarini aralashtirganda aniqlik qanchaga tushadi. SHAP: har bashoratga belgining hissasi (log-odds), mutlaq qiymatlar o'rtachasi.

   ```python
   pi = permutation_importance(gb, Xte, yte, n_repeats=8, random_state=0)
   ```
3. Haqiqatda ta'sirsiz belgi eng past o'rinda bo'lishi kerak — bu usullar 'shovqinni' muhim deb ko'rsatmasligini tasdiqlaydi. Reyting mosligini kendalltau bilan o'lchang (kutilgan ~0,6).
4. Har safar Xtr dan qaytarib tanlab yangi model o'qiting, SHAP top-3 to'plamini solishtiring. Kutilgan ~100% mos; o'zgarsa izohga ishonib bo'lmaydi.

   ```python
   idx = rng.integers(0, len(Xtr), len(Xtr))
   ```
5. Surrogat murakkab modelni taqlid qiladi: y sifatida gb.predict(Xtr) bering. Sodiqlik = testda surrogat va model bashoratlari mosligi ulushi (kutilgan ~0,89).
6. Bitta omil qiymatini panjara bo'yicha siljiting (qarz yukini 0 tomon, kechikishlarni kamaytirib, daromadni 3 baravargacha) va birinchi p >= 0,5 nuqtani qayd qiling. Yosh kabi o'zgarmas omillar bo'yicha bermang.

Bajargach o'ylang: Sizning global muhimlik reytingingiz va tushuntirish xatingiz namunadagi bilan mos keldimi? Real arizachiga beriladigan xatda qaysi ma'lumotni qoldirib, qaysisini olib tashlagan bo'lardingiz?

## 6-qadam. Muvofiqlik tekshiruv ro'yxati va gap-tahlil hisoboti (ISO/IEC 42001, 23894, EU AI Act)

1. Joriy holat — CURRENT dagi yetuklik (0-5) va dalil; maqsad — xavf toifasi va talab vazniga bog'liq TARGET lug'ati (yuqori xavfda vazn 3 uchun 4).
2. 'Hujjatlashtirilmagan — bajarilmagan': dalil bo'sh va baho >= 2 bo'lsa, uni 1 ga tushiring. Bu o'z-o'zini baholashdagi optimizmni cheklaydi.

   ```python
   if cur >= 2 and not ev: cur = 1
   ```
3. Ustuvorlik = gap x vazn (majburiylik). Saralashda teng bo'lsa vazn bo'yicha qo'shimcha tartiblang. Kutilgan eng ustuvorlar: A.5 (ta'sirni baholash) va 73-modda.
4. Yo'q — joriy qiymatni maqsad bilan cheklang (clip), keyin vaznli yig'indini vaznli maqsad yig'indisiga bo'ling. Kutilgan ~42/100.

   ```python
   df.joriy.clip(upper=df.maqsad)
   ```
5. manba ustunidan regex bilan guruh nomini ajrating (str.extract), keyin groupby. Rahbariyat qaysi standart bo'yicha ko'proq ish kerakligini ko'radi.
6. Vazn 3 va holati 'bajarilmagan' talab bo'lsa — joriy etish to'xtatiladi. Kutilgan 7 ta majburiy talab bajarilmagan.

Bajargach o'ylang: Sizning indeksingiz va top-8 ro'yxatingiz namunadagi bilan mos keldimi? Crosswalk (bir chora bir necha manbani yopishi) takomillashtirish rejasini qanday arzonlashtiradi?

## 7-qadam. Takomillashtirish rejasi: byudjet ostida optimal choralar, bog'liqliklar bilan Gantt va PDCA

1. Bog'liqlikli 'ryukzak' (knapsack): qiymat (yopilgan vaznli gap + kamaygan xavf) maksimal, narx yig'indisi <= 250 mln, har chora bog'liqliklari ham tanlangan bo'lishi shart.
2. Yopilgan darajalarni gap bo'yicha yig'ib, min(yig'indi, gap) x vazn oling; xavf kamayishi ham xavf - 1 bilan cheklanadi (xavf 1 dan kam bo'lmaydi).
3. 2^13 = 8192 — juda kam. Har k uchun itertools.combinations bilan yurib, feasible bo'lganlarning qiymatini solishtiring.

   ```python
   for k in range(len(M) + 1):
       for sel in itertools.combinations([m[0] for m in M], k): ...
   ```
4. Ochko'z usul faqat joriy qadamdagi qiymat/narx ga qaraydi; bog'liqlik ochib beradigan kelajak qiymatni ko'rmaydi. Kutilgan: optimal ~94,5, ochko'z ~77,5.
5. Boshlanish = max(jamoa bo'shagan vaqt, bog'liqliklar tugagan vaqt). Har safar tayyor choralardan eng erta boshlanadiganini eng erta bo'shagan jamoaga bering. Kutilgan reja ~14 hafta.

   ```python
   j = int(np.argmin(free)); start[c] = max(free[j], t_dep)
   ```
6. Oldin = sum(gap x vazn), keyin = sum(max(0, gap - yopilgan) x vazn); xavf keyin = max(1, xavf - kamayish). Kutilgan ~80% gap yopiladi, R01 va R07 yuqori qoladi.

Bajargach o'ylang: Sizning optimal to'plamingiz va jadvalingiz namunadagi bilan mos keldimi? Choralar soni 40 ta bo'lsa to'liq qidiruv o'rniga qaysi usulni (butun sonli dasturlash, dinamik dasturlash) ishlatardingiz?
