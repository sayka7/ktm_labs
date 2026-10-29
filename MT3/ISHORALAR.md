# MT3. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Arxitekturalar pasporti va Roofline modeli

1. Biri — hisoblash cho'qqisi (TOP/s), ikkinchisi — xotira o'tkazuvchanligi bilan AI ko'paytmasi. Birliklarga e'tibor bering: TOP/s = 1e12, GB/s = 1e9.
2. Ridge = cho'qqi / o'tkazuvchanlik (op/bayt). Hailo-8 da BW atigi 4 GB/s — shuning uchun ridge juda katta chiqadi.

   ```python
   df["ridge_int8"] = df.int8_tops * 1e12 / (df.bw_gbs * 1e9)
   ```
3. Har bir chiqish elementi uchun K ta ko'paytirish va K ta qo'shish. Ideal holda A, B va C matritsalar bir martadan o'qiladi/yoziladi; INT8 da element = 1 bayt.
4. Oddiy conv: 2·H·W·Ci·Co·k² amal. Depthwise da har kanal alohida (Ci = Co = C, kanallar aralashmaydi) — shuning uchun AI keskin tushadi.
5. P = min(P_cho'qqi, AI·BW). Agar cho'qqi AI·BW dan kichik yoki teng bo'lsa — hisoblash chegarasi. BW ni GB/s dan TB/s ga o'tkazish uchun 1e3 ga bo'ling.

   ```python
   p = min(q.int8_tops, ai * q.bw_gbs / 1e3)
   ```
6. 7B parametr INT8 da ≈ 7 GB. token/s ≈ BW / model hajmi; hisoblash cho'qqisi bu yerda deyarli rol o'ynamaydi.
7. GEMM 4096³: 2·4096³ / (3·4096²) ≈ 2731. GEMV: 2·4096² / (4096² + ...) ≈ 2. Agar shunday bo'lmasa — birliklarni (1e12/1e9/1e3) tekshiring.

Bajargach o'ylang: Namunada chegara turi P ni hisoblash bilan bir xil ifodadan aniqlanadi — sizda ham shundaymi? FP16 (b=2) bo'lsa qaysi yuklamalar chegara turini o'zgartirishi mumkinligini taxmin qiling.

## 2-qadam. TPU sistolik massivi: takt darajasidagi simulyator va MXU foydalanish modeli

1. Natija (acc) har PE ichida to'planadi; A qiymatlari o'ngga, B qiymatlari pastga siljiydi.
2. i-satr t − i taktda boshlanadi (skew). Oxirgi ko'paytma t = (K−1)+(R−1)+(C−1) da keladi, demak taktlar soni shundan bitta ko'p.
3. Chap ustun bo'shaydi va unga yangi qiymat kiradi; qolganlari bir ustun o'ngga ko'chadi. Ustma-ust tushgan kesmalar nusxasiz buzilishi mumkin.

   ```python
   a_reg[:, 1:] = a_reg[:, :-1].copy()
   ```
4. Chegaradan tashqarida 0 kiritiladi. A va B da nol yo'q (1..8), shuning uchun (a_reg != 0) & (b_reg != 0) — faol MAC belgisi.
5. Plitkalar = ceil(K/128)·ceil(N/128); har plitka M + 128 + 128 − 2 takt (+128 yuklash, agar ikki bufer bo'lmasa). Foydalanish = M·K·N / (128·128·jami_takt).
6. MXU modelida BERT 4096 token ~94 %, GEMV M=1 ~0,4 %, N=130 ~45 % bo'lishi kerak. GEMV nega shunchalik past — M=1 da fill/drain qancha ulush oladi?

Bajargach o'ylang: Namunada skew kirish indeksidagi t − i orqali amalga oshirilgan — siz uni boshqacha (masalan, A ni oldindan siljitib) qildingizmi? N=130 holatida padding isrofini qanday kamaytirish mumkin?

## 3-qadam. FPGA: quvur (pipeline), II va fixed-point aniqlik simulyatsiyasi

1. Boshlanish = i·II, tugash = boshlanish + bosqichlar soni. Jami takt — oxirgi kirishning tugash vaqti.
2. Kutilgan qiymat: 8 kirish 11 taktda (to'ldirish 3 takt + 8). II=2, II=4 da o'tkazuvchanlik taxminan II marta tushadi.
3. 1 MAC = 2 amal. INT8 packing'da bitta DSP 2 MAC bajaradi, FP32 esa ~3 DSP talab qiladi. INT8 packing uchun ~12 TOP/s kutiladi.

   ```python
   tops = DSP * ulush * mac_dsp * 2 * F / II / 1e12
   ```
4. x·2^n ni yaxlitlang (np.round), keyin ishorali jami-bit diapazoniga np.clip bilan to'yintiring. To'yinganlar sonini ham hisoblang.

   ```python
   q = np.round(x * (1 << kasr))
   ```
5. Ko'paytmada kasr bitlar qo'shiladi: kasr_a + kasr_w. ReLU dan keyin >> kasr_w qilib yana Q kasr_a ga qaytasiz — FPGA'da siljitish deyarli bepul.
6. Yig'indi to'yinib, chiqishlar buziladi. SQNR = 10·log10(Σ etalon² / Σ (etalon − kvant)²), bu yerda kvant chiqish 2^(kasr_a+kasr_w) ga bo'linadi.
7. Q0.7 da butun qism yo'q — |w| ≥ 1 bo'lgan og'irliklar to'yinadi. Og'irliklar diapazonini chiqarib ko'ring.

Bajargach o'ylang: Namunadagi fixed_mlp bilan solishtiring: bias va siljitishni siz ham xuddi shu tartibda qildingizmi? Q0.7 o'rniga Q1.6 tanlash qaysi muammoni hal qiladi?

## 4-qadam. NPU uchun INT8 kvantlash: aniqlik, hajm, kechikish va grafik bo'linishi

1. Faollashtirishlar ham kvantlanadi; ularning diapazoni (min/max) kuzatuvchilar (observer) orqali kalibrlash namunalaridan yig'iladi.
2. state_dict ni io.BytesIO ga torch.save qiling. Bias, masshtab va zero_point'lar qo'shimcha joy oladi.

   ```python
   b = io.BytesIO(); torch.save(model.state_dict(), b)
   ```
3. fuse_modules → prepare → kalibrlash (bir marta forward) → convert. Birlashtirilgan qatlam oraliq tenzorni alohida kvantlamaydi.

   ```python
   q = tq.fuse_modules(q, [["c1", "r1"], ...])
   ```
4. Per-channel har chiqish kanaliga alohida masshtab beradi: PerChannelMinMaxObserver + torch.per_channel_symmetric. Faollashtirish uchun quint8 MinMaxObserver.
5. s = max|w| / 127; wq = clip(round(w/s), −127, 127)·s; SQNR = 10·log10(Σw² / Σ(w−wq)²). ~40–47 dB kutiladi.
6. Har operator: NPU yoki CPU hisoblash vaqti. Joriy joy (CPU/NPU) o'zgarsa — chiqish tenzori MB / PCIe GB/s ms qo'shiladi. Kirish CPU xotirasida turadi.
7. Batch=1 da INT8 sekinroq chiqsa — bu xato emas: kichik modelda quant/dequant xarajati ustun keladi.

Bajargach o'ylang: Namunadagi kvantla() bilan solishtiring: kalibrlash torch.no_grad() ichida bajarilganmi? 8 va 256 namunali kalibrlash orasida farq kichik chiqqani nimani anglatadi?

## 5-qadam. Real CPU'da GEMM va konvolyutsiya o'lchovlari, boshqa arxitekturalarga ekstrapolyatsiya

1. Amaliy cho'qqi — eng tez ishlagan GEMM ning GFLOP/s qiymati; amaliy BW — STREAM triad. Ikkalasi birgalikda «o'lchangan Roofline» ni beradi.
2. 1–2 marta isitish (kesh, xotira ajratish), keyin bir necha o'lchov va median — tasodifiy uzilishlarga chidamli.

   ```python
   t0 = time.perf_counter(); f(); t.append(time.perf_counter() - t0)
   ```
3. b va c o'qiladi, a yoziladi: 3·n·4 bayt. out= parametri yangi xotira ajratmaslik uchun kerak.

   ```python
   torch.add(b, c, alpha=1.5, out=a)
   ```
4. ops = 2·batch·H·W·Co·(Ci/g)·k²; bayt = 4·(kirish + og'irlik + chiqish elementlari). Depthwise da g = Ci, shuning uchun AI juda kichik.
5. Kichik matritsalar L2/L3 keshga sig'adi va RAM triad BW dan tezroq o'qiladi. Depthwise samaradorligi < 10 % kutiladi.
6. FP32 baytlari 4 ga bo'linib, FP16 da 2 ga, INT8 da 1 ga ko'paytiriladi. Kichik yadrolar tezlatkichda 5–10 us «pol» ga uriladi.
7. 1e9 ga bo'lish, float32 = 4 bayt, GEMM da 2·N³ amal — eng ko'p uchraydigan xatolar shu yerda.

Bajargach o'ylang: Namunadagi e ni 5..100 % ga cheklash qanday maqsadga xizmat qiladi? Sizning ekstrapolyatsiyangiz PCIe nusxalashni hisobga oladimi va bu natijani qanday o'zgartirardi?

## 6-qadam. Energiya samaradorligi (TOP/s/Vt, mJ/tasvir) va 3 yillik TCO

1. Real unumdorlik = cho'qqi · u (TOP/s); 1 TOP = 1000 GOP. Uni 8,2 GOP/tasvir ga bo'ling.

   ```python
   ips = pk * u * 1e3 / GOP_TASVIR
   ```
2. math.ceil(TALAB / ips). Har dona uchun karta + xost ulushi Vt qo'shiladi.
3. Energiya/tasvir = quvvat (Vt = J/s) / tasvir/s. 1e3 ga ko'paytirib mJ ga o'ting. Kutilgan: NPU ~3,5, GPU ~10, CPU ~470 mJ.
4. Egalik = CAPEX (karta + xost) · n + elektr (P · soat / 1000 · PUE · narx). Bulut = n · USD/soat · soat. Narxi None bo'lgan variantni hisoblamang.
5. TALAB · SOAT · 3600 tasvir; eng arzon TCO (egalik yoki bulut) ni shu songa bo'lib, mln va sentga o'tkazing.
6. Sezgirlik: CAPEX o'zgarmaydi, elektr qismi narxga proporsional. Bulut faqat ishlagan soat uchun to'lanadi, egalik CAPEX esa baribir to'lanadi.
7. Kutilgan natija: faqat NPU mos. TPU v4 faqat bulutda mavjud (narx None).

Bajargach o'ylang: Namunadagi tco = min(egalik, bulut) yondashuvini o'zingiznikiga solishtiring. Qaysi faraz (foydalanish koeffitsienti, PUE yoki narx) natijaga eng kuchli ta'sir qiladi deb o'ylaysiz?

## 7-qadam. Yakuniy tanlov matritsasi: WSM, TOPSIS va vaznlar sezgirligi

1. Har mezonni [0, 1] ga keltiring: 1 — eng yaxshi. «Kami yaxshi» (−1) mezonlar uchun 1 − z oling.
2. Diapazon bir necha tartibga cho'zilgan; chiziqli min-max da qolganlar hammasi ~0 ga yopishib qoladi.

   ```python
   x = np.log10(df[m].astype(float))
   ```
3. Avval filtr: mos kelmaydigan nomzod normallash diapazoniga ham ta'sir qilmasligi kerak. JADVAL.apply(cheklov, axis=1) mantiqiy niqob beradi.
4. Ustunlarni √Σx² ga bo'lib, vaznga ko'paytiring; ideal = yaxshi yo'nalishdagi ekstremum. Ball = d⁻ / (d⁺ + d⁻).
5. rng.dirichlet(W0 * 50) — o'rtacha W0 atrofida, yig'indisi 1 bo'lgan vazn. Konsentratsiya qancha katta bo'lsa, tebranish shuncha kichik.

   ```python
   wv = rng.dirichlet(W0 * 50)
   ```
6. Chekkada faqat 2 nomzod qoladi — min-max ularni 0/1 ga aylantiradi, shuning uchun WSM va TOPSIS turli javob berishi mumkin.

Bajargach o'ylang: Namunadagi WSM va TOPSIS natijalarini o'zingiznikiga solishtiring. Qaysi usul nomzodlar kam bo'lganda ishonchliroq va nega?
