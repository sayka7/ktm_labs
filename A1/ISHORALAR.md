# A1. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Talablardan yuklama hisobi: GFLOP, bayt, TFLOP/s

1. Bir sekundda nechta inferens bajariladi? KAMERALAR va FPS dan INFERENS_S kelib chiqadi; GFLOP/inferens x inferens/s = GFLOP/s.
2. Har konvolyutsiya chiqish xaritasining har pikselida bajariladi; piksel soni H*W ga, ya'ni imgsz^2 ga proporsional.

   ```python
   gflop = gflop640 * (imgsz / 640) ** 2
   ```
3. 1 TFLOP = 1000 GFLOP. 28,6 x 160 = 4576 GFLOP/s.
4. Platforma reklama qilingan cho'qqining faqat bir qismini real erishadi: cho'qqi = talab / samaradorlik (~10,2 TOPS).
5. FP16 — 2 bayt, INT8 — 1 bayt har parametrga; natijani 1e6 ga bo'lib MB qiling (yolov8s FP16 ~22,3 MB).
6. Ikki tsiklli list comprehension -> pd.DataFrame; so'ng mantiqiy niqob bilan filtrlang.

   ```python
   df[(df.model == "yolov8s") & (df.imgsz == 640)].iloc[0]
   ```
7. 1920*1080*3 bayt ~6,2 MB kadr; 160 kadr/s ~1 GB/s; tarmoq 64 Mbit/s — 1 Gbit/s portning 6,4 %.

Bajargach o'ylang: Natijangiz namunadagi 4,58 TFLOP/s va ~10,2 TOPS bilan mos keldimi? SAMARADORLIK 0,3 yoki FPS 25 bo'lsa qaysi model hali ham Orin NX (50 TOPS) ga sig'adi?

## 2-qadam. Platformalar jadvali va talablar matritsasi bo'yicha tanlash skripti

1. Platforma 4,58 TFLOP/s ni umuman bera olmasa, boshqa mezonlarda yuqori ball olishi ahamiyatsiz — zaxira >= 1 sharti.
2. 50 x 0,45 = 22,5; 22,5 / 4,58 ~4,9. pandas'da ustunlar orasidagi amal butun ustunga qo'llanadi.
3. Kadr/s bo'yicha nisbat 3 ga teng; demak nvdec * 3 >= 16 bo'lishi kerak.
4. J = Vt * s; sekundiga inferens = tops_real*1e3 / GFLOP. Energiya = vt / (inferens/s).
5. (z - 1) / 2 ni 1 bilan yuqoridan cheklang; z=2 -> 0,5.

   ```python
   min(1.0, (z - 1) / 2)
   ```
6. lambda r: ... har satrga qo'llanadi; sum(ball*og'irlik) / sum(og'irlik) * 100. iterrows() bilan aylaning.

   ```python
   for _, r in platformalar.iterrows():
       ballar = {m: f(r) * w for m, w, f in MEZONLAR}
   ```
7. sort_values(["yaroqli", "jami_%"], ascending=False) va jadval[jadval.yaroqli].iloc[0].

Bajargach o'ylang: Namunada narx mezoni 1 - narx/2000 deb olingan — siz boshqacha baholash funksiyasini tanladingizmi? Qaysi og'irlik o'zgarsa AGX Orin birinchi o'ringa chiqadi?

## 3-qadam. Kechikish byudjeti va Monte-Karlo simulyatsiyasi

1. Talab '200 ms dan oshmasin' — bu dumdagi (tail) kadrlar haqida; o'rtacha qiymat ularni yashiradi.
2. Kechikish manfiy bo'lmaydi va o'ng tomonga cho'zilgan. Lognormalning mean parametri log-shkalada: log(median).

   ```python
   rng.lognormal(np.log(med), sigma, N)
   ```
3. Namunalarni element bo'yicha qo'shib, keyin yig'indining persentilini oling — p95 lar yig'indisi pessimistik bo'ladi.

   ```python
   np.percentile(jami, 95)
   ```
4. (batch-1)/INFERENS_S sekund; kutish 0 dan shu qiymatgacha tekis (uniform) taqsimlangan deb oling (~94 ms).
5. Bir kadrga inferens vaqti (1 + 0,25*(batch-1))/batch marta kamayadi — GPU katta paketda samaraliroq.
6. Mantiqiy massivning o'rtachasi ulushni beradi.

   ```python
   np.mean(e2e > BYUDJET_MS) * 100
   ```

Bajargach o'ylang: Namunada jadval va simulyatsiya uchun alohida namunalar olingan — siz bitta funksiyadan foydalandingizmi? sigma 0,2 dan 0,5 ga oshsa, qaysi batch hali 200 ms ichida qoladi?

## 4-qadam. Arifmetik intensivlik va Roofline modeli

1. Hisoblash shifti P_peak va xotira 'qiyaligi' AI*B_mem; AI = bajarilgan FLOP / xotiradan o'tgan bayt.
2. P_peak*1e12 / (B_mem*1e9) — FLOP/s ni bayt/s ga bo'lish; Orin NX ~490.
3. Vaznlar (PARAMS*bayt_param) bir marta o'qiladi; faollashuvlar AKTIV_BAYT_FP16*batch*(bayt_param/2) — INT8 da yarmi.
4. GEMM: 2n^3 FLOP, 3 ta matritsa x n^2 x 4 bayt -> n/6. GELU: har elementga bir necha amal, 4 bayt o'qish + 4 yozish.
5. AI * bw / 1e3 TFLOP/s beradi; agar u >= peak bo'lsa — H, aks holda X.

   ```python
   erish = min(peak, ai * bw / 1e3)
   ```
6. np.logspace(-1, 4, 200) bo'yicha np.minimum(peak, x*bw/1e3) ni chizing; ax.set_xscale('log'), set_yscale('log').

Bajargach o'ylang: Namunadagi soddalashtirilgan trafik modeli (60 MB faollashuv) sizning taxminingizdan farq qildimi? INT8 ga o'tish nega Orin NX'da tezlikni deyarli ikki barobar oshiradi, A100'da esa unchalik emas?

## 5-qadam. Kichik CNN uchun FLOP va parametrlarni torch bilan hisoblash

1. Har chiqish elementi Kh*Kw*(Cin/groups) ta MAC; 1 MAC = 2 FLOP. Chiqish elementlari Cout*Hout*Wout*batch.
2. Forward hook (mod, inp, out) oladi; out.shape[-2:] — haqiqiy chiqish o'lchami, out.shape[0] — batch.

   ```python
   kh, kw = mod.kernel_size
   hout, wout = out.shape[-2:]
   ```
3. model.modules() barcha ichki modullarni qaytaradi; isinstance bilan tekshiring.

   ```python
   m.register_forward_hook(conv_hook)
   ```
4. sum(p.numel() for p in model.parameters()); FP32 — 4 bayt/parametr.
5. Kontekst menejeri ichida modelni chaqiring, so'ng get_total_flops(); farq 0 % bo'lishi kerak (u ham 2*MAC sanaydi).

   ```python
   with torch.no_grad(), FlopCounterMode(display=False) as fc:
       model(x)
   ```
6. Birinchi chaqiruvlar xotira ajratish/kesh bilan sekin; time.perf_counter() bilan 10 marta o'lchab median oling. FLOP ~imgsz^2 -> 4 marta.

Bajargach o'ylang: Namunada Linear uchun ham hook bor — sizning tarmog'ingizda u kerak bo'ldimi? Nega batch=8 da GFLOP/s yuqoriroq va bu 4-qadamdagi arifmetik intensivlik bilan qanday bog'liq?

## 6-qadam. Hardware–Software Co-design: izlanish fazosini grid search bilan ko'rib chiqish

1. Model/imgsz/format/batch — dasturiy, platforma — apparat. INT8 ni qo'llamaydigan platformada INT8 modeli ma'nosiz va hokazo.
2. 3 x 3 x 2 x 3 x 6 = 324, format qo'llanmaganlar tashlanadi -> 270. itertools.product Dekart ko'paytmasini beradi.

   ```python
   for m, s, f, b, p in itertools.product(MODELLAR, IMGSZ, FORMATLAR, BATCHLAR, PLATFORMALAR):
   ```
3. inferens_ms = batch*gflop/(tops_real*1e3)*1e3 (batch=1 da 1,3 ga ko'paytiring); kutish = (batch-1)/INFERENS_S*1e3.
4. {nom: bool} lug'ati: yaroqli = all(values), buzilgan = vergul bilan ajratilgan False kalitlar; keyin df.buzilgan.str.contains(k).
5. Arzonroq va sifatliroq boshqa variant bo'lmasa — nuqta Pareto-optimal. Narx o'sib borganda mAP oldingisidan katta bo'lsagina qo'shing.
6. A100 — quvvat, narx, harorat; Coral — zaxira. groupby("platforma") va set ayirmasi yordam beradi.

Bajargach o'ylang: MIN_ZAXIRA ni 1,5 dan 3 ga yoki MAKS_NARX ni 1000 ga o'zgartiring — izlanish fazosi qanchaga torayadi? Namunadagi Pareto qidiruvi bilan sizning usulingiz qaysi holatda farq qiladi (bir xil narxli nuqtalar)?

## 7-qadam. Edge va markazlashgan arxitekturaning 3 yillik TCO taqqoslovi

1. Apparat bir martalik (CapEx), energiya/sovutish, tarmoq ijarasi va xizmat — har yillik (OpEx), 3 yilga ko'paytiriladi.
2. Server yarim bo'lib sotilmaydi — yuqoriga yaxlitlash.

   ```python
   int(np.ceil(sexlar / SIGIM_SEX[nom]))
   ```
3. PUE — sovutish va infratuzilma ko'paytmasi. Vt*PUE/1000 = kVt; x 8760 soat x 0,08 $ x 3 yil (~92 $).
4. Edge da video joyida tahlil qilinadi (tarmoq 0); markazda har sex markazga kanal ijarasini to'laydi — sexlar soniga ko'paytiring.
5. jami / (sexlar*16) / (YIL*12).
6. np.arange(1, 25) bo'yicha massivlar, (a100 < edge).any() va np.argmax birinchi True indeksini beradi.

Bajargach o'ylang: Elektr narxini 0,2 $ ga, tarmoq ijarasini 0 ga tushiring — tenglik nuqtasi paydo bo'ladimi? TCO'da hisobga olinmagan omillar (kechikish, yagona nosozlik nuqtasi) qarorga qanday ta'sir qiladi?
