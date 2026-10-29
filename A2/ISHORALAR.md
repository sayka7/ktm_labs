# A2. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Sof Python vs NumPy vektorlash

1. Birinchi chaqiruvda kesh, xotira ajratish va BLAS ishga tushishi qo'shiladi; tasodifiy uzilishlar o'rtachani buzadi, median esa chidamli.

   ```python
   t0 = time.perf_counter(); f(); v.append(time.perf_counter() - t0)
   sorted(v)[len(v) // 2]
   ```
2. Har C[i][j] uchun n ta ko'paytirish va n ta qo'shish bor; FLOP ni vaqtga va 1e9 ga bo'ling.
3. A[i][k] ichki siklda o'zgarmaydi, B[k] va C[i] qatorlari ketma-ket yuriladi — indekslash xarajati kamayadi.
4. Kutilgan tezlanish 250–500×; ikkala natija farqining maksimal modulini (np.abs(...).max()) chiqaring — ~1e-13 bo'lishi kerak.
5. Taxminan 8 ta oraliq massiv, har biri 4 MB, har biri yoziladi va o'qiladi: 8·2·N·4 bayt.

   ```python
   0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x ** 3)))
   ```
6. Matmul hisobga bog'liq (har element ko'p marta qayta ishlatiladi), GELU esa xotira o'tkazuvchanligiga bog'liq.

Bajargach o'ylang: Namuna bilan solishtiring: siz o'lchov funksiyasini qanday yozdingiz va takror sonini n ga qarab o'zgartirdingizmi? Interpretatorning bitta element uchun ~100 ns sarfi sizning raqamlaringizda ko'rindimi?

## 2-qadam. Numba @njit va prange — OpenMP'ning Python muqobili

1. Birinchi chaqiruvda argument turlari bo'yicha LLVM kompilyatsiya bo'ladi; uni alohida o'lchab, keyingi chaqiruvlarni solishtiring.
2. Dekoratorga parallel=True va tashqi i siklida range o'rniga prange; har ip o'z C qatorlariga yozadi.

   ```python
   @njit(parallel=True)
   for i in prange(n): ...
   ```
3. prange ichidagi `s +=` ni Numba avtomatik reduction(+:s) deb taniydi — har ip o'z qisman yig'indisini oladi.
4. numba.set_num_threads(iplar); ip yaratish/sinxronizatsiya va umumiy kesh/xotira raqobati.

   ```python
   numba.set_num_threads(iplar)
   ```
5. BLAS registr/kesh bloklash, FMA vektorlash va qo'lda optimallashtirilgan mikroyadrolardan foydalanadi.
6. Har element 1 marta o'qiladi (FP64 = 8 bayt), amal juda kam — o'tkazuvchanlik cheklaydi; x.nbytes / t ga e'tibor bering.

Bajargach o'ylang: Namuna bilan solishtiring: kompilyatsiya vaqtini qanday ajratdingiz va olchov funksiyasi C ni har safar nolga tushirishi nega muhim? Sizning tezlanish raqamlaringiz kutilgan 1,6–1,9× ga tushdimi?

## 3-qadam. Kesh bloklash (tiled) vs naive matritsa ko'paytirish

1. Qator-major tartibda ustun bo'ylab yurish har qadamda n·4 bayt sakraydi — har element yangi 64 baytli satr.
2. Ichki j-sikl C[i,:] va B[k,:] bo'ylab ketma-ket — kompilyator uni SIMD bilan vektorlaydi.
3. Tashqi ii, kk, jj — range(0, n, bs); ichki i, k — blok ichida; eng ichkisi j — bs uzunlikdagi bo'lak.

   ```python
   for ii in range(0, n, bs):
       for kk in range(0, n, bs): ...
   ```
4. Indeks manfiy bo'lishi mumkin deb wraparound tekshiruvi qo'shiladi; 0 dan boshlanadigan indeks bilan bu tekshiruv yo'qoladi.

   ```python
   Ci = C[i, jj:jj + bs]
   Bk = B[k, jj:jj + bs]
   ```
5. 3·bs²·4 bayt; kichik bs da sikl boshqaruvi xarajati foydali ishdan ko'p bo'ladi.
6. Katta L3 kesh n=1024 matritsalarni (12 MB) sig'diradi; bloklash foydasi matritsa keshdan katta bo'lganda ko'rinadi.

Bajargach o'ylang: Namuna bilan solishtiring: bloklangan variantingizda ichki sikl vektorlandimi (GFLOP/s 2 dan yuqorimi)? Eng yaxshi bs qiymati kesh hajmi bilan qanday bog'liq chiqdi?

## 4-qadam. Konvolyutsiyani to'rt usulda parallellashtirish

1. Har piksel K·K·CIN ta MAC; MAC = 2 FLOP; chiqish elementlari soniga ko'paytiring — ~81,4 MFLOP.
2. co → i → j → (ci, di, dj); har co alohida chiqish kanaliga yozadi, shuning uchun co bo'ylab bo'lish xavfsiz.
3. Har (ci, di, dj) uchun siljigan oyna bitta qator bo'ladi: (CIN·K·K) × (HO·WO); og'irliklar (COUT) × (CIN·K·K).

   ```python
   cols[r] = x[ci, di:di + HO, dj:dj + WO].ravel()
   ```
4. Asosiy ish BLAS GEMM'ga o'tadi (bloklangan, vektorlangan); cuDNN implicit GEMM shuni xotirada nusxa yaratmasdan qiladi.
5. torch.set_num_threads(...) ni almashtiring; batch vaqtini 8 ga bo'ling; torch.no_grad() ichida o'lchang.
6. F.conv2d natijasini etalon qilib np.abs(y - etalon).max(); FP32 da yig'indi tartibi farqi tufayli ~5e-6.

Bajargach o'ylang: Namuna bilan solishtiring: qaysi usul sizda eng tez chiqdi va tezlanishlar zanjiri (prange 2× → im2col ~16× → torch ~20×) saqlandimi? im2col'ni yanada tezlashtirish uchun nima qilish mumkin edi?

## 5-qadam. PyTorch: GPU o'rniga ko'p yadro — 1 vs 2 ip, formatlar, yadro birlashtirish

1. Ishni iplarga bo'lish va sinxronizatsiya qat'iy xarajatga ega; ish hajmi n³ bilan o'sadi.

   ```python
   torch.set_num_threads(iplar)
   ```
2. Har iplar soni uchun avval 2 marta chaqirib, keyin median olish; oneDNN birinchi chaqiruvda ish joyini tayyorlaydi.
3. Masshtab s = max|A|/127; A/s ni yaxlitlab [-128,127] ga qisib int8; natijani sa·sb ga ko'paytiring.

   ```python
   A8 = (A / sa).round().clamp(-128, 127).to(torch.int8)
   ```
4. Format foydasi apparatda maxsus buyruq (AVX512-BF16, VNNI, Tensor Core) bo'lsagina; aks holda emulyatsiya qilinadi.
5. Har amal 1 o'qish + 1 yozish; ~7 amal → 7·2·N·4 bayt; birlashtirilgan F.gelu esa faqat 2·N·4.
6. Samarali GB/s ni chiqaring: ikkalasida ham o'tkazuvchanlik yaqin bo'lsa, farq — o'tishlar sonida.

Bajargach o'ylang: Namuna bilan solishtiring: _int_mm qo'llab-quvvatlanmasa dasturingiz yiqilmaydimi (try/except)? GPU'da shu tajriba natijalari qaysi qatorlarda tubdan o'zgarar edi?

## 6-qadam. Ma'lumot parallelligi: multiprocessing bilan 2 jarayon

1. Kichik NumPy massivlar ustida ko'p amal — vaqtning katta qismi Python interpretatorida; GIL bir vaqtda faqat bitta ipga ruxsat beradi.
2. gx uchun o'ng va chap qo'shni ustunlar farqi, gy uchun pastki va yuqori qatorlar; o'rtacha — 9 ta siljigan bo'lak yig'indisi / 9.

   ```python
   gx[:, 1:-1] = x[:, 2:] - x[:, :-2]
   ```
3. Har bo'lak pickle qilinib IPC orqali yuboriladi; kichik bo'lak — ko'p xabar va ko'p xarajat, katta bo'lak — yuk muvozanati yomonlashishi mumkin.

   ```python
   with ctx.Pool(n_proc) as pool:
       natija = pool.map(kadr_ishlov, kadrlar, chunksize=cs)
   ```
4. E = S / p; har jarayon kamroq ma'lumot bilan ishlaganda kesh yaxshiroq ishlashi mumkin (superchiziqli effekt).
5. map tartibni saqlaydi — ro'yxatlarni == bilan solishtiring (assert); parallellashtirish to'g'rilikni buzmasligi kerak.
6. Yaratish xarajatini ketma-ket vaqt (~0,6 s) bilan solishtiring; DDP'da process group init.

Bajargach o'ylang: Namuna bilan solishtiring: isinish (pool.map(kadrlar[:20])) ni qo'shdingizmi va u o'lchovga qanday ta'sir qiladi? Eng yaxshi chunksize sizda qaysi bo'ldi va nega?

## 7-qadam. Amdahl va Gustafson qonunlari: nazariya va 2 yadroda o'lchov

1. Ketma-ket qism o'zgarmaydi, parallel qism n ga bo'linadi; S = T(1)/T(n).
2. p/n nolga intiladi — qoladigan faqat ketma-ket ulush; p=0,95 da 20.
3. Vaqt emas, masala hajmi o'sadi: parallel qism n marta ko'p ish bajaradi; S = (1−p) + p·n.
4. 1/S = (1−p) + p/n dan p = (1 − 1/S)/(1 − 1/n); o'lchov shovqini p > 1 berishi mumkin.

   ```python
   p = min(1.0, max(0.0, (1 - 1 / S2) / (1 - 1 / 2)))
   ```
5. amdahl(0.97, 1024) ni hisoblang (~30×); 3 % ketma-ket ish minglab yadroni bekor qiladi.
6. n 1 dan 1024 gacha o'zgaradi — log o'q to'yinishni yaqqol ko'rsatadi; nuqta o'zingiz baholagan p egri chizig'ida.

   ```python
   ax1.set_xscale("log", base=2); ax1.plot([2], [S2], "ko")
   ```

Bajargach o'ylang: Namuna bilan solishtiring: ketma-ket qismning alohida o'lchangan vaqti Amdahl bahosidagi (1−p) ulushga mos keldimi? GPU'da ketma-ket qismlarni (NMS, ma'lumot yuklash) kamaytirish uchun qanday choralar ko'rgan bo'lardingiz?
