# A5. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. DDP: gloo backend bilan 2 jarayonli taqsimlangan o'qitish

1. MASTER_ADDR/MASTER_PORT muhit o'zgaruvchilari tayyor; endi har jarayon o'z rank'i bilan jarayonlar guruhini ishga tushiradi. CPU uchun backend — gloo.

   ```python
   dist.init_process_group("gloo", rank=rank, world_size=world)
   ```
2. DistributedSampler to'plamni num_replicas bo'lakka kesishmaydigan qilib bo'ladi; DataLoader'ga sampler= orqali bering (shuffle= emas).
3. Sampler urug'i epoxa raqamiga bog'liq; chaqirilmasa har epoxada bir xil tartib takrorlanadi.
4. Chiziqli LR qoidasi: lr = baza_lr × world. DDP o'ralganda 0-rank parametrlari broadcast qilinadi; backward() da gradientlar all-reduce bo'ladi.
5. Barcha parametrlarni bitta vektorga yoyib, uning nusxalariga ReduceOp.MAX va ReduceOp.MIN bilan all_reduce qiling; farq 0 bo'lishi kerak.

   ```python
   dist.all_reduce(mx, op=dist.ReduceOp.MAX)
   ```
6. Kutilgan: har rank 4096 (2 epoxa × 2048), jami 8192; aniqlik ~0.95; farq 0.00e+00 -> 'BIR XIL (DDP sinxron)'.

Bajargach o'ylang: Namunada jami statistikalar va parametr farqi qanday yig'ilgan — siz ham all_reduce ishlatdingizmi yoki boshqa yo'l tanladingizmi? GPU klasterida (nccl, torchrun) kodingizdan nima o'zgaradi?

## 2-qadam. Masshtablash: 1 vs 2 jarayon, S(N) va E(N)

1. Kuchli: global paket o'zgarmaydi (256 -> har rankka 128); kuchsiz: har rankda to'liq paket (256), global 512.
2. fork kontekstidagi mp Queue yarating, worker argumentlariga uzating, rank 0 q.put(dt) qilsin, asosiy jarayon q.get() bilan oladi.

   ```python
   q = mp.get_context("fork").Queue()
   ```
3. MASTER_ADDR/PORT, GLOO_SOCKET_IFNAME='lo', set_num_threads(1), init_process_group('gloo'), DDP(make_model()), train_steps, destroy_process_group.
4. Kuchsiz rejimda ish hajmi 2 baravar. O'tkazuvchanlik = global_paket × STEPS / vaqt; S = thr / thr_baza; E = S / N.
5. Bu bir xil 2 yadroni boshqa usulda (matritsa amallari ichidagi parallellik) ishlatish; aloqa xarajati yo'q.
6. Kutilgan: baza ~4-5 s; kuchli S(2)≈1.6-1.8 (E≈80-90 %), kuchsiz E≈90 %, intra-op ~1.9x. Katta model — hisob/aloqa nisbati yaxshiroq.

Bajargach o'ylang: Sizning S(N) jadvalingiz namunadagi bilan mos keldimi? Kuchli masshtablashda samaradorlik nega pastroq — har rankka tushadigan ish va all-reduce hajmi orqali tushuntiring.

## 3-qadam. Halqa all-reduce ni NumPy'da qo'lda yozish va aloqa hajmi

1. N bo'lak; N-1 qadamdan keyin har ishchida bitta bo'lak barcha ishchilar bo'yicha to'liq yig'ilgan bo'ladi.
2. Bo'lak raqami (i − k) mod N; qabul qilish chap qo'shnidan: (i − 1) mod N. Avval hamma paketlarni yig'ib, keyin qo'llang — aks holda bir qadamda ikki marta qo'shib yuborasiz.

   ```python
   c = chunks[(i - k) % N]
   ```
3. Almashtiriladi (buf[i][c] = data). Reduce-scatter'dan keyin i-ishchida (i + 1) mod N bo'lak to'liq — all-gather shundan boshlanadi.
4. sent[i] ga har yuborishda len(c) × 4 qo'shing; nazariya: 2(N−1)/N × S, S = L × 4 bayt. N katta bo'lsa 2S ga intiladi.
5. vaqt = 2(N−1)/N × S_bayt / o'tkazuvchanlik (bayt/s). Gbit/s ni 8 ga bo'lib bayt/s ga o'tkazing.
6. Kutilgan: N=2,4,8 da 'natija to'g'ri=True' (np.allclose(o, ref, atol=1e-4)), baytlar formulaga aynan teng; 10 GbE da 8 GPU ~143 ms, NVLink ~0.3 ms.

Bajargach o'ylang: Namuna bilan solishtiring: paketlarni 'avval yig'ib, keyin qo'llash' tartibi sizda ham bormi? Sodda 'hammaga yuborish' usuli bilan solishtirganda halqa nega N katta bo'lganda yutadi?

## 4-qadam. Gradient to'plash va ma'lumotlar parallelligi ekvivalentligi

1. To'liq o'rtacha = to'rtta bo'lak o'rtachalarining o'rtachasi (bo'laklar teng bo'lsa). Demak har bo'lak hissasi 1/K vazn oladi.
2. .grad yig'iladi (qo'shiladi); mikro-paketlar orasida zero_grad chaqirilmaydi. Har yo'qotishni K ga bo'lib backward qiling.

   ```python
   (lossf(mB(xb), yb) / K).backward()
   ```
3. fresh() bir xil urug' bilan yaratadi (DDP broadcast'ining muqobili). Har ishchi o'z 64 namunasida backward qiladi.
4. all-reduce(SUM)/world = o'rtacha gradient.

   ```python
   gC = torch.stack(grads).mean(0)
   ```
5. Kutilgan: ~1e-8 (float32 yaxlitlash), xato variantda farq ~0.3 va ||gE||/||gA|| = 4.00 = K.

Bajargach o'ylang: Sizning yechimingiz namunadan nimasi bilan farq qiladi? Agar modelda BatchNorm bo'lganda (B) va (C) nega (A) ga teng chiqmasdi?

## 5-qadam. Parallel inferens: sharding, ishchilar puli, natijalarni yig'ish

1. Har so'rov mustaqil ('noqulay parallel'): ma'lumotni bo'laklarga bo'lib (scatter), har birini alohida hisoblab, natijalarni yig'ish (gather) yetarli.
2. h = ReLU(X @ W1 + b1); chiqish = (h @ W2 + b2).argmax(1). nn.Linear og'irligi (out, in) shaklda, .T bilan (in, out) bo'ladi.

   ```python
   h = np.maximum(X @ W1 + b1, 0)
   ```
3. Pool(initializer=init_worker, initargs=(W,)) — har ishchi ishga tushganda global o'zgaruvchiga oladi.
4. (idx, bo'lak) juftligini yuboring va natijalarni idx bo'yicha saralab, np.concatenate bilan birlashtiring (all-gather).
5. thr = so'rovlar / vaqt; S = thr / thr(1 ishchi); E = S / N. Birinchi chaqiruvda ishchilar ishga tushishi va kesh isishi vaqt oladi.
6. Kutilgan: ~0.5-0.6 s va ~0.3 s; S(2)≈1.7-1.9, E≈85-95 %; aniqlik ikkalasida bir xil (0.993) — farq bo'lsa, tartib buzilgan.

Bajargach o'ylang: Bo'lak hajmini (SHARD) kichraytirsangiz yoki kattalashtirsangiz S(2) va p99 qanday o'zgaradi? Namunadagi IPC xarajati haqidagi xulosani o'z o'lchovingiz bilan solishtiring.

## 6-qadam. Amdal va Gustafson qonunlari grafigi

1. Ketma-ket qism o'zgarmaydi, parallel qism N ga bo'linadi: T(N)/T(1) = (1−p) + p/N. Tezlashish — buning teskarisi.
2. p/N → 0, demak S → 1/(1−p). p=0.90 da 10x.
3. Ketma-ket qism (1−p) bir marta, parallel qism esa N baravar ko'p: S = (1−p) + p·N.
4. Faqat arifmetik amallar ishlating (if/for siz) — NumPy ularni elementma-element bajaradi; log-log o'qlar: ax.set(xscale='log', yscale='log').
5. Kutilgan: p=0.90 da S(256)≈9.7 (chegara 10x), p=0.99 da ≈72; Gustafson deyarli chiziqli.

Bajargach o'ylang: 2-qadamdagi o'z S(2) o'lchovingizdan p ni baholang (Amdal formulasini p ga nisbatan yeching). Bu p bilan 64 ishchida qanday tezlashish kutiladi — namuna xulosasiga mosmi?

## 7-qadam. Aloqa/hisob nisbati kalkulyatori (GPU klasteri uchun baho)

1. T_step = T_comp + (1 − overlap) · T_comm; samaradorlik E = T_comp / T_step.
2. Forward + backward ≈ 3 × forward. T_comp = 3 × GFLOP × 1e9 × batch × accum / (PEAK × MFU).
3. 2(N−1)/N × params × bytes_per_grad; vaqt = shu / bw. N=1 da aloqa yo'q.

   ```python
   t_comm = (2 * (N - 1) / N) * params * bpg / bw if N > 1 else 0.0
   ```
4. Hisob accum baravar o'sadi, all-reduce esa har to'plangan qadamda faqat bir marta — r accum ga teskari proporsional kamayadi.
5. Aloqa parametrlar soniga, hisob FLOPs × batch ga proporsional; FLOPs/parametr nisbati katta modellarda yuqoriroq.
6. Kutilgan: NVLink'da E≈99 %, 10 GbE da ResNet-50 E≈26 %, accum=8 bilan ≈74 %.

Bajargach o'ylang: O'z klasteringiz (masalan, PCIe orqali ulangan 2-4 GPU) parametrlarini kiritsangiz E qancha chiqadi? Namunadagi analitik model haqiqiy profiler o'lchovidan qaysi omillarni hisobga olmaydi?
