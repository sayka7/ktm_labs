# MT5. Mustaqil bajarish uchun variantlar (20)

Variant raqami = talabaning jurnal raqami (21 va undan katta bo'lsa, 20 ga bo'lgandagi qoldiq; qoldiq 0 bo'lsa 20-variant). Mustaqil ishning s1–s7 qadamlari kodi asos qilib olinadi: bilim bazasi (FACTS, DISTRACT), savollar, bo'laklash, qidiruv usuli, tau va injection to'plami o'z variantingiz parametrlariga almashtiriladi. Topshiriq .py fayllar + hisobot (PDF yoki Markdown: jadvallar, grafiklar, xulosa) ko'rinishida topshiriladi; internet va haqiqiy LLM ishlatilmaydi, barcha raqamlar taxminiy.

## 1-variant. Bank call-markazi: tarif va karta qoidalari bo'yicha RAG yordamchi (oson)

Soha: Moliya (bank)

Bank operatorlari mijoz savollariga (karta limitlari, komissiyalar, kredit muddatlari) tez javob topishi kerak. Ichki yo'riqnomalar o'zbek tilida, 5 hujjat va 20 ta fakt. Javob faqat yo'riqnomaga tayanishi, sonlar (foiz, summa) xato bo'lmasligi shart; mijoz chatiga kiritilgan matnlar orqali injection bo'lishi mumkin.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 5 hujjat, 20 fakt (karta, kredit, omonat, o'tkazma, komissiya) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | jumla-2 |
| Top-k | 3 |
| Qidiruv usuli | BM25 |
| Injection fokusi | unut/ignore |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (5 hujjat, 20 fakt (karta, kredit, omonat, o'tkazma, komissiya)); jumla-2 va kamida 2 boshqa strategiyani BM25 bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «unut/ignore» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 2-variant. Poliklinika: bemorlarga qabul tartibi bo'yicha savol-javob (oson)

Soha: Tibbiyot (poliklinika)

Poliklinika saytida bemorlar qabul vaqti, tahlil topshirish, sug'urta polisi bo'yicha savol beradi. Bilim bazasi 4 ta ichki nizomdan iborat. Tizim tashxis qo'ymasligi, javobsiz savollarda shifokorga yo'naltirishi va har javobda manba ko'rsatishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 4 hujjat, 16 fakt (qabul, tahlil, sug'urta, shoshilinch yordam) |
| Kontekst oynasi | 4096 token |
| Bo'laklash | so'z-60/15 |
| Top-k | 3 |
| Qidiruv usuli | TF-IDF(char) |
| Injection fokusi | rol o'yini |
| Rad etish tau (boshlang'ich) | 0.35 |
| KV-kesh modeli | Llama-2-7B (MHA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 4096 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-2-7B (MHA) uchun 4k, 4k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (4 hujjat, 16 fakt (qabul, tahlil, sug'urta, shoshilinch yordam)); so'z-60/15 va kamida 2 boshqa strategiyani TF-IDF(char) bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.35 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «rol o'yini» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 3-variant. Universitet: talabalar uchun reglament yordamchisi (oson)

Soha: Ta'lim (oliy ta'lim)

Talabalar kredit-modul tizimi, qayta topshirish, stipendiya va akademik ta'til qoidalari haqida so'raydi. Reglament 6 bo'limli, ko'p raqamli muddatlar bor. Tizim noto'g'ri muddat aytmasligi va javobsiz savollarda dekanatga yo'naltirishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt (kredit, imtihon, stipendiya, ta'til, yotoqxona, kutubxona) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | jumla-3 |
| Top-k | 3 |
| Qidiruv usuli | Gibrid RRF |
| Injection fokusi | tizim prompti |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt (kredit, imtihon, stipendiya, ta'til, yotoqxona, kutubxona)); jumla-3 va kamida 2 boshqa strategiyani Gibrid RRF bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «tizim prompti» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 4-variant. Soliq xizmati: tadbirkorlar uchun FAQ yordamchi (oson)

Soha: Davlat xizmatlari (soliq)

Kichik tadbirkorlar soliq hisobotlari muddati, imtiyozlar va jarimalar haqida savol beradi. Bilim bazasi 5 ta qisqa yo'riqnoma. Raqamlar (stavka, muddat) aynan manbadagidek bo'lishi, gallyutsinatsiya bo'lsa ogohlantirish chiqishi talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 5 hujjat, 20 fakt (QQS, aylanma soliq, muddatlar, jarimalar, imtiyozlar) |
| Kontekst oynasi | 16384 token |
| Bo'laklash | so'z-80/20 |
| Top-k | 5 |
| Qidiruv usuli | BM25 |
| Injection fokusi | bilvosita (hujjatda) |
| Rad etish tau (boshlang'ich) | 0.32 |
| KV-kesh modeli | Llama-3-70B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 16384 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-70B (GQA) uchun 4k, 16k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (5 hujjat, 20 fakt (QQS, aylanma soliq, muddatlar, jarimalar, imtiyozlar)); so'z-80/20 va kamida 2 boshqa strategiyani BM25 bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=5 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.32 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «bilvosita (hujjatda)» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 5-variant. Internet-provayder: texnik yordam bilim bazasi (oson)

Soha: Telekom

Abonentlar router sozlash, tarif o'zgartirish va uzilishlar haqida yozadi. Bilim bazasida o'zbekcha va ruscha (lotin yozuvida) aralash maqolalar bor. Tokenizator samaradorligi va kontekstga nechta maqola sig'ishi alohida hisoblanishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 22 fakt (router, tarif, uzilish, to'lov, IPTV, mobil) |
| Kontekst oynasi | 4096 token |
| Bo'laklash | jumla-2 |
| Top-k | 3 |
| Qidiruv usuli | TF-IDF(char) |
| Injection fokusi | niqoblangan |
| Rad etish tau (boshlang'ich) | 0.28 |
| KV-kesh modeli | Llama-2-7B (MHA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 4096 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-2-7B (MHA) uchun 4k, 4k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 22 fakt (router, tarif, uzilish, to'lov, IPTV, mobil)); jumla-2 va kamida 2 boshqa strategiyani TF-IDF(char) bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.28 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «niqoblangan» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 6-variant. Logistika kompaniyasi: yuk jo'natish qoidalari (oson)

Soha: Logistika

Mijozlar va menejerlar yuk o'lchamlari, taqiqlangan tovarlar, yetkazish muddatlari va sug'urta haqida so'raydi. Hujjatlar uzun, jadvalga o'xshash jumlalar ko'p. Bo'laklash strategiyasi recall@3 va kontekst hajmi bo'yicha tanlanishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 5 hujjat, 20 fakt (o'lcham, vazn, taqiq, muddat, sug'urta) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | so'z-40/10 |
| Top-k | 5 |
| Qidiruv usuli | Gibrid RRF |
| Injection fokusi | unut/ignore |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (5 hujjat, 20 fakt (o'lcham, vazn, taqiq, muddat, sug'urta)); so'z-40/10 va kamida 2 boshqa strategiyani Gibrid RRF bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=5 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «unut/ignore» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 7-variant. Mehmonxona tarmog'i: mehmonlar uchun konsyerj-bot (oson)

Soha: Turizm (mehmonxona)

Mehmonlar check-in vaqti, nonushta, transfer va bekor qilish shartlari haqida so'raydi. Savollar o'zbek va ingliz tillarida keladi. Bot mehmonxonada yo'q xizmatni «o'ylab topmasligi» va rad etish chegarasi to'g'ri tanlanishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 4 hujjat, 16 fakt (bron, bekor qilish, nonushta, transfer) |
| Kontekst oynasi | 4096 token |
| Bo'laklash | jumla-2 |
| Top-k | 3 |
| Qidiruv usuli | BM25 |
| Injection fokusi | rol o'yini |
| Rad etish tau (boshlang'ich) | 0.33 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 4096 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 4k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (4 hujjat, 16 fakt (bron, bekor qilish, nonushta, transfer)); jumla-2 va kamida 2 boshqa strategiyani BM25 bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.33 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «rol o'yini» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 8-variant. Neft-gaz korxonasi: mehnat muhofazasi yo'riqnomalari (o'rta)

Soha: Sanoat (neft-gaz)

Ishchilar smenada himoya vositalari, ruxsatnoma-naryad va avariya tartibi haqida so'raydi. Xato javob hayotga xavf tug'diradi, shuning uchun javobsiz savollarda tizim albatta rad etishi va xavfsizlik muhandisiga yo'naltirishi kerak. Hujjatlarda o'xshash, lekin boshqa sexga tegishli jumlalar ko'p.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt + 24 chalg'ituvchi (sex, burg'ulash, gaz, yong'in, SIZ, ruxsatnoma) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | jumla-3 |
| Top-k | 3 |
| Qidiruv usuli | Gibrid RRF |
| Injection fokusi | bilvosita (hujjatda) |
| Rad etish tau (boshlang'ich) | 0.38 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt + 24 chalg'ituvchi (sex, burg'ulash, gaz, yong'in, SIZ, ruxsatnoma)); jumla-3 va kamida 2 boshqa strategiyani Gibrid RRF bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.38 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «bilvosita (hujjatda)» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 9-variant. Sug'urta kompaniyasi: polis shartlari bo'yicha agent yordamchisi (o'rta)

Soha: Sug'urta

Agentlar mijozga KASKO, OSAGO va tibbiy sug'urta shartlarini tushuntiradi. Shartlarda ko'p raqamli limitlar va istisnolar bor; son almashgan javob (masalan, franshiza) jiddiy moliyaviy oqibatga olib keladi. Hallucination detektori son tekshiruviga alohida e'tibor berishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 5 hujjat, 20 fakt (KASKO, OSAGO, tibbiy, franshiza, to'lov) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | so'z-60/15 |
| Top-k | 3 |
| Qidiruv usuli | BM25 |
| Injection fokusi | unut/ignore |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-70B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-70B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (5 hujjat, 20 fakt (KASKO, OSAGO, tibbiy, franshiza, to'lov)); so'z-60/15 va kamida 2 boshqa strategiyani BM25 bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «unut/ignore» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 10-variant. Elektron tijorat: qaytarish va yetkazish siyosati chatboti (o'rta)

Soha: Chakana savdo (e-tijorat)

Onlayn do'kon mijozlari buyurtma holati, qaytarish muddati, kafolat va to'lov usullari haqida yozadi. Sharh va mahsulot tavsiflari orqali bilvosita injection tushishi mumkin. Detektor ko'rilmagan injection turlarida baholanishi va 3 ta prompt shabloni taqqoslanishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt (yetkazish, qaytarish, kafolat, to'lov, chegirma, sharhlar) |
| Kontekst oynasi | 16384 token |
| Bo'laklash | jumla-2 |
| Top-k | 5 |
| Qidiruv usuli | Gibrid RRF |
| Injection fokusi | bilvosita (hujjatda) |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 16384 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 16k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt (yetkazish, qaytarish, kafolat, to'lov, chegirma, sharhlar)); jumla-2 va kamida 2 boshqa strategiyani Gibrid RRF bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=5 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «bilvosita (hujjatda)» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 11-variant. Energetika: dispetcherlar uchun avariya reglamenti (o'rta)

Soha: Energetika

Elektr tarmog'i dispetcherlari uzilishlar, yuklama cheklovi va ta'mirlash tartibi bo'yicha reglamentdan tezkor ma'lumot oladi. Javob kechikishi muhim: 2x24 GB GPU da kontekst 8k bo'lganda nechta parallel so'rov sig'ishi hisoblanadi. Javoblar faqat reglamentga tayanadi.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 5 hujjat, 20 fakt (uzilish, yuklama, ta'mir, ruxsat, aloqa) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | so'z-80/20 |
| Top-k | 3 |
| Qidiruv usuli | BM25 |
| Injection fokusi | tizim prompti |
| Rad etish tau (boshlang'ich) | 0.32 |
| KV-kesh modeli | Llama-3-70B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-70B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (5 hujjat, 20 fakt (uzilish, yuklama, ta'mir, ruxsat, aloqa)); so'z-80/20 va kamida 2 boshqa strategiyani BM25 bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.32 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «tizim prompti» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 12-variant. Huquqiy firma: shartnoma shablonlari bo'yicha qidiruv (o'rta)

Soha: Huquq (yuridik xizmat)

Yuristlar ichki shablon va izohlardan band, muddat va jarima qiymatlarini qidiradi. Hujjatlar uzun jumlali, takroriy iboralar ko'p. Bo'lak hajmi va overlap ta'siri, BM25 parametrlari (k1, b) va recall@k hisobotda ko'rsatilishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt (ijara, oldi-sotdi, xizmat, maxfiylik, jarima, nizolar) |
| Kontekst oynasi | 32768 token |
| Bo'laklash | so'z-160/40 |
| Top-k | 5 |
| Qidiruv usuli | BM25 |
| Injection fokusi | unut/ignore |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 32768 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 32k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt (ijara, oldi-sotdi, xizmat, maxfiylik, jarima, nizolar)); so'z-160/40 va kamida 2 boshqa strategiyani BM25 bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=5 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «unut/ignore» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 13-variant. Aviakompaniya: bagaj va reys qoidalari yordamchisi (o'rta)

Soha: Transport (aviatsiya)

Yo'lovchilar bagaj normasi, reys kechikishi kompensatsiyasi va chipta almashtirish haqida so'raydi. Savollar uch tilda; inglizcha-markazli tokenizatorda o'zbekcha savollar ko'proq token oladi — narx va kontekst byudjeti hisoblanishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 5 hujjat, 20 fakt (bagaj, qo'l yuki, kompensatsiya, almashtirish, maxsus yuk) |
| Kontekst oynasi | 4096 token |
| Bo'laklash | jumla-2 |
| Top-k | 3 |
| Qidiruv usuli | TF-IDF(char) |
| Injection fokusi | niqoblangan |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-2-7B (MHA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 4096 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-2-7B (MHA) uchun 4k, 4k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (5 hujjat, 20 fakt (bagaj, qo'l yuki, kompensatsiya, almashtirish, maxsus yuk)); jumla-2 va kamida 2 boshqa strategiyani TF-IDF(char) bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «niqoblangan» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 14-variant. Qishloq xo'jaligi: fermerlar uchun agronom maslahatchi (o'rta)

Soha: Agro

Fermerlar ekish muddati, o'g'it me'yori va sug'orish rejimi haqida so'raydi. Bilim bazasi agronomik tavsiyalar to'plami; noto'g'ri me'yor hosilga zarar yetkazadi. Javob manbaga tayanishi va sonlar tekshirilishi, javobsiz savollarda agronomga yo'naltirish kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt (g'alla, paxta, sabzavot, o'g'it, sug'orish, zararkunanda) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | jumla-3 |
| Top-k | 3 |
| Qidiruv usuli | Gibrid RRF |
| Injection fokusi | rol o'yini |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt (g'alla, paxta, sabzavot, o'g'it, sug'orish, zararkunanda)); jumla-3 va kamida 2 boshqa strategiyani Gibrid RRF bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «rol o'yini» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 15-variant. IT-kompaniya: ichki DevOps runbook yordamchisi (o'rta)

Soha: IT (DevOps)

Muhandislar deploy, rollback, monitoring ogohlantirishlari va kirish huquqlari bo'yicha runbook'dan javob oladi. Matnlarda buyruq va kod bo'laklari bor, bu char n-gram TF-IDF uchun qulay, lekin injection uchun ham xavfli. Detektor qiyin zararsiz so'rovlarda (git ignore, system prompt nima) FPR bo'yicha baholanadi.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt (deploy, rollback, alert, kirish, backup, insident) |
| Kontekst oynasi | 16384 token |
| Bo'laklash | so'z-60/15 |
| Top-k | 5 |
| Qidiruv usuli | TF-IDF(char) |
| Injection fokusi | tizim prompti |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 16384 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 16k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt (deploy, rollback, alert, kirish, backup, insident)); so'z-60/15 va kamida 2 boshqa strategiyani TF-IDF(char) bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=5 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «tizim prompti» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod (barcha qadamlar 60 s ichida) — 30, O'lchovlar va jadvallar to'liqligi — 25, Tahlil va xulosa (raqamlar bilan asoslangan) — 25, Hisobot sifati — 20 (jami 100)

## 16-variant. Davlat portali: fuqarolarga xizmatlar navigatori (ko'p tilli) (qiyin)

Soha: Davlat xizmatlari (portal)

Fuqarolar pasport, ro'yxatdan o'tish, nafaqa va ma'lumotnomalar olish tartibini so'raydi. Savollar o'zbek (lotin), ingliz va rus (lotin transliteratsiya) tillarida. Ikki tokenizator fertilligi, kontekst byudjeti, gibrid qidiruv va to'liq RAG baholash hisoboti (recall, aniqlik, rad etish, faithfulness) talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 8 hujjat, 32 fakt (pasport, propiska, nafaqa, ma'lumotnoma, to'lov, muddatlar, onlayn xizmat, shikoyat) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | jumla-2 va so'z-80/20 |
| Top-k | 5 |
| Qidiruv usuli | Gibrid RRF |
| Injection fokusi | barcha 5 oila |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (8 hujjat, 32 fakt (pasport, propiska, nafaqa, ma'lumotnoma, to'lov, muddatlar, onlayn xizmat, shikoyat)); jumla-2 va so'z-80/20 va kamida 2 boshqa strategiyani Gibrid RRF bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=5 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «barcha 5 oila» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali; qo'shimcha ravishda ssenariydagi murakkab talabni (ko'p bosqichli himoya / reranker / ablyatsiya / to'liq baholash hisoboti) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod — 25, Murakkab talab (ssenariydagi qo'shimcha vazifa) — 20, O'lchovlar va jadvallar — 20, Tahlil va xulosa — 20, Hisobot sifati — 15 (jami 100)

## 17-variant. Kiberxavfsizlik SOC: insident javob playbook'lari (qiyin)

Soha: Xavfsizlik (SOC)

SOC tahlilchilari fishing, ransomware, sizib chiqish insidentlari bo'yicha playbook qadamlarini so'raydi. Hujjatlarning o'zida hujum misollari (injection matnlari) bor — bilvosita injection filtri toza bo'laklarni noto'g'ri olib tashlamasligi kerak. Ko'p bosqichli himoya (filtr + shablon) zaharlangan bo'laklar ulushi 0/10/30% da baholanadi.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt + hujum misollari (fishing, ransomware, DDoS, sizib chiqish, eskalatsiya, tiklash) |
| Kontekst oynasi | 16384 token |
| Bo'laklash | jumla-3 |
| Top-k | 5 |
| Qidiruv usuli | Gibrid RRF |
| Injection fokusi | bilvosita + niqoblangan |
| Rad etish tau (boshlang'ich) | 0.32 |
| KV-kesh modeli | Llama-3-70B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 16384 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-70B (GQA) uchun 4k, 16k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt + hujum misollari (fishing, ransomware, DDoS, sizib chiqish, eskalatsiya, tiklash)); jumla-3 va kamida 2 boshqa strategiyani Gibrid RRF bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=5 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.32 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «bilvosita + niqoblangan» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali; qo'shimcha ravishda ssenariydagi murakkab talabni (ko'p bosqichli himoya / reranker / ablyatsiya / to'liq baholash hisoboti) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod — 25, Murakkab talab (ssenariydagi qo'shimcha vazifa) — 20, O'lchovlar va jadvallar — 20, Tahlil va xulosa — 20, Hisobot sifati — 15 (jami 100)

## 18-variant. Farmatsevtika: dori qo'llanmalari bo'yicha provizor yordamchisi (qiyin)

Soha: Tibbiyot (dorixona)

Provizorlar doza, o'zaro ta'sir va saqlash sharoitini qo'llanmalardan tekshiradi. Son yoki dori nomi almashgan javob xavfli, shuning uchun hallucination detektori jumla darajasida ishlashi, obyekt (dori nomi) almashtirishni aniqlovchi qo'shimcha qoida qo'shilishi va chegara precision >= 0.95 bo'yicha kalibrlanishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt (doza, o'zaro ta'sir, saqlash, qarshi ko'rsatma, bolalar, homiladorlik) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | jumla-2 |
| Top-k | 3 |
| Qidiruv usuli | BM25 |
| Injection fokusi | rol o'yini |
| Rad etish tau (boshlang'ich) | 0.36 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt (doza, o'zaro ta'sir, saqlash, qarshi ko'rsatma, bolalar, homiladorlik)); jumla-2 va kamida 2 boshqa strategiyani BM25 bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.36 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «rol o'yini» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali; qo'shimcha ravishda ssenariydagi murakkab talabni (ko'p bosqichli himoya / reranker / ablyatsiya / to'liq baholash hisoboti) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod — 25, Murakkab talab (ssenariydagi qo'shimcha vazifa) — 20, O'lchovlar va jadvallar — 20, Tahlil va xulosa — 20, Hisobot sifati — 15 (jami 100)

## 19-variant. Sud tizimi: protsessual muddatlar bo'yicha kotib yordamchisi (qiyin)

Soha: Adliya

Sud kotiblari da'vo berish, apellyatsiya va ijro muddatlarini kodeks izohlaridan qidiradi. Muddat xatosi jiddiy huquqiy oqibatga ega. Semantik bo'laklash, reranker va rad etish chegarasini birga sozlab, 30 savolli (24 javobli + 6 javobsiz) to'plamda umumiy aniqlik >= 0.9 ga erishish talab qilinadi.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt (da'vo, apellyatsiya, kassatsiya, ijro, muddat tiklash, to'lov) |
| Kontekst oynasi | 32768 token |
| Bo'laklash | semantik bo'laklash |
| Top-k | 5 |
| Qidiruv usuli | BM25 + reranker |
| Injection fokusi | unut/ignore |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-70B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 32768 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-70B (GQA) uchun 4k, 32k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt (da'vo, apellyatsiya, kassatsiya, ijro, muddat tiklash, to'lov)); semantik bo'laklash va kamida 2 boshqa strategiyani BM25 + reranker bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=5 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «unut/ignore» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali; qo'shimcha ravishda ssenariydagi murakkab talabni (ko'p bosqichli himoya / reranker / ablyatsiya / to'liq baholash hisoboti) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod — 25, Murakkab talab (ssenariydagi qo'shimcha vazifa) — 20, O'lchovlar va jadvallar — 20, Tahlil va xulosa — 20, Hisobot sifati — 15 (jami 100)

## 20-variant. Korporativ HR portali: shablonlarni avtomatik tanlash va ablyatsiya (qiyin)

Soha: Bandlik (HR)

HR portali xodimlarga ta'til, maosh, sug'urta va o'qitish bo'yicha javob beradi. Ehtimoliy LLM simulyatori (85–95% bo'ysunish) bilan kamida 16 ta prompt shabloni 5 martadan yuritilib, yakuniy ball, std va token narxi bo'yicha Pareto tahlili va element ablyatsiyasi o'tkazilishi kerak.

| Parametr | Qiymat |
|---|---|
| Bilim bazasi | 6 hujjat, 24 fakt (ta'til, maosh, sug'urta, trening, xizmat safari, intizom) |
| Kontekst oynasi | 8192 token |
| Bo'laklash | jumla-2 |
| Top-k | 3 |
| Qidiruv usuli | TF-IDF(char) |
| Injection fokusi | barcha 5 oila |
| Rad etish tau (boshlang'ich) | 0.3 |
| KV-kesh modeli | Llama-3-8B (GQA) |

Qadamlar:

1. s1 kodini o'z sohangiz matnlari bilan (kamida 12 o'zbekcha va 12 inglizcha jumla) ishga tushiring; fertillik jadvali va 8192 tokenli oynada nechta bo'lak sig'ishini hisoblang.
2. s2 kalkulyatori bilan Llama-3-8B (GQA) uchun 4k, 8k kontekstda KV-kesh hajmi va 2x24 GB GPU'da parallel so'rovlar sonini hisoblang (GPU — hisob-kitob, simulyatsiya).
3. s3 kodidagi FACTS va DISTRACT ni sohangizga almashtiring (6 hujjat, 24 fakt (ta'til, maosh, sug'urta, trening, xizmat safari, intizom)); jumla-2 va kamida 2 boshqa strategiyani TF-IDF(char) bilan recall@1/3/5, MRR va kontekst hajmi bo'yicha taqqoslang.
4. s4 mini-RAG ni top-k=3 bilan yig'ing, 6 ta javobsiz savol qo'shing va tau ni 0.3 atrofida to'r bo'yicha tanlang; xato javoblarni tahlil qiling.
5. s5 hallucination detektorini sohangiz javoblarida (to'g'ri, parafraz, son/obyekt almashgan, to'qilgan) ishga tushiring: AUC, precision/recall va turlar jadvali.
6. s6 injection detektorini «barcha 5 oila» oilasiga e'tibor bilan qayta yig'ing; ko'rilmagan oila va maqsadda recall hamda qiyin zararsizlarda FPR ni o'lchang.
7. s7 shablon baholovchisini sohangiz uchun moslang: kamida 4 shablon, 36 holatli regressiya testi va yakuniy ball jadvali; qo'shimcha ravishda ssenariydagi murakkab talabni (ko'p bosqichli himoya / reranker / ablyatsiya / to'liq baholash hisoboti) bajaring.

Topshiriladi: Kod (.py, har qadam alohida); Bilim bazasi va savollar to'plami (JSON yoki .py); Natijalar jadvallari (fertillik, recall/MRR, aniqlik, AUC, injection recall); Grafiklar (kamida 3 ta); Xulosa (>= 150 so'z): haqiqiy LLM bilan nima o'zgarishi va qanday xavflar qolishi

Baholash: To'g'ri ishlaydigan kod — 25, Murakkab talab (ssenariydagi qo'shimcha vazifa) — 20, O'lchovlar va jadvallar — 20, Tahlil va xulosa — 20, Hisobot sifati — 15 (jami 100)
