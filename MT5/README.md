# MT5. LLM asosidagi AI tizimlarining muhandislik muammolari: prompt engineering, RAG arxitekturasi, LLM xavfsizligi va hallucination

📄 Bosma qo'llanma (bosqichlar, kod, mashqlar, 20 variant): `MT5_laboratoriya_qollanma.pdf`
🦉 Bosqichlar Suqrot usulida: har `sN_*.py` — karkas (TODO larni o'zingiz yozasiz), savollar shu faylda, ishoralar `ISHORALAR.md` da.
Bu mustaqil ishda talaba LLM tizimining asosiy muhandislik qismlarini katta modelsiz, CPU'da qo'lda quradi: BPE tokenizatorini o'qitib kontekst byudjetini hisoblaydi, KV-kesh xotirasini baholaydi, sintetik korporativ bilim bazasida bo'laklash va BM25/TF-IDF qidiruvini recall@k va MRR bilan o'lchaydi, ekstraktiv mini-RAG, hallucination va prompt injection detektorlarini yasaydi hamda prompt shablonlarini regressiya testi bilan baholaydi. Real labdan farqi: internet va haqiqiy LLM yo'q — generator ekstraktiv, LLM xatti-harakati qoidalar bilan simulyatsiya qilinadi; har qadamda «haqiqiy LLM bilan qanday bo'ladi» izohi berilgan. Barcha raqamlar kichik sintetik ma'lumotda olingan va taxminiy.

## Muhit

Muhit: Python 3.12, CPU (2 yadro, 1,5 GB), tarmoq yo'q, vaqt chegarasi 90 s. Kutubxonalar: torch, numpy, pandas, scipy, scikit-learn, matplotlib, fairlearn, shap, psutil, pyyaml, fastapi, httpx, tqdm, tabulate (transformers, sentence-transformers, faiss, nltk YO'Q — tokenizator, BM25 va detektorlar qo'lda yoziladi).

Mahalliy kompyuterda: `pip install -r ../requirements.txt` (yoki ildizdagi Dockerfile bilan). Saytdagi virtual laboratoriya: https://aiscan.airi.uz/ktm/#/c/mis2104/lab/MT5

## Bosqichlar

### 1-qadam. Tokenizatsiya: BPE ni qo'lda o'qitish va kontekst oynasi byudjeti

Byte-Pair Encoding (BPE) algoritmi kichik o'zbekcha va inglizcha korpusda qo'lda o'qitiladi: ikki tokenizator — «inglizcha markazli» (90% inglizcha matn) va «muvozanatli» (50/50). Ko'rilmagan jumlalarda belgi, so'z va BPE tokenlari soni hamda **fertillik** (token/so'z) taqqoslanadi. So'ng 8192 tokenli kontekst oynasida RAG prompti byudjeti hisoblanadi: tizim prompti, suhbat tarixi, javob zaxirasi va bo'laklar sig'imi; shartli narx bo'yicha kunlik xarajat. Talaba merge sonini, korpus nisbatini va CTX/CHUNK_WORDS ni o'zgartirib ko'radi.

Fayl: `s1_1_qadam_tokenizatsiya_bpe_ni_qolda_oqiti.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s1_1_qadam_tokenizatsiya_bpe_ni_qolda_oqiti.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Tokenizator 'xavfsizligi' so'zini qanday bo'laklarga ajratishi nimaga bog'liq: so'zning o'ziga yoki tokenizator qaysi matnda o'qitilganiga?
2. BPE ning bitta iteratsiyasida aniq nima sodir bo'ladi: nimani sanaysiz va nimani birlashtirasiz?
3. Agar eng ko'p uchragan juftlik atigi 1 marta uchrasa, birlashtirishni davom ettirish mantiqlimi?
4. Ko'rilmagan so'zni kodlaganda merges ro'yxatini qaysi tartibda qo'llash kerak va nega tartib muhim?
5. Fertillik (token/so'z) ni qanday hisoblaysiz va inglizcha-markazli tokenizatorda o'zbekcha fertillik inglizchadan necha baravar katta chiqishini kutasiz?
6. 8192 tokenli oynada bo'laklar uchun qancha joy qoladi va 200 so'zli bo'lak o'zbekchada necha tokenga teng?
7. Agar korpusni 50/50 qilsangiz yoki merge sonini 600 ga oshirsangiz, o'zbekcha fertillik va kunlik narx qanday o'zgaradi?

Kutilgan natija: Inglizcha-markazli tokenizatorda o'zbekcha fertillik inglizchadan ~2 baravar yuqori (bu kichik korpusda raqamlar katta chiqadi, taxminan 5 va 2,6); shu sabab o'zbekcha bo'laklardan promptga ~5 ta, inglizchadan ~12 ta sig'adi. Grafik: merge soni vs fertillik.

Tajriba qilib ko'ring:

- `corp_en_centric` dagi nisbatni 50/50 dan 10/90 gacha o'zgartiring: o'zbekcha fertillik qanday kamayadi?
- CTX=32768 va ANSWER_RESERVE=2000 qilib, o'zbekcha bo'laklar sonini qayta hisoblang.
- Ko'rilmagan so'zlarni (`xavfsizligi`, `shifrlangan`) muvozanatli tokenizator bilan bo'laklang va farqni yozing.

### 2-qadam. KV-kesh xotira kalkulyatori va keshning generatsiya tezligiga ta'siri

KV-kesh formulasi `2 x qatlam x KV_bosh x head_dim x ketma-ketlik x batch x bayt` ochiq e'lon qilingan konfiguratsiyalar (Llama-2-7B MHA, Llama-3-8B va 70B GQA) uchun hisoblanadi: token boshiga KB, 4k/32k/128k kontekstda GB, 2x24 GB GPU'da 8k kontekst bilan nechta so'rov sig'ishi (fp16 va int4 vaznlar). Keyin 4 qatlamli kichik decoder CPU'da KV-kesh bilan va keshsiz generatsiya qilinadi — natija tokenlari bir xilligi va tezlanish o'lchanadi.

Fayl: `s2_2_qadam_kv_kesh_xotira_kalkulyatori_va_k.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s2_2_qadam_kv_kesh_xotira_kalkulyatori_va_k.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Generatsiyada har yangi token uchun oldingi tokenlarning qaysi oraliq natijalari qayta kerak bo'ladi va nega aynan ularni saqlash mumkin?
2. KV-kesh hajmi qaysi kattaliklarga chiziqli bog'liq? Formuladagi '2' nimani bildiradi?
3. Llama-2-7B (32 KV bosh) va Llama-3-8B (8 KV bosh, GQA) da token boshiga KB necha baravar farq qilishi kerak? Q boshlar soni formulaga kiradimi?
4. 2x24 GB GPU'da 70B model fp16 vaznlari bilan nechta so'rovga joy qoladi? Bo'sh joy manfiy chiqsa nima qilasiz?
5. Kichik decoderda keshni qo'llashda yangi k, v ni qaysi o'q bo'yicha ulash kerak va bitta yangi token uchun causal maska nega shart emas?
6. Keshli generatsiyada yangi tokenning pozitsiya embeddingi uchun `start` qiymati qanday bo'lishi kerak? Xato bo'lsa natija qanday bilinadi?
7. Keshsiz generatsiya vaqti yangi tokenlar soniga qanday o'sadi, keshli esa qanday? 256 tokenda taxminan qancha tezlanish kutasiz?

Kutilgan natija: GQA (8 KV bosh) KV-keshni MHA ga nisbatan 4 baravar kamaytiradi (128 vs 512 KB/token); 70B fp16 2x24 GB ga sig'maydi. Kichik decoderda KV-kesh 256 token uchun ~10x tezlanish beradi (CPU'ga qarab o'zgaradi).

Tajriba qilib ko'ring:

- `bytes_per=1` (int8/fp8 KV-kesh) bilan so'rovlar sonini qayta hisoblang.
- Prompt uzunligini 64 dan 512 ga oshiring: keshsiz generatsiya vaqti qanday o'sadi (kvadratik)?
- MODELS lug'atiga o'zingiz tanlagan ochiq modelning config.json qiymatlarini qo'shing.

### 3-qadam. Bo'laklash (chunking) va qidiruv: BM25 (qo'lda), TF-IDF, gibrid RRF; recall@k va MRR

6 hujjatli sintetik korporativ bilim bazasi (24 fakt + 24 chalg'ituvchi jumla + umumiy shovqin) yasaladi. Olti bo'laklash strategiyasi (so'z oynasi 40/80/160, overlap bilan va siz; 2 va 4 jumlali) va uch qidiruv usuli — qo'lda yozilgan Okapi BM25 (sodda o'zbekcha qo'shimcha kesish bilan), char n-gram TF-IDF va Reciprocal Rank Fusion — taqqoslanadi. Har savol uchun oltin jumlani to'liq saqlagan bo'lak o'rni bo'yicha recall@1/3/5, MRR va top-3 kontekst hajmi (so'z) hisoblanadi.

Fayl: `s3_3_qadam_bolaklash_chunking_va_qidiruv_bm.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s3_3_qadam_bolaklash_chunking_va_qidiruv_bm.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Bu bosqichda 'to'g'ri topildi' deb nimani hisoblaymiz: bo'lakda javob soni borligini yoki oltin jumla to'liq borligini? Bu tanlov natijaga qanday ta'sir qiladi?
2. Overlap 0 va 40 so'zli oynada nega recall@5 ~0,83 da qoladi, overlap 10 qo'shilsa ~1,0 ga chiqadi?
3. BM25 formulasidagi k1 va b nimani boshqaradi? Uzun bo'lakda so'z ko'p uchrasa ball cheksiz o'saveradimi?
4. Nega so'z bo'yicha BM25 dan tashqari char n-gram TF-IDF ham sinaladi? O'zbekcha qo'shimchalar ('ta'tilga', 'ta'til') bunga qanday aloqador?
5. Ikki usul ballari turli shkalada bo'lsa (BM25 ~10, kosinus ~0,3), ularni qanday birlashtirasiz? Nega ballarning o'zi emas, o'rni ishlatiladi?
6. MRR qanday hisoblanadi va topilmagan savol unga qanday hissa qo'shadi? recall@k dan qanday farqi bor?
7. Katta bo'laklar MRR=1 beradi, lekin top-3 kontekst ~390 so'z. Qaysi strategiyani 'eng tejamkor' deysiz va buni qanday raqam bilan ifodalaysiz?

Kutilgan natija: Overlapsiz 40 so'zli bo'laklarda recall@5 ~0,83 (jumla kesiladi); overlap qo'shilsa ~1,0. Katta bo'laklar MRR=1 beradi, lekin top-3 kontekst ~390 so'z; 2 jumlali bo'laklar ~40 so'zda MRR ~0,9 — eng tejamkor.

Tajriba qilib ko'ring:

- BM25 da `k1` ni 0.8 va 2.0, `b` ni 0 va 1 qilib MRR ga ta'sirini kuzating.
- `tokens(..., use_stem=False)` bilan BM25 ni qayta baholang — qo'shimcha kesish qanchalik yordam beradi?
- `build_docs(filler_per_fact=8)` bilan shovqinni oshiring va qaysi usul barqarorroq ekanini aniqlang.

### 4-qadam. Mini-RAG: qidiruv, prompt yig'ish, ekstraktiv generator va rad etish chegarasi

To'liq RAG zanjiri yig'iladi: 3 jumlali bo'laklar, TF-IDF qidiruv, top-k bo'lak jumlalaridan savolga eng o'xshashini tanlovchi **ekstraktiv generator** (son so'ralsa sonli jumlaga bonus). Javobsiz 6 ta savol qo'shilib, rad etish chegarasi `tau` to'ri bo'yicha to'g'ri javob, noto'g'ri javob, asossiz rad va to'g'ri rad soni hisoblanadi. Oxirida haqiqiy LLM'ga yuboriladigan prompt (rol, tayanish ko'rsatmasi, manba identifikatorlari) chop etiladi.

Fayl: `s4_4_qadam_mini_rag_qidiruv_prompt_yigish_e.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s4_4_qadam_mini_rag_qidiruv_prompt_yigish_e.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Bizda katta LLM yo'q. Unda 'generator' sifatida nima ishlatiladi va bunday ekstraktiv javobning afzalligi va cheklovi nimada?
2. Qidiruv va jumla tanlash uchun bitta TF-IDF vektorizatorni nimalar ustida o'qitish kerak, toki jumla vektorlari ham bir xil fazoda bo'lsin?
3. Savolda 'necha', 'qancha' kabi so'z bo'lsa, nega sonli jumlaga kichik bonus berish foydali? Bu qanday xatoga olib kelishi mumkin?
4. k (bo'laklar soni) ni 1 dan 5 ga oshirsangiz, to'g'ri javoblar ulushi doim o'sadimi? Nega?
5. Javobsiz savolga tizim nima qilishi kerak va buni qaysi kattalik bilan boshqarasiz? Chegarani juda baland qo'ysangiz nima bo'ladi?
6. Umumiy aniqlikni qanday hisoblaysiz: javobli savollarda nima 'to'g'ri', javobsizlarda nima 'to'g'ri' hisoblanadi? tau ~0,3 da qancha kutasiz?
7. Haqiqiy LLM ga yuboriladigan promptda qaysi qismlar gallyutsinatsiyani kamaytiradi va javobni tekshirib bo'ladigan qiladi?

Kutilgan natija: Javobli savollarda aniqlik ~0,88; xatolar chalg'ituvchi jumlalardan («Kichik summali xaridlar…»). tau ~0,3 da umumiy aniqlik ~0,8; ba'zi javobsiz savollarga baribir noto'g'ri jumla qaytadi — leksik o'xshashlik chegarasining zaifligi.

Tajriba qilib ko'ring:

- Sonli jumla bonusini (0.05) olib tashlang va qaysi savollar buzilishini yozing.
- Generatorni 2 ta eng yaxshi jumlani birlashtiradigan qiling — aniqlik va javob uzunligi qanday o'zgaradi?
- Rad etishni sentence balli o'rniga top-1 va top-2 bo'lak ballari farqiga (margin) asoslang.

### 5-qadam. Hallucination detektori: javob jumlalarining manbaga tayanishi (leksik tekshiruv)

Har fakt uchun 6 turdagi javob yasaladi: to'g'ri, to'g'ri parafraz, son almashgan, obyekt almashgan, boshqa hujjatdan va to'qib chiqarilgan. Detektor NLI modeli o'rniga uch belgi ishlatadi: mazmunli so'zlar qamrovi, sonlar mosligi va manbada yo'q bigrammalar ulushi; ular «tayanish balli»ga birlashtiriladi. ROC-AUC (har belgi va birlashgan), chegara bo'yicha precision/recall, turlar kesimida aniqlash ulushi va aralash javobni jumla darajasida belgilash ko'rsatiladi.

Fayl: `s5_5_qadam_hallucination_detektori_javob_ju.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s5_5_qadam_hallucination_detektori_javob_ju.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Javob 'manbaga tayanmagan' degani nimani anglatadi? Bizda NLI modeli yo'q — qaysi kuzatiladigan belgilar bunga ishora qilishi mumkin?
2. Parol muddati '90' o'rniga '95' deb yozilgan javobda so'z qamrovi baribir yuqori chiqadi. Qaysi belgi bu xatoni ushlaydi va uni ballga qanday qo'shasiz?
3. Yangi bigrammalar ulushi nimani o'lchaydi va nega parafrazlarda u biroz oshadi?
4. Detektorni baholash uchun ijobiy (gallyutsinatsiya) va salbiy misollarni qanday yasaysiz? Qaysi tur eng qiyin bo'lishi mumkin?
5. ROC-AUC ni hisoblaganda ball yo'nalishi muhim: yuqori 'tayanish balli' ijobiy klassga mos keladimi yoki teskarisi?
6. Chegara 0,6 da son almashganlarning qancha qismi, obyekt almashganlarning qancha qismi va parafrazlarning qancha qismi 'asossiz' deb belgilanishini kutasiz?
7. Nega butun javobni emas, har jumlani alohida tekshirish foydali? Aralash javobda nimani ko'rishni kutasiz?

Kutilgan natija: Birlashgan ball AUC ~0,98; son almashganlarning ~94% aniqlanadi, obyekt almashganlarning faqat ~1/3 qismi (so'zlar boshqa hujjatda emas, shu hujjatda ham uchrashi mumkin); parafrazlarning ~8% noto'g'ri «asossiz» deb belgilanadi.

Tajriba qilib ko'ring:

- Chegara TH ni 0.4 va 0.8 qilib precision/recall murozasini jadvalga yozing.
- Obyekt almashtirishni aniqlash uchun savoldagi kalit so'z va javobdagi sub'ekt mosligini tekshiruvchi qoida qo'shing.
- Kontekstni butun hujjat o'rniga faqat top-3 bo'lak qiling — FP soni qanday o'zgaradi?

### 6-qadam. Prompt injection detektori: qoidalar + kichik klassifikator, ko'rilmagan hujum turlarida baholash

Besh oila (unut/ignore, tizim promptini so'rash, rol o'yini, bilvosita — hujjat ichida, niqoblangan) va 6 maqsaddan iborat sintetik injection'lar hamda zararsiz va «qiyin zararsiz» (o'xshash so'zli) so'rovlar yasaladi. Regex qoidalar, char n-gram TF-IDF + logistik regressiya va ularning OR birikmasi (A) tasodifiy bo'linishda, (B) ko'rilmagan oila va ko'rilmagan maqsadda baholanadi. Oxirida bilim bazasi jumlalariga yashirilgan bilvosita injection skanerlanadi.

Fayl: `s6_6_qadam_prompt_injection_detektori_qoida.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s6_6_qadam_prompt_injection_detektori_qoida.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Prompt injection ni oddiy foydalanuvchi so'rovidan nima ajratadi? Bu bosqichda qaysi 5 oila hujum turi bor va 'qiyin zararsiz' so'rovlar nima uchun kerak?
2. Regex qoidalar qaysi oilalarni yaxshi ushlaydi va qaysilarida (masalan, '1gn0re prev10us', harflar orasida bo'shliq) ojiz qoladi?
3. Nega so'z emas, char n-gram TF-IDF tanlanadi? Niqoblangan matn va o'zbek/ingliz aralashmasiga bu qanday yordam beradi?
4. Tasodifiy 70/30 bo'linishda klassifikator F1 ~1,0 berishi nega aldamchi bo'lishi mumkin? Testdagi misollar o'qitishdagilardan qanchalik farq qiladi?
5. Ko'rilmagan hujumga umumlashtirishni qanday adolatli o'lchaysiz? O'qitish va test to'plamini qaysi ikki o'lchov bo'yicha ajratasiz?
6. B) sinovda klassifikator recall'i qancha tushishini kutasiz va qoidalar bilan OR birikmasi nima beradi?
7. Bilvosita injection hujjat ichida yashirilgan bo'lsa, uni qayerda va qachon tekshirish kerak? Skaner zararsiz jumlani ham belgilasa nima qilasiz?

Kutilgan natija: Tasodifiy bo'linishda klassifikator F1 ~1,0 (aldamchi!), ko'rilmagan oila+maqsadda uning recall'i 0–0,25 gacha tushadi; qoidalar ~0,67–1,0, lekin qiyin zararsizlarda FPR ~0,2. Skanerda zaharlangan jumla topiladi, bitta zararsiz jumla ham shubhali chiqadi.

Tajriba qilib ko'ring:

- RULES ro'yxatidan maqsadga oid qoidalarni (api key, attacker@) olib tashlang — B jadvali qanday o'zgaradi?
- Klassifikatorga so'z darajasidagi TF-IDF qo'shing (FeatureUnion) va B tajribasini takrorlang.
- Niqoblangan oila uchun normalizatsiya (1->i, 0->o, bo'shliqlarni olib tashlash) qo'shib recall'ni oshiring.

### 7-qadam. Prompt shablonlari va baholovchi: statik linter + LLM simulyatori bilan regressiya testi

Besh shablon (naiv, rol, tayanish, tayanish+himoya+JSON, ortiqcha uzun) statik linter bilan 7 element (rol, tayanish, rad etish, ajratgich, injection himoyasi, manba, format) bo'yicha tekshiriladi. So'ng 36 holatli regressiya to'plami (24 javobli, 6 javobsiz, 6 zaharlangan kontekst) **LLM simulyatori** orqali yuritiladi: aniqlik, to'g'ri rad, to'qish, injection muvaffaqiyati, manba, JSON format va token narxi; vaznli yakuniy ball hisoblanadi. Simulyator haqiqiy model emas — ko'rsatmalarga bo'ysunishni qoidalar bilan modellashtiradi.

Fayl: `s7_7_qadam_prompt_shablonlari_va_baholovchi.py` — karkas: `TODO` joylarini o'zingiz to'ldiring, so'ng `python3 s7_7_qadam_prompt_shablonlari_va_baholovchi.py`

Yo'naltiruvchi savollar (avval o'zingiz javob bering):

1. Prompt shablonini 'yaxshi' deb baholash uchun qanday ikki xil tekshiruv bor: matnning o'zini ko'rib chiqish va uni model bilan sinash. Ular nimasi bilan farq qiladi?
2. Linter har bir elementni qanday aniqlaydi? Masalan, 'injection himoyasi' bor-yo'qligini shablondagi qaysi so'z ko'rsatadi?
3. Simulyator injection ga qachon 'bo'ysunadi'? Faqat himoya ko'rsatmasi yetarlimi yoki kontekst ajratgichi ham kerakmi — nega?
4. Javobsiz savolda tayanish va rad etish ko'rsatmasi bo'lmagan shablon bilan simulyator nima qaytaradi va bu qaysi metrikaga tushadi?
5. JSON formatdagi javobda to'g'rilikni qanday tekshirasiz? Format buzilsa nima qilasiz?
6. Yakuniy ballda qaysi metrikalarga qancha vazn berasiz va nega ortiqcha uzun shablon (T5) jarimalanishi kerak?
7. Qaysi shablon eng yuqori ball olishini kutasiz (taxminan nechchi) va haqiqiy LLM bilan bu natijani bir marta yuritish yetarlimi?

Kutilgan natija: T1/T2 javobsiz savollarning yarmiga to'qilgan javob beradi va injection'ga 100% bo'ysunadi; T3 to'qishni yo'qotadi, lekin injection qoladi; T4 injection 0 va JSON 100% — eng yuqori ball (~84). T5 xuddi T4 natijasini ikki baravar ko'p token bilan beradi.

Tajriba qilib ko'ring:

- Yakuniy ball vaznlarini o'zgartiring (masalan, injection 0.4) — reyting o'zgaradimi?
- Simulyatorga ehtimollik qo'shing: ko'rsatma bo'lsa ham 10% hollarda unga bo'ysunmasin; har shablonni 5 marta yuritib o'rtacha va std ni hisoblang.
- O'z shabloningizni (T6) yozing va T4 dan kam token bilan bir xil ball oling.

## Qo'shimcha mashqlar (10 ta, jami 100 ball)

Laboratoriya bosqichlari asosida bajariladigan kichik mashqlar.

| № | Mashq | Daraja | Ball |
|---|---|---|---|
| 1 | **Fertillik va narx jadvali** — 1-qadam kodida o'z mavzungizdagi 3 ta o'zbekcha va 3 ta inglizcha jumlani test to'plamiga qo'shing. Ikki tokenizator (100/300/600 merge) uchun fertillik jadvalini tuzing va kuniga 50 000 so'rov (har biri ~800 so'z) uchun o'zbekcha va inglizcha holatda shartli narx farqini hisoblang. | oson | 5 |
| 2 | **KV-kesh rejasi** — 2-qadam kalkulyatoridan foydalanib Llama-3-8B (GQA) uchun 1x24 GB, 2x24 GB va 1x80 GB GPU'da 4k, 16k, 32k kontekstda bir vaqtdagi so'rovlar sonini fp16 va int4 vaznlar, fp16 va int8 KV-kesh bilan hisoblang (jadval) va qaysi konfiguratsiya 20 ta parallel 16k so'rovga yetishini yozing. | oson | 5 |
| 3 | **BM25 parametrlari** — 3-qadamda jumla-2 bo'laklash uchun k1 = 0.5, 1.0, 1.5, 2.0 va b = 0, 0.5, 0.75, 1 to'rida BM25 MRR qiymatlarini hisoblang, 4x4 issiqlik xaritasini chizing va eng yaxshi juftlikni asoslang. | oson | 5 |
| 4 | **Semantik bo'laklash** — 3-qadamga yangi bo'laklash strategiyasini qo'shing: qo'shni jumlalar TF-IDF o'xshashligi chegaradan past tushganda yangi bo'lak boshlansin (semantik chunking). Uni so'z-80/20 va jumla-2 bilan recall@1, MRR va top-3 kontekst hajmi bo'yicha taqqoslang (jadval + 3 jumlali xulosa). | o'rta | 10 |
| 5 | **Qayta tartiblovchi (reranker)** — 4-qadam pipeline'iga ikkinchi bosqich qo'shing: top-10 bo'lak savol bilan umumiy so'zlar soni, sonlar mosligi va savol so'zi turi (kim/qancha/qachon) belgilariga asoslangan logistik regressiya bilan qayta tartiblansin (o'qitish uchun faktlarning yarmi, test uchun qolgani). Reranker bilan va siz aniqlikni solishtiring. | o'rta | 10 |
| 6 | **Detektor kalibrlash** — 5-qadam detektori uchun chegara TH ni 0.3 dan 0.9 gacha 0.05 qadam bilan o'zgartirib precision, recall, F1 egri chiziqlarini chizing; F1 maksimal bo'lgan va recall >= 0.95 bo'lgan chegaralarni toping. Har gallyutsinatsiya turi uchun eng ko'p o'tkazib yuborilgan misollarni tahlil qiling. | o'rta | 10 |
| 7 | **Injection'ga qarshi normalizatsiya** — 6-qadamga kirish matnini normalizatsiya qiluvchi funksiya qo'shing (leetspeak 1->i, 0->o, 3->e; harflar orasidagi bo'shliqlarni olib tashlash; katta-kichik harf; base64 bo'lakni dekodlash). B tajribasini normalizatsiya bilan va siz takrorlab, oila bo'yicha recall jadvalini va qiyin zararsizlardagi FPR ni taqqoslang. | o'rta | 10 |
| 8 | **To'liq RAG baholash hisoboti** — 3–5 qadam kodlarini birlashtirib bitta skript yozing: 24 javobli + 6 javobsiz savol uchun qidiruv (recall@3, MRR), javob aniqligi, rad etish to'g'riligi va har javob jumlasi uchun tayanish balli (o'rtacha faithfulness) hisoblansin. Natijani Markdown jadval ko'rinishida chop eting va 3 xil bo'laklash strategiyasi uchun taqqoslang. Umumiy ish vaqti 60 s dan oshmasin. | qiyin | 15 |
| 9 | **Ko'p bosqichli injection himoyasi** — 6 va 7-qadamlarni birlashtiring: qidirilgan bo'laklar avval injection detektoridan o'tsin (shubhali bo'lak olib tashlanadi), so'ng T3 shabloni bilan simulyatorga berilsin. Zaharlangan bo'laklar ulushi 0%, 10%, 30% bo'lganda injection muvaffaqiyati, javob aniqligi va noto'g'ri olib tashlangan toza bo'laklar ulushini o'lchang; «faqat shablon himoyasi», «faqat filtr» va «ikkalasi» variantlarini jadvalda solishtiring. | qiyin | 15 |
| 10 | **Prompt shablonini avtomatik tanlash** — 7-qadam simulyatoriga tasodifiylik qo'shing (har ko'rsatmaga 85–95% bo'ysunish ehtimoli). 7 element (rol, tayanish, rad etish, ajratgich, himoya, manba, format) kombinatsiyalaridan kamida 16 ta shablon yasab, har birini 5 marta yurgizing; yakuniy ball o'rtachasi va std ni hisoblang, eng yaxshi shablonni token narxi bilan Pareto chizig'ida ko'rsating va qaysi element eng katta hissa qo'shishini (ablyatsiya) aniqlang. | qiyin | 15 |

Hisobot: har topshiriq uchun kod + natija jadvali/grafik + 3–5 jumla xulosa; o'qituvchiga belgilangan tartibda topshiriladi.

## Mustaqil topshiriqlar — 20 ta variant

Har talaba o'z variantini bajaradi (jurnal raqami bo'yicha; 20 dan katta bo'lsa — 20 ga bo'lgandagi qoldiq, 0 → 20). To'liq matn pastda (va `VARIANTS.md` da).


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
