# MT5. Ishoralar (Suqrot usuli)

Avval savolga o'zingiz javob bering; ishorani faqat qiynalganda o'qing. To'liq namuna yechim saytdagi virtual laboratoriyada bosqichni bajarganingizdan keyin ochiladi.

## 1-qadam. Tokenizatsiya: BPE ni qo'lda o'qitish va kontekst oynasi byudjeti

1. BPE lug'ati korpusdagi chastotalardan quriladi: korpusda ko'p uchragan harf birikmalari bitta tokenga aylanadi. Korpusning 90% i inglizcha bo'lsa, qaysi birikmalar g'olib chiqadi?
2. Har so'z — belgilar kortejidan iborat (oxirida '</w>'), so'z chastotasi bilan. Barcha qo'shni juftliklar chastotasini yig'ing, eng kattasini tanlang va lug'atdagi barcha so'zlarda shu juftlikni bitta belgiga almashtiring.

   ```python
   pairs = collections.Counter()
   for w, f in vocab.items():
       for a, b in zip(w, w[1:]): pairs[(a, b)] += f
   ```
3. Bunday birlashtirish faqat bitta so'zni 'yodlab' oladi. Namunada pairs[best] < 2 bo'lsa tsikl to'xtaydi — shu sabab haqiqiy merge soni so'ralgandan kam chiqishi mumkin.
4. Birlashtirishlar o'qitishdagi tartibda qo'llanadi: keyingi merge oldingi merge natijasiga tayanadi (masalan, ('x','a') -> 'xa', keyin ('xa','v')).
5. Test matnidagi BPE tokenlari sonini words() bergan so'zlar soniga bo'ling. Kutilgan natija: o'zbekcha ~5, inglizcha ~2,6 — taxminan 2 baravar.
6. Avval tizim prompti, savol, javob zaxirasi va 6 ta tarix navbatini ayiring; so'ng bo'sh joyni ceil(200 x fertillik) ga butun bo'ling. Kutilgan: o'zbekcha ~5, inglizcha ~12 bo'lak.

   ```python
   free = CTX - SYSTEM - QUESTION - ANSWER_RESERVE - HISTORY_TURNS * TOK_PER_TURN
   k_max = free // math.ceil(CHUNK_WORDS * fert)
   ```
7. Muvozanatli korpusda o'zbekcha birikmalar ham lug'atga kiradi — token kam, demak bir xil matn arzonroq va kontekstga ko'proq bo'lak sig'adi.

Bajargach o'ylang: Sizning BPE ingiz namunadagi bilan bir xil merge sonini berdimi? Fertillik farqi RAG tizimida kontekst va narx hisobini qanday o'zgartiradi — o'zbekcha hujjatlar uchun qanday tokenizator tanlagan bo'lardingiz?

## 2-qadam. KV-kesh xotira kalkulyatori va keshning generatsiya tezligiga ta'siri

1. Attention'da yangi tokenning Q si oldingi BARCHA tokenlarning K va V ga qaraydi; ular o'zgarmaydi (causal), shuning uchun bir marta hisoblab saqlanadi.
2. Har qatlam, har KV bosh, head_dim o'lchamli vektor, har token va har so'rov uchun alohida — K va V ikkalasi. fp16 da har son 2 bayt.

   ```python
   2 * L * kv * hd * seq * batch * bytes_per
   ```
3. Formulada faqat KV boshlar soni bor: 32/8 = 4 baravar (512 vs 128 KB/token). GQA da bir nechta Q bosh bitta K/V juftini bo'lishadi.
4. Vazn = 70.6e9 x 2 bayt ~ 131 GB — 48 GB dan katta. Bo'sh joy <= 0 bo'lsa so'rovlar soni 0; int4 (0,5 bayt) bilan qayta hisoblang.
5. Tensor shakli (B, H, T, head_dim) — vaqt o'qi dim=2. Bitta yangi token barcha oldingi tokenlarga qarashi mumkin, kelajak tokenlar keshda yo'q.

   ```python
   k = torch.cat([cache["k"], k], 2)
   ```
6. Yangi token ketma-ketlikdagi oxirgi o'rinda: start = ids.shape[1] - 1. Xato bo'lsa 'Natija tokenlari bir xilmi' True bo'lmaydi.
7. Keshsiz har qadamda butun ketma-ketlik qayta hisoblanadi (umumiy ish ~kvadratik), keshli — faqat bitta token. Kutilgan ~10x (CPU ga bog'liq).

Bajargach o'ylang: Sizning kesh mantiqingiz namunadagidan nimasi bilan farq qiladi? Real serverlarda (vLLM) KV-kesh nega sahifalab boshqariladi va GQA/int8 KV-kesh qaysi cheklovni yumshatadi?

## 3-qadam. Bo'laklash (chunking) va qidiruv: BM25 (qo'lda), TF-IDF, gibrid RRF; recall@k va MRR

1. Namunada bo'lak oltin jumlani TO'LIQ (s in text) saqlasagina relevant. Bo'lak chegarasi jumlani kesib qo'ysa, u hech bir bo'lakda to'liq bo'lmaydi.
2. Jumla oyna chegarasiga to'g'ri kelsa ikkiga bo'linadi. Overlap qo'shni bo'laklarni ustma-ust qiladi — qisqa jumla hech bo'lmasa bittasida butun qoladi.

   ```python
   step = max(1, size - overlap)
   for i in range(0, len(w), step): ...
   ```
3. k1 — chastotaning to'yinishi (f ortgani sari hissa (k1+1) ga yaqinlashadi), b — uzunlik normallashtirishi (len/avg). idf = log(1 + (N-n+0.5)/(n+0.5)).
4. Char 3-5-gramlar so'z o'zagi umumiy bo'lsa ham moslik beradi — sodda stem() ilmasligi mumkin bo'lgan shakllarni qamraydi.

   ```python
   TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
   ```
5. RRF: har usulda bo'lak o'rnini topib, 1/(60 + rank) larni qo'shing — shkala muammosi yo'qoladi.

   ```python
   ranks = np.empty(len(s)); ranks[np.argsort(-s)] = np.arange(1, len(s) + 1)
   ```
6. Har savol uchun 1/rank (topilmasa 0), so'ng o'rtacha. recall@k faqat 'top-k ichidami' deydi, MRR esa birinchi o'rinni ham qadrlaydi.
7. MRR ni top-3 kontekst so'zlari soniga nisbatini solishtiring; kutilgan: 2 jumlali bo'laklar ~40 so'zda MRR ~0,9.

Bajargach o'ylang: Sizning BM25 ingiz namunadagi bilan bir xil MRR berdimi? Real RAG da TF-IDF o'rniga dense embedding (bge-m3) va reranker qo'shilsa, qaysi bo'laklash strategiyasini tanlardingiz va nega?

## 4-qadam. Mini-RAG: qidiruv, prompt yig'ish, ekstraktiv generator va rad etish chegarasi

1. Javob — qidirilgan bo'laklardagi mavjud jumlalardan biri. U yangi narsa 'to'qiy olmaydi', lekin ikki jumlani birlashtira ham olmaydi.
2. Namunada vektorizator bo'laklar matni + ularning alohida jumlalari ustida fit qilinadi; so'ng bo'lak matritsasi M = VEC.transform(texts).

   ```python
   sc = (M @ VEC.transform([q]).T).toarray().ravel()
   ```
3. Ko'p savollar son so'raydi, shovqin jumlalarda esa son yo'q. Lekin chalg'ituvchi jumlalarda ham son bor ('Kichik summali xaridlar...', '30 kun saqlanadi').
4. Ko'proq bo'lak — oltin jumla ichida bo'lish ehtimoli ko'proq, lekin chalg'ituvchi jumlalar ham ko'payadi. Kutilgan k=3 da ~0,88.
5. Eng yaxshi jumla balli tau dan past bo'lsa — rad. tau baland bo'lsa javobli savollar ham asossiz rad qilinadi; to'r bo'yicha umumiy aniqlikni maksimallang.

   ```python
   for tau in np.round(np.arange(0.10, 0.46, 0.05), 2): ...
   ```
6. (to'g'ri javoblar + to'g'ri radlar) / (24 + 6). Kutilgan ~0,8; ba'zi javobsiz savollarga baribir noto'g'ri jumla qaytadi.
7. Rol, 'faqat KONTEKST ga tayan', 'topilmasa — Ma'lumot topilmadi de', har bo'lak oldida [Hujjat#n] identifikatori va aniq KONTEKST BOSHI/OXIRI ajratgichlari.

Bajargach o'ylang: Sizning tau tanlovingiz namunadagi bilan mos keldimi? Leksik o'xshashlik chegarasi javobsiz savollarda nega zaif va uni nima bilan (reranker, NLI, LLM-hakam) mustahkamlash mumkin?

## 5-qadam. Hallucination detektori: javob jumlalarining manbaga tayanishi (leksik tekshiruv)

1. Javobdagi mazmunli so'zlar kontekstda bormi, javobdagi sonlar kontekstda bormi, javobda kontekstda uchramaydigan yangi so'z birikmalari qanchalik ko'p.
2. Sonlar mosligi: javobdagi har bir son kontekstdagi sonlar ichida bo'lishi kerak. Mos kelmasa ballni keskin kamaytiring (masalan, x0,3).

   ```python
   num_ok = all(n in numbers(ctx) for n in numbers(sent))
   ```
3. Jumla qo'shni so'z juftlaridan kontekstda uchramaganlari ulushi. Parafrazda kirish so'zi qo'shilib, bitta so'z tushiriladi — bir-ikki yangi juftlik paydo bo'ladi.

   ```python
   big = set(zip(ct, ct[1:]))
   ```
4. Har fakt: to'g'ri, parafraz (0) va son/obyekt almashgan, boshqa hujjatdan, to'qilgan (1). Obyekt almashganda yangi so'z shu hujjatning boshqa joyida ham uchrashi mumkin.
5. 1 = gallyutsinatsiya bo'lgani uchun roc_auc_score(y, 1 - S) ishlating; bigrammalar ulushi esa to'g'ridan-to'g'ri. Kutilgan birlashgan AUC ~0,98.

   ```python
   roc_auc_score(y, 1 - S)
   ```
6. Kutilgan: son almashgan ~94%, obyekt almashgan ~1/3, parafrazlarning ~8% i noto'g'ri belgilanadi. Turlar bo'yicha pred[idx].mean() ni chiqaring.
7. Bitta to'g'ri jumla butun javob ballini ko'tarib yuborishi mumkin. sentences() bilan bo'lib, har birini [OK]/[ASOSSIZ] deb belgilang — to'qilgan jumla ajralib chiqishi kerak.

Bajargach o'ylang: Sizning ball formulangiz namunadagidan farq qiladimi va AUC qanday chiqdi? Real tizimda NLI yoki 'LLM-hakam' qaysi turdagi xatolarni (sinonim, obyekt almashuvi) yaxshiroq ushlaydi?

## 6-qadam. Prompt injection detektori: qoidalar + kichik klassifikator, ko'rilmagan hujum turlarida baholash

1. Injection modelning oldingi ko'rsatmalarini bekor qilib, boshqa maqsadga yo'naltiradi. 'Parolimni unutdim' kabi so'rovlar o'xshash so'zli, lekin zararsiz — ular FPR ni tekshiradi.
2. Qoida faqat muallif oldindan bilgan iborani ushlaydi. Niqoblangan va bilvosita oilalar uchun alohida qoidalar yozing va qiyin zararsizlarda FPR ni kuzating.

   ```python
   re.search(r"ignore (all )?(previous|prior|above)", t.lower())
   ```
3. Belgi n-gramlari qisman buzilgan so'zlarda ham umumiy bo'laklarni topadi. min_df=2 va class_weight='balanced' sinflar nomutanosibligini yumshatadi.

   ```python
   make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)), LogisticRegression(...))
   ```
4. Bir xil shablon va maqsad iboralari train va testga tushadi — model shablonni 'yodlab' oladi, hujum g'oyasini emas.
5. Leave-one-family-out: test oilasi o'qitishda yo'q, va maqsadlar ham ajratilgan (train 0-2, test 3-5). Faqat injection lar bo'lgani uchun recall = pred.mean().

   ```python
   tr = (fam != f) & (gl < 3); te = (fam == f) & (gl >= 3)
   ```
6. Kutilgan: klassifikator 0-0,25 gacha, qoidalar ~0,67-1,0. OR ikkalasining kuchini birlashtiradi, lekin FPR ham qo'shiladi.
7. Qidirilgan bo'laklar promptga qo'shilishidan oldin skanerlanadi. Detektor — faqat bir qatlam: imtiyozlarni cheklash, kontekstni ajratish, chiqish filtri bilan birga.

Bajargach o'ylang: Sizning qoidalaringiz va klassifikatoringiz B) sinovda namunaga nisbatan qanday natija berdi? Nega baholashni doim ko'rilmagan hujum turlarida qilish kerak?

## 7-qadam. Prompt shablonlari va baholovchi: statik linter + LLM simulyatori bilan regressiya testi

1. Statik linter — shablonda kerakli elementlar (rol, tayanish, rad etish, ajratgich, himoya, manba, format) bormi; regressiya testi — shu shablon bilan javoblar qanday chiqadi.
2. Har element uchun shablon konstantalaridagi o'ziga xos ibora qidiriladi: 'Faqat KONTEKST', 'topilmadi', '=== KONTEKST', 'bajarma', '[manba]', 'JSON'.

   ```python
   {"tayanish": "Faqat KONTEKST" in t, "format": "JSON" in t, ...}
   ```
3. Namunada injection faqat himoya VA ajratgich bo'lganda e'tiborsiz qoldiriladi: model qayerdan 'ma'lumot' boshlanishini bilishi kerak. T3 da himoya yo'q — injection qoladi.
4. Ball 0,30 dan past bo'lsa: rad etish ko'rsatmasi bor — 'Ma'lumot topilmadi', yo'q — to'qilgan '30 kun' javobi (to'qish). Kutilgan T1/T2 da javobsizlarning yarmi to'qiladi.
5. json.loads bilan o'qing, muvaffaqiyatli bo'lsa format_ok ni oshiring va 'javob' maydonini tekshiring; xato bo'lsa xom matn bilan davom eting.

   ```python
   try: obj = json.loads(out); text = obj["javob"] or ""
   except Exception: text = out
   ```
6. Namunada: 0,35 aniqlik + 0,2 to'g'ri rad + 0,25 (1 - injection) + 0,1 manba + 0,1 format, minus 300 tokendan oshgan har token uchun 0,02. T5 T4 natijasini ~2 baravar token bilan beradi.
7. Kutilgan T4 ~84. Haqiqiy LLM ehtimoliy — har holatni bir necha marta yuritib, o'rtacha va tarqoqlikni hisoblash kerak.

Bajargach o'ylang: Sizning simulyatoringiz va ball formulangiz namunadagi tartibni (T4 > T3 > T2 ~ T1) takrorladimi? Haqiqiy LLM bilan ishlaganda bu regressiya to'plamini qachon va qanday qayta ishga tushirasiz?
