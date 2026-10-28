"""MT5 — 1-qadam. Tokenizatsiya: BPE ni qo'lda o'qitish va kontekst oynasi byudjeti

Byte-Pair Encoding (BPE) algoritmi kichik o'zbekcha va inglizcha korpusda qo'lda o'qitiladi: ikki tokenizator — «inglizcha markazli» (90% inglizcha matn) va «muvozanatli» (50/50). Ko'rilmagan jumlalarda belgi, so'z va BPE tokenlari soni hamda **fertillik** (token/so'z) taqqoslanadi. So'ng 8192 tokenli kontekst oynasida RAG prompti byudjeti hisoblanadi: tizim prompti, suhbat tarixi, javob zaxirasi va bo'laklar sig'imi; shartli narx bo'yicha kunlik xarajat. Talaba merge sonini, korpus nisbatini va CTX/CHUNK_WORDS ni o'zgartirib ko'radi.

Kutilgan natija: Inglizcha-markazli tokenizatorda o'zbekcha fertillik inglizchadan ~2 baravar yuqori (bu kichik korpusda raqamlar katta chiqadi, taxminan 5 va 2,6); shu sabab o'zbekcha bo'laklardan promptga ~5 ta, inglizchadan ~12 ta sig'adi. Grafik: merge soni vs fertillik.
"""
# 1-qadam. Tokenizatsiya (BPE ni qo'lda o'qitish) va kontekst oynasi byudjeti
# Maqsad: LLM matnni so'z emas, TOKEN bilan o'lchaydi. Tokenizator qaysi tilda o'qitilgan bo'lsa,
# o'sha tilda tejamkor; boshqa tilda (masalan, o'zbekcha) bir so'z ko'p tokenga bo'linadi -> kontekst tez to'ladi.
import re, collections, math
import matplotlib.pyplot as plt

UZ = """Sun'iy intellekt tizimlari korxonada hujjatlarni qidirish va savollarga javob berish uchun ishlatiladi.
Xodim savol yozadi, tizim tegishli hujjat bo'laklarini topadi va til modeli javob tayyorlaydi.
Til modeli matnni tokenlarga ajratadi va har bir token uchun keyingi tokenni bashorat qiladi.
Kontekst oynasi cheklangan bo'lgani uchun tizim faqat eng muhim bo'laklarni promptga qo'shadi.
Hisobotlar, buyruqlar va yo'riqnomalar o'zbek tilida yozilgan, shuning uchun tokenizator samaradorligi muhim.
Ta'til uchun ariza kamida o'n kun oldin bo'lim boshlig'iga topshiriladi.
Xavfsizlik qoidalariga ko'ra parollar har to'qson kunda almashtirilishi kerak.
Server xonasiga kirish faqat ruxsatnomasi bor xodimlar uchun ochiq.
Moliyaviy hisobotlar har chorak oxirida buxgalteriyaga taqdim etiladi.
Yangi xodimlar birinchi hafta davomida axborot xavfsizligi bo'yicha o'qitiladi.
Mijozlarning shaxsiy ma'lumotlari shifrlangan holda saqlanadi va uchinchi shaxslarga berilmaydi.
Model javobi manbaga tayanmasa, foydalanuvchiga ogohlantirish ko'rsatiladi."""
EN = """Artificial intelligence systems are used in the company to search documents and answer questions.
An employee writes a question, the system finds relevant document chunks and the language model prepares an answer.
The language model splits text into tokens and predicts the next token for each position.
Because the context window is limited, the system adds only the most important chunks to the prompt.
Reports, orders and instructions are written in English, so tokenizer efficiency matters.
A vacation request must be submitted to the head of department at least ten days in advance.
According to security rules, passwords must be changed every ninety days.
Access to the server room is open only to employees with a permit.
Financial reports are submitted to accounting at the end of each quarter.
New employees are trained on information security during their first week.
Personal data of customers is stored encrypted and is not shared with third parties.
If the model answer is not grounded in the source, a warning is shown to the user."""

def words(text):
    return re.findall(r"[\w']+|[^\w\s]", text.lower())

def train_bpe(corpus_words, n_merges):
    """Klassik BPE (Sennrich, 2016): eng ko'p uchragan qo'shni juftlikni ketma-ket birlashtirish."""
    vocab = collections.Counter(tuple(w) + ("</w>",) for w in corpus_words)
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for w, f in vocab.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        if pairs[best] < 2:
            break
        merges.append(best)
        new = collections.Counter()
        for w, f in vocab.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1]); i += 2
                else:
                    out.append(w[i]); i += 1
            new[tuple(out)] += f
        vocab = new
    return merges

def encode_word(w, merges):
    sym = list(w) + ["</w>"]
    for a, b in merges:                       # birlashtirishlar o'qitish tartibida qo'llanadi
        i, out = 0, []
        while i < len(sym):
            if i < len(sym) - 1 and sym[i] == a and sym[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(sym[i]); i += 1
        sym = out
    return sym

def n_tokens(text, merges):
    return sum(len(encode_word(w, merges)) for w in words(text))

uz_lines, en_lines = UZ.split("\n"), EN.split("\n")
# o'qitish: birinchi 9 jumla, test: oxirgi 3 jumla (tokenizator ularni ko'rmagan)
tr_uz, te_uz = " ".join(uz_lines[:9]), " ".join(uz_lines[9:])
tr_en, te_en = " ".join(en_lines[:9]), " ".join(en_lines[9:])
corp_en_centric = words(tr_en) * 9 + words(tr_uz) * 1     # 90% inglizcha — "inglizcha markazli" tokenizator
corp_balanced = words(tr_en) * 5 + words(tr_uz) * 5        # 50/50

print("=== Tokenlar soni: belgi / so'z / BPE ===")
print(f"{'tokenizator':<26}{'merge':>6}{'UZ token':>10}{'UZ tok/so`z':>12}{'EN token':>10}{'EN tok/so`z':>12}")
nw_uz, nw_en = len(words(te_uz)), len(words(te_en))
print(f"{'belgi (char)':<26}{'-':>6}{len(te_uz):>10}{len(te_uz)/nw_uz:>12.2f}{len(te_en):>10}{len(te_en)/nw_en:>12.2f}")
print(f"{'so`z (whitespace)':<26}{'-':>6}{nw_uz:>10}{1:>12.2f}{nw_en:>10}{1:>12.2f}")
res = {}
for name, corp in [("BPE inglizcha-markazli", corp_en_centric), ("BPE muvozanatli", corp_balanced)]:
    for m in (100, 300, 600):
        mg = train_bpe(corp, m)
        tu, tn = n_tokens(te_uz, mg), n_tokens(te_en, mg)
        res[(name, m)] = (tu / nw_uz, tn / nw_en)
        print(f"{name:<26}{len(mg):>6}{tu:>10}{tu/nw_uz:>12.2f}{tn:>10}{tn/nw_en:>12.2f}")
mg = train_bpe(corp_en_centric, 300)
print("\nNamuna (inglizcha-markazli, 300 merge):")
for w in ["xavfsizligi", "shifrlangan", "security", "encrypted"]:
    print(f"  {w:<14} -> {encode_word(w, mg)}")

# ---- Kontekst oynasi byudjeti ----
print("\n=== Kontekst oynasi byudjeti (RAG prompti) ===")
CTX = 8192            # model kontekst oynasi (token)
SYSTEM, QUESTION, ANSWER_RESERVE = 350, 60, 700
HISTORY_TURNS, TOK_PER_TURN = 6, 120
CHUNK_WORDS = 200
for name, (fu, fe) in [("inglizcha matn", (None, res[("BPE inglizcha-markazli", 300)][1])),
                       ("o'zbekcha matn", (res[("BPE inglizcha-markazli", 300)][0], None))]:
    fert = fu if fu else fe
    chunk_tok = math.ceil(CHUNK_WORDS * fert)
    free = CTX - SYSTEM - QUESTION - ANSWER_RESERVE - HISTORY_TURNS * TOK_PER_TURN
    k_max = free // chunk_tok
    print(f"{name:<16} fertillik={fert:.2f} tok/so'z  bo'lak={chunk_tok} token  bo'sh joy={free}  -> max k = {k_max} bo'lak")
PRICE_IN, PRICE_OUT = 0.50, 1.50       # SHARTLI narx, $ / 1M token (o'quv misoli, real narx emas)
req_in, req_out, n_req = 5000, 400, 20000
cost = n_req * (req_in * PRICE_IN + req_out * PRICE_OUT) / 1e6
print(f"Kunlik {n_req} so'rov x ({req_in} kirish + {req_out} chiqish token) ~ ${cost:.1f}/kun (shartli narxda)")

fig, ax = plt.subplots(figsize=(7, 3.5))
ms = [100, 300, 600]
for name, st in [("BPE inglizcha-markazli", "-o"), ("BPE muvozanatli", "--s")]:
    ax.plot(ms, [res[(name, m)][0] for m in ms], st, label=f"{name}: UZ")
    ax.plot(ms, [res[(name, m)][1] for m in ms], st, alpha=.5, label=f"{name}: EN")
ax.set_xlabel("merge soni (lug'at hajmi)"); ax.set_ylabel("token / so'z"); ax.legend(fontsize=7); ax.grid(alpha=.3)
ax.set_title("Tokenizator fertilligi")
plt.tight_layout()
print("\nIzoh: haqiqiy LLM'larda (GPT, Llama, Qwen) lug'at 32k-150k token; o'zbekcha matn odatda inglizchadan")
print("sezilarli ko'p token oladi (aniq nisbat tokenizatorga bog'liq) — kontekst va narx hisobida buni hisobga oling.")
