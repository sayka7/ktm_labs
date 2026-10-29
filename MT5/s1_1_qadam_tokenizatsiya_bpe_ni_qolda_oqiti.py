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

# TODO 1: klassik BPE — so'zlarni belgilar ketma-ketligi (+ "</w>") ko'rinishida sanang, har iteratsiyada
#         eng ko'p uchragan qo'shni juftlikni topib birlashtiring; merges ro'yxatini qaytaring
def train_bpe(corpus_words, n_merges):
    raise NotImplementedError("TODO 1: BPE o'qitish (juftliklarni sanash va birlashtirish)")

# TODO 2: yangi so'zni belgilarga bo'lib, merges ni O'QITISH TARTIBIDA qo'llang
def encode_word(w, merges):
    raise NotImplementedError("TODO 2: so'zni BPE tokenlariga bo'lish")

# TODO 3: matndagi barcha so'zlar tokenlari yig'indisi
def n_tokens(text, merges):
    raise NotImplementedError("TODO 3: matn tokenlari soni")

uz_lines, en_lines = UZ.split("\n"), EN.split("\n")
tr_uz, te_uz = " ".join(uz_lines[:9]), " ".join(uz_lines[9:])
tr_en, te_en = " ".join(en_lines[:9]), " ".join(en_lines[9:])
corp_en_centric = words(tr_en) * 9 + words(tr_uz) * 1     # 90% inglizcha
corp_balanced = words(tr_en) * 5 + words(tr_uz) * 5        # 50/50

print("=== Tokenlar soni: belgi / so'z / BPE ===")
nw_uz, nw_en = len(words(te_uz)), len(words(te_en))
res = {}
# TODO 4: ikki korpus x merge soni (100, 300, 600) uchun UZ va EN test matnida token soni va fertillikni
#         (token/so'z) hisoblang, res[(nom, m)] = (fert_uz, fert_en) ga yozing va jadval qilib chiqaring
for name, corp in [("BPE inglizcha-markazli", corp_en_centric), ("BPE muvozanatli", corp_balanced)]:
    for m in (100, 300, 600):
        mg = train_bpe(corp, m)
        ...

print("\n=== Kontekst oynasi byudjeti (RAG prompti) ===")
CTX = 8192
SYSTEM, QUESTION, ANSWER_RESERVE = 350, 60, 700
HISTORY_TURNS, TOK_PER_TURN = 6, 120
CHUNK_WORDS = 200
# TODO 5: bo'sh joy = CTX - (tizim + savol + javob zaxirasi + tarix); bo'lak tokeni = CHUNK_WORDS x fertillik;
#         UZ va EN uchun promptga nechta bo'lak sig'ishini chiqaring
PRICE_IN, PRICE_OUT = 0.50, 1.50       # SHARTLI narx, $ / 1M token
req_in, req_out, n_req = 5000, 400, 20000
# TODO 6: kunlik xarajat = so'rovlar x (kirish x narx_kirish + chiqish x narx_chiqish) / 1e6
