"""MT5 — 4-qadam. Mini-RAG: qidiruv, prompt yig'ish, ekstraktiv generator va rad etish chegarasi

To'liq RAG zanjiri yig'iladi: 3 jumlali bo'laklar, TF-IDF qidiruv, top-k bo'lak jumlalaridan savolga eng o'xshashini tanlovchi **ekstraktiv generator** (son so'ralsa sonli jumlaga bonus). Javobsiz 6 ta savol qo'shilib, rad etish chegarasi `tau` to'ri bo'yicha to'g'ri javob, noto'g'ri javob, asossiz rad va to'g'ri rad soni hisoblanadi. Oxirida haqiqiy LLM'ga yuboriladigan prompt (rol, tayanish ko'rsatmasi, manba identifikatorlari) chop etiladi.

Kutilgan natija: Javobli savollarda aniqlik ~0,88; xatolar chalg'ituvchi jumlalardan («Kichik summali xaridlar…»). tau ~0,3 da umumiy aniqlik ~0,8; ba'zi javobsiz savollarga baribir noto'g'ri jumla qaytadi — leksik o'xshashlik chegarasining zaifligi.
"""
# 4-qadam. Mini-RAG: qidiruv -> prompt yig'ish -> ekstraktiv "generator" (jumla tanlash) -> rad etish chegarasi
# ---- Sintetik korporativ bilim bazasi (6 hujjat, 24 fakt) — bu blok har qadamda bir xil ----
import re, math, random, collections
FACTS = [  # (hujjat, fakt jumlasi, savol (boshqacha so'zlar bilan), kutilgan javob bo'lagi)
 ("Kadrlar", "Yillik mehnat ta'tili 24 kalendar kunni tashkil etadi.", "Yillik ta'til necha kun beriladi?", "24"),
 ("Kadrlar", "Ta'til uchun ariza kamida 10 kun oldin bo'lim boshlig'iga topshiriladi.", "Ta'tilga arizani qancha oldin berish kerak?", "10 kun"),
 ("Kadrlar", "Sinov muddati yangi xodimlar uchun 3 oy davom etadi.", "Yangi xodimning sinov muddati qancha?", "3 oy"),
 ("Kadrlar", "Ish kuni soat 9:00 da boshlanib 18:00 da tugaydi.", "Ish kuni soat nechada boshlanadi?", "9:00"),
 ("Xavfsizlik", "Parollar har 90 kunda almashtirilishi shart.", "Parolni qanchalik tez-tez almashtirish kerak?", "90"),
 ("Xavfsizlik", "Parol kamida 12 belgidan iborat bo'lishi kerak.", "Parolning eng kam uzunligi nechta belgi?", "12"),
 ("Xavfsizlik", "Server xonasiga kirish faqat IT bo'limi ruxsatnomasi bilan amalga oshiriladi.", "Server xonasiga kirish uchun kimning ruxsati kerak?", "IT bo'limi"),
 ("Xavfsizlik", "Shubhali xat haqida 15 daqiqa ichida xavfsizlik markaziga xabar beriladi.", "Shubhali elektron xat kelsa qancha vaqtda xabar qilinadi?", "15 daqiqa"),
 ("Moliya", "Xizmat safari kunlik xarajati 350 ming so'm miqdorida qoplanadi.", "Xizmat safarida bir kunlik xarajat qancha qoplanadi?", "350 ming"),
 ("Moliya", "Avans hisoboti safardan qaytgach 3 ish kuni ichida topshiriladi.", "Safardan keyin avans hisobotini qachongacha topshirish kerak?", "3 ish kuni"),
 ("Moliya", "Moliyaviy hisobotlar har chorak oxirida buxgalteriyaga taqdim etiladi.", "Moliyaviy hisobotlar qachon buxgalteriyaga beriladi?", "chorak"),
 ("Moliya", "5 million so'mdan yuqori xaridlar moliya direktori tomonidan tasdiqlanadi.", "Katta summali xaridni kim tasdiqlaydi?", "moliya direktori"),
 ("IT", "Noutbuk ta'miri uchun so'rov Helpdesk tizimi orqali yuboriladi.", "Noutbuk buzilsa ta'mir so'rovi qayerga yuboriladi?", "Helpdesk"),
 ("IT", "Zaxira nusxalar har kecha soat 02:00 da avtomatik olinadi.", "Zaxira nusxa soat nechada olinadi?", "02:00"),
 ("IT", "VPN ulanishi faqat korporativ qurilmalardan ruxsat etiladi.", "VPN ga qaysi qurilmalardan ulanish mumkin?", "korporativ qurilma"),
 ("IT", "Elektron pochta qutisi hajmi 50 GB bilan cheklangan.", "Pochta qutisining hajmi qancha?", "50 GB"),
 ("Mehnat", "Ishlab chiqarish sexida kaska va himoya ko'zoynagi taqish majburiy.", "Sexda qaysi himoya vositalari majburiy?", "kaska"),
 ("Mehnat", "Yong'in mashqlari yiliga 2 marta o'tkaziladi.", "Yong'in mashqi yilda necha marta bo'ladi?", "2 marta"),
 ("Mehnat", "Baxtsiz hodisa haqida 1 soat ichida mehnat muhofazasi muhandisiga xabar qilinadi.", "Baxtsiz hodisa bo'lsa kimga xabar qilinadi?", "mehnat muhofazasi muhandisi"),
 ("Mehnat", "Tibbiy ko'rik har 12 oyda bir marta o'tkaziladi.", "Tibbiy ko'rikdan qanchalik tez-tez o'tiladi?", "12 oy"),
 ("Mijoz", "Mijoz shikoyatiga 48 soat ichida javob beriladi.", "Mijoz shikoyatiga javob berish muddati qancha?", "48 soat"),
 ("Mijoz", "Mijozlarning shaxsiy ma'lumotlari shifrlangan holda 5 yil saqlanadi.", "Mijozlarning shaxsiy ma'lumotlari necha yil saqlanadi?", "5 yil"),
 ("Mijoz", "Qo'ng'iroq markazi dushanbadan shanbagacha ishlaydi.", "Qo'ng'iroq markazi qaysi kunlari ishlaydi?", "dushanbadan shanbagacha"),
 ("Mijoz", "Qaytarilgan mahsulot uchun pul 7 ish kunida qaytariladi.", "Mahsulot qaytarilganda pul necha kunda qaytariladi?", "7 ish kuni"),
]
FILLER = ["Ushbu bo'lim kompaniyaning ichki tartibini belgilaydi.", "Qoidalar barcha xodimlar uchun majburiy hisoblanadi.",
 "O'zgarishlar direktor buyrug'i bilan tasdiqlanadi.", "Savollar bo'yicha tegishli bo'limga murojaat qilish mumkin.",
 "Hujjat har yili qayta ko'rib chiqiladi.", "Qoidabuzarlik intizomiy javobgarlikka sabab bo'ladi.",
 "Xodimlar ushbu tartib bilan imzo qo'yib tanishadi.", "Muddatlar ish kunlarida hisoblanadi, agar boshqacha ko'rsatilmagan bo'lsa.",
 "Bo'lim boshliqlari nazoratni ta'minlaydi.", "Elektron shakllar ichki portalda joylashgan.",
 "Kompaniya xalqaro standartlarga muvofiq ishlaydi.", "Har bir xodim o'z ma'lumotlarini yangilab turishi kerak."]

DISTRACT = [  # o'xshash so'zli, lekin boshqa savolga tegishli "chalg'ituvchi" jumlalar
 ("Kadrlar", "Homiladorlik va tug'ish ta'tili alohida tartibda 126 kun beriladi."), ("Kadrlar", "Ariza shakli kadrlar bo'limi portalida joylashgan."),
 ("Kadrlar", "Sinov muddati davomida xodim yillik ta'tilga chiqa olmaydi."), ("Kadrlar", "Ish kuni davomida tushlik tanaffusi 1 soat."),
 ("Xavfsizlik", "Bank kartasi PIN-kodi 6 raqamdan iborat bo'ladi."), ("Xavfsizlik", "Mehmon Wi-Fi paroli har oy yangilanadi."),
 ("Xavfsizlik", "Server xonasida harorat 22 darajadan oshmasligi kerak."), ("Xavfsizlik", "Shubhali qurilma topilsa qo'riqlash xizmatiga xabar beriladi."),
 ("Moliya", "Xizmat safarida mehmonxona xarajati alohida qoplanadi."), ("Moliya", "Yillik moliyaviy hisobot auditorlar tomonidan tekshiriladi."),
 ("Moliya", "Kichik summali xaridlar bo'lim boshlig'i tomonidan tasdiqlanadi."), ("Moliya", "Avans miqdori safar xarajatlari smetasiga qarab belgilanadi."),
 ("IT", "Zaxira nusxalar 30 kun davomida saqlanadi."), ("IT", "Mehmon tarmog'iga shaxsiy qurilmalardan ulanish mumkin."),
 ("IT", "Pochta xatiga ilova hajmi 25 MB bilan cheklangan."), ("IT", "Printer kartriji omborxonadan olinadi."),
 ("Mehnat", "Omborda xavfsizlik poyabzali taqish majburiy."), ("Mehnat", "Evakuatsiya rejasi har qavatda osib qo'yiladi."),
 ("Mehnat", "Kichik jarohatlar tibbiy jurnalda qayd etiladi."), ("Mehnat", "Ko'z ko'rigi har 24 oyda o'tkaziladi."),
 ("Mijoz", "Mijoz so'roviga chat orqali 10 daqiqada javob beriladi."), ("Mijoz", "Xodimlarning shaxsiy ma'lumotlari arxivda 75 yil saqlanadi."),
 ("Mijoz", "Qo'ng'iroq markazi bayram kunlari dam oladi."), ("Mijoz", "Mahsulotni qaytarish uchun 14 kun muddat beriladi."),
]

def build_docs(seed=0, filler_per_fact=3):
    """Har hujjat: faktlar + chalg'ituvchi jumlalar + umumiy (shovqin) jumlalar aralashtiriladi."""
    rng = random.Random(seed); docs = collections.OrderedDict()
    for d, s, q, a in FACTS:
        docs.setdefault(d, []).extend([s] + rng.sample(FILLER, filler_per_fact))
    for d, s in DISTRACT:
        docs[d].append(s)
    for d in docs:
        rng.shuffle(docs[d])
    return {d: " ".join(v) for d, v in docs.items()}

def sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]

SUFFIX = sorted({"lari", "larni", "larga", "lar", "ning", "ni", "ga", "da", "dan", "dagi", "gacha", "si", "i", "ini"}, key=lambda s: (-len(s), s))
def stem(w):
    """Juda sodda o'zbekcha qo'shimcha kesish (o'quv maqsadida; haqiqiy morfologik tahlilchi emas)."""
    for s in SUFFIX:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[: -len(s)]
    return w

def tokens(text, use_stem=True):
    ws = re.findall(r"[a-zA-Z0-9'ʻʼ:]+", text.lower())
    return [stem(w) if use_stem else w for w in ws]
# ---- bilim bazasi tugadi ----
# Katta LLM yo'q: javob sifatida qidirilgan bo'laklardan eng mos JUMLA tanlanadi (ekstraktiv QA).
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS = build_docs(seed=0)
CHUNKS = []
for d, t in DOCS.items():
    s = sentences(t)
    for i in range(0, len(s), 3):                      # 3 jumlali bo'laklar
        CHUNKS.append({"id": f"{d}#{i // 3 + 1}", "text": " ".join(s[i:i + 3])})
texts = [c["text"] for c in CHUNKS]
VEC = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts + [s for t in texts for s in sentences(t)])
M = VEC.transform(texts)

def retrieve(q, k=3):
    sc = (M @ VEC.transform([q]).T).toarray().ravel()
    idx = np.argsort(-sc)[:k]
    return [(CHUNKS[i], float(sc[i])) for i in idx]

QWORDS = {"necha", "qancha", "nechta", "nechada", "qanchalik"}
def answer(q, k=3, tau=0.0):
    """Ekstraktiv generator: top-k bo'lak jumlalari ichidan savolga eng o'xshashini tanlaydi."""
    hits = retrieve(q, k); best = (-1, None, None)
    qv = VEC.transform([q])
    for ch, _ in hits:
        for s in sentences(ch["text"]):
            sc = float((VEC.transform([s]) @ qv.T).toarray()[0, 0])
            if set(tokens(q, False)) & QWORDS and re.search(r"\d", s): sc += 0.05   # son so'ralsa — sonli jumla
            if sc > best[0]: best = (sc, s, ch["id"])
    if best[0] < tau:
        return "Kechirasiz, bilim bazasida bu savolga javob topilmadi.", None, best[0]
    return best[1], best[2], best[0]

UNANSWERABLE = ["Korporativ avtomobildan foydalanish tartibi qanday?", "Kompaniyaning bosh direktori kim?",
                "Ofisga uy hayvonini olib kelish mumkinmi?", "Yangi yil mukofoti qancha beriladi?",
                "Avtoturargoh uchun oylik to'lov qancha?", "Sport zali qaysi binoda joylashgan?"]

print("=== Mini-RAG: top-k ta'siri (javobsiz savollarsiz) ===")
for k in (1, 3, 5):
    ok = sum(a.lower() in answer(q, k)[0].lower() for d, s, q, a in FACTS)
    print(f"k={k}: to'g'ri javob {ok}/{len(FACTS)} = {ok/len(FACTS):.2f}")
print("Xato tanlangan jumlalar (k=3):")
for d, s, q, a in FACTS:
    txt = answer(q, 3)[0]
    if a.lower() not in txt.lower(): print(f"  S: {q}\n     -> {txt}")

print("\n=== Rad etish chegarasi (tau) tanlash: javobli + javobsiz savollar ===")
ans_sc = [answer(q)[2] for d, s, q, a in FACTS]; una_sc = [answer(q)[2] for q in UNANSWERABLE]
print(f"javobli savollar balli: min={min(ans_sc):.3f} o'rtacha={np.mean(ans_sc):.3f};  javobsiz: max={max(una_sc):.3f} o'rtacha={np.mean(una_sc):.3f}")
print(f"{'tau':>6}{'to`g`ri':>9}{'noto`g`ri':>11}{'asossiz rad':>13}{'to`g`ri rad':>13}{'umumiy':>8}")
best_tau, best_acc = 0, -1
for tau in np.round(np.arange(0.10, 0.46, 0.05), 2):
    c = w = fr = tr = 0
    for d, s, q, a in FACTS:
        txt, src, _ = answer(q, 3, tau)
        if src is None: fr += 1
        elif a.lower() in txt.lower(): c += 1
        else: w += 1
    tr = sum(answer(q, 3, tau)[1] is None for q in UNANSWERABLE)
    acc = (c + tr) / (len(FACTS) + len(UNANSWERABLE))
    if acc > best_acc: best_tau, best_acc = tau, acc
    print(f"{tau:>6.2f}{c:>9}{w:>11}{fr:>13}{tr:>13}{acc:>8.2f}")
print(f"Tanlangan tau = {best_tau} (umumiy aniqlik {best_acc:.2f})")

print("\n=== Namuna javoblar (manba havolasi bilan) ===")
for q in [FACTS[4][2], FACTS[13][2], UNANSWERABLE[1]]:
    txt, src, sc = answer(q, 3, best_tau)
    print(f"S: {q}\nJ: {txt}" + (f"  [manba: {src}, ball={sc:.2f}]" if src else f"  [ball={sc:.2f} < tau]"))

q = FACTS[8][2]; hits = retrieve(q, 3)
prompt = ("Sen kompaniya ichki yordamchisisan. Faqat KONTEKST bo'limidagi ma'lumotga tayan.\n"
          "Javob topilmasa: 'Ma'lumot topilmadi' de. Har da'vodan keyin [manba] ko'rsat.\n=== KONTEKST BOSHI ===\n" +
          "\n".join(f"[{c['id']}] {c['text']}" for c, _ in hits) + f"\n=== KONTEKST OXIRI ===\nSavol: {q}\nJavob:")
print("\n=== Haqiqiy LLM uchun yig'ilgan prompt ===")
print(prompt)
print(f"(prompt uzunligi ~{len(prompt.split())} so'z)")
print("\nIzoh: haqiqiy LLM ushbu promptdan erkin matnli (abstraktiv) javob yozadi; bizning ekstraktiv generator esa")
print("faqat mavjud jumlani qaytaradi — shuning uchun u 'o'ylab topa olmaydi', lekin jumlalarni birlashtira ham olmaydi.")
