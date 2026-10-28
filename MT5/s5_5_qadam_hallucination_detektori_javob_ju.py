"""MT5 — 5-qadam. Hallucination detektori: javob jumlalarining manbaga tayanishi (leksik tekshiruv)

Har fakt uchun 6 turdagi javob yasaladi: to'g'ri, to'g'ri parafraz, son almashgan, obyekt almashgan, boshqa hujjatdan va to'qib chiqarilgan. Detektor NLI modeli o'rniga uch belgi ishlatadi: mazmunli so'zlar qamrovi, sonlar mosligi va manbada yo'q bigrammalar ulushi; ular «tayanish balli»ga birlashtiriladi. ROC-AUC (har belgi va birlashgan), chegara bo'yicha precision/recall, turlar kesimida aniqlash ulushi va aralash javobni jumla darajasida belgilash ko'rsatiladi.

Kutilgan natija: Birlashgan ball AUC ~0,98; son almashganlarning ~94% aniqlanadi, obyekt almashganlarning faqat ~1/3 qismi (so'zlar boshqa hujjatda emas, shu hujjatda ham uchrashi mumkin); parafrazlarning ~8% noto'g'ri «asossiz» deb belgilanadi.
"""
# 5-qadam. Hallucination (asossiz da'vo) detektori: javob jumlalari manbaga tayanadimi? (leksik tekshiruv)
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
# NLI modeli yo'q — o'rniga: (1) mazmunli so'zlar qamrovi, (2) sonlar mosligi, (3) yangi bigrammalar ulushi.
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

DOCS = build_docs(seed=0); rng = random.Random(1)
STOP = {"va", "uchun", "bilan", "har", "bir", "faqat", "kamida", "holda", "ham", "bo'lishi", "kerak", "etiladi"}
FABRIC = ["Ushbu qoida 2019 yilda Jahon banki tavsiyasi bilan joriy etilgan.", "Qoidani buzgan xodimga 2 million so'm jarima solinadi.",
          "Bu talab faqat Toshkent filialida amal qiladi.", "Tartib xalqaro ISO 9001 sertifikati talabi hisoblanadi.",
          "Istisno holatlarda bosh direktor muddatni 3 baravar uzaytiradi.", "Masofaviy ishlovchilar bu talabdan to'liq ozod qilinadi."]

def content(text):
    return [w for w in tokens(text) if len(w) >= 3 and w not in STOP and not w.isdigit()]
def numbers(text):
    return re.findall(r"\d+(?::\d+)?", text)

def features(sent, ctx):
    """Bitta javob jumlasi uchun 'tayanish' belgilari."""
    cw, ctx_set = content(sent), set(tokens(ctx))
    support = np.mean([w in ctx_set for w in cw]) if cw else 1.0
    nums = numbers(sent); num_ok = all(n in numbers(ctx) for n in nums)
    ct = tokens(ctx); big = set(zip(ct, ct[1:])); st = tokens(sent)
    sb = list(zip(st, st[1:])); novel = np.mean([b not in big for b in sb]) if sb else 0.0
    return support, num_ok, novel

def score(sent, ctx):
    """Yakuniy 'tayanish balli' (1 = to'liq tayangan). Son mos kelmasa — kuchli jarima."""
    s, n, nov = features(sent, ctx)
    return s * (1.0 if n else 0.3) * (1 - 0.5 * nov)

PREFIX = ["Qoidaga ko'ra,", "Hujjatda yozilishicha,", "Ichki tartibga muvofiq,", "Aniqlashtirsak,"]
def paraphrase(s):
    """Manbaga tayangan, lekin so'zma-so'z bo'lmagan javob: kirish so'zi qo'shiladi, bitta so'z tushiriladi."""
    w = s.rstrip(".").split(); i = rng.randrange(1, len(w) - 1)
    if not re.search(r"\d", w[i]): w.pop(i)
    return f"{rng.choice(PREFIX)} {w[0].lower()} {' '.join(w[1:])}."

# ---- Sintetik javoblar: to'g'ri va 4 xil gallyutsinatsiya ----
non_num = [a for d, s, q, a in FACTS if not re.search(r"\d", a)]
samples = []   # (tur, jumla, kontekst, yorliq: 1 = gallyutsinatsiya)
for d, s, q, a in FACTS:
    ctx = DOCS[d]
    samples.append(("to'g'ri", s, ctx, 0))
    samples.append(("to'g'ri (parafraz)", paraphrase(s), ctx, 0))
    if numbers(s):
        alt = re.sub(r"\d+", lambda m: str(int(m.group()) + rng.choice([1, 2, 5, 10])), s, count=1)
        samples.append(("son almashgan", alt, ctx, 1))
    else:
        other = rng.choice([x for x in non_num if x != a])
        samples.append(("obyekt almashgan", re.sub(re.escape(a), other, s, flags=re.I), ctx, 1))
    other_fact = rng.choice([f for f in FACTS if f[0] != d])[1]
    samples.append(("boshqa hujjatdan", other_fact, ctx, 1))
    samples.append(("to'qib chiqarilgan", rng.choice(FABRIC), ctx, 1))

y = np.array([l for *_, l in samples])
S = np.array([score(snt, c) for _, snt, c, _ in samples])
F_ = np.array([features(snt, c) for _, snt, c, _ in samples], dtype=float)
print(f"Namunalar: {len(samples)} (to'g'ri={sum(y==0)}, gallyutsinatsiya={sum(y==1)})")
print("=== Detektor sifati (ROC-AUC, 1 = gallyutsinatsiya) ===")
print(f"faqat so'z qamrovi           AUC = {roc_auc_score(y, 1 - F_[:, 0]):.3f}")
print(f"faqat son tekshiruvi         AUC = {roc_auc_score(y, 1 - F_[:, 1]):.3f}")
print(f"faqat yangi bigrammalar      AUC = {roc_auc_score(y, F_[:, 2]):.3f}")
print(f"birlashgan tayanish balli    AUC = {roc_auc_score(y, 1 - S):.3f}")
TH = 0.6
pred = (S < TH).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f"\nChegara: ball < {TH} -> 'asossiz'.  TP={tp} FP={fp} FN={fn} TN={tn}  precision={tp/(tp+fp):.2f} recall={tp/(tp+fn):.2f}")
print("=== Turlar bo'yicha aniqlash ulushi ===")
for t in ["to'g'ri", "to'g'ri (parafraz)", "son almashgan", "obyekt almashgan", "boshqa hujjatdan", "to'qib chiqarilgan"]:
    idx = [i for i, s in enumerate(samples) if s[0] == t]
    print(f"{t:<20} n={len(idx):3d}  'asossiz' deb belgilangan: {pred[idx].mean():.2f}   o'rtacha ball={S[idx].mean():.2f}")

print("\n=== Jumla darajasida tekshiruv: aralash javob ===")
d, s, q, a = FACTS[9]
ans = s + " " + FABRIC[1] + " " + "Hisobotga chiptalar nusxasi ilova qilinadi."
for snt in sentences(ans):
    sc = score(snt, DOCS[d]); print(f"[{'OK ' if sc >= TH else 'ASOSSIZ'}] {sc:.2f}  {snt}")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(S[y == 0], bins=15, alpha=.6, label="to'g'ri"); ax.hist(S[y == 1], bins=15, alpha=.6, label="gallyutsinatsiya")
ax.axvline(TH, color="k", ls="--"); ax.set_xlabel("tayanish balli"); ax.legend(); ax.set_title("Tayanish balli taqsimoti"); plt.tight_layout()
print("\nIzoh: real tizimda leksik qamrov o'rniga NLI (entailment) modeli yoki 'LLM-hakam' ishlatiladi; ular sinonim va")
print("parafrazni tushunadi, lekin o'zi ham xato qiladi. Sonlarni alohida tekshirish oddiy, ammo juda samarali qoida.")
