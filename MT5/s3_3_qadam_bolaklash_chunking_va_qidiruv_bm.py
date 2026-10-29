"""MT5 — 3-qadam. Bo'laklash (chunking) va qidiruv: BM25 (qo'lda), TF-IDF, gibrid RRF; recall@k va MRR

6 hujjatli sintetik korporativ bilim bazasi (24 fakt + 24 chalg'ituvchi jumla + umumiy shovqin) yasaladi. Olti bo'laklash strategiyasi (so'z oynasi 40/80/160, overlap bilan va siz; 2 va 4 jumlali) va uch qidiruv usuli — qo'lda yozilgan Okapi BM25 (sodda o'zbekcha qo'shimcha kesish bilan), char n-gram TF-IDF va Reciprocal Rank Fusion — taqqoslanadi. Har savol uchun oltin jumlani to'liq saqlagan bo'lak o'rni bo'yicha recall@1/3/5, MRR va top-3 kontekst hajmi (so'z) hisoblanadi.

Kutilgan natija: Overlapsiz 40 so'zli bo'laklarda recall@5 ~0,83 (jumla kesiladi); overlap qo'shilsa ~1,0. Katta bo'laklar MRR=1 beradi, lekin top-3 kontekst ~390 so'z; 2 jumlali bo'laklar ~40 so'zda MRR ~0,9 — eng tejamkor.
"""
# 3-qadam. Bo'laklash (chunking) + qidiruv (BM25 qo'lda, TF-IDF, gibrid RRF) va recall@k / MRR baholash
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

DOCS = build_docs(seed=0)

# TODO 1: so'z oynasi bilan bo'laklash: qadam = size - overlap; har bo'lak (doc_id, matn)
def chunk_words(doc_id, text, size, overlap):
    raise NotImplementedError("TODO 1: so'z oynasi bo'laklash")

# TODO 2: jumlalar bo'yicha bo'laklash (har n_sent jumla bitta bo'lak)
def chunk_sentences(doc_id, text, n_sent):
    raise NotImplementedError("TODO 2: jumla bo'laklash")

class BM25:
    """Okapi BM25 (k1, b) — qo'lda yozing."""
    def __init__(self, texts, k1=1.5, b=0.75):
        # TODO 3: har bo'lak uchun tokens() chastotasi, uzunligi, o'rtacha uzunlik va idf
        raise NotImplementedError("TODO 3: BM25 indeksi")
    def scores(self, q):
        # TODO 4: so'rov so'zlari bo'yicha idf * f*(k1+1) / (f + k1*(1-b+b*len/avg)) yig'indisi
        raise NotImplementedError("TODO 4: BM25 ballari")

# TODO 5: Reciprocal Rank Fusion — har ro'yxatdagi o'rin (rank) bo'yicha 1/(k+rank) yig'indisi
def rrf(*score_lists, k=60):
    raise NotImplementedError("TODO 5: RRF")

# TODO 6: har savol uchun ballar -> tartib; oltin jumlani TO'LIQ saqlagan bo'lakning o'rni; recall@1/3/5, MRR, top-3 so'zlar
def evaluate(chunks, method):
    raise NotImplementedError("TODO 6: recall@k va MRR")

STRATS = {"so'z-40/0": lambda d, t: chunk_words(d, t, 40, 0), "so'z-40/10": lambda d, t: chunk_words(d, t, 40, 10),
          "so'z-80/20": lambda d, t: chunk_words(d, t, 80, 20), "so'z-160/40": lambda d, t: chunk_words(d, t, 160, 40),
          "jumla-2": lambda d, t: chunk_sentences(d, t, 2), "jumla-4": lambda d, t: chunk_sentences(d, t, 4)}
print("=== Qidiruv sifati: bo'laklash x usul ===")
table, cost = {}, {}
for sname, fn in STRATS.items():
    chunks = [c for d, t in DOCS.items() for c in fn(d, t)]
    for method in ("BM25", "TF-IDF(char)", "Gibrid RRF"):
        rec, mrr, cw = evaluate(chunks, method); table[(sname, method)] = mrr; cost[(sname, method)] = cw
        ...
# TODO 7: eng yuqori MRR va eng tejamkor (MRR / kontekst so'zlari) kombinatsiyani "Eng yaxshi MRR:" bilan chiqaring
