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

# TODO 1: jumla uchun 3 belgi: (a) mazmunli so'zlarning kontekstda bo'lish ulushi, (b) barcha sonlar kontekstda bormi,
#         (c) jumla bigrammalaridan kontekstda YO'Q bo'lganlari ulushi
def features(sent, ctx):
    raise NotImplementedError("TODO 1: tayanish belgilari")

# TODO 2: belgilarni bitta 'tayanish balli' (0..1) ga birlashtiring; son mos kelmasa kuchli jarima
def score(sent, ctx):
    raise NotImplementedError("TODO 2: tayanish balli")

PREFIX = ["Qoidaga ko'ra,", "Hujjatda yozilishicha,", "Ichki tartibga muvofiq,", "Aniqlashtirsak,"]
# TODO 3a: parafraz — kirish so'zi (PREFIX) qo'shib, sonsiz bitta so'zni tushiring (manbaga tayangan, lekin so'zma-so'z emas)
def paraphrase(s):
    raise NotImplementedError("TODO 3a: parafraz")

non_num = [a for d, s, q, a in FACTS if not re.search(r"\d", a)]
samples = []   # (tur, jumla, kontekst, yorliq: 1 = gallyutsinatsiya)
# TODO 3: har fakt uchun 6 tur javob yasang: to'g'ri, parafraz, son almashgan (yoki obyekt almashgan),
#         boshqa hujjatdan, to'qib chiqarilgan (FABRIC)
assert samples, "TODO 3: avval sintetik javoblarni yasang"

print("=== Detektor sifati (ROC-AUC, 1 = gallyutsinatsiya) ===")
# TODO 4: har belgi alohida va birlashgan ball uchun roc_auc_score (yo'nalishga e'tibor: yuqori ball = tayangan)
TH = 0.6
# TODO 5: ball < TH -> 'asossiz'; TP/FP/FN/TN, precision, recall va turlar bo'yicha aniqlash ulushi

print("\n=== Jumla darajasida tekshiruv: aralash javob ===")
# TODO 6: FACTS[9] fakti + to'qilgan jumla + manbada yo'q jumladan iborat javobni jumlalarga bo'lib, har birini baholang
