"""MT6 — 6-qadam. Muvofiqlik tekshiruv ro'yxati va gap-tahlil hisoboti (ISO/IEC 42001, 23894, EU AI Act)

22 talab (ISO/IEC 42001 4–10 bandlari va A.2–A.10 nazorat sohalari, ISO/IEC 23894 xavf jarayoni, EU AI Act 12, 14, 15, 17, 73-moddalari) qisqa parafraz ko'rinishida beriladi. Tashkilotning joriy yetukligi (0–5) va dalili CURRENT lug'atidan olinadi; dalilsiz yuqori baho avtomatik pasaytiriladi. Xavf toifasiga qarab maqsadli yetuklik belgilanadi, gap va ustuvorlik (gap x vazn) hisoblanib, Markdown hisobot (qamrov, top-8 bo'shliq, xulosa) va standartlar orasidagi bog'lanishlar chiqadi.

Kutilgan natija: Umumiy muvofiqlik indeksi ~42/100; eng ustuvor bo'shliqlar — ta'sirni baholash (A.5) va jiddiy hodisalar tartibi (73-modda); 7 ta majburiy talab bajarilmagan -> «joriy etishdan oldin yopish SHART».
"""
# 6-qadam. Muvofiqlik tekshiruv ro'yxati va gap-tahlil hisoboti (Markdown generatsiya)
# Talablar: ISO/IEC 42001 (4-10 bandlar va A ilova nazorat sohalari), ISO/IEC 23894, EU AI Act (yuqori xavf moddalari).
# Talablar matni QISQARTIRILGAN parafraz — standart matnining o'zi emas. Joriy holat (yetuklik 0-5) tashkilot tomonidan to'ldiriladi.
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

REQ = [  # (id, manba, soha, talab (qisqa), majburiylik vazni 1-3, tegishli boshqa manbalar)
 ("Q4", "ISO/IEC 42001 4", "Kontekst", "Tashkilot konteksti, manfaatdor tomonlar va AIMS doirasi aniqlangan", 2, ""),
 ("Q5", "ISO/IEC 42001 5", "Rahbariyat", "Rahbariyat majburiyati va AI siyosati tasdiqlangan", 2, "A.2"),
 ("Q6", "ISO/IEC 42001 6", "Rejalashtirish", "AI xavflarini baholash va ishlov berish jarayoni, maqsadlar", 3, "ISO/IEC 23894; AI Act 9"),
 ("Q7", "ISO/IEC 42001 7", "Qo'llab-quvvatlash", "Resurslar, kompetensiya, xabardorlik, hujjatlashtirilgan ma'lumot", 2, "AI Act 4"),
 ("Q8", "ISO/IEC 42001 8", "Amaliyot", "Operatsion rejalashtirish, AI tizimiga ta'sirni baholash", 3, "AI Act 27"),
 ("Q9", "ISO/IEC 42001 9", "Baholash", "Monitoring, ichki audit, rahbariyat tahlili", 2, "AI Act 72"),
 ("Q10", "ISO/IEC 42001 10", "Takomillashtirish", "Nomuvofiqlik, tuzatuvchi harakatlar, uzluksiz takomillashtirish", 2, ""),
 ("A2", "ISO/IEC 42001 A.2", "Siyosat", "AI bo'yicha siyosatlar", 2, ""),
 ("A3", "ISO/IEC 42001 A.3", "Tashkiliy tuzilma", "Rollar, mas'uliyat, muammolarni xabar qilish", 2, ""),
 ("A4", "ISO/IEC 42001 A.4", "Resurslar", "Ma'lumot, vosita, hisoblash va inson resurslari hujjatlashtirilgan", 1, ""),
 ("A5", "ISO/IEC 42001 A.5", "Ta'sirni baholash", "AI tizimining shaxs va jamiyatga ta'sirini baholash", 3, "AI Act 27"),
 ("A6", "ISO/IEC 42001 A.6", "Hayot sikli", "Talablar, loyihalash, verifikatsiya/validatsiya, joriy etish, monitoring", 3, "AI Act 11, 15"),
 ("A7", "ISO/IEC 42001 A.7", "Ma'lumotlar", "Ma'lumot sifati, kelib chiqishi, tayyorlash", 3, "AI Act 10"),
 ("A8", "ISO/IEC 42001 A.8", "Ma'lumot berish", "Foydalanuvchi va manfaatdor tomonlar uchun ma'lumot", 2, "AI Act 13, 50"),
 ("A9", "ISO/IEC 42001 A.9", "Foydalanish", "AI tizimidan mas'uliyatli foydalanish jarayonlari", 2, "AI Act 14, 26"),
 ("A10", "ISO/IEC 42001 A.10", "Uchinchi tomon", "Yetkazib beruvchi va mijozlar bilan mas'uliyat taqsimoti", 2, ""),
 ("R1", "ISO/IEC 23894", "Xavf jarayoni", "Xavflarni aniqlash, tahlil, baholash, ishlov, monitoring sikli", 3, "AI Act 9"),
 ("E12", "EU AI Act 12", "Jurnal", "Hodisalarni avtomatik qayd qilish (log) va saqlash", 3, ""),
 ("E14", "EU AI Act 14", "Inson nazorati", "Inson nazorati choralari va to'xtatish imkoniyati", 3, ""),
 ("E15", "EU AI Act 15", "Mustahkamlik", "Aniqlik, mustahkamlik va kiberxavfsizlik darajasi e'lon qilingan va sinalgan", 3, ""),
 ("E17", "EU AI Act 17", "Sifat tizimi", "Sifat menejmenti tizimi (QMS)", 3, "ISO/IEC 42001"),
 ("E73", "EU AI Act 73", "Hodisalar", "Jiddiy hodisalar haqida xabar berish tartibi", 3, ""),
]
# Joriy holat: yetuklik 0 (yo'q) .. 5 (optimallashgan) va dalil — talaba o'z tashkiloti uchun to'ldiradi
CURRENT = {"Q4": (3, "doira hujjati v1"), "Q5": (2, "siyosat loyihasi"), "Q6": (1, "xavf ro'yxati Excel'da"), "Q7": (2, "treninglar qisman"),
           "Q8": (1, ""), "Q9": (1, ""), "Q10": (1, ""), "A2": (2, "siyosat loyihasi"), "A3": (3, "RACI jadvali"), "A4": (3, "resurs reyestri"),
           "A5": (0, ""), "A6": (2, "MLOps pipeline"), "A7": (2, "ma'lumot kartasi qisman"), "A8": (1, ""), "A9": (2, "operator yo'riqnomasi"),
           "A10": (1, "shartnoma shablon"), "R1": (1, "xavf ro'yxati Excel'da"), "E12": (3, "ELK loglari 90 kun"), "E14": (1, ""),
           "E15": (2, "faqat aniqlik testi"), "E17": (1, ""), "E73": (0, "")}
RISK_CLASS = "yuqori"                       # EU AI Act toifasi (2-qadamdan): yuqori xavfda maqsadli yetuklik yuqori
TARGET = {1: 3, 2: 3, 3: 4} if RISK_CLASS == "yuqori" else {1: 2, 2: 3, 3: 3}

def md_table(d, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows(): out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)

rows = []
# TODO 1: har talab uchun joriy yetuklik va dalil (CURRENT); dalilsiz >= 2 baho 1 ga tushiriladi
# TODO 2: maqsad = TARGET[vazn], gap = max(0, maqsad - joriy), ustuvorlik = gap x vazn,
#         holat: gap 0 -> bajarilgan, joriy >= 2 -> qisman, aks holda bajarilmagan
for rid, src, area, text, w, rel in REQ:
    raise NotImplementedError("TODO 1-2: talab qatorini baholash")
df = pd.DataFrame(rows)

# TODO 3: manba guruhi (ISO/IEC 42001, ISO/IEC 23894, EU AI Act) bo'yicha qamrov foizi
# TODO 4: umumiy muvofiqlik indeksi = vaznli (min(joriy, maqsad)) / vaznli maqsad x 100
# TODO 5: top-8 bo'shliq va xulosa bilan Markdown hisobot ("# Muvofiqlik gap-tahlili hisoboti") yasang
# TODO 6: crosswalk — boshqa manbalarga bog'langan talablar ro'yxati
