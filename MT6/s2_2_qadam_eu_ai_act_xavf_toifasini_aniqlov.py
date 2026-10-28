"""MT6 — 2-qadam. EU AI Act xavf toifasini aniqlovchi qoida dvigateli (15 ssenariy, avtomatik testlar bilan)

Regulation (EU) 2024/1689 mantig'i qoidalar ketma-ketligi sifatida yoziladi: 5-modda taqiqlari (a–h bandlar, 5(1)(f) dagi tibbiy/xavfsizlik istisnosi bilan), 6(1)+I ilova, 6(2)+III ilova (8 soha), 6(3) istisnosi (profillash bo'lsa qo'llanmaydi), 50-modda shaffoflik majburiyatlari va minimal xavf; GPAI modellari (53, 51/55-moddalar) alohida belgilanadi. 15 ssenariy tasniflanadi, kutilgan javob bilan solishtiriladi, kredit skoringi uchun majburiyatlar ro'yxati va asosiy muddatlar chiqadi. Bu o'quv soddalashtirishi, yuridik xulosa emas.

Kutilgan natija: 15/15 ssenariy kutilgan toifaga mos; kredit skoringi — YUQORI XAVF, 12 majburiyat (27-modda FRIA bilan); chatbot ortidagi katta model — qo'shimcha GPAI tizimli xavf belgisi.
"""
# 2-qadam. EU AI Act (Regulation (EU) 2024/1689) xavf darajasini aniqlovchi qoida dvigateli
# Tartib muhim: avval taqiqlangan amaliyotlar (5-modda), keyin yuqori xavf (6-modda, I va III ilovalar),
# keyin shaffoflik majburiyatlari (50-modda), qolgani — minimal xavf. GPAI modellari alohida (51-55-moddalar).
# Diqqat: bu o'quv soddalashtirishi, yuridik maslahat emas; moddalar matnini asl manbadan tekshiring.
import pandas as pd
import matplotlib.pyplot as plt

PROHIBITED = [  # (atribut, 5(1) bandi, izoh)
 ("subliminal_manipulyatsiya", "5(1)(a)", "ongsiz/manipulyativ usullar bilan zarar yetkazuvchi ta'sir"),
 ("zaiflikdan_foydalanish", "5(1)(b)", "yosh, nogironlik, ijtimoiy holat zaifligidan foydalanish"),
 ("ijtimoiy_reyting", "5(1)(c)", "ijtimoiy xulq bo'yicha reyting va noqulay munosabat"),
 ("faqat_profil_jinoyat_bashorati", "5(1)(d)", "faqat profillash asosida jinoyat sodir etish xavfini bashorat"),
 ("yuz_rasmlarini_maqsadsiz_yigish", "5(1)(e)", "internet/CCTV dan yuz rasmlarini maqsadsiz yig'ish"),
 ("ish_yoki_talimda_hissiyot", "5(1)(f)", "ish joyi va ta'limda hissiyotni aniqlash (tibbiy/xavfsizlik istisno)"),
 ("biometrik_sezgir_toifa", "5(1)(g)", "biometrik ma'lumotdan irq, siyosiy qarash va h.k. xulosa"),
 ("realvaqt_rbi_huquq", "5(1)(h)", "jamoat joyida real vaqt masofaviy biometrik identifikatsiya (qat'iy istisnolar bilan)"),
]
ANNEX_III = {"biometrika": "III-1", "kritik_infratuzilma": "III-2", "talim": "III-3", "bandlik": "III-4",
             "asosiy_xizmatlar": "III-5", "huquq_tartibot": "III-6", "migratsiya": "III-7", "adliya_demokratiya": "III-8"}
HIGH_OBLIG = ["9 xavflarni boshqarish tizimi", "10 ma'lumotlar va boshqaruv", "11 texnik hujjatlar", "12 jurnal (log) yuritish",
              "13 shaffoflik va foydalanuvchiga ma'lumot", "14 inson nazorati", "15 aniqlik, mustahkamlik, kiberxavfsizlik",
              "17 sifat menejmenti tizimi", "43 muvofiqlikni baholash", "49 EU bazasida ro'yxatdan o'tkazish", "72 bozordan keyingi monitoring"]

def classify(s):
    """Ssenariy (lug'at) -> (toifa, asos, majburiyatlar)."""
    for key, art, note in PROHIBITED:
        if s.get(key):
            if key == "ish_yoki_talimda_hissiyot" and s.get("tibbiy_yoki_xavfsizlik"):
                continue                                           # 5(1)(f) istisnosi
            return "TAQIQLANGAN", f"5-modda {art}: {note}", ["Joriy etish mumkin emas (99-modda: 35 mln EUR yoki 7% gacha jarima)"]
    if s.get("I_ilova_xavfsizlik_komponenti"):
        return "YUQORI XAVF", "6(1)-modda + I ilova: mahsulot xavfsizlik komponenti, uchinchi tomon baholovi", HIGH_OBLIG
    area = s.get("soha")
    if area in ANNEX_III:
        if s.get("tor_protsedura_vazifa") and not s.get("profillash"):
            return "YUQORI XAVF EMAS*", f"6(3)-modda istisnosi ({ANNEX_III[area]}), hujjatlashtirish va ro'yxat shart", ["6(4): baho hujjatlashtiriladi", "49(2): ro'yxatdan o'tkazish"]
        ob = HIGH_OBLIG + (["27 asosiy huquqlarga ta'sirni baholash (FRIA)"] if s.get("davlat_yoki_kredit") else [])
        return "YUQORI XAVF", f"6(2)-modda + {ANNEX_III[area]} ilova", ob
    tr = []
    if s.get("odam_bilan_muloqot"): tr.append("50(1): foydalanuvchiga AI bilan muloqot ekanini bildirish")
    if s.get("sintetik_kontent"): tr.append("50(2): sintetik kontentni mashina o'qiy oladigan belgilash")
    if s.get("deepfake"): tr.append("50(4): deepfake ekanini oshkor qilish")
    if s.get("hissiyot_aniqlash"): tr.append("50(3): hissiyot aniqlash haqida xabardor qilish")
    if tr: return "CHEKLANGAN (shaffoflik)", "50-modda", tr
    return "MINIMAL", "maxsus majburiyat yo'q (ixtiyoriy kodekslar, 95-modda)", ["AI savodxonligi (4-modda) — barcha uchun"]

def gpai(s):
    if not s.get("gpai_model"): return ""
    return "GPAI: 53-modda" + (" + tizimli xavf 51/55-moddalar (>1e25 FLOP)" if s.get("flop", 0) > 1e25 else "")

SCEN = [  # (nom, atributlar, kutilgan toifa — dvigatelni sinash uchun)
 ("Davlat: fuqarolarning ijtimoiy reytingi", {"ijtimoiy_reyting": True}, "TAQIQLANGAN"),
 ("Metro: politsiya uchun real vaqt yuz tanish", {"realvaqt_rbi_huquq": True, "soha": "biometrika"}, "TAQIQLANGAN"),
 ("Call-markaz: operatorlar hissiyotini kuzatish", {"ish_yoki_talimda_hissiyot": True}, "TAQIQLANGAN"),
 ("Zavod: haydovchi charchoqini aniqlash (xavfsizlik)", {"ish_yoki_talimda_hissiyot": True, "tibbiy_yoki_xavfsizlik": True, "hissiyot_aniqlash": True}, "CHEKLANGAN (shaffoflik)"),
 ("HR: rezyumelarni saralash", {"soha": "bandlik", "profillash": True}, "YUQORI XAVF"),
 ("Bank: kredit skoringi", {"soha": "asosiy_xizmatlar", "profillash": True, "davlat_yoki_kredit": True}, "YUQORI XAVF"),
 ("Universitet: imtihon ishlarini baholash", {"soha": "talim"}, "YUQORI XAVF"),
 ("Elektr tarmog'i: yuklamani boshqarish", {"soha": "kritik_infratuzilma"}, "YUQORI XAVF"),
 ("Tibbiy qurilma: rentgen tahlili (MDR)", {"I_ilova_xavfsizlik_komponenti": True}, "YUQORI XAVF"),
 ("HR: ariza fayllarini formatga keltirish", {"soha": "bandlik", "tor_protsedura_vazifa": True}, "YUQORI XAVF EMAS*"),
 ("Bank sayti: mijozlar chatboti", {"odam_bilan_muloqot": True}, "CHEKLANGAN (shaffoflik)"),
 ("Media: deepfake video generator", {"sintetik_kontent": True, "deepfake": True}, "CHEKLANGAN (shaffoflik)"),
 ("Pochta: spam filtri", {}, "MINIMAL"),
 ("O'yin: NPC xatti-harakati", {}, "MINIMAL"),
 ("Chatbot ortidagi katta til modeli (provayder)", {"odam_bilan_muloqot": True, "gpai_model": True, "flop": 3e25}, "CHEKLANGAN (shaffoflik)"),
]
rows = []
for name, attrs, exp in SCEN:
    cat, basis, ob = classify(attrs)
    rows.append(dict(ssenariy=name, toifa=cat, asos=basis, GPAI=gpai(attrs), majburiyat_soni=len(ob), mos=cat == exp))
df = pd.DataFrame(rows)
print("=== EU AI Act: xavf toifasi qoida dvigateli ===")
with pd.option_context("display.width", 220, "display.max_colwidth", 60):
    print(df[["ssenariy", "toifa", "asos", "majburiyat_soni", "mos"]].to_string(index=False))
print(f"\nDvigatel testlari: {df.mos.sum()}/{len(df)} ssenariy kutilgan toifaga mos")
for _, r in df[df.GPAI != ""].iterrows(): print(f"Qo'shimcha: {r.ssenariy} -> {r.GPAI}")

print("\n=== Namuna: 'Bank: kredit skoringi' majburiyatlari (moddalar) ===")
for o in classify(SCEN[5][1])[2]: print("  -", o)
print("\n=== Asosiy muddatlar (2024/1689 matni bo'yicha; keyingi o'zgartirish takliflarini tekshiring) ===")
for d, t in [("2024-08-01", "kuchga kirish"), ("2025-02-02", "taqiqlar (5-modda) va AI savodxonligi"), ("2025-08-02", "GPAI majburiyatlari"),
             ("2026-08-02", "asosiy qism, III ilova yuqori xavf tizimlari"), ("2027-08-02", "I ilova (mahsulot xavfsizligi) yuqori xavf tizimlari")]:
    print(f"  {d}  {t}")
cnt = df.toifa.value_counts()
fig, ax = plt.subplots(figsize=(6.5, 3)); ax.barh(cnt.index, cnt.values, color=["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#7f8c8d"][:len(cnt)])
ax.set_xlabel("ssenariylar soni"); ax.set_title("Ssenariylar xavf toifalari bo'yicha"); plt.tight_layout()
print("\nIzoh: real tasnifda har ssenariy yurist va soha mutaxassisi bilan ko'rib chiqiladi; Komissiya yo'riqnomalari")
print("va milliy organlar talqinlari (6-modda bo'yicha misollar ro'yxati) hisobga olinadi.")
