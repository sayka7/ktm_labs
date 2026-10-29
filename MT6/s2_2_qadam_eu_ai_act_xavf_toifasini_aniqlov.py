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

# TODO 1: 5-modda taqiqlari (5(1)(f) uchun tibbiy/xavfsizlik istisnosi bilan) -> "TAQIQLANGAN"
# TODO 2: I ilova xavfsizlik komponenti -> "YUQORI XAVF" (6(1))
# TODO 3: III ilova sohasi -> "YUQORI XAVF" (davlat/kredit bo'lsa + 27 FRIA); tor protsedura va profillash yo'q -> 6(3) istisnosi
# TODO 4: 50-modda shaffoflik (muloqot, sintetik kontent, deepfake, hissiyot) -> "CHEKLANGAN (shaffoflik)"; qolgani "MINIMAL"
def classify(s):
    """Ssenariy (lug'at) -> (toifa, asos, majburiyatlar)."""
    raise NotImplementedError("TODO 1-4: qoida dvigateli")

# TODO 5: GPAI model bo'lsa 53-modda; flop > 1e25 bo'lsa tizimli xavf (51/55-moddalar)
def gpai(s):
    raise NotImplementedError("TODO 5: GPAI belgisi")

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
# TODO 6: har ssenariyni tasniflab, kutilgan toifa bilan solishtiring (mos ustuni) va DataFrame ga yig'ing
for name, attrs, exp in SCEN:
    cat, basis, ob = classify(attrs)
    ...
df = pd.DataFrame(rows)
print("=== EU AI Act: xavf toifasi qoida dvigateli ===")
# TODO 7: jadvalni va "Dvigatel testlari: <mos>/<jami>" qatorini chiqaring; kredit skoringi majburiyatlarini sanab bering
