"""MT6 — 1-qadam. AI tizimi xavf reyestri generatori (ISO/IEC 23894 uslubi): ehtimollik x ta'sir, qoldiq xavf, Monte Karlo

Kredit skoring tizimi profili (shaxsiy ma'lumot, avtomatik qaror, inson nazorati yo'q, tashqi API, drift) asosida 12 ta xavfdan iborat katalog baholanadi: profil modifikatorlari ehtimollik (L) va ta'sirni (I) o'zgartiradi, xavf = L x I, daraja (past/o'rta/yuqori/kritik). Nazorat choralari qoldiq xavfni hisoblaydi; xavf ishtahasi (8) bo'yicha ishlov qarori chiqadi (qabul, kamaytirish, eskalatsiya). Ekspert bahosidagi +-1 noaniqlik Monte Karlo bilan baholanadi, ikki issiqlik xaritasi (oldin/keyin) chiziladi.

Kutilgan natija: Jami xavf ~126 dan ~75 ga tushadi (~40%); choralari yo'q R08, R02, R10 eskalatsiyaga chiqadi; R08 uchun P(qoldiq > 8) ~0,8. Ikki 5x5 issiqlik xaritasi.
"""
# 1-qadam. AI tizimi xavf reyestri generatori (ISO/IEC 23894 uslubi: aniqlash -> tahlil -> baholash -> ishlov)
# Ehtimollik (L) va ta'sir (I) 1..5 shkalada; xavf = L x I. Tizim profili xavf ballarini o'zgartiradi.
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# ---- Tizim profili (talaba o'z tizimiga moslaydi) ----
PROFILE = {"nomi": "Bank: kredit arizalarini avtomatik skoring qilish", "shaxsiy_malumot": True, "avtomatik_qaror": True,
           "inson_nazorati": False, "tashqi_model_api": True, "foydalanuvchilar": 50000, "malumot_drift": True}

# ---- Xavf katalogi: (id, manba, tavsif, hayot sikli bosqichi, bazaviy L, bazaviy I, kim zarar ko'radi) ----
CATALOG = [
 ("R01", "Ma'lumot sifati", "O'quv ma'lumotida tarixiy tarafkashlik — guruhlarga nisbatan adolatsiz qaror", "ma'lumot", 3, 4, "shaxs"),
 ("R02", "Ma'lumot sifati", "Yorliq xatolari va to'liq bo'lmagan yozuvlar", "ma'lumot", 3, 3, "tashkilot"),
 ("R03", "ML ga xos", "Ma'lumot/kontsept drifti — aniqlikning vaqt o'tib pasayishi", "ishlatish", 3, 3, "tashkilot"),
 ("R04", "Shaffoflik", "Qaror sababini tushuntirib bo'lmasligi (qora quti)", "loyihalash", 3, 3, "shaxs"),
 ("R05", "Avtomatlashtirish", "Inson nazoratisiz noto'g'ri rad etish", "ishlatish", 2, 4, "shaxs"),
 ("R06", "Xavfsizlik", "Adversarial manipulyatsiya (ariza belgilarini sun'iy o'zgartirish)", "ishlatish", 2, 4, "tashkilot"),
 ("R07", "Maxfiylik", "Shaxsiy ma'lumotlarning sizib chiqishi / a'zolik hujumi", "ishlatish", 2, 5, "shaxs"),
 ("R08", "Uchinchi tomon", "Tashqi model API ishlamay qolishi yoki o'zgarishi", "ishlatish", 3, 3, "tashkilot"),
 ("R09", "Muvofiqlik", "Normativ talablarga (masalan, EU AI Act yuqori xavf) mos kelmaslik", "boshqaruv", 2, 5, "tashkilot"),
 ("R10", "Mustahkamlik", "Ko'rilmagan kirish ma'lumotlarida ishonchsiz bashorat", "sinov", 3, 3, "shaxs"),
 ("R11", "Inson omili", "Operatorlarning modelga ortiqcha ishonishi (automation bias)", "ishlatish", 3, 3, "shaxs"),
 ("R12", "Hayot sikli", "Model versiyalari va ma'lumot kelib chiqishi hujjatlashtirilmagan", "boshqaruv", 3, 2, "tashkilot"),
]
# profil bo'yicha modifikatorlar (qoida: shart -> (xavf id, dL, dI))
MODS = [("shaxsiy_malumot", "R07", 1, 0), ("avtomatik_qaror", "R05", 1, 0), ("avtomatik_qaror", "R11", 0, 1),
        ("inson_nazorati", "R05", -2, 0), ("tashqi_model_api", "R08", 1, 0), ("malumot_drift", "R03", 1, 0)]
# ---- Nazorat choralari: (xavf id, chora, L kamayishi, I kamayishi, narx shartli birlik) ----
CONTROLS = [("R01", "Fairness audit + qayta tortish (reweighing)", 1, 1, 3), ("R03", "Drift monitoringi va qayta o'qitish", 2, 0, 2),
            ("R04", "SHAP tushuntirish + qaror sabablari xati", 1, 1, 2), ("R05", "Rad etishda majburiy inson tekshiruvi", 2, 1, 4),
            ("R06", "Kirish validatsiyasi + adversarial test", 1, 0, 2), ("R07", "Differensial maxfiylik / kirishni cheklash", 1, 1, 4),
            ("R09", "Muvofiqlik gap-tahlili va texnik hujjat", 1, 1, 3), ("R11", "Operator treningi va ishonch ko'rsatkichi", 1, 0, 1),
            ("R12", "Model registri va ma'lumot kartasi", 2, 0, 1)]
APPETITE = 8          # xavf ishtahasi: qoldiq xavf <= 8 bo'lsa qabul qilinadi

# TODO 1: xavf darajasi: >=20 kritik, >=12 yuqori, >=6 o'rta, aks holda past
def level(s):
    raise NotImplementedError("TODO 1: xavf darajasi")

rows = []
for rid, src, desc, stage, L, I, who in CATALOG:
    # TODO 2: profil shartlari bajarilgan MODS bo'yicha L va I ni o'zgartiring, 1..5 oralig'ida cheklang
    # TODO 3: shu xavfga tegishli CONTROLS bo'yicha qoldiq rL, rI va qoldiq xavf = rL x rI
    # TODO 4: ishlov qarori: xavf <= APPETITE -> qabul; chora yo'q -> eskalatsiya; qoldiq <= APPETITE -> kamaytirish;
    #         aks holda kamaytirish + qo'shimcha chora
    raise NotImplementedError("TODO 2-4: reyestr qatorini hisoblash")
reg = pd.DataFrame(rows)
print(f"Tizim: {PROFILE['nomi']}")
print("=== Xavf reyestri (ISO/IEC 23894 uslubi) ===")
# TODO 5: reyestrni xavf bo'yicha saralab chiqaring; jami xavf -> qoldiq, kamayish foizi, choralar narxi

print("\n=== Monte Karlo: qoldiq xavf ishtahadan oshish ehtimoli (L va I +-1, 5000 simulyatsiya) ===")
# TODO 6: eng katta 6 xavf uchun rL, rI ga -1/0/+1 tasodifiy xato qo'shib (1..5 cheklab), P(qoldiq > APPETITE) ni baholang
# TODO 7: 5x5 issiqlik xaritasi (oldin: L-I, keyin: rL-rI), katakda xavf id lari
