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

def level(s):
    return "kritik" if s >= 20 else "yuqori" if s >= 12 else "o'rta" if s >= 6 else "past"

rows = []
for rid, src, desc, stage, L, I, who in CATALOG:
    for key, mid, dL, dI in MODS:
        if mid == rid and PROFILE.get(key):
            L, I = L + dL, I + dI
    L, I = int(np.clip(L, 1, 5)), int(np.clip(I, 1, 5))
    ctl = [c for c in CONTROLS if c[0] == rid]
    rL = int(np.clip(L - sum(c[2] for c in ctl), 1, 5)); rI = int(np.clip(I - sum(c[3] for c in ctl), 1, 5))
    res = rL * rI
    if L * I <= APPETITE: treat = "qabul (monitoring)"
    elif not ctl: treat = "chora yo'q -> eskalatsiya"
    else: treat = "kamaytirish" if res <= APPETITE else "kamaytirish + qo'shimcha chora"
    rows.append(dict(id=rid, manba=src, bosqich=stage, L=L, I=I, xavf=L * I, daraja=level(L * I), chora="; ".join(c[1] for c in ctl) or "-",
                     rL=rL, rI=rI, qoldiq=res, q_daraja=level(res), ishlov=treat, narx=sum(c[4] for c in ctl), kim=who, tavsif=desc))
reg = pd.DataFrame(rows).sort_values(["xavf", "id"], ascending=[False, True])
print(f"Tizim: {PROFILE['nomi']}")
print("=== Xavf reyestri (ISO/IEC 23894 uslubi) ===")
with pd.option_context("display.width", 200, "display.max_colwidth", 38):
    print(reg[["id", "manba", "bosqich", "L", "I", "xavf", "daraja", "rL", "rI", "qoldiq", "q_daraja", "ishlov"]].to_string(index=False))
print(f"\nJami xavf (yig'indi): {reg.xavf.sum()} -> qoldiq: {reg.qoldiq.sum()}  (kamayish {1 - reg.qoldiq.sum() / reg.xavf.sum():.0%}), choralar narxi = {reg.narx.sum()} birlik")
print("Ishtahadan yuqori qoldiq xavflar:", ", ".join(reg[reg.qoldiq > APPETITE].id) or "yo'q")

# ---- Noaniqlik: L bahosi +-1 xato bo'lishi mumkin (ekspert bahosi) -> Monte Karlo ----
print("\n=== Monte Karlo: qoldiq xavf ishtahadan oshish ehtimoli (L va I +-1, 5000 simulyatsiya) ===")
for _, r in reg.head(6).iterrows():
    Ls = np.clip(r.rL + rng.integers(-1, 2, 5000), 1, 5); Is = np.clip(r.rI + rng.integers(-1, 2, 5000), 1, 5)
    print(f"{r.id}: qoldiq={r.qoldiq:2d}  P(qoldiq > {APPETITE}) = {(Ls * Is > APPETITE).mean():.2f}")

def heat(ax, col_L, col_I, title):
    M = np.zeros((5, 5), int)
    for _, r in reg.iterrows(): M[r[col_I] - 1, r[col_L] - 1] += 1
    bg = np.array([[ (l + 1) * (i + 1) for l in range(5)] for i in range(5)])
    ax.imshow(bg, cmap="RdYlGn_r", origin="lower", alpha=.6, vmin=1, vmax=25)
    for i in range(5):
        for l in range(5):
            ids = [r.id for _, r in reg.iterrows() if r[col_L] == l + 1 and r[col_I] == i + 1]
            if ids: ax.text(l, i, "\n".join(ids), ha="center", va="center", fontsize=6)
    ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6)); ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
    ax.set_xlabel("ehtimollik L"); ax.set_ylabel("ta'sir I"); ax.set_title(title)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2)); heat(ax[0], "L", "I", "Boshlang'ich xavf"); heat(ax[1], "rL", "rI", "Qoldiq xavf (choralardan keyin)")
plt.tight_layout()
print("\nIzoh: real tashkilotda L va I bahosi ekspertlar seminarida, dalillar (incidentlar, test natijalari) asosida")
print("beriladi; ISO/IEC 23894 ISO 31000 jarayonini AI ga moslaydi — reyestr muntazam qayta ko'rib chiqiladi.")
