"""A6 — 7-qadam. Etik tekshiruv ro'yxatini avtomatik baholash

20 bandli ro'yxat (ISO/IEC 42001, 23894, EU AI Act, NIST AI RMF) skript bilan baholanadi: texnik bandlar (7, 8, 9, 11, 20) skript ichida tez o'lchangan metrikalardan (toza/FGSM aniqlik, DP/EO farqi, urug' va versiyalar) chegaralar bo'yicha avtomatik, tashkiliy bandlar talaba to'ldiradigan DALIL lug'atidan olinadi; dalilsiz «ha» avtomatik «qisman»ga tushiriladi. Umumiy ball /100 va xulosa chiqadi.

Kutilgan natija: 20 qatorli jadval, umumiy ball ~60/100, bajarilmagan bandlar ro'yxati (masalan 9, 12, 14, 18) va xulosa.
"""
# 7-qadam. Etik tekshiruv ro'yxatini (20 band) avtomatik baholovchi skript
# Texnik bandlar (7, 8, 9, 11, 20) o'lchov natijalaridan avtomatik aniqlanadi; tashkiliy bandlar DALIL lug'atidan.
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
torch.manual_seed(0); np.random.seed(0)

# ---- 1) Tezkor o'lchovlar ----
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data / 16.0, d.target, test_size=0.3, random_state=0, stratify=d.target)
Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr.astype(np.float32), Xte.astype(np.float32), ytr, yte))
# TODO 1: kichik MLP o'qiting; toza aniqlik va FGSM (eps=0.1) aniqligini o'lchang
# TODO 2: sintetik kredit ma'lumotida (A, income, history) LogisticRegression; DP va EO farqlarini o'lchang
OLCHOV = {"toza_aniqlik": None, "fgsm_aniqlik_eps0.1": None, "dp_farqi": None, "eo_farqi": None,
          "urug_belgilangan": True, "versiyalar_qayd": {"torch": torch.__version__}}

# ---- 2) Dalillar (talaba to'ldiradi) ----
DALIL = {
 1: ("ha", "model_card.md: maqsad — raqam tasnifi, foydalanuvchi — laboratoriya"),
 2: ("ha", "xavf toifasi: minimal (o'quv demo); kredit modeli bo'lsa — yuqori (AI Act III ilova)"),
 3: ("ha", "sklearn digits (BSD litsenziya), sintetik kredit ma'lumoti (generator kodi)"),
 4: ("qisman", "sinflar vakilligi tekshirilgan, guruhlar bo'yicha yo'q"),
 5: ("ha", "sezgir atribut A, vakil belgi — income (korr. bilan)"),
 6: ("ha", "mezon: equalized odds — noto'g'ri rad etish zarari og'irroq"),
 10: ("ha", "gradient-saliency xaritasi (4-qadam)"),
 12: ("yo'q", ""), 13: ("qisman", "interfeysda AI haqida yozuv rejalashtirilgan"),
 14: ("yo'q", ""), 15: ("qisman", "drift monitoringi rejasi yozilgan, amalga oshirilmagan"),
 16: ("ha", "shaxsiy ma'lumot yo'q (ochiq to'plam)"), 17: ("qisman", "model card bor, datasheet yo'q"),
 18: ("yo'q", ""), 19: ("qisman", "mas'ul shaxs ko'rsatilgan, shikoyat mexanizmi yo'q"),
}
BANDLAR = {1: "Maqsad va ta'sir ko'lami hujjatlashtirilgan", 2: "Xavf darajasi aniqlangan", 3: "Ma'lumot manbai/litsenziyasi",
 4: "Guruhlar vakilligi tahlili", 5: "Sezgir atributlar va proxy aniqlangan", 6: "Adolatlilik mezoni asoslangan",
 7: "Adolatlilik ko'rsatkichlari o'lchangan (chegara: EO<0.05)", 8: "Toza va adversarial aniqlik o'lchangan",
 9: "Himoya chorasi sinovdan o'tgan (adv aniqlik >= 50%)", 10: "XAI tushuntirish mexanizmi mavjud",
 11: "Tushuntirish ishonchliligi tekshirilgan", 12: "Inson nazorati nuqtalari", 13: "Foydalanuvchi AI'ni biladi",
 14: "Audit izi (jurnal)", 15: "Drift/adolatlilik monitoringi", 16: "Shaxsiy ma'lumot minimallashtirilgan",
 17: "Model card va datasheet", 18: "Nosozlik rejimi (fallback)", 19: "Mas'ul shaxs va shikoyat mexanizmi",
 20: "Takrorlanuvchanlik (urug', versiyalar)"}

def baholash(n):
    """Har band uchun (holat, dalil) qaytaradi."""
    # TODO 3: texnik bandlar (7, 8, 9, 11, 20) — OLCHOV va band nomidagi chegaralar bo'yicha avtomatik holat
    # TODO 4: qolganlari DALIL dan; dalilsiz "ha" -> "qisman"
    raise NotImplementedError("TODO 3-4: baholash")

BALL = {"ha": 5, "qisman": 2.5, "yo'q": 0}
print("=== Etik tekshiruv ro'yxati (ISO/IEC 42001, 23894, EU AI Act, NIST AI RMF) ===")
# TODO 5: 20 bandni baholab jadvalni chiqaring, jami ballni va bajarilmagan bandlarni yig'ing; "Umumiy ball: .. / 100" va xulosa daraja
for n in range(1, 21):
    h, dl = baholash(n)
