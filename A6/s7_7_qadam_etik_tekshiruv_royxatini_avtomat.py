"""A6 — 7-qadam. Etik tekshiruv ro'yxatini avtomatik baholash

20 bandli ro'yxat (ISO/IEC 42001, 23894, EU AI Act, NIST AI RMF) skript bilan baholanadi: texnik bandlar (7, 8, 9, 11, 20) skript ichida tez o'lchangan metrikalardan (toza/FGSM aniqlik, DP/EO farqi, urug' va versiyalar) chegaralar bo'yicha avtomatik, tashkiliy bandlar talaba to'ldiradigan DALIL lug'atidan olinadi; dalilsiz «ha» avtomatik «qisman»ga tushiriladi. Umumiy ball /100 va xulosa chiqadi.

Kutilgan natija: 20 qatorli jadval, umumiy ball ~60/100, bajarilmagan bandlar ro'yxati (masalan 9, 12, 14, 18) va xulosa.
"""
# 7-qadam. Etik tekshiruv ro'yxatini (20 band) avtomatik baholovchi skript
# Texnik bandlar (7, 8, 9, 11, 20) o'lchov natijalaridan avtomatik aniqlanadi; tashkiliy bandlar DALIL lug'atidan.
# Talaba DALIL lug'atini o'z tizimi uchun to'ldiradi: holat ("ha"/"qisman"/"yo'q") va dalil matni.
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
torch.manual_seed(0); np.random.seed(0)

# ---- 1) Tezkor o'lchovlar (laboratoriya tizimi uchun) ----
d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data / 16.0, d.target, test_size=0.3, random_state=0, stratify=d.target)
Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr.astype(np.float32), Xte.astype(np.float32), ytr, yte))
m = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10)); opt = torch.optim.Adam(m.parameters(), 3e-3)
for _ in range(250): opt.zero_grad(); F.cross_entropy(m(Xtr), ytr).backward(); opt.step()
m.eval()
def fgsm(x, y, eps):
    x = x.clone().requires_grad_(True); g, = torch.autograd.grad(F.cross_entropy(m(x), y), x)
    return (x + eps * g.sign()).clamp(0, 1).detach()
with torch.no_grad(): clean_acc = (m(Xte).argmax(1) == yte).float().mean().item()
adv_acc = (m(fgsm(Xte, yte, 0.1)).argmax(1) == yte).float().mean().item()

rng = np.random.default_rng(0); n = 4000
A = rng.binomial(1, 0.4, n); inc = rng.normal(5 - 0.8 * A, 1, n); h = rng.normal(0, 1, n)
yf = ((0.9 * inc + h + rng.normal(0, 1, n)) > 4.5).astype(int)
Xf = np.c_[inc, h]; clf = LogisticRegression().fit(Xf[:3000], yf[:3000]); pf = clf.predict(Xf[3000:])
dp = demographic_parity_difference(yf[3000:], pf, sensitive_features=A[3000:])
eo = equalized_odds_difference(yf[3000:], pf, sensitive_features=A[3000:])
OLCHOV = {"toza_aniqlik": clean_acc, "fgsm_aniqlik_eps0.1": adv_acc, "dp_farqi": dp, "eo_farqi": eo,
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
    """Har band uchun (holat, dalil) qaytaradi. Texnik bandlar avtomatik."""
    o = OLCHOV
    if n == 7:  return ("ha" if o["eo_farqi"] < 0.05 else "qisman"), f"EO farqi={o['eo_farqi']:.3f}, DP farqi={o['dp_farqi']:.3f}"
    if n == 8:  return "ha", f"toza={o['toza_aniqlik']:.3f}, FGSM eps=0.1 -> {o['fgsm_aniqlik_eps0.1']:.3f}"
    if n == 9:  return ("ha" if o["fgsm_aniqlik_eps0.1"] >= 0.5 else "yo'q"), f"adversarial aniqlik={o['fgsm_aniqlik_eps0.1']:.3f}"
    if n == 11: return "qisman", "saliency korrelyatsiyasi o'lchangan, sodiqlik (deletion) o'lchanmagan"
    if n == 20: return ("ha" if o["urug_belgilangan"] and o["versiyalar_qayd"] else "yo'q"), f"seed=0, torch={o['versiyalar_qayd']['torch']}"
    holat, dalil = DALIL.get(n, ("yo'q", ""))
    if holat == "ha" and not dalil.strip(): holat = "qisman"      # dalilsiz "bajarilgan" qabul qilinmaydi
    return holat, dalil or "-"

BALL = {"ha": 5, "qisman": 2.5, "yo'q": 0}
print("=== Etik tekshiruv ro'yxati (ISO/IEC 42001, 23894, EU AI Act, NIST AI RMF) ===")
print(f"{'№':>2s} {'Band':58s} {'Holat':8s} Dalil"); jami = 0; yoq = []
for n in range(1, 21):
    h, dl = baholash(n); jami += BALL[h]; yoq += [n] if h == "yo'q" else []
    print(f"{n:2d} {BANDLAR[n]:58s} {h:8s} {dl[:60]}")
print(f"\nUmumiy ball: {jami:.1f} / 100   bajarilmagan bandlar: {yoq}")
daraja = "yetarli (ishlab chiqarishga tayyor emas, lekin audit boshlanishi mumkin)" if jami >= 70 else \
         "qoniqarsiz — yuqori xavfli sohada ishlatish mumkin emas" if jami < 50 else "qisman — tashkiliy bandlar to'ldirilsin"
print(f"Xulosa: {daraja}")
print("Eslatma: yuqori xavfli tizim (kredit, tibbiyot, bandlik) uchun 12, 14, 18-bandlar majburiy (AI Act 9, 12, 14-m.).")
