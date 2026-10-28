"""MT1 — 3-qadam. Onlayn tajriba: A/B test, statistik quvvat va peeking xatosi

Offline metrika yakuniy hukm emas — qarorni onlayn A/B test beradi. Simulyatorda har foydalanuvchining «haqiqiy» didi ma'lum, shuning uchun ikki siyosatning (A — mashhurlik, B — xatoli shaxsiy model) haqiqiy CTR farqi oldindan bilinadi. Tajriba 6000 va 40 000 seans bilan o'tkaziladi: ikki proporsiya z-testi, 95% ishonch oralig'i, so'ngra MDE bo'yicha kerakli tanlanma hajmi hisoblanadi. A/A testda 20 marta oraliq natijaga qarash (peeking) I tur xatoni 5% dan ~25% gacha oshirishi va yangilik effekti ko'rsatiladi.

Kutilgan natija: Haqiqiy o'sish ~+11%; 6000 seansda farq ahamiyatsiz, 40 000 da p < 0,001; MDE 10% uchun har guruhga ~25 ming seans; peeking soxta g'alabani ~0,05 dan ~0,25 ga oshiradi (taxminiy).
"""
# 3-qadam. Onlayn tajriba: A/B test simulyatsiyasi (mashhurlik vs shaxsiy tavsiya), statistik quvvat va «peeking» xatosi
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
# --- «Haqiqiy dunyo»: foydalanuvchining har filmga haqiqiy bahosi (simulyatorda bilamiz, real hayotda — yo'q)
n_u, n_i, k = 600, 400, 6
P, Q = rng.normal(0, 1, (n_u, k)), rng.normal(0, 1, (n_i, k))
pop = 1.0 / np.arange(1, n_i + 1) ** 0.9; pop = pop[rng.permutation(n_i)]
lp = (np.log(pop) - np.log(pop).mean()) / np.log(pop).std()
bi = 0.35 * rng.normal(0, 1, n_i) + 0.35 * lp                   # mashhur filmlar o'rtacha biroz yaxshiroq
haqiqiy = 3.5 + rng.normal(0, 0.35, n_u)[:, None] + bi[None, :] + P @ Q.T / np.sqrt(k) * 0.9

def p_click(baho):
    """Bosish ehtimoli: haqiqiy baho 4 dan yuqori bo'lsa keskin oshadi (taxminiy model)."""
    return 0.10 / (1 + np.exp(-1.5 * (baho - 4.0)))

# --- Ikki siyosat: A — hamma uchun bir xil top-10 mashhur; B — model bahosi (xatoli: shovqin 1,4 — mukammal emas)
topA = np.argsort(-pop)[:10]
taxmin = haqiqiy + rng.normal(0, 1.4, haqiqiy.shape)
topB = np.argsort(-taxmin, axis=1)[:, :10]
ctrA = p_click(haqiqiy[:, topA]).mean(1)                    # har foydalanuvchi uchun kutilgan CTR
ctrB = p_click(np.take_along_axis(haqiqiy, topB, 1)).mean(1)
print("=== A/B test sozlamasi ===")
print(f"Haqiqiy (simulyatorda ma'lum) CTR: A = {ctrA.mean():.4f}, B = {ctrB.mean():.4f}, nisbiy o'sish = {ctrB.mean() / ctrA.mean() - 1:+.1%}")

# --- Tajriba: har seans tasodifiy foydalanuvchi, 50/50 taqsimlash; ikki xil davomiylik bilan
print("\n=== A/B test natijasi ===")
for n in (6000, 40000):                                      # har guruhga seanslar
    ua, ub = rng.integers(0, n_u, n), rng.integers(0, n_u, n)
    xa, xb = rng.random(n) < ctrA[ua], rng.random(n) < ctrB[ub]
    pa, pb = xa.mean(), xb.mean(); pp = (xa.sum() + xb.sum()) / (2 * n)
    z = (pb - pa) / np.sqrt(pp * (1 - pp) * 2 / n); pval = 2 * (1 - stats.norm.cdf(abs(z)))
    se = np.sqrt(pa * (1 - pa) / n + pb * (1 - pb) / n)
    print(f"[n = {n} har guruhga] CTR A = {pa:.4f}, CTR B = {pb:.4f}, farq = {pb - pa:+.4f}, "
          f"95% CI [{pb - pa - 1.96 * se:+.4f}; {pb - pa + 1.96 * se:+.4f}], z = {z:.2f}, p = {pval:.4f}")
    print("   Qaror:", "B ni joriy qilish (farq statistik ahamiyatli)" if pval < 0.05 else "farq ahamiyatsiz — tanlanma kichik (quvvat yetarli emas)")

# --- Statistik quvvat: berilgan MDE uchun har guruhga kerakli seanslar soni
def kerakli_n(p0, mde_nisbiy, alfa=0.05, quvvat=0.8):
    p1 = p0 * (1 + mde_nisbiy); za, zb = stats.norm.ppf(1 - alfa / 2), stats.norm.ppf(quvvat)
    return int(np.ceil((za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2))
print("\n=== Statistik quvvat (alfa 0,05, quvvat 0,8) ===")
print(pd.DataFrame([(f"{m:.0%}", kerakli_n(pa, m)) for m in (0.02, 0.05, 0.10, 0.20, 0.30)],
                   columns=["MDE (nisbiy)", "Har guruhga seans"]).to_string(index=False))

# --- «Peeking»: A/A testda (farq yo'q) har 500 seansda natijaga qarab, birinchi p<0,05 da to'xtatish
eks, qadam, qarash, p0 = 2000, 500, 20, 0.05
ca = np.cumsum(rng.binomial(qadam, p0, (eks, qarash)), 1); cb = np.cumsum(rng.binomial(qadam, p0, (eks, qarash)), 1)
nn = qadam * np.arange(1, qarash + 1)
ppool = (ca + cb) / (2 * nn); zz = (cb - ca) / nn / np.sqrt(ppool * (1 - ppool) * 2 / nn)
fp_bir = np.mean(np.abs(zz[:, -1]) > 1.96); fp_peek = np.mean((np.abs(zz) > 1.96).any(1))
print("\n=== Peeking xatosi (A/A test, 2000 tajriba) ===")
print(f"Faqat oxirida tekshirish: soxta «g'alaba» ulushi = {fp_bir:.3f} (nazariy 0,05)")
print(f"20 marta oraliq qarash:    soxta «g'alaba» ulushi = {fp_peek:.3f}  <- I tur xato bir necha marta oshadi")

# --- Yangilik effekti: B ning ustunligi dastlabki kunlarda bo'rttirilgan (foydalanuvchi yangi interfeysga qiziqadi)
kun = np.arange(1, 15); lift = (ctrB.mean() / ctrA.mean() - 1) * (1 + 0.8 * np.exp(-kun / 3))
print("Yangilik effekti: 1-kun o'sish", f"{lift[0]:+.1%}", "-> 14-kun", f"{lift[-1]:+.1%}", "(kamida 2 hafta o'tkazing)")
plt.figure(figsize=(5.5, 3.2)); plt.plot(nn, np.mean(np.abs(zz) > 1.96, 0), label="har qarashdagi FP")
plt.plot(nn, np.cumsum(np.abs(zz) > 1.96, 1).astype(bool).mean(0), label="to'plangan FP (peeking)")
plt.axhline(0.05, ls="--", c="k"); plt.xlabel("Har guruhdagi seans"); plt.ylabel("Soxta g'alaba ulushi"); plt.legend(); plt.tight_layout()
