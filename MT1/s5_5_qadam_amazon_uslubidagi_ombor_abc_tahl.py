"""MT1 — 5-qadam. Amazon uslubidagi ombor: ABC tahlili, slotting va zaxira siyosati

300 SKU uchun 120 kunlik sintetik talab bo'yicha ABC tahlili bajariladi. So'ngra 300 joyli omborda tovarlarni joylashtirish (slotting) uch usulda solishtiriladi: tasodifiy, talab bo'yicha saralash va Venger usuli (`linear_sum_assignment`); 3000 buyurtma bo'yicha o'rtacha terish yo'li o'lchanadi. A sinf tovarlari uchun xavfsizlik zaxirasi, qayta buyurtma nuqtasi (ROP) va EOQ hisoblanadi va (s, Q) siyosati barqaror hamda tasodifiy yetkazish muddati bilan simulyatsiya qilinadi.

Kutilgan natija: A sinf: ~17% SKU qiymatning 80% i; talab bo'yicha slotting terish yo'lini ~55% qisqartiradi; tasodifiy yetkazish muddatida fill rate ~1,0 dan ~0,93 ga tushadi (taxminiy). Slotting xaritasi.
"""
# 5-qadam. Amazon uslubidagi ombor: ABC tahlili, talab bo'yicha joylashtirish (slotting) va zaxira siyosati
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
# --- 300 SKU, 120 kunlik sintetik talab: Zipf mashhurlik + hafta kuni mavsumiyligi
n_sku, D = 300, 120
mashhur = 1.0 / np.arange(1, n_sku + 1) ** 1.1; mashhur = mashhur[rng.permutation(n_sku)]
kunlik_o = 4000 * mashhur / mashhur.sum()                                  # kuniga ~4000 dona jami
hafta = 1 + 0.25 * np.sin(2 * np.pi * np.arange(D) / 7)
talab = rng.poisson(np.outer(hafta, kunlik_o))                             # D x n_sku
narx = rng.lognormal(3.0, 0.8, n_sku)                                      # dona narxi, $ (taxminiy)

# --- ABC tahlili (yillik qiymat bo'yicha): A — qiymatning 80%, B — keyingi 15%, C — qolgan 5%
qiymat = talab.sum(0) * narx
tart = np.argsort(-qiymat); ulush = np.cumsum(qiymat[tart]) / qiymat.sum()
sinf = np.empty(n_sku, "<U1"); sinf[tart] = np.where(ulush <= 0.8, "A", np.where(ulush <= 0.95, "B", "C"))
print("=== ABC tahlili ===")
abc = pd.DataFrame(dict(sinf=sinf, qiymat=qiymat, dona=talab.sum(0))).groupby("sinf").agg(
    SKU=("qiymat", "size"), qiymat_ulushi=("qiymat", "sum"), dona_ulushi=("dona", "sum"))
abc["qiymat_ulushi"] /= qiymat.sum(); abc["dona_ulushi"] /= talab.sum()
print(abc.round(3).to_string())

# --- Slotting: 300 joy (20 qator x 15 ustun), qadoqlash stantsiyasi (0,0) da; masofa — Manhattan (m)
xy = np.array([(r * 3.0, c * 2.0) for r in range(20) for c in range(15)])
masofa = np.abs(xy).sum(1)
tezlik = talab[:90].mean(0)                                                # birinchi 90 kun — o'rganish davri
def yigish_masofasi(joy):
    """Oxirgi 30 kun buyurtmalari: har buyurtmada 1–4 SKU, terish yo'li = eng uzoq joygacha borib-qaytish (S-shakl yaqinlashuvi)."""
    p = talab[90:].sum(0) / talab[90:].sum(); jami = 0.0; r = np.random.default_rng(0)
    for _ in range(3000):
        sku = r.choice(n_sku, r.integers(1, 5), p=p)
        jami += 2 * masofa[joy[sku]].max()
    return jami / 3000
tasodifiy = rng.permutation(n_sku)
abc_joy = np.empty(n_sku, int); abc_joy[np.argsort(-tezlik)] = np.argsort(masofa)   # tez aylanadigan — yaqinga
# Optimal tayinlash (Venger usuli): xarajat = kunlik terishlar x masofa
r_idx, c_idx = linear_sum_assignment(np.outer(tezlik, masofa)); opt_joy = np.empty(n_sku, int); opt_joy[r_idx] = c_idx
print("\n=== Slotting: o'rtacha terish yo'li (m/buyurtma) ===")
rows = [(n, yigish_masofasi(j)) for n, j in [("Tasodifiy joylashtirish", tasodifiy),
                                            ("Talab bo'yicha saralash", abc_joy), ("Venger usuli (optimal)", opt_joy)]]
base = rows[0][1]
print(pd.DataFrame([(n, m, 1 - m / base) for n, m in rows], columns=["Usul", "Yo'l, m", "Tejash"]).round(3).to_string(index=False))
print("Izoh: chiziqli xarajatda saralash va Venger bir xil yechim beradi; amalda og'irlik, hajm, birga buyurtma qilinish cheklovlari qo'shiladi.")

# --- Zaxira siyosati (A sinf uchun): xavfsizlik zaxirasi, qayta buyurtma nuqtasi va EOQ
L, xizmat, S_buy, h = 5, 0.95, 50.0, 0.25                                  # yetkazish 5 kun, servis 95%, buyurtma xarajati $, saqlash 25%/yil
z = stats.norm.ppf(xizmat)
A = np.where(sinf == "A")[0][:8]
d, sig = talab[:90, A].mean(0), talab[:90, A].std(0)
ss = z * sig * np.sqrt(L); rop = d * L + ss; eoq = np.sqrt(2 * d * 365 * S_buy / (h * narx[A]))
print("\n=== Zaxira siyosati (A sinfdan 8 SKU) ===")
print(pd.DataFrame(dict(SKU=A, kunlik_talab=d, sigma=sig, xavfsizlik_zaxirasi=ss, ROP=rop, EOQ=eoq)).round(1).to_string(index=False))
# Simulyatsiya: oxirgi 30 kunda (s, Q) siyosati; yetkazish muddati barqaror (5 kun) yoki tasodifiy (3–9 kun)
def simulyatsiya(lt_tasodifiy, seed=0):
    r = np.random.default_rng(seed); tanq = jami = 0; tsikl = tsikl_tanq = 0
    for j, sku in enumerate(A):
        zaxira, yolda = rop[j], []
        for k in range(90, D):
            zaxira += sum(q for (kel, q) in yolda if kel == k); yolda = [(kel, q) for (kel, q) in yolda if kel > k]
            sotildi = min(zaxira, talab[k, sku]); tanq += talab[k, sku] - sotildi; jami += talab[k, sku]; zaxira -= sotildi
            if zaxira + sum(q for _, q in yolda) <= rop[j]:
                lt = int(r.integers(3, 10)) if lt_tasodifiy else L
                yolda.append((k + lt, eoq[j])); tsikl += 1
            tsikl_tanq += int(sotildi < talab[k, sku])
    return 1 - tanq / jami, tsikl, tsikl_tanq
for nom, f in [("barqaror 5 kun", False), ("tasodifiy 3–9 kun", True)]:
    fr, ts, tt = simulyatsiya(f)
    print(f"Simulyatsiya (30 kun, yetkazish {nom}): fill rate = {fr:.3f}, buyurtmalar = {ts}, tanqislik kunlari = {tt}")
print("Xulosa: yetkazish muddati noaniqligi formulaga kiritilmasa (sigma_L), servis darajasi rejadan past bo'ladi.")

plt.figure(figsize=(5, 4)); plt.scatter(xy[:, 1], xy[:, 0], c=tezlik[np.argsort(opt_joy)], cmap="viridis", s=25)
plt.colorbar(label="kunlik terish (dona)"); plt.scatter([0], [0], marker="*", s=200, c="red"); plt.title("Optimal slotting xaritasi")
plt.xlabel("ustun, m"); plt.ylabel("qator, m"); plt.tight_layout()
