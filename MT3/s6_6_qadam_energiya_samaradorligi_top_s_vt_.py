"""MT3 — 6-qadam. Energiya samaradorligi (TOP/s/Vt, mJ/tasvir) va 3 yillik TCO

ResNet-50 sinfidagi tasnif (~8,2 GOP/tasvir) va 5000 tasvir/s talab uchun har arxitekturada kerakli qurilmalar soni, umumiy quvvat, energiya (mJ/tasvir), CAPEX, elektr xarajati (PUE bilan) va egalik/bulut TCO hisoblanadi; natija bitta mln tasvir narxiga keltiriladi. Elektr narxi bo'yicha sezgirlik, 30 % yuklamada bulut va egalik solishtiruvi hamda chekka stsenariysi (120 tasvir/s, 30 Vt) ko'rib chiqiladi. Narx, quvvat va real foydalanish koeffitsientlari — taxminiy faraz, ular kod boshida o'zgartiriladi.

Kutilgan natija: TOP/s/Vt: NPU ~10, GPU ~2,8, CPU ~0,08; mJ/tasvir NPU ~3,5, GPU ~10, CPU ~470; 24/7 da GPU egaligi bulutdan arzon, 30 % yuklamada bulut afzal; chekkada faqat NPU mos.
"""
# 6-qadam. Energiya samaradorligi (TOP/s/Vt, J/inferens) va 3 yillik TCO taqqoslash
# Ssenariy: ResNet-50 sinfidagi tasvir tasnifi (~4,1 GMAC = 8,2 GOP/tasvir), talab — 5000 tasvir/s, 24/7.
# Barcha narx, quvvat va "real foydalanish" koeffitsientlari TAXMINIY faraz — o'z qiymatlaringizni qo'ying.
import math
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 170)
GOP_TASVIR = 8.2           # ResNet-50, bitta tasvir, GOP
TALAB = 5000               # tasvir/s
SOAT = 3 * 365 * 24        # 3 yil, soat
NARX_KVT = 0.10            # USD / kVt*soat (taxminiy)
PUE = 1.4                  # ma'lumot markazi: sovutish va boshqa yo'qotishlar koeffitsienti

# nom: cho'qqi INT8 TOP/s, real foydalanish (MLPerf'dagi kabi tartib, taxminiy), yuklamadagi quvvat Vt,
#      karta narxi USD, xost ulushi USD, xost quvvat ulushi Vt, bulut USD/soat (yo'q = None)
Q = {
    "CPU EPYC 9654":    (29.5, 0.30, 360, 11800, 4000, 150, None),
    "GPU H100 SXM":     (1979.0, 0.35, 700, 30000, 6000, 150, 3.00),
    "TPU v4 (bulut)":   (275.0, 0.50, 200, None, None, 0, 3.22),
    "NPU Hailo-8":      (26.0, 0.50, 2.5, 250, 300, 3, None),
    "Jetson AGX Orin":  (138.0, 0.30, 60, 2000, 0, 0, None),
    "FPGA Alveo U250":  (33.3, 0.60, 225, 9000, 3000, 100, None),
}
satr = []
for nom, (pk, u, vt, narx, xost, xvt, bulut) in Q.items():
    ips = pk * u * 1e3 / GOP_TASVIR                     # tasvir/s bitta qurilmada
    n = math.ceil(TALAB / ips)
    P = n * (vt + xvt)                                  # umumiy quvvat, Vt
    energiya_kvt = P * SOAT / 1e3 * PUE
    capex = None if narx is None else n * (narx + xost)
    opex_el = energiya_kvt * NARX_KVT
    tco_own = None if capex is None else capex + opex_el
    tco_cloud = None if bulut is None else n * bulut * SOAT
    tco = min(x for x in (tco_own, tco_cloud) if x is not None)
    satr.append({"qurilma": nom, "TOPS/Vt (cho'qqi)": pk / vt, "tasvir/s/qurilma": ips, "soni": n,
                 "quvvat_kVt": P / 1e3, "mJ/tasvir": 1e3 * (vt + xvt) / ips, "CAPEX_k$": (capex or 0) / 1e3,
                 "elektr_k$": opex_el / 1e3, "egalik_TCO_k$": None if tco_own is None else tco_own / 1e3,
                 "bulut_TCO_k$": None if tco_cloud is None else tco_cloud / 1e3,
                 "c/mln tasvir": 100 * tco / (TALAB * SOAT * 3600 / 1e6)})     # sent (cent) / 1 mln tasvir
df = pd.DataFrame(satr)
print(f"=== Energiya va TCO: talab {TALAB} tasvir/s, 3 yil, {NARX_KVT} $/kVt*soat, PUE {PUE} (taxminiy) ===")
print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}", na_rep="—"))

# --- Sezgirlik: elektr narxi va o'rtacha yuklama (load factor) TCO ga qanday ta'sir qiladi ---
print("\n=== Sezgirlik: egalik TCO (k$) elektr narxiga qarab ===")
print(f"{'qurilma':18s}" + "".join(f"{p:>10.2f}$" for p in (0.05, 0.10, 0.20, 0.40)))
for _, r in df.iterrows():
    if pd.isna(r["egalik_TCO_k$"]): continue
    capex = r["CAPEX_k$"]; el = r["elektr_k$"]
    print(f"{r.qurilma:18s}" + "".join(f"{capex + el * p / NARX_KVT:11.1f}" for p in (0.05, 0.10, 0.20, 0.40)))

print("\n=== Yuklama 30 % bo'lsa (kunduzgi cho'qqi, tunda bo'sh): bulut vs egalik ===")
for _, r in df.iterrows():
    if pd.isna(r["bulut_TCO_k$"]): continue
    own = "—" if pd.isna(r["egalik_TCO_k$"]) else f"{r['egalik_TCO_k$']:.1f}"
    print(f"{r.qurilma:18s} egalik {own:>8s} k$ | bulut (faqat ish vaqti) "
          f"{0.3 * r['bulut_TCO_k$']:8.1f} k$")
print("Xulosa: doimiy 24/7 yuklamada egalik arzonroq bo'lishi mumkin, o'zgaruvchan yuklamada bulut afzal.")

# --- Chekka (edge) cheklovi: 4 kamera x 30 kadr/s, quvvat byudjeti 30 Vt ---
print("\n=== Chekka stsenariysi: 120 tasvir/s, quvvat <= 30 Vt ===")
for nom, (pk, u, vt, narx, xost, xvt, bulut) in Q.items():
    ips = pk * u * 1e3 / GOP_TASVIR; n = math.ceil(120 / ips); P = n * (vt + xvt)
    holat = "MOS" if P <= 30 and bulut is None else ("bulut — chekka emas" if narx is None else "quvvat oshadi")
    print(f"{nom:18s} {n:3d} dona, {P:7.1f} Vt -> {holat}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(df.qurilma, df["mJ/tasvir"]); ax[0].set_yscale("log"); ax[0].set_title("Energiya, mJ/tasvir (log)")
ax[1].bar(df.qurilma, df["c/mln tasvir"], color="tab:orange"); ax[1].set_title("Narx, sent / 1 mln tasvir")
for a in ax: a.tick_params(axis="x", rotation=45, labelsize=7)
plt.tight_layout()
print("\n=== TCO tahlili tugadi ===")
