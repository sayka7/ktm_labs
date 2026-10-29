"""MT3 — 6-qadam. Energiya samaradorligi (TOP/s/Vt, mJ/tasvir) va 3 yillik TCO

ResNet-50 sinfidagi tasnif (~8,2 GOP/tasvir) va 5000 tasvir/s talab uchun har arxitekturada kerakli qurilmalar soni, umumiy quvvat, energiya (mJ/tasvir), CAPEX, elektr xarajati (PUE bilan) va egalik/bulut TCO hisoblanadi; natija bitta mln tasvir narxiga keltiriladi. Elektr narxi bo'yicha sezgirlik, 30 % yuklamada bulut va egalik solishtiruvi hamda chekka stsenariysi (120 tasvir/s, 30 Vt) ko'rib chiqiladi. Narx, quvvat va real foydalanish koeffitsientlari — taxminiy faraz, ular kod boshida o'zgartiriladi.

Kutilgan natija: TOP/s/Vt: NPU ~10, GPU ~2,8, CPU ~0,08; mJ/tasvir NPU ~3,5, GPU ~10, CPU ~470; 24/7 da GPU egaligi bulutdan arzon, 30 % yuklamada bulut afzal; chekkada faqat NPU mos.
"""
# 6-qadam. Energiya samaradorligi (TOP/s/Vt, J/inferens) va 3 yillik TCO taqqoslash
# Ssenariy: ResNet-50 sinfidagi tasnif (~8,2 GOP/tasvir), talab — 5000 tasvir/s, 24/7. Barcha raqamlar TAXMINIY faraz.
import math
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 170)
GOP_TASVIR = 8.2           # ResNet-50, bitta tasvir, GOP
TALAB = 5000               # tasvir/s
SOAT = 3 * 365 * 24        # 3 yil, soat
NARX_KVT = 0.10            # USD / kVt*soat (taxminiy)
PUE = 1.4                  # ma'lumot markazi: sovutish va boshqa yo'qotishlar koeffitsienti

# nom: cho'qqi INT8 TOP/s, real foydalanish, yuklamadagi quvvat Vt,
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
    # TODO 1: bitta qurilmaning tasvir/s ini va talab uchun kerakli qurilmalar sonini toping
    # TODO 2: umumiy quvvat, 3 yillik energiya (PUE bilan, kVt*soat) va mJ/tasvir ni hisoblang
    # TODO 3: CAPEX, elektr xarajati, egalik va bulut TCO, sent / 1 mln tasvir ni hisoblang (None larni hisobga oling)
    raise NotImplementedError("TODO 1-3: energiya va TCO")
df = pd.DataFrame(satr)
print(f"=== Energiya va TCO: talab {TALAB} tasvir/s, 3 yil, {NARX_KVT} $/kVt*soat, PUE {PUE} (taxminiy) ===")
print(df.to_string(index=False))

print("\n=== Sezgirlik: egalik TCO (k$) elektr narxiga qarab ===")
# TODO 4: elektr narxi 0.05, 0.10, 0.20, 0.40 $ bo'lganda egalik TCO ni chiqaring
# TODO 5: 30 % yuklamada bulut (faqat ish vaqti) va egalik TCO ni solishtiring

print("\n=== Chekka stsenariysi: 120 tasvir/s, quvvat <= 30 Vt ===")
# TODO 6: har qurilma uchun 120 tasvir/s ga kerakli soni va quvvatni topib, 30 Vt byudjetga mosligini ayting

print("\n=== TCO tahlili tugadi ===")
