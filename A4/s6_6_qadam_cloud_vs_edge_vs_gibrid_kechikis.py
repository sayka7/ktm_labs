"""A4 — 6-qadam. Cloud vs Edge vs Gibrid: kechikish, trafik, energiya va TCO kalkulyatori

A1 SHV ssenariysi (16 kamera, 160 inferens/s, 3 yil) uchun uch arxitektura hisoblanadi: uchdan-uchga kechikish (RTT + navbat + inferens), tarmoq trafigi, mJ/inferens (PUE bilan), TCO (GPU soati, egress, qo'shimcha xizmatlar; Jetson narxi, energiya, texnik xizmat, zaxira). TCO 1/2/3/5 yil, sezgirlik tahlili (kameralar x2/x8, bulut narxi −30 %, hodisa bo'yicha oqim, xizmat narxi, RTT) va kesishish nuqtasi. Jetson/TPU yo'q — bu simulyatsiya, raqamlar o'zgartiriladigan parametrlar.

Kutilgan natija: Cloud TCO Edge'dan 30–50 marta qimmat, kechikish ~6 marta katta; bulut narxi −30 % ham nisbatni saqlaydi; grafik: TCO yillar bo'yicha va sezgirlik ustunlari.
"""
# 6-qadam. Cloud vs Edge vs Gibrid: kechikish, trafik, energiya va TCO kalkulyatori (A1 SHV ssenariysi) + sezgirlik tahlili
import math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- Berilgan ssenariy va narxlar (taxminiy). Hammasini o'zgartirib ko'ring.
S = dict(
    kameralar=16, fps_har_kamera=10,
    yillar=3, kvt_soat_usd=0.08, pue_cloud=1.3,
    gpu_soat_usd=0.50, gpu_sigim_inf_s=380, zaxira_nusxa=2, rtt_ms=80, bulut_qoshimcha_foiz=30,
    oqim_mbit_kamera=4.0, egress_usd_gb=0.09, gpu_quvvat_w=70, inferens_ms_cloud=3.0, navbat_ms_cloud=25,
    edge_qurilma_usd=700, edge_korpus_usd=300, edge_quvvat_w=25, edge_sigim_inf_s=380, edge_inferens_ms=19,
    edge_xizmat_usd_yil=150, edge_zaxira=1,
    shubhali_ulush=0.05,
)

def hisob(S):
    inf_s = S["kameralar"] * S["fps_har_kamera"]
    soat = S["yillar"] * 365 * 24
    # TODO 1: Cloud — GPU soni (zaxira bilan), GPU soatlari narxi, uzluksiz oqim egress GB, qo'shimcha foiz, kechikish (RTT+navbat+inferens), mJ/inferens (PUE)
    # TODO 2: Edge — qurilmalar soni, kVt*soat energiya, qurilma+korpus (zaxira bilan) + energiya + xizmat, kechikish, mJ/inferens
    # TODO 3: Gibrid — Edge TCO + bulutda faqat shubhali kadrlar uchun qisman GPU va kam egress
    # TODO 4: uch qatorli DataFrame qaytaring: arxitektura, kechikish_p95_ms, trafik_mbit_s, TCO_usd, mJ_inferens, ...
    raise NotImplementedError("TODO 1-4: hisob")

print("=== Cloud vs Edge ===")
df = hisob(S); print(df.to_string(index=False))

print("\n=== TCO yillar bo'yicha (USD) ===")
# TODO 5: 1, 2, 3, 5 yil uchun cloud/edge/gibrid TCO jadvali (dy)

print("\n=== Sezgirlik tahlili (3 yil, TCO USD) ===")
HOLATLAR = (("asosiy", {}), ("kameralar x2", dict(kameralar=32)), ("kameralar x8 (128)", dict(kameralar=128)),
            ("bulut narxi -30 %", dict(gpu_soat_usd=0.35, egress_usd_gb=0.063)), ("hodisa bo'yicha oqim (0.5 Mbit/kamera)", dict(oqim_mbit_kamera=0.5)),
            ("Edge xizmat 500 $/yil", dict(edge_xizmat_usd_yil=500)), ("RTT 150 ms (uzoq mintaqa)", dict(rtt_ms=150)))
# TODO 6: har holat uchun {**S, **kw} bilan hisoblab cloud, edge, gibrid, cloud_edge nisbati va cloud_p95 ni chiqaring

print("\n=== Kesishish nuqtasi ===")
# TODO 7: 1, 2, 4, 8 kamera (zaxirasiz) uchun qaysi arxitektura arzonligini aniqlang
