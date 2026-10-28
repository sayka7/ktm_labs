"""A4 — 6-qadam. Cloud vs Edge vs Gibrid: kechikish, trafik, energiya va TCO kalkulyatori

A1 SHV ssenariysi (16 kamera, 160 inferens/s, 3 yil) uchun uch arxitektura hisoblanadi: uchdan-uchga kechikish (RTT + navbat + inferens), tarmoq trafigi, mJ/inferens (PUE bilan), TCO (GPU soati, egress, qo'shimcha xizmatlar; Jetson narxi, energiya, texnik xizmat, zaxira). TCO 1/2/3/5 yil, sezgirlik tahlili (kameralar x2/x8, bulut narxi −30 %, hodisa bo'yicha oqim, xizmat narxi, RTT) va kesishish nuqtasi. Jetson/TPU yo'q — bu simulyatsiya, raqamlar o'zgartiriladigan parametrlar.

Kutilgan natija: Cloud TCO Edge'dan 30–50 marta qimmat, kechikish ~6 marta katta; bulut narxi −30 % ham nisbatni saqlaydi; grafik: TCO yillar bo'yicha va sezgirlik ustunlari.
"""
# 6-qadam. Cloud vs Edge vs Gibrid: kechikish, trafik, energiya va TCO kalkulyatori (A1 SHV ssenariysi) + sezgirlik tahlili
import math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# --- Ssenariy va narxlar (ochiq narxlar, taxminiy; hisobotda sana va manba ko'rsatiladi). Hammasini o'zgartirib ko'ring.
S = dict(
    kameralar=16, fps_har_kamera=10,          # 16 kamera x 10 kadr/s = 160 inferens/s
    yillar=3, kvt_soat_usd=0.08, pue_cloud=1.3,
    # Cloud (T4 GPU, K8s)
    gpu_soat_usd=0.50, gpu_sigim_inf_s=380, zaxira_nusxa=2, rtt_ms=80, bulut_qoshimcha_foiz=30,   # egress, LB, loglar, monitoring
    oqim_mbit_kamera=4.0, egress_usd_gb=0.09, gpu_quvvat_w=70, inferens_ms_cloud=3.0, navbat_ms_cloud=25,
    # Edge (Jetson Orin NX)
    edge_qurilma_usd=700, edge_korpus_usd=300, edge_quvvat_w=25, edge_sigim_inf_s=380, edge_inferens_ms=19,
    edge_xizmat_usd_yil=150, edge_zaxira=1,     # texnik xizmat + bitta zaxira qurilma
    # Gibrid: chekkada hamma, bulutga shubhali kadrlar ulushi
    shubhali_ulush=0.05,
)

def hisob(S):
    inf_s = S["kameralar"] * S["fps_har_kamera"]
    soat = S["yillar"] * 365 * 24
    # ---- Cloud
    gpu_n = max(S["zaxira_nusxa"], math.ceil(inf_s / S["gpu_sigim_inf_s"]))
    gpu_usd = gpu_n * S["gpu_soat_usd"] * soat
    trafik_mbit = S["kameralar"] * S["oqim_mbit_kamera"]                                   # uzluksiz oqim
    egress_gb = trafik_mbit / 8 / 1024 * 3600 * soat
    cloud_usd = (gpu_usd + egress_gb * S["egress_usd_gb"]) * (1 + S["bulut_qoshimcha_foiz"] / 100)
    cloud_lat = S["rtt_ms"] + S["navbat_ms_cloud"] + S["inferens_ms_cloud"]
    cloud_mj = S["gpu_quvvat_w"] * gpu_n / inf_s * 1000 * S["pue_cloud"]                   # to'liq (GPU band bo'lmasa ham ishlaydi)
    # ---- Edge
    edge_n = math.ceil(inf_s / S["edge_sigim_inf_s"])
    kwh = S["edge_quvvat_w"] * edge_n * soat / 1000
    edge_usd = (edge_n + S["edge_zaxira"]) * (S["edge_qurilma_usd"] + S["edge_korpus_usd"]) + kwh * S["kvt_soat_usd"] + S["edge_xizmat_usd_yil"] * S["yillar"] * edge_n
    edge_lat = S["edge_inferens_ms"]
    edge_mj = S["edge_quvvat_w"] * edge_n / inf_s * 1000
    # ---- Gibrid: Edge + bulutda faqat shubhali kadrlar (kichik GPU ulushi, egress kam)
    sh_inf_s = inf_s * S["shubhali_ulush"]
    gib_gpu_usd = 1 * S["gpu_soat_usd"] * soat * max(S["shubhali_ulush"] * 4, 0.1)        # 1 GPU qisman band (serverless/spot taxmini)
    gib_egress_gb = egress_gb * S["shubhali_ulush"] / 10                                   # faqat kesilgan kadrlar
    gib_usd = edge_usd + (gib_gpu_usd + gib_egress_gb * S["egress_usd_gb"]) * (1 + S["bulut_qoshimcha_foiz"] / 100)
    return pd.DataFrame([
        dict(arxitektura="Cloud (T4, K8s)", kechikish_p95_ms=cloud_lat, trafik_mbit_s=round(trafik_mbit, 1), TCO_usd=round(cloud_usd),
             mJ_inferens=round(cloud_mj), oflayn="yo'q", maxfiylik="video chiqadi", nusxa=f"{gpu_n} GPU"),
        dict(arxitektura="Edge (Jetson Orin NX)", kechikish_p95_ms=edge_lat, trafik_mbit_s=0.01, TCO_usd=round(edge_usd),
             mJ_inferens=round(edge_mj), oflayn="ha", maxfiylik="video binoda", nusxa=f"{edge_n}+{S['edge_zaxira']} qurilma"),
        dict(arxitektura="Gibrid (Edge + bulut tekshiruv)", kechikish_p95_ms=f"{edge_lat} / {cloud_lat}", trafik_mbit_s=round(trafik_mbit * S["shubhali_ulush"] / 10, 2),
             TCO_usd=round(gib_usd), mJ_inferens=round(edge_mj * 1.1), oflayn="asosiy ha", maxfiylik="faqat kadrlar", nusxa=f"{edge_n} qurilma + 1 GPU qisman"),
    ])

print("=== Cloud vs Edge ===")
print(f"Ssenariy: {S['kameralar']} kamera x {S['fps_har_kamera']} kadr/s = {S['kameralar'] * S['fps_har_kamera']} inferens/s, {S['yillar']} yil")
df = hisob(S); print(df.to_string(index=False))
nisbat = df.TCO_usd[0] / df.TCO_usd[1]
print(f"\nCloud TCO Edge'dan {nisbat:.1f} marta qimmat; kechikish {df.kechikish_p95_ms[0] / df.kechikish_p95_ms[1]:.1f} marta katta.")
print("Cloud mJ/inferens katta, chunki 2 ta zaxira GPU 160 inferens/s da kam band (sig'im 760/s) — bo'sh quvvat ham hisobga kiradi (PUE 1,3 bilan).")

# --- TCO yillar bo'yicha (1, 3, 5): Edge kapital xarajat oldinda, Cloud chiziqli o'sadi
print("\n=== TCO yillar bo'yicha (USD) ===")
rows = []
for y in (1, 2, 3, 5):
    d = hisob({**S, "yillar": y}); rows.append(dict(yil=y, cloud=d.TCO_usd[0], edge=d.TCO_usd[1], gibrid=d.TCO_usd[2]))
dy = pd.DataFrame(rows); print(dy.to_string(index=False))

# --- Sezgirlik tahlili
print("\n=== Sezgirlik tahlili (3 yil, TCO USD) ===")
rows = []
for nomi, kw in (("asosiy", {}), ("kameralar x2", dict(kameralar=32)), ("kameralar x8 (128)", dict(kameralar=128)),
                 ("bulut narxi -30 %", dict(gpu_soat_usd=0.35, egress_usd_gb=0.063)), ("hodisa bo'yicha oqim (0.5 Mbit/kamera)", dict(oqim_mbit_kamera=0.5)),
                 ("Edge xizmat 500 $/yil", dict(edge_xizmat_usd_yil=500)), ("RTT 150 ms (uzoq mintaqa)", dict(rtt_ms=150))):
    d = hisob({**S, **kw}); rows.append(dict(holat=nomi, cloud=d.TCO_usd[0], edge=d.TCO_usd[1], gibrid=d.TCO_usd[2],
                                            cloud_edge=round(d.TCO_usd[0] / d.TCO_usd[1], 1), cloud_p95=d.kechikish_p95_ms[0]))
ds = pd.DataFrame(rows); print(ds.to_string(index=False))

# Qachon Cloud arzonroq? Kameralar soni juda kam bo'lsa (bitta kamera uchun ham Jetson sotib olinadi) yoki yuklama notekis bo'lsa
print("\n=== Kesishish nuqtasi ===")
for kam in (1, 2, 4, 8):
    d = hisob({**S, "kameralar": kam, "zaxira_nusxa": 1, "edge_zaxira": 0})
    print(f"  {kam} kamera, zaxirasiz: cloud {d.TCO_usd[0]:>7} $   edge {d.TCO_usd[1]:>6} $   -> {'Cloud' if d.TCO_usd[0] < d.TCO_usd[1] else 'Edge'} arzon")
print("Xulosa: SHV ssenariysida (uzluksiz 160 inferens/s, maxfiy video) Edge aniq ustun; Cloud 10 000 foydalanuvchili mobil ilova kabi")
print("  notekis, geografik tarqoq yuklamada oqilona. Gibrid: chekkada 95 % qaror, bulutda sifat nazorati va qayta o'qitish uchun ma'lumot.")
print("Eslatma: bu simulyatsiya — Jetson/TPU/GPU o'lchovlari emas; real labda raqamlar tegrastats, nvidia-smi va bulut hisob-fakturasidan olinadi.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(dy.yil, dy.cloud, "o-", label="Cloud"); ax[0].plot(dy.yil, dy.edge, "s-", label="Edge"); ax[0].plot(dy.yil, dy.gibrid, "^-", label="Gibrid")
ax[0].set_xlabel("yil"); ax[0].set_ylabel("TCO, USD"); ax[0].set_title("TCO yillar bo'yicha"); ax[0].legend(); ax[0].grid(alpha=.3)
xx = np.arange(len(ds)); w = 0.27
ax[1].bar(xx - w, ds.cloud, w, label="Cloud"); ax[1].bar(xx, ds.edge, w, label="Edge"); ax[1].bar(xx + w, ds.gibrid, w, label="Gibrid")
ax[1].set_xticks(xx); ax[1].set_xticklabels(ds.holat, rotation=25, ha="right", fontsize=7); ax[1].set_yscale("log"); ax[1].set_ylabel("TCO 3 yil, USD (log)")
ax[1].set_title("Sezgirlik tahlili"); ax[1].legend(); ax[1].grid(axis="y", alpha=.3)
plt.tight_layout()
