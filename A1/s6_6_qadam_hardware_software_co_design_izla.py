"""A1 — 6-qadam. Hardware–Software Co-design: izlanish fazosini grid search bilan ko'rib chiqish

Dasturiy qarorlar (3 model × 3 imgsz × 2 format × 3 batch) va apparat qarorlari (6 platforma) birgalikda — 270 ta konfiguratsiya — zaxira, kechikish, quvvat, narx, harorat va format cheklovlari bo'yicha baholanadi. Yaroqli konfiguratsiyalar orasidan mAP bo'yicha eng yaxshilari, har platforma uchun eng yaxshi variant va narx–sifat Pareto chegarasi topiladi; grafikda yaroqli (rangli) va yaroqsiz (kulrang) nuqtalar ko'rsatiladi. Cheklov qiymatlarini o'zgartirib, izlanish fazosi qanday torayishini kuzating.

Kutilgan natija: 270 konfiguratsiya, ~118 yaroqli; top-10 jadval (yolov8m AGX Orin / Orin NX INT8), Pareto ro'yxati (Hailo-8 → Orin NX → AGX Orin), scatter grafik.
"""
# 6-qadam. Hardware-Software Co-design: izlanish fazosini to'liq ko'rib chiqish (grid search)
# Dasturiy qarorlar (model, kirish o'lchami, format, batch) x apparat qarorlari (platforma) birgalikda
# baholanadi. Cheklovlar: o'tkazuvchanlik zaxirasi, inferens kechikishi, quvvat, narx, harorat, format.
# Maqsad: cheklovlarni qanoatlantiruvchi eng sifatli (mAP) va arzon konfiguratsiyalar (Pareto chegarasi).
import itertools
import pandas as pd
import matplotlib.pyplot as plt

KAMERALAR, FPS = 16, 10
INFERENS_S = KAMERALAR * FPS
INFERENS_BYUDJET_MS = 60       # kechikish byudjetidan inferensga ajratilgan qism
MIN_ZAXIRA = 1.5               # o'tkazuvchanlik zaxirasi (pik yuklama uchun)
MAKS_VT, MAKS_NARX = 60, 2000

# model: (GFLOP @640, COCO mAP @640)
MODELLAR = {"yolov8n": (8.7, 37.3), "yolov8s": (28.6, 44.9), "yolov8m": (78.9, 50.2)}
IMGSZ = (416, 512, 640)
FORMATLAR = {"FP16": 2, "INT8": 1}            # bayt/parametr (ma'lumot uchun)
BATCHLAR = (1, 4, 8)
# platforma: INT8 dense TOPS, samaradorlik, Vt, narx, NVDEC 1080p30 oqim, FP16 bor, INT8 bor, sanoat harorati
PLATFORMALAR = {
    "Orin NX 16GB":  (50,  0.45, 25,  650,   7, True,  True,  True),
    "AGX Orin 64GB": (138, 0.45, 60,  2000, 14, True,  True,  True),
    "A100 PCIe":     (624, 0.50, 300, 10000, 10, True,  True,  False),
    "Coral TPU":     (4,   0.50, 2,   80,    0, False, True,  False),
    "Hailo-8":       (26,  0.50, 3,   200,   0, False, True,  True),
    "x86 CPU 8c":    (2.5, 0.60, 65,  400,   0, True,  True,  True),
}


def baholash(model, imgsz, fmt, batch, plat):
    gflop640, map640 = MODELLAR[model]
    tops, eff, vt, narx, nvdec, fp16, int8, sanoat = PLATFORMALAR[plat]
    gflop = gflop640 * (imgsz / 640) ** 2
    # sifat (mAP) proxysi: kichik kirishda pasayadi, INT8 kvantlash ~1 punkt yo'qotadi
    mAP = map640 - 12 * (1 - imgsz / 640) - (1.0 if fmt == "INT8" else 0)
    # format qo'llab-quvvatlanishi
    if (fmt == "FP16" and not fp16) or (fmt == "INT8" and not int8):
        return None
    tops_real = tops * eff * (0.5 if fmt == "FP16" else 1.0)   # FP16 odatda INT8 ning yarmi
    talab = gflop * INFERENS_S / 1e3
    zaxira = tops_real / talab
    # inferens kechikishi: paket hisoblash + paket to'lguncha kutish
    inferens_ms = batch * gflop / (tops_real * 1e3) * 1e3 * (1.3 if batch == 1 else 1.0)
    kutish_ms = (batch - 1) / INFERENS_S * 1e3
    kechikish = inferens_ms + kutish_ms
    dekod = nvdec * 3 >= KAMERALAR
    cpu_dekod_vt = 0 if dekod else 20            # apparat dekoder bo'lmasa CPU dekodlash qo'shimcha quvvati
    jami_vt = vt + cpu_dekod_vt
    cheklovlar = {
        "zaxira": zaxira >= MIN_ZAXIRA,
        "kechikish": kechikish <= INFERENS_BYUDJET_MS,
        "quvvat": jami_vt <= MAKS_VT,
        "narx": narx <= MAKS_NARX,
        "harorat": sanoat,
    }
    return dict(model=model, imgsz=imgsz, fmt=fmt, batch=batch, platforma=plat, mAP=round(mAP, 1),
                zaxira=round(zaxira, 2), kechikish_ms=round(kechikish, 1), Vt=jami_vt, narx=narx,
                yaroqli=all(cheklovlar.values()),
                buzilgan=",".join(k for k, v in cheklovlar.items() if not v) or "-")


qatorlar = []
for model, imgsz, fmt, batch, plat in itertools.product(MODELLAR, IMGSZ, FORMATLAR, BATCHLAR, PLATFORMALAR):
    r = baholash(model, imgsz, fmt, batch, plat)
    if r:
        qatorlar.append(r)
df = pd.DataFrame(qatorlar)

print("=== Co-design izlanish fazosi ===")
print(f"Jami konfiguratsiyalar: {len(df)}, yaroqli: {df.yaroqli.sum()}")
print("\nCheklov buzilishlari (necha konfiguratsiyada):")
for k in ("zaxira", "kechikish", "quvvat", "narx", "harorat"):
    print(f"  {k:10s}: {df.buzilgan.str.contains(k).sum()}")

yaroqli = df[df.yaroqli].sort_values(["mAP", "narx"], ascending=[False, True])
print("\n--- Eng yaxshi 10 ta yaroqli konfiguratsiya (mAP bo'yicha) ---")
print(yaroqli.head(10).drop(columns=["yaroqli", "buzilgan"]).to_markdown(index=False))

# Pareto chegarasi: narx kamroq va mAP yuqoriroq
pareto = []
for _, r in yaroqli.sort_values("narx").iterrows():
    if not pareto or r.mAP > pareto[-1].mAP:
        pareto.append(r)
print("\n--- Pareto chegarasi (narx vs mAP) ---")
for r in pareto:
    print(f"  {r.narx:6d} $  mAP {r.mAP:5.1f}  {r.platforma:14s} {r.model} @{r.imgsz} {r.fmt} batch {r.batch}  zaxira {r.zaxira}")

print("\n--- Har platforma uchun eng yaxshi yaroqli variant ---")
for plat, g in yaroqli.groupby("platforma"):
    r = g.iloc[0]
    print(f"  {plat:14s}: {r.model} @{r.imgsz} {r.fmt} batch {r.batch}  mAP {r.mAP}  zaxira {r.zaxira}  {r.kechikish_ms} ms")
yoq = set(PLATFORMALAR) - set(yaroqli.platforma)
print("  Yaroqli varianti yo'q:", ", ".join(sorted(yoq)))

fig, ax = plt.subplots(figsize=(8, 5))
for plat, g in df.groupby("platforma"):
    ok = g[g.yaroqli]
    yo = g[~g.yaroqli]
    ax.scatter(yo.narx, yo.mAP, marker="x", alpha=0.3, color="gray")
    ax.scatter(ok.narx, ok.mAP, label=plat, s=40)
ax.plot([r.narx for r in pareto], [r.mAP for r in pareto], "k--", lw=1, label="Pareto chegarasi")
ax.set_xscale("log")
ax.set_xlabel("Narx, $ (log)")
ax.set_ylabel("Sifat (mAP proxysi)")
ax.set_title("Co-design: yaroqli (rangli) va yaroqsiz (kulrang x) konfiguratsiyalar")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
