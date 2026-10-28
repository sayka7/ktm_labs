"""MT2 — 6-qadam. Energiya samaradorligi: J/inferens, EDP, DVFS va quvvat rejimlari

RAPL/nvidia-smi yo'qligi sababli quvvat modeli ishlatiladi: P = P_bosh + (P_max − P_bosh)·bandlik, bandlik = CPU vaqti / (devor vaqti·yadro). FP32 va INT8 statik, 1/2 oqim, batch 1/32 uchun µJ/inferens (to'liq va dinamik), inf/J va EDP (energiya x kechikish) hisoblanadi. So'ngra DVFS simulyatsiyasi: P_din ~ f·V², kechikish ~ 1/f — SLO ichida eng tejamkor chastota va «race-to-idle» sharti; oxirida Jetson uslubidagi nvpmodel rejimlari (faraziy raqamlar) bo'yicha tanlov.

Kutilgan natija: Eng kam energiya — INT8 statik, 1 oqim, batch 32 (~500 µJ/inf, taxminiy); 2 oqim batch 1 da energiya oshadi; DVFS optimumi ~1,2 GHz, statik quvvat katta bo'lsa optimum yuqori chastotaga siljiydi.
"""
# 6-qadam. Energiya samaradorligi: CPU-vaqt asosidagi quvvat modeli, J/inferens, EDP, DVFS va «race-to-idle» simulyatsiyasi
import warnings; warnings.filterwarnings("ignore")
import io, time, logging, contextlib, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

logging.disable(logging.WARNING); ort.set_default_logger_severity(3)
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# --- Sintetik akselerometr (HAR) ma'lumoti: 3 o'q x 128 nuqta (50 Hz, 2,56 s), 6 harakat sinfi
SINFLAR = ["tinch", "yurish", "yugurish", "zina_yuqori", "zina_past", "yiqilish"]
def har_yarat(n=3000, seed=0):
    r = np.random.default_rng(seed); t = np.arange(128) / 50.0
    X = np.zeros((n, 3, 128), np.float32); y = r.integers(0, 6, n)
    for j in range(n):
        c, f, ph = y[j], r.uniform(1.6, 2.4), r.uniform(0, 2 * np.pi)
        amp = {0: 0.05, 1: 0.6, 2: 1.0, 3: 0.65, 4: 0.65, 5: 0.3}[c] * r.uniform(0.7, 1.3)
        if c == 2: f *= 1.3
        s = amp * np.sin(2 * np.pi * f * t + ph)
        X[j, 0] = s; X[j, 1] = 0.5 * amp * np.sin(2 * np.pi * f * t + ph + 1.2); X[j, 2] = 1.0 + 0.6 * amp * np.cos(2 * np.pi * 2 * f * t + ph)
        if c == 3: X[j, 2] += np.linspace(0, 0.4, 128)          # yuqoriga — vertikal tezlanish trendi
        if c == 4: X[j, 2] -= np.linspace(0, 0.4, 128)
        if c == 5: k = r.integers(20, 100); X[j, :, k:k + 6] += r.normal(3.0, 0.8, (3, 1)); X[j, 2, k + 6:] -= 1.0   # zarba, keyin yotish
        X[j] += r.normal(0, 0.6, (3, 128))                      # sensor shovqini
    return X, y

# --- Model: depthwise-separable 1D CNN (MobileNet g'oyasi, chekka qurilmalar uchun)
class HARNet(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        def ds(ci, co):  # depthwise (har kanal alohida) + pointwise (1x1) konvolyutsiya
            return nn.Sequential(nn.Conv1d(ci, ci, 5, padding=2, groups=ci), nn.Conv1d(ci, co, 1), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
        self.f = nn.Sequential(nn.Conv1d(3, w, 5, padding=2), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2),
                               ds(w, 2 * w), ds(2 * w, 2 * w), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(2 * w, 6)
    def forward(self, x):
        return self.head(self.f(x))

def oqit(model, Xtr, ytr, epoxa=5, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr); lf = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(epoxa):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = perm[i:i + 64]; opt.zero_grad(); lf(model(Xt[b]), yt[b]).backward(); opt.step()
    return model.eval()

# Energiya vaznlarga bog'liq emas — o'qitmaymiz; INT8 uchun kalibrlash sintetik HAR namunalarida
X, _ = har_yarat(400)
torch.onnx.export(HARNet().eval(), torch.tensor(X[:1]), "fp32.onnx", input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
quant_pre_process("fp32.onnx", "prep.onnx")
class Kalib(CalibrationDataReader):
    def __init__(self): self.it = iter([{"x": X[i:i + 32]} for i in range(0, 320, 32)])
    def get_next(self): return next(self.it, None)
quantize_static("prep.onnx", "int8.onnx", Kalib(), quant_format=QuantFormat.QOperator, per_channel=True,
                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax)

# --- Quvvat modeli (RAPL/INA219 yo'q — TAXMINIY): P = P_bosh + (P_max - P_bosh) * bandlik, bandlik = CPU vaqti / (devor vaqti * yadro)
P_BOSH, P_MAX, YADRO = 6.0, 28.0, 2           # W — noutbuk/kichik server CPU paketi uchun faraz
def olch(fayl, intra, batch, n_inf=12800):
    so = ort.SessionOptions(); so.intra_op_num_threads = intra
    s = ort.InferenceSession(fayl, so, providers=["CPUExecutionProvider"])
    xb = np.random.randn(batch, 3, 128).astype(np.float32)
    for _ in range(20): s.run(None, {"x": xb})
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(n_inf // batch): s.run(None, {"x": xb})
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    band = min(cpu / (wall * YADRO), 1.0); P = P_BOSH + (P_MAX - P_BOSH) * band
    E = P * wall / n_inf                                               # J/inferens (bosh holat quvvati bilan)
    Ed = (P_MAX - P_BOSH) * cpu / YADRO / n_inf                        # faqat dinamik (marginal) qism
    lat = wall / (n_inf // batch) * 1e3
    return dict(bandlik=band, quvvat_W=P, lat_ms=lat, inf_s=n_inf / wall, uJ_inf=E * 1e6, uJ_dinamik=Ed * 1e6,
                inf_J=1 / E, EDP=E * 1e6 * lat)
rows = []
for nom, f in (("FP32", "fp32.onnx"), ("INT8 statik", "int8.onnx")):
    for intra in (1, 2):
        for batch in (1, 32):
            rows.append(dict(model=nom, intra=intra, batch=batch, **olch(f, intra, batch)))
E = pd.DataFrame(rows)
print("=== Energiya bahosi (quvvat modeli TAXMINIY: P_bosh 6 W, P_max 28 W) ===")
print(E.round(3).to_string(index=False))
eng = E.loc[E.uJ_inf.idxmin()]; edp = E.loc[E.EDP.idxmin()]
print(f"Eng kam energiya: {eng.model}, intra={eng.intra}, batch={eng.batch} -> {eng.uJ_inf:.1f} uJ/inf ({eng.inf_J:.0f} inf/J)")
print(f"Eng yaxshi EDP (energiya x kechikish): {edp.model}, intra={edp.intra}, batch={edp.batch}")
print("Izoh: 2 oqimda bandlik ~2x, tezlik esa < 2x bo'lsa — energiya/inferens oshadi; bosh holat quvvati tez ishlashni rag'batlantiradi.")

# --- DVFS: chastota f pasayganda P_dinamik ~ f*V^2, kechikish ~ 1/f (hisobga bog'liq yuk uchun faraz)
F = np.array([0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2])                   # GHz
V = 0.70 + (1.10 - 0.70) * (F - F[0]) / (F[-1] - F[0])            # kuchlanish, V (chiziqli faraz)
P_STAT, P_DYN_MAX = 3.0, 22.0                                      # W: statik (oqish) va maks. dinamik
Pd = P_DYN_MAX * (F / F[-1]) * (V / V[-1]) ** 2
base = E[(E.model == "FP32") & (E.intra == 1) & (E.batch == 1)].iloc[0]
lat_f = base.lat_ms * (F[-1] / F)                                  # o'lchangan kechikish maks. chastotada deb faraz
yuk = 2000                                                          # so'rov/s — davriy yuk
band_f = np.minimum(yuk * lat_f / 1000, 1.0)                       # bir yadro bandligi
P_ish = P_STAT + Pd                                                # ishlayotgandagi quvvat
P_bosh_holat = P_STAT + 0.05 * P_DYN_MAX                           # chuqur uyqu bo'lmagan bosh holat
E_sek = band_f * P_ish + (1 - band_f) * P_bosh_holat                # 1 soniyalik energiya, J
D = pd.DataFrame(dict(f_GHz=F, V=V, P_ish_W=P_ish, lat_ms=lat_f, bandlik=band_f, uJ_inf_faqat_ish=P_ish * lat_f * 1e3,
                      J_sekund=E_sek, uJ_inf_jami=E_sek / yuk * 1e6))
SLO = 0.25
D["SLO"] = np.where(D.lat_ms <= SLO, "ha", "yo'q")
print(f"\n=== DVFS simulyatsiyasi (yuk {yuk} so'rov/s, SLO kechikish <= {SLO} ms; parametrlar FARAZIY) ===")
print(D.round(3).to_string(index=False))
ok = D[D.SLO == "ha"]; best = ok.loc[ok.uJ_inf_jami.idxmin()]
print(f"SLO ichida eng tejamkor chastota: {best.f_GHz} GHz ({best.uJ_inf_jami:.1f} uJ/inf); maks. chastotada: {D.uJ_inf_jami.iloc[-1]:.1f} uJ/inf")
# Chuqur uyqu (C-state) mavjud bo'lsa bosh holat ~0,5 W; statik (oqish) quvvati faqat ish paytida — u katta bo'lsa tez ishlash foydali
for p_stat in (1.0, 3.0, 10.0, 20.0):
    Eb = band_f * (p_stat + Pd) + (1 - band_f) * 0.5
    print(f"  statik quvvat {p_stat:4.1f} W, uyqu 0,5 W -> eng tejamkor chastota {F[np.argmin(np.where(lat_f <= SLO, Eb, np.inf))]} GHz")
print("Race-to-idle: ish paytidagi statik quvvat katta va uyqu chuqur bo'lsa — tez ishlab «uxlash», aks holda past chastotada sekin ishlash foydali.")

# --- Jetson uslubidagi quvvat rejimlari (nvpmodel) — FARAZIY illyustratsiya, real raqamlar tegrastats bilan o'lchanadi
J = pd.DataFrame(dict(rejim=["10 W", "15 W", "25 W", "MAXN"], quvvat_W=[10.0, 15.0, 25.0, 40.0], nisbiy_tezlik=[0.45, 0.65, 0.85, 1.00]))
J["inf_s"] = 30000 * J.nisbiy_tezlik; J["inf_J"] = J.inf_s / J.quvvat_W
print("\n=== Jetson uslubidagi quvvat rejimlari (FARAZIY raqamlar) ===")
print(J.round(2).to_string(index=False))
kerak = 15000; mos = J[J.inf_s >= kerak]
print(f"Talab {kerak} inf/s: mos rejimlar {list(mos.rejim)}, eng tejamkori — {mos.loc[mos.inf_J.idxmax()].rejim}")
print("Real stendda: sudo nvpmodel -m <rejim>; tegrastats yoki tashqi quvvat o'lchagich bilan W ni yozib, inf/J hisoblanadi.")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
lab = E.model + " i" + E.intra.astype(str) + " b" + E.batch.astype(str)
ax[0].barh(lab, E.uJ_inf, color="tab:orange"); ax[0].set_xlabel("uJ/inferens (taxminiy)"); ax[0].set_title("Energiya/inferens")
ax[1].plot(D.f_GHz, D.uJ_inf_jami, marker="o"); ax[1].set_xlabel("Chastota, GHz"); ax[1].set_ylabel("uJ/inf (jami)"); ax[1].set_title("DVFS: optimal chastota"); plt.tight_layout()
