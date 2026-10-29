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

# TODO 1: olch(fayl, intra, batch) — n_inf inferensda CPU va devor vaqtini o'lchang; bandlik, quvvat P,
#         J/inferens (to'liq va dinamik), kechikish, inf/s, inf/J va EDP ni lug'atda qaytaring
def olch(fayl, intra, batch, n_inf=12800):
    raise NotImplementedError("TODO 1: energiya o'lchovi")

# TODO 2: FP32/INT8 x intra 1/2 x batch 1/32 jadvalini tuzing, eng kam energiya va eng yaxshi EDP ni ayting
E = None
raise NotImplementedError("TODO 2: energiya jadvali")
print("=== Energiya bahosi (quvvat modeli TAXMINIY: P_bosh 6 W, P_max 28 W) ===")

# TODO 3: DVFS — chastotalar F (0,8..3,2 GHz), kuchlanish V chiziqli; P_din ~ f*V^2, kechikish ~ 1/f;
#         yuk 2000 so'rov/s da bandlik va 1 soniyalik energiyani hisoblang, SLO (0,25 ms) ichida eng tejamkor f ni toping
print("\n=== DVFS simulyatsiyasi ===")
# TODO 4: statik quvvat 1, 3, 10, 20 W bo'lganda optimal chastota qanday siljiydi (race-to-idle)?

# TODO 5: Jetson uslubidagi rejimlar (10/15/25 W, MAXN — faraziy) uchun inf/J ni hisoblang va 15000 inf/s talabiga eng tejamkorini tanlang
print("\n=== Jetson uslubidagi quvvat rejimlari (FARAZIY raqamlar) ===")
