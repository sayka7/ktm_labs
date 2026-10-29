"""MT2 — 5-qadam. Parallellik: intra-op oqimlar, sessiyalar va LATENCY/THROUGHPUT rejimlari

Kengroq model (w=64) uchun 5 konfiguratsiya batch 1 va 32 da o'lchanadi: 1 sessiya/1 oqim, intra=2, umumiy sessiyaga 2 Python oqimi, 2 sessiya x 1 oqim va ortiqcha obuna (intra=2 x 2 oqim). Natija OpenVINO'ning NUM_STREAMS va PERFORMANCE_HINT (LATENCY/THROUGHPUT), TensorRT'ning bir nechta execution context + CUDA stream tushunchalari bilan jadvalda bog'lanadi.

Kutilgan natija: Eng past kechikish — intra=2 (1 oqim); eng yuqori throughput — 2 sessiya x 1 oqim (~2x); ortiqcha obunada p99 keskin oshadi (taxminiy).
"""
# 5-qadam. Parallellik: intra-op oqimlar, bir nechta so'rov oqimi (streams) va «latency» / «throughput» rejimlari
import warnings; warnings.filterwarnings("ignore")
import time, threading, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ort.set_default_logger_severity(3)
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

# Kengroq model (w=64): oqimlararo parallellik sezilishi uchun hisob ko'proq bo'lsin
torch.onnx.export(HARNet(w=64).eval(), torch.randn(1, 3, 128), "har64.onnx", input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)

# TODO 1: sessiya(intra, inter=1, parallel=False) — berilgan oqimlar soni bilan "har64.onnx" uchun ORT sessiyasi
def sessiya(intra, inter=1, parallel=False):
    raise NotImplementedError("TODO 1: SessionOptions bilan sessiya")

# TODO 2: yuklama(sessiyalar, ishchi, batch, jami) — ishchi ta Python oqimi jami // batch chaqiruvni bo'lishib bajaradi
#         (umumiy hisoblagich lock bilan); p50, p99 (ms) va inf/s ni qaytaring
def yuklama(sessiyalar, ishchi, batch, jami=1200):
    raise NotImplementedError("TODO 2: ko'p oqimli yuklama")

# TODO 3: A–E konfiguratsiyalar ro'yxati: (nom, sessiyalarni yaratuvchi funksiya, ishchi soni)
CFG = []
# TODO 4: batch 1 va 32 uchun barcha konfiguratsiyalarni o'lchang, nisbiy throughput qo'shing,
#         har batch uchun eng yuqori throughput va eng past p50 ni ayting
raise NotImplementedError("TODO 4: konfiguratsiyalarni o'lchash")
print("=== Parallellik konfiguratsiyalari (2 yadroli CPU) ===")

# TODO 5: ORT sozlamalarini OpenVINO va TensorRT/Triton analoglari bilan jadvalda bog'lang
print("\n=== OpenVINO / TensorRT dagi analoglari ===")
