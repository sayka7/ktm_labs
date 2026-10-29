"""MT2 — 2-qadam. ONNX eksport, grafik optimallashtirish va operatorlar profili

Model ONNX'ga (opset 17, dinamik batch) eksport qilinadi. ORT ning 4 ta optimallashtirish darajasi uchun optimallashtirilgan grafik faylga saqlanib, tugunlar va operatorlar ro'yxati (Conv+Relu → FusedConv) solishtiriladi — bu TensorRT'ning engine qurishdagi va OpenVINO'ning kompilyatsiyadagi qatlam birlashtirishiga o'xshaydi. So'ngra `enable_profiling` bilan har operatorning vaqt ulushi hisoblanadi (trtexec --dumpProfile analogi) va Amdahl qonuni bo'yicha optimallashtirish nuqtasi aniqlanadi.

Kutilgan natija: 14 tugun → 11 (FusedConv:3); ORT PyTorch eager'dan ~8 marta tez (~0,07 ms); profilda Conv/FusedConv vaqtning ~80% ini oladi (taxminiy).
"""
# 2-qadam. ONNX eksport, ORT grafik optimallashtirish (fusion) va operatorlar bo'yicha profil
import warnings; warnings.filterwarnings("ignore")
import os, json, time, glob, numpy as np, pandas as pd, torch, torch.nn as nn, onnx, onnxruntime as ort

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)
ort.set_default_logger_severity(3)                 # ORT ogohlantirishlarini yashiramiz

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

# Kechikish va profil vaznlarga bog'liq emas — bu qadamda model o'qitilmaydi (faqat arxitektura)

# Kechikish va profil vaznlarga bog'liq emas — bu qadamda model o'qitilmaydi (faqat arxitektura)
model = HARNet().eval(); x1 = torch.randn(1, 3, 128)

# TODO 1: modelni "har.onnx" ga eksport qiling (opset 17, kirish "x", chiqish "logits", dinamik batch o'qi) va tekshiring
raise NotImplementedError("TODO 1: ONNX eksport")

# TODO 2: operatorlar(mm) — grafikdagi tugunlarning op_type bo'yicha sonini lug'at qilib qaytaring
def operatorlar(mm):
    raise NotImplementedError("TODO 2: operatorlarni sanash")
print("=== ONNX eksport ===")

# TODO 3: 4 ta GraphOptimizationLevel uchun sessiya yarating (optimized_model_filepath bilan),
#         isinish + 500 chaqiruvda p50/p99 ni o'lchang, saqlangan grafikdagi tugunlar va operatorlarni yozing
print("\n=== Optimallashtirish darajalari (batch 1, 1 oqim) ===")

# TODO 4: PyTorch eager p50 bilan ORT ALL p50 ni solishtirib, necha marta tezligini chiqaring

# TODO 5: enable_profiling bilan batch 64 da 100 chaqiruv qiling, profil JSON'dan "_kernel_time" hodisalarini
#         op_name bo'yicha yig'ib, vaqt ulushini hisoblang va eng og'ir operatorni ayting
print("\n=== ORT profil (operatorlar bo'yicha, batch 64, 100 chaqiruv) ===")
