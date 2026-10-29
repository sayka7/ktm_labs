"""MT3 — 4-qadam. NPU uchun INT8 kvantlash: aniqlik, hajm, kechikish va grafik bo'linishi

Kichik CNN digits ustida o'qitilib, PyTorch eager statik kvantlash (qnnpack) bilan INT8 ga o'tkaziladi: conv+relu birlashtiriladi, per-tensor va per-channel og'irlik kvantlash, 8 va 256 namunali kalibrlash solishtiriladi. Aniqlik, model hajmi va batch 1/64 kechikishi o'lchanadi. So'ng har qatlam og'irliklarining simmetrik INT8 SQNR'i hisoblanadi. Oxirida NPU qo'llamaydigan operatorlar (SiLU, NMS) CPU'ga qaytganda PCIe orqali qatnov kechikishi modellashtiriladi va SiLU -> ReLU almashtirish samarasi ko'rsatiladi.

Kutilgan natija: FP32 va INT8 aniqligi deyarli teng (~0,95), hajm ~3,5x kichik; CPU'da batch=1 da INT8 tez bo'lmasligi mumkin, batch 64 da tezroq; SQNR ~40-47 dB; bo'linish modelida o'tishlar 5 -> 3, kechikish ~3,0 -> 2,4 ms.
"""
# 4-qadam. NPU uchun INT8 kvantlash: aniqlik, hajm, kechikish va operatorlarni bo'lish (partitioning)
# Kichik CNN'ni PyTorch statik kvantlash (qnnpack) bilan INT8 ga o'tkazamiz: per-tensor vs per-channel,
# kalibrlash hajmi ta'siri va NPU qo'llamaydigan operatorlar CPU'ga qaytganda (fallback) yo'qotilgan vaqt.
import warnings; warnings.filterwarnings("ignore")
import copy, io, time
import numpy as np
import torch, torch.nn as nn
import torch.ao.quantization as tq
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0); torch.set_num_threads(2)
torch.backends.quantized.engine = "qnnpack"      # AVX'siz CPU uchun
torch.backends.nnpack.set_flags(False)

X, y = load_digits(return_X_y=True); X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
Ttr, Tytr, Tte = torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(Xte)


class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.quant = tq.QuantStub(); s.dequant = tq.DeQuantStub()
        s.c1 = nn.Conv2d(1, 16, 3, padding=1); s.r1 = nn.ReLU()
        s.c2 = nn.Conv2d(16, 32, 3, padding=1); s.r2 = nn.ReLU(); s.p = nn.MaxPool2d(2)
        s.fc1 = nn.Linear(32 * 4 * 4, 64); s.r3 = nn.ReLU(); s.fc2 = nn.Linear(64, 10)

    def forward(s, x):
        x = s.quant(x)
        x = s.p(s.r2(s.c2(s.r1(s.c1(x)))))
        x = s.r3(s.fc1(x.reshape(x.shape[0], -1)))
        return s.dequant(s.fc2(x))


m = CNN(); opt = torch.optim.Adam(m.parameters(), 2e-3); lf = nn.CrossEntropyLoss()
for ep in range(6):
    perm = torch.randperm(len(Ttr))
    for i in range(0, len(Ttr), 64):
        idx = perm[i:i + 64]; opt.zero_grad(); lf(m(Ttr[idx]), Tytr[idx]).backward(); opt.step()
m.eval()

# TODO 1: aniqlik(model), hajm_kb(model) (state_dict -> BytesIO) va kechikish_ms(model, batch) (median, ms) yozing
def aniqlik(model): raise NotImplementedError("TODO 1")
def hajm_kb(model): raise NotImplementedError("TODO 1")
def kechikish_ms(model, batch=1, n=300): raise NotImplementedError("TODO 1")


def kvantla(model, qconfig, n_kalib):
    # TODO 2: nusxa oling, qconfig bering, conv+relu / linear+relu ni birlashtiring (fuse_modules)
    # TODO 3: prepare -> n_kalib namuna bilan kalibrlash -> convert
    raise NotImplementedError("TODO 2-3: statik kvantlash")


# TODO 4: per-tensor (MinMaxObserver) va per-channel (PerChannelMinMaxObserver) QConfig'larini tuzing
per_tensor = None
per_channel = None

print("=== INT8 statik kvantlash (qnnpack, CPU — NPU o'rniga) ===")
print(f"{'FP32':32s} {aniqlik(m):8.4f} {hajm_kb(m):8.1f} {kechikish_ms(m):8.3f} {kechikish_ms(m, 64, 100):8.3f}")
# TODO 5: 2 xil qconfig x kalibrlash (8, 256) uchun aniqlik, hajm, b=1 va b=64 kechikishini chiqaring

print("\n=== Qatlam og'irliklari: INT8 kvantlash xatosi (SQNR, dB) ===")
# TODO 6: har 2D+ parametr uchun simmetrik INT8 (masshtab = max|w|/127) va SQNR ni hisoblang

print("\n=== NPU grafik bo'linishi modeli (taxminiy, Hailo-8 kabi M.2 NPU) ===")
NPU_TOPS, PCIE_GBS, CPU_GOPS = 26 * 0.3, 3.0, 50   # NPU real foydalanish ~30 %, PCIe ~3 GB/s, xost CPU ~50 GOP/s
OPLAR = [  # (nom, amallar GOP, chiqish tenzori MB, NPU qo'llaydimi)
    ("conv_stem", 0.9, 1.6, True), ("conv_blok1", 2.1, 0.8, True), ("SiLU (qo'llanmaydi)", 0.01, 0.8, False),
    ("conv_blok2", 2.4, 0.4, True), ("NMS/postprocess", 0.05, 0.05, False), ("conv_head", 0.6, 0.2, True)]
# TODO 7: grafikni ketma-ket yurib, har CPU<->NPU o'tishda mb/PCIE_GBS qo'shing; SiLU -> ReLU variantini solishtiring

print("\n=== Kvantlash tahlili tugadi ===")
