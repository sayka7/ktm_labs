"""MT3 — 4-qadam. NPU uchun INT8 kvantlash: aniqlik, hajm, kechikish va grafik bo'linishi

Kichik CNN digits ustida o'qitilib, PyTorch eager statik kvantlash (qnnpack) bilan INT8 ga o'tkaziladi: conv+relu birlashtiriladi, per-tensor va per-channel og'irlik kvantlash, 8 va 256 namunali kalibrlash solishtiriladi. Aniqlik, model hajmi va batch 1/64 kechikishi o'lchanadi. So'ng har qatlam og'irliklarining simmetrik INT8 SQNR'i hisoblanadi. Oxirida NPU qo'llamaydigan operatorlar (SiLU, NMS) CPU'ga qaytganda PCIe orqali qatnov kechikishi modellashtiriladi va SiLU -> ReLU almashtirish samarasi ko'rsatiladi.

Kutilgan natija: FP32 va INT8 aniqligi deyarli teng (~0,95), hajm ~3,5x kichik; CPU'da batch=1 da INT8 tez bo'lmasligi mumkin, batch 64 da tezroq; SQNR ~40-47 dB; bo'linish modelida o'tishlar 5 -> 3, kechikish ~3,0 -> 2,4 ms.
"""
# 4-qadam. NPU uchun INT8 kvantlash: aniqlik, hajm, kechikish va operatorlarni bo'lish (partitioning)
# NPU'lar (Hailo, Edge TPU, Ethos, DLA) asosan INT8 da ishlaydi. Biz kichik CNN'ni PyTorch statik kvantlash
# (qnnpack) bilan INT8 ga o'tkazamiz: per-tensor vs per-channel, kalibrlash hajmi ta'siri va NPU qo'llab-quvvatlamaydigan
# operatorlar CPU'ga qaytganda (fallback) qancha vaqt yo'qotilishini modellashtiramiz.
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


def aniqlik(model):
    with torch.no_grad():
        return (model(Tte).argmax(1).numpy() == yte).mean()


def hajm_kb(model):
    b = io.BytesIO(); torch.save(model.state_dict(), b); return len(b.getvalue()) / 1024


def kechikish_ms(model, batch=1, n=300):
    x = Tte[:batch]
    with torch.no_grad():
        for _ in range(20): model(x)
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); model(x); t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t))


def kvantla(model, qconfig, n_kalib):
    q = copy.deepcopy(model); q.qconfig = qconfig
    q = tq.fuse_modules(q, [["c1", "r1"], ["c2", "r2"], ["fc1", "r3"]])
    tq.prepare(q, inplace=True)
    with torch.no_grad():
        q(Ttr[:n_kalib])                       # kalibrlash: faollashtirish diapazonlari yig'iladi
    tq.convert(q, inplace=True)
    return q


per_tensor = tq.QConfig(activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
                        weight=tq.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
per_channel = tq.QConfig(activation=tq.MinMaxObserver.with_args(dtype=torch.quint8),
                         weight=tq.PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric))

print("=== INT8 statik kvantlash (qnnpack, CPU — NPU o'rniga) ===")
print(f"{'model':32s} {'aniqlik':>8s} {'hajm KB':>8s} {'b=1 ms':>8s} {'b=64 ms':>8s}")
print(f"{'FP32':32s} {aniqlik(m):8.4f} {hajm_kb(m):8.1f} {kechikish_ms(m):8.3f} {kechikish_ms(m, 64, 100):8.3f}")
for nom, qc in (("INT8 per-tensor", per_tensor), ("INT8 per-channel", per_channel)):
    for nk in (8, 256):
        q = kvantla(m, qc, nk)
        print(f"{nom + f', kalib={nk}':32s} {aniqlik(q):8.4f} {hajm_kb(q):8.1f} {kechikish_ms(q):8.3f} {kechikish_ms(q, 64, 100):8.3f}")
print("Hajm ~4x kamayadi. CPU'da INT8 har doim tezroq emas (qnnpack, kichik model) — NPU'da esa INT8 yagona tez yo'l.")

# --- Kvantlash xatosi qatlam bo'yicha (simmetrik per-tensor, NPU uslubi) ---
print("\n=== Qatlam og'irliklari: INT8 kvantlash xatosi (SQNR, dB) ===")
for nom, p in m.named_parameters():
    if p.dim() < 2: continue
    w = p.detach().numpy(); s = np.abs(w).max() / 127
    wq = np.clip(np.round(w / s), -127, 127) * s
    sqnr = 10 * np.log10((w ** 2).sum() / ((w - wq) ** 2).sum())
    print(f"{nom:12s} shakl={str(tuple(w.shape)):18s} masshtab={s:.5f}  SQNR={sqnr:5.1f} dB")

# --- Operatorlarni bo'lish: NPU qo'llamaydigan op CPU'da bajariladi, tenzor PCIe orqali qatnaydi ---
print("\n=== NPU grafik bo'linishi modeli (taxminiy, Hailo-8 kabi M.2 NPU) ===")
NPU_TOPS, PCIE_GBS, CPU_GOPS = 26 * 0.3, 3.0, 50   # NPU real foydalanish ~30 %, PCIe ~3 GB/s, xost CPU ~50 GOP/s
OPLAR = [  # (nom, amallar GOP, chiqish tenzori MB, NPU qo'llaydimi)
    ("conv_stem", 0.9, 1.6, True), ("conv_blok1", 2.1, 0.8, True), ("SiLU (qo'llanmaydi)", 0.01, 0.8, False),
    ("conv_blok2", 2.4, 0.4, True), ("NMS/postprocess", 0.05, 0.05, False), ("conv_head", 0.6, 0.2, True)]


def ms(gop, npu):  # amal vaqti, ms
    return gop / (NPU_TOPS * 1e3) * 1e3 if npu else gop / CPU_GOPS * 1e3


RELU = [("ReLU", g, mb, True) if "SiLU" in n else (n, g, mb, npu) for n, g, mb, npu in OPLAR]   # SiLU -> ReLU (NPU qo'llaydi)
for nom_v, oplar in (("asl grafik (SiLU + NMS CPU'da)", OPLAR), ("SiLU -> ReLU almashtirilgan", RELU)):
    jami = 0.0; joy = "CPU"; utish = 0           # kirish tasviri xost (CPU) xotirasida
    for nom, gop, mb, npu in oplar:
        yangi = "NPU" if npu else "CPU"
        if yangi != joy:
            jami += mb / PCIE_GBS; utish += 1
        jami += ms(gop, npu); joy = yangi
    print(f"{nom_v:34s} kechikish ~{jami:6.2f} ms, CPU<->NPU o'tishlar: {utish}")
print("Xulosa: bitta qo'llanmaydigan aktivatsiya grafikni bo'lib, PCIe orqali qo'shimcha nusxalashga olib keladi.")
print("\n=== Kvantlash tahlili tugadi ===")
