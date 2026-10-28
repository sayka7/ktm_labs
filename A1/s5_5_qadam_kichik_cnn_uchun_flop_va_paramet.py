"""A1 — 5-qadam. Kichik CNN uchun FLOP va parametrlarni torch bilan hisoblash

MiniYolo nomli kichik detektor tarmog'i uchun forward hook'lar bilan har qatlam FLOP'i formuladan (2·Kh·Kw·Cin·Cout·Hout·Wout) hisoblanadi va torch FlopCounterMode bilan 0 % farq tasdiqlanadi. So'ng kirish o'lchami (64/128/256) va batch (1/8) uchun haqiqiy CPU vaqti o'lchanib erishilgan GFLOP/s topiladi. Kutilgan natija: 128×128 da ~41,6 MFLOP, 61 K parametr; FLOP imgsz² ga proporsional; batch paketlash bir inferens vaqtini kamaytiradi.

Kutilgan natija: Qatlamlar jadvali (5 ta Conv2d), jami 41,58 MFLOP hook va FlopCounterMode bir xil; o'lchov jadvali 12–40 GFLOP/s oralig'ida.
"""
# 5-qadam. Kichik CNN uchun FLOP va parametrlarni hisoblash (torch) va CPU'da o'lchash
# Formula: Conv2d FLOP = 2*Kh*Kw*Cin/groups*Cout*Hout*Wout, Linear FLOP = 2*Nin*Nout.
# Qo'lda yozilgan hook'lar bilan hisoblab, torch'ning FlopCounterMode natijasi bilan solishtiramiz,
# so'ng haqiqiy CPU vaqtini o'lchab erishilgan GFLOP/s ni topamiz.
import time
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

torch.manual_seed(0)
torch.set_num_threads(2)
torch.backends.nnpack.set_flags(False)   # sandbox CPU'da NNPACK yo'q — ogohlantirishlarni o'chiramiz


class MiniYolo(nn.Module):
    """Juda kichik detektor-uslubidagi tarmoq: 4 ta conv + bosh (head)."""
    def __init__(self, cin=3, sinflar=4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(cin, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.head = nn.Conv2d(64, 5 + sinflar, 1)      # har hujayra: 4 koordinata + obyektlik + sinflar

    def forward(self, x):
        return self.head(self.backbone(x))


# --- Hook'lar bilan FLOP hisoblash ---
flop_jadval = []

def conv_hook(mod, inp, out):
    kh, kw = mod.kernel_size
    cin = mod.in_channels // mod.groups
    hout, wout = out.shape[-2:]
    flop = 2 * kh * kw * cin * mod.out_channels * hout * wout * out.shape[0]
    flop_jadval.append((mod.__class__.__name__, tuple(out.shape[1:]), flop))

def linear_hook(mod, inp, out):
    flop = 2 * mod.in_features * mod.out_features * out.shape[0]
    flop_jadval.append(("Linear", tuple(out.shape[1:]), flop))

model = MiniYolo().eval()
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        m.register_forward_hook(conv_hook)
    elif isinstance(m, nn.Linear):
        m.register_forward_hook(linear_hook)

IMGSZ = 128
x = torch.randn(1, 3, IMGSZ, IMGSZ)
with torch.no_grad():
    model(x)

print("=== CNN FLOP hisobi ===")
print(f"{'Qatlam':8s} {'chiqish shakli':>18s} {'MFLOP':>9s}")
for nom, shakl, flop in flop_jadval:
    print(f"{nom:8s} {str(shakl):>18s} {flop / 1e6:9.2f}")
jami_hook = sum(f for _, _, f in flop_jadval)
params = sum(p.numel() for p in model.parameters())
print(f"Jami (hook): {jami_hook / 1e6:.2f} MFLOP, parametrlar: {params / 1e3:.1f} K, vazn FP32 {params * 4 / 1e6:.2f} MB")

# --- torch FlopCounterMode bilan tekshirish ---
flop_jadval.clear()
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    model(x)
jami_torch = fc.get_total_flops()
print(f"Jami (torch FlopCounterMode): {jami_torch / 1e6:.2f} MFLOP  -> farq {abs(jami_torch - jami_hook) / jami_hook * 100:.2f} %")

# --- FLOP kirish o'lchamiga qanday bog'liq ---
print("\n--- Kirish o'lchami va paket ta'siri (o'lchangan CPU vaqti, 2 ip) ---")
print(f"{'imgsz':>6s} {'batch':>5s} {'MFLOP':>9s} {'ms/inferens':>12s} {'GFLOP/s':>8s}")
for imgsz in (64, 128, 256):
    for batch in (1, 8):
        xx = torch.randn(batch, 3, imgsz, imgsz)
        flop_jadval.clear()
        with torch.no_grad():
            model(xx)
        flop = sum(f for _, _, f in flop_jadval)
        with torch.no_grad():
            for _ in range(3):                 # isinish
                model(xx)
            vaqtlar = []
            for _ in range(10):
                t0 = time.perf_counter()
                model(xx)
                vaqtlar.append(time.perf_counter() - t0)
        t = sorted(vaqtlar)[len(vaqtlar) // 2]
        print(f"{imgsz:6d} {batch:5d} {flop / 1e6:9.1f} {t / batch * 1e3:12.3f} {flop / t / 1e9:8.2f}")

print("\nXulosa: FLOP kirish maydoniga (imgsz^2) proporsional; paketlash bir inferens vaqtini kamaytiradi,")
print("chunki vaznlar keshda qoladi va BLAS yadrolari kattaroq matritsalarda samaraliroq ishlaydi.")
