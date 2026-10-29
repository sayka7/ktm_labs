"""A1 — 5-qadam. Kichik CNN uchun FLOP va parametrlarni torch bilan hisoblash

MiniYolo nomli kichik detektor tarmog'i uchun forward hook'lar bilan har qatlam FLOP'i formuladan (2·Kh·Kw·Cin·Cout·Hout·Wout) hisoblanadi va torch FlopCounterMode bilan 0 % farq tasdiqlanadi. So'ng kirish o'lchami (64/128/256) va batch (1/8) uchun haqiqiy CPU vaqti o'lchanib erishilgan GFLOP/s topiladi. Kutilgan natija: 128×128 da ~41,6 MFLOP, 61 K parametr; FLOP imgsz² ga proporsional; batch paketlash bir inferens vaqtini kamaytiradi.

Kutilgan natija: Qatlamlar jadvali (5 ta Conv2d), jami 41,58 MFLOP hook va FlopCounterMode bir xil; o'lchov jadvali 12–40 GFLOP/s oralig'ida.
"""
# 5-qadam. Kichik CNN uchun FLOP va parametrlarni hisoblash (torch) va CPU'da o'lchash
# Formula: Conv2d FLOP = 2*Kh*Kw*Cin/groups*Cout*Hout*Wout, Linear FLOP = 2*Nin*Nout.
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


flop_jadval = []

def conv_hook(mod, inp, out):
    # TODO 1: formuladan Conv2d FLOP ini hisoblab, (nom, chiqish shakli, flop) ni flop_jadval ga qo'shing
    raise NotImplementedError("TODO 1: conv_hook")


model = MiniYolo().eval()
# TODO 2: modeldagi har Conv2d ga conv_hook ni forward hook sifatida ulang
for m in model.modules():
    ...

IMGSZ = 128
x = torch.randn(1, 3, IMGSZ, IMGSZ)
with torch.no_grad():
    model(x)

print("=== CNN FLOP hisobi ===")
# TODO 3: qatlamlar jadvali, jami MFLOP (hook) va parametrlar sonini chiqaring
jami_hook = None

# TODO 4: FlopCounterMode bilan jami FLOP ni oling va hook natijasi bilan farqini (%) chiqaring
jami_torch = None
print(f"Jami (torch FlopCounterMode): {jami_torch / 1e6:.2f} MFLOP")

print("\n--- Kirish o'lchami va paket ta'siri (o'lchangan CPU vaqti, 2 ip) ---")
# TODO 5: imgsz (64,128,256) x batch (1,8) uchun isinish + 10 o'lchov mediani -> ms/inferens va GFLOP/s
