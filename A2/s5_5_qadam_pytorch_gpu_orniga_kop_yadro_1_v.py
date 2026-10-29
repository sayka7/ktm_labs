"""A2 — 5-qadam. PyTorch: GPU o'rniga ko'p yadro — 1 vs 2 ip, formatlar, yadro birlashtirish

torch.cuda.is_available() False ekani ko'rsatiladi va GPU'dagi o'zgarishlar izohlanadi. CPU'da: (a) matmul n=128/512/1024 uchun torch.set_num_threads(1) vs (2) tezlanishi; (b) FP32 vs BF16 vs INT8 (torch._int_mm) — CPU'da maxsus buyruqlar (AVX512-BF16/VNNI) bo'lmasa past formatlar sekinroq, bu apparat–format mosligi (Co-design) darsidir; (c) GELU eager (7 alohida amal) vs birlashtirilgan F.gelu — xotira trafigi 7× kam. Kutilgan natija: 2 ip katta n da ~2×, BF16 CPU'da 20× sekin, F.gelu ~3× tez.

Kutilgan natija: Iplar jadvali (1024 da 1,98× tezlanish), formatlar jadvali (FP32 ~50 GFLOP/s, BF16 ~2, INT8 ~4), GELU: eager ~130–220 ms, F.gelu ~45–80 ms.
"""
# 5-qadam. PyTorch: GPU o'rniga ko'p yadro — torch.set_num_threads(1) vs 2, formatlar va yadro birlashtirish
# Sandbox'da CUDA yo'q. CPU'da: (a) iplar soni, (b) FP32 vs BF16 vs INT8, (c) GELU eager vs birlashtirilgan.
import time
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.backends.nnpack.set_flags(False)
print("=== torch iplar: 1 vs 2 ===")
# TODO 1: torch versiyasi, torch.cuda.is_available() va standart iplar sonini chiqaring


def olchov(f, takror=7):
    # TODO 2: 2 marta isinish, so'ng takror marta o'lchab median qaytaring
    raise NotImplementedError("TODO 2: olchov")


# TODO 3: n = 128, 512, 1024 uchun A @ B ni 1 va 2 ipda o'lchang: ms, GFLOP/s, 1 ipga nisbatan tezlanish
print(f"{'n':>5s} {'iplar':>5s} {'ms':>9s} {'GFLOP/s':>8s} {'tezlanish':>9s}")
for n in (128, 512, 1024):
    A = torch.randn(n, n); B = torch.randn(n, n)
    olchov(lambda: A @ B)

# TODO 4: n=512, 2 ip: FP32, BF16 va INT8 (simmetrik masshtab + torch._int_mm) vaqti va nisbiy xatosi

# TODO 5: 8M elementli GELU: eager formulasi va F.gelu(approximate="tanh") ni 1 va 2 ipda, samarali GB/s bilan
N = 8 * 1024 * 1024
print(f"\n--- GELU, {N // 2 ** 20}M element ---")
