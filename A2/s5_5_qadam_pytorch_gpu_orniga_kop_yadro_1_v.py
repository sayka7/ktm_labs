"""A2 — 5-qadam. PyTorch: GPU o'rniga ko'p yadro — 1 vs 2 ip, formatlar, yadro birlashtirish

torch.cuda.is_available() False ekani ko'rsatiladi va GPU'dagi o'zgarishlar izohlanadi. CPU'da: (a) matmul n=128/512/1024 uchun torch.set_num_threads(1) vs (2) tezlanishi; (b) FP32 vs BF16 vs INT8 (torch._int_mm) — CPU'da maxsus buyruqlar (AVX512-BF16/VNNI) bo'lmasa past formatlar sekinroq, bu apparat–format mosligi (Co-design) darsidir; (c) GELU eager (7 alohida amal) vs birlashtirilgan F.gelu — xotira trafigi 7× kam. Kutilgan natija: 2 ip katta n da ~2×, BF16 CPU'da 20× sekin, F.gelu ~3× tez.

Kutilgan natija: Iplar jadvali (1024 da 1,98× tezlanish), formatlar jadvali (FP32 ~50 GFLOP/s, BF16 ~2, INT8 ~4), GELU: eager ~130–220 ms, F.gelu ~45–80 ms.
"""
# 5-qadam. PyTorch: GPU o'rniga ko'p yadro — torch.set_num_threads(1) vs 2, formatlar va yadro birlashtirish
# Sandbox'da CUDA mavjud emas (torch.cuda.is_available() == False). Real labdagi Tensor Core o'lchovlari
# o'rniga CPU'da: (a) iplar soni ta'siri, (b) FP32 vs BF16 vs INT8 (CPU'da!), (c) GELU eager vs birlashtirilgan.
import time
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.backends.nnpack.set_flags(False)
print("=== torch iplar: 1 vs 2 ===")
print(f"torch {torch.__version__}, CUDA mavjud: {torch.cuda.is_available()}, CPU iplar (standart): {torch.get_num_threads()}")
print("GPU bo'lsa: dev='cuda', tensorlar .to(dev), vaqt oldidan torch.cuda.synchronize(); FP16/BF16 Tensor Core'ga o'tadi.\n")


def olchov(f, takror=7):
    for _ in range(2):
        f()
    v = []
    for _ in range(takror):
        t0 = time.perf_counter(); f(); v.append(time.perf_counter() - t0)
    return sorted(v)[len(v) // 2]


# --- (a) Matritsa ko'paytirish: iplar soni va o'lcham ---
print(f"{'n':>5s} {'iplar':>5s} {'ms':>9s} {'GFLOP/s':>8s} {'tezlanish':>9s}")
for n in (128, 512, 1024):
    A = torch.randn(n, n); B = torch.randn(n, n)
    t1 = None
    for iplar in (1, 2):
        torch.set_num_threads(iplar)
        t = olchov(lambda: A @ B)
        if t1 is None:
            t1 = t
        print(f"{n:5d} {iplar:5d} {t * 1e3:9.3f} {2 * n ** 3 / t / 1e9:8.1f} {t1 / t:9.2f}")
print("Kichik n da 2 ip foyda bermaydi (ip sinxronizatsiyasi xarajati), katta n da ~2x ga yaqinlashadi.")

# --- (b) Formatlar: FP32 vs BF16 vs INT8 — CPU'da tezlanish bormi? ---
torch.set_num_threads(2)
n = 512
A = torch.randn(n, n); B = torch.randn(n, n)
etalon = A @ B
print(f"\n--- Formatlar, n={n}, 2 ip (CPU) ---")
print(f"{'format':8s} {'ms':>9s} {'GFLOP/s':>8s} {'nisbiy xato':>12s}")
t32 = olchov(lambda: A @ B)
print(f"{'FP32':8s} {t32 * 1e3:9.3f} {2 * n ** 3 / t32 / 1e9:8.1f} {0.0:12.1e}")
Ab, Bb = A.bfloat16(), B.bfloat16()
tb = olchov(lambda: Ab @ Bb)
xb = ((Ab @ Bb).float() - etalon).norm() / etalon.norm()
print(f"{'BF16':8s} {tb * 1e3:9.3f} {2 * n ** 3 / tb / 1e9:8.1f} {xb.item():12.1e}")
# INT8: dinamik kvantlash (A3 mavzusi): masshtab * int8 matritsa
sa, sb = A.abs().max() / 127, B.abs().max() / 127
A8 = (A / sa).round().clamp(-128, 127).to(torch.int8)
B8 = (B / sb).round().clamp(-128, 127).to(torch.int8)
try:
    t8 = olchov(lambda: torch._int_mm(A8, B8))
    x8 = ((torch._int_mm(A8, B8).float() * sa * sb) - etalon).norm() / etalon.norm()
    print(f"{'INT8':8s} {t8 * 1e3:9.3f} {2 * n ** 3 / t8 / 1e9:8.1f} {x8.item():12.1e}  (torch._int_mm)")
except Exception as e:
    print(f"{'INT8':8s} CPU'da _int_mm qo'llab-quvvatlanmaydi: {type(e).__name__}")
print("CPU'da BF16/INT8 tezlanish bermasligi mumkin (AVX512-BF16/VNNI yo'q) — GPU Tensor Core'da 2-8x. Bu apparat-format moslik (Co-design) misoli.")

# --- (c) Elementlararo amal: GELU eager (6-7 alohida yadro) vs birlashtirilgan F.gelu ---
N = 8 * 1024 * 1024                                   # 8M element = 32 MB FP32
x = torch.randn(N)
def gelu_eager(x):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x ** 3)))
print(f"\n--- GELU, {N // 2 ** 20}M element ---")
print(f"{'variant':22s} {'iplar':>5s} {'ms':>8s} {'samarali GB/s':>14s}")
for iplar in (1, 2):
    torch.set_num_threads(iplar)
    for nom, f, otish in (("eager (7 yadro)", lambda: gelu_eager(x), 7 * 2), ("F.gelu (1 yadro)", lambda: F.gelu(x, approximate="tanh"), 2)):
        t = olchov(f, takror=5)
        print(f"{nom:22s} {iplar:5d} {t * 1e3:8.2f} {otish * N * 4 / t / 1e9:14.1f}")
print("Eager rejimda har amal 32 MB o'qib-yozadi; birlashtirilgan yadro bitta o'tishda — xotira trafigi 7x kam.")
print("GPU bo'lsa: torch.compile Triton bilan shunday birlashtirishni avtomatik qiladi (sandbox'da C kompilyatori yo'q).")
