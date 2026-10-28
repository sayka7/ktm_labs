"""A5 — 3-qadam. Halqa all-reduce ni NumPy'da qo'lda yozish va aloqa hajmi

N ta ishchining gradient vektorlari uchun halqa all-reduce (N-1 reduce-scatter + N-1 all-gather qadami) simulyatsiya qilinadi, natija oddiy yig'indi bilan solishtiriladi va har ishchi yuborgan baytlar `2(N-1)/N * S` formulasiga mos kelishi ko'rsatiladi. So'ng ResNet-50 gradientlari (102 MB) uchun NVLink, InfiniBand va 10 GbE tarmoqlarda aloqa vaqti hisoblanadi.

Kutilgan natija: N=2,4,8 uchun «natija to'g'ri=True» va yuborilgan baytlar formulaga aynan teng; 10 GbE da 8 GPU uchun ~143 ms, NVLink'da 0.3 ms.
"""
# 3-qadam. Halqa (ring) all-reduce ni NumPy'da qo'lda amalga oshirish va aloqa hajmini hisoblash
import numpy as np

def ring_allreduce(grads):
    """grads: ro'yxat, har bir ishchining gradient vektori (bir xil uzunlik).
    Qaytaradi: har ishchidagi natija (yig'indi) va har ishchi yuborgan baytlar soni."""
    N = len(grads); L = len(grads[0])
    chunks = np.array_split(np.arange(L), N)           # vektor N bo'lakka bo'linadi
    buf = [g.astype(np.float32).copy() for g in grads]  # har ishchining mahalliy buferi
    sent = [0] * N
    # 1-bosqich: reduce-scatter — N-1 qadam, har qadamda ishchi i bo'lak (i-k) ni o'ng qo'shnisiga yuboradi
    for k in range(N - 1):
        pkts = []
        for i in range(N):
            c = chunks[(i - k) % N]
            pkts.append((c, buf[i][c].copy())); sent[i] += len(c) * 4
        for i in range(N):
            c, data = pkts[(i - 1) % N]                 # chap qo'shnidan kelgan bo'lak
            buf[i][c] += data                           # qo'shib qo'yiladi (reduce)
    # 2-bosqich: all-gather — N-1 qadam, tayyor bo'lgan bo'lak halqa bo'ylab tarqatiladi
    for k in range(N - 1):
        pkts = []
        for i in range(N):
            c = chunks[(i + 1 - k) % N]
            pkts.append((c, buf[i][c].copy())); sent[i] += len(c) * 4
        for i in range(N):
            c, data = pkts[(i - 1) % N]
            buf[i][c] = data                            # almashtiriladi (gather)
    return buf, sent

print("=== Ring all-reduce tekshiruvi ===")
rng = np.random.default_rng(0)
for N in (2, 4, 8):
    L = 1000
    grads = [rng.standard_normal(L) for _ in range(N)]
    out, sent = ring_allreduce(grads)
    ref = np.sum(grads, axis=0)
    ok = all(np.allclose(o, ref, atol=1e-4) for o in out)
    S = L * 4
    theory = 2 * (N - 1) / N * S
    print(f"N={N:2d}: natija to'g'ri={ok}  har ishchi yubordi={sent[0]} bayt  formula 2(N-1)/N*S={theory:.0f} bayt  "
          f"(sodda 'hammaga yuborish' bo'lsa {(N-1)*S} bayt)")

print("\n=== Aloqa hajmi va vaqti: ResNet-50 (25.6 mln parametr, FP32 = 102 MB) ===")
S_bytes = 25.6e6 * 4
nets = {"NVLink (600 GB/s)": 600e9, "InfiniBand 100 Gbit/s": 100e9 / 8, "Ethernet 10 Gbit/s": 10e9 / 8}
print(f"{'N GPU':>6s} {'bayt/GPU, MB':>13s} " + " ".join(f"{k:>24s}" for k in nets))
for N in (2, 4, 8, 16, 64):
    per = 2 * (N - 1) / N * S_bytes
    print(f"{N:6d} {per / 1e6:13.1f} " + " ".join(f"{per / bw * 1e3:22.1f}ms" for bw in nets.values()))
print("\nXulosa: N oshganda har GPU yuboradigan hajm 2S ga intiladi (ishchilar sonidan deyarli bog'liq emas),")
print("lekin tarmoq sekin bo'lsa (10 GbE) qadam sayin ~160 ms aloqa — hisoblash vaqtiga teng yoki undan ko'p.")
print("BF16 gradient yuborilsa hajm 2 baravar kamayadi; gradient to'plash all-reduce chastotasini kamaytiradi.")
