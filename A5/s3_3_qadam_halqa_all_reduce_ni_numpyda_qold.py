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
    chunks = np.array_split(np.arange(L), N)
    buf = [g.astype(np.float32).copy() for g in grads]
    sent = [0] * N
    # TODO 1: reduce-scatter — N-1 qadam: har ishchi bitta bo'lakni o'ng qo'shnisiga yuboradi, qabul qilgan bo'lakni o'zinikiga qo'shadi
    # TODO 2: all-gather — N-1 qadam: tayyor bo'laklar halqa bo'ylab tarqatiladi (qo'shilmaydi, almashtiriladi)
    raise NotImplementedError("TODO 1-2: ring all-reduce")
    return buf, sent

print("=== Ring all-reduce tekshiruvi ===")
rng = np.random.default_rng(0)
for N in (2, 4, 8):
    L = 1000
    grads = [rng.standard_normal(L) for _ in range(N)]
    out, sent = ring_allreduce(grads)
    # TODO 3: natijani np.sum bilan solishtiring va yuborilgan baytlarni 2(N-1)/N*S formulasi bilan taqqoslang
    #         (chop eting: "N=..: natija to'g'ri=..  har ishchi yubordi=.. bayt  formula ..")

print("\n=== Aloqa hajmi va vaqti: ResNet-50 (25.6 mln parametr, FP32 = 102 MB) ===")
nets = {"NVLink (600 GB/s)": 600e9, "InfiniBand 100 Gbit/s": 100e9 / 8, "Ethernet 10 Gbit/s": 10e9 / 8}
# TODO 4: N = 2, 4, 8, 16, 64 uchun har GPU yuboradigan MB va har tarmoqdagi aloqa vaqtini (ms) jadvalda chiqaring
