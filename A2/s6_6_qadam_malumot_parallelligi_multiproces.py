"""A2 — 6-qadam. Ma'lumot parallelligi: multiprocessing bilan 2 jarayon

600 ta sintetik kadr uchun Python+NumPy oldindan ishlov (Sobel, filtr, gistogramma) ketma-ket va multiprocessing.Pool (fork, 1 va 2 jarayon, chunksize 1/25/150) bilan bajariladi; tezlanish, samaradorlik va Pool yaratish xarajati o'lchanadi, natija ketma-ket bilan taqqoslanib tekshiriladi. Kutilgan natija: 2 jarayonda ~1,9–2,4× (chunksize kattaroq bo'lsa yaxshiroq), Pool ochish ~10–50 ms. Bu DDP'dagi «har qurilma o'z ulushini ishlaydi» g'oyasining CPU muqobili.

Kutilgan natija: Ketma-ket ~0,6 s; jadval 6 qator (jarayonlar × chunksize), 2 jarayonda ~2× tezlanish; Pool xarajati xabari.
"""
# 6-qadam. Ma'lumot parallelligi: multiprocessing.Pool bilan kadrlarni 2 jarayonga bo'lish
# Python GIL tufayli iplar CPU-og'ir Python kodini tezlashtirmaydi; jarayonlar (fork) esa
# har biri o'z interpretatoriga ega. Bu ko'p-GPU DDP'dagi "har qurilma o'z ulushini ishlaydi" g'oyasining CPU muqobili.
import os
import time
import multiprocessing as mp
import numpy as np

KADRLAR = 600                  # 'video kadrlar' soni
H = W = 96


def kadr_ishlov(kadr):
    """Bitta kadr uchun oldindan ishlov: normalizatsiya, Sobel gradient, 3x3 o'rtacha filtr, gistogramma.
    Kichik NumPy massivlar ustida ko'p amal — BLAS iplari ishlamaydi, bu Python+NumPy yukidir."""
    x = kadr.astype(np.float32) / 255.0
    gx = np.zeros_like(x); gy = np.zeros_like(x)
    for _ in range(6):                                   # ishni og'irlashtirish (real quvurda bir nechta bosqich)
        gx[:, 1:-1] = x[:, 2:] - x[:, :-2]
        gy[1:-1, :] = x[2:, :] - x[:-2, :]
        mag = np.sqrt(gx * gx + gy * gy)
        x = (x[:-2, :-2] + x[1:-1, :-2] + x[2:, :-2] + x[:-2, 1:-1] + x[1:-1, 1:-1] + x[2:, 1:-1]
             + x[:-2, 2:] + x[1:-1, 2:] + x[2:, 2:]) / 9.0
        x = np.pad(x, 1)
    hist, _ = np.histogram(mag, bins=16, range=(0, 1))
    return float(mag.mean()), hist.argmax()


def ketma_ket(kadrlar):
    return [kadr_ishlov(k) for k in kadrlar]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    kadrlar = [rng.integers(0, 256, (H, W), dtype=np.uint8) for _ in range(KADRLAR)]

    print("=== Ma'lumot parallelligi (multiprocessing) ===")
    print(f"CPU yadrolar (ko'rinadigan): {os.cpu_count()}, ruxsat etilgan: {len(os.sched_getaffinity(0))}, kadrlar: {KADRLAR}")

    t0 = time.perf_counter(); etalon = ketma_ket(kadrlar); t_seq = time.perf_counter() - t0
    print(f"\nKetma-ket (1 jarayon): {t_seq * 1e3:7.0f} ms  ({KADRLAR / t_seq:.0f} kadr/s)")

    ctx = mp.get_context("fork")
    print(f"\n{'jarayonlar':>10s} {'chunksize':>9s} {'ms':>7s} {'kadr/s':>7s} {'tezlanish':>9s} {'samaradorlik':>12s}")
    for n_proc in (1, 2):
        for chunksize in (1, 25, 150):
            with ctx.Pool(n_proc) as pool:
                pool.map(kadr_ishlov, kadrlar[:20])          # isinish
                t0 = time.perf_counter()
                natija = pool.map(kadr_ishlov, kadrlar, chunksize=chunksize)
                t = time.perf_counter() - t0
            assert natija == etalon, "natija mos emas!"
            print(f"{n_proc:10d} {chunksize:9d} {t * 1e3:7.0f} {KADRLAR / t:7.0f} {t_seq / t:9.2f} {t_seq / t / n_proc * 100:11.0f} %")

    # Jarayon yaratish xarajati (DDP'da bu 'process group init' ga o'xshaydi)
    t0 = time.perf_counter()
    with ctx.Pool(2) as pool:
        pool.map(kadr_ishlov, kadrlar[:2])
    print(f"\nPool(2) yaratish + yopish xarajati: {(time.perf_counter() - t0) * 1e3:.0f} ms — qisqa vazifalar uchun parallelizm o'zini oqlamaydi.")
    print("chunksize=1 da har kadr alohida pickle/IPC orqali yuboriladi; katta bo'lak IPC xarajatini kamaytiradi.")
    print("\nGPU bo'lsa: ma'lumot parallelligi = DistributedDataParallel (har GPU o'z paketini ishlaydi, gradientlar all-reduce);")
    print("bu yerda jarayonlar natijani faqat oxirida jamlaydi (reduce), aloqa xarajati ancha kichik.")
