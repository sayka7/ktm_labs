"""A2 — 7-qadam. Amdahl va Gustafson qonunlari: nazariya va 2 yadroda o'lchov

S(n)=1/((1−p)+p/n) jadvali p=0,5…0,99 uchun chiqariladi; so'ng «dastur» = Python ketma-ket qism + torch matmul 1 va 2 ipda o'lchanib, S(2) dan parallel ulush p baholanadi va shu p bilan 8 hamda 1024 yadrodagi (GPU) tezlanish bashorat qilinadi. Grafikda Amdahl egri chiziqlari, o'lchangan nuqta va Gustafson chiziqlari ko'rsatiladi. Kutilgan natija: faqat matmul p≈1, ketma-ket qism qo'shilganda p≈0,97 → 1024 yadroda tezlanish atigi ~30×.

Kutilgan natija: Amdahl jadvali (p=0,95: n=inf da 20), o'lchov jadvali S(2) ~1,9–2,0, p baholari, 2 panelli grafik.
"""
# 7-qadam. Amdahl va Gustafson qonunlari: nazariy egri chiziqlar + 2 yadroda o'lchangan parallel ulush
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)


def amdahl(p, n):
    # TODO 1: Amdahl formulasi bo'yicha tezlanish
    raise NotImplementedError("TODO 1: amdahl")


def gustafson(p, n):
    # TODO 2: Gustafson (masshtablangan) tezlanish
    raise NotImplementedError("TODO 2: gustafson")


print("=== Amdahl qonuni ===")
# TODO 3: p = 0.5..0.99 va n = 2, 4, 8, 64, 1024, inf jadvalini chiqaring
for p in (0.5, 0.8, 0.9, 0.95, 0.99):
    print(p, amdahl(p, 2))


def olchov(f, takror=5):
    f(); f()
    v = []
    for _ in range(takror):
        t0 = time.perf_counter(); f(); v.append(time.perf_counter() - t0)
    return sorted(v)[len(v) // 2]

n = 768
A = torch.randn(n, n); B = torch.randn(n, n)

def ketma_ket_qism():
    s = 0.0
    for i in range(60000):
        s += (i % 7) * 0.5
    return s

def dastur(ketma_ket=True):
    if ketma_ket:
        ketma_ket_qism()
    for _ in range(4):
        C = A @ B
    return C

# TODO 4: "faqat matmul" va "ketma-ket qism + matmul" uchun 1 va 2 ipda o'lchab S(2) ni toping
# TODO 5: S(2) dan p ni teskari hisoblang (0..1 ga qising) va S(8), S(1024) ni bashorat qiling; "p (baho)" ustunli jadval
# TODO 6: 2 panelli grafik: Amdahl egri chiziqlari + o'lchangan nuqta; Gustafson vs Amdahl (log2 o'qlar)
