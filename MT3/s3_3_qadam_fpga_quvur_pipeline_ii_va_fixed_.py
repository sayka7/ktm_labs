"""MT3 — 3-qadam. FPGA: quvur (pipeline), II va fixed-point aniqlik simulyatsiyasi

Birinchi qismda 4 bosqichli MAC quvuri vaqt diagrammasi bilan simulyatsiya qilinadi: initiation interval (II) 1, 2, 4 bo'lganda o'tkazuvchanlik qanday tushishi ko'rsatiladi, Alveo U250 (~12288 DSP, 300 MHz) uchun INT16/INT8/FP32 rejimlarda TOP/s bahosi chiqariladi. Ikkinchi qismda digits ustida o'qitilgan MLP 64-32-10 butunlay butun sonli (fixed-point) arifmetikada bajariladi: Qm.n formatlari, to'yinish (saturation), siljitish orqali masshtab va akkumulyator kengligi. Har format uchun aniqlik, SQNR va to'yingan og'irliklar soni jadvalda.

Kutilgan natija: II=1 da 8 kirish 11 taktda; INT8 packing ~12 TOP/s (taxminiy); 16 bitda aniqlik suzuvchi nuqta bilan bir xil (~0,97), Q0.7 da to'yinish va keskin tushish, 12 bitli akkumulyatorda ~0,6.
"""
# 3-qadam. FPGA: quvur (pipeline) va fixed-point aniqlik simulyatsiyasi
# (a) MAC yo'li 4 bosqichga bo'lingan (o'qish -> ko'paytirish -> qo'shish -> yozish). II — yangi kirish oralig'i.
# (b) digits MLP og'irlik va faollashtirishlarini Qm.n formatga keltirib, to'yinishli butun sonli arifmetikada bajaramiz.
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ---------------- (a) Quvur simulyatori ----------------
def quvur(n_kirish, bosqich=4, II=1):
    """Qaytaradi: jami taktlar, har kirishning boshlanish takti."""
    # TODO 1: har kirish qachon boshlanadi va oxirgisi qachon tugaydi?
    raise NotImplementedError("TODO 1: quvur vaqt modeli")


print("=== (a) Quvur: 4 bosqichli MAC, 8 ta kirish ===")
# TODO 2: II = 1, 2, 4 uchun jami taktlar va o'tkazuvchanlikni (natija/takt) chiqaring; II=1 vaqt diagrammasini chizing

print("\n=== FPGA o'tkazuvchanlik modeli (taxminiy) ===")
DSP, F = 12288, 300e6
REJIM = [("INT16, 1 MAC/DSP", 1, 0.8, 1), ("INT8, 2 MAC/DSP (packing)", 2, 0.8, 1),
         ("INT8, II=2 (xotira cheklovi)", 2, 0.8, 2), ("FP32 (~3 DSP/MAC)", 1 / 3, 0.8, 1)]
# TODO 3: har rejim uchun TOP/s = DSP * ulush * MAC/DSP * 2 op * F / II ni hisoblang

# ---------------- (b) Fixed-point aniqlik ----------------
X, y = load_digits(return_X_y=True); X = (X / 16.0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0).fit(Xtr, ytr)
W1, W2 = mlp.coefs_; b1, b2 = mlp.intercepts_
acc_fp = (mlp.predict(Xte) == yte).mean()


def qkvant(x, butun, kasr, jami):
    """Ishorali fixed-point: jami bit, 'kasr' bit kasr qism. Qaytaradi: butun sonlar, to'yinganlar soni."""
    # TODO 4: 2^kasr ga ko'paytirib yaxlitlang va [-2^(jami-1), 2^(jami-1)-1] ga to'yintiring
    raise NotImplementedError("TODO 4: Qm.n kvantlash")


def fixed_mlp(X, jami, kasr_w, kasr_a, akk_bit):
    """Butun sonli inferens. Qaytaradi: aniqlik, SQNR (dB), to'yingan og'irliklar soni."""
    # TODO 5: x, W1, W2 ni kvantlang; bias'ni Q(kasr_a+kasr_w) ga keltiring; akkumulyatorni akk_bit da to'yintiring
    # TODO 6: ReLU dan keyin kasr_w bitga o'ngga siljitib masshtabni qaytaring, ikkinchi qatlamni hisoblang
    # TODO 7: suzuvchi nuqta etalon bilan SQNR = 10*log10(signal / shovqin) ni toping
    raise NotImplementedError("TODO 5-7: fixed-point MLP")


print(f"\n=== (b) Fixed-point MLP 64-32-10 (digits), suzuvchi nuqta aniqligi: {acc_fp:.4f} ===")
for jami, kw, ka, akk in [(16, 12, 12, 48), (12, 8, 8, 32), (8, 6, 6, 24), (8, 4, 5, 24), (8, 7, 7, 24),
                          (6, 4, 4, 20), (4, 2, 2, 16), (8, 6, 6, 12)]:
    a, sq, sat = fixed_mlp(Xte, jami, kw, ka, akk)
    # TODO 8: natijalarni (bit, Q_w, Q_a, akk_bit, aniqlik, SQNR, to'yingan_w) jadvalga yig'ib chiqaring

print("\n=== FPGA simulyatsiyasi tugadi ===")
