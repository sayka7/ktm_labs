"""MT3 — 3-qadam. FPGA: quvur (pipeline), II va fixed-point aniqlik simulyatsiyasi

Birinchi qismda 4 bosqichli MAC quvuri vaqt diagrammasi bilan simulyatsiya qilinadi: initiation interval (II) 1, 2, 4 bo'lganda o'tkazuvchanlik qanday tushishi ko'rsatiladi, Alveo U250 (~12288 DSP, 300 MHz) uchun INT16/INT8/FP32 rejimlarda TOP/s bahosi chiqariladi. Ikkinchi qismda digits ustida o'qitilgan MLP 64-32-10 butunlay butun sonli (fixed-point) arifmetikada bajariladi: Qm.n formatlari, to'yinish (saturation), siljitish orqali masshtab va akkumulyator kengligi. Har format uchun aniqlik, SQNR va to'yingan og'irliklar soni jadvalda.

Kutilgan natija: II=1 da 8 kirish 11 taktda; INT8 packing ~12 TOP/s (taxminiy); 16 bitda aniqlik suzuvchi nuqta bilan bir xil (~0,97), Q0.7 da to'yinish va keskin tushish, 12 bitli akkumulyatorda ~0,6.
"""
# 3-qadam. FPGA: quvur (pipeline) va fixed-point aniqlik simulyatsiyasi
# (a) Takt darajasidagi quvur: MAC yo'li 4 bosqichga bo'lingan (o'qish -> ko'paytirish -> qo'shish -> yozish).
#     II (initiation interval) = yangi kirish qabul qilish oralig'i. Throughput = f_clk * parallel / II.
# (b) Fixed-point: digits MLP og'irlik va faollashtirishlarini Qm.n formatga keltirib, to'yinishli (saturating)
#     butun sonli arifmetikada bajaramiz — FPGA'dagi DSP bloklari aynan shunday hisoblaydi.
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# ---------------- (a) Quvur simulyatori ----------------
def quvur(n_kirish, bosqich=4, II=1):
    """Har kirish II taktda bir qabul qilinadi; har bosqich 1 takt. Qaytaradi: jami taktlar, vaqt chizig'i."""
    boshlash = [i * II for i in range(n_kirish)]
    tugash = [b + bosqich for b in boshlash]
    return tugash[-1], boshlash


print("=== (a) Quvur: 4 bosqichli MAC, 8 ta kirish ===")
for II in (1, 2, 4):
    T, bosh = quvur(8, 4, II)
    print(f"II={II}: jami {T:3d} takt; o'tkazuvchanlik {8 / T:.2f} natija/takt")
T, bosh = quvur(6, 4, 1)
print("\nVaqt diagrammasi (II=1): O-o'qish, K-ko'paytirish, Q-qo'shish, Y-yozish")
for i, b in enumerate(bosh):
    print(f"kirish {i}: " + "." * b + "OKQY" + "." * (T - b - 4))
print("Quvursiz (har kirish 4 takt, ketma-ket): 6 kirish = 24 takt; quvur bilan = "
      f"{T} takt -> {24 / T:.1f}x tezroq")

# --- FPGA resurs/o'tkazuvchanlik modeli: Alveo U250 (~12288 DSP, taxminiy), 300 MHz ---
print("\n=== FPGA o'tkazuvchanlik modeli (taxminiy) ===")
DSP, F = 12288, 300e6
print(f"{'rejim':34s} {'MAC/DSP':>8s} {'band DSP':>9s} {'II':>3s} {'TOP/s':>7s}")
for nom, mac_dsp, ulush, II in [("INT16, 1 MAC/DSP", 1, 0.8, 1), ("INT8, 2 MAC/DSP (packing)", 2, 0.8, 1),
                                 ("INT8, II=2 (xotira cheklovi)", 2, 0.8, 2), ("FP32 (~3 DSP/MAC)", 1 / 3, 0.8, 1)]:
    tops = DSP * ulush * mac_dsp * 2 * F / II / 1e12
    print(f"{nom:34s} {mac_dsp:8.2f} {int(DSP * ulush):9d} {II:3d} {tops:7.2f}")
print("Ishlab chiqaruvchi INT8 cho'qqisi (33,3 TOP/s) yuqoriroq chastota va to'liq DSP foydalanishni nazarda tutadi.")

# ---------------- (b) Fixed-point aniqlik ----------------
X, y = load_digits(return_X_y=True); X = (X / 16.0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=0).fit(Xtr, ytr)
W1, W2 = mlp.coefs_; b1, b2 = mlp.intercepts_
acc_fp = (mlp.predict(Xte) == yte).mean()


def qkvant(x, butun, kasr, jami):
    """Ishorali fixed-point: jami bit, 'kasr' bit kasr qism. To'yinish (saturation) bilan butun songa."""
    q = np.round(x * (1 << kasr))
    lo, hi = -(1 << (jami - 1)), (1 << (jami - 1)) - 1
    return np.clip(q, lo, hi).astype(np.int64), int(((q < lo) | (q > hi)).sum())


def fixed_mlp(X, jami, kasr_w, kasr_a, akk_bit):
    """Butun sonli inferens: x(Q kasr_a) * w(Q kasr_w) -> akkumulyator (Q kasr_a+kasr_w), to'yinish akk_bit da."""
    xq, _ = qkvant(X, 0, kasr_a, jami)
    w1q, s1 = qkvant(W1, 0, kasr_w, jami); w2q, s2 = qkvant(W2, 0, kasr_w, jami)
    lo, hi = -(1 << (akk_bit - 1)), (1 << (akk_bit - 1)) - 1
    h = np.clip(xq @ w1q + np.round(b1 * (1 << (kasr_a + kasr_w))).astype(np.int64), lo, hi)
    h = np.maximum(h, 0) >> kasr_w                      # ReLU va masshtabni qaytarish (siljitish — FPGA'da bepul)
    h = np.clip(h, -(1 << (jami - 1)), (1 << (jami - 1)) - 1)
    o = np.clip(h @ w2q + np.round(b2 * (1 << (kasr_a + kasr_w))).astype(np.int64), lo, hi)
    hf = np.maximum(X @ W1 + b1, 0); of = hf @ W2 + b2   # suzuvchi nuqta etalon
    oq = o / float(1 << (kasr_a + kasr_w))
    sqnr = 10 * np.log10((of ** 2).sum() / max(((of - oq) ** 2).sum(), 1e-30))
    return (o.argmax(1) == yte).mean(), sqnr, s1 + s2


print(f"\n=== (b) Fixed-point MLP 64-32-10 (digits), suzuvchi nuqta aniqligi: {acc_fp:.4f} ===")
print(f"Og'irliklar diapazoni: [{min(W1.min(), W2.min()):.2f}, {max(W1.max(), W2.max()):.2f}]")
satr = []
for jami, kw, ka, akk in [(16, 12, 12, 48), (12, 8, 8, 32), (8, 6, 6, 24), (8, 4, 5, 24), (8, 7, 7, 24),
                          (6, 4, 4, 20), (4, 2, 2, 16), (8, 6, 6, 12)]:
    a, sq, sat = fixed_mlp(Xte, jami, kw, ka, akk)
    satr.append({"bit": jami, "Q_w": f"Q{jami - kw - 1}.{kw}", "Q_a": f"Q{jami - ka - 1}.{ka}", "akk_bit": akk,
                 "aniqlik": a, "farq_pp": 100 * (a - acc_fp), "SQNR_dB": sq, "to'yingan_w": sat})
print(pd.DataFrame(satr).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("Kuzating: kasr bit ko'paysa butun qism qisqaradi -> katta og'irliklar to'yinadi (Q0.7);")
print("akkumulyator torligi (12 bit) natijani buzadi — FPGA'da akkumulyator kengligini to'g'ri tanlash shart.")
print("\n=== FPGA simulyatsiyasi tugadi ===")
