"""MT2 — 4-qadam. Paket va oqim tahlili: S(b) modeli va dinamik paketlash simulyatsiyasi

Batch 1–128 uchun xizmat vaqti o'lchanadi va chiziqli model S(b) = c0 + c1·b moslanadi (c0 — har chaqiruv ustamasi, c1 — namuna narxi). So'ngra Puasson oqimi bilan navbat simulyatsiyasi: paketsiz, dinamik paketlash (b<=16, kutish 0,3 ms) va (b<=64, kutish 1 ms) siyosatlari turli yuklamada p50/p99 va throughput bo'yicha solishtiriladi (Triton dynamic_batching, OpenVINO avtomatik paketlash g'oyasi). Xuddi shu simulyatsiya GPU uslubidagi S(b) (katta c0, kichik c1 — faraziy) uchun takrorlanadi.

Kutilgan natija: Bu CPU'da c0 kichik (~0,02 ms), throughput batch bilan ~15k → ~18,5k inf/s; GPU uslubida dinamik paketlash SLO ichidagi yuklamani ~1000 dan ~60 000 so'rov/s gacha oshiradi (taxminiy).
"""
# 4-qadam. Paket (batch) va oqim tahlili: xizmat vaqti modeli S(b), dinamik paketlash simulyatsiyasi, SLO bo'yicha maksimal yuklama
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ort.set_default_logger_severity(3)
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# --- Sintetik akselerometr (HAR) ma'lumoti: 3 o'q x 128 nuqta (50 Hz, 2,56 s), 6 harakat sinfi
SINFLAR = ["tinch", "yurish", "yugurish", "zina_yuqori", "zina_past", "yiqilish"]
def har_yarat(n=3000, seed=0):
    r = np.random.default_rng(seed); t = np.arange(128) / 50.0
    X = np.zeros((n, 3, 128), np.float32); y = r.integers(0, 6, n)
    for j in range(n):
        c, f, ph = y[j], r.uniform(1.6, 2.4), r.uniform(0, 2 * np.pi)
        amp = {0: 0.05, 1: 0.6, 2: 1.0, 3: 0.65, 4: 0.65, 5: 0.3}[c] * r.uniform(0.7, 1.3)
        if c == 2: f *= 1.3
        s = amp * np.sin(2 * np.pi * f * t + ph)
        X[j, 0] = s; X[j, 1] = 0.5 * amp * np.sin(2 * np.pi * f * t + ph + 1.2); X[j, 2] = 1.0 + 0.6 * amp * np.cos(2 * np.pi * 2 * f * t + ph)
        if c == 3: X[j, 2] += np.linspace(0, 0.4, 128)          # yuqoriga — vertikal tezlanish trendi
        if c == 4: X[j, 2] -= np.linspace(0, 0.4, 128)
        if c == 5: k = r.integers(20, 100); X[j, :, k:k + 6] += r.normal(3.0, 0.8, (3, 1)); X[j, 2, k + 6:] -= 1.0   # zarba, keyin yotish
        X[j] += r.normal(0, 0.6, (3, 128))                      # sensor shovqini
    return X, y

# --- Model: depthwise-separable 1D CNN (MobileNet g'oyasi, chekka qurilmalar uchun)
class HARNet(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        def ds(ci, co):  # depthwise (har kanal alohida) + pointwise (1x1) konvolyutsiya
            return nn.Sequential(nn.Conv1d(ci, ci, 5, padding=2, groups=ci), nn.Conv1d(ci, co, 1), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
        self.f = nn.Sequential(nn.Conv1d(3, w, 5, padding=2), nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2),
                               ds(w, 2 * w), ds(2 * w, 2 * w), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Linear(2 * w, 6)
    def forward(self, x):
        return self.head(self.f(x))

model = HARNet().eval()
torch.onnx.export(model, torch.randn(1, 3, 128), "har.onnx", input_names=["x"], output_names=["logits"],
                  dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
so = ort.SessionOptions(); so.intra_op_num_threads = 1
s = ort.InferenceSession("har.onnx", so, providers=["CPUExecutionProvider"])

# TODO 1: batch (1..128) bo'yicha xizmat vaqtini o'lchang: isinish, takrorlar, S_p50, S_p99 va inf/s
rows = []
S = pd.DataFrame(rows, columns=["batch", "S_p50, ms", "S_p99, ms", "inf/s"])
# TODO 2: S(b) = c0 + c1*b chiziqli modelini moslang
c0, c1 = None, None
raise NotImplementedError("TODO 1–2: S(b) o'lchovi va chiziqli moslash")
print("=== Paket o'lchami: xizmat vaqti va throughput ===")

# TODO 3: simul(lam, max_b, max_wait_ms, c0, c1) — Puasson oqimi, bitta ishchi, dinamik paketlash;
#         p50, p99 kechikish, o'rtacha paket va throughput'ni qaytaring
def simul(lam, max_b, max_wait_ms, c0, c1, n=6000, seed=0):
    raise NotImplementedError("TODO 3: navbat simulyatsiyasi")

SLO = 2.0                                                             # p99 <= 2 ms (taxminiy talab)
SIYOSAT = [("paketsiz", 1, 0.0), ("dinamik b<=16, kutish 0,3 ms", 16, 0.3), ("dinamik b<=64, kutish 1 ms", 64, 1.0)]
# TODO 4: jadval(c0, c1, yuklamalar) — har yuklama x siyosat uchun simul natijasi va SLO bajarilishi,
#         har siyosat uchun SLO ni saqlagan maksimal yuklamani chiqaring
def jadval(c0_, c1_, yuklamalar):
    raise NotImplementedError("TODO 4: siyosatlar jadvali")
print(f"\n=== Dinamik paketlash simulyatsiyasi: bu CPU (o'lchangan S(b)), SLO p99 <= {SLO} ms ===")
# TODO 5: xuddi shu jadvalni GPU uslubidagi S(b) = 0.40 + 0.004*b ms uchun takrorlang
