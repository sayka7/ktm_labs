"""A3 — 1-qadam. Etalon: PyTorch FP32 o'qitish va ONNX eksport

Kichik CNN sklearn digits to'plamida 3 epoxa o'qitiladi, batch 1 kechikishi (p50/p95) o'lchanadi, so'ngra model `torch.onnx.export` bilan dinamik paket o'qi va opset 17 bilan eksport qilinadi. `onnx.checker` grafikni tekshiradi, ONNX Runtime chiqishi PyTorch bilan `np.allclose` orqali solishtiriladi. O'zgartirib ko'ring: epoxa sonini, `dynamic_axes` ni olib tashlab kirish shaklini, `do_constant_folding=False` da tugunlar sonini.

Kutilgan natija: 3 epoxadan keyin ~94 % aniqlik, PyTorch eager p50 ~0,15 ms; ONNX grafikda 10 tugun (BatchNorm Conv'ga birlashgan), maksimal farq ~1e-6, «MOS».
"""
# 1-qadam. Etalon: kichik CNN'ni PyTorch'da o'qitish, ONNX'ga eksport qilish va moslikni tekshirish
import warnings; warnings.filterwarnings("ignore")
import os, time, numpy as np, torch, torch.nn as nn, onnx, onnxruntime as ort
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2)                  # sandbox'da 2 yadro bor
torch.backends.nnpack.set_flags(False)    # eski CPU'da NNPACK ogohlantirishini o'chiramiz

# --- Ma'lumot: sklearn digits — 1797 ta 8x8 kulrang rasm, 10 sinf (ImageNet o'rniga, CPU uchun)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32).reshape(-1, 1, 8, 8)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr, Xte, ytr, yte))

# --- Model: ResNet-50 o'rniga kichik CNN (Conv-BN-ReLU-Pool x2 + 2 ta Linear)
class KichikCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.belgilar = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.tasnif = nn.Sequential(nn.Flatten(), nn.Linear(16 * 2 * 2, 32), nn.ReLU(), nn.Linear(32, 10))

    def forward(self, x):
        return self.tasnif(self.belgilar(x))

def oqit(model, epoxa=3, lr=3e-3):
    """Oddiy o'qitish sikli: Adam, paket 64, har epoxadan keyin test aniqligi."""
    opt = torch.optim.Adam(model.parameters(), lr); lossf = nn.CrossEntropyLoss()
    for ep in range(epoxa):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad(); loss = lossf(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        print(f"  epoxa {ep + 1}: loss {loss.item():.3f}   test aniqlik {acc * 100:.1f} %")
    return acc

print("=== PyTorch FP32 etalon ===")
model = KichikCNN()
t0 = time.perf_counter(); acc_fp32 = oqit(model)
print(f"O'qitish vaqti: {time.perf_counter() - t0:.1f} s")
print(f"Parametrlar soni: {sum(p.numel() for p in model.parameters())}")

# Latency: batch 1, 300 takror (avval 30 ta isinish)
x1 = Xte[:1]
with torch.inference_mode():
    for _ in range(30): model(x1)
    t = []
    for _ in range(300):
        a = time.perf_counter(); model(x1); t.append(time.perf_counter() - a)
t = np.array(t) * 1e3
print(f"PyTorch eager, batch 1: p50 {np.percentile(t, 50):.3f} ms   p95 {np.percentile(t, 95):.3f} ms")

print("\n=== ONNX eksport ===")
# dynamic_axes — paket o'lchami o'zgaruvchan (keyingi qadamlarda batch 1/16/64 bilan sinaymiz)
# dynamo=False — TorchScript asosidagi eksport (sandbox'da onnxscript o'rnatilmagan)
torch.onnx.export(model, x1, "cnn_fp32.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=17, do_constant_folding=True, dynamo=False)
m = onnx.load("cnn_fp32.onnx"); onnx.checker.check_model(m)
print("Opset:", m.opset_import[0].version, "  tugunlar soni:", len(m.graph.node))
print("Operatorlar:", [n.op_type for n in m.graph.node])
print(f"Fayl hajmi: {os.path.getsize('cnn_fp32.onnx') / 1024:.1f} KB")
print("Kirish shakli:", [d.dim_param or d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim])

# Moslik: PyTorch va ONNX Runtime chiqishlari bir xilmi?
sess = ort.InferenceSession("cnn_fp32.onnx", providers=["CPUExecutionProvider"])
with torch.inference_mode():
    ref = model(Xte[:64]).numpy()
out = sess.run(None, {"input": Xte[:64].numpy()})[0]
farq = np.abs(ref - out).max()
mos = np.allclose(ref, out, rtol=1e-3, atol=1e-4)
print(f"PyTorch vs ORT maksimal farq: {farq:.2e}  -> {'MOS' if mos else 'MOS EMAS'}")
acc_ort = (sess.run(None, {"input": Xte.numpy()})[0].argmax(1) == yte.numpy()).mean()
print(f"Test aniqlik: PyTorch {acc_fp32 * 100:.1f} %   ORT {acc_ort * 100:.1f} %")
print("\nIzoh: eval rejimida BatchNorm eksport paytida Conv ichiga birlashtirildi — ONNX grafikda BN tuguni yo'q.")
print("GPU bo'lsa: xuddi shu fayl TensorRT (trtexec --onnx=cnn_fp32.onnx) va ORT CUDAExecutionProvider'da ishlaydi.")
