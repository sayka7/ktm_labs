"""A3 — 3-qadam. PyTorch'da INT8 dinamik kvantlash (Linear)

MLP (Linear og'ir model) `torch.ao.quantization.quantize_dynamic` bilan INT8 ga o'tkaziladi: kvantlash formulasi q = round(w/s) qo'lda tekshiriladi, hajm (state_dict baytlari), aniqlik va batch 1/256 kechikishi FP32 bilan solishtiriladi. Sandbox CPU'da AVX yo'qligi sababli `qnnpack` dvigateli majburan tanlanadi — real serverda bu qator kerak emas.

Kutilgan natija: Hajm ~3,8 marta kichik (335 → 89 KB), aniqlik o'zgarmaydi (±0,3 p.p.), tezlik bu CPU'da 0,7–1,0x (VNNI yo'q).
"""
# 3-qadam. PyTorch'da dinamik INT8 kvantlash (Linear qatlamlar): hajm, aniqlik, tezlik
import warnings; warnings.filterwarnings("ignore")
import io, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)
# Sandbox CPU'da AVX/VNNI yo'q: fbgemm/onednn dvigatellari ishlamaydi, qnnpack (ARM/eski x86 uchun) ishlaydi.
# Haqiqiy serverda (AVX-512 VNNI) bu qator kerak emas — 'x86' dvigateli avtomatik tanlanadi.
torch.backends.quantized.engine = "qnnpack"
print("Kvantlash dvigateli:", torch.backends.quantized.engine, "| mavjud:", torch.backends.quantized.supported_engines)

X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
Xtr, Xte, ytr, yte = map(torch.tensor, (Xtr, Xte, ytr, yte))

# MLP — Linear (MatMul) og'ir model: dinamik kvantlash aynan shunday modellarga mo'ljallangan (BERT, LSTM kabi)
model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
opt = torch.optim.Adam(model.parameters(), 2e-3); lossf = nn.CrossEntropyLoss()
for ep in range(3):
    model.train(); perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i + 64]; opt.zero_grad(); lossf(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
model.eval()

def aniqlik(m):
    with torch.inference_mode():
        return (m(Xte).argmax(1) == yte).float().mean().item() * 100

def hajm_kb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 1024

def latency(m, x, takror=300):
    with torch.inference_mode():
        for _ in range(30): m(x)
        t = []
        for _ in range(takror):
            a = time.perf_counter(); m(x); t.append(time.perf_counter() - a)
    return np.percentile(np.array(t) * 1e3, 50)

print("\n=== PyTorch INT8 dinamik kvantlash ===")
qmodel = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)   # vaznlar INT8, faollashuvlar ish vaqtida kvantlanadi
print("Kvantlangan model tuzilmasi:")
for nomi, mod in qmodel.named_children():
    print(f"  {nomi}: {repr(mod)}")

# Kvantlash formulasini qo'lda tekshiramiz: q = round(w / s), s = max|w| / 127 (simmetrik, per-tensor)
w = model[0].weight.detach()
s = w.abs().max() / 127
q = torch.round(w / s).clamp(-128, 127)
print(f"\n1-qatlam vazni: diapazon [{w.min():.3f}, {w.max():.3f}]  masshtab s = {s:.5f}")
print(f"  INT8 -> FP32 qayta tiklash xatosi: max {((q * s) - w).abs().max():.5f}, o'rtacha {((q * s) - w).abs().mean():.6f}")
print(f"  Noyob qiymatlar: FP32 {w.unique().numel()} ta -> INT8 {q.unique().numel()} ta")

x1, xb = Xte[:1], Xte[:256]
rows = []
for nomi, m in (("FP32", model), ("INT8 dinamik", qmodel)):
    rows.append(dict(model=nomi, hajm_KB=round(hajm_kb(m), 1), aniqlik=round(aniqlik(m), 2),
                     p50_batch1_ms=round(latency(m, x1), 3), p50_batch256_ms=round(latency(m, xb, 100), 3)))
df = pd.DataFrame(rows); print("\n" + df.to_string(index=False))
print(f"\nHajm {df.hajm_KB[0] / df.hajm_KB[1]:.1f} marta kichraydi (nazariy chegara 4x; bias va masshtablar FP32 da qoladi).")
print(f"Aniqlik farqi: {df.aniqlik[1] - df.aniqlik[0]:+.2f} p.p.   Tezlik (batch 256): {df.p50_batch256_ms[0] / df.p50_batch256_ms[1]:.2f}x")
print("Izoh: bu CPU'da VNNI buyruqlari yo'q, shuning uchun INT8 tezlanishi kichik yoki hatto salbiy bo'lishi mumkin —")
print("  bu A3 matnidagi 5-xato ('VNNI yo'qligini tekshirmaslik') ning jonli namunasi. VNNI/AMX CPU'da 2-3x kutiladi.")
