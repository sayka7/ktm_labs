"""A3 — 2-qadam. ONNX Runtime: optimallashtirish darajalari, paket, oqimlar, Little qonuni

ONNX Runtime'ning 4 ta grafik optimallashtirish darajasi (DISABLE_ALL, BASIC, EXTENDED, ALL) batch 1 da o'lchanadi va optimallashtirilgan grafikdagi tugunlar soni sanaladi; keyin paket 1/16/64/256 uchun throughput va Little qonuni (N = X·R), intra_op oqimlar 1 va 2 taqqoslanadi (GPU parallelligi o'rniga). Oxirida 2000 so'rov/s talabi uchun nusxalar soni rejalashtiriladi.

Kutilgan natija: ORT ALL PyTorch eager'dan 6–9 marta tez; EXTENDED/ALL da tugunlar 10 → 7–8; batch 256 da throughput 10+ marta yuqori, latency ham oshadi; 2 oqim ~1,9x.
"""
# 2-qadam. ONNX Runtime: grafik optimallashtirish darajalari, paket o'lchami, oqimlar soni va Little qonuni
import warnings; warnings.filterwarnings("ignore")
import time, math, numpy as np, pandas as pd, torch, torch.nn as nn, onnx, onnxruntime as ort

torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

# Berilgan model (latency vaznlarga bog'liq emas — o'qitilmaydi)
class KichikCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.belgilar = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.tasnif = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.tasnif(self.belgilar(x))

model = KichikCNN().eval()
x1 = torch.randn(1, 1, 8, 8)
torch.onnx.export(model, x1, "cnn_fp32.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)

def olch(fn, takror=300, isinish=30):
    # TODO 1: fn() kechikish taqsimotini o'lchab p50/p95/p99/mean (ms) lug'atini qaytaring
    raise NotImplementedError("TODO 1: olch")

def ort_sess(opt_level, threads=2, saqla=None):
    # TODO 2: SessionOptions — graph_optimization_level, intra_op_num_threads, optimized_model_filepath
    raise NotImplementedError("TODO 2: ort_sess")

print("=== ONNX Runtime o'lchovlari ===")
L = ort.GraphOptimizationLevel
# TODO 3: PyTorch eager va 4 daraja (DISABLE_ALL, BASIC, EXTENDED, ALL) uchun batch 1 jadvali + tugunlar soni
# TODO 4: "--- Paket o'lchami ---": batch 1/16/64/256 uchun throughput va little_N = throughput x latency
# TODO 5: intra_op oqimlar 1 va 2 ni batch 256 da taqqoslang
# TODO 6: "Little qonuni" bo'yicha 2000 so'rov/s uchun kerakli nusxalar soni (math.ceil)
