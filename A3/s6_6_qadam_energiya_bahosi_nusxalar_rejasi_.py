"""A3 — 6-qadam. Energiya bahosi, nusxalar rejasi va GPU/TensorRT yo'l xaritasi

RAPL/nvidia-smi yo'qligi sababli quvvat MODEL bilan baholanadi: P = P_bosh + (P_to'liq − P_bosh)·CPU_bandlik, bandlik `time.process_time()/devor vaqti` dan olinadi. FP32 va INT8 dinamik, 1 va 2 oqim, batch 1 va 64 uchun µJ/inferens (to'liq va dinamik) hamda inferens/s/W hisoblanadi; Little qonuni bilan 5000 so'rov/s uchun nusxalar va quvvat; oxirida GPU'da TensorRT FP16/INT8 va quvvat o'lchash qadamlari (buyruqlar va kutilgan raqamlar).

Kutilgan natija: Batch 64 da energiya batch 1 dan 5–15 marta kam; 2 oqim batch 1 da energiya ortadi (bandlik 2x, tezlik oshmaydi); ustunli grafik.
"""
# 6-qadam. Energiya bahosi (CPU vaqti asosida), nusxalar rejasi va GPU/TensorRT yo'l xaritasi
import warnings; warnings.filterwarnings("ignore")
import time, math, logging, numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

logging.disable(logging.WARNING)
torch.manual_seed(0); torch.set_num_threads(2); torch.backends.nnpack.set_flags(False)

class KichikCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.belgilar = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.tasnif = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    def forward(self, x):
        return self.tasnif(self.belgilar(x))

model = KichikCNN().eval(); x1 = torch.randn(1, 1, 8, 8)
torch.onnx.export(model, x1, "cnn_fp32.onnx", input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
quant_pre_process("cnn_fp32.onnx", "cnn_prep.onnx")
quantize_dynamic("cnn_prep.onnx", "cnn_dyn.onnx", weight_type=QuantType.QInt8)

# Berilgan quvvat modeli parametrlari: P = P_bosh + (P_tolik - P_bosh) * CPU_bandlik
P_TOLIK_W = 15.0
P_BOSH_W = 5.0

def energiya(fn, batch, takror):
    # TODO 1: devor vaqti va CPU vaqtini (process_time) o'lchab, 2 yadroga nisbatan bandlikni toping
    # TODO 2: P, uJ_tolik, uJ_dinamik va inf_s_W ni hisoblab lug'at qaytaring
    raise NotImplementedError("TODO 1-2: energiya")

print("=== Energiya bahosi ===")
# TODO 3: FP32 va INT8 dinamik, 1/2 oqim, batch 1/64 uchun jadval tuzing va eng tejamkorini toping
rows = []
df = pd.DataFrame(rows)
raise NotImplementedError("TODO 3: energiya jadvali")

print("\n=== Nusxalar rejasi (Little qonuni) ===")
lam = 5000
# TODO 4: har konfiguratsiya (batch 1, 2 oqim) uchun N = lam * latency, nusxalar soni va umumiy quvvat

print("\n=== GPU'da bu qadam qanday bo'ladi ===")
# TODO 5: trtexec (FP32/FP16/INT8) va nvidia-smi bilan quvvat o'lchash qadamlarini jadval qilib chiqaring
# TODO 6: uJ/inferens ustunli grafigi (log shkala, to'liq va dinamik)
