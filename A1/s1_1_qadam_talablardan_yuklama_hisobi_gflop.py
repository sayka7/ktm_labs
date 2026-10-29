"""A1 — 1-qadam. Talablardan yuklama hisobi: GFLOP, bayt, TFLOP/s

Ssenariy talablari (16 kamera × 10 kadr/s) YOLOv8 variantlari uchun bir inferens GFLOP, vazn hajmi (FP16/INT8) va jami TFLOP/s talabiga aylantiriladi; samaradorlik koeffitsiyenti orqali kerakli cho'qqi TOPS topiladi. Kutilgan natija: yolov8s @ 640 uchun 28,6 GFLOP, 4,58 TFLOP/s haqiqiy, ~10 TOPS cho'qqi. Kodda KAMERALAR, FPS, SAMARADORLIK va MODELLAR jadvalini o'zgartirib, talab qanday o'sishini kuzating.

Kutilgan natija: 12 qatorli jadval (4 model × 3 imgsz), etalon yolov8s@640: 4,58 TFLOP/s, cho'qqi 10,2 TOPS; tarmoq 64 Mbit/s, xom kadrlar ~1 GB/s.
"""
# 1-qadam. Ssenariy talablaridan hisoblash yuklamasini (GFLOP, TFLOP/s) hisoblash
# Sandbox'da tarmoq yo'q: Ultralytics hujjatlaridagi rasmiy (640x640) qiymatlar jadval sifatida berilgan.
import pandas as pd

# --- Ssenariy talablari (o'zgartirib ko'ring) ---
KAMERALAR = 16            # kameralar soni
FPS = 10                  # har kamera uchun kadr/s
SAMARADORLIK = 0.45       # platformaning cho'qqi unumdorligidan real erishiladigan ulush (30-60 %)
INFERENS_S = KAMERALAR * FPS

# Berilgan ma'lumotlar (o'zgartirmang): (parametrlar soni, GFLOP @640, COCO mAP50-95)
MODELLAR = {
    "yolov8n": (3.15e6, 8.7, 37.3),
    "yolov8s": (11.17e6, 28.6, 44.9),
    "yolov8m": (25.90e6, 78.9, 50.2),
    "yolov8l": (43.70e6, 165.2, 52.9),
}


def model_profili(nomi, imgsz=640):
    """Bitta model varianti uchun statik yuklama ko'rsatkichlari (lug'at qaytaradi)."""
    params, gflop640, mAP = MODELLAR[nomi]
    # TODO 1: imgsz uchun GFLOP ni 640 dagi qiymatdan hisoblang (FLOP kirish maydoniga proporsional)
    # TODO 2: 160 inferens/s uchun jami TFLOP/s talabini toping
    # TODO 3: FP16 va INT8 vazn hajmini (MB) parametrlar sonidan hisoblang
    # TODO 4: SAMARADORLIK ni hisobga olib kerakli cho'qqi TOPS ni toping
    # Kalitlar: model, imgsz, params_M, vazn_FP16_MB, vazn_INT8_MB, GFLOP_inferens,
    #           TFLOP_s_talab, kerakli_choqqi_TOPS, mAP
    raise NotImplementedError("TODO 1-4: model_profili ni yozing")


# TODO 5: 4 model x (416, 512, 640) uchun 12 qatorli DataFrame yarating
df = None

print("=== Yuklama hisobi ===")
print(df.to_markdown(index=False))

# TODO 6: etalon yolov8s @ 640 qatorini ajratib, GFLOP, TFLOP/s va cho'qqi TOPS ni chiqaring
# TODO 7: tarmoq (16 x 4 Mbit/s) va dekodlangan 1920x1080x3 kadrlar oqimini (GB/s) hisoblang
