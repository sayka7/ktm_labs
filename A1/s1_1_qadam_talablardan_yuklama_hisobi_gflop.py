"""A1 — 1-qadam. Talablardan yuklama hisobi: GFLOP, bayt, TFLOP/s

Ssenariy talablari (16 kamera × 10 kadr/s) YOLOv8 variantlari uchun bir inferens GFLOP, vazn hajmi (FP16/INT8) va jami TFLOP/s talabiga aylantiriladi; samaradorlik koeffitsiyenti orqali kerakli cho'qqi TOPS topiladi. Kutilgan natija: yolov8s @ 640 uchun 28,6 GFLOP, 4,58 TFLOP/s haqiqiy, ~10 TOPS cho'qqi. Kodda KAMERALAR, FPS, SAMARADORLIK va MODELLAR jadvalini o'zgartirib, talab qanday o'sishini kuzating.

Kutilgan natija: 12 qatorli jadval (4 model × 3 imgsz), etalon yolov8s@640: 4,58 TFLOP/s, cho'qqi 10,2 TOPS; tarmoq 64 Mbit/s, xom kadrlar ~1 GB/s.
"""
# 1-qadam. Ssenariy talablaridan hisoblash yuklamasini (GFLOP, TFLOP/s) hisoblash
# Real labda ultralytics YOLO(...).info() modelni yuklab GFLOP'ni beradi. Sandbox'da tarmoq yo'q,
# shuning uchun Ultralytics hujjatlaridagi rasmiy (640x640) qiymatlarni jadval sifatida kiritamiz
# va boshqa kirish o'lchamlari uchun FLOP ~ (H*W) proporsionalligidan foydalanamiz.
import pandas as pd

# --- Ssenariy talablari (o'zgartirib ko'ring) ---
KAMERALAR = 16            # kameralar soni
FPS = 10                  # har kamera uchun kadr/s
SAMARADORLIK = 0.45       # platformaning cho'qqi unumdorligidan real erishiladigan ulush (30-60 %)
INFERENS_S = KAMERALAR * FPS

# Ultralytics hujjatlari, 640x640 kirish: (parametrlar soni, GFLOP bir inferens uchun, COCO mAP50-95)
MODELLAR = {
    "yolov8n": (3.15e6, 8.7, 37.3),
    "yolov8s": (11.17e6, 28.6, 44.9),
    "yolov8m": (25.90e6, 78.9, 50.2),
    "yolov8l": (43.70e6, 165.2, 52.9),
}


def model_profili(nomi, imgsz=640):
    """Bitta model varianti uchun statik yuklama ko'rsatkichlari."""
    params, gflop640, mAP = MODELLAR[nomi]
    gflop = gflop640 * (imgsz / 640) ** 2          # konvolyutsiya FLOP'i kirish maydoniga proporsional
    talab_tflops = gflop * INFERENS_S / 1e3         # 160 inferens/s uchun jami talab
    return {
        "model": nomi,
        "imgsz": imgsz,
        "params_M": round(params / 1e6, 2),
        "vazn_FP16_MB": round(params * 2 / 1e6, 1),   # FP16: 2 bayt/parametr
        "vazn_INT8_MB": round(params * 1 / 1e6, 1),   # INT8: 1 bayt/parametr
        "GFLOP_inferens": round(gflop, 1),
        "TFLOP_s_talab": round(talab_tflops, 2),
        # samaradorlik koeffitsiyentini hisobga olgan holda kerakli CHO'QQI unumdorlik
        "kerakli_choqqi_TOPS": round(talab_tflops / SAMARADORLIK, 1),
        "mAP": mAP,
    }


qatorlar = [model_profili(n, s) for n in MODELLAR for s in (416, 512, 640)]
df = pd.DataFrame(qatorlar)

print("=== Yuklama hisobi ===")
print(f"Ssenariy: {KAMERALAR} kamera x {FPS} kadr/s = {INFERENS_S} inferens/s, samaradorlik {SAMARADORLIK:.0%}")
print(df.to_markdown(index=False))

# Talablar matritsasidagi chegara: YOLOv8s @ 640
etalon = df[(df.model == "yolov8s") & (df.imgsz == 640)].iloc[0]
print("\n--- Etalon konfiguratsiya: yolov8s @ 640 ---")
print(f"Bir inferens: {etalon.GFLOP_inferens} GFLOP, vazn FP16 {etalon.vazn_FP16_MB} MB")
print(f"Jami talab: {etalon.TFLOP_s_talab} TFLOP/s haqiqiy -> cho'qqi kamida {etalon.kerakli_choqqi_TOPS} TOPS")

# Kirish oqimi (dekoderdan oldin va keyin) — tarmoq va xotira talabi
bitrate_mbit = 4.0                                   # bitta 1080p10 H.264 oqim, Mbit/s
xom_kadr_MB = 1920 * 1080 * 3 / 1e6                  # dekodlangan RGB kadr
print(f"\nTarmoq: {KAMERALAR} x {bitrate_mbit} = {KAMERALAR * bitrate_mbit:.0f} Mbit/s (1 Gbit/s portning {KAMERALAR * bitrate_mbit / 10:.1f} %)")
print(f"Dekodlangan xom kadrlar: {INFERENS_S} x {xom_kadr_MB:.1f} MB = {INFERENS_S * xom_kadr_MB / 1e3:.2f} GB/s xotira yozuvi")
df.to_csv("yuklama.csv", index=False)
print("yuklama.csv saqlandi")
