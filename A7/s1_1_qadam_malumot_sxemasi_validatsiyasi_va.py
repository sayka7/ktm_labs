"""A7 — 1-qadam. Ma'lumot sxemasi validatsiyasi va bemor bo'yicha bo'lish

Sintetik «bemor» manifesti yaratiladi (har bemorda 1–3 tasvir, 6 ta tasvir belgisi, yosh, jins, topilma). Pandera uslubidagi sxema tekshiruvi toza ma'lumotda o'tadi, ataylab buzilgan partiyada esa har xatoni aniq nomlaydi. So'ng `GroupShuffleSplit` bilan BEMOR bo'yicha bo'linadi va sizish `assert` bilan tekshiriladi; taqqoslash uchun tasodifiy (tasvir bo'yicha) bo'lishda nechta bemor sizishi hisoblanadi. O'zgartirib ko'ring: `test_size`, buzilgan qiymatlar, `n_bemor`.

Kutilgan natija: Sxema toza ma'lumotda O'TDI, buzilgan partiyada 6 ta xato satri; bemor bo'yicha bo'lishda sizish 0, tasodifiy bo'lishda ~35% bemor sizadi; split × sex va yosh guruhi × split jadvallari.
"""
# 1-qadam. Ma'lumot sxemasi validatsiyasi va BEMOR bo'yicha bo'lish (sizishsiz)
# Sandbox'da haqiqiy rentgen tasvirlari yo'q: har tasvir o'rniga 6 ta sonli belgi sintetik yaratiladi.
# Bitta bemorda 1–3 ta tasvir bo'ladi — xuddi NIH ChestX-ray14 dagidek.
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """Bemor-darajali sintetik manifest (berilgan, o'zgartirmang)."""
    rng = np.random.default_rng(seed); rows = []
    for pid in range(n_bemor):
        age = int(np.clip(rng.normal(50, 18), 1, 95)); sex = "M" if rng.random() < 0.56 else "F"
        z = rng.normal(0, 1) + 0.02 * (age - 50)
        for k in range(int(rng.integers(1, 4))):
            e = rng.normal(0, 2.2 if age < 18 else 1.0, 6)
            f = [0.9*z+e[0], -0.6*z+e[1], 0.5*z+e[2], 0.7*z+e[3], 0.2*z+e[4], 0.4*z+e[5]]
            rows.append([f"{pid:08d}_{k:03d}.png", pid, age, sex, *np.round(f, 4), int(z + rng.normal(0, 0.6) > 0.3)])
    return pd.DataFrame(rows, columns=["image_id", "patient_id", "age", "sex", *FEATS, "finding"])

# Sxema qoidalari (berilgan)
SXEMA = {
    "image_id":   dict(dtype="str",   regex=r"^\d{8}_\d{3}\.png$", unique=True),
    "patient_id": dict(dtype="int",   min=0),
    "age":        dict(dtype="int",   min=0, max=100),
    "sex":        dict(dtype="str",   isin={"M", "F"}),
    "finding":    dict(dtype="int",   isin={0, 1}),
    **{f: dict(dtype="float", min=-15, max=15) for f in FEATS},
}

# TODO 1: har ustun uchun qoidalarni (ustun bormi, NaN, son turi, min/max, isin, regex, unique) tekshiring
def validate(df, sxema=SXEMA):
    """Buzilishlar ro'yxatini qaytaradi (bo'sh = to'g'ri); har xato ustun nomi bilan boshlansin."""
    raise NotImplementedError("TODO 1: sxema qoidalarini tekshiring")

df = sintetik_bemorlar()
print("=== Manifest ===")
print(f"tasvirlar: {len(df)}, bemorlar: {df.patient_id.nunique()}, topilma ulushi: {df.finding.mean():.3f}")
print("\n=== Sxema tekshiruvi (toza ma'lumot) ===")
x = validate(df)
print("natija:", "O'TDI" if not x else x)

buzuq = df.sample(200, random_state=1).copy()
# TODO 2: buzuq partiyaga 6 xil tipik xato kiriting (yosh, jins, fayl nomi, yorliq kodi, NaN belgi)
print("\n=== Sxema tekshiruvi (buzilgan partiya) ===")
for xat in validate(buzuq):
    print(" -", xat)

# TODO 3: GroupShuffleSplit (test_size=0.3, random_state=42) bilan patient_id bo'yicha bo'ling,
#         "split" ustuniga train/test yozing, testdagi juft patient_id larni "val" qiling
df["split"] = None

# TODO 4: train bemorlari test va val bemorlari bilan kesishmasligini assert bilan tekshiring

# TODO 5: taqqoslash — tasvir bo'yicha tasodifiy 70/30 bo'lishda nechta bemor ikki tomonda qolishini sanang
sizgan = None

print("\n=== Bo'lish natijasi ===")
# TODO 6: split × sex jadvali (finding count/mean), split bo'yicha bemorlar soni va sizish natijasini chiqaring
print(f"Bemor bo'yicha bo'lish: train∩test bemorlar = 0 (sizish yo'q)")

# TODO 7: yosh guruhlari (0-18, 19-40, 41-60, 61+) × split crosstab va manifest.csv ni /tmp/triage/data ga saqlash
