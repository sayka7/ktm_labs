"""A7 — 1-qadam. Ma'lumot sxemasi validatsiyasi va bemor bo'yicha bo'lish

Sintetik «bemor» manifesti yaratiladi (har bemorda 1–3 tasvir, 6 ta tasvir belgisi, yosh, jins, topilma). Pandera uslubidagi sxema tekshiruvi toza ma'lumotda o'tadi, ataylab buzilgan partiyada esa har xatoni aniq nomlaydi. So'ng `GroupShuffleSplit` bilan BEMOR bo'yicha bo'linadi va sizish `assert` bilan tekshiriladi; taqqoslash uchun tasodifiy (tasvir bo'yicha) bo'lishda nechta bemor sizishi hisoblanadi. O'zgartirib ko'ring: `test_size`, buzilgan qiymatlar, `n_bemor`.

Kutilgan natija: Sxema toza ma'lumotda O'TDI, buzilgan partiyada 6 ta xato satri; bemor bo'yicha bo'lishda sizish 0, tasodifiy bo'lishda ~35% bemor sizadi; split × sex va yosh guruhi × split jadvallari.
"""
# 1-qadam. Ma'lumot sxemasi validatsiyasi va BEMOR bo'yicha bo'lish (sizishsiz)
# Sandbox'da haqiqiy rentgen tasvirlari yo'q: har tasvir o'rniga undan olingan 6 ta
# sonli belgi (opaklik, o'pka maydoni, kardiotorakal nisbat, tekstura, yorqinlik, qirra zichligi)
# sintetik tarzda yaratiladi. Bitta bemorda 1–3 ta tasvir bo'ladi — xuddi NIH ChestX-ray14 dagidek.
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """Bemor-darajali sintetik manifest: image_id, patient_id, age, sex, 6 belgi, finding."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_bemor):
        age = int(np.clip(rng.normal(50, 18), 1, 95))
        sex = "M" if rng.random() < 0.56 else "F"
        z = rng.normal(0, 1) + 0.02 * (age - 50)          # bemorning yashirin "patologiya" omili
        for k in range(int(rng.integers(1, 4))):            # 1–3 ta tasvir
            shovqin = 2.2 if age < 18 else 1.0               # bolalar tasviri shovqinliroq (kesim farqi uchun)
            e = rng.normal(0, shovqin, 6)
            f = [0.9*z+e[0], -0.6*z+e[1], 0.5*z+e[2], 0.7*z+e[3], 0.2*z+e[4], 0.4*z+e[5]]
            finding = int(z + rng.normal(0, 0.6) > 0.3)
            rows.append([f"{pid:08d}_{k:03d}.png", pid, age, sex, *np.round(f, 4), finding])
    return pd.DataFrame(rows, columns=["image_id", "patient_id", "age", "sex", *FEATS, "finding"])

# --- Sxema: Pandera uslubidagi kichik tekshiruvchi (sandbox'da pandera yo'q) ---
SXEMA = {
    "image_id":   dict(dtype="str",   regex=r"^\d{8}_\d{3}\.png$", unique=True),
    "patient_id": dict(dtype="int",   min=0),
    "age":        dict(dtype="int",   min=0, max=100),
    "sex":        dict(dtype="str",   isin={"M", "F"}),
    "finding":    dict(dtype="int",   isin={0, 1}),
    **{f: dict(dtype="float", min=-15, max=15) for f in FEATS},
}

def validate(df, sxema=SXEMA):
    """Har ustun uchun qoidalarni tekshiradi; buzilishlar ro'yxatini qaytaradi (bo'sh = to'g'ri)."""
    xatolar = []
    for col, q in sxema.items():
        if col not in df.columns:
            xatolar.append(f"{col}: ustun yo'q"); continue
        s = df[col]
        if s.isna().any():
            xatolar.append(f"{col}: {int(s.isna().sum())} ta bo'sh qiymat")
        if q["dtype"] in ("int", "float") and not pd.api.types.is_numeric_dtype(s):
            xatolar.append(f"{col}: son emas"); continue
        if "min" in q and (s < q["min"]).any():
            xatolar.append(f"{col}: {int((s < q['min']).sum())} ta qiymat < {q['min']}")
        if "max" in q and (s > q["max"]).any():
            xatolar.append(f"{col}: {int((s > q['max']).sum())} ta qiymat > {q['max']}")
        if "isin" in q and (~s.isin(q["isin"])).any():
            xatolar.append(f"{col}: ruxsatsiz qiymatlar {sorted(set(s[~s.isin(q['isin'])].astype(str)))[:5]}")
        if "regex" in q and (~s.astype(str).str.match(q["regex"])).any():
            xatolar.append(f"{col}: {int((~s.astype(str).str.match(q['regex'])).sum())} ta nom formati noto'g'ri")
        if q.get("unique") and s.duplicated().any():
            xatolar.append(f"{col}: {int(s.duplicated().sum())} ta takror")
    return xatolar

df = sintetik_bemorlar()
print("=== Manifest ===")
print(f"tasvirlar: {len(df)}, bemorlar: {df.patient_id.nunique()}, topilma ulushi: {df.finding.mean():.3f}")
print(df.head(5).to_string(index=False))

print("\n=== Sxema tekshiruvi (toza ma'lumot) ===")
x = validate(df)
print("natija:", "O'TDI" if not x else x)

# Buzilgan «yangi partiya»: shifoxona arxividan kelgan ma'lumotda tipik xatolar
buzuq = df.sample(200, random_state=1).copy()
buzuq.loc[buzuq.index[:3], "age"] = [412, -1, 150]        # yosh xato kiritilgan
buzuq.loc[buzuq.index[3:5], "sex"] = ["U", "m"]            # noma'lum / kichik harf
buzuq.loc[buzuq.index[5], "image_id"] = "scan_final.jpg"   # nom formati
buzuq.loc[buzuq.index[6], "finding"] = 2                   # yorliq kodi
buzuq.loc[buzuq.index[7], "opacity"] = np.nan              # yo'qolgan belgi
print("\n=== Sxema tekshiruvi (buzilgan partiya) ===")
for xat in validate(buzuq):
    print(" -", xat)

# --- Bo'lish: BEMOR bo'yicha (GroupShuffleSplit), so'ng testning yarmi validatsiya ---
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_idx, rest_idx = next(gss.split(df, groups=df["patient_id"]))
df["split"] = "train"
df.iloc[rest_idx, df.columns.get_loc("split")] = "test"
val_mask = (df["split"] == "test") & (df["patient_id"] % 2 == 0)
df.loc[val_mask, "split"] = "val"

# Sizish testi: bitta bemor ikki to'plamda bo'lmasligi shart
tr_p = set(df[df.split == "train"].patient_id)
assert tr_p.isdisjoint(df[df.split == "test"].patient_id) and tr_p.isdisjoint(df[df.split == "val"].patient_id)

# Taqqoslash uchun NOTO'G'RI (tasodifiy, tasvir bo'yicha) bo'lish — nechta bemor sizadi?
rng = np.random.default_rng(0)
tasodifiy = rng.random(len(df)) < 0.7
sizgan = len(set(df[tasodifiy].patient_id) & set(df[~tasodifiy].patient_id))

print("\n=== Bo'lish natijasi ===")
jad = df.groupby(["split", "sex"])["finding"].agg(count="count", mean="mean").round(3)
print(jad.to_string())
print("\nBemorlar soni:", df.groupby("split")["patient_id"].nunique().to_dict())
print(f"Bemor bo'yicha bo'lish: train∩test bemorlar = 0 (sizish yo'q)")
print(f"Tasodifiy (tasvir bo'yicha) bo'lishda sizadigan bemorlar: {sizgan} ta ({sizgan/df.patient_id.nunique():.0%})")

df["age_group"] = pd.cut(df.age, [0, 18, 40, 60, 100], labels=["0-18", "19-40", "41-60", "61+"])
print("\n=== Vakillik jadvali: yosh guruhi × split (tasvirlar soni) ===")
print(pd.crosstab(df.age_group, df.split).to_string())

import os
os.makedirs("/tmp/triage/data", exist_ok=True)
df.to_csv("/tmp/triage/data/manifest.csv", index=False)
print("\nSaqlandi: /tmp/triage/data/manifest.csv", df.shape)
