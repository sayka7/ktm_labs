"""A7 — 5-qadam. Monitoring: PSI/KS bilan drift aniqlash va ogohlantirish mantiqi

Validatsiya to'plami mos yozuv bo'ladi. 8 kunlik ishlab chiqarish oqimi simulyatsiya qilinadi: 4-kundan «yangi apparat» (opaklik kontrasti −30% va siljish, yorqinlik oshishi) kuchayib boradi. Har kun uchun belgilar va model ballari PSI, KS-test p-qiymati, HIGH ulushi hisoblanib, darajali qoida (OK/WARN/ALERT) ishlaydi; sezgirlik yorliq bilan qayta hisoblanib, driftning haqiqiy zarari ko'rsatiladi. O'zgartirib ko'ring: siljish kuchi, qoidalar chegarasi, chelaklar soni.

Kutilgan natija: 1–4 kun OK, 5-kundan ALERT (intensity/opacity PSI > 0.3), 8-kunda ball PSI ~0.23, KS p ~1e-14, sezgirlik 0.86 -> 0.68; kun × belgi PSI jadvali va grafik.
"""
# 5-qadam. Monitoring: PSI va KS-test bilan drift aniqlash, ogohlantirish mantiqi
# Mos yozuv (reference) — validatsiya to'plamidagi belgilar va model ballari.
# Simulyatsiya: shifoxona yangi rentgen apparati o'rnatdi -> kontrast pasaydi, yorqinlik siljidi
# (bizda: 'opacity' va 'intensity' belgilari siljiydi). Drift kun sayin kuchayadi; har kun — monitoring oynasi.
import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings("ignore")
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, recall_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FEATS = ["opacity", "lung_area", "ct_ratio", "texture", "intensity", "edge_density"]

def sintetik_bemorlar(n_bemor=1500, seed=0):
    """1-qadamdagi generator: bemor-darajali sintetik manifest."""
    rng = np.random.default_rng(seed); rows = []
    for pid in range(n_bemor):
        age = int(np.clip(rng.normal(50, 18), 1, 95)); sex = "M" if rng.random() < 0.56 else "F"
        z = rng.normal(0, 1) + 0.02 * (age - 50)
        for k in range(int(rng.integers(1, 4))):
            e = rng.normal(0, 2.2 if age < 18 else 1.0, 6)
            f = [0.9*z+e[0], -0.6*z+e[1], 0.5*z+e[2], 0.7*z+e[3], 0.2*z+e[4], 0.4*z+e[5]]
            rows.append([f"{pid:08d}_{k:03d}.png", pid, age, sex, *np.round(f, 4), int(z + rng.normal(0, 0.6) > 0.3)])
    return pd.DataFrame(rows, columns=["image_id", "patient_id", "age", "sex", *FEATS, "finding"])

def psi(ref, cur, bins=10):
    """Population Stability Index: mos yozuv kvantillari bo'yicha chelaklar; <0.1 barqaror, 0.1–0.2 kuzatuv, >=0.2 drift."""
    e = np.quantile(ref, np.linspace(0, 1, bins + 1)); e[0], e[-1] = -np.inf, np.inf
    r = np.histogram(ref, e)[0] / len(ref) + 1e-6; c = np.histogram(cur, e)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))

# --- Model va mos yozuv ---
df = sintetik_bemorlar(3000)
tr_i, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(df, groups=df.patient_id))
tr, ref = df.iloc[tr_i], df.iloc[rest]
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(tr[FEATS], tr.finding)
ref_scores = model.predict_proba(ref[FEATS])[:, 1]
THRESHOLD = 0.30
print(f"Mos yozuv: {len(ref)} tasvir, o'rtacha ball {ref_scores.mean():.3f}, HIGH ulushi {(ref_scores>=THRESHOLD).mean():.3f}, AUC {roc_auc_score(ref.finding, ref_scores):.3f}")

# --- 8 kunlik ishlab chiqarish oqimi: 4-kundan yangi apparat (kontrast −30%, yorqinlik siljishi) ---
QOIDA = {"psi_warn": 0.10, "psi_alert": 0.20, "ks_p": 0.01, "pos_rate_rel": 0.25}   # ogohlantirish qoidalari

def yangi_apparat(df, kuch):
    """Boshqa apparat taqlidi: opaklik kontrasti −30% va siljish, yorqinlik oshadi, tekstura silliqlashadi (kuch 0..1)."""
    cur = df.copy()
    cur["opacity"] = cur["opacity"] * (1 - 0.30 * kuch) - 0.9 * kuch
    cur["intensity"] = cur["intensity"] + 0.9 * kuch
    cur["texture"] = cur["texture"] - 0.5 * kuch
    return cur

jurnal, psi_jadval = [], []
pos_ref = float((ref_scores >= THRESHOLD).mean())
print("\n=== Kunlik monitoring oynalari ===")
print(f"{'kun':>3} {'n':>4} {'PSI(ball)':>9} {'PSI maks belgi':>22} {'KS p(ball)':>10} {'HIGH ulushi':>11} {'sezg.*':>6}  holat")
for kun in range(1, 9):
    kuch = 0.0 if kun <= 3 else min(1.0, (kun - 3) / 3)             # 4-kundan boshlab kuchayadigan siljish
    cur = yangi_apparat(sintetik_bemorlar(250, seed=100 + kun), kuch)
    sc = model.predict_proba(cur[FEATS])[:, 1]
    psi_f = {f: round(psi(ref[f].values, cur[f].values), 3) for f in FEATS}; psi_jadval.append({"kun": kun, **psi_f})
    fmax = max(psi_f, key=psi_f.get)
    psi_s = psi(ref_scores, sc); ks_p = ks_2samp(ref_scores, sc).pvalue
    pos = float((sc >= THRESHOLD).mean())
    rec = recall_score(cur.finding, sc >= THRESHOLD)                  # * haqiqiy yorliq odatda kechikib keladi; bu yerda «oltin» yorliq bor deb faraz
    # Ogohlantirish mantiqi: darajali (OK / WARN / ALERT) + sabab
    sabab = []
    if psi_s >= QOIDA["psi_alert"] or psi_f[fmax] >= QOIDA["psi_alert"]: holat = "ALERT"; sabab.append("PSI>=0.2")
    elif psi_s >= QOIDA["psi_warn"] or psi_f[fmax] >= QOIDA["psi_warn"]: holat = "WARN"; sabab.append("PSI>=0.1")
    else: holat = "OK"
    if ks_p < QOIDA["ks_p"]: sabab.append("KS p<0.01"); holat = "ALERT" if holat != "OK" else "WARN"
    if abs(pos - pos_ref) / pos_ref > QOIDA["pos_rate_rel"]: sabab.append("HIGH ulushi ±25%"); holat = "ALERT" if holat != "OK" else "WARN"
    jurnal.append(dict(kun=kun, n=len(cur), psi_score=round(psi_s, 3), psi_feat_max=fmax, psi_feat=psi_f[fmax],
                       ks_p=float(ks_p), pos_rate=round(pos, 3), recall=round(rec, 3), status=holat, reasons=sabab))
    print(f"{kun:>3} {len(cur):>4} {psi_s:>9.3f} {fmax:>14} {psi_f[fmax]:>7.3f} {ks_p:>10.1e} {pos:>11.3f} {rec:>6.3f}  {holat} {' '.join(sabab)}")

birinchi = next((j for j in jurnal if j["status"] == "ALERT"), None)
print("\n=== Ogohlantirish xulosasi ===")
print("Qoidalar:", QOIDA)
print("Birinchi ALERT:", f"{birinchi['kun']}-kun ({', '.join(birinchi['reasons'])})" if birinchi else "yo'q")
print("Eng ko'p siljigan belgi:", pd.Series([j["psi_feat_max"] for j in jurnal[3:]]).mode()[0],
      "-> runbook: apparat sozlamalarini tekshirish, mos yozuvni yangilash yoki qayta o'qitish")
print(f"Sezgirlik: 1-kun {jurnal[0]['recall']} -> 8-kun {jurnal[-1]['recall']} (SLO 0.90). Yorliq kechikib keladi — PSI buni yorliqsiz, oldinroq ko'rsatdi.")

print("\n=== PSI belgilar bo'yicha (kun × belgi) ===")
print(pd.DataFrame(psi_jadval).set_index("kun").to_string())

# Grafik: ball PSI va HIGH ulushi dinamikasi
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot([j["kun"] for j in jurnal], [j["psi_score"] for j in jurnal], "o-"); ax[0].axhline(0.2, color="r", ls="--", label="ALERT 0.2"); ax[0].axhline(0.1, color="orange", ls="--", label="WARN 0.1")
ax[0].set_title("PSI (model balli)"); ax[0].set_xlabel("kun"); ax[0].legend()
ax[1].plot([j["kun"] for j in jurnal], [j["pos_rate"] for j in jurnal], "s-", color="purple"); ax[1].axhline(pos_ref, color="gray", ls=":", label="mos yozuv")
ax[1].set_title("HIGH ulushi"); ax[1].set_xlabel("kun"); ax[1].legend(); plt.tight_layout()
json.dump(jurnal, open("/tmp/drift_log.json", "w"), indent=1)
print("\nDrift jurnali saqlandi: /tmp/drift_log.json")
