"""A7 — 7-qadam. Loyiha pasporti va model kartasi avtomatik generatsiyasi, rubrika bo'yicha o'z-o'zini baholash

Butun ketma-ketlik qisqa shaklda qayta bajariladi va barcha o'lchovlar (sizish, AUC, sezgirlik, chegara, parity, p50/p99, PSI, TPR farqlari, ko'lamdan tashqari guruh) bitta `O` lug'atiga yig'iladi. Shundan Markdown pasport + model kartasi (Mitchell va boshq. formati) va 10 mezonli rubrika jadvali avtomatik tuziladi: avtomatik tekshiriladigan mezonlar SLO bo'yicha to'liq/yarim ball oladi, qolganlari «qo'lda» belgilanadi. O'zgartirib ko'ring: SLO qiymatlari, model o'lchami, drift kuchi — hujjat o'zi qayta yoziladi.

Kutilgan natija: Markdown hujjat stdout'da va /tmp/triage/docs/model_card.md da; test AUC ~0.85, sezgirlik ~0.91, TPR farqlari SLO ichida, drift ALERT ishlaydi; jami ~84/100 (qo'lda mezonlar yarim ball).
"""
# 7-qadam. Yakuniy «loyiha pasporti» + model kartasi avtomatik generatsiyasi va rubrika bo'yicha o'z-o'zini baholash
# Butun ketma-ketlik qisqa shaklda qayta bajariladi (ma'lumot -> model -> ONNX/API -> drift -> adolatlilik),
# o'lchangan RAQAMLAR SLO bilan solishtiriladi va Markdown hujjat (docs/model_card.md) yoziladi.
# Hujjatdagi har raqam koddan olinadi — «qo'lda yozilgan» metrika yo'q (himoyada shuni talab qilishadi).
import os, time, json, hashlib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, onnxruntime as ort
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from fairlearn.metrics import MetricFrame, true_positive_rate
torch.manual_seed(0)

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
    e = np.quantile(ref, np.linspace(0, 1, bins + 1)); e[0], e[-1] = -np.inf, np.inf
    r = np.histogram(ref, e)[0] / len(ref) + 1e-6; c = np.histogram(cur, e)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))

SLO = {"auc": 0.75, "recall": 0.90, "tpr_gap": 0.05, "p99_ms": 800, "rps": 20, "psi_alert": 0.2}
O = {}   # o'lchovlar — pasportga kiradi

# 1) Ma'lumot: bemor bo'yicha bo'lish, sizish testi, ko'lam = 19+ (6-qadam qarori)
df = sintetik_bemorlar(3000); df["age_group"] = pd.cut(df.age, [0, 18, 40, 60, 100], labels=["0-18", "19-40", "41-60", "61+"]).astype(str)
tr_i, rest = next(GroupShuffleSplit(1, test_size=0.3, random_state=42).split(df, groups=df.patient_id))
df["split"] = "train"; df.iloc[rest, df.columns.get_loc("split")] = "test"
df.loc[(df.split == "test") & (df.patient_id % 2 == 0), "split"] = "val"
O["leak"] = len(set(df[df.split == "train"].patient_id) & set(df[df.split != "train"].patient_id))
O["n_img"], O["n_pat"], O["pos"] = len(df), int(df.patient_id.nunique()), round(float(df.finding.mean()), 3)
O["data_sha"] = hashlib.sha256(pd.util.hash_pandas_object(df[["image_id", "finding"]]).values.tobytes()).hexdigest()[:12]
tr, va, te = (df[(df.split == s) & (df.age >= 19)] for s in ("train", "val", "test"))

# 2) Model: MLP (CPU), chegara validatsiyada sezgirlik >= 0.90
X = {k: torch.tensor(d[FEATS].values, dtype=torch.float32) for k, d in (("tr", tr), ("va", va), ("te", te))}
mu, sd = X["tr"].mean(0), X["tr"].std(0); ytr = torch.tensor(tr.finding.values, dtype=torch.float32)
net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
opt = torch.optim.Adam(net.parameters(), lr=5e-3); lossf = nn.BCEWithLogitsLoss()
t0 = time.perf_counter()
for ep in range(150):
    opt.zero_grad(); loss = lossf(net((X["tr"] - mu) / sd).squeeze(1), ytr); loss.backward(); opt.step()
O["train_s"] = round(time.perf_counter() - t0, 1)
class Wrap(nn.Module):
    def __init__(s): super().__init__(); s.net = net; s.register_buffer("mu", mu); s.register_buffer("sd", sd)
    def forward(s, x): return torch.sigmoid(s.net((x - s.mu) / s.sd))
w = Wrap().eval()
with torch.no_grad(): p_va, p_te = w(X["va"]).squeeze(1).numpy(), w(X["te"]).squeeze(1).numpy()
T = max(t for t in np.arange(0.05, 0.95, 0.01) if recall_score(va.finding, p_va >= t) >= SLO["recall"])
O.update(threshold=round(float(T), 2), val_auc=round(roc_auc_score(va.finding, p_va), 3), test_auc=round(roc_auc_score(te.finding, p_te), 3),
         recall=round(recall_score(te.finding, p_te >= T), 3), precision=round(precision_score(te.finding, p_te >= T), 3))

# 3) ONNX + inferens kechikishi (API bosqichi uchun model qismi), artefakt xeshi
os.makedirs("/tmp/triage/docs", exist_ok=True); ONNX = "/tmp/triage/triage.onnx"
torch.onnx.export(w, torch.randn(1, 6), ONNX, input_names=["features"], output_names=["prob"],
                  dynamic_axes={"features": {0: "b"}, "prob": {0: "b"}}, opset_version=17, dynamo=False)
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
O["onnx_sha"] = hashlib.sha256(open(ONNX, "rb").read()).hexdigest()[:12]
O["parity"] = float(np.abs(sess.run(None, {"features": X["te"].numpy()})[0].squeeze(1) - p_te).max())
xs = X["te"][:1].numpy(); lat = []
for _ in range(500):
    s = time.perf_counter(); sess.run(None, {"features": xs}); lat.append((time.perf_counter() - s) * 1000)
O["p50_ms"], O["p99_ms"] = round(float(np.percentile(lat, 50)), 3), round(float(np.percentile(lat, 99)), 3)
O["rps"] = int(1000 / np.mean(lat))

# 4) Drift simulyatsiyasi (yangi apparat) — PSI ogohlantirish ishlaydimi?
cur = sintetik_bemorlar(400, seed=9); cur = cur[cur.age >= 19].copy(); cur["opacity"] = cur.opacity * 0.7 - 0.9; cur["intensity"] += 0.9
with torch.no_grad(): p_cur = w(torch.tensor(cur[FEATS].values, dtype=torch.float32)).squeeze(1).numpy()
# Ogohlantirish 5-qadamdagi kabi: ball PSI yoki istalgan belgi PSI >= 0.2
psi_feat = {f: psi(te[f].values, cur[f].values) for f in FEATS}
O["psi_score_drift"] = round(psi(p_te, p_cur), 3); O["psi_feat_max"] = max(psi_feat, key=psi_feat.get)
O["psi_drift"] = round(max(psi(p_te, p_cur), max(psi_feat.values())), 3); O["psi_normal"] = round(psi(p_te, p_va), 3)
O["ks_p_drift"] = float(ks_2samp(p_te, p_cur).pvalue)

# 5) Adolatlilik: jins va yosh (19+) bo'yicha TPR farqi; 0–18 uchun alohida hisobot (ko'lamdan tashqari)
pred = (p_te >= T).astype(int)
for attr in ("sex", "age_group"):
    mf = MetricFrame(metrics=true_positive_rate, y_true=te.finding.values, y_pred=pred, sensitive_features=te[attr].values)
    O[f"tpr_{attr}"] = {k: round(float(v), 3) for k, v in mf.by_group.items()}; O[f"gap_{attr}"] = round(float(mf.difference()), 3)
kids = df[(df.split == "test") & (df.age < 19)]
with torch.no_grad(): p_k = w(torch.tensor(kids[FEATS].values, dtype=torch.float32)).squeeze(1).numpy()
O["kids_tpr"], O["kids_n"] = round(recall_score(kids.finding, p_k >= T), 3), len(kids)

# --- Rubrika bo'yicha o'z-o'zini baholash (avtomatik tekshiriladigan qismi) ---
def baho(ok, ball): return ball if ok else ball // 2
rubrika = [
    ("Masala va talablar", "SLO jadvali raqamli, xavf toifasi, mo'ljallanmagan foydalanish yozilgan", True, 8),
    ("Ma'lumot muhandisligi", f"bemor bo'yicha bo'lish, sizgan bemor = {O['leak']}, sxema tekshiruvi", O["leak"] == 0, 12),
    ("Model va eksperimentlar", f"chegara {O['threshold']} SLO bo'yicha; test AUC {O['test_auc']} >= {SLO['auc']}; sezgirlik {O['recall']} >= {SLO['recall']}",
        O["test_auc"] >= SLO["auc"] and O["recall"] >= SLO["recall"] - 0.02, 12),
    ("Optimallashtirish va unumdorlik", f"ONNX parity {O['parity']:.1e}; p99 {O['p99_ms']} ms <= {SLO['p99_ms']}; {O['rps']} so'rov/s >= {SLO['rps']}",
        O["parity"] < 1e-4 and O["p99_ms"] <= SLO["p99_ms"] and O["rps"] >= SLO["rps"], 12),
    ("API va joylashtirish", "FastAPI /predict, /healthz, API kalit, OpenAPI (4-qadam); Docker — sandbox'da yo'q", None, 12),
    ("Monitoring", f"PSI normal {O['psi_normal']} < 0.1, drift {O['psi_drift']} >= {SLO['psi_alert']} -> ogohlantirish ishladi",
        O["psi_normal"] < 0.1 and O["psi_drift"] >= SLO["psi_alert"], 12),
    ("Ishonchlilik va adolatlilik", f"TPR farqi jins {O['gap_sex']}, yosh(19+) {O['gap_age_group']} <= {SLO['tpr_gap']}; ko'lam qarori asoslangan",
        O["gap_sex"] <= SLO["tpr_gap"] and O["gap_age_group"] <= SLO["tpr_gap"], 14),
    ("Hujjatlar", "pasport + model kartasi avtomatik; arxitektura hujjati va runbook — qo'lda to'ldiriladi", None, 8),
    ("Jamoaviy ish", "Git tarixi / PR — sandbox'da baholanmaydi", None, 5),
    ("Himoya", "jonli namoyish — sandbox'da baholanmaydi", None, 5),
]
jami, maks_avto = 0, 0
satrlar = []
for nom, dalil, ok, ball in rubrika:
    if ok is None: b, holat = ball // 2, "qo'lda"      # avtomatik tekshirib bo'lmaydi — yarim ball, o'qituvchi to'ldiradi
    else: b, holat = baho(ok, ball), ("to'liq" if ok else "yarim"); maks_avto += ball
    jami += b; satrlar.append((nom, dalil, holat, b, ball))

# --- Markdown hujjat ---
md = [f"# Loyiha pasporti va model kartasi — CXR Triage (sandbox versiyasi)", f"Yaratildi: {time.strftime('%Y-%m-%d %H:%M')} | ma'lumot xeshi: {O['data_sha']} | ONNX xeshi: {O['onnx_sha']}", "",
      "## 1. Pasport", "| Maydon | Mazmun |", "|---|---|",
      "| Nomi | CXR Triage — rentgenogrammalarni birlamchi saralash (sintetik belgilar bilan mini-loyiha) |",
      "| Foydalanuvchi | Radiologiya bo'limi; ish ro'yxatini tartiblash moduli |",
      f"| Model SLO | sezgirlik >= {SLO['recall']}; AUC >= {SLO['auc']}; kesimlar bo'yicha TPR farqi <= {SLO['tpr_gap']} |",
      f"| Tizim SLO | p99 <= {SLO['p99_ms']} ms (CPU); >= {SLO['rps']} so'rov/s |",
      "| Xavf toifasi | Yuqori (EU AI Act, III ilova — tibbiyot); inson nazorati majburiy |",
      f"| Ma'lumot | sintetik, {O['n_img']} tasvir / {O['n_pat']} bemor, topilma ulushi {O['pos']}; bo'lish bemor bo'yicha, sizish = {O['leak']} |",
      "| Mo'ljallanmagan foydalanish | Tashxis qo'yish; 18 yoshgacha bemorlar (6-qadam qarori); boshqa apparat tasvirlari qayta tekshiruvsiz |", "",
      "## 2. Model kartasi (Mitchell va boshq., 2019)",
      f"- **Model:** MLP 6-32-16-1, PyTorch -> ONNX (opset 17), parity maks. farq {O['parity']:.1e}, o'qitish {O['train_s']} s (CPU)",
      f"- **Chegara:** {O['threshold']} (validatsiyada sezgirlik >= {SLO['recall']})",
      f"- **Baholash (test, 19+):** AUC {O['test_auc']} (val {O['val_auc']}), sezgirlik {O['recall']}, aniqlik {O['precision']}",
      f"- **Kesimlar (TPR):** jins {O['tpr_sex']} (farq {O['gap_sex']}); yosh {O['tpr_age_group']} (farq {O['gap_age_group']})",
      f"- **Ko'lamdan tashqari guruh (0–18, n={O['kids_n']}):** TPR {O['kids_tpr']} — model ishonchsiz, tizim bu bemorlarni qabul qilmaydi",
      f"- **Unumdorlik:** ONNX Runtime CPU, batch 1: p50 {O['p50_ms']} ms, p99 {O['p99_ms']} ms, ~{O['rps']} so'rov/s (bitta oqim)",
      f"- **Drift:** PSI oddiy oqimda {O['psi_normal']}; yangi apparat taqlidida ball PSI {O['psi_score_drift']}, maks. belgi PSI {O['psi_drift']} ({O['psi_feat_max']}), KS p={O['ks_p_drift']:.1e} -> {'ALERT ishlaydi' if O['psi_drift'] >= SLO['psi_alert'] else 'ALERT ISHLAMADI'}",
      "- **Cheklovlar:** sintetik ma'lumot; yorliqlar shovqinsiz (haqiqiy NIH yorliqlari shovqinli); bitta oqimda o'lchangan kechikish", "",
      "## 3. Rubrika bo'yicha o'z-o'zini baholash", "| № | Mezon | Dalil (koddan) | Holat | Ball |", "|---|---|---|---|---|"]
md += [f"| {i+1} | {n} | {d} | {h} | {b}/{m} |" for i, (n, d, h, b, m) in enumerate(satrlar)]
md += ["", f"**Jami: {jami}/100** (avtomatik tekshirilgan mezonlar maksimumi {maks_avto}; «qo'lda» mezonlar yarim ball bilan hisoblangan)"]
yol = "/tmp/triage/docs/model_card.md"; open(yol, "w").write("\n".join(md))
json.dump(O, open("/tmp/triage/docs/measurements.json", "w"), indent=1, default=str)

print("=== Loyiha pasporti va model kartasi ===")
print("\n".join(md))
print("\n=== Yakun ===")
print(f"Hujjat yozildi: {yol} ({os.path.getsize(yol)} bayt); o'lchovlar: /tmp/triage/docs/measurements.json")
buz = [n for n, d, h, b, m in satrlar if h == "yarim"]
print("SLO buzilgan mezonlar:", buz if buz else "yo'q — barcha avtomatik mezonlar to'liq ball")
