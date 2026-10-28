"""MT1 — 7-qadam. Joylash va hujjat: API xizmat, kanareyka relizi, model kartasi va hayot sikli hisoboti

Ikki model versiyasi (v1 — mashhurlik, v2 — MF) FastAPI xizmatiga joylanadi va `TestClient` bilan tekshiriladi (Docker o'rniga jarayon ichidagi xizmat). Kanareyka relizi 0% → 10% → 50% → 100% bosqichlarida foydalanuvchi id xeshi bo'yicha barqaror taqsimlanadi, har bosqichda kechikish va xatolik «qo'riqchi» metrikalari tekshiriladi. Oxirida YAML model kartasi, Netflix/Uber/Amazon misollarining umumiy taqqosi va hayot sikli hisoboti (JSON) yaratiladi.

Kutilgan natija: Barcha bosqichlarda xato 0, p95 ~4 ms (jarayon ichida, tarmoqsiz); YAML model kartasi va 8 bosqichli hisobot jadvali chiqadi (raqamlar taxminiy).
"""
# 7-qadam. Joylash va hujjatlash: FastAPI tavsiya xizmati, kanareyka relizi, model kartasi va hayot sikli hisoboti
import warnings; warnings.filterwarnings("ignore")
import time, json, numpy as np, pandas as pd, yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

rng = np.random.default_rng(1)
n_u, n_i, k = 300, 200, 8
# Ikki model versiyasi (registr): v1 — mashhurlik, v2 — MF omillari (bu yerda tayyor vaznlar sifatida sintetik)
pop = rng.gamma(1.0, 1.0, n_i)
P, Q = rng.normal(0, 0.3, (n_u, k)), rng.normal(0, 0.3, (n_i, k))
REGISTR = {
    "v1": {"tur": "mashhurlik", "baho": lambda u: pop},
    "v2": {"tur": "matrix factorization k=8", "baho": lambda u: Q @ P[u] + 0.3 * np.log1p(pop)},
}
holat = {"kanareyka_ulushi": 0.0, "sorovlar": {"v1": 0, "v2": 0}}

app = FastAPI(title="Tavsiya xizmati")

@app.get("/health")
def health():
    return {"status": "ok", "modellar": list(REGISTR), "kanareyka": holat["kanareyka_ulushi"]}

@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    if not 0 <= user_id < n_u:
        raise HTTPException(404, "foydalanuvchi topilmadi (sovuq start -> mashhurlik siyosati)")
    # Kanareyka: foydalanuvchi id xeshi bo'yicha barqaror taqsimlash (bir odam doim bitta versiyani ko'radi)
    ver = "v2" if (user_id * 2654435761 % 2**32) / 2**32 < holat["kanareyka_ulushi"] else "v1"
    holat["sorovlar"][ver] += 1
    top = np.argsort(-REGISTR[ver]["baho"](user_id))[:n]
    return {"user": user_id, "model": ver, "items": top.tolist()}

c = TestClient(app)
print("=== Xizmat tekshiruvi ===")
print("GET /health ->", c.get("/health").json())
print("GET /recommend/5 ->", c.get("/recommend/5?n=5").json())
print("GET /recommend/9999 ->", c.get("/recommend/9999").status_code, "(kutilgan 404)")

# --- Kanareyka relizi: 0% -> 10% -> 50% -> 100%; har bosqichda kechikish va xatolik «qo'riqchi» metrikalari
print("\n=== Kanareyka relizi ===")
rows = []
for ulush in (0.0, 0.1, 0.5, 1.0):
    holat["kanareyka_ulushi"] = ulush; holat["sorovlar"] = {"v1": 0, "v2": 0}
    lat, xato = [], 0
    for _ in range(300):
        a = time.perf_counter(); r = c.get(f"/recommend/{int(rng.integers(0, n_u))}"); lat.append((time.perf_counter() - a) * 1e3)
        xato += r.status_code != 200
    p95 = np.percentile(lat, 95); ok = xato == 0 and p95 < 50
    rows.append((f"{ulush:.0%}", holat["sorovlar"]["v2"], round(float(np.median(lat)), 2), round(float(p95), 2), xato, "davom" if ok else "ORTGA QAYTARISH"))
print(pd.DataFrame(rows, columns=["v2 ulushi", "v2 so'rovlari", "p50, ms", "p95, ms", "xato", "qaror"]).to_string(index=False))
print("Izoh: TestClient jarayon ichida ishlaydi — tarmoq kechikishi yo'q; real xizmatda p95 bir necha barobar katta bo'ladi.")

# --- Model kartasi (YAML) — raqamlar 2–6 qadamlardagi taxminiy natijalardan
karta = {
    "model": {"nomi": "film-tavsiya", "versiya": "v2", "turi": REGISTR["v2"]["tur"], "egasi": "ML jamoasi"},
    "maqsad": "foydalanuvchiga shaxsiy top-10 film; biznes metrikasi — CTR va tomosha vaqti",
    "malumot": {"manba": "sintetik reytinglar (Netflix uslubida)", "bolish": "vaqt bo'yicha (oxirgi 60 kun — test)"},
    "offline_metrikalar": {"RMSE": "~0,89", "precision@10": "~0,09 (gibrid)", "coverage": "~0,13–0,21"},
    "onlayn": {"A/B": "CTR +8..+12% (simulyatsiya)", "minimal_tanlanma": "har guruhga ~25 000 seans (MDE 10%)"},
    "cheklovlar": ["sovuq start (yangi foydalanuvchi/film)", "mashhurlik siljishi", "baholash xulqi vaqt bilan o'zgaradi"],
    "monitoring": {"PSI_film": "> 0,2 -> ogohlantirish", "RMSE": "etalondan +10% -> qayta o'qitish", "p95_latency_ms": "< 50"},
    "etika": ["filtr pufagi xavfi: coverage va novelty kuzatiladi", "shaxsiy ma'lumot: faqat anonim id"],
}
print("\n=== Model kartasi (YAML) ===")
print(yaml.safe_dump(karta, allow_unicode=True, sort_keys=False).strip())

# --- Hayot sikli hisoboti: har bosqich uchun artefakt va holat
print("\n=== Hayot sikli hisoboti ===")
hisobot = pd.DataFrame([
    ("Muammo ta'rifi", "biznes metrikasi va cheklovlar", "bajarildi"),
    ("Ma'lumot", "EDA: siyraklik ~93%, uzun dum", "bajarildi"),
    ("Modellash", "bias, item-kNN, MF, gibrid", "bajarildi"),
    ("Offline baholash", "RMSE, precision@k, coverage", "bajarildi"),
    ("Onlayn tajriba", "A/B, quvvat, peeking", "bajarildi"),
    ("Joylash", "API + kanareyka 0->100%", "bajarildi" if all(r[-1] == "davom" for r in rows) else "to'xtatildi"),
    ("Monitoring", "PSI, KS, haftalik RMSE", "rejalashtirilgan (6-qadam ssenariysi)"),
    ("Qayta o'qitish", "drift signali bo'yicha", "rejalashtirilgan"),
], columns=["Bosqich", "Artefakt", "Holat"])
print(hisobot.to_string(index=False))

# --- Real dunyo misollari: ommaga ma'lum manbalar asosida umumiy taqqos (tafsilotlar kompaniya ichki ma'lumoti)
print("\n=== Real dunyo misollari (umumiy taqqos) ===")
misol = pd.DataFrame([
    ("Netflix", "tavsiya (top-N, shaxsiy sahifa)", "Netflix Prize 2006–2009: RMSE ni 10% yaxshilash tanlovi; g'olib ansamblda MF uslubidagi modellar bor edi", "A/B test — asosiy qaror vositasi"),
    ("Uber", "talab prognozi, narx, ETA", "Michelangelo ML platformasi (Uber muhandislik blogi, 2017)", "real vaqt bashorati, doimiy monitoring"),
    ("Amazon", "item-to-item tavsiya, ombor", "Linden, Smith, York, IEEE Internet Computing, 2003", "katta katalogda masshtablanish"),
], columns=["Kompaniya", "AI vazifasi", "Ommaga ma'lum manba", "Hayot siklidagi urg'u"])
print(misol.to_string(index=False))
json.dump({"karta": karta, "hisobot": hisobot.to_dict("records")}, open("hayot_sikli_hisoboti.json", "w"), ensure_ascii=False, indent=1)
print("Saqlandi: hayot_sikli_hisoboti.json")
