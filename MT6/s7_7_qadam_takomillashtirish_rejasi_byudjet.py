"""MT6 — 7-qadam. Takomillashtirish rejasi: byudjet ostida optimal choralar, bog'liqliklar bilan Gantt va PDCA

13 chora (narx, muddat, qaysi gap va xavflarni yopishi, bog'liqliklar) orasidan 250 mln so'm (shartli) byudjetda eng katta qiymat beradigan to'plam to'liq qidiruv (8192 kombinatsiya) bilan topiladi va ochko'z usul bilan taqqoslanadi. Tanlangan choralar 2 jamoa va bog'liqliklarni hisobga olib haftalarga joylashtiriladi (ro'yxatli rejalashtirish), PDCA bosqichiga ajratiladi; Markdown yo'l xaritasi, vaznli gap va xavf «oldin/keyin», KPI ro'yxati va Gantt diagrammasi chiqadi.

Kutilgan natija: Optimal to'plam qiymati ~94,5, ochko'z usul ~77,5 (bog'liqlik tufayli M2 ni o'tkazib yuboradi); reja ~14 hafta, vaznli gap ~80% yopiladi, R01 va R07 ishtahadan yuqori qoladi.
"""
# 7-qadam. Takomillashtirish rejasi: byudjet ostida choralarni optimal tanlash, bog'liqliklar bilan jadval (Gantt), PDCA
# Kirish: 6-qadamdagi bo'shliqlar (gap) va 1-qadamdagi xavflar — bu yerda qisqa nusxasi berilgan (qadam mustaqil).
import itertools, numpy as np, pandas as pd
import matplotlib.pyplot as plt

GAP = {"A5": (4, 3), "E73": (4, 3), "Q6": (3, 3), "Q8": (3, 3), "R1": (3, 3), "E14": (3, 3), "E17": (3, 3), "A6": (2, 3),
       "A7": (2, 3), "E15": (2, 3), "Q9": (2, 2), "A8": (2, 2), "Q10": (2, 2)}            # id: (gap, vazn)
RISK = {"R01": 12, "R03": 12, "R05": 12, "R07": 15, "R08": 12, "R11": 12, "R10": 9}       # boshlang'ich xavf (L x I)
M = [  # (kod, chora, narx (mln so'm, shartli), hafta, yopadigan gap {id: daraja}, xavf kamayishi {id: ball}, bog'liqlik)
 ("M1", "AI siyosati va AIMS doirasini tasdiqlash", 20, 3, {"Q10": 1}, {}, []),
 ("M2", "ISO/IEC 23894 xavf jarayoni + xavf reyestri vositasi", 45, 5, {"Q6": 3, "R1": 3}, {"R08": 4}, ["M1"]),
 ("M3", "AI ta'sirini baholash (A.5 / FRIA) metodikasi", 40, 4, {"A5": 4, "Q8": 2}, {"R01": 2}, ["M1"]),
 ("M4", "Jiddiy hodisalar va eskalatsiya tartibi", 15, 2, {"E73": 4}, {}, ["M1"]),
 ("M5", "Rad etishda inson tekshiruvi + to'xtatish tugmasi", 35, 4, {"E14": 3}, {"R05": 8, "R11": 4}, []),
 ("M6", "Fairness audit + reweighing konveyerga", 30, 3, {"A7": 1}, {"R01": 6}, ["M2"]),
 ("M7", "Drift monitoringi va qayta o'qitish siyosati", 40, 4, {"Q9": 2, "A6": 1}, {"R03": 6}, []),
 ("M8", "Mustahkamlik/adversarial test to'plami (CI)", 35, 3, {"E15": 2, "A6": 1}, {"R10": 5}, []),
 ("M9", "Ma'lumot kartalari va kelib chiqish (lineage)", 25, 3, {"A7": 1}, {}, []),
 ("M10", "Sifat menejmenti tizimi (QMS) hujjatlari", 60, 6, {"E17": 3, "Q10": 1}, {}, ["M1", "M2"]),
 ("M11", "Differensial maxfiylik / kirishni cheklash", 50, 5, {}, {"R07": 7}, []),
 ("M12", "Foydalanuvchiga tushuntirish xati (SHAP)", 20, 2, {"A8": 2}, {"R11": 2}, ["M8"]),
 ("M13", "Zaxira model va tashqi API SLA", 30, 3, {}, {"R08": 5}, []),
]
BUDGET, TEAM = 250, 2          # mln so'm; bir vaqtda ishlaydigan jamoalar soni

def value(sel):
    """Qiymat = yopilgan gap x vazn (gapdan oshmaydi) + 0.5 x kamaygan xavf ballari."""
    closed = {}
    for code in sel:
        for g, d in M[idx[code]][4].items(): closed[g] = closed.get(g, 0) + d
    v = sum(min(d, GAP[g][0]) * GAP[g][1] for g, d in closed.items())
    red = {}
    for code in sel:
        for r, d in M[idx[code]][5].items(): red[r] = red.get(r, 0) + d
    return v + 0.5 * sum(min(d, RISK[r] - 1) for r, d in red.items())
idx = {m[0]: i for i, m in enumerate(M)}
def feasible(sel):
    s = set(sel)
    return sum(M[idx[c]][2] for c in s) <= BUDGET and all(d in s for c in s for d in M[idx[c]][6])

# ---- 1) To'liq qidiruv (2^13 = 8192 kombinatsiya) — aniq optimum ----
best, bestv = (), -1
for k in range(len(M) + 1):
    for sel in itertools.combinations([m[0] for m in M], k):
        if feasible(sel):
            v = value(sel)
            if v > bestv or (v == bestv and sum(M[idx[c]][2] for c in sel) < sum(M[idx[c]][2] for c in best)): best, bestv = sel, v
# ---- 2) Ochko'z (greedy) usul: qiymat/narx bo'yicha, bog'liqlik va byudjetni hisobga olib ----
greedy, left = [], set(m[0] for m in M)
while True:
    cand = [(value(greedy + [c]) - value(greedy)) / M[idx[c]][2] for c in left if feasible(greedy + [c])]
    opts = [c for c in left if feasible(greedy + [c])]
    if not opts: break
    c = opts[int(np.argmax(cand))]
    if value(greedy + [c]) - value(greedy) <= 0: break
    greedy.append(c); left.remove(c)
cost = lambda s: sum(M[idx[c]][2] for c in s)
print("=== Choralarni tanlash (byudjet {} mln so'm) ===".format(BUDGET))
print(f"Optimal (to'liq qidiruv): {sorted(best, key=lambda c: idx[c])}  qiymat={bestv:.1f}  narx={cost(best)}")
print(f"Ochko'z usul:             {sorted(greedy, key=lambda c: idx[c])}  qiymat={value(greedy):.1f}  narx={cost(greedy)}")
allv = value([m[0] for m in M]); print(f"Hamma choralar (byudjetsiz): qiymat={allv:.1f}  narx={cost([m[0] for m in M])}")

# ---- 3) Jadval: bog'liqliklar + TEAM ta parallel jamoa (ro'yxatli rejalashtirish) ----
sel = sorted(best, key=lambda c: idx[c]); start, end, free = {}, {}, [0] * TEAM
pending = list(sel)
while pending:
    ready = [c for c in pending if all(d in end for d in M[idx[c]][6] if d in sel)]
    c = min(ready, key=lambda c: (max([end[d] for d in M[idx[c]][6] if d in end] + [0]), idx[c]))   # eng erta boshlanadigani
    t_dep = max([end[d] for d in M[idx[c]][6] if d in end] + [0]); j = int(np.argmin(free))
    start[c] = max(free[j], t_dep); end[c] = start[c] + M[idx[c]][3]; free[j] = end[c]; pending.remove(c)
PDCA = lambda c: "Plan" if c in ("M1", "M2", "M3") else ("Check" if c in ("M7", "M8") else "Do")

# ---- 4) Natija: muvofiqlik va xavf "oldin/keyin" ----
closed = {}
for c in sel:
    for g, d in M[idx[c]][4].items(): closed[g] = closed.get(g, 0) + d
gap_before = sum(g * w for g, w in GAP.values()); gap_after = sum(max(0, GAP[g][0] - closed.get(g, 0)) * GAP[g][1] for g in GAP)
red = {}
for c in sel:
    for r, d in M[idx[c]][5].items(): red[r] = red.get(r, 0) + d
risk_after = {r: max(1, v - red.get(r, 0)) for r, v in RISK.items()}
print(f"\n# Takomillashtirish yo'l xaritasi (jami {max(end.values())} hafta, {TEAM} jamoa)")
print("| kod | chora | PDCA | boshlanish (hafta) | tugash | narx |\n|---|---|---|---|---|---|")
for c in sorted(sel, key=lambda c: start[c]):
    print(f"| {c} | {M[idx[c]][1]} | {PDCA(c)} | {start[c]} | {end[c]} | {M[idx[c]][2]} |")
print(f"\nVaznli gap: {gap_before} -> {gap_after} ({1 - gap_after / gap_before:.0%} yopildi)")
print(f"Xavflar yig'indisi: {sum(RISK.values())} -> {sum(risk_after.values())};  ishtahadan (8) yuqori qolganlar: {[r for r, v in risk_after.items() if v > 8] or 'yo`q'}")
print("Kuzatiladigan KPI: ochiq yuqori xavflar soni, DI >= 0.8, drift ogohlantirishiga javob <= 48 soat, audit nomuvofiqliklari soni")
print("Tashlab ketilgan choralar:", [c for c in idx if c not in sel])

fig, ax = plt.subplots(figsize=(9, 3.6)); col = {"Plan": "#2980b9", "Do": "#27ae60", "Check": "#e67e22"}
for i, c in enumerate(sorted(sel, key=lambda c: start[c])):
    ax.barh(i, end[c] - start[c], left=start[c], color=col[PDCA(c)]); ax.text(start[c] + 0.1, i, c, va="center", fontsize=7, color="white")
ax.set_yticks([]); ax.set_xlabel("hafta"); ax.set_title("Yo'l xaritasi (Gantt): ko'k=Plan, yashil=Do, to'q sariq=Check"); ax.grid(axis="x", alpha=.3)
plt.tight_layout()
print("\nIzoh: real rejada narx va muddatlar smeta asosida, qiymat esa xavf egalari bilan kelishilgan holda beriladi;")
print("reja ISO/IEC 42001 10-bandi (uzluksiz takomillashtirish) bo'yicha har chorakda qayta ko'rib chiqiladi.")
