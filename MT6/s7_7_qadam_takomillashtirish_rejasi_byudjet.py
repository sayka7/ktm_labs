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

idx = {m[0]: i for i, m in enumerate(M)}
cost = lambda s: sum(M[idx[c]][2] for c in s)

# TODO 1: qiymat = sum(min(yopilgan daraja, gap) x vazn) + 0.5 x sum(min(xavf kamayishi, xavf - 1))
def value(sel):
    raise NotImplementedError("TODO 1: to'plam qiymati")

# TODO 2: to'plam byudjetdan oshmasin va har choraning bog'liqliklari to'plamda bo'lsin
def feasible(sel):
    raise NotImplementedError("TODO 2: bajariluvchanlik")

# TODO 3: to'liq qidiruv: 2^13 kombinatsiyaning bajariluvchanlari ichidan eng katta qiymatlisi (teng bo'lsa arzoni)
# TODO 4: ochko'z usul: har qadamda (qiymat ortishi / narx) eng katta bajariluvchan chorani qo'shish
best, bestv = (), -1
print("=== Choralarni tanlash (byudjet {} mln so'm) ===".format(BUDGET))
print(f"Optimal (to'liq qidiruv): qiymat={value(best):.1f}")

# TODO 5: ro'yxatli rejalashtirish: bog'liqliklari tugagan choralarni TEAM ta jamoaning eng erta bo'shaganiga joylang
# TODO 6: PDCA bosqichi, Markdown yo'l xaritasi ("# Takomillashtirish yo'l xaritasi"), vaznli gap va xavf oldin/keyin
# TODO 7: Gantt diagrammasi (barh, left=start)
