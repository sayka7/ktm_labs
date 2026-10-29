"""A5 — 4-qadam. Gradient to'plash va ma'lumotlar parallelligi ekvivalentligi

Bir xil 128 namunali global paket uchun uch usul taqqoslanadi: bitta katta paket; 4 ta mikro-paketda gradient to'plash (yo'qotish 1/K ga ko'paytiriladi); 2 «ishchi»da 64 tadan hisoblab gradientlarni o'rtachalash (DDP all-reduce'ning qo'lda nusxasi). Farqlar 1e-7 dan kichik chiqishi, 1/K ga bo'lmaslik esa gradientni K baravar kattalashtirishi ko'rsatiladi.

Kutilgan natija: max|gA-gB| va max|gA-gC| ~1e-8; xato variantda farq ~0.3 va norma nisbati 4.00.
"""
# 4-qadam. Gradient to'plash (accumulation) va ma'lumotlar parallelligi ekvivalentligi
# Uch usul bir xil 128 namunali global paket uchun bir xil gradient berishi kerak:
#  (A) bitta katta paket 128;  (B) 4 ta mikro-paket 32;  (C) "2 ishchi" har biri 64 — gradientlar o'rtachalanadi.
import torch, torch.nn as nn
torch.manual_seed(0)
N_FEAT, N_CLS, GB = 32, 5, 128
X = torch.randn(GB, N_FEAT); y = torch.randint(0, N_CLS, (GB,))
lossf = nn.CrossEntropyLoss()     # 'mean' — o'rtacha yo'qotish

def fresh():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(N_FEAT, 64), nn.ReLU(), nn.Linear(64, N_CLS))

def flat_grad(m):
    return torch.cat([p.grad.flatten() for p in m.parameters()])

# TODO 1: (A) to'liq paket 128 bo'yicha gradient gA
gA = None
# TODO 2: (B) K=4 mikro-paket (32 tadan), har yo'qotishni to'g'ri masshtablab gradient to'plang -> gB
gB = None
# TODO 3: (C) 2 ta bir xil boshlangan "ishchi", har biri 64 namuna; gradientlarni all-reduce kabi o'rtachalang -> gC
gC = None
# TODO 4: (B') xato variant — masshtablashsiz to'plash -> gE
gE = None

print("=== Gradient ekvivalentligi ===")
print(f"||gA|| (to'liq paket 128)            = {gA.norm():.6f}")
print(f"max|gA - gB| (to'plash 4x32)         = {(gA - gB).abs().max():.2e}")
print(f"max|gA - gC| (2 ishchi x 64, o'rtacha)= {(gA - gC).abs().max():.2e}")
print(f"max|gA - gE| (to'plash, 1/K siz!)    = {(gA - gE).abs().max():.2e}   <- ||gE||/||gA|| = {gE.norm() / gA.norm():.2f}")
