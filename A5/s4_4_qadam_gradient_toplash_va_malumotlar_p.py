"""A5 — 4-qadam. Gradient to'plash va ma'lumotlar parallelligi ekvivalentligi

Bir xil 128 namunali global paket uchun uch usul taqqoslanadi: bitta katta paket; 4 ta mikro-paketda gradient to'plash (yo'qotish 1/K ga ko'paytiriladi); 2 «ishchi»da 64 tadan hisoblab gradientlarni o'rtachalash (DDP all-reduce'ning qo'lda nusxasi). Farqlar 1e-7 dan kichik chiqishi, 1/K ga bo'lmaslik esa gradientni K baravar kattalashtirishi ko'rsatiladi.

Kutilgan natija: max|gA-gB| va max|gA-gC| ~1e-8; xato variantda farq ~0.3 va norma nisbati 4.00.
"""
# 4-qadam. Gradient to'plash (accumulation) va ma'lumotlar parallelligi ekvivalentligi
# Uch usul bir xil 128 namunali global paket uchun bir xil gradient berishi kerak:
#  (A) bitta katta paket 128;  (B) 4 ta mikro-paket 32 — gradient to'planadi;
#  (C) "2 ishchi" har biri 64 — gradientlar o'rtachalanadi (DDP all-reduce ning qo'lda nusxasi).
import torch, torch.nn as nn
torch.manual_seed(0)
N_FEAT, N_CLS, GB = 32, 5, 128
X = torch.randn(GB, N_FEAT); y = torch.randint(0, N_CLS, (GB,))
lossf = nn.CrossEntropyLoss()     # 'mean' — o'rtacha yo'qotish, shuning uchun bo'laklar vazni muhim

def fresh():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(N_FEAT, 64), nn.ReLU(), nn.Linear(64, N_CLS))

def flat_grad(m):
    return torch.cat([p.grad.flatten() for p in m.parameters()])

# (A) to'liq paket
mA = fresh(); lossf(mA(X), y).backward(); gA = flat_grad(mA)

# (B) gradient to'plash: 4 x 32, har mikro-paket yo'qotishi 1/4 ga ko'paytiriladi
mB = fresh(); K = 4
for i in range(K):
    xb, yb = X[i*32:(i+1)*32], y[i*32:(i+1)*32]
    (lossf(mB(xb), yb) / K).backward()        # .grad ichida yig'iladi — zero_grad chaqirilmaydi
gB = flat_grad(mB)

# (C) ma'lumotlar parallelligi: 2 ishchi, har biri 64 namuna, so'ng all-reduce (o'rtacha)
workers = [fresh(), fresh()]                   # DDP broadcast — bir xil boshlang'ich og'irliklar
grads = []
for r, m in enumerate(workers):
    lossf(m(X[r*64:(r+1)*64]), y[r*64:(r+1)*64]).backward()
    grads.append(flat_grad(m))
gC = torch.stack(grads).mean(0)                # all-reduce(SUM)/world_size

# (B') xato variant: 1/K ga bo'lmasdan to'plash — gradient K baravar katta chiqadi
mE = fresh()
for i in range(K):
    lossf(mE(X[i*32:(i+1)*32]), y[i*32:(i+1)*32]).backward()
gE = flat_grad(mE)

print("=== Gradient ekvivalentligi ===")
print(f"||gA|| (to'liq paket 128)            = {gA.norm():.6f}")
print(f"max|gA - gB| (to'plash 4x32)         = {(gA - gB).abs().max():.2e}")
print(f"max|gA - gC| (2 ishchi x 64, o'rtacha)= {(gA - gC).abs().max():.2e}")
print(f"max|gA - gE| (to'plash, 1/K siz!)    = {(gA - gE).abs().max():.2e}   <- xato: ||gE||/||gA|| = {gE.norm() / gA.norm():.2f}")
print("\nXulosa: 1 x 128 == 4 x 32 (to'plash) == 2 ishchi x 64 (DDP). Shuning uchun")
print("GPU yetmasa gradient to'plash bilan katta global paketni taqlid qilish mumkin, aloqa chastotasi esa kamayadi.")
print("Eslatma: BatchNorm bo'lsa ekvivalentlik buziladi (statistika mikro-paketga bog'liq).")
