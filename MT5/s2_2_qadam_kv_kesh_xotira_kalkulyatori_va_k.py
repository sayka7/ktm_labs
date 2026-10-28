"""MT5 — 2-qadam. KV-kesh xotira kalkulyatori va keshning generatsiya tezligiga ta'siri

KV-kesh formulasi `2 x qatlam x KV_bosh x head_dim x ketma-ketlik x batch x bayt` ochiq e'lon qilingan konfiguratsiyalar (Llama-2-7B MHA, Llama-3-8B va 70B GQA) uchun hisoblanadi: token boshiga KB, 4k/32k/128k kontekstda GB, 2x24 GB GPU'da 8k kontekst bilan nechta so'rov sig'ishi (fp16 va int4 vaznlar). Keyin 4 qatlamli kichik decoder CPU'da KV-kesh bilan va keshsiz generatsiya qilinadi — natija tokenlari bir xilligi va tezlanish o'lchanadi.

Kutilgan natija: GQA (8 KV bosh) KV-keshni MHA ga nisbatan 4 baravar kamaytiradi (128 vs 512 KB/token); 70B fp16 2x24 GB ga sig'maydi. Kichik decoderda KV-kesh 256 token uchun ~10x tezlanish beradi (CPU'ga qarab o'zgaradi).
"""
# 2-qadam. KV-kesh xotira kalkulyatori va kichik transformerda KV-keshning tezlikka ta'siri
# KV-kesh hajmi = 2 (K va V) x qatlamlar x KV_boshlar x bosh_o'lchami x ketma-ketlik x batch x bayt
import time, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
torch.backends.nnpack.set_flags(False); torch.manual_seed(0); torch.set_num_threads(2)

# Ochiq e'lon qilingan konfiguratsiyalar (model kartalari asosida, taxminiy):
MODELS = {
    #  nom               qatlam  q_bosh  kv_bosh  head_dim  parametr(mlrd)
    "Llama-2-7B (MHA)":  (32,    32,     32,      128,      6.7),
    "Llama-3-8B (GQA)":  (32,    32,     8,       128,      8.0),
    "Llama-3-70B (GQA)": (80,    64,     8,       128,      70.6),
}
def kv_bytes(L, kv, hd, seq, batch, bytes_per=2):
    return 2 * L * kv * hd * seq * batch * bytes_per

GB = 1024 ** 3
print("=== KV-kesh xotirasi (fp16, batch=1) ===")
print(f"{'model':<20}{'KB/token':>10}{'4k':>9}{'32k':>9}{'128k':>9}   (GB)")
for name, (L, qh, kv, hd, P) in MODELS.items():
    per_tok = kv_bytes(L, kv, hd, 1, 1) / 1024
    print(f"{name:<20}{per_tok:>10.0f}" + "".join(f"{kv_bytes(L, kv, hd, s, 1)/GB:>9.2f}" for s in (4096, 32768, 131072)))

print("\n=== 2x24 GB GPU'da sig'adigan bir vaqtdagi so'rovlar (kontekst 8k) ===")
GPU_MEM = 2 * 24 * GB; OVERHEAD = 0.10            # 10% aktivatsiya/fragmentatsiya uchun zaxira (taxmin)
for name, (L, qh, kv, hd, P) in MODELS.items():
    for wbytes, wname in [(2, "fp16"), (0.5, "int4")]:
        weights = P * 1e9 * wbytes
        free = GPU_MEM * (1 - OVERHEAD) - weights
        per_req = kv_bytes(L, kv, hd, 8192, 1)
        n = int(free // per_req) if free > 0 else 0
        print(f"{name:<20} vazn {wname}: {weights/GB:6.1f} GB  bo'sh {max(free,0)/GB:6.1f} GB  -> {n:4d} so'rov")

# ---- Kichik decoder: KV-kesh bilan va keshsiz generatsiya ----
D, H, LAYERS, VOCAB = 128, 4, 4, 500
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D)
        s.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)); s.n1 = nn.LayerNorm(D); s.n2 = nn.LayerNorm(D)
    def forward(s, x, cache=None):
        B, T, _ = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        if cache is not None:
            if "k" in cache:                       # oldingi tokenlar K/V qayta hisoblanmaydi
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        causal = cache is None or T > 1
        a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        x = x + s.o(a.transpose(1, 2).reshape(B, T, D))
        return x + s.ff(s.n2(x))
class TinyLM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(VOCAB, D); s.pos = nn.Embedding(1024, D)
        s.blocks = nn.ModuleList(Block() for _ in range(LAYERS)); s.head = nn.Linear(D, VOCAB)
    def forward(s, ids, start=0, caches=None):
        x = s.emb(ids) + s.pos(torch.arange(start, start + ids.shape[1]))
        for i, b in enumerate(s.blocks):
            x = b(x, None if caches is None else caches[i])
        return s.head(x)

@torch.no_grad()
def generate(m, prompt, n_new, use_cache):
    ids = prompt.clone()
    if use_cache:
        caches = [dict() for _ in range(LAYERS)]
        logits = m(ids, 0, caches)
        for _ in range(n_new):
            nxt = logits[:, -1].argmax(-1, keepdim=True); ids = torch.cat([ids, nxt], 1)
            logits = m(nxt, ids.shape[1] - 1, caches)      # faqat YANGI token
    else:
        for _ in range(n_new):
            logits = m(ids)                                 # butun ketma-ketlik qayta hisoblanadi
            ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return ids

m = TinyLM().eval()
prompt = torch.randint(0, VOCAB, (1, 64))
a = generate(m, prompt, 32, False); b = generate(m, prompt, 32, True)
print("\n=== Kichik decoder: KV-kesh bilan / keshsiz ===")
print("Natija tokenlari bir xilmi:", bool(torch.equal(a, b)))
rows = []
for n_new in (32, 64, 128, 256):
    t = time.perf_counter(); generate(m, prompt, n_new, False); t0 = time.perf_counter() - t
    t = time.perf_counter(); generate(m, prompt, n_new, True); t1 = time.perf_counter() - t
    rows.append((n_new, t0, t1))
    print(f"yangi token={n_new:4d}  keshsiz {t0*1000:7.0f} ms   KV-kesh {t1*1000:6.0f} ms   tezlanish x{t0/t1:4.1f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
seq = [2 ** i for i in range(10, 18)]
for name, (L, qh, kv, hd, P) in MODELS.items():
    ax[0].plot(seq, [kv_bytes(L, kv, hd, s, 1) / GB for s in seq], "-o", label=name)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("kontekst (token)"); ax[0].set_ylabel("KV-kesh, GB"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].plot([r[0] for r in rows], [r[1] * 1000 for r in rows], "-o", label="keshsiz")
ax[1].plot([r[0] for r in rows], [r[2] * 1000 for r in rows], "-s", label="KV-kesh")
ax[1].set_xlabel("yangi tokenlar"); ax[1].set_ylabel("ms"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout()
print("\nIzoh: real serverlarda (vLLM PagedAttention va h.k.) KV-kesh sahifalab boshqariladi; xotira odatda")
print("vaznlardan keyingi asosiy cheklov — GQA, int8/fp8 KV-kesh va kontekstni qisqartirish uni kamaytiradi.")
