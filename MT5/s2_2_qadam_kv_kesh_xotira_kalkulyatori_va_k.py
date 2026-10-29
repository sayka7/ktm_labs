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
# TODO 1: KV-kesh baytlari formulasini yozing (K va V, qatlamlar, KV boshlar, head_dim, ketma-ketlik, batch, bayt)
def kv_bytes(L, kv, hd, seq, batch, bytes_per=2):
    raise NotImplementedError("TODO 1: KV-kesh hajmi formulasi")

GB = 1024 ** 3
print("=== KV-kesh xotirasi (fp16, batch=1) ===")
print(f"{'model':<20}{'KB/token':>10}{'4k':>9}{'32k':>9}{'128k':>9}   (GB)")
# TODO 2: har model uchun token boshiga KB va 4k/32k/128k kontekstda GB ni chiqaring
for name, (L, qh, kv, hd, P) in MODELS.items():
    per_tok = kv_bytes(L, kv, hd, 1, 1) / 1024
    ...

print("\n=== 2x24 GB GPU'da sig'adigan bir vaqtdagi so'rovlar (kontekst 8k) ===")
GPU_MEM = 2 * 24 * GB; OVERHEAD = 0.10
# TODO 3: vaznlar (parametr x bayt) ni ayirib, qolgan joyga 8192 tokenli KV-keshdan nechta sig'ishini hisoblang (fp16 va int4)

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
            # TODO 4: keshda oldingi K/V bo'lsa, yangi k, v ni ular bilan vaqt o'qi (dim=2) bo'yicha ulang
            #         va keshni yangilang; bitta yangi token uchun causal maska kerakmi — o'ylang
            raise NotImplementedError("TODO 4: KV-keshni yangilash")
        causal = True
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
    # TODO 5: use_cache=False — har qadamda butun ketma-ketlikni modeldan o'tkazing;
    #         use_cache=True — promptni bir marta o'tkazib, keyin faqat YANGI tokenni (to'g'ri pozitsiya bilan) bering
    raise NotImplementedError("TODO 5: keshli va keshsiz greedy generatsiya")

m = TinyLM().eval()
prompt = torch.randint(0, VOCAB, (1, 64))
a = generate(m, prompt, 32, False); b = generate(m, prompt, 32, True)
print("\n=== Kichik decoder: KV-kesh bilan / keshsiz ===")
print("Natija tokenlari bir xilmi:", bool(torch.equal(a, b)))
# TODO 6: n_new = 32, 64, 128, 256 uchun ikkala usul vaqtini time.perf_counter() bilan o'lchab, tezlanishni chiqaring
