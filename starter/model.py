"""GPT variant: RoPE, RMSNorm, SwiGLU MLP, scaled residual init, weight
tying. Each change targets THIS assignment's specific constraints (2M
params, 2,000 steps, CPU, bilingual corpus) — see reasoning in our
earlier discussion. Interface (GPT(cfg), forward(idx, targets), n_params())
is unchanged so evaluate.py keeps working.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 8192     # OVERWRITTEN at runtime by the trained tokenizer
    block_size = 256
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.0
    tie_weights = True
    rope_theta = 10000.0
    norm_eps = 1e-5
    mlp_mult = 4
    use_swiglu = True


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def build_rope_cache(seq_len, head_dim, theta):
    """cos/sin tables of shape (seq_len, head_dim/2)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE pairing"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, n_head, T, head_dim). cos/sin: (T, head_dim/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.stack([rx1, rx2], dim=-1)
    return out.flatten(-2)


# ---------------------------------------------------------------------------
# Norm / MLP variants
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 3 matmuls (gate, up, down) vs. a GELU MLP's 2; shrink hidden by
        # ~2/3 so total MLP params stay roughly comparable to a 4x-GELU
        # MLP at the same n_embd (standard SwiGLU convention).
        hidden = int(round(cfg.mlp_mult * cfg.n_embd * 2 / 3))
        hidden = max(hidden - hidden % 8, 8)
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class GELUMLP(nn.Module):
    """Kept for ablation against SwiGLU (cfg.use_swiglu=False)."""
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.n_embd
        self.fc1 = nn.Linear(cfg.n_embd, hidden)
        self.fc2 = nn.Linear(hidden, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ---------------------------------------------------------------------------
# Attention / Block
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = SelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg) if cfg.use_swiglu else GELUMLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = build_rope_cache(cfg.block_size, head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # GPT-2 style scaled residual init: keeps residual-stream variance
        # from compounding with depth. Applied after the generic init.
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0,
                             std=0.05 / math.sqrt(2 * cfg.n_layer))
            down = block.mlp.down if hasattr(block.mlp, "down") else block.mlp.fc2
            nn.init.normal_(down.weight, mean=0.0,
                             std=0.05 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, (
            f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.drop(self.tok_emb(idx))
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def muon_params(self):
        """2D weight matrices strictly inside transformer blocks — eligible
        for Muon. Excludes embeddings/head and 1D params (norms)."""
        for blk in self.blocks:
            for p in blk.parameters():
                if p.ndim == 2:
                    yield p

    def adamw_params(self):
        """Everything else: token embedding (+ tied head), norm weights."""
        muon_ids = {id(p) for p in self.muon_params()}
        seen = set()
        for p in self.parameters():
            if id(p) not in muon_ids and id(p) not in seen:
                seen.add(id(p))
                yield p