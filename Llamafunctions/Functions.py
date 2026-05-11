import torch
import torch.nn as nn
import math


import torch
import math

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_RoPE(q, k, start_pos, RoPE_theta=500000.0, RoPE_scaling=None):
    B, H, T, D = q.shape
    device, dtype = q.device, q.dtype

    inv_freq = 1.0 / (
        RoPE_theta ** (torch.arange(0, D, 2, device=device, dtype=dtype) / D)
    )

    if RoPE_scaling is not None:
        factor = RoPE_scaling["factor"]
        low_freq_factor = RoPE_scaling["low_freq_factor"]
        high_freq_factor = RoPE_scaling["high_freq_factor"]
        old_context_len = RoPE_scaling["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)

        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama

        is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
        inv_freq = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    pos = torch.arange(start_pos, start_pos + T, device=device, dtype=dtype)
    freqs = torch.outer(pos, inv_freq)   # [T, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

def rep_kv(x, n_rep):
    return x.repeat_interleave(n_rep, dim = 1)
    

#---------------RMSNorm-------------------#
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x
        
#--------------Attention-------------------#
class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, RoPE_scaling):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.RoPE_scaling = RoPE_scaling
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        
        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
    def forward(self, x, start_pos, K_cache = None, V_cache = None):
        
        B, T, C = x.shape
        
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)                # split into heads
        K = K.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)        
        
        Q, K = apply_RoPE(Q, K, start_pos, RoPE_theta = 500000, RoPE_scaling = self.RoPE_scaling)
        
        if K_cache is not None:
            K = torch.cat([K_cache, K], dim=2)
            V = torch.cat([V_cache, V], dim=2)
                
        K_exp = rep_kv(K, self.n_rep)
        V_exp = rep_kv(V, self.n_rep)
        
        scores = (Q @ K_exp.transpose(-2, -1))/math.sqrt(self.head_dim)                         #score calculation
        
        if K_cache is None:
            mask = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
            mask =  mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        weights = torch.softmax(scores, dim =-1)
        out = weights @ V_exp                                                       #weighted sum
        
        out = out.transpose(1,2).contiguous().view(B, T, C)                       #merging heads
        
        return self.wo(out), K, V
        
#--------------MLP(with SwiGLU)-------------#
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias = False)
        self.w_down = nn.Linear(hidden_dim, dim, bias = False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias = False)
        
    def forward(self, x):
        return self.w_down(torch.nn.functional.silu(self.w_gate(x))*self.w_up(x))                        #silu is x*sigmoid(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, RoPE_scaling, hidden_dim):
        super().__init__()
        
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, n_kv_heads, RoPE_scaling)
        
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FFN(dim, hidden_dim)
       
    def forward(self, x, start_pos, kcache = None, vcache = None):
        attn_output, new_kcache, new_vcache = self.attn(self.attn_norm(x), start_pos, kcache, vcache)
        
        x = x + attn_output
        
        x = x + self.ffn(self.ffn_norm(x))
        
        return x, new_kcache, new_vcache
        
class MiniLlama(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, RoPE_scaling, hidden_dim):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, RoPE_scaling, hidden_dim)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids, start_pos = 0, kv_cache = None):
        x = self.embed(input_ids)
        
        new_kv_cache = []
        cur_pos = start_pos
        
        for i,layer in enumerate(self.layers):
            k_temp = None
            v_temp = None
            
            if kv_cache is not None:
                k_temp, v_temp = kv_cache[i]
            
            x, new_k, new_v = layer(x, cur_pos, k_temp, v_temp)
            new_kv_cache.append((new_k, new_v))

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_cache