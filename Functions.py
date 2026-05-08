import torch
import torch.nn as nn
import math


def rotate_half(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
    
def apply_RoPE(q, k):
    B, H, T, D = q.shape
    
    freqs = torch.arange(0,D,2).float()
    freqs = 1.0/(500000**(freqs/D))
    
    pos = torch.arange(T).float()
    
    angles = pos[:, None]*freqs[None, :]
    
    sine = torch.sin(angles)
    cosine = torch.cos(angles)
    
    sine = sine[None, None, :, :]
    cosine = cosine[None, None, :, :]
    
    q = (q * cosine.repeat_interleave(2, dim=-1)) + (rotate_half(q)*sine.repeat_interleave(2, dim=-1))
    k = (k * cosine.repeat_interleave(2, dim=-1)) + (rotate_half(k)*sine.repeat_interleave(2, dim=-1))
    
    return q, k

#---------------RMSNorm-------------------#
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x
        
#--------------Attention-------------------#
class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        
        B, T, C = x.shape
        
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)                # split into heads
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)        
        
        Q, K = apply_RoPE(Q, K)
        
        scores = (Q @ K.transpose(-2, -1))/math.sqrt(self.head_dim)                         #score calculation
        mask = torch.tril(torch.ones(T,T))
        mask =  mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill( mask == 0, float('-inf'))
        
        weights = torch.softmax(scores, dim =-1)
        out = weights @ V                                                       #weighted sum
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)                       #merging heads
        
        return self.wo(out)
        
#--------------MLP(with SwiGLU)-------------#
class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias = False)
        self.w_down = nn.Linear(hidden_dim, dim, bias = False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias = False)
        
    def forward(self, x):
        return self.w_down(torch.nn.functional.silu(self.w_up(x))*self.w_gate(x))                        #silu is x*sigmoid(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, n_heads)
        
        self.ffn_norm = RMSNorm(dim)
        self.ffn = FFN(dim, hidden_dim)
       
    def forward(self, x):
        
        x = x + self.attn(self.attn_norm(x))
        
        x = x + self.ffn(self.ffn_norm(x))
        
        return x
        
class MiniLlama(nn.Module):
    def __init__(self, vocab_size, dim, n_layers, n_heads, hidden_dim):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)