import torch
import torch.nn as nn
import math



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
        
        scores = (q @ k.transpose(-2, -1))/math.sqrt(self.head_dim)                         #score calculation
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
        
    def forward(self,x):
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