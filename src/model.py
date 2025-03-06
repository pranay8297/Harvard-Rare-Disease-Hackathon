import torch
import random
import numpy as np

from einops import rearrange
from torch import nn
from torch.nn import functional as F


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Initialize linear and embedding weights with Kaiming initialization
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        # Initialize normalization layer parameters
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class MultiHeadAttention(nn.Module): # Verified

    def __init__(self, d_model, nhead):

        super().__init__()
        assert d_model % nhead == 0

        self.kqv = nn.Linear(d_model, 3*d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.nhead = nhead

    def forward(self, x, attention_mask):

        # x -> (b, s, e) -> (b s, h, e/h)
        kqv = self.kqv(x) # (b, s, 3*d_model)
        k, q, v = torch.chunk(kqv, 3, dim = -1) # (3, b, s, d_model)

        k = rearrange(k, 'b s (h e) -> b h s e', h = self.nhead)
        q = rearrange(q, 'b s (h e) -> b h s e', h = self.nhead)
        v = rearrange(v, 'b s (h e) -> b h s e', h = self.nhead)

        # Attention claculation - # TODO: make is_casual true in case of finetuning - Very important
        y = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask[:, None, None, :], is_causal = False) # flash attention
        y = rearrange(y, 'b h s e -> b s (h e)', h = self.nhead)

        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, d_model):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.layer(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        # Self Attn
        # MLP
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.mlp = MLP(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn = self.self_attn(x, mask)
        x = self.norm1(x + attn)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x

class Net(nn.Module):
    def __init__(self, vocab_size, emb_dim = 512, n_classes = 12687, n_layers = 8, n_heads = 16):

        super(Net, self).__init__()

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.transformers = nn.ModuleList([EncoderLayer(d_model = emb_dim, nhead = n_heads) for _ in range(n_layers)])
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x, mask):
        # breakpoint()
        x = self.embedding(x) # (b, s, e)
        x = self.norm(x) # (b, s, e)
        mask = mask.bool()
        for transformer in self.transformers:
            x = transformer(x, mask)

        x = x[:, 0, :].squeeze() # (b, s, e) -> (b, e)
        x = self.classifier(x) # (b, e) -> (b, num_classes)
        return x

class POCActor(nn.Module):
    def __init__(self, vocab_size, emb_dim = 512, n_classes = 12687, n_layers = 8, n_heads = 16):

        super().__init__()

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.transformers = nn.ModuleList([EncoderLayer(d_model = emb_dim, nhead = n_heads) for _ in range(n_layers)])

        self.final_norm = nn.LayerNorm(emb_dim)

        self.emb_transform_mean = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(1024, 512)
        )
        self.emb_transform_log_std = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(1024, 512)
        )

    def load_backbone(self, path = './drive/MyDrive/hrd_hack/model_checkpoint_epoch_99.pt'):
        checkpoint = torch.load(path, map_location = 'cpu')['model_state_dict']
        self_state_dict = self.state_dict()
        for k, v in checkpoint.items():
            if k in self_state_dict:
                if v.shape == self_state_dict[k].shape:
                    self_state_dict[k].copy_(v)

    def adjust_log_std(self, log_std):
        log_std_min, log_std_max = (-5, 2)  # From SpinUp / Denis Yarats
        return log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

    def forward(self, x, mask):
        x = self.embedding(x) # (b, s, e)
        x = self.norm(x) # (b, s, e)
        mask = mask.bool()
        for transformer in self.transformers:
            x = transformer(x, mask)

        x = x[:, 0, :].squeeze() # (b, s, e) -> (b, e)

        x = self.final_norm(x).squeeze() # (b, e)

        mean = self.emb_transform_mean(x) # (b, e) -> (b, e)

        log_std = self.emb_transform_log_std(x) # (b, e) -> (b, e)
        log_std = torch.tanh(log_std)
        return mean, self.adjust_log_std(log_std)

    def get_action(self, x, mask):
        mean, log_std = self.forward(x, mask)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        log_prob = normal.log_prob(x_t)
        return x_t, log_prob

class POCCritic(nn.Module):
    def __init__(self, vocab_size, emb_dim = 512, n_classes = 12687, n_layers = 8, n_heads = 16):

        super().__init__()

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.disease_transform = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(1024, 512)
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.transformers = nn.ModuleList([EncoderLayer(d_model = emb_dim, nhead = n_heads) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(emb_dim)
        self.q_value = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.GELU(approximate = 'tanh'),
            nn.Linear(1024, 1)
        )

    def forward(self, x, mask, d_emb):
        # breakpoint()

        x = self.embedding(x) # (b, s, e)

        d_emb = self.disease_transform(d_emb) # (b, e)
        d_emb = d_emb.unsqueeze(1) # (b, s + 1, e)
        d_emb_mask = torch.ones(x.shape[0], 1)

        mask = torch.cat([mask, d_emb_mask], dim = -1)# (b, s+1, 1)
        x = torch.cat([x, d_emb], dim = 1)

        x = self.norm(x) # (b, s+1, e)
        mask = mask.bool()
        for transformer in self.transformers:
            x = transformer(x, mask)

        x = x[:, 0, :].squeeze() # (b, s, e) -> (b, e)
        x = self.final_norm(x)
        return self.q_value(x)

