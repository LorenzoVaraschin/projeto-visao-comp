import torch
from torch import nn
import math 
import numpy as np
from timm.models.layers import trunc_normal_

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, num_frames, N, masked, attn_type): 
        super().__init__()
        assert d_model % h == 0, f"d_model needs to be a multiple of h, but got d_model={d_model} | h={h}"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = self.d_k 
        self.masked = masked
        self.num_frames = num_frames
        self.N = N #number of patches per frame
        self.attn = attn_type
        
        self.W_key = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_query = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_value = nn.Linear(in_features=d_model, out_features=d_model)
        
        self.softmax = nn.Softmax(dim=-1)
        self.linear_head = nn.Linear(in_features=d_model, out_features=d_model)
        
    def forward(self, query, key, value): 
        queries = self.W_query(query).reshape(query.shape[0], query.shape[1], self.h, self.d_k).permute(0, 2, 1, 3)
        keys = self.W_key(key).reshape(key.shape[0], key.shape[1], self.h, self.d_k).permute(0, 2, 1, 3)
        values = self.W_value(value).reshape(value.shape[0], value.shape[1], self.h, self.d_v).permute(0, 2, 1, 3)

        dp_scores = torch.matmul(queries, keys.mT) / math.sqrt(self.d_k)

        # apply causal mask (only for the decoder self-attention)
        if self.masked == False:
            attention = self.softmax(dp_scores)
        else:
            attention = self.softmax(self.apply_causal_mask(dp_scores))

        # b, num_heads, seq_length(query), d_k
        output = torch.matmul(attention, values).transpose(1, 2).reshape(query.shape[0], query.shape[1], self.h*self.d_k)
        output = self.linear_head(output)
        return queries, keys, values, attention, output 

    def apply_causal_mask(self, dp_scores):
        mask = torch.ones(dp_scores.shape[2], dp_scores.shape[3])
        if self.attn == "temporal":
            mask = torch.triu(mask, diagonal=1)
        elif self.attn == "space_time":
            for i in range(self.num_frames):
                mask[i*self.N:, i*self.N:(i+1)*self.N] = 0
        elif self.attn == "spatial":
            return dp_scores
        dp_scores[:, :, mask == 1] = -np.inf
        return dp_scores 

class GLU(nn.Module):
    def __init__(self, d_model, d_hidden, dropout): 
        super().__init__()
        self.in_linear1 = nn.Linear(in_features=d_model, out_features=d_hidden)
        self.in_linear2 = nn.Linear(in_features=d_model, out_features=d_hidden)
        self.out_linear = nn.Linear(in_features=d_hidden, out_features=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.silu = nn.SiLU()


    def forward(self, x):
        x1 = self.in_linear1(x)
        x2 = self.silu(self.in_linear2(x))
        x = self.dropout1(x1*x2)
        out = self.out_linear(x)
        return self.dropout2(out)

class TransformerBlock(nn.Module):
    def __init__(self, h, d_model, d_hidden, attn_type, num_frames, N, transformer_type, dropout=0.1):
        super().__init__()

        if transformer_type == "encoder_only":
            masked = False
        elif transformer_type == "decoder_only":
            masked = True
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.sublayer1 = MultiHeadAttention(h=h, d_model=d_model, num_frames=num_frames, N=N, masked=masked, attn_type=attn_type)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.sublayer2 = GLU(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x_norm = self.layer_norm1(x)
        x = self.sublayer1(x_norm, x_norm, x_norm)[-1] + x
        x_norm = self.layer_norm2(x)
        output = self.sublayer2(x_norm) + x
        return output

# PREDFORMER ORIGINAL POSITIONAL ENCODING
def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe = pe.unsqueeze(0)
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return pe
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_frames, num_patches, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(sinusoidal_embedding(num_frames * num_patches, d_model),
                                               requires_grad=False).view(1, num_frames, num_patches, d_model)
        
    def forward(self, x):
        if self.pos_embedding.device != x.device:
            self.pos_embedding = self.pos_embedding.to(x.device)
        x = x + self.pos_embedding
        return x

class LinearPatchEncoder(nn.Module):
    def __init__(self, d_model, patch_size, image_size, num_frames=10):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_frames= num_frames
        self.n_h = image_size // patch_size
        self.n_w = image_size // patch_size

        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.norm_ = nn.LayerNorm(d_model)

    def forward(self, x):
        # here, x is the batched image_sequences
        # reshape x (B, num_frames, 3, H, W) -> (B*num_frames, 3, H, W)
        batch_size = x.shape[0]
        x = x.reshape(batch_size*self.num_frames, 3, self.image_size, self.image_size)
        patch_embeddings = self.conv_proj(x)
        # (B*num_frames, d_model, n_h, n_w) -> (B, num_frames, d_model, n_h*n_w)
        patch_embeddings = patch_embeddings.reshape(batch_size, self.num_frames, self.d_model, self.n_h*self.n_w)
        patch_embeddings = patch_embeddings.permute(0, 1, 3, 2)
        return self.norm_(patch_embeddings)

class LinearPatchDecoder(nn.Module):
    def __init__(self, d_model, patch_size, image_size):
        super().__init__()
        self.linear = nn.Linear(in_features=d_model, out_features=patch_size*patch_size*3) 
        self.patch_size = patch_size
        self.image_size = image_size
        self.d_model = d_model
        
    def forward(self, x):
        B, num_frames, N, d_model = x.shape
        n_h = self.image_size // self.patch_size
        n_w = n_h
        # here, x has shape (B, num_frames, n_h*n_w, d_model)
        x = self.linear(x)
        # (B, num_frames, n_h*n_w, patch_size*_patch_size*3) -> (B, num_frames, n_h, n_w, 3, patch_size, patch_size)
        x = x.reshape(B, num_frames, n_h, n_w, 3, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        x = x.reshape(x.shape[0], x.shape[1], 3, self.image_size, self.image_size)
        return x
