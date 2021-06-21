import torch
import torch.nn as nn
import numpy as np

from utils import *


class ViT(nn.Module):
    # Visual Transformer implemented using nn.Transformer
    def __init__(self, input_resolution=224, patch_size=32, width=512, layers=12, heads=16, output_dim=10, pos_emb=True, dropout=0):
        super(ViT, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=True)
        self.num_layers = layers
        
        scale = width ** -0.5
        self.cls_emb = nn.Parameter(scale*torch.randn(width))
        if pos_emb:
            self.position_emb = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        else:
            self.position_emb = None
        
        # self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, heads, dropout=dropout, activation='gelu'), layers)
        self.transformer = Transformer(width, layers, heads, dropout=dropout)
        self.ln_post = nn.LayerNorm(width)
        self.final_dropout = nn.Dropout(dropout)
        # self.proj = nn.Parameter(scale*torch.randn(width, output_dim))
        self.proj = nn.Linear(width, output_dim)
        
    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.cls_emb.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if self.position_emb is not None:
            x = x + self.position_emb.to(x.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            if isinstance(self.proj, torch.Tensor):
                x = self.final_dropout(x) @ self.proj
            else:
                x = self.proj(self.final_dropout(x))

        return x
    
    def load_pretrain(self, weights):
        # load the first convolutional layer
        self.conv1.weight.data.copy_(np2pt(weights['embedding/kernel'], ifconv=True))
        if self.conv1.bias is not None:
            self.conv1.bias.data.copy_(np2pt(weights['embedding/bias']))
        
        # load the embedding parameters
        self.cls_emb.data.copy_(np2pt(weights['cls']).squeeze())
        self.position_emb.data.copy_(np2pt(weights['Transformer/posembed_input/pos_embedding']).squeeze())
        
        # load the transformer parameters
        with torch.no_grad():
            for i in range(self.num_layers):
                block_name = 'Transformer/encoderblock_'+str(i)+'/'
                # load multi-head-attention part
                w, b, out_w, out_b = combine_qkv(weights=weights, layer=i)
                self.transformer.resblocks[i].attn.in_proj_weight.data.copy_(w.t())
                self.transformer.resblocks[i].attn.in_proj_bias.data.copy_(b)
                self.transformer.resblocks[i].attn.out_proj.weight.data.copy_(out_w.t())
                self.transformer.resblocks[i].attn.out_proj.bias.data.copy_(out_b)
                # load layernorm part
                self.transformer.resblocks[i].ln_1.weight.data.copy_(np2pt(weights[block_name+'LayerNorm_0/scale']))
                self.transformer.resblocks[i].ln_1.bias.data.copy_(np2pt(weights[block_name+'LayerNorm_0/bias']))
                self.transformer.resblocks[i].ln_2.weight.data.copy_(np2pt(weights[block_name+'LayerNorm_2/scale']))
                self.transformer.resblocks[i].ln_2.bias.data.copy_(np2pt(weights[block_name+'LayerNorm_2/bias']))
                # load mlp part
                self.transformer.resblocks[i].mlp.c_fc.weight.data.copy_(np2pt(weights[block_name+'MlpBlock_3/Dense_0/kernel']).t())
                self.transformer.resblocks[i].mlp.c_fc.bias.data.copy_(np2pt(weights[block_name+'MlpBlock_3/Dense_0/bias']))
                self.transformer.resblocks[i].mlp.c_proj.weight.data.copy_(np2pt(weights[block_name+'MlpBlock_3/Dense_1/kernel']).t())
                self.transformer.resblocks[i].mlp.c_proj.bias.data.copy_(np2pt(weights[block_name+'MlpBlock_3/Dense_1/bias']))

            # load the post-LayerNorm parameters
            self.ln_post.weight.data.copy_(np2pt(weights['Transformer/encoder_norm/scale']))
            self.ln_post.bias.data.copy_(np2pt(weights['Transformer/encoder_norm/bias']))
            
    def freeze_module(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.proj.parameters():
            p.requires_grad = True
            
    def unfreeze_module(self):
        for p in self.parameters():
            p.requires_grad = True