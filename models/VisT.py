import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, Union
from .ResNet import BasicBlock, Bottleneck, ResNet18
from utils import ResidualAttentionBlock

class Visual_Tokenizer(nn.Module):
    def __init__(self, in_dim, static=True, hidden_dim=None):
        super(Visual_Tokenizer, self).__init__()
        self.static = static
        if self.static:
            assert(hidden_dim is not None)
            self.token_weight = nn.Linear(in_dim, hidden_dim, bias=False)
        else:
            self.token_weight = nn.Linear(in_dim, in_dim, bias=False)
    
    def forward(self, x, t=None):
        n = x.shape[0]
        c = x.shape[-1]
        x = x.reshape(n, -1, c) # [N, HW, C]
        if not self.static:
            assert(t is not None)
            token_map = self.token_weight(t) # [N, L, C]
            token_map = token_map.permute(0, 2, 1) # [N, C, L]
            attention_weight = torch.bmm(x, token_map)  # [N, HW, L]
            attention_weight = attention_weight.permute(0, 2, 1).softmax(dim=-1) # [N, L, HW]
        else:
            attention_weight = self.token_weight(x) # [N, HW, L]
            attention_weight = attention_weight.permute(0, 2, 1).softmax(dim=-1) # [N, L, HW]
        t_out = torch.bmm(attention_weight, x)
        return t_out

class Attention_layer(nn.Module):
    # the original transformer layer in arxiv:2006.03677
    def __init__(self, in_dim, static, num_tokens, hidden_dim, dropout=0):
        super(Attention_layer, self).__init__()
        self.static = static
        self.tokenizer = Visual_Tokenizer(in_dim, static, num_tokens)
        self.in_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.in_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_dim, hidden_dim)),
            ("relu", nn.ReLU()),
            ('dropout', nn.Dropout(dropout)),
            ("c_proj", nn.Linear(hidden_dim, hidden_dim))
        ]))
        self.out_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, x, t=None):
        t_in = self.tokenizer(x, t) # [N, L, C]
        
        in_query = self.in_Q(t_in) # [N, L, C]
        in_key = self.in_K(t_in) # [N, L, C]
        attn_weight1 = torch.bmm(in_query, in_key.permute(0, 2, 1)) # [N, L, L]
        attn_weight1 = attn_weight1.softmax(dim = -1)
        t_out_p = t_in + torch.bmm(attn_weight1, t_in) # [N, L, C]
        t_out = t_out_p + self.mlp(t_out_p) # [N, L, C]
        
        out_query = self.out_Q(x) # [N, HW, C]
        out_key = self.out_K(t_out) # [N, L, C]
        attn_weight2 = torch.bmm(out_query, out_key.permute(0, 2, 1))
        attn_weight2 = attn_weight2.softmax(dim = -1)
        x_out = x + torch.bmm(attn_weight2, t_out)
        
        return x_out, t_out
    
class MHAttention_layer(nn.Module):
    # the multi-head transformer layer in arxiv:2006.03677
    def __init__(self, in_dim, static, num_tokens, hidden_dim, num_heads=4, dropout=0):
        super(MHAttention_layer, self).__init__()
        self.static = static
        self.tokenizer = Visual_Tokenizer(in_dim, static, num_tokens)
        self.attn = ResidualAttentionBlock(hidden_dim, num_heads)
        self.out_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, x, t=None):
        t_in = self.tokenizer(x, t) # [N, L, C]
        t_out = self.attn(t_in) # [N, L, C]
        
        out_query = self.out_Q(x) # [N, HW, C]
        out_key = self.out_K(t_out) # [N, L, C]
        attn_weight2 = torch.bmm(out_query, out_key.permute(0, 2, 1))
        attn_weight2 = attn_weight2.softmax(dim = -1)
        x_out = x + torch.bmm(attn_weight2, t_out)
        
        return x_out, t_out

class T_ViT(nn.Module):
    def __init__(self, hidden_dim=256, n_token=16, multi_head=False, num_heads=None, output_dim=10, dropout=0):
        super(T_ViT, self).__init__()
        resnet = ResNet18()
        self.base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.layer1, resnet.layer2, resnet.layer3)
        self.base[4][1] = BasicBlock(in_planes=256, planes=hidden_dim, stride=1)
        
        # The first attn layer is always static since no previous information available
        if not multi_head:
            self.attn_layer1 = Attention_layer(hidden_dim, True, n_token, hidden_dim, dropout)
            self.attn_layer2 = Attention_layer(hidden_dim, False, n_token, hidden_dim, dropout)
        else:
            assert(num_heads is not None)
            self.attn_layer1 = MHAttention_layer(hidden_dim, True, n_token, hidden_dim, num_heads, dropout)
            self.attn_layer2 = MHAttention_layer(hidden_dim, False, n_token, hidden_dim, num_heads, dropout)

        self.attn_layer3 = None

        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # base cnn part
        x = self.base(x)
        n = x.shape[0]
        c = x.shape[1]
        x = x.reshape(n, c, -1).permute(0, 2, 1)
        
        # VT part
        x, t = self.attn_layer1(x)
        x, t = self.attn_layer2(x, t)
        if self.attn_layer3 is not None:
            x, t = self.attn_layer3(x, t)
            
        # predict head
        logits = self.fc(t.mean(dim=1))
        return logits
