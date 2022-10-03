import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


class MyNet(nn.Module):
    def __init__(self, num_node_s, num_node_t, num_feat, num_hid1, num_hid2, p):
        super().__init__()
        self.num_node_s = num_node_s
        self.num_node_t = num_node_t
        self.num_feat = num_feat
        self.num_hid1 = num_hid1
        self.num_hid2 = num_hid2
        self.gcn1 = MyGCN(num_feat, num_hid1)
        self.gcn2 = MyGCN(num_hid1, num_hid2)
        self.module_list = nn.ModuleList([self.gcn1, self.gcn2])
        self.p = p
        params_init(self.modules())


    def forward(self, lap, data):
        input_ = data.clone()
        hid = self.gcn1(lap, input_)
        hid = F.dropout(hid, p=self.p, training=self.training)
        output = self.gcn2(lap, hid)
        return output

class MyGCN(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.lin = nn.Linear(num_in, num_out, bias = True)
        self.act = nn.Tanh()
        params_init(self.modules())

    def forward(self, lap, data): #laps is normalized laplacian matrixs.
        output = self.lin(data)
        output = torch.matmul(lap, output)
        output = self.act(output)
        return output

class Attention(nn.Module):
    def __init__(self, num_node_s, num_node_t, num_out):
        super().__init__()
        self.attention_weight = nn.Parameter(torch.randn((1, (num_node_s + num_node_t) * num_out)), requires_grad=True)
        # nn.init.normal_(self.attention_weight.data)

    def forward(self, output_s, output_t):
        output_cat = torch.cat((output_s, output_t)).reshape((-1, 1))
        alpha = torch.nn.LeakyReLU(negative_slope = 0.2, inplace=False)(torch.matmul(self.attention_weight, output_cat))
        return alpha

class Reconstruction_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, recA):
        loss = (A - recA) ** 2
        loss = loss.sum() / len(A)
        return loss

class Fisher_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Hs, Ht):
        miu_s = torch.mean(Hs, dim=0)
        var_s = torch.var(Hs, dim=0)
        miu_t = torch.mean(Ht, dim=0)
        var_t = torch.var(Ht, dim=0)
        loss = torch.sum((miu_s - miu_t) ** 2) / torch.sum(var_s + var_t)
        return loss

class Refine_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hs, ht):
        # print(hs.shape)
        loss = 1 - torch.cosine_similarity(hs, ht, dim = 0)
        return loss



def params_init(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('tanh'))
            if m.bias is not None:
                m.bias.data = nn.init.constant_(m.bias.data, 0.0)
