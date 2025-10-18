import torch
import torch.nn as nn
from torch_scatter import scatter
from einops.layers.torch import Rearrange

import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

BN = True

# encoders

def DiscreteEncoder(nin, nhid):
    return nn.Embedding(nin, nhid)


def LinearEncoder(nin, nhid):
    return nn.Linear(nin, nhid)

def FeatureEncoder(TYPE, nin, nhid):
    models = {
        'Discrete': DiscreteEncoder,
        'Linear': LinearEncoder
    }

    return models[TYPE](nin, nhid)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class GNN(nn.Module):
    def __init__(self,
                 nin,
                 nout,
                 nlayer_gnn,
                 gnn_type,
                 bn=BN,
                 dropout=0.0,
                 res=True):
        super().__init__()
        self.dropout = dropout
        self.res = res

        self.convs = nn.ModuleList([GINEConv(
            nin, nin, bias=not bn) for _ in range(nlayer_gnn)])
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(nin) if bn else Identity() for _ in range(nlayer_gnn)])
        self.output_encoder = nn.Linear(nin, nout)

    def forward(self, x, edge_index, edge_attr):
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x
                previous_x = x

        x = self.output_encoder(x)
        return x


class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                     n_hid if i < nlayer-1 else nout,
                                     # TODO: revise later
                                               bias=True if (i == nlayer-1 and not with_final_activation and bias)
                                               or (not with_norm) else False)  # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer-1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: check if need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

        # if self.residual:
        #     x = x + previous_x
        return x

class TransformerConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=8):
        super().__init__()
        self.layer = TransformerConv(
            in_channels=nin, out_channels=nout//nhead, heads=nhead, edge_dim=nin, bias=bias)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)


class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = gnn.GINEConv(self.nn, train_eps=True)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)



class MPGNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 node_type, edge_type,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean'):

        super().__init__()
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.pooling = pooling

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = GNN(nin=nhid, nout=nhid, nlayer_gnn=nlayer_gnn,
                        gnn_type=gnn_type, bn=bn, dropout=dropout, res=res)

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = self.gnns(x, data.edge_index, edge_attr)

        # graph leval task
        x = scatter(x, data.batch, dim=0, reduce=self.pooling)
        x = self.output_decoder(x)

        return x