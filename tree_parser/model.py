import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from tree_parser.utils import dep_edges, pos_tags


class GraphTransformerModel(nn.Module):

    def __init__(self, language_model, ninp=200, nhead=2, nhid=200, nlayers=2, dropout=0.2, adj_dropout=0.1):
        super().__init__()
        self.language_model = language_model
        self.model_type = 'TreeTransformer'
        self.adj_dropout = adj_dropout
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder_in = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder_out = TransformerEncoder(encoder_layers, nlayers)

        self.input_linear = nn.Linear(768, ninp)
        self.output_linear = nn.Linear(ninp, 30522)

        self.sa_layer1_linear = nn.Linear(ninp, nhid)

        self.linearK = nn.Linear(ninp, nhid)
        self.linearQ = nn.Linear(ninp, nhid)
        self.linearBiaff = nn.Linear(ninp, nhid)
        self.linearV = nn.Linear(ninp, nhid)
        self.linearEdges = nn.Linear(nhid, len(dep_edges))
        self.linearPOS = nn.Linear(nhid, len(pos_tags))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.linearV.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.language_model(src)[0]
        output = self.input_linear(output)
        output = F.relu(output)
        output = self.transformer_encoder_in(output)

        v = self.linearV(output)
        v = F.relu(v)

        pos = self.linearPOS(v)
        pos = F.softmax(pos, dim=-1)

        edges = self.linearEdges(v)
        edges = F.softmax(edges, dim=-1)

        k = self.linearK(output)
        q = self.linearQ(output)
        W = self.linearBiaff(q)
        adj = torch.bmm(k, W.transpose(2, 1))
        adj = F.softmax(adj, dim=-1)


        return edges, adj, pos
