import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax


import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax

from torch_geometric.nn import aggr

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # X: [N, in_channels]
        # edge_index: [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1/2)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

         # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] OUT PUT DIMS = [N, out_channels]
        # import pdb; pdb.set_trace()
        return aggr_out


class GCNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, 
                 num_node, device, num_class):
        super(GCNDecoder, self).__init__()
        self.num_node = num_node
        self.num_class = num_class
        self.embedding_dim = embedding_dim

        self.internal_conv_first, self.internal_conv_block, self.internal_conv_last = self.build_internal_conv_layer(hidden_dim)
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        self.x_internal_norm_first = nn.BatchNorm1d(hidden_dim)
        self.x_internal_norm_block = nn.BatchNorm1d(hidden_dim)
        self.x_internal_norm_last = nn.BatchNorm1d(hidden_dim)

        self.x_norm_first = nn.BatchNorm1d(hidden_dim)
        self.x_norm_block = nn.BatchNorm1d(hidden_dim)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim)

        # Simple aggregations
        self.mean_aggr = aggr.MeanAggregation()
        self.max_aggr = aggr.MaxAggregation()
        # Learnable aggregations
        self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)
        self.powermean_aggr = aggr.PowerMeanAggregation(learn=True)

        self.internal_transform = nn.Linear(input_dim, hidden_dim)

        self.merge_modality = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.graph_prediction = nn.Linear(embedding_dim, num_class)

    def reset_parameters(self):
        self.internal_conv_first.reset_parameters()
        self.internal_conv_block.reset_parameters()
        self.internal_conv_last.reset_parameters()
        self.x_internal_norm_first.reset_parameters()
        self.x_internal_norm_block.reset_parameters()
        self.x_internal_norm_last.reset_parameters()
        self.conv_first.reset_parameters()
        self.conv_block.reset_parameters()
        self.conv_last.reset_parameters()
        self.x_norm_first.reset_parameters()
        self.x_norm_block.reset_parameters()
        self.x_norm_last.reset_parameters()
        self.internal_transform.reset_parameters()
        self.graph_prediction.reset_parameters()

    def build_internal_conv_layer(self, hidden_dim):
        internal_conv_first = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        internal_conv_block = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        internal_conv_last = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        return internal_conv_first, internal_conv_block, internal_conv_last

    def build_conv_layer(self, hidden_dim, embedding_dim):
        conv_first = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        conv_block = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        conv_last = GCNConv(in_channels=hidden_dim, out_channels=embedding_dim)
        return conv_first, conv_block, conv_last

    def forward(self, x, internal_edge_index, all_edge_index):
        ### Internal message passing
        x_internal = self.internal_transform(x)
        # Internal graph - Layer 1
        x_internal = self.internal_conv_first(x_internal, internal_edge_index)
        x_internal = self.x_internal_norm_first(x_internal)
        x_internal = self.act(x_internal)
        # Internal graph - Layer 2
        x_internal = self.internal_conv_block(x_internal, internal_edge_index)
        x_internal = self.x_internal_norm_block(x_internal)
        x_internal = self.act(x_internal)
        # Internal graph - Layer 3
        x_internal = self.internal_conv_last(x_internal, internal_edge_index)
        x_internal = self.x_internal_norm_last(x_internal)
        x_internal = self.act(x_internal)

        ### Internal - Initial embedding merging
        x_cat = torch.cat([x, x_internal], dim=-1)
        x_cat = self.merge_modality(x_cat)

        ### Global message passing
        # Global graph - Layer 1
        x_global = self.conv_first(x_cat, all_edge_index)
        x_global = self.x_norm_first(x_global)
        x_global = self.act2(x_global)
        # Global graph - Layer 2
        x_global = self.conv_block(x_global, all_edge_index)
        x_global = self.x_norm_block(x_global)
        x_global = self.act2(x_global)
        # Global graph - Layer 3
        x_global = self.conv_last(x_global, all_edge_index)
        x_global = self.x_norm_last(x_global)
        x_global = self.act2(x_global)
        
        # Embedding decoder to [ypred]
        x_global = x_global.view(-1, self.num_node, self.embedding_dim)
        x_global = self.powermean_aggr(x_global).view(-1, self.embedding_dim)
        output = self.graph_prediction(x_global)
        _, ypred = torch.max(output, dim=1)
        return output, ypred


    def loss(self, output, label):
        num_class = self.num_class
        # Use weight vector to balance the loss
        weight_vector = torch.zeros([num_class]).to(device='cuda')
        label = label.long()
        for i in range(num_class):
            n_samplei = torch.sum(label == i)
            if n_samplei == 0:
                weight_vector[i] = 0
            else:
                weight_vector[i] = len(label) / (n_samplei)
        # Calculate the loss
        output = torch.log_softmax(output, dim=-1)
        loss = F.nll_loss(output, label, weight_vector)
        return loss