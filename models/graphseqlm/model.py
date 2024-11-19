import os
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv
)

from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax

import math
from tqdm.auto import tqdm
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax

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


class TransformerConv(MessagePassing):
    _alpha: OptTensor
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights=None,
    ):
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor or (torch.Tensor, torch.Tensor)): The input node
                features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads,
                                                      self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GraphSeqLM:
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 5,
        embedding_dim: int = 5,
        num_nodes: int = 2111,
        num_heads: int = 1,
        num_classes: int = 2,
        dna_seq_model = None,
        rna_seq_model = None,
        protein_seq_model = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dna_seq_model = dna_seq_model
        self.rna_seq_model = rna_seq_model
        self.protein_seq_model = protein_seq_model
        
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        self.x_norm_first = nn.BatchNorm1d(hidden_dim * num_heads)
        self.x_norm_block = nn.BatchNorm1d(hidden_dim * num_heads)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim * num_heads)

        # Simple aggregations
        self.mean_aggr = aggr.MeanAggregation()
        self.max_aggr = aggr.MaxAggregation()
        # Learnable aggregations
        self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)
        self.powermean_aggr = aggr.PowerMeanAggregation(learn=True)

        self.pretrain_transform = nn.Linear(1, 1)
        self.graph_prediction = nn.Linear(embedding_dim * num_heads, num_classes)

    def reset_parameters(self):
        self.dna_seq_model.reset_parameters()
        self.rna_seq_model.reset_parameters()
        self.protein_seq_model.reset_parameters()
        self.conv_first.reset_parameters()
        self.conv_block.reset_parameters()
        self.conv_last.reset_parameters()
        self.pretrain_transform.reset_parameters()
        self.graph_prediction.reset_parameters()

    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=self.num_heads)
        conv_block = TransformerConv(in_channels=hidden_dim*self.num_heads, out_channels=hidden_dim, heads=self.num_heads)
        conv_last = TransformerConv(in_channels=hidden_dim*self.num_heads, out_channels=embedding_dim, heads=self.num_heads)
        return conv_first, conv_block, conv_last

    def forward(self, x, embedding, all_edge_index, seq):
        # Convert seq from numpy 1D strings to list of strings
        seq = seq.tolist()
        # DNA sequence model
        dna_seq = seq[:self.num_nodes, ]
        embed_dna_seq = self.dna_seq_model(dna_seq)
        # RNA sequence model
        rna_seq = seq[self.num_nodes:2*self.num_nodes, ]
        embed_rna_seq = self.rna_seq_model(rna_seq)
        # Protein sequence model
        protein_seq = seq[2*self.num_nodes:, ]
        embed_protein_seq = self.protein_seq_model(protein_seq)
        # Concatenate the embeddings
        embed_seq = embed_dna_seq + embed_rna_seq + embed_protein_seq
        # Convert the embeddings to torch tensor (shape: [num_nodes, 3])
        embed_seq = torch.tensor(embed_seq).float()
        # Concatenate the x (shape: [batch_size, num_nodes, 1]) and embed_seq (shape: [num_nodes, 3])
        x_cat = torch.cat((x, embed_seq), dim=-1)
        # Concatenate the x_cat (shape: [batch_size, num_nodes, 4]) and embedding (shape: [batch_size, num_nodes, 1])
        pretrain_embedding = self.pretrain_transform(embedding)
        x_cat_embed = torch.cat((x_cat, pretrain_embedding), dim=-1)
        # Graph encoder
        x = self.conv_first(x, all_edge_index)
        x = self.x_norm_first(x)
        x = self.act2(x)

        x = self.conv_block(x, all_edge_index)
        x = self.x_norm_block(x)
        x = self.act2(x)

        x = self.conv_last(x, all_edge_index)
        x = self.x_norm_last(x)
        x = self.act2(x)
        
        # Embedding decoder to [ypred]
        x = x.view(-1, self.num_node, self.embedding_dim * self.num_head)
        x = self.powermean_aggr(x).view(-1, self.embedding_dim * self.num_head)
        output = self.graph_prediction(x)
        _, ypred = torch.max(output, dim=1)
        return output, ypred


    def loss(self, output, label):
        num_class = self.num_classes
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
    
        
