import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv
)

import math
from typing import Optional, Tuple, Union
from torch import Tensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from torch_geometric.utils import softmax

import torch
import torch.nn as nn
from torch_geometric.nn import aggr
from torch_geometric.nn import MessagePassing


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


class GraphSeqLM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dna_seq_dim: int,
        rna_seq_dim: int,
        protein_seq_dim: int,
        lm_dim: int,
        num_nodes: int,
        num_heads: int,
        num_classes: int,
        device: str
    ):
        super(GraphSeqLM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dna_seq_dim = dna_seq_dim
        self.rna_seq_dim = rna_seq_dim
        self.protein_seq_dim = protein_seq_dim
        self.lm_dim = lm_dim
        self.num_nodes = num_nodes
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.device = device

        self.internal_conv_first, self.internal_conv_block, self.internal_conv_last = self.build_internal_conv_layer(hidden_dim)
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        self.x_internal_norm_first = nn.BatchNorm1d(hidden_dim * num_heads)
        self.x_internal_norm_block = nn.BatchNorm1d(hidden_dim * num_heads)
        self.x_internal_norm_last = nn.BatchNorm1d(hidden_dim * num_heads)

        self.x_norm_first = nn.BatchNorm1d(hidden_dim * num_heads)
        self.x_norm_block = nn.BatchNorm1d(hidden_dim * num_heads)
        self.x_norm_last = nn.BatchNorm1d(embedding_dim * num_heads)

        # Simple aggregations
        self.mean_aggr = aggr.MeanAggregation()
        self.max_aggr = aggr.MaxAggregation()

        # Learnable aggregations
        self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)
        self.powermean_aggr = aggr.PowerMeanAggregation(learn=True)

        # Sequence models linear transformations
        self.dna_seq_transform = nn.Linear(dna_seq_dim, lm_dim) # 15659
        self.rna_seq_transform = nn.Linear(rna_seq_dim, lm_dim) # 15659
        self.protein_seq_transform = nn.Linear(protein_seq_dim, lm_dim) # 50257
        # Graph language model linear transformation
        self.glm_transform = nn.Linear(input_dim + lm_dim, hidden_dim)
        # Internal embedding linear transformation  
        self.internal_transform = nn.Linear(hidden_dim * num_heads, hidden_dim)
        # Modality merging linear transformation
        self.modality_transform = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.graph_prediction = nn.Linear(embedding_dim * num_heads, num_classes)

    def reset_parameters(self):
        super().reset_parameters()
        self.dna_seq_transform.reset_parameters()
        self.rna_seq_transform.reset_parameters()
        self.protein_seq_transform.reset_parameters()
        self.internal_transform.reset_parameters()

        self.x_internal_norm_first.reset_parameters()
        self.x_internal_norm_block.reset_parameters()
        self.x_internal_norm_last.reset_parameters()
        self.internal_conv_first.reset_parameters()
        self.internal_conv_block.reset_parameters()
        self.internal_conv_last.reset_parameters()

        self.modality_transform.reset_parameters()

        self.x_norm_first.reset_parameters()
        self.x_norm_block.reset_parameters()
        self.x_norm_last.reset_parameters()
        self.conv_first.reset_parameters()
        self.conv_block.reset_parameters()
        self.conv_last.reset_parameters()

        self.graph_prediction.reset_parameters()

    def build_internal_conv_layer(self, hidden_dim):
        internal_conv_first = TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=self.num_heads)
        internal_conv_block = TransformerConv(in_channels=hidden_dim*self.num_heads, out_channels=hidden_dim, heads=self.num_heads)
        internal_conv_last = TransformerConv(in_channels=hidden_dim*self.num_heads, out_channels=hidden_dim, heads=self.num_heads)
        return internal_conv_first, internal_conv_block, internal_conv_last

    def build_conv_layer(self, hidden_dim, embedding_dim):
        conv_first = TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=self.num_heads)
        conv_block = TransformerConv(in_channels=hidden_dim*self.num_heads, out_channels=hidden_dim, heads=self.num_heads)
        conv_last = TransformerConv(in_channels=hidden_dim*self.num_heads, out_channels=embedding_dim, heads=self.num_heads)
        return conv_first, conv_block, conv_last

    def forward(self, x, internal_edge_index, all_edge_index, dna_embedding, rna_embedding, protein_embedding):
        # Initial parameters
        num_nodes = int(self.num_nodes)
        batch_size = int(x.size(0) / num_nodes)

        ### LM sequence embedding
        # DNA sequence embedding
        embed_dna_seq = self.dna_seq_transform(dna_embedding)
        # RNA sequence embedding
        embed_rna_seq = self.rna_seq_transform(rna_embedding)
        # Protein sequence embedding
        embed_protein_seq = self.protein_seq_transform(protein_embedding)
        # Concatenate the embeddings
        embed_seq = torch.cat((embed_dna_seq, embed_rna_seq, embed_protein_seq), dim=0)
        # Expand the embed_seq to [batch_size, num_nodes, 3]
        expand_embed_seq = embed_seq.expand(batch_size, -1, -1)
        # Reshape the expand_embed_seq to [batch_size * num_nodes, 1]
        reshaped_embed_seq = expand_embed_seq.reshape(-1, self.lm_dim)

        ### Modality merging
        # Concatenate the x (shape: [batch_size * num_nodes, 1]) and embed_seq (shape: [num_nodes, 3])
        x_glm = torch.cat((x, reshaped_embed_seq), dim=-1)
        x_glm = self.glm_transform(x_glm)

        ### Internal message passing
        # Internal - layer1
        x_internal = self.internal_conv_first(x_glm, internal_edge_index)
        x_internal = self.act2(x_internal)
        x_internal = self.x_internal_norm_first(x_internal)
        # Internal - layer2
        x_internal = self.internal_conv_block(x_internal, internal_edge_index)
        x_internal = self.act2(x_internal)
        x_internal = self.x_internal_norm_block(x_internal)
        # Internal - layer3
        x_internal = self.internal_conv_last(x_internal, internal_edge_index)
        x_internal = self.act2(x_internal)
        x_internal = self.x_internal_norm_last(x_internal)
        x_internal_embedding = self.internal_transform(x_internal)

        ### All modality merging
        # Concatenate the x_internal_embedding (shape: [batch_size, num_nodes, 1]) and embedding (shape: [batch_size, num_nodes, 1])\
        x_glm_all = torch.cat((x_glm, x_internal_embedding), dim=-1)
        x_glm_all = self.modality_transform(x_glm_all)

        # Graph encoder - layer1 
        x_embed = self.conv_first(x_glm_all, all_edge_index)
        x_embed = self.x_norm_first(x_embed)
        x_embed = self.act2(x_embed)
        # Graph encoder - layer2
        x_embed = self.conv_block(x_embed, all_edge_index)
        x_embed = self.x_norm_block(x_embed)
        x_embed = self.act2(x_embed)
        # Graph encoder - layer3
        x_embed = self.conv_last(x_embed, all_edge_index)
        x_embed = self.x_norm_last(x_embed)
        x_embed = self.act2(x_embed)
        
        # Embedding decoder to [ypred]
        x_embed = x_embed.view(-1, num_nodes, self.embedding_dim * self.num_heads)
        x_embed = self.powermean_aggr(x_embed).view(-1, self.embedding_dim * self.num_heads)
        output = self.graph_prediction(x_embed)
        _, ypred = torch.max(output, dim=1)

        return output, ypred


    def loss(self, output, label):
        num_class = self.num_classes
        # Use weight vector to balance the loss
        weight_vector = torch.zeros([num_class]).to(device=self.device)
        label = label.long()
        for i in range(num_class):
            n_samplei = torch.sum(label == i)
            if n_samplei == 0:
                weight_vector[i] = 0
            else:
                weight_vector[i] = len(label) / (n_samplei * 1.5)
        # Calculate the loss
        output = torch.log_softmax(output, dim=-1)
        loss = F.nll_loss(output, label, weight_vector)
        return loss