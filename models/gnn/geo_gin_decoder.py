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

from typing import Callable, Optional, Union

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import spmm

from torch_geometric.nn import aggr

class GINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

    or

    .. math::
        \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

    here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, in_channels, out_channels, eps: float = 0., train_eps: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer('eps', torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        x = self.lin(x)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
        out = F.relu(out)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GINDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, 
                 num_node, device, num_class):
        super(GINDecoder, self).__init__()
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
        internal_conv_first = GINConv(in_channels=hidden_dim, out_channels=hidden_dim)
        internal_conv_block = GINConv(in_channels=hidden_dim, out_channels=hidden_dim)
        internal_conv_last = GINConv(in_channels=hidden_dim, out_channels=hidden_dim)
        return internal_conv_first, internal_conv_block, internal_conv_last

    def build_conv_layer(self, hidden_dim, embedding_dim):
        conv_first = GINConv(in_channels=hidden_dim, out_channels=hidden_dim)
        conv_block = GINConv(in_channels=hidden_dim, out_channels=hidden_dim)
        conv_last = GINConv(in_channels=hidden_dim, out_channels=embedding_dim)
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