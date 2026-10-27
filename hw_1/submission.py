import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter


class MyGCNConv(MessagePassing):
    """
    X' = D^{-1/2} * (A + I) * D^{-1/2} * X * W
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, norm=norm)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j


class MySAGEConv(MessagePassing):
    """
    x_i' = W_self * x_i + W_neigh * mean_{j in N(i)} x_j
    out = lin_l(mean_j) + lin_r(x).
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__(aggr='mean')
        self.lin_self  = Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_self.reset_parameters()
        self.lin_neigh.reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        out_self  = self.lin_self(x)
        out_neigh = self.propagate(edge_index, x=x, size=(x.size(0), x.size(0)))
        out_neigh = self.lin_neigh(out_neigh)
        out = out_self + out_neigh
        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j