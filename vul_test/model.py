import torch.nn as nn
from torch.nn import Module, ModuleList, Linear, LayerNorm
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, GCNConv, global_mean_pool, GlobalAttention, AttentionalAggregation, GATConv
from torch_geometric.explain import Explainer
import torch
import torch_geometric.nn as pyg_nn
from  dgl.nn.pytorch.conv import GatedGraphConv as dgl_ggnn
from dgl.nn.pytorch import GlobalAttentionPooling as dgl_gap
from torch_geometric.explain.algorithm import GNNExplainer
from mp_ import MessagePassing
# from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag


class GONN(Module):
    def __init__(self, params, name):
        super().__init__()
        self.name = name
        self.params = params
        self.linear_trans_in = ModuleList()
        self.linear_trans_out = Linear(params['hidden_channel'], params['out_channel'])
        self.norm_input = ModuleList()
        self.convs = ModuleList()

        self.tm_norm = ModuleList()
        self.tm_net = ModuleList()
        self.gatconv = ModuleList()
        # self.gatconv = GATConv(params[''], params[''], heads=params['heads'], concat=True, dropout=params['dropout_rate'])
        self.linear_trans_in.append(Linear(params['in_channel'], params['hidden_channel']))

        self.norm_input.append(LayerNorm(params['hidden_channel']))                            ##层归一化

        for i in range(params['num_layers_input']-1):
            self.linear_trans_in.append(Linear(params['hidden_channel'], params['hidden_channel']))
            self.norm_input.append(LayerNorm(params['hidden_channel']))

        if params['global_gating']==True:
            tm_net = Linear(2*params['hidden_channel'], params['chunk_size'])

        for i in range(params['num_layers']):
            self.tm_norm.append(LayerNorm(params['hidden_channel']))
            if params['gat'] == True:
                self.gatconv.append(GATConv(params['hidden_channel'], params['hidden_channel'], heads=params['heads'], concat=True, dropout=params['dropout_rate']))
            if params['global_gating']==False:
                self.tm_net.append(Linear(2*params['hidden_channel'], params['chunk_size']))
            else:
                self.tm_net.append(tm_net)
            
            if params['model']=="ONGNN":

                self.convs.append(ONGNNConv(tm_net=self.tm_net[i], tm_norm=self.tm_norm[i], params=params))

        self.params_conv = list(set(list(self.convs.parameters())+list(self.tm_net.parameters())))
        self.params_others = list(self.linear_trans_in.parameters())+list(self.linear_trans_out.parameters())

    def forward(self, x, edge_index, batch):
        check_signal = []
        for i in range(len(self.linear_trans_in)):
            x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x = F.relu(self.linear_trans_in[i](x))
            x = self.norm_input[i](x)
        tm_signal = x.new_zeros(self.params['chunk_size'])

        for j in range(len(self.convs)):
            if self.params['dropout_rate2']!='None':
                x = F.dropout(x, p=self.params['dropout_rate2'], training=self.training)
            else:
                x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
            x, tm_signal = self.convs[j](x, edge_index, last_tm_signal=tm_signal)
            check_signal.append(dict(zip(['tm_signal'], [tm_signal])))

        lineout = self.linear_trans_out(x)
        # print(lineout.shape)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.params['dropout_rate'], training=self.training)
        x = self.linear_trans_out(x)
        # print(x.shape)
        encode_values = dict(zip(['x', 'check_signal'], [x, check_signal]))
        # print(x.shape)
        # print(lineout.shape)
        return x, lineout
class ONGNNConv(MessagePassing):
    def __init__(self, tm_net, tm_norm, params):
        super(ONGNNConv, self).__init__('mean')
        self.params = params
        self.tm_net = tm_net
        self.tm_norm = tm_norm

    def forward(self, x, edge_index, last_tm_signal):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            if self.params['add_self_loops']==True:
                edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            if self.params['add_self_loops']==True:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        if self.params['tm']==True:
            if self.params['simple_gating']==True:
                tm_signal_raw = F.sigmoid(self.tm_net(torch.cat((x, m), dim=1)))    
            else:
                tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
                tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
                if self.params['diff_or']==True:
                    tm_signal_raw = last_tm_signal+(1-last_tm_signal)*tm_signal_raw
            tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.params['hidden_channel']/self.params['chunk_size']), dim=1)
            out = x*tm_signal + m*(1-tm_signal)
        else:
            out = m
            tm_signal_raw = last_tm_signal

        out = self.tm_norm(out)

        return out, tm_signal_raw

class GatedGraphResidualBlock(nn.Module):   #残差连接层
    def __init__(self, hidden_dim, num_layers) -> None:
        super(GatedGraphResidualBlock, self).__init__()
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ggnn = GatedGraphConv(hidden_dim, num_layers)
    def forward(self, node_embed, edge_index):
        out = self.ggnn(node_embed, edge_index)
        return out + node_embed
                
class GGNNModel(nn.Module):
    def __init__(self, input_dim, num_layers, hidden_dim, output_dim, name, device = None) -> None:
        super(GGNNModel, self).__init__()
        self.name = name
        self.device = device
        self.nodeEmbed = nn.Linear(input_dim, hidden_dim)
        self.ggnn = self.residual_layers(hidden_dim, num_layers)
        self.cat_linear = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_nn = nn.Linear(hidden_dim, 1)
        self.pool = AttentionalAggregation(self.gate_nn)
        hidden_size = int(hidden_dim / 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.outfc = nn.Linear(hidden_size, output_dim)
        self.lineout = nn.Linear(hidden_dim, output_dim)
    def residual_layers(self, hidden_dim, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(GatedGraphResidualBlock(hidden_dim, num_layers))
        return nn.ModuleList(layers)
    def forward(self, node_embed, edge_index, batch):
        node_embed = self.nodeEmbed(node_embed)
        layer_outputs = []
        for layer in self.ggnn:
            out = layer(node_embed, edge_index)
            layer_outputs.append(out)
        hidden = self.cat_linear(torch.cat(layer_outputs, dim=-1))
        lineout = self.lineout(hidden)
        # print(lineout.shape)
        hidden = self.pool(hidden, batch)
        hidden = self.mlp(hidden)
        out = self.outfc(hidden)
        # print(out.shape)
        return out, lineout

class SimpleGGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(SimpleGGNN, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.ggnn = GatedGraphConv(hidden_dim, num_layers)
        self.output_linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, node_embed, edge_index, batch):
        node_embed = self.input_linear(node_embed)
        node_embed = self.ggnn(node_embed, edge_index)
        node_embed = global_mean_pool(node_embed, batch)  
        node_embed = F.dropout(node_embed, p=0.5, training=self.training)
        out = self.output_linear(node_embed)
        return out
        
class GGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GGNN, self).__init__()
     
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.ggnn = GatedGraphConv(hidden_dim, num_layers)
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, node_embed, edge_matrix):
        
        
        node_embed = self.input_linear(node_embed)
        edge_index = edge_matrix
        # edge_index = self.input_linear(edge_index)
        node_state = node_embed
        for _ in range(self.num_layers):
            node_state = self.ggnn(node_state, edge_index.to(torch.int64))
        node_state = node_state.mean(dim=0, keepdim=True)
        # graph_embedding = pyg_nn.global_add_pool(node_state, batch)
        # graph_embedding = torch.sum(node_state, dim=1)
        out = self.output_linear(node_state) ## 200*2
        return out

class GCN(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
