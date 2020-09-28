import torch
from torch.functional import F
torch.set_printoptions(profile="full")
from torch_geometric.nn.conv.graph_conv import GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadd
from math import floor
from utils.Linear_masked_weight import Linear_masked_weight

class PGC_GNN(torch.nn.Module):
    '''
    Definition of the Polynomial Graph Convolutional GNN
    '''
    def __init__(self, in_channels, out_channels, input_proj_dim, n_class=2, drop_prob=0.5, k=3, output=None, device=None):
        '''
        Function that define the PGC-GNN
        :param in_channels: input size
        :param out_channels: number of hidden units of the PGC
        :param input_proj_dim: number of hidden units for the first graph convolution layer
        :param n_class: number of classes
        :param drop_prob: dropout probability
        :param k: k parameter
        :param output: type of output stage
        :param device: device [CPU, GPU]
        '''
        super(PGC_GNN, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_proj_dim = input_proj_dim
        self.n_class = n_class
        self.output = output
        self.k = k
        self.dropout = torch.nn.Dropout(p=drop_prob)

        # first layer conv
        self.conv0 = GraphConv(self.in_channels, self.input_proj_dim)
        self.norm0 = torch.nn.BatchNorm1d(self.input_proj_dim)

        # PGC-layer
        self.lin=Linear_masked_weight(self.input_proj_dim * (k), self.out_channels * (k))
        xhi_layer_mask=[]
        for i in range(k):
            mask_ones = torch.ones(out_channels, input_proj_dim * (i + 1)).to(self.device)
            mask_zeros=torch.zeros(out_channels, input_proj_dim * (k - (i + 1))).to(self.device)
            xhi_layer_mask.append(torch.cat([mask_ones,mask_zeros],dim=1))
        self.xhi_layer_mask=torch.cat(xhi_layer_mask,dim=0).to(self.device)

        # batch normalizations
        self.bn_hidden_rec = torch.nn.BatchNorm1d(self.out_channels * k)
        self.bn_out = torch.nn.BatchNorm1d(self.out_channels * k * 3)

        # output function
        self.out_fun = torch.nn.LogSoftmax(dim=1)

        # readout layers
        self.lin1 = torch.nn.Linear(self.out_channels * k * 3, self.out_channels * k * 2)
        self.lin2 = torch.nn.Linear(self.out_channels * k * 2, self.out_channels * k)
        self.lin3 = torch.nn.Linear(self.out_channels * k, self.n_class)

        if output == "restricted_funnel":
            self.lin1 = torch.nn.Linear(self.out_channels * k * 3, floor(self.out_channels / 2) * k)

            self.lin2 = torch.nn.Linear(floor(self.out_channels/2) * k, self.n_class)

        self.reset_parameters()

    def reset_parameters(self):
        '''
        method that reset the model parameters
        '''

        print("reset parameters")
        self.norm0.reset_parameters()
        self.bn_hidden_rec.reset_parameters()
        self.bn_out.reset_parameters()
        self.conv0.reset_parameters()
        self.lin.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        '''
        model forward method
        :param data: current batch
        :return: the model output given the batch
        '''
        X = data.x

        edge_index = data.edge_index
        X= self.norm0(self.conv0(X,edge_index))

        k = self.k

        # compute adjacency matrix A
        adjacency_indexes = data.edge_index
        A_rows = adjacency_indexes[0]
        A_data = [1] * A_rows.shape[0]
        v_index = torch.FloatTensor(A_data).to(self.device)
        A_shape=[X.shape[0],X.shape[0]]
        A = torch.sparse.FloatTensor(adjacency_indexes, v_index, torch.Size(A_shape)).to_dense()

        H = [X]

        # compute the parts of matrix H for each k
        for i in range(k-1):
            xhi_layer_i=torch.mm(torch.matrix_power(A,i+1),X)
            H.append(xhi_layer_i)
        # project H by W
        H=self.bn_hidden_rec(self.lin(torch.cat(H, dim=1), self.xhi_layer_mask))

        # compute the graph layer representation using 3 different pooling strategies
        H_avg=gap(H, data.batch)
        H_add=gadd(H, data.batch)
        H_max=gmp(H, data.batch)
        H=torch.cat([H_avg, H_add, H_max],dim=1)

        #compute the readout
        if self.output=="funnel" or self.output is None:
            return self.funnel_output(H)
        elif self.output == "restricted_funnel":
            return self.restricted_funnel_output(H)
        else:
            assert False, "error in output stage"

    def funnel_output(self,H):
        '''
        readout part composed of a sequence of layer with dimension m*2, m, n_class, respectively
        :param H: the graph layer representation computed by the PGC-layer
        :return: the output of the model
        '''

        x = self.bn_out(H)

        x = (F.relu(self.lin1(x)))

        x = self.dropout(x)

        x = (F.relu(self.lin2(x)))
        x = self.dropout(x)

        x = self.out_fun(self.lin3(x))

        return x

    def restricted_funnel_output(self, H):
        '''
        readout part composed of a sequence of layers with dimension m/2, n_class, respectively
        :param H: the graph layer representation computed by the PGC-layer
        :return: the output of the model
        '''
        x = self.bn_out(H)

        x = self.dropout(x)

        x = (F.relu(self.lin1(x)))

        x = self.dropout(x)

        x = self.out_fun(self.lin2(x))

        return x
