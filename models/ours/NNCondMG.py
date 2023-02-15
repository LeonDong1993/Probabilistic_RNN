import numpy as np
from functools import partial
from copy import deepcopy
from pdb import set_trace

import tinylib
from Gaussians import MultivariateGaussain
from utmLib import utils
from utmLib.clses import MyObject
from utmLib.ml.GBN import GBN
from utmLib.ml.graph import Graph

import torch
import torch.nn as nn
torch.set_num_threads(1)


def create_graph(data, max_parents, corr_thresh):
    D = data.shape[1]
    G = Graph(digraph = False).make_complete_graph(D)
    # remove dependencies below correlation threshold
    corrs = tinylib.analyze_correlation(data)
    for i,j in utils.halfprod( range(D) ):
        if corrs[i,j] < corr_thresh:
            G.remove_edge(i,j)

    # make this graph directed
    total_corr = np.sum(corrs, axis = 0)
    nid = np.argmax(total_corr)
    G = G.todirect(nid)

    # restrict the number of parents if need
    if max_parents > 0:
        for i in range(D):
            par = G.find_parents(i)
            if len(par) <= max_parents:
                continue
            rank = np.argsort([corrs[i,p] for p in par])
            for j in rank[max_parents:] :
                G.remove_edge(par[j], i)
    return G

# data batch index loader
def minibatch_index(total, size, with_last = False):
    indices = np.arange(total)
    np.random.shuffle(indices)
    st = 0 ; ed = size
    while st < total:
        if (ed <= total) or (with_last):
            yield indices[st:ed]
        st = ed
        ed += size

class BuildingBlock(nn.Module):
    def __init__(self, insize, outsize):
        super(BuildingBlock, self).__init__()

        layers = []
        layers.append( nn.Linear(insize, outsize) )
        layers.append( nn.BatchNorm1d(outsize) )
        layers.append( nn.ReLU() )
        self.encoder = nn.Sequential(*layers)

    def forward(self, X):
        return self.encoder(X)

# NN model used to generate the params for GBN
class PGNN(nn.Module):
    def __init__(self, nn_insize, node_parents, cf):
        '''
        cf - config object includes following attributes
            depth
            compress_rate
            feature_size
            device
            max_header_size
            prec_thresh
        '''
        super(PGNN, self).__init__()
        node_num = len(node_parents)
        depth = cf.depth
        cr = cf.compress_rate
        fsize = cf.feature_size
        layer_size = [fsize * (cr ** (depth - i)) for i in range(depth+1)]
        layer_size = list(map(int, layer_size))

        layers = list()
        szin = nn_insize
        for i,szout in enumerate(layer_size):
            layers.append( BuildingBlock(szin, szout) )
            if cf.drop_out > 0 and i == ( len(layer_size) // 2 ):
                layers.append( nn.Dropout(cf.drop_out) )
            szin = szout
        self.encoder = nn.Sequential(*layers)

        def construct_head(Nin, Nout):
            L = []
            L.append( nn.Linear(Nin, Nout) )
            L.append( nn.BatchNorm1d(Nout) )
            L.append( nn.ReLU() )
            L.append( nn.Linear(Nout, Nout) )
            header = nn.Sequential(*L)
            return header

        # merge small headers
        # less expressive but more efficient
        thresh = cf.max_header_size
        header_size = []
        cur_sum = node_parents[0]
        assert(thresh >= 0 )
        for n in node_parents[1:]:
            assert( n < node_num )
            if (cur_sum + n) > thresh:
                header_size.append(cur_sum)
                cur_sum = n
            else:
                cur_sum += n
        header_size.append(cur_sum)

        # three types of heads
        self.bias_head = construct_head(fsize, node_num)
        self.weight_head = nn.ModuleList([construct_head(fsize, n) for n in header_size])
        self.prec_head = construct_head(fsize, node_num)
        self.up_thresh = torch.Tensor([[ cf.prec_thresh[1] ]]).to(cf.device)
        self.down_thresh = torch.Tensor([[ cf.prec_thresh[0] ]]).to(cf.device)
        self.device = cf.device

    def forward(self, X):
        features = self.encoder(X)
        bias = self.bias_head(features)
        weights = torch.cat([H(features) for H in self.weight_head], dim = 1)
        prec = self.prec_head(features)
        prec = torch.minimum(prec, self.up_thresh)
        prec = torch.maximum(prec, self.down_thresh)
        return weights, bias, prec

    def np_forward(self, arg):
        # make input 2d array
        if len(arg.shape) == 1:
            arg = arg.reshape(1,-1)

        # convert to float tensor
        X = torch.from_numpy(arg).to(device = self.device, dtype = torch.float)
        with torch.no_grad():
            A,B,P = self.forward(X)
        A = A.cpu()
        B = B.cpu()
        P = P.cpu()
        del X
        return A,B,P

    def save_checkpoint(self):
        # save current model status in the current directory
        self.last_state = deepcopy(self.state_dict())

    def load_checkpoint(self):
        # load the saved status
        self.load_state_dict(self.last_state)

    def moveto(self, device):
        self.device = device
        self.up_thresh = self.up_thresh.to(device)
        self.down_thresh = self.down_thresh.to(device)
        self.to(device)

    @staticmethod
    # this function can also be used to calculate log mass
    def loss_func(nn_out, ydata, G, return_mass = False):
        A,B,P = nn_out
        selector = np.concatenate([G.V[i].parents for i in range(G.N)])
        aug_ydata = ydata[:,selector]
        result = A * aug_ydata

        offset = 0
        mean = torch.zeros_like(B)
        for i in range(G.N):
            psize = len(G.V[i].parents)
            if psize > 0:
                mean[:, i] = torch.sum(result[:, offset:(offset+psize)], dim = 1)
                offset += psize
        mean = mean + B

        log_mass = -0.5 * ( torch.log(2 * np.pi / P) + torch.square(ydata - mean) * P )

        if return_mass:
            ret = torch.sum(log_mass, dim = 1)
        else:
            ret = -torch.sum(log_mass) / A.shape[0]
        return ret

class NeuralNetCondMG:
    def fit(self, ydata, xdata, verbose = 0, corr_thresh = 0, max_parents = None, device = 'cuda:0', drop_out = 0,
            prec_thresh = (1e-2,1e+2), depth = 2, compress_rate = 2, feature_size = None, max_header_size = None,
            pre_train = 4, pre_train_epoch = 0.15, max_epoch = 120, maxlr = 1e-3, **kwargs):

        N, x_size = xdata.shape
        N_, y_size = ydata.shape
        assert(N == N_), "Invalid train data"

        if max_parents is None:
            max_parents = int(0.5 * y_size) + 1

        if feature_size is None:
            feature_size = int(y_size * np.sqrt(max_parents))

        if max_header_size is None:
            max_header_size = y_size

        if not isinstance(pre_train_epoch, int):
            pre_train_epoch = int(pre_train_epoch * max_epoch)

        nn_structure_conf = MyObject()
        nn_structure_conf.depth = depth
        nn_structure_conf.compress_rate = compress_rate
        nn_structure_conf.feature_size = feature_size
        nn_structure_conf.prec_thresh = prec_thresh
        nn_structure_conf.device = device
        nn_structure_conf.max_header_size = max_header_size
        nn_structure_conf.drop_out = drop_out

        # build a GBN based on this graph
        G = create_graph(ydata, max_parents, corr_thresh)
        # G will be enhanced here
        self.gbn = GBN(G).fit(ydata)
        num_parents = [len(G.V[i].parents) for i in range(G.N)]

        if verbose:
            print('Pretrain the models ...')

        # # convert data
        # trainX = torch.from_numpy(xdata).to(device = device, dtype = torch.float)
        # trainY = torch.from_numpy(ydata).to(device = device, dtype = torch.float)

        # def get_data_loader_old(data_index, batch_size, with_last = False):
        #     def dataloader():
        #         if data_index is not None:
        #             dataX = trainX[data_index]
        #             dataY = trainY[data_index]
        #         else:
        #             dataX = trainX
        #             dataY = trainY
                
        #         for ind in minibatch_index(dataX.shape[0], batch_size, with_last):
        #             X = dataX[ind]
        #             Y = dataY[ind]
        #             yield (X,Y)
        #     return dataloader

        # this new dataloader is slower but saves cuda memory
        def get_data_loader(data_index, batch_size, with_last = False):
            def dataloader():
                if data_index is not None:
                    selected = np.array(data_index)
                else:
                    selected = np.arange(N_)
                
                for ind in minibatch_index(selected.size, batch_size, with_last):
                    X = torch.from_numpy(xdata[selected[ind]]).to(device = device, dtype = torch.float)
                    Y = torch.from_numpy(ydata[selected[ind]]).to(device = device, dtype = torch.float)
                    yield(X,Y)
                    del X,Y
            return dataloader

        candidates = list()
        for _ in range(pre_train):
            model = PGNN(x_size, num_parents, nn_structure_conf)
            model.loss = partial(PGNN.loss_func, G=G)
            score = tinylib.train_model(model, N_, get_data_loader, stop = pre_train_epoch, verbose = verbose, 
                            maxlr = maxlr, max_epoch = max_epoch, device = device, **kwargs)
            candidates.append( (score, model) )

        if verbose:
            print('Pre-train model scores:')
            print(np.round([s for s,_ in candidates], 4))

        candidates.sort(reverse = True)
        best_model = candidates[0][1]

        ret_score = tinylib.train_model(best_model, N_, get_data_loader, start = pre_train_epoch, verbose = verbose, 
                                    maxlr = maxlr, max_epoch = max_epoch, device = device, **kwargs)
        self.nn = best_model
        best_model.moveto('cpu')
        if verbose:
            print('Final model score:{}'.format(ret_score))

        return self

    def __call__(self, arg, auto_forward = True):
        # arg can be a single instance x or nn_out
        if auto_forward:
            assert(len(arg.shape) == 1)
            A,B,P = self.nn.np_forward(arg)
            arg = (A[0], B[0], P[0])

        # change param of gbn based on nn output
        A,B,P = arg
        G = self.gbn.g

        offset = 0
        for i in range(G.N):
            psize = len(G.V[i].parents)
            if psize > 0:
                weight = A[offset:(offset+psize)]
                weight = weight.numpy()
            else:
                weight = None
            offset += psize

            bias = B[i].item()
            var = 1.0 / P[i].item()
            self.gbn.potential[i].para = (weight, bias, var)

        mg = MultivariateGaussain.GBN2MG(self.gbn)
        return mg

    def mass(self, Y, X, logmode = False, batch_size = 500):
        # convert data into tensor and move to specific device.
        N = X.shape[0]
        X = torch.from_numpy(X).to(device = self.nn.device, dtype=torch.float)
        Y = torch.from_numpy(Y).to(device = self.nn.device, dtype=torch.float)

        ret = np.zeros( shape = (N,) )
        for idx in minibatch_index(N, batch_size, with_last=True):
            x = X[idx,:]
            y = Y[idx,:]
            with torch.no_grad():
                out = self.nn.forward(x)
                sub = self.nn.loss(out, y, return_mass = True)
            ret[idx] = sub.numpy()

        if not logmode:
            ret = np.exp(ret)
        del X, Y
        return ret

