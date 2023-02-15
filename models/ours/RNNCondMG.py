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

# Parameter Generating RNN (for GBN only)
class PGRNN(nn.Module):
    def __init__(self, input_size, node_parents, cf):
        '''
        cf - config object includes following attributes
            latent_size
            depth
            device
            max_header_size
            prec_thresh
        '''
        super(PGRNN, self).__init__()

        def construct_head(Nin, Nout):
            L = []
            L.append( nn.Linear(Nin, Nout) )
            # L.append( nn.BatchNorm1d(Nout) )
            L.append( nn.ReLU() )
            L.append( nn.Linear(Nout, Nout) )
            header = nn.Sequential(*L)
            return header

        # merge small headers, less expressive but more efficient
        node_num = len(node_parents)
        thresh = cf.max_header_size
        header_size = []
        cur_sum = node_parents[0]
        assert( thresh >= 0 )
        for n in node_parents[1:]:
            assert( n < node_num )
            if (cur_sum + n) > thresh:
                header_size.append(cur_sum)
                cur_sum = n
            else:
                cur_sum += n
        header_size.append(cur_sum)

        # the output of GRU would be of size N*L*H where
        # N is the batch size (number of sequence)
        # L is length of the sequence
        # H is the latent(hidden) size
        self.gru = nn.GRU(input_size = input_size, hidden_size = cf.latent_size, num_layers = cf.depth, batch_first = True)
        # three types of heads
        self.bias_head = construct_head(cf.latent_size, node_num)
        self.weight_head = nn.ModuleList([construct_head(cf.latent_size, n) for n in header_size])
        self.prec_head = construct_head(cf.latent_size, node_num)
        self.up_thresh = torch.Tensor([[ cf.prec_thresh[1] ]]).to(cf.device)
        self.down_thresh = torch.Tensor([[ cf.prec_thresh[0] ]]).to(cf.device)
        self.device = cf.device

        if cf.drop_out > 0:
            assert(cf.drop_out <= 0.7), "Too large dropout ratio!"
            self.drop = nn.Dropout(cf.drop_out)
        else:
            self.drop = None

    def forward(self, X, inH = None, with_hidden = False):
        # input should be of size N*L*F
        # F is number of Features
        if inH is None:
            features, H = self.gru(X)
        else:
            features, H = self.gru(X, inH)

        if self.drop is not None:
            features = self.drop(features)

        bias = self.bias_head(features)
        weights = torch.cat([H(features) for H in self.weight_head], dim = 2)
        prec = self.prec_head(features)
        prec = torch.maximum(prec, self.down_thresh)
        prec = torch.minimum(prec, self.up_thresh)
        # the dimension of these output is N*L*[~]
        if not with_hidden:
            return weights, bias, prec
        else:
            return (weights, bias, prec), H

    def np_forward(self, arg):
        # input can be:
        # 1. single time sequence (numpy array)
        # 2. a list of time sequences
        if isinstance(arg, np.ndarray):
            arg = [arg]
        arg = np.array(arg)

        # convert to float tensor
        X = torch.from_numpy(arg).to(device = self.device, dtype = torch.float32)
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
    def loss_func(nn_out, yseq, G, return_mass = False):
        N, T, _ = yseq.shape
        A,B,P = nn_out
        selector = np.concatenate([G.V[i].parents for i in range(G.N)])
        aug_ydata = yseq[:,:,selector]
        result = A * aug_ydata

        offset = 0
        mean = torch.zeros_like(B)
        for i in range(G.N):
            psize = len(G.V[i].parents)
            if psize > 0:
                mean[:, :, i] = torch.sum(result[:, :, offset:(offset+psize)], dim = 2)
                offset += psize
        mean = mean + B

        log_mass = -0.5 * ( torch.log(2 * np.pi / P) + torch.square(yseq - mean) * P )

        if return_mass:
            ret = torch.sum(log_mass, dim = (1,2))
        else:
            ret = -torch.sum(log_mass) / (N*T)
        return ret


class RecurrentNNCondMG:

    def fit(self, yseq_list, xseq_list, verbose = 0, corr_thresh = 0, max_parents = None, device = 'cuda:0',
            prec_thresh = (1e-2, 1e+2), depth = 2, hidden_size = None, max_header_size = None, drop_out = 0,
            pre_train = 4, pre_train_epoch = 0.15, max_epoch = 120, maxlr = 1e-3, **kwargs):

        xsize = xseq_list[0].shape[1]
        ysize = yseq_list[0].shape[1]
        N_total = len(xseq_list)

        if max_parents is None:
            max_parents = int(0.5 * ysize) + 1

        if hidden_size is None:
            hidden_size = int(ysize * np.sqrt(max_parents))

        if max_header_size is None:
            max_header_size = ysize

        if not isinstance(pre_train_epoch, int):
            pre_train_epoch = int(pre_train_epoch * max_epoch)

        nn_structure_conf = MyObject()
        nn_structure_conf.depth = depth
        nn_structure_conf.latent_size = hidden_size
        nn_structure_conf.prec_thresh = prec_thresh
        nn_structure_conf.device = device
        nn_structure_conf.max_header_size = max_header_size
        nn_structure_conf.drop_out = drop_out

        # build a GBN based on this graph
        stacked_ydata = np.vstack( yseq_list )
        G = create_graph(stacked_ydata, max_parents, corr_thresh)
        # G will be enhanced here
        self.gbn = GBN(G).fit( stacked_ydata ) # G will be enhanced
        num_parents = [len(G.V[i].parents) for i in range(G.N)]

        if verbose:
            print('Pretrain the models ...')

        # max_seq_len = max([seq.shape[0] for seq in xseq_list])
        # allX = np.array([tinylib.pad_sequence(seq, max_seq_len) for seq in xseq_list])
        # allY = np.array([tinylib.pad_sequence(seq, max_seq_len) for seq in yseq_list])
        # allX = torch.from_numpy(allX).to(device = device, dtype = torch.float32)
        # allY = torch.from_numpy(allY).to(device = device, dtype = torch.float32)
        
        # def get_data_loader_old(data_index, batch_size, with_last = False):
        #     def dataloader():
        #         if data_index is not None:
        #             dataX = allX[data_index]
        #             dataY = allY[data_index]
        #             seq_len = [xseq_list[i].shape[0] for i in data_index]
        #         else:
        #             dataX = allX
        #             dataY = allY
        #             seq_len = [seq.shape[0] for seq in xseq_list]
                
        #         for sL,ind in tinylib.group_batch_index(seq_len, batch_size, with_last):
        #             X = dataX[ind, :sL]
        #             Y = dataY[ind, :sL]
        #             yield (X,Y)
        #     return dataloader

        # this version of dataloader saves more memory
        def get_data_loader(data_index, batch_size, with_last = False):
            def dataloader():
                if data_index is not None:
                    selected = np.array(data_index)
                else:
                    selected = np.arange(N_total)
                
                seq_len = [xseq_list[i].shape[0] for i in selected]
                for _,ind in tinylib.group_batch_index(seq_len, batch_size, with_last):
                    X = np.array([xseq_list[i] for i in selected[ind]])
                    Y = np.array([yseq_list[i] for i in selected[ind]])
                    X = torch.from_numpy(X).to(device = device, dtype = torch.float)
                    Y = torch.from_numpy(Y).to(device = device, dtype = torch.float)
                    yield (X,Y)
                    del X,Y
            return dataloader

        candidates = list()
        for _ in range(pre_train):
            model = PGRNN(xsize, num_parents, nn_structure_conf)
            model.loss = partial(PGRNN.loss_func, G = G)
            score = tinylib.train_model(model, N_total, get_data_loader, stop = pre_train_epoch, verbose = verbose, 
                            maxlr = maxlr, max_epoch = max_epoch, device = device, **kwargs)
            candidates.append( (score, model) )

        if verbose:
            print('Pre-train model scores:')
            print(np.round([s for s,_ in candidates], 4))

        candidates.sort(reverse = True)
        best_model = candidates[0][1]

        ret_score = tinylib.train_model(best_model, N_total, get_data_loader, start = pre_train_epoch, verbose = verbose, 
                                    maxlr = maxlr, max_epoch = max_epoch, device = device, **kwargs)
        self.nn = best_model
        best_model.moveto('cpu')
        if verbose:
            print('Final model score:{}'.format(ret_score))
        return self

    def __call__(self, arg, auto_forward = True):
        if auto_forward:
            assert(len(arg.shape) == 2)
            A,B,P = self.nn.np_forward(arg)
            arg = (A[0], B[0], P[0])

        def get_mg(A,B,P):
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

            return MultivariateGaussain.GBN2MG(self.gbn)

        T = arg[0].shape[0]
        As,Bs,Ps = arg
        MGs = [get_mg(As[i], Bs[i], Ps[i]) for i in range(T)]
        return MGs

    def mass(self, yseq, xseq, logmode = False):
        '''
        return the (log)likelihood of one sequence
        accumulated over all time steps and all variables
        '''
        yseq = np.array([yseq])
        Y = torch.from_numpy(yseq).to(dtype = torch.float)
        nn_out = self.nn.np_forward(xseq)
        ret = self.nn.loss(nn_out, Y, return_mass = True)
        ret = ret.cpu().numpy()

        if not logmode:
            ret = np.exp(ret)
        return float(ret)



