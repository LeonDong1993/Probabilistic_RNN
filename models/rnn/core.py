import warnings
import numpy as np
from copy import deepcopy
from pdb import set_trace

import tinylib
from Gaussians import MultivariateGaussain, MixMG
from utmLib.clses import MyObject

import torch
import torch.nn as nn
torch.set_num_threads(1)

# Parameter Generating RNN (for indpendent MG)
class PGRNN(nn.Module):
    def __init__(self, input_size, node_num, cf):
        '''
        cf - config object includes following attributes
            latent_size
            depth
            device
            prec_thresh
            num_comps
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

        # the output of GRU would be of size N*L*H where
        # N is the batch size (number of sequence)
        # L is length of the sequence
        # H is the latent(hidden) size
        self.gru = nn.GRU(input_size = input_size, hidden_size = cf.latent_size, num_layers = cf.depth, batch_first = True)

        # here the precision is \frac{1}{\sigma ^ 2}, cannot be too big
        self.mu_head = nn.ModuleList([construct_head(cf.latent_size, node_num) for _ in range(cf.num_comps)])
        self.prec_head = nn.ModuleList([construct_head(cf.latent_size, node_num) for _ in range(cf.num_comps)])
        self.up_thresh = torch.Tensor([[ cf.prec_thresh[1] ]]).to(device = cf.device, dtype=torch.float)
        self.down_thresh = torch.Tensor([[ cf.prec_thresh[0] ]]).to(device = cf.device, dtype=torch.float)
        if cf.num_comps > 1: 
            self.comp_weight_head = construct_head(cf.latent_size, cf.num_comps)
        self.device = cf.device

        if cf.drop_out > 0:
            assert(cf.drop_out <= 0.7), "Too large dropout ratio!"
            self.drop = nn.Dropout(cf.drop_out)
        else:
            self.drop = None

    def forward(self, X, with_hidden=False, inH = None):
        # input should be of size N*L*F
        # F is number of Features
        # the dimension of these output is N*L*[~]
        
        if inH is None:
            features, H = self.gru(X)
        else:
            features, H = self.gru(X, inH)

        if self.drop is not None:
            features = self.drop(features)

        if len(self.mu_head) > 1:
            mu = torch.stack([HF(features) for HF in self.mu_head], dim = 1)
            prec = torch.stack([HF(features) for HF in self.prec_head], dim = 1)
            cw = self.comp_weight_head(features)
            cw = nn.functional.normalize(cw.abs() + 1e-3, p = 1, dim = 2)
        else:
            mu = self.mu_head[0](features)
            prec = self.prec_head[0](features)
            cw = None

        prec = torch.maximum(self.down_thresh, prec)
        prec = torch.minimum(self.up_thresh, prec)
        if not with_hidden:
            return (mu, prec, cw)
        else:
            return (mu,prec,cw), H   

    def np_forward(self, arg):
        # input can be:
        # 1. single time sequence (numpy array)
        # 2. a list of time sequences
        if isinstance(arg, np.ndarray):
            arg = [arg]
        arg = np.array(arg)

        if self.device != 'cpu':
            warnings.warn("Model is assumed to be on cpu, but actually not!")

        # convert to float tensor
        X = torch.from_numpy(arg).to(dtype = torch.float)
        with torch.no_grad():
            result = self.forward(X)
        del X
        return result

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
    def loss_func(nn_out, yseq, return_mass = False):
        # input shape is N*L*D
        # N is number of sequences
        # L is number if time steps
        # D is output size of header, related to number of features in yseq

        def compute_LL(out_U, out_P):
            ll_parts = -0.5 * ( torch.log(2 * np.pi / out_P)  + torch.square( yseq - out_U ) * out_P )
            # sum over features only, output shape is N*L
            return ll_parts.sum(dim = 2)

        U, P, W  = nn_out

        if W is not None:
            K = U.shape[1]
            W = torch.log(W)
            comp_ll = torch.stack([compute_LL(U[:,i], P[:,i])  for i in range(K)], dim = 2)
            comp_ll = comp_ll + W
            logmass = torch.logsumexp(comp_ll, dim = 2)
        else:
            logmass = compute_LL(U,P)

        # logmass is of shape N x L
        if return_mass:
            ret = logmass.sum(dim = 1)
        else:
            ret = -torch.mean(logmass) 
        return ret


class RNNCondIndMG:

    def fit(self, yseq_list, xseq_list, verbose = 0, device = 'cuda:0', drop_out = 0,
            prec_thresh = (1e-2,1e+2), depth = 2, hidden_size = None, n_comps = 1,
            pre_train = 4, pre_train_epoch = 0.15, max_epoch = 100, maxlr = 1e-3, **kwargs):

        xsize = xseq_list[0].shape[1]
        ysize = yseq_list[0].shape[1]
        N_total = len(xseq_list)

        if hidden_size is None:
            hidden_size = ysize

        if not isinstance(pre_train_epoch, int):
            pre_train_epoch = int(pre_train_epoch * max_epoch)

        nn_structure_conf = MyObject()
        nn_structure_conf.depth = depth
        nn_structure_conf.latent_size = hidden_size
        nn_structure_conf.prec_thresh = prec_thresh
        nn_structure_conf.device = device
        nn_structure_conf.num_comps = n_comps
        nn_structure_conf.drop_out = drop_out

        # max_seq_len = max([seq.shape[0] for seq in xseq_list])
        # allX = np.array([tinylib.pad_sequence(seq, max_seq_len) for seq in xseq_list])
        # allY = np.array([tinylib.pad_sequence(seq, max_seq_len) for seq in yseq_list])
        # allX = torch.from_numpy(allX).to(device = device, dtype = torch.float32)
        # allY = torch.from_numpy(allY).to(device = device, dtype = torch.float32)
        
        # def get_data_loader(data_index, batch_size, with_last = False):
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

        if verbose:
            print('Pretrain the models ...')

        candidates = list()
        for _ in range(pre_train):
            model = PGRNN(xsize, ysize, nn_structure_conf)
            model.loss = PGRNN.loss_func
            score = tinylib.train_model(model, N_total, get_data_loader, stop = pre_train_epoch, verbose = verbose, 
                                    maxlr = maxlr, max_epoch = max_epoch, device = device,  **kwargs)
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
            U,P,W = self.nn.np_forward(arg)
            if W is not None:
                arg = (U[0], P[0], W[0])
            else:
                arg = (U[0], P[0], None)

        # U,P,W = [item.numpy() for item in arg]
        U,P,W = arg
        U = U.numpy()
        P = P.numpy()

        if W is None:
            K = 1
            T = U.shape[0]
        else:
            W = W.numpy()
            K = U.shape[0]
            T = U.shape[1]

        ret = []
        for i in range(T):
            if K == 1:
                M = MultivariateGaussain()
                M.mu = U[i]
                M.S = np.diag(1.0 / P[i])
            else:
                M = MixMG()
                M.models = []
                M.W = W[i]
                for k in range(K):
                    mg = MultivariateGaussain()
                    mg.mu = U[k,i]
                    mg.S = np.diag(1.0 / P[k,i])
                    # P_arr = np.clip(P[k,i], 0.01, 100)
                    # mg.S = np.diag(1.0 / P_arr)
                    # if np.linalg.det(mg.S) == 0 or np.linalg.det(mg.S) == np.inf:
                    #     # still underflow or overflow
                    #     # clip to nearly identical matrix
                    #     diag_elements = np.clip(np.diag(mg.S), 0.9, 1.1)
                    #     mg.S = np.diag(diag_elements)
                    M.models.append(mg)
            ret.append(M)
        return ret

    def mass(self, yseq, xseq, logmode = False):
        '''
        support one input sequence only, i.e. 2D input
        return the (log)-likelihood for the sequence
        which is the (sum)/product over all time steps and variables
        '''
        yseq = np.array([yseq])
        Y = torch.from_numpy(yseq).to(dtype = torch.float)
        nn_out = self.nn.np_forward(xseq)
        ret = self.nn.loss(nn_out, Y, return_mass = True)
        ret = ret.numpy()

        if not logmode:
            ret = np.exp(ret)
        return float(ret)


class RNN_Regressor(nn.Module):
    def __init__(self, in_size, latent_size, drop_out, depth = 2) -> None:
        super(RNN_Regressor, self).__init__()
        self.gru = nn.GRU(input_size = in_size, hidden_size = latent_size, batch_first = True, num_layers = depth)
        L = []
        L.append( nn.Linear(latent_size, in_size) )
        # L.append( nn.BatchNorm1d(Nout) )
        L.append( nn.ReLU() )
        L.append( nn.Linear(in_size, in_size) )
        self.head = nn.Sequential(*L)
        
        if drop_out > 0:
            assert(drop_out <= 0.7), "Too large dropout ratio!"
            self.drop = nn.Dropout(drop_out)
        else:
            self.drop = None
        
    def forward(self, X, with_hidden = False, inH = None):
        if inH is None:
            features, H = self.gru(X)
        else:
            features, H = self.gru(X, inH)

        if self.drop is not None:
            features = self.drop(features)
        out = self.head(features)
        
        if not with_hidden:
            return out
        else:
            return out, H
    
    def save_checkpoint(self):
        # save current model status in the current directory
        self.last_state = deepcopy(self.state_dict())

    def load_checkpoint(self):
        # load the saved status
        self.load_state_dict(self.last_state)
    
    def moveto(self, device):
        self.device = device
        self.to(device)
        return

class RNN_Model:
    def fit(self, data, ev, **kwargs):
        total = len(data)
        D = data[0].shape[1]

        latent_size = kwargs['hidden_size']
        drop_out = kwargs['drop_out']
        device = kwargs['device']

        model = RNN_Regressor(D, latent_size, drop_out)
        model.loss = nn.MSELoss(reduction = 'mean')

        yseq_list = data
        xseq_list = []
        for item in data:
            x_ = np.zeros_like(item)
            x_[1:] = item[:-1]
            xseq_list.append(x_)

        # max_seq_len = max([seq.shape[0] for seq in xseq_list])
        # allX = np.array([tinylib.pad_sequence(seq, max_seq_len) for seq in xseq_list])
        # allY = np.array([tinylib.pad_sequence(seq, max_seq_len) for seq in yseq_list])
        # allX = torch.from_numpy(allX).to(device = device, dtype = torch.float32)
        # allY = torch.from_numpy(allY).to(device = device, dtype = torch.float32)
        
        # def get_data_loader(data_index, batch_size, with_last = False):
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

        def get_data_loader(data_index, batch_size, with_last = False):
            def dataloader():
                if data_index is not None:
                    selected = np.array(data_index)
                else:
                    selected = np.arange(total)
                
                seq_len = [xseq_list[i].shape[0] for i in selected]
                for _,ind in tinylib.group_batch_index(seq_len, batch_size, with_last):
                    X = np.array([xseq_list[i] for i in selected[ind]])
                    Y = np.array([yseq_list[i] for i in selected[ind]])
                    X = torch.from_numpy(X).to(device = device, dtype = torch.float)
                    Y = torch.from_numpy(Y).to(device = device, dtype = torch.float)
                    yield (X,Y)
                    del X,Y
            return dataloader
        
        tinylib.train_model(model, total, get_data_loader, **kwargs)
        self.nn = model
        model.moveto('cpu')
        return self
    
    def forecast(self, obs, n_step = 1):
        # if N observation is given, we can make N+1 predictions 
        # only return prediction from start (inclusive)
        
        N, D = obs.shape
        data = np.zeros(shape = (N+1,D))
        data[1:] = obs
        X = torch.from_numpy(data).to(self.nn.device, dtype=torch.float)
        if n_step == 1:
            with torch.no_grad():
                pred = self.nn.forward(X)
            pred = pred.numpy()
            return pred[1:]
        else:
            with torch.no_grad():
                pred, H = self.nn.forward(X, with_hidden=True)
            
            ret = [pred[-1].reshape(1,-1)]
            for _ in range(n_step-1):
                with torch.no_grad():
                    pred, H = self.nn.forward(ret[-1], with_hidden=True, inH = H)
                ret.append(pred)
            return np.vstack(ret)


    def predict_with_evi(self, seq, imputed):
        N, D = seq.shape
        K, D2 = imputed.shape
        assert(D == D2)
        # not sure how to use the partially observed xt to predict xt

        # forecast input x0 to xn-1
        # it returns predicted x1 to xn 

        obs = seq[:-1]
        res = self.forecast(obs, 1)
        # return the last K predictions 
        return res[-K:]

    def fill_sequence(self, seq, mask, n_iter = 0):
        N, D = seq.shape
        data = np.vstack( [np.zeros(shape = (1,D)), seq ] )
        X = torch.from_numpy(data).to(dtype = torch.float)
        for i in range(N+1):
            # fill the missing part using predicted values 
            cond = torch.isnan(X[i])
            if cond.any():
                X[i, cond] = pred[0, cond]
            
            # conduct next step prediction
            with torch.no_grad():
                if i == 0:
                    pred, H = self.nn.forward(X[i:(i+1)], with_hidden=True)
                else:
                    pred, H = self.nn.forward(X[i:(i+1)], with_hidden=True, inH = H)
        
        return X[1:].numpy()

