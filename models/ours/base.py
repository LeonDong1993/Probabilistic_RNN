from copy import deepcopy
import numpy as np
import torch
import pdb

from utmLib import utils
from .NNCondMG import NeuralNetCondMG
from .RNNCondMG import RecurrentNNCondMG

DEBUG = 0

if DEBUG:
    print('== Currently In DEBUG mode ==')


class DynamicNeuralMG:
    def fit(self, data, evi_var, **kwargs):
        '''
        data - a list of time sequence
        evi_var - evidence variable
        '''
        D = data[0].shape[1]
        _e = evi_var
        _x = utils.notin(range(D), _e)

        # create transition sequence
        ztm1 = []
        for seq in data:
            new_seq = np.zeros_like(seq)
            new_seq[1:] = seq[0:-1]
            ztm1.append(new_seq)

        ztm1 = np.vstack(ztm1)
        zt = np.vstack(data)

        # learn p(x^t | x^{t-1})
        pr = NeuralNetCondMG().fit(zt[:,_x], ztm1[:,_x], **kwargs )

        # learn p(e^t | x^t, x^{t-1})
        if len(_e) > 0:
            xtxtm1 = np.hstack( [zt[:,_x], ztm1[:,_x]] )
            pd = NeuralNetCondMG().fit(zt[:,_e], xtxtm1, **kwargs)
        else:
            pd = None

        self.dist = (pr, pd)
        self.vars = (_x,_e)
        return self

    def mass(self, seq, logmode = False):
        '''
        support only one sequence a time
        '''
        pr, pd = self.dist
        _x, _e = self.vars

        zt = seq
        ztm1 = np.zeros_like(seq)
        ztm1[1:] = zt[0:-1]

        ret = np.sum( pr.mass(zt[:,_x], ztm1[:,_x], logmode=True) )

        if DEBUG:
            tmp = 0.0
            for i in range(seq.shape[0]):
                mg = pr(ztm1[i,_x])
                tmp += mg.mass(zt[i,_x], logmode = 1)
            assert( abs(tmp - ret) < 1e-3 ), '{} v.s {}'.format(tmp, ret)

        if len(_e) > 0:
            ret += np.sum( pd.mass(zt[:,_e], np.hstack([zt[:,_x], ztm1[:,_x]]), logmode=True) )
        ret = ret / seq.shape[0]

        if not logmode:
            ret = np.exp(ret)
        return ret
    
    def forecast(self, prev_obs, n_step = 1):
        pr, pd = self.dist
        _x, _e = self.vars

        assert( len(_e) == 0 ), "Under Implmentation"

        if n_step == 1:
            pred = np.zeros_like(prev_obs)
            for i,v in enumerate(prev_obs):
                mg = pr(v)
                pred[i] = mg.mu
        else:
            ret = []
            prev = prev_obs[-1]
            for i in range(n_step):
                mg = pr(prev)
                ret.append(mg.mu)
                prev = mg.mu
            pred = np.array(ret)

        return pred
    
    def predict_with_evi(self, seq, mask):
        pr, pd = self.dist
        _x, _e = self.vars

        assert( len(_e) == 0 ), "Under Implmentation"
        K = mask.shape[0]
        N = seq.shape[0]
        i = 0
        ret = []
        while i < K:
            mg = pr( seq[N-K-1+i] )
            x = seq[N-K+i].copy()
            x[ mask[i] ] = np.nan
            ret.append(mg.predict( x ))
            i += 1
        return np.array(ret)
        
    def fill_sequence(self, seq, mask, n_iter = 50, maxlr = 0.01):
        pr = self.dist[0]
        
        # first, run simple forward to fill the missing values 
        N, D = seq.shape 
        for i in range(N):
            cond = np.isnan(seq[i])
            if not cond.any():
                continue

            if i == 0:
                mg = pr( np.zeros(D) )
            else:
                mg = pr( seq[i-1] )
            
            pred = mg.predict( seq[i] )
            seq[i, cond] = pred[cond]
        
        # return seq
        # second, run gradient to optimize the assignment
        # 1. fix the parameters inside the nn
        for p in pr.nn.parameters():
            p.requires_grad = False
        
        # 2. create the variables
        _x,_y = mask
        maskp1 = (_x+1, _y)
        data = np.zeros( (N+1, D))
        data[1:] = seq
        data = torch.from_numpy(data).to( dtype = torch.float )
        X = data[maskp1]
        X.requires_grad = True

        # 3. define objective function 
        def objective():
            tmp = deepcopy(data)
            tmp[maskp1]  = X
            _X = tmp[0:N]
            _Y = tmp[1:]

            out = pr.nn.forward(_X)
            loss = pr.nn.loss(out, _Y)
            return loss
        
        # 4. start gradient ascent
        t = 0
        best_val = objective().item()
        best_result = deepcopy(X)
        optimizer = torch.optim.Adam([X], lr = maxlr)
        
        while t < n_iter:
            optimizer.zero_grad()
            val = objective()
            val.backward()
            optimizer.step()
            t += 1
            if val.item() < best_val:
                best_val = val.item()
                best_result = deepcopy(X)
        
        seq[mask] = best_result.detach().cpu().numpy()
        return seq


class DynamicRnnMG:
    def fit(self, data, evi_var, **kwargs):
        N = len(data)
        D = data[0].shape[1]
        _e = evi_var
        _x = utils.notin(range(D), _e)

        # create transition sequence
        ztm1 = []
        for seq in data:
            new_seq = np.zeros_like(seq)
            new_seq[1:] = seq[0:-1]
            ztm1.append(new_seq)
        zt = data

        # learn p(x^t | x^{t-1})
        pr = RecurrentNNCondMG().fit([seq[:,_x] for seq in zt],
                    [seq[:,_x] for seq in ztm1], **kwargs)

        # learn p(e^t | x^t, x^{t-1})
        if len(_e) > 0:
            pd = RecurrentNNCondMG().fit([seq[:,_e] for seq in zt],
                    [np.hstack([zt[i][:,_x], ztm1[i][:,_x]]) for i in range(N)], **kwargs)
        else:
            pd = None

        self.dist = (pr, pd)
        self.var = (_x, _e)
        return self

    def mass(self, seq, logmode = False):
        _x, _e = self.var
        pr, pd = self.dist

        zt = seq
        ztm1 = np.zeros_like(seq)
        ztm1[1:] = zt[0:-1]

        ret = pr.mass(zt[:,_x], ztm1[:,_x], logmode=1)

        if DEBUG:
            tmp = 0.0
            MGs = pr(ztm1[:,_x])
            for i,mg in enumerate(MGs):
                tmp += mg.mass(zt[i,_x], logmode = 1)
            assert( abs(tmp - ret) < 1e-3 ), '{} v.s {}'.format(tmp, ret)

        if len(_e) > 0:
            ret += pd.mass(zt[:,_e], np.hstack( [zt[:,_x], ztm1[:,_x]] ), logmode=1)
        ret = ret / seq.shape[0]

        if not logmode:
            ret = np.exp(ret)

        return ret

    def forecast(self, obs, n_step = 1):
        pr, pd = self.dist
        _x, _e = self.var
        assert( len(_e) == 0 ), "Under Implementation"
       
        N, D = obs.shape
        data = np.zeros(shape = (N+1,D))
        data[1:] = obs
        
        if n_step == 1:
            models = pr(data)[1:]
            ret = np.array([mg.mu for mg in models])
            return ret
        else:
            X = torch.from_numpy(np.array([data])).to(dtype = torch.float)
            with torch.no_grad():
                params, H = pr.nn.forward(X, with_hidden = True)
            params = [item[0] for item in params]
            Model = pr(params, auto_forward=False)[-1]
            ret = [Model.mu]

            for _ in range(n_step - 1):
                with torch.no_grad():
                    prev = torch.from_numpy( np.array([[ret[-1]]])).to(dtype = torch.float)
                    params, H = pr.nn.forward(prev, with_hidden = True, inH = H)
                params = [item[0] for item in params]
                Model = pr(params, auto_forward=False)[-1]
                ret.append(Model.mu)
            return np.vstack(ret) 

    def predict_with_evi(self, seq, mask):
        pr, pd = self.dist
        _x, _e = self.var
        assert( len(_e) == 0 ), "Under Implementation"

        N = seq.shape[0]
        K = mask.shape[0]
        obs = np.zeros_like(seq)
        obs[1:] = seq[:-1]
        
        # input is x-1 to x n-1
        # this is the distribution over x0 to xn
        models = pr(obs)[-K:]
        ret = []

        i = 0 
        while i < K:
            x = seq[N-K+i].copy()
            x [ mask[i] ] = np.nan
            ret.append( models[i].predict(x) )
            i+=1
        return np.array(ret)

    def fill_sequence(self, seq, mask, n_iter = 50, maxlr = 0.01):
        pr = self.dist[0]
        # first complete missing part using simple forward
        N,D = seq.shape
        data = np.zeros( (N+1,D)) 
        data[1:] = seq

        X = torch.from_numpy(data).to(dtype = torch.float)
        for i in range(N+1):
            cond = torch.isnan(X[i])
            if cond.any():
                P = [item[0] for item in P]
                mg = pr(P, auto_forward=False)[0]
                pred = mg.predict(seq[i-1])
                X[i] = torch.from_numpy(pred).to(dtype = torch.float)
            
            with torch.no_grad():
                if i == 0:
                    P, H = pr.nn.forward( X[i].reshape(1,1,-1), with_hidden = True)
                else:
                    P, H = pr.nn.forward( X[i].reshape(1,1,-1), with_hidden = True, inH = H)
            
        # return X[1:].numpy()

        # second, conduct gradient ascent
        # 1. fix the parameters inside the nn
        for p in pr.nn.parameters():
            p.requires_grad = False
        
        # 2. create the variables
        _x, _y = mask
        maskp1 = (_x+1, _y)
        data = X
        X = data[maskp1]
        X.requires_grad = True

        # 3. define objective function 
        def objective():
            tmp = deepcopy(data)
            tmp[maskp1]  = X
            _X = tmp[0:N].reshape(1,N,-1)
            _Y = tmp[1:].reshape(1,N,-1)

            out = pr.nn.forward(_X)
            loss = pr.nn.loss(out, _Y)
            return loss
        
        # 4. start gradient ascent
        t = 0
        best_val = objective().item()
        best_result = deepcopy(X)
        optimizer = torch.optim.Adam([X], lr = maxlr)
        
        while t < n_iter:
            optimizer.zero_grad()
            val = objective()
            val.backward()
            optimizer.step()
            t += 1
            if val.item() < best_val:
                best_val = val.item()
                best_result = deepcopy(X)
        
        seq[mask] = best_result.detach().numpy()
        return seq
        
