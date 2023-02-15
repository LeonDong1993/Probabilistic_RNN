from pdb import set_trace
import numpy as np
import torch.multiprocessing as mp
import torch, copy

from scipy.special import logsumexp
from functools import partial
from utmLib import utils
from .core import RNNCondIndMG
DEBUG = 0

if DEBUG:
    print('== Currently In DEBUG mode ==')


def _mp_model_predict(item, obj, n_step = 1):
    return obj.forecast(item, n_step = n_step)

class DynamicRNNIndMG:
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
        pr = RNNCondIndMG().fit([seq[:,_x] for seq in zt],
                    [seq[:,_x] for seq in ztm1], **kwargs)

        # learn p(e^t | x^t, x^{t-1})
        if len(_e) > 0:
            pd = RNNCondIndMG().fit([seq[:,_e] for seq in zt],
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
            models = pr(ztm1[:,_x])
            for i,M in enumerate(models):
                tmp += M.mass(zt[i,_x], logmode = 1)
            assert( abs(tmp - ret) < 1e-3 ), "Loglikelihood computation error."

        if len(_e) > 0:
            ret += pd.mass(zt[:,_e], np.hstack( [zt[:,_x], ztm1[:,_x]] ), logmode=1)
        ret  = ret / seq.shape[0]

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
            ret = np.array([M.predict(None) for M in models])
            return ret 
        else:
            X = torch.from_numpy(np.array([data])).to(dtype = torch.float)
            with torch.no_grad():
                params, H = pr.nn.forward(X, with_hidden = True)
            params = [item[0] if item is not None else None for item in params]
            Model = pr(params, auto_forward=False)[-1]
            ret = [Model.predict(None)]

            for _ in range(n_step - 1):
                with torch.no_grad():
                    prev = torch.from_numpy( np.array([[ret[-1]]]))
                    params, H = pr.nn.forward(prev, with_hidden = True, inH = H)
                params = [item[0] if item is not None else None for item in params]
                Model = pr(params, auto_forward=False)[-1]
                ret.append(Model.predict(None))
            return np.vstack(ret)

    def forecast_wrapper(self, obs_list, n_step = 1, parallel = 1):
        if parallel == 1:
            ret = [self.forecast(x) for x in obs_list]
        else:
            assert(parallel > 0)
            func = partial(_mp_model_predict, obj = self, n_step = n_step)
            chk_size = int(len(obs_list) / parallel) + 1
            with mp.get_context('spawn').Pool(processes = parallel) as pool:
                ret = list(pool.imap(func, obs_list, chk_size))
        return ret
         
    def particle_filtering(self, obs, n_step, n_particles=100):
        pr, pd = self.dist
        _x, _e = self.var
        assert( len(_e) == 0 ), "Under Implementation"
        assert( n_step > 1 ), "User forecast function for 1-step prediction"

        # construct padded data
        N, D = obs.shape
        data = np.zeros(shape = (N+1,D))
        data[1:] = obs

        # pass data into model, obtain h_{t+1}
        X = torch.from_numpy(np.array([data])).to(dtype = torch.float)
        with torch.no_grad():
            params, H = pr.nn.forward(X, with_hidden = True)
        params = [item[0] if item is not None else None for item in params]
        pt_1 = pr(params, auto_forward=False)[-1]

        # start particle filtering
        samples = pt_1.rvs(n_particles)
        masses = pt_1.mass(samples, logmode=1)
        pathes = [[i] for i in range(n_particles)]
        traces = [samples]
        caches = {}

        for s in range(n_step - 1):
            j = 0 # the sample index 
            new_mass = []
            new_samples = []
            new_pathes = []
            # sample_dist = np.exp(masses)
            # sample_dist = sample_dist/np.sum(sample_dist)
            # the above way of calculating sample dist has chance to get underflow
            # sample_dist = masses - np.mean(masses)
            # sample_dist = np.exp(sample_dist)
            # sample_dist = sample_dist/np.sum(sample_dist)
            # this ways still have overflow underflow issue.
            # this gurantee the sample_dist has total probability 1.0
            sample_dist = masses - logsumexp(masses)
            sample_dist = np.exp(sample_dist)

            chosed_ind = np.random.choice(n_particles, size=n_particles, replace=True, p = sample_dist)
            unique_ind, ind_counts = np.unique(chosed_ind, return_counts=True)
            for i,n in zip(unique_ind, ind_counts):
                if s > 0:
                    prev_H = caches[s-1,i]
                else:
                    prev_H = H
                
                x = torch.from_numpy(samples[i].reshape(1,1,-1)).to(dtype=torch.float)
                with torch.no_grad():
                    params, new_H = pr.nn.forward(x, with_hidden=True, inH = prev_H)
                params = [item[0] if item is not None else None for item in params]
                cur_model = pr(params, auto_forward=False)[0]
                cur_samples = cur_model.rvs(n+1)
                cur_mass = cur_model.mass(cur_samples, logmode=1)
                for _i in range(n):
                    _s = cur_samples[_i]
                    new_pathes.append( pathes[i] + [j] )
                    new_samples.append(_s)
                    caches[s,j] = new_H
                    new_mass.append( masses[i] + cur_mass[_i] )
                    j += 1

            masses = np.array(new_mass)
            samples = new_samples
            pathes = new_pathes
            traces.append(samples)

        ind = np.argmax(masses)
        # print(masses[ind])
        path = pathes[ind]
        ret = np.zeros(shape = (n_step, D))
        for i,j in enumerate(path):
            ret[i] = traces[i][j]

        return ret

    def gradient_ascent(self, obs, init, n_iter = 50, maxlr = 0.01):
        pr, pd = self.dist
        _x, _e = self.var
        assert( len(_e) == 0 ), "Under Implementation"

        N, D = init.shape
        assert( N > 1 ), "User forecast function for 1-step prediction"

        # construct padded data
        N, D = obs.shape
        data = np.zeros(shape = (N+1,D))
        data[1:] = obs

        # pass data into model, obtain h_{t+1}
        data = torch.from_numpy(np.array([data])).to(dtype = torch.float)
        with torch.no_grad():
            params, H0 = pr.nn.forward(data, with_hidden = True)
        
        # extract the parameters of last time step
        _tmp = []
        for item in params:
            if item is None:
                _tmp.append(None)
            else:
                if len(item.shape) > 3:
                    _tmp.append(item[:, :, -1:])
                else:
                    _tmp.append(item[:, -1:])
        params0 = _tmp

        # create the variable, 
        init_copy = copy.copy(init)
        X = torch.from_numpy(init_copy).to(dtype = torch.float)
        X.requires_grad = True

        # fixed all parameters inside nn 
        nn_model = pr.nn
        for p in nn_model.parameters():
            p.requires_grad = False

        # define objective function
        def objective():
            N, _ = X.shape
            param = params0
            H = H0
            ret = 0
            for i in range(N):
                ret += nn_model.loss(param, X[i])
                if i < N-1: 
                    param, H = nn_model.forward(X[i].reshape(1,1,-1), with_hidden=True, inH = H)
            # this value is the negative LL of the whole assignment
            return ret
        
        # conduct gradient descent on the objective is enough
        # keep track what we found best so far
        # lr_func = lambda e: 1/(1+0.25*e)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
        optimizer = torch.optim.Adam([X], lr = maxlr)

        t = 0
        best_val = objective().item()
        best_result = copy.copy(X.data.numpy())
        while t < n_iter:
            optimizer.zero_grad()
            val = objective()
            val.backward()
            optimizer.step()
            # scheduler.step()
            t += 1
            if val.item() < best_val:
                # print('Update!')
                best_val = val.item()
                best_result = copy.copy(X.data.numpy())
        # set_trace()
        return best_result
    
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
                P = [item[0] for item in P]  # this only works if more than 1 component
                mixmg = pr(P, auto_forward=False)[0]
                pred = mixmg.predict(seq[i-1])
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
            tmp = copy.deepcopy(data)
            tmp[maskp1]  = X
            _X = tmp[0:N].reshape(1,N,-1)
            _Y = tmp[1:].reshape(1,N,-1)

            out = pr.nn.forward(_X)
            loss = pr.nn.loss(out, _Y)
            return loss
        
        # 4. start gradient ascent
        t = 0
        best_val = objective().item()
        best_result = copy.deepcopy(X)
        optimizer = torch.optim.Adam([X], lr = maxlr)
        
        while t < n_iter:
            optimizer.zero_grad()
            val = objective()
            val.backward()
            optimizer.step()
            t += 1
            if val.item() < best_val:
                best_val = val.item()
                best_result = copy.deepcopy(X)
        
        seq[mask] = best_result.detach().numpy()
        return seq


class Bagged_RNNIndMG:
    def fit(self, data, evi_var, n_estimator = 3, **kwargs):
        models = []
        N_data = len(data)
        for _ in range(n_estimator):
            selected = np.random.choice(N_data, size = N_data, replace=True)
            sub_data = [data[i] for i in selected]
            M = DynamicRNNIndMG().fit(sub_data, evi_var, **kwargs)
            models.append(M)
        
        selected = np.linspace(0, N_data-1, num = min(N_data, 500)).astype(int)
        
        weight = np.array([sum([M.mass(data[i]) for i in selected]) for M in models])
        abs_sum = np.sum(np.abs(weight))
        weight = np.sqrt( np.exp( weight/abs_sum) )
        weight = weight / np.sum(weight)
        
        self.W = weight
        self.models = models
        # assert(min(self.W) >= 0)
        return self

    def mass(self, seq, logmode = False):
        component_mass = np.array([M.mass(seq, logmode) for M in self.models])
        ret = sum(component_mass * self.W)
        return ret

    def forecast_wrapper(self, obs_list, parallel = 1):
        preds = []
        for M in self.models:
            preds.append(M.forecast_wrapper(obs_list, parallel))
        
        N = len(obs_list)
        K = len(self.models)
        ret = []
        for i in range(N):
            cur = 0
            for k in range(K):
                cur = self.W[k] * preds[k][i] + cur 
            ret.append(cur)

        return ret



