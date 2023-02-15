from pdb import set_trace
from copy import deepcopy
import numpy as np
import torch

import tinylib
from Gaussians import MultivariateGaussain

from utmLib import utils
from utmLib.ml.GBN import GBN
from utmLib.ml.graph import Graph
from utmLib.ml.potential import CLG


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


'''
TODO :
create GBNs with multiple connected components
and check if message passing and densities works in this case,
this can be also checked together with gbn2mg function
'''

class CondGBN:
    def fit(self, Y, X, gY):
        '''
        fit a conditional GBN where every variable y_i in Y
        condition on Par(y_i) and X
        '''
        gY.enhance()
        potential = [None] * gY.N

        for i in range(gY.N):
            parents = gY.V[i].parents
            pt = CLG()
            pt.dom = [i] + parents
            pt.fit(np.hstack([Y[:, pt.dom] , X]))
            potential[i] = pt

        self.g = gY
        self.gbn = GBN(gY).fit(Y)
        self.P = potential
        return self

    def mass(self, Y, X, logmode = False):
        '''
        calculate p(y_i|X = x_i), each row is a input
        support batch calculation (suggested for efficiency)
        '''
        ret = 0.0
        G = self.g
        for i in range(G.N):
            pt = self.P[i]
            A,b,s = pt.para
            if len(pt.dom) > 1:
                _cond = pt.dom[1:]
                V = np.hstack([Y[:,_cond], X])
            else:
                V = X
            mu = V.dot( A.reshape(-1,1) ).flatten() + b
            vec = Y[:,i] - mu
            density = -0.5 * (np.multiply(vec,vec) / s + np.log(2*s*np.pi))
            ret = density + ret

        if not logmode:
            ret = np.exp(ret)
        return ret

    def _predict_one_(self, y, x):
        """Compute MPE assignment for unknown variables (with value np.nan)
        in y given assignment of condition variables X=x.

        Parameters
        ----------
        y : np.ndarray
            the partially observed values for Y variables ( can be all nan)
        x : np.ndarray
            the assignment to contional variable X

        Returns
        -------
        np.ndarray
            the MPE assignment for unknown Y variables
        """

        G = self.g
        x_size = x.size
        unk_idx = np.isnan(y).nonzero()[0]

        if not len(unk_idx):
            # not unknown variables, directly return
            return y

        # obtain the distribution p(Y|X=x)
        # it is represented using a GBN
        for i in range(G.N):
            A,b,s = self.P[i].para
            xA = A[-x_size:]
            b += np.dot(xA, x)
            self.gbn.potential[i].para = (A[:-x_size], b, s)

        # convert the GBN into MG for inference
        mg = MultivariateGaussain.GBN2MG(self.gbn)
        if len(unk_idx) < G.N:
            # we have partial evidence
            y_pred = mg.predict(y)
        else:
            # all of them are unknown, prediction is just mean
            y_pred = mg.mu.copy()

        return y_pred

    def predict(self, Y, X):
        N = X.shape[0]
        assert( N == Y.shape[0]), "Input size not match"
        return np.array([self._predict_one_(Y[i], X[i]) for i in range(N)])


def loss_func(nn_out, ydata, G, return_mass = False):
    A,B,P = nn_out
    selector = np.concatenate([G.V[i].parents for i in range(G.N)]).astype(int)
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


class DynamicGBN:
    def fit(self, data, evi_var, max_parents = 1, corr_thresh = 0.0):
        '''
        data - a list of sequence (np.ndarray), the length of sequence might differ
        works even when evi_var is empty
        '''
        D = data[0].shape[1]
        _e = evi_var
        _x = utils.notin(range(D), _e)

        # learn p(z0)
        Z = np.vstack(data)
        g = create_graph(Z, max_parents, corr_thresh)
        z0 = np.array([seq[0] for seq in data])
        p0 = GBN(g).fit(z0)

        # learn transition distribution
        ztm1 = []
        for seq in data:
            new_seq = np.zeros_like(seq)
            new_seq[1:] = seq[0:-1]
            ztm1.append(new_seq)

        ztm1 = np.vstack(ztm1)
        zt = np.vstack(data)
        pt = CondGBN().fit(zt, ztm1[:,_x], g)
        # pt = CondGBN().fit(zt, ztm1, g)

        self.dists = (p0, pt)
        self.vars = (_x,_e)
        return self


    def mass(self, seq, logmode = False):
        '''
        only one sequence a time
        '''
        p0, pt = self.dists
        _x, _e = self.vars

        ret = 0.0
        # ret += np.log(p0.mass(seq[0]))
        # ztm1 = seq[0:-1]
        # zt = seq[1:]
        ztm1 = np.zeros_like(seq)
        ztm1[1:] = seq[0:-1]
        zt = seq

        # ret += np.sum(pt.mass(zt, ztm1, logmode = 1))
        ret += np.sum(pt.mass(zt, ztm1[:,_x], logmode = 1))
        ret = ret/seq.shape[0]

        if not logmode:
            ret = np.exp(ret)
        return ret

    def forecast(self, prev_obs, n_step = 1):
        """Predict the future outcomes given previous observation

        Args:
            prev_obs (np.array): previous obvious z_0 to z_{t-1}

        Returns:
            np.array: predicted outcome z_1 to z_t
        """

        pt = self.dists[1]
        _x = self.vars[0]
        if n_step == 1:
            pred = np.zeros_like(prev_obs)
            pred[:] = np.nan
            X = prev_obs[:, _x]
            ret = pt.predict(pred, X)
        else:
            prev = prev_obs[-1]
            ret = []
            for _ in range(n_step):
                tmp = np.zeros_like(prev)
                tmp[:] = np.nan
                new = pt._predict_one_(tmp, prev[_x])
                ret.append(new)
                prev = new
            ret = np.array(ret)
        return ret
    
    def vectorize(self):
        pt = self.dists[1]
        _x = self.vars[0]
        cond_size = len(_x)

        Ay = []
        Ax = []
        B = []
        S = []

        for i in range(pt.g.N):
            A,b,s = pt.P[i].para
            B.append(b)
            S.append(s)
            Ax.append( A[-cond_size:] )
            Ay.append( A[:-cond_size] )

        Ay = np.concatenate(Ay).reshape(1,-1)
        B = np.array(B).reshape(1,-1)
        S = np.array(S).reshape(1,-1)
        Ax = np.array(Ax)
        self._vec_ = (Ay,B,S,Ax)
        return 

    def fill_sequence(self, seq, mask, n_iter = 50, maxlr = 0.01):
        # transform all things into vector form 
        if not hasattr(self, '_vec_'):
            self.vectorize()

        _x = self.vars[0]
        N, D = seq.shape 
        data = np.zeros(shape = (N+1, D))
        data[1:] = seq

        Ay,B,S,Ax = self._vec_
        B = deepcopy(np.broadcast_to(B, (N,D)))
        S = deepcopy(np.broadcast_to(S, (N,D)))
        Ay = deepcopy(np.broadcast_to(Ay, (N,Ay.size)))
        
        G = self.dists[1].g
        B = torch.from_numpy( B ).to(dtype = torch.float)
        S = torch.from_numpy( 1.0/S ).to(dtype = torch.float)
        Ay = torch.from_numpy( Ay ).to(dtype = torch.float)
        Ax = torch.from_numpy( Ax ).to(dtype = torch.float)

        # first, run simple forward prediction
        pt = self.dists[1]
        for i in range(N+1):
            cond = np.isnan(data[i])
            if cond.any():
                data[i] = pt._predict_one_(data[i], data[i-1][_x])
        
        # return data[1:]
        # print( self.mass(data[1:], logmode=1) )

        # second, conduct GA using the vector form
        # 1. create the variables
        _mx, _my = mask
        maskp1 = (_mx+1, _my)
        data = torch.from_numpy(data).to( dtype = torch.float )
        X = data[maskp1]
        X.requires_grad = True

        # 3. define objective function 
        def objective():
            tmp = deepcopy(data)
            tmp[maskp1]  = X
            _X = tmp[0:N, _x]
            _Y = tmp[1:]

            newB = B + _X @ Ax.T
            loss = loss_func((Ay, newB, S), _Y, G)
            return loss
        
        # 4. start gradient ascent
        t = 0
        best_val = objective().item()
        best_result = deepcopy(X)
        optimizer = torch.optim.Adam([X], lr = maxlr)
        # print(best_val)

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

