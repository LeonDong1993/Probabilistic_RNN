import pdb
import warnings
from functools import partial

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from scipy.stats import multivariate_normal
from scipy.special import logsumexp


import torch
import torch.multiprocessing as mp
torch.set_num_threads(1)

def _mp_mixmg_predict(args, fixed):
    # a wrapper function to achieve parallel
    # args - a single x
    # fixed, the object as well as max_iter
    x = args
    obj, max_iter = fixed
    return obj._predict_(x, max_iter)

# some functions for self test
def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

# test if x is the local maximum of function f
def is_local_max(f, x, dom, scale = 0.5, N = 500, tol = 0.05):
    """Test if x is the local maximum of function f with tolerance

    Args:
        f (object): the callable function object
        x (np.ndarray): the assumed local maximum
        dom (array-like): the dimension we test against with
        scale (float, optional): perturbation scale . Defaults to 0.5.
        N (int, optional): number of trials. Defaults to 500.
        tol (float, optional): tolerance level. Defaults to 0.05.

    Returns:
        bool: Ture if it is local maximum (approximately)
    """
    assert(tol <= 0.1)
    val_max = f(x) * (1+tol)
    y = x.copy()
    for _ in range(N):
        perturbation = (np.random.rand( len(dom) ) - 0.5) * 2 * scale
        y[dom] = x[dom] + perturbation
        if f(y) >= val_max:
            return False
    return True


class MultivariateGaussain:
    def __init__(self):
        self.mass_correction = -100
        self.logpdf = multivariate_normal.logpdf

    def fit(self, data, covfix = 1e-6):
        self.mu = np.mean(data, axis = 0)
        M = np.cov(data, rowvar = False, bias = False)
        if covfix > 0:
            np.fill_diagonal(M, M.diagonal() + covfix)
        self.S = M
        return self

    def mass(self, X, logmode = False):
        try:
            ret = self.logpdf(X, mean = self.mu, cov = self.S, allow_singular=True)
        except Exception:
            warnings.warn("MultivariateGaussain mass computation error")
            if X.ndim == 2:
                ret = np.array([-self.mass_correction] * X.shape[0])
            else:
                ret = -self.mass_correction

        if not logmode:
           ret = np.exp(ret)
        return ret

    @staticmethod
    def GBN2MG(gbn):
        """Convert a Gaussian Bayesian Network into a Multivariate Gaussian Object
           Sparse GBN and pure independent GBN supported as well

        Args:
            gbn (utmLib.ml.GBN.GBN): fitted Gaussian Bayesian Network object 

        Returns:
            MultivariateGaussain object: the corresponding MG represent the same distribution
        """
        node_num = gbn.g.N
        mu = np.zeros(shape = (node_num, ))
        S = np.zeros(shape = (node_num, node_num))

        for i,nid in enumerate(gbn.g._topo_order_):
            prev_nodes = gbn.g._topo_order_[0:i]
            parents = gbn.g.V[nid].parents
            A, b, s = gbn.potential[nid].para

            if A is None:
                mu[nid] = b
                S[nid,nid] = s
            else:
                # support sparse gbn
                if len(parents) < len(prev_nodes):
                    # ind_par = utils.notin(prev_nodes, parents)
                    ind_par = list(np.setdiff1d(prev_nodes, parents))
                    parents = parents + ind_par
                    A = np.concatenate([A, np.zeros(len(ind_par))])

                mu[nid] = b + np.sum(A * mu[parents])
                Sp = S[np.ix_( parents, parents)]
                beta = A.reshape(-1,1)
                S[nid,nid] = s + beta.T @ Sp @ beta

                Sx = S[np.ix_(prev_nodes, parents)]
                cov_vec = Sx @ beta
                S[np.ix_([nid], prev_nodes)] = cov_vec.T
                S[np.ix_(prev_nodes, [nid])] = cov_vec

        obj = MultivariateGaussain()
        obj.mu = mu
        obj.S = S
        return obj

    @staticmethod
    def CLG(mu, sigma, var_idx, cond_idx = None):
        ''' Compute the parameters of Conditional Linear Gaussian P(Y|X) given the
        parameters of Joint Gaussian distribution over Y and X. This function is 
        numerical stable.

        Args:
        mu: array-like
            the mean vector of the joint gaussain distribution 
        sigma: array-like
            the covariance matrix of the joint gaussain distribution 
        var_idx: array-like
            the ids of Y variables, the order matters
        cond_idx: array-like, optional
            the ids of X variables, the order matters
            if None is given, it will be automatically set to Not(Y) in increasing order

        Returns:
        1. the coef vector A (np.array)
        2. the bias vector B (np.array)
        3. the cov matrix S (np.array)
        '''

        if mu.size == 1:
            A = None
            B = float(mu)
            S = float(sigma)
        else:
            if cond_idx is None:
                # cond_idx = np.array( utils.notin(range(mu.size), var_idx) )
                cond_idx = np.setdiff1d(range(mu.size), var_idx)
            mu = mu.reshape(-1,1)
            mu1 = mu[var_idx,:]
            mu2 = mu[cond_idx, :]
            s11 = sigma[np.ix_(var_idx, var_idx)]
            s12 = sigma[np.ix_(var_idx, cond_idx)]
            s21 = s12.T
            s22 = sigma[np.ix_(cond_idx, cond_idx)]
            # fix explicit calculation of inverse @ 09-25-21 15:05
            # Orginal to calculate: A = s12 @ inv(S22)
            # convert into A * S22 = s12 --> S22 * (A.T) = s12.T
            A = np.linalg.lstsq(s22, s21, rcond=None)[0].T
            B = mu1 - A.dot(mu2)
            S = s11 - A.dot(s21)
        return A,B,S

    def rvs(self, N = 1, evi = None):
        """Generate samples from this MG distribution, partial evidence supported

        Args:
            N (int, optional): number of samples to generate. Defaults to 1.
            evi (np.ndarray or None, optional): the partial evidence given as an array, 
                if i^th entry is np.nan, it considered as missing. Defaults to None.
                If all entry is missing, set evi to None.
                Also, there should be at least one entry in evi that is np.nan

        Returns:
            np.ndarray: generated full samples corresponds to evidence.
                it's a 2D array if N > 1, otherwise, 1D array is returned
        """

        if evi is None:
            evi_idx = np.array([])
            var_idx = np.arange(self.mu.size)
        else:
            cond_array = np.isnan(evi)
            var_idx = cond_array.nonzero()[0]
            evi_idx = (~cond_array).nonzero()[0]

        if evi_idx.size == self.mu.size:
            warnings.warn("All variables have given value, no need to generate sample.")
            return evi

        if evi_idx.size == 0:
            full_samples = multivariate_normal.rvs(mean = self.mu, cov = self.S, size = N)
        else:
            A,B,S = MultivariateGaussain.CLG(self.mu, self.S, var_idx, evi_idx)
            evi_val = evi[evi_idx].reshape(-1,1)
            mu = A.dot(evi_val) + B
            partial_samples = multivariate_normal.rvs(mean = mu.flatten(), cov = S, size = N)
            full_samples = np.zeros(shape = (N, self.mu.size))
            full_samples[:,var_idx] = partial_samples.reshape(N,-1)
            full_samples[:,evi_idx] = evi_val.reshape(1,-1)
            # preserve shape
            if N == 1: full_samples = full_samples[0]

        return full_samples

    def predict(self, test):
        """ Fill up the missing value (np.nan) in test data with MPE values

        Args:
            test (np.ndarray): partial observed data, have nan inside

        Returns:
            np.ndarray: predicted data without nan values 
        """
        if test is None:
            test = np.zeros_like(self.mu)
            test[:] = np.nan

        if len(test.shape) == 1:
            test = test.reshape(1,-1)

        self._cache_ = {}
        ret = np.array( [self._predict_(x) for x in test] )
        del self._cache_

        # preserve input shape
        if ret.shape[0] == 1:
            ret = ret[0]
        return ret

    def _predict_(self, x):
        cond = np.isnan(x)
        evi = (~cond).nonzero()[0]
        unk = cond.nonzero()[0]

        # no evidence case, simply return the mean vector
        if evi.size == 0:
            return self.mu.copy()
        
        if unk.size == 0:
            warnings.warn("Nothing to predict, all evidence given!")
            return x 

        key = tuple(unk)
        if key in self._cache_:
            A,B,S = self._cache_[key]
        else:
            A,B,S = MultivariateGaussain.CLG(self.mu, self.S, unk, evi)
            self._cache_[key] = (A,B,S)

        evi_val = x[evi].reshape(-1,1)
        mu = A.dot(evi_val) + B
        ret = x.copy()
        ret[unk] = mu.flatten()
        return ret

    def marginalize(self, rv_ids):
        mu = self.mu[rv_ids]
        S = self.S[np.ix_(rv_ids, rv_ids)]
        obj = MultivariateGaussain()
        obj.mu = mu
        obj.S = S
        return obj

    @staticmethod
    def self_test():
        N_vars = 6
        mu = np.random.rand(N_vars)
        while True:
            S  = np.random.rand(N_vars, N_vars)
            S = S + S.T
            if is_pos_def(S):
                break

        mg = MultivariateGaussain()
        mg.mu = mu
        mg.S = S
        print("Init done.")

        # test CLG and prediction function simutaneously
        N_sample = 100
        unknowns = []
        samples = np.random.rand(N_sample, N_vars)

        for i in range(N_sample):
            unk = np.random.choice(N_vars, size = int(N_vars/2), replace = False)
            unknowns.append(unk)
            samples[i, unk] = np.nan

        predicted = mg.predict(samples)
        for i,p in enumerate(predicted):
            assert( is_local_max(mg.mass, p, unknowns[i]) )
        print('Passed prediction test')

        # test the rvs function
        for idx in [3, 7, 9]:
            x = predicted[idx]
            unk = unknowns[idx]
            y = x.copy() 
            y[unk] = np.nan
            samples = mg.rvs(1000000, evi = y)
            mean_hat = np.mean(samples, axis = 0)
            print('Estimated:{} Expected:{}'.format( mean_hat[unk], x[unk] ))
        
        # test sample and refit
        data = mg.rvs(int(1e6), evi=None)
        mg_re = MultivariateGaussain().fit(data)
        
        print("Original mu and covariance matrix is:")
        print(mg.mu)
        print(mg.S)

        print("Fitted mu and covariance matrix is:")
        print(mg_re.mu)
        print(mg_re.S)
        
        return mg


class MixMG:
    def fit(self, data, n_comps = -10, cov_type = 'full', tol = 0.03, verbose = False):
        """Fit a Mixture of MG using cross validation to determine the number of components

        Args:
            data (np.ndarray): training data where each row is a data point
            n_comps (int, optional): if less than 0, using cross validation where the
                number of components is from [-n_comps, 2]. if greater than 0, use given 
                values directly and no CV procedure. Defaults to -10.
            cov_type (str, optional): Covariance matrix type for GMM model. Defaults to 'full'.
            tol (float, optional): score difference toleration. Defaults to 0.03.
            verbose (bool, optional): if set to True, show auto determined number of components. 
                Defaults to False.

        Returns:
            self: this object
        """

        assert(isinstance(n_comps, int) and n_comps != 0)
        assert(tol <= 0.1), "Too high tolerance!"
        kwargs = {'covariance_type':cov_type, 'n_init':3, 'init_params':'kmeans', 'warm_start':True}

        if n_comps < 0:
            assert( n_comps <= -2 ), "Nothing to tune, pass positive integer for n_comps"
            cv_score = []
            kf = KFold(n_splits=5, shuffle=True)
            fold_idx = list(kf.split(data))
            for n in range(2, 1-n_comps):
                gmm = GaussianMixture(n_components = n, **kwargs)
                tmp = [gmm.fit(data[_t]).score(data[_v]) for _t,_v in fold_idx]
                cv_score.append( np.mean(tmp) - 0.05*np.std(tmp) )

            # favor less number of components if score does not differs too much
            min_score, max_score = min(cv_score), max(cv_score)
            cv_score = (np.array(cv_score) - min_score) / (max_score - min_score)
            cand_idx = (cv_score >= (1-tol)).nonzero()[0]
            n_comps = np.min(cand_idx) + 2

        if verbose:
            print('Fit a MixMG with {} components'.format(n_comps))

        m = GaussianMixture(n_components = n_comps, **kwargs).fit(data)
        self.models = []
        for i in range(n_comps):
            mu = m.means_[i]
            sigma = m.covariances_[i]
            mg = MultivariateGaussain()
            mg.mu = mu
            mg.S = sigma
            self.models.append(mg)
        self.W  = m.weights_
        return self

    def mass(self, X, logmode = 0):
        """Compute the (log) density of input data, this function has been
           verified against naive non-vectorized implmentaion. 

        Args:
            X (np.ndarray): Input data, can be 2D or 1D (i.e. 1 sample)
            logmode (bool, optional): if True, retrurn log(P) instead of P. Defaults to 0.

        Returns:
            np.ndarray or scaler: the (log) density of input
        """
        K = len(self.W)
        comp_ll = np.array([m.mass(X, logmode = 1) for m in self.models])
        comp_ll = comp_ll.reshape(K, -1)
        log_w = np.log(self.W).reshape(K,1)
        
        ret = logsumexp( comp_ll + log_w, axis = 0)
        if not logmode:
            ret = np.exp(ret)
        
        if ret.size == 1:
            # turn into a single float number
            # if only one sample is provided
            ret = ret[0]
        return ret

    def rvs(self, N=1, evi=None):
        """Generate samples from this MixMG distribution, partial evidence supported

        Args:
            N (int, optional): number of samples to generate. Defaults to 1.
            evi (np.ndarray or None, optional): the partial evidence given as an array, 
                if i^th entry is np.nan, it considered as missing. Defaults to None.
                If all entry is missing, set evi to None.
                Also, there should be at least one entry in evi that is np.nan

        Returns:
            np.ndarray: generated full samples corresponds to evidence.
                it's a 2D array if N > 1, otherwise, 1D array is returned
        """

        comp_ids = np.random.choice(self.W.size, size = N, p = self.W, replace = True)
        val, idcounts = np.unique( comp_ids, return_counts=True)
        
        sample_batches = []
        for cid, count in zip(val,idcounts):
            samples = self.models[cid].rvs(count, evi)
            sample_batches.append(samples.reshape(count, -1))
        ret = np.vstack(sample_batches)
        
        # preserve the input shape 
        if N == 1: ret = ret[0]
        return ret

    # def gradient(self, x, i):
    #     def _npd_(func, v, i):
    #         # numerical partial derivative
    #         def wrap_func(x):
    #             item = v.copy()
    #             item[i] = x
    #             return func(item)
    #         return derivative(wrap_func, v[i], dx = 1e-5)
    #     return _npd_(self.mass, x, i)

        # exact gradient suffers from inverse, give complete wrong solution
        # ret = 0
        # for i, mg in enumerate(self.models):
        #     vec = (mg.mu - x).reshape(-1,1)
        #     ret = ret + self.W[i] * np.linalg.inv(mg.S).dot(vec)
        # return ret.flatten()

    def marginalize(self, rv_ids):
        obj = MixMG()
        obj.models = [m.marginalize(rv_ids) for m in self.models]
        obj.W = self.W.copy()
        return obj

    def predict(self, X, max_iter = 50, parallel = 1):
        """ Fill out missing values in X through MPE inference 

        Args:
            X (np.ndarray): partial observed data with missing values
            max_iter (int, optional): Max number of iteration to conduct GA. Defaults to 50.
            parallel (int, optional): How many cores to use for prediction. Defaults to 1.

        Returns:
            np.ndarray: full array with predicted values 
        """

        if not hasattr(self, 'tensor_'):
            self._predict_setup_()

        # predict all variable
        if X is None:
            X = np.zeros_like(self.models[0].mu, dtype=float)
            X[:] = np.nan
        
        # handles vector input
        if len(X.shape) == 1:
            X = X.reshape(1,-1)

        # setup prediction cache manully
        for m in self.models:
            m._cache_ = {}

        # conduct prediction tasks
        if parallel == 1:
            ret = [self._predict_(x, max_iter) for x in X]
        else:
            assert(parallel > 0)
            func = partial(_mp_mixmg_predict, fixed = (self, max_iter))
            chk_size = int(X.shape[0] / parallel) + 1
            with mp.get_context('spawn').Pool(processes = parallel) as pool:
                ret = list(pool.imap(func, X, chunksize = chk_size))
            
        # clean component prediction cache
        for m in self.models:
            del m._cache_

        # preserve input shape
        if len(ret) == 1:
            ret = ret[0]
        else:
            ret = np.array(ret)
        return ret

    def _predict_setup_(self, **kwargs):
        # set up neccessary tensor used for pytorch to do optimization 
        N = self.W.size
        try:
            list_mu = [torch.from_numpy(self.models[i].mu).to(dtype=torch.float) for i in range(N)]
            list_P = [torch.from_numpy(np.linalg.inv(self.models[i].S)).to(dtype=torch.float) for i in range(N)]
            cov_det = [np.linalg.det(2*np.pi*self.models[i].S) for i in range(N)]
            assert(min(cov_det) > 0)
            constant = np.array([1.0/np.sqrt( val ) for val in cov_det])
            constant = torch.log(torch.from_numpy(constant).to(dtype=torch.float))
            weight = torch.log( torch.from_numpy(self.W).to(dtype=torch.float))
            self.tensor_ = (list_mu, list_P, constant, weight)
            self.pred_args_ = kwargs
        except:
            self.tensor_ = None

    def _predict_(self, x, max_iter = 50):
        """ Fill out the missing values in one sample point through MPE
            Suggested max_iter ~ 20 * sqrt(#Features)
        Args:
            x (np.ndarray): a single data point with missing values 
            max_iter (int, optional): if less than 0, no GA will be used. Defaults to 50.

        Returns:
            array: predicted sample without nan values 
        """

        N = self.W.size
        cond = np.isnan(x)
        unk = cond.nonzero()[0]

        if unk.size == 0:
            warnings.warn("Nothing to predict, all evidence given!")
            return x 

        # construct candidate initial point
        init_points = [m._predict_(x) for m in self.models]
        init_mass = self.mass(init_points, logmode=1)
        ind = np.argmax(init_mass)
        best_val = init_mass[ind]
        best_pred = init_points[ind]

        # if we have singular component or max_iter is zero
        # return the best one in those init_points, no GA applied
        if self.tensor_ is None or max_iter <= 0:
            return best_pred

        # remove some useless init points for perf speedup
        # M = np.exp(init_mass)
        # M = M / M.sum()
        # selected = np.unique( np.random.choice(N, size=N, replace=True, p = M) )
        # init_points = [init_points[i] for i in selected]

        # conduct GA using pytorch optim
        U,P,C,W = self.tensor_

        if 'maxlr' in self.pred_args_:
            maxlr = self.pred_args_['maxlr']
        else:
            maxlr = 1.0
        
        if 'lr_func' in self.pred_args_:
            lr_func = self.pred_args_['lr_func']
        else:
            lr_func = lambda e: 1/(1+0.25*e)

        # basically the loglikelihood
        def objective(p):
            y = torch.from_numpy(x).to(dtype=torch.float)
            y[unk] = p
            comp_ll = torch.zeros(N)
            for i in range(N):
                vec = (y - U[i]).reshape(-1,1)
                comp_ll[i] = -0.5 * (vec.T @ P[i] @ vec)
            comp_ll = comp_ll + C + W
            LL = torch.logsumexp(comp_ll, dim=0)
            return LL, y

        for point in init_points:
            z = torch.nn.Parameter( torch.from_numpy(point[unk]).to(dtype=torch.float) )
            optimizer = torch.optim.Adam([z], maximize = True, lr = maxlr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
            
            t = 0
            while t < max_iter:
                optimizer.zero_grad()
                val, _z = objective(z)
                val.backward()
                optimizer.step()
                scheduler.step()
                
                t += 1
                if z.grad.abs().max() < 1e-3:
                    break
                if val > best_val:
                    best_val = val
                    best_pred = _z.data.numpy()
                    # double verify the correctness
                    ll_ = self.mass(best_pred, logmode=1)
                    diff = abs(ll_ - val.item())
                    pct_deviation = diff / abs(ll_) 
                    if pct_deviation > 0.05:
                        warnings.warn("One of the component might has near singular covariance matrix.")
        return best_pred
   
    @staticmethod
    def self_test():
        from utmLib.clses import Timer, ProgressIndicator
        N_comps = 3
        N_vars = 5
        scale = 3
        clock = Timer()

        MGs = []
        for _ in range(N_comps):
            mu = np.random.rand(N_vars) * scale 
            while 1:
                S = np.random.rand(N_vars, N_vars) * scale
                S = S + S.T
                if is_pos_def(S):
                    break
            mg = MultivariateGaussain()
            mg.mu = mu
            mg.S = S
            MGs.append(mg)
        weights = np.random.rand(N_comps)
        weights /= np.sum(weights)

        # create a random MixMG distribution
        mixmg = MixMG()
        mixmg.models = MGs
        mixmg.W = weights
        clock.ring("Init done.")

        # conduct some prediction task
        mode = mixmg.predict(X = None)
        assert( is_local_max(mixmg.mass, mode, list(range(N_vars))) )
        print("Predicted mode LL is: {}".format(mixmg.mass(mode, logmode=1)))
        data = mixmg.rvs(N = int(1e5))
        print('Best sample LL is: {}'.format(mixmg.mass(data, logmode=1).max()) )
        clock.ring()
        
        K = 200
        data = mixmg.rvs(N=K)
        unknown = []
        for x in data:
            unk = np.random.choice(N_vars, size = 4, replace=True)
            x[unk] = np.nan
            unknown.append(unk)
        
        pred = mixmg.predict(data, parallel=4)
        clock.ring("Predition Done.")

        pi = ProgressIndicator(K, msg="Verifying")
        for i in range(K):
            unk = unknown[i]
            pi.at(i+1)
            assert( is_local_max(mixmg.mass, pred[i], unk) )
        clock.ring("Passed prediction task verification.")

        # gerneate samples and refit
        data = mixmg.rvs(int(1e5), evi=None)
        mixmg_re = MixMG().fit(data, n_comps=N_comps)

        # compare the fitted object and original object
        print('Original mean vector and cov matrix of each component:')
        print(mixmg.W)
        for i in range(N_comps):
            print(mixmg.models[i].mu)
            print(mixmg.models[i].S)
            print('-' * 50)
        
        print('Fitted mean vector and cov matrix of each component:')
        print(mixmg_re.W)
        for i in range(N_comps):
            print(mixmg_re.models[i].mu)
            print(mixmg_re.models[i].S)
            print('-' * 50)

        # that result might differs a lot, let's try evaluate the LL difference
        test = np.random.rand(10000, N_vars) * scale
        ori_LL = mixmg.mass(test, logmode = 1)
        re_LL = mixmg_re.mass(test, logmode = 1)
        pct_diff = abs(ori_LL - re_LL)/abs(ori_LL)
        print('Max LL relative difference is: {:.4f}%'.format( pct_diff.max() * 100 ))
        clock.ring()

        mixmg_auto = MixMG().fit(data, verbose = 1)
        re_LL = mixmg_auto.mass(test, logmode = 1)
        pct_diff = abs(ori_LL - re_LL)/abs(ori_LL)
        print('Max LL relative difference for auto model is: {:.4f}%'.format( pct_diff.max() * 100 ))
        clock.ring()
        return mixmg



if __name__ == '__main__':
    # MultivariateGaussain.self_test()
    MixMG.self_test()
