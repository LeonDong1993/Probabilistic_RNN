import numpy as np
import warnings
from copy import deepcopy
from pdb import set_trace
# coding: utf-8

# TODO: unify all potentials that take x as a tuple

class PotentialWrapper:
    def __init__(self,cont_ids,discrete_ids):
        self.cid = cont_ids
        self.did = discrete_ids
        self.onept = len(self.did) == 0
        self.pt = {}

    def add_potential(self,c,p):
        if self.onept:
            self.pt = p
        else:
            self.pt[c] = p

    def prob(self,x):
        valx = tuple(map(lambda v:x[v], self.cid))
        if self.onept:
            p = self.pt
        else:
            conf = tuple(map(lambda v:x[v], self.did))
            if len(conf) == 1: conf = conf[0]
            p = self.pt[conf]
        if len(valx) == 1:
            valx = float(valx[0])
        return p.prob(valx)

class SoftMax:
    def __init__(self,domain):
        self.domain = domain

    def fit(self,traindata):
        # assume the first column is Y while the rest is X
        from sklearn.linear_model import LogisticRegression
        _,F = traindata.shape
        if F == 1 :
            self.univar = True
            pt = CPT([self])
            pt.fit(traindata)
        else:
            self.univar = False
            pt = LogisticRegression(multi_class= 'auto', solver = 'lbfgs')
            pt.fit(traindata[:,1:],traindata[:,0])
            T = list(pt.classes_)
            self.idx = dict(zip(T,range(len(T))))

        self.P = pt

    def prob(self,x):
        if self.univar:
            p = self.P.prob(x)
        else:
            data = x[1:]
            data = np.array(data).reshape(1,-1)
            probs = self.P.predict_proba(data)
            y = x[0]
            p = probs[0,self.idx[y]]
        return p

class CPT:
    def __init__(self,rvlist):
        """ Learn a conditional discrete probability distribution,
            we assume the first random variable condition on the others
        Args:
            rvlist(list): list of discrete R.V. object
        """
        self.I = []
        self.N = []
        for rv in rvlist:
            n = len(rv.domain)
            indexdict = dict(zip(rv.domain,range(n)))
            self.I.append(indexdict)
            self.N.append(n)

        self.univar = len(self.N) == 1
        self.P = np.random.random(tuple(self.N))
        self.normalize(smooth = False)

    def normalize(self,smooth = False):
        if smooth:
            addv = 0.1
            self.P += addv

        if self.univar:
            self.P /= np.sum(self.P)
        else:
            summations = np.sum(self.P, axis = 0)
            for npi,npv in np.ndenumerate(summations):
                idx = tuple([slice(None)]) + npi
                self.P[idx] /= npv

    def fit(self,train_data):
        """ train_data(numpy array): array of train data,
            each column is treated as samples for r.v.
        """
        self.P[:] = 0
        N,D = train_data.shape
        assert(D == len(self.N))
        for n in range(N):
            idx = self._index(train_data[n,:])
            self.P[idx] += 1
        self.normalize(smooth=True)

    def prob(self,x):
        if self.univar:
            idx = self._index([x])
        else:
            idx = self._index(x)
        return self.P[idx]

    def random(self,N=1,cond_val=[]):
        candidates = list(self.I[0].keys())
        dist = [self.prob([v]+cond_val) for v in candidates]
        samples = np.random.choice(candidates,p=dist,size=N)
        if N==1:
            samples = float(samples)
        else:
            samples = list(samples)
        return samples

    def _index(self,x):
        return tuple(map(lambda i,xi: self.I[i][xi],range(len(x)), x))

class CLG:
    def __init__(self, A = None, b = 0, s = 1):
        self.var_thresh = 0.1
        self.mass_correction = 1e-100
        self.biased = False
        if A is not None:
            if hasattr(A, '__len__'):
                A = np.array(A)
            else:
                A = np.array([A])
        self.para = (A,b,s)

    @staticmethod
    def corr_coef(data, magnitude = True, weight = None):
        assert(data.shape[1] == 2)
        var = np.cov(data, rowvar=False, aweights = weight)

        if any(np.isclose(np.diag(var), 0)) :
            coef = 0.0
        else:
            coef = var[0,1] / np.sqrt(var[0,0]*var[1,1])

        if magnitude:
            coef = np.abs(coef)

        if np.abs(coef) > 1.0001:
            warnings.warn("Correlation coef > 1.0 !")
        return coef

    @staticmethod
    def product_gaussian(G):
        """ product N 1-D gaussian distributions
        Args:
            G (list): list of (mu,variance) tuples
        Returns:
            tuple: the mean and variance of result gaussian (mu,var)
        """
        X,P = 0.0,0.0
        for m,var in G:
            prec = 1.0/var
            P += prec
            X += m*prec
        V = 1.0/P
        mu = X*V
        return mu,V

    @staticmethod
    def get_conditional_gaussian(mu,sigma):
        if mu.size == 1:
            A = None
            b = float(mu)
            s = float(sigma)
        else:
            if np.linalg.det(sigma) <= 0:
                warnings.warn("Covariance matrix is singular" , RuntimeWarning)
                N = sigma.shape[0]
                m = np.diag([1e-5]*N)
                sigma += m

            mu = mu.reshape((mu.size,1))
            mu1 = mu[0,:]
            mu2 = mu[1:,:]
            s11 = sigma[0,0]
            s12 = sigma[0:1,1:]
            s21 = s12.T
            s22 = sigma[1:,1:]
            try:
                A = s12.dot(np.linalg.inv(s22))
            except:
                N,_ = s22.shape
                A = s12.dot(np.identity(N))
            b = float(mu1 - A.dot(mu2))
            s = float(s11 - A.dot(s21))
            A = A.flatten()
        return (A,b,s)

    def fit(self,train_data,weight = None):
        N,D = train_data.shape
        if weight is None:
            mu = np.mean(train_data,axis = 0)
            sigma = np.cov(train_data,rowvar = False, bias = self.biased)
            '''
            mu = mu.reshape((1, mu.size))
            data = train_data - mu
            mat = data.T.dot(data)
            mat /= (data.shape[0] - 1)
            '''
        else:
            W = deepcopy(weight)
            W = W.reshape( (weight.size,1) )
            W /= np.sum(weight)

            mu = np.sum(train_data * W, axis = 0)
            mu = mu.reshape((1, mu.size))
            data = train_data - mu
            data *= np.sqrt(W)
            sigma = data.T.dot(data)
            if not self.biased:
                sigma *= N/(N-1) # make it unbiased

        A,b,s = CLG.get_conditional_gaussian(mu,sigma)
        if s < self.var_thresh:
            s = self.var_thresh
        self.para = (A,b,s)
        if A is not None:
            assert( np.sum( np.abs(A) ) > 0), "Data is not valid"
        return self

    def rvs(self, N = None, cond_val = []):
        """ Generate N samples given specific conditioned values
        Args:
            N (int): the number of samples needed
            cond_val (list/1-D numpy array): the value of conditioned R.V.
        Returns:
            float/list: the samples generated
        """
        mu = self._calculate_mean(cond_val)
        std = float(np.sqrt(self.para[2]))
        samples = np.random.normal(mu,std,size = N)
        return samples

    def mass(self,x, logmode = False):
        """ evaluate the probability density give the value of all R.V.
        Args:
            x(float/list/tuple): the value of all R.V.
        Returns:
            float: the density
        """
        A,b,s = self.para
        if A is None:
            v = x ; cond_val = []
        else:
            v = x[0] ; cond_val = x[1:]

        mu = self._calculate_mean(cond_val)

        density = -0.5*(v-mu)*(v-mu)/s - np.log( np.sqrt(2*np.pi*s) )
        if not logmode:
            density = float(np.exp(density))
            density += self.mass_correction
        return density

    # calculate the gradient respect to the i^th variable
    def gradient(self, x, i = 0, logf = False):
        A,b,s = self.para
        if A is None:
            v = x ; cond_val = []
            assert( i == 0 )
        else:
            v = x[0] ; cond_val = x[1:]
            assert(i >= 0 and i < x.size)

        mu = self._calculate_mean(cond_val)

        if logf:
            gd = 1.0
        else:
            gd = self.mass(x)

        gd = gd * (mu-v)/s
        if i > 0:
            gd *= -A[i-1]
        return gd

    def _calculate_mean(self,cond_val):
        A,b,_ = self.para
        mu = b
        if A is not None:
            mu += np.sum(A * np.array(cond_val))
        return mu

    def __repr__(self):
        return 'CLG:<{}>'.format(self.para)

    @staticmethod
    def selftest():
        clg = CLG(0.9,1.3,2)
        cond_vals = np.random.random(int(10e4))
        reponse_vals = [clg.rvs(cond_val = v) for v in cond_vals]
        samples = np.array([reponse_vals,cond_vals]).T
        clg_fit = CLG().fit(samples)
        print('Truth:{} Fitted:{}'.format( clg.para, clg_fit.para ))

        from scipy.stats import norm
        clg = CLG()
        assert( np.isclose( clg.mass(0.713), norm.pdf(0.713) ))
        print('Mass function test passed')

        g1 = clg.gradient(0.713)
        g2 = norm.pdf(0.713) * -0.713
        assert( np.isclose( g1, g2 ) ), 'Gradient not equal: {} {}'.format(g1, g2)
        print('Gradient function test passed')

if __name__ == "__main__":
    CLG.selftest()

