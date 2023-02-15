import numpy as np
from copy import deepcopy
from pdb import set_trace
from sklearn.cluster import KMeans

from utmLib import utils
from utmLib.clses import Timer
from utmLib.ml.graph import Graph
# from utmLib.parmapper import Xmap
from utmLib.ml.potential import CLG
from utmLib.ml.solver import GaBP

class GBN:
    def __init__(self, g):
        assert(g.digraph), "Only directed graph allowed"
        g.enhance()
        self.g = g
        self.potential = {}
        self.mass_correction = 1e-100

    def fit(self, train_data, weight = None, var_thresh = 0.1):
        _,D = train_data.shape
        assert(self.g.N == D), "Input data not valid"

        for i in range(self.g.N):
            parents = self.g.V[i].parents
            domain = [i] + parents
            clg = CLG()
            clg.var_thresh = var_thresh
            clg.fit(train_data[:,domain], weight = weight)
            clg.domain = domain
            self.potential[i] = clg

        fg = self.g.factorize(self.potential)
        solver = GaBP(fg)
        if fg.istree():
            solver.infer = solver.tree_infer
        else:
            solver.infer = solver.general_infer
        self._solver_ = solver

        return self

    def predict(self, test, unknown):
        return np.array( [self._predict_(x,unknown) for x in test] )

    def _predict_(self, x, unknown):
        solver = self._solver_
        solver.set_value(x, unknown)
        solver.infer()

        ret = deepcopy(x)
        for i in unknown:
            ret[i] = solver.MPE(i)
        return ret

    def gradient(self,x,i):
        gd = self.mass(x)
        summation = 0
        pt = self.potential[i]
        summation += pt.gradient(x[pt.domain], 0, logf = True)
        children = self.g.V[i].children
        for c in children:
            pt = self.potential[c]
            summation += pt.gradient(x[pt.domain], 1, logf = True)
        gd *= summation
        return gd

    def mass(self,x):
        density = 1.0
        for i in range(self.g.N):
            p = self.potential[i]
            dom = p.domain
            density *= p.mass(x[dom])

        density += self.mass_correction
        return density

    def rvs(self):
        topo_order = self.g._topo_order_
        ret = np.zeros(self.g.N)
        for n in topo_order:
            pt = self.potential[n]
            cond_val = ret[pt.domain[1:]]
            ret[n] = pt.rvs(cond_val = cond_val)
        return ret

    @staticmethod
    def chowliu_tree(data, root=0, weight=None):
        _,D = data.shape
        g = Graph(digraph = False)
        for i in range(D):
            g.add_vertice(i)

        allpair = utils.halfprod(range(D))
        for i,j in allpair:
            coef = CLG.corr_coef(data[:,(i,j)], weight = weight)
            g.add_edge(i,j, weight = coef)

        g = g.max_spanning_tree()
        g = g.todirect(root)
        return g

    @staticmethod
    def selftest():
        M = 5
        given = [1,2]
        unknown = utils.notin(range(M),given)
        # generate fake dataset
        N = 100000
        dataset =[]
        for i in range(N):
            X0 = np.random.normal(100,5)
            X1 = np.random.normal(3*X0-20,10)
            X2 = np.random.normal(0.2*X0+30,5)
            X3 = np.random.normal(2*X1+10,5)
            X4 = np.random.normal(X1+7,5)
            dataset.append([X0,X1,X2,X3,X4])

        # automatically learn the tree structure
        data = np.array(dataset,dtype='float')
        cut = int(0.8*N)
        trainset = data[0:cut,:]
        testset =  data[cut:,:]

        g = GBN.chowliu_tree(data)
        gbn = GBN(g).fit(trainset)
        pred = gbn.predict(testset, unknown)

        # calculate deviation percentage
        ideal = testset[:,unknown]
        result = pred[:,unknown]
        diff = abs(result - ideal)
        pct_diff = np.sum(diff)/np.sum(abs(ideal))
        print('Average diff is {:.2f}%'.format(100*np.mean(pct_diff)))
        exit(0)


class MixGBN:
    def __init__(self, converge_thresh = 1e-3, structure_update_thresh = 1e-2, structure_update_interval = 0):
        self.converge_thresh = converge_thresh
        self.structure_update_thresh = structure_update_thresh
        self.structure_update_interval = structure_update_interval

    def fit(self, X, n_component = 5, base_maxiter = 50, verbose = False, parallel = 1):
        if verbose: clock = Timer()

        n_samples = X.shape[0]
        # max_iter = int(max_iter_rate * n_component)
        log_base = 10
        iter_rate = max( np.log(n_component) / np.log(log_base) , 1.0)
        max_iter = int( base_maxiter * iter_rate )

        if self.structure_update_interval <= 0:
            self.structure_update_interval = min(n_component, 5)

        # Use Kmeans to do initialization
        clf = KMeans(n_clusters = n_component).fit(X)
        matW = np.zeros( (n_samples, n_component) )
        for i in range(n_samples):
            for j in range(n_component):
                matW[i,j] = np.linalg.norm( X[i,:] - clf.cluster_centers_[j,:] )
        Z = np.sum(matW, axis=1).reshape(-1,1)
        matW = matW / Z
        if verbose: clock.ring('Kmeans init')

        # initialize components using above results
        comps = list()
        for i in range(n_component):
            g = GBN.chowliu_tree(X, weight = matW[:,i])
            comps.append(GBN(g).fit(X, weight = matW[:,i]))
        if verbose: clock.ring('Init component')

        # varibles needed for em
        L = np.ones((n_component,)) / n_component
        Q = np.ones((n_samples,n_component)) / n_component
        V = np.zeros((n_samples,n_component))

        converged = False
        prev_val = 0.0
        change_rate = 1.0
        converge_count = 0
        prev_update_iter = 0
        n_iter = 0

        def em_obj_val():
            obj_val = 0.0
            for j in range(n_component):
                obj_val += np.sum( Q[:,j] * np.log( L[j] * V[:,j] / Q[:,j] ))
            return obj_val

        while not converged:
            n_iter += 1

            if parallel == 1:
                # E-Step, update Q
                for i in range(n_samples):
                    for j in range(n_component):
                        V[i,j] = comps[j].mass(X[i,:])
            else:
                assert(0)
                # items = list(utils.product([comps,X]))
                # cksize = int(len(items) / parallel) + 1
                # ret = list(Xmap(GBN.mass, items, N = parallel, star=True, chunksize = cksize))
                # V = np.array(ret).reshape(n_component,-1).T

            Z = V.dot(L)
            for j in range(n_component):
                Q[:,j] = L[j] * V[:,j] / Z

            # smooth Q a little bit
            smooth_amount = 0.01 * (1.0/n_component)
            Q += smooth_amount
            amp_rate = (n_samples + smooth_amount * Q.size)/ n_samples
            Q /= amp_rate
            if verbose: clock.ring('E-Step')

            # M-Step, Update L and gbn
            L = np.sum(Q, axis=0) / n_samples

            structure_update = change_rate < self.structure_update_thresh and (n_iter - prev_update_iter) >= self.structure_update_interval
            if structure_update:
                prev_update_iter = n_iter

            for j,gbn in enumerate(comps):
                if structure_update:
                    g = GBN.chowliu_tree(X, weight = Q[:,j])
                    comps[j] = GBN(g).fit(X, weight = Q[:,j])
                else:
                    gbn.fit(X, weight = Q[:,j])
            if verbose: clock.ring('M-Step')

            # stop criteria
            now_val = em_obj_val()
            change_rate = np.abs( ( now_val - prev_val) / now_val)
            prev_val = now_val

            if verbose:
                print('EM object value: {:.5f}, Change Rate:{}'.format(now_val, change_rate))

            if change_rate < self.converge_thresh:
                converge_count += 1
            else:
                converge_count = 0

            if converge_count >= 2 and prev_update_iter > 0:
                converged = True

            if n_iter >= max_iter:
                converged = True

        self.weight = L
        self.component = comps
        return self

    def predict(self, X, unknown, verbose = False, GA = False , max_iter = 20, parallel = 1):
        # a wrapper function calls self.mpe to do prediction
        if parallel == 1:
            ret = [self.mpe(x, unknown, verbose, GA, max_iter) for x in X]
        else:
            assert(0)
            cksize = int(X.shape[0]/parallel) +1
            # ret = list(Xmap(self.mpe, X , args=(unknown, False, GA, max_iter), chunksize=cksize, N=parallel  ))
        return np.array(ret)

    def mpe(self, x, unknown, verbose = False, GA = False, max_iter = 20):
        if verbose: clock = Timer()

        candidates = []
        for item in self.component:
            px = item._predict_(x, unknown)
            m = self.mass(px)
            candidates.append( (px,m) )
        if verbose: clock.ring('Find component MAP')

        # compute mass center
        X = np.array([x for x,m in candidates])
        W = np.array([m for x,m in candidates])
        W = W.reshape(-1,1)
        W /= np.sum(W)
        mass_center = np.sum(W * X, axis = 0)
        candidates.append( (mass_center, self.mass(mass_center)) )

        # computer weight center
        W = self.weight.reshape(-1,1)
        weight_center = np.sum(W * X, axis = 0)
        candidates.append( (weight_center, self.mass(weight_center)) )
        candidates.sort(key = lambda x:x[1], reverse = True)
        if verbose: clock.ring('Computer mass and weight center')

        if not GA:
            ret = candidates[0][0]
        else:
            # start GA from the best candidate
            n_iter = 0
            dec_rate = 0.3
            converged = False
            y = deepcopy(candidates[0][0])
            G = np.zeros( (y.size,) )

            while not converged:
                n_iter += 1
                for i in unknown:
                    G[i] = self.gradient(y,i)

                # determine actual step size using line backtracking
                init_mass = self.mass(y)
                grad_norm_sqaure = np.linalg.norm(G) ** 2
                y_norm_square = np.linalg.norm(y) ** 2
                norm_ratio = np.sqrt(grad_norm_sqaure/y_norm_square)
                lrate = self.converge_thresh/norm_ratio * 36

                while (self.mass( y+G*lrate ) - init_mass) < (0.5 * lrate * grad_norm_sqaure) \
                and (lrate * norm_ratio) >= self.converge_thresh:
                    lrate *= dec_rate

                if (lrate * norm_ratio) < self.converge_thresh:
                    converged = True
                else:
                    y += lrate * G

                if n_iter >= max_iter:
                    converged = True

                if verbose:
                    current_mass = self.mass(y)
                    print('Mass:{} Step:{} X:{} G:{}'.format(current_mass, lrate, y[unknown], G[unknown]) )

            ret = y
        if verbose: clock.ring('GA time (if turn on)')
        return ret

    def gradient(self,x,i):
        G = np.array([item.gradient(x,i) for item in self.component])
        grad = np.sum(G * self.weight)
        return grad

    def mass(self, x):
        return np.sum(self.weight * np.array([item.mass(x) for item in self.component]))

if __name__ == "__main__":
    GBN.selftest()