# coding: utf-8
import os,platform
import numpy as np
from copy import deepcopy
from utmLib.clses import MyObject
from utmLib.utils import notin, halfprod
import networkx as nx
import networkx.algorithms as nxalg

class Node(MyObject):
    def __init__(self,desc = '?'):
        self.desc = desc

    def __repr__(self):
        return self.desc

class Graph:
    def __init__(self,digraph):
        self.digraph = digraph
        self.N = 0
        self.V = {}
        self.E = {}

    def make_complete_graph(self, n):
        assert(self.N == 0)
        for i in range(n):
            self.add_vertice(i)
        for i,j in halfprod( range(n) ):
            self.add_edge(i,j)
        return self

    def add_vertice(self,node = None):
        if node is None:
            node = Node( 'x{}'.format(self.N) )
        elif isinstance(node,int):
            node = Node( 'x{}'.format(node) )
        elif isinstance(node, str):
            node = Node(node)
        else:
            assert( isinstance(node, Node) )

        self.V[self.N] = node
        self.E[self.N] = {}
        self.N += 1
        return self.N - 1

    def add_edge(self,s,t,weight=1.0):
        if t not in self.E[s]:
            self.E[s][t]=weight

            if not self.digraph:
                self.E[t][s]=weight
    
    def remove_edge(self, s, t):
        if s not in self.E:
            return

        if self.digraph:
            self.E[s].pop(t, None)
        else:
            if t not in self.E:
                return 
            self.E[s].pop(t, None)
            self.E[t].pop(s, None)
        return

    def remove_all_edges(self):
        self.E = {}
        for i in range(self.N):
            self.E[i] = {}
        return

    def get_edges(self,weight=False):
        edges = []
        for s,v in self.E.items():
            for t in v:
                info = [s,t]
                if weight: info.append(v[t])
                edges.append( tuple(info) )
        return edges

    def roots(self):
        # return a list: nodes that has no parents
        assert(self.digraph)
        not_root = set()
        for vid in range(self.N):
            for x in self.E[vid].keys():
                not_root.add(x)
        roots = notin(range(self.N), not_root)
        return roots

    def find_leaves(self):
        assert(self.digraph)
        ret = []
        for i in range(self.N):
            if len(self.E[i]) == 0:
                ret.append(i)
        return ret

    def merge(self,g2):
        g = deepcopy(self)
        mapping = {}
        for nid,node in g2.V.items():
            vid = g.add_vertice(node)
            mapping[nid] = vid

        for sid,v in g2.E.items():
            for tid,weight in v.items():
                g.add_edge( mapping[sid],mapping[tid], weight= weight )
        return g

    def find_parents(self,x):
        assert(self.digraph)
        parents = []
        for k,v in self.E.items():
            if x in v:
                parents.append(k)
        return parents

    def find_children(self,x):
        assert(self.digraph)
        children = list(self.E[x].keys())
        return children

    def find_neighbor(self,v):
        if self.digraph:
            neighbors = self.find_parents(v) + self.find_children(v)
        else:
            neighbors = list(self.E[v].keys())
        return neighbors

    def to_nxGraph(self):
        if self.digraph:
            G = nx.DiGraph()
        else:
            G = nx.Graph()

        for i in range(self.N):
            G.add_node(i)

        for i, j, w in self.get_edges(weight=True):
            G.add_edge(i,j, weight = w)

        return G

    def enhance(self):
        '''
        call this function ONLY when graph structure is not gonna change
        pre-compute all graph structure info to speed up furture operation
        '''
        # computer node neighbors, parents and childs if digraph
        for i in range(self.N):
            node = self.V[i]
            node.nb = self.find_neighbor(i)
            if self.digraph:
                node.children = self.find_children(i)
                node.parents = self.find_parents(i)
        # add topo sort information as well
        self._topo_order_ = self.toposort()

    def istree(self):
        G = self.to_nxGraph()
        flag = nxalg.tree.recognition.is_tree(G)
        return flag

    def todirect(self,root):
        assert(not self.digraph)
        newg = deepcopy(self)
        newg.remove_all_edges()
        newg.digraph=True

        unmarked = [root] + list(range(newg.N))
        while len(unmarked) > 0:
            root = unmarked[0]
            info = self.BFS(root,walkinfo=True)
            visited = list(info['dist'].keys())
            unmarked = notin(unmarked,visited)
            for i,j in info['walk']:
                newg.add_edge(i,j)
        return newg

    def BFS(self,v,walkinfo = False):
        assert(not self.digraph)
        info = {'walk':[],'dist':{}}
        Q = [(v,0)]; marked = info['dist']

        while len(Q) > 0:
            node,distance = Q.pop(0)
            if node not in marked:
                marked[node] = distance
            for nb in self.find_neighbor(node):
                if nb not in marked:
                    Q.append((nb,distance+1))
                    if walkinfo:
                        info['walk'].append((node,nb))
        return info

    def toposort(self,reverse = False):
        assert(self.digraph)
        G = self.to_nxGraph()
        topo_order = list(nxalg.dag.topological_sort(G))
        if reverse:
            topo_order.reverse()
        return topo_order

    def max_spanning_tree(self):
        assert(not self.digraph)
        G = self.to_nxGraph()
        mst = nxalg.tree.maximum_spanning_edges(G)

        newg = deepcopy(self)
        newg.remove_all_edges()
        for i,j,_ in mst:
            newg.add_edge(i,j)
        return newg

    def factorize(self,potential):
        """ Create a factor graph b
        Args:
            potential(dict): potential[i] is the potential object associated with node i and its parent
        Returns:
            a graph object: the factor graph created
        """
        assert(self.digraph)
        fg = Graph(digraph=False)
        for i in range(self.N):
            node = deepcopy(self.V[i])
            if not hasattr(node,'attr'):
                node.attr = 'RV'
            else:
                assert(0), "Vertice object in graph already has attr attribute!"
            fg.add_vertice(node)

        for i in range(self.N):
            node = Node()
            node.attr = 'FACTOR'
            node.potential = potential[i]
            fid = fg.add_vertice(node)
            node.desc = f"F{fid}"
            # connect factor with related R.V.
            domain = self.find_parents(i) + [i]
            for nid in domain:
                fg.add_edge(fid,nid)

        fg.rvs = []; fg.factors = []
        for i in range(fg.N):
            node = fg.V[i]
            node.nb = fg.find_neighbor(i)
            if node.attr != 'FACTOR':
                fg.rvs.append(i)
            else:
                fg.factors.append(i)
        return fg

    def expand(self,intercnt,T):
        assert(self.digraph),"Directed graph needed"
        g = deepcopy(self)
        for t in range(T):
            offset = (t+1)*self.N
            for i in range(self.N):
                # copy node
                node = deepcopy(self.V[i])
                nid = g.add_vertice(node)
                assert(nid == offset + i)
                # copy edge
                for n,w in self.E[i].items():
                    g.E[nid][n+offset] = w
                # add inter connection
                for i,j in intercnt:
                    i = i+offset - self.N
                    j = j+offset
                    g.add_edge(i,j)
        return g

    def dump(self, fpath, weight=False):
        name = 'tmp'
        fh = open(name+'.dot','w')
        if self.digraph:
            fh.write("digraph %s { \n" % name)
        else:
            fh.write('graph %s { \n' % name)

        for v in range(self.N):
            fh.write('{} [label="{}#{}"] \n'.format(v,v,self.V[v]))

        if self.digraph:
            for s in self.E:
                edges = self.E[s]
                for t in edges:
                    if weight:
                        fh.write('{} -> {} [label = {}]\n'.format(s,t,edges[t]))
                    else:
                        fh.write('{} -> {}\n'.format(s,t))
        else:
            Edges = []
            for s in self.E:
                edges = self.E[s]
                for t in edges:
                    if (s,t) not in Edges and (t,s) not in Edges:
                        if weight:
                            fh.write('{} -- {} [label = {}]\n'.format(s,t,edges[t]))
                        else:
                            fh.write('{} -- {}\n'.format(s,t))
                        Edges.append( (s,t) )

        fh.write('}\n')
        fh.close()
        # require to install graphviz
        os.system('dot -Tpng -o {} {}.dot'.format(fpath, name))

        if 'Win' in platform.platform():
            os.system('del {}.dot'.format(name))
        else:
            os.system('rm {}.dot'.format(name))
