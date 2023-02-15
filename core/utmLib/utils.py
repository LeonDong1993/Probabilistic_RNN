# coding:utf-8
import sys, pickle, time
from utmLib.clses import MyObject

def str_clean(s, to, chars = "!#$%^&*()"):
    for c in chars:
        s = s.replace(c, to)
    return s

def read_text(fpath, splitter = ',', header = False, encoding = 'utf-8', comment = '#'):
    try:
        fh = open(fpath, 'r', encoding = encoding)
    except TypeError:
        # support python2
        import io
        fh = io.open(fpath, 'r', encoding = encoding)

    content = fh.readlines()
    fh.close()

    ret = []
    for line in content:
        if not line.startswith(comment):
            ret.append(line.strip().split(splitter))

    if header:
        ret = ret[1:]

    return ret

def require_verison(major, minor, flag):
    assert(flag in ['eq','le','ge']), "Not supported"
    a = major
    b = minor
    pyver = sys.version_info
    if flag == 'eq':
        assert(pyver >= (a,b-1))
        assert(pyver <= (a,b+1))
    if flag == 'ge':
        assert(pyver >= (a,b-1))
    if flag == 'le':
        assert(pyver <= (a,b+1))
    return

def allin(x,y):
    # return True if all v in x also in y
    ret = all(v in y for v in x)
    return ret

def notin(x,y):
    # return elements of x that not in y
    ret = [v for v in x if v not in y]
    return ret

def pkdump(obj, fpath):
    pickle.dump( obj, open( fpath, "wb" ))

def pkload(fpath):
    return pickle.load( open( fpath, "rb" ) )

def product(*args):
    assert( len(args) > 1), "Invalid input"
    D = [(item, ) for item in args[0]]
    for item in args[1:]:
        D = crossprod(D,item)
    return D

def crossprod(X, Y):
    for x in X:
        for y in Y:
            yield x + (y,)

def halfprod(X):
    ret=[] ; N = len(X)
    for i in range(N):
        for j in range(i+1,N):
            ret.append( (X[i],X[j]) )
    return ret

def diffprod(X):
    ret=[] ; N = len(X)
    for i in range(N):
        for j in range(N):
            if i!=j:
                ret.append( (X[i],X[j]) )
    return ret

def dict2obj(d):
    obj = MyObject()
    obj.__dict__.update(**d)
    return obj

def keep_alive(drives):
    for d in drives:
        fpath = '{}:/.keep_alive'.format(d)
        print('Write keep alive msg for drive {} ...'.format(d))
        with open(fpath, 'a+') as f:
            f.write('..... heartbeat .....\n')
    return

if __name__ == '__main__':
    import sys
    D = sys.argv[1].split(',')
    while 1:
        keep_alive(D)
        time.sleep(30)