import numpy as np
import configparser, os

from pdb import set_trace
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer

from utmLib import utils
from utmLib.clses import ProgressIndicator, MyObject
from utmLib.ml.potential import CLG

import time
import torch
import torch.optim as optim

def high_corr_detection(data, exclude_vars, thresh = 0.98):
    # high correlated variable detection
    # returns the vars need to be removed such that no two 
    # variables shares a correlation higher than [thresh]
    D = data.shape[1]
    var_corr = analyze_correlation(data)

    hc_vars = []
    for idx in utils.halfprod(range(D)):
        v = var_corr[idx]
        if v > thresh:
            hc_vars.append( (idx,v) )

    elims = set()
    if len(hc_vars) > 0:
        # give elimination suggestion
        for item in hc_vars.copy():
            (a,b), _ = item
            flag1 = a in exclude_vars
            flag2 = b in exclude_vars
            if flag1 + flag2 == 2:
                print('Warning, non-elim variable correlated.')
                hc_vars.remove(item)

            if flag1 + flag2 == 1:
                elims.add( item[0][int(flag1)] )
                hc_vars.remove(item)

        for item in hc_vars.copy():
            (a,b), _ = item
            if a in elims or b in elims:
                hc_vars.remove(item)

        while len(hc_vars) > 0:
            counter = np.zeros(shape = (D,))
            for (a,b), _ in hc_vars:
                counter[a] += 1
                counter[b] += 1
            idx = np.argmax(counter)
            elims.add(idx)
            for item in hc_vars.copy():
                (a,b), _ = item
                if a == idx or b == idx:
                    hc_vars.remove(item)
    return list(elims)

def show_dataset_info(data):
    dim = data[0].shape[1]
    N = len(data)
    seq_len = [item.shape[0] for item in data]
    maxL = np.max(seq_len)
    minL = np.min(seq_len)
    avgL = int(np.round(np.mean(seq_len)))
    print(' Seqs:{} Dim:{} Len. Avg:{} Min:{} Max:{} #:{:.3f}'.format(N,dim,
        avgL,minL,maxL,np.log(N*dim*avgL)))

def pre_processing(train, test, std_rate = 2, iqr_rate = 2, outlier_pct = 10, thresh = 0.01):
    # first, remove outliers
    data = np.vstack(train + test)
    # std outlier detection
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    upper_std = mean + std_rate*std
    lower_std = mean - std_rate*std
    
    # iqr outlier detection
    q_left = outlier_pct / 2
    tmp = np.percentile(data, axis=0, q = [q_left,100-q_left])
    mean = np.mean(data, axis=0)
    diff = tmp[1] - tmp[0]
    upper_iqr = mean + iqr_rate * diff
    lower_iqr = mean - iqr_rate * diff

    # combine both method in an convervative way
    upper = np.maximum(upper_iqr, upper_std)
    lower = np.minimum(lower_iqr, lower_std)
    upper = upper.reshape(1,-1)
    lower = lower.reshape(1,-1)

    def _do(seq_list):
        remove = set()
        N = len(seq_list)
        for i in range(N):
            seq = seq_list[i]
            # if (seq < lower).any() or (seq > upper).any():
                # remove.add(i)
            n_out = np.sum(seq < lower) + np.sum(seq > upper)
            if (n_out/seq.size) > thresh:
                remove.add(i)
        
        print('{} sequences removed'.format(len(remove)))
        ret = []
        for i in range(N):
            if i not in remove:
                ret.append(seq_list[i])
        return ret

    train = _do(train)
    test = _do(test)

    # second detect higher correlation vars
    data = np.vstack(train + test)
    elims = high_corr_detection(data, exclude_vars=[], thresh = 0.95)
    print('Removed {} variables.'.format(len(elims)))

    D = train[0].shape[1]
    selector = np.setdiff1d(np.arange(D), elims)
    train = [item[:, selector] for item in train]
    test = [item[:, selector] for item in test]

    print("Train Set:")
    show_dataset_info(train)
    print("Test Set:")
    show_dataset_info(test)
    return train, test

def pad_sequence(seq, target_len):
    N, D = seq.shape
    assert(N <= target_len)
    if N == target_len:
        return seq
    ret = np.zeros(shape = (target_len, D))
    ret[:N] = seq
    return ret

def group_batch_index(seq_len, batch_size, with_last = False, shuffle = True):
    groups = {}
    for i,v in enumerate(seq_len):
        if v not in groups:
            groups[v] = []
        groups[v].append(i)

    seqL = list(groups.keys())
    if shuffle:
        np.random.shuffle(seqL)

    ind_list = []
    for kl in seqL:
        seq_ids = groups[kl]
        if shuffle:
            np.random.shuffle(seq_ids)
        total = len(seq_ids)
        st = 0 ; ed = batch_size
        while st < total: # fix @ 07-12-22 13:13
            if with_last or ed <= total or st == 0:
                ind_list.append( (kl, seq_ids[st:ed]))
            st = ed
            ed += batch_size
    return ind_list

def train_model(model, N_data, get_dataloader, conf = 'default', start = 0, stop = 0, verbose = 0, **kwargs):
    # model must have loss function attached, in the form _name_(nn_out, Y)

    # make a default configuration, no default maxlr and max_epoch 
    default_conf = MyObject()
    default_conf.batch_num = 75
    default_conf.batch_size = 10
    default_conf.weight_decay = 1e-4
    default_conf.init_epoch = 5
    default_conf.last_epoch = 5
    default_conf.train_ratio = 0.8
    default_conf.tune_epoch = 5
    default_conf.tol_epoch = 15
    default_conf.device = 'cuda:0'
    
    if conf == 'default':
        conf = default_conf
        for k,v in kwargs.items():
            conf[k] = v

    # pre-train detection
    if stop == 0:
        stop = conf.max_epoch
        warm_up = False
    else:
        warm_up = True

    # attach optimizer and scheduler to the model
    if start == 0:
        final_epoches = conf.max_epoch - (conf.init_epoch + conf.last_epoch)
        scheduler = LrateScheduler(conf.maxlr, conf.init_epoch, final_epoches)
        optimizer = optim.Adam(model.parameters(), lr=conf.maxlr, weight_decay = conf.weight_decay)
        model.device = conf.device
        model.optimizer = optimizer
        model.lrs = scheduler
        model.train_loss = []
        model.valid_loss = []

    # prepare training
    model.moveto(model.device)
    optimizer = model.optimizer
    scheduler = model.lrs
    batch_size = max(conf.batch_size, int(N_data/conf.batch_num)+1) 

    # split data into train and valid
    if conf.tune_epoch > 0:
        train_ind = np.linspace(0, N_data-1, num = int(conf.train_ratio * N_data)).astype(int)
        valid_ind = np.setdiff1d(np.arange(N_data), train_ind)
        train_loader = get_dataloader(train_ind, batch_size)
        valid_loader = get_dataloader(valid_ind, batch_size, with_last = True)
        N_train = train_ind.size
        N_valid = valid_ind.size
    else:
        train_loader = get_dataloader(None, batch_size)
        N_train = N_data

    # nn training
    if verbose and not warm_up:
        print('Train configuration:')
        conf.show()

    if verbose > 1:
        pi = ProgressIndicator(stop - start)

    best_score = -np.inf
    last_update = 0
    for epoch in range(start, stop):
        model.train()
        train_loss = 0
        st_time = time.time()

        lrate = scheduler.get_lrate(epoch)
        for g in optimizer.param_groups:
            g['lr'] = lrate

        for X,Y in train_loader():
            optimizer.zero_grad()
            nn_out = model.forward(X)
            loss = model.loss(nn_out, Y)
            loss.backward()
            optimizer.step()
            train_loss += X.shape[0] * loss.item()

        avg_train_loss = train_loss/N_train
        model.train_loss.append( avg_train_loss )

        if conf.tune_epoch > 0:
            # evaluate on validation 
            model.eval()
            with torch.no_grad():
                valid_loss = 0
                for X,Y in valid_loader():
                    nn_out = model.forward(X)
                    loss = model.loss(nn_out, Y)
                    valid_loss += X.shape[0] * loss.item()
            
            avg_valid_loss = valid_loss / N_valid
            model.valid_loss.append( avg_valid_loss )
        else:
            avg_valid_loss = avg_train_loss
        ed_time = time.time()

        cur_score = -(0.8*avg_valid_loss + 0.2*avg_train_loss)
        if cur_score > best_score:
            best_score = cur_score
            model.save_checkpoint()
            last_update = epoch
        elif epoch - last_update > conf.tol_epoch:
            # early stop if no better results shown 
            break

        if verbose > 1:
            msg = 'E{:2d} time={:.2f}s lrate={:.7f} train={:.2f} valid={:.2f} best={:.2f}'.format(
                epoch+1, ed_time - st_time , lrate, avg_train_loss, avg_valid_loss, -best_score)
            pi.at(epoch-start, info = msg)
    
    # load the best status if possible 
    model.load_checkpoint()
    model.eval()
    model.score = best_score
    return best_score
    if warm_up or conf.tune_epoch <= 0:
        return best_score

    # fine tune the model
    full_loader = get_dataloader(None, batch_size)
    if verbose > 1:
        pi = ProgressIndicator(conf.tune_epoch)

    for epoch in range(stop, stop + conf.tune_epoch):
        model.train()
        train_loss = 0
        st_time = time.time()

        lrate = scheduler.get_lrate(epoch)
        for g in optimizer.param_groups:
            g['lr'] = lrate * 0.2 

        for X,Y in full_loader():
            optimizer.zero_grad()
            nn_out = model.forward(X)
            loss = model.loss(nn_out, Y)
            loss.backward()
            optimizer.step()
            train_loss += X.shape[0] * loss.item()

        avg_train_loss = train_loss/N_data
        model.train_loss.append( avg_train_loss )
        ed_time = time.time()

        if verbose > 1:
            msg = 'FineTune - E{:2d} time={:.2f}s lrate={:.7f} train={:.2f}'.format(
                epoch+1, ed_time - st_time , lrate, avg_train_loss)
            pi.at(epoch-stop, info = msg)

    model.eval()
    return best_score

class LrateScheduler:
    def __init__(self, lr_max, init_epoches=5, final_epoches=90, init_scale=0.1, final_scale=0.1):
        self.lr_max = lr_max
        self.init_scale = init_scale
        self.init_epoches = init_epoches
        self.final_scale = final_scale
        self.final_epoch = final_epoches
        self.init_lr = lr_max * init_scale
        self.final_lr = lr_max * final_scale
        self.total_epoch = final_epoches + init_epoches

    def get_lrate(self,epoch):
        # linear warmup followed by cosine decay
        if epoch < self.init_epoches:
            lr = (self.lr_max - self.init_lr) * float(epoch) / self.init_epoches + self.init_lr
        elif epoch < self.total_epoch:
            lr = (self.lr_max - self.final_lr)*max(0.0, np.cos(((float(epoch) -
                    self.init_epoches)/(self.final_epoch - 1.0))*(np.pi/2.0))) + self.final_lr
        else:
            lr = self.final_lr
        return lr

def read_arff(fpath):
    def clean(s):
        for c in ['"', "'"]:
            s = s.replace(c,'')
        return s
        
    with open(fpath, 'r') as fh:
        content = fh.readlines()
    
    data = []
    flag = 0
    for line in content:
        if line.startswith('@data'):
            flag = 1
            continue
            
        if flag:
            # start parse sequence
            columns = line.strip().split('\\n')
            L = None
            seq = []
            for col in columns:
                col = clean(col)
                
                if '?' in col:
                    k = col.index('?')
                    col = col[0 : (k-1)]

                items = col.split(',')
                if L is None:
                    L = len(items)
                else:
                    items = items[:L]
                seq.append( list(map(float, items)))
            seq = np.array(seq).T
            # check the sequence is not padded or have NaN
            assert( not np.isnan(seq).any() ), "NaN Inside"
            if (np.isclose(seq,0).sum(axis=1) == seq.shape[1]).any():
                Warning.warn("Sequences might be padded.")
            
            data.append(seq)
    return data

def down_sample(data_list, ratio):
    N = len(data_list)
    if ratio >= 1:
        N_sub = ratio
    else:
        N_sub =  int(N*ratio)
    
    chosen = np.linspace(0, N-1, N_sub).astype(int)
    return [data_list[i] for i in chosen]

def standardize_TS(train, test, method):
    """standardize time sequence with zero mean and unit variance

    Args:
        train (list of np.array): list of time sequences used for training
        test (list of np.array): list of time seqeuences used for testing
        method (str) : which method use to standardize data, must in ['std', 'minmax']

    Returns:
        (new_train, new_test): standardize train and test sequence of same data format
    """
    assert( method in ['std', 'minmax'] )
    if method == 'std':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    stacked_train = np.vstack(train)
    scaler.fit(stacked_train)
    
    new_train = [scaler.transform(item) for item in train]
    new_test = [scaler.transform(item) for item in test]
    return new_train, new_test

def find_integer_attr(data):
    _, D = data.shape
    ret = []
    for i in range(D):
        col_int = data[:,i].astype(int)
        if np.all(np.isclose(data[:,i], col_int, rtol=1e-4)):
            ret.append(i)
    return ret

def pca_analysis(data, percentiles):
    D = data.shape[1]
    obj = PCA().fit(data)
    acc_sum = [] ; S = 0
    for v in obj.explained_variance_ratio_:
        S += v
        acc_sum.append(S)
    acc_sum = np.array(acc_sum)

    ret = []
    for p in percentiles:
        left = np.sum(acc_sum > p) - 1
        num_p = D - left
        ret.append(num_p)
    return ret

def analyze_correlation(data):
    # return the pairwise P-correlation of variables
    _,D = data.shape
    corr = np.zeros((D,D), dtype=float)
    for i,j in utils.halfprod(range(D)):
        corr[i,j] = CLG.corr_coef(data[:,(i,j)])
        corr[j,i] = corr[i,j]
    return corr

def compute_rmse(a, b):
    assert(a.shape == b.shape)
    sse = np.sum(np.square(a-b))
    mse = sse / a.size
    rmse = np.sqrt(mse)
    return rmse

def evaluate_result(truth, estimated, query = None, output_var = False, cat_vars = []):
    # evaluate the average rmse of query variables, missing is not considered here
    N, _ = truth.shape
    rmse = []
    right = 0
    total = 0

    if query is None:
        # warnings.warn('Evaluate on all variables since query is None.')
        query = [list(range(truth.shape[1]))] * N

    for i in range(N):
        idx = query[i]
        if len(cat_vars) > 0:
            cont_ind = utils.notin(idx, cat_vars)
            cat_ind = utils.notin(idx, cont_ind)
        else:
            cont_ind = idx

        if len(cont_ind) > 0:
            rmse.append(compute_rmse(truth[i,cont_ind], estimated[i,cont_ind] ))
        else:
            rmse.append(0.0)

        if len(cat_vars) > 0:
            total += len(cat_ind)
            right += np.sum(np.isclose(truth[i,cat_ind], estimated[i,cat_ind]))

    avg_rmse = np.mean(rmse)

    ret = avg_rmse
    if output_var:
        std_rmse = np.std(rmse)
        ret = (ret, std_rmse)
    if len(cat_vars) > 0:
        ret = (ret, 1-right/total)
    return ret

def logmass_statistic(logmass):
    # return the 25%, median, 75% percentile and average of logmass
    p25 = np.percentile(logmass, 25)
    median = np.median(logmass)
    p75 = np.percentile(logmass, 75)
    average = np.mean(logmass)
    std = np.std(logmass)
    ret = (p25, median, p75, average, std)
    return list(np.round(ret, 4))

def print_array(X):
    print('-------------------------------')
    for row in X:
        print(','.join(row.astype(str)))
    return

def read_ini(fpath, section = 'expconf'):
    parser = configparser.ConfigParser()
    parser.read(fpath)
    parsed_conf = dict(parser[section])
    return parsed_conf

def load_exp_conf(fpath):
    parsed_conf = read_ini(fpath)

    if 'include' in parsed_conf:
        extra_ini = parsed_conf['include'].split(',')
        path_frag = fpath.split('/')
        for item in extra_ini:
            path_frag[-1] = item
            extra_fpath = '/'.join(path_frag)
            for k,v in read_ini(extra_fpath).items():
                if k not in parsed_conf:
                    parsed_conf[k] = v
        del parsed_conf['include']

    conf_dict = {}
    for k,v in parsed_conf.items():
        conf_dict[k] = eval(v)
    options = utils.dict2obj(conf_dict)

    # take care of root dir simplification
    if options.root_dir == 'auto':
        file_path = os.path.realpath(__file__)
        up2_dir = file_path.split('/')[0:-3]
        options.root_dir = '/'.join(up2_dir)

    return options

def _do_mask_(data, query, missing):
    masked = data.copy()
    unknown = np.concatenate([query,missing], axis = 1)

    for i, r in enumerate(masked):
        u = unknown[i]
        r[u] = np.nan
    return masked

def mask_dataset(data, Q, M):
    # create masked dataset give query and missing setting
    N, D = data.shape

    if isinstance(Q, list):
        # fix query var
        query = [Q] * N
    else:
        qsize = int(np.round(D*Q))
        query = [np.random.choice(D, size=qsize, replace=False) for i in range(N)]

    msize = int(np.round(D*M))
    missing = []
    for i in range(N):
        pool = utils.notin(range(D), query[i])
        missing.append( np.random.choice(pool, size=msize, replace=False) )

    masked = _do_mask_(data, query, missing)
    return masked, query, missing
