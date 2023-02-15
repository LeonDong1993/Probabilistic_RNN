# coding: utf-8

# system imports
import pdb
import numpy as np
import torch, random
# import torch.multiprocessing as mp
import matplotlib.pyplot as plt
# np.set_printoptions(suppress=True, linewidth=120, precision=4)
# mp.set_sharing_strategy('file_system')

# system from imports
from copy import deepcopy

# personal imports
import tinylib

# personal from imports
from utmLib.clses import Timer, Logger
from utmLib import utils, shell
# from models.dgbn.base import DynamicGBN
# from models.ours.base import DynamicNeuralMG, DynamicRnnMG
# from models.rnn.base import DynamicRNNIndMG, Bagged_RNNIndMG
# from models.rnn.core import RNN_Model

# set all seed of commonly used libraries 
my_seed = 7
torch.manual_seed(my_seed)
random.seed(my_seed)
np.random.seed(my_seed)
torch.cuda.manual_seed_all(my_seed)



def drop_side(arr, drop = 0):
    # input is np.array with float dtype
    # default scheme, smaller is better
    
    if drop > 0:
        best_cut, worst_cut = np.percentile(arr, [drop/2, 100-drop/2])
        best_cut_idx = arr < best_cut
        worst_cut_idx = arr > worst_cut
        arr[best_cut_idx] = np.nan
        arr[worst_cut_idx] = np.nan
        arr = arr[~np.isnan(arr)]
    return arr
    
def sequence_RMSE(truth, pred, mask, drop = 0):
    """
    Evaluate the average (over both time and variables) RMSE of time series 
    Drop control the percent of best/worse prediction we ignore in total 
    (i.e. each side (drop/2) %) 
    """
    diff = abs(truth[mask] - pred[mask])
    diff = drop_side(diff, drop = drop)
    return np.sqrt( diff.mean() )
    
def sequence_within(truth, pred, mask, thresh, fixed = False):
    """
    Determine how many values (the ratio) in pred lies within 
    thresh% of deviation in terms of truth value.
    """
    if fixed:
        max_diff = thresh
    else:
        max_diff = abs(truth[mask]) * thresh / 100
    diff = abs(truth[mask] - pred[mask])
    num_in = ( diff <= max_diff ).sum()
    total = diff.size
    return (num_in/total)*100

def show_pred_result(name, pred, truth, mask):
    # print all evaluation statistics
    for d in drop_list:
        rmse_d = [sequence_RMSE(g,p,m, drop = d) for g,p,m in zip(truth,pred,mask)]
        log.write(f'{name} RMSE-{100-d}%: ' + LL_fmt_str.format( *tinylib.logmass_statistic(rmse_d) ), echo=1)
    
    for t in thresh_list:
        rate_t = [sequence_within(g,p,m, thresh = t) for g,p,m in zip(truth,pred,mask)]
        log.write(f'{name} ISIN-{t}%: ' + LL_fmt_str.format( *tinylib.logmass_statistic(rate_t) ), echo=1)
    
    # for t in thresh_list:
    #     rate_t = [sequence_within(g,p,m, thresh = t/100, fixed=True) for g,p,m in zip(truth,pred,mask)]
    #     log.write(f'{name} ISIN-{t/100}: ' + LL_fmt_str.format( *tinylib.logmass_statistic(rate_t) ), echo=1)

def mask_and_impute(seq, K, evi_ratio, imputer):
    N,D = seq.shape
    assert(K < N)
    
    # generate masked indexes 
    n_hidden = int( D * (1-evi_ratio) )
    mask = np.array([np.random.choice(D, size = n_hidden, replace=False)  for _ in range(K)])

    truth = seq[(N-K):]
    data = deepcopy(truth)
    assert(data.shape[0] == K)
    for i, _ind in enumerate(mask):
        data[i, _ind] = np.nan
    imputed_data= imputer.transform(data)

    return mask, imputed_data, truth
    
def get_mask(seq, hidden_ratio, steps, N_miss, tail_ratio):
    # seq:ndarray - input sequence 
    # keep_init:float - percentage of seqeucen that has no missing value from begining 
    for x in [hidden_ratio, tail_ratio]:
        assert( 0 <= x and x <= 1)
    for x in [steps, N_miss]:
        assert(isinstance(x, int) and x >= 0 )

    N, D = seq.shape 
    ind_ed = N-steps
    if tail_ratio == 0:
        ind_st = 0
    else:
        ind_st = max(0, ind_ed - int(np.ceil(N_miss/tail_ratio)))

    # T: int - number of total missing time slices
    # K: int - number of missing variables in each time slice 
    T = min(N_miss, ind_ed - ind_st)
    K = int(D * hidden_ratio)
    
    missing_time = np.random.choice( list(range(ind_st, ind_ed)), size = T, replace = False )
    axis_x = []
    axis_y = []
    for x in missing_time:
        miss_var = np.random.choice(D, size = K, replace = False)
        axis_x.append( [x] *  K)
        axis_y.append(miss_var)
    
    if n_step > 0:
        extra_axis_x = [[i]*D for i in range(ind_ed, N)]
        extra_axis_y = [list(range(D)) for _ in range(n_step)]
    else:
        extra_axis_x = []
        extra_axis_y = []

    _x = np.concatenate(axis_x + extra_axis_x)
    _y = np.concatenate(axis_y + extra_axis_y)
    mask_all = (_x,_y)
    
    if steps == 0:
        mask_query = mask_all
    else:
        _x = np.concatenate(extra_axis_x)
        _y = np.concatenate(extra_axis_y)
        mask_query = (_x,_y)

    return mask_all, mask_query


# update 09-05-22 22:26
if __name__ == "__main__":
    DEBUG = 1
    ########################################################################################
    drop_list = [0]
    thresh_list = [10,20]

    n_step = 0
    hidden_var_ratio = 0.5
    N_missing_slice = 10
    missing_at_tail = 0.5
    ########################################################################################

    import sys
    if len(sys.argv) <= 1:
        print('Usage: python main.py path_to_data [out_dir_name] [model_path]')
        print('Usage: if model_path is given, out_dir_name must be given as well')
        exit(0)
    
    data_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_dir = sys.argv[2]
    else:
        out_dir = Timer.get_time(fmt = "%m_%d_%H_%M")

    # G_pred = {}
    clock = Timer()
    output_dir = f'./results/raw/{out_dir}'
    shell.makedir(output_dir)
    log = Logger(f'{output_dir}/outputs.txt', with_time = False)
    LL_fmt_str = 'p25 {} p50 {} p75 {} avg {} std {}'

    train, test = utils.pkload(data_path)
    log.write(f'Working on {data_path} ...')
    train, test = tinylib.standardize_TS(train, test, method = 'std')
    # reduce to maximum 300 test points, for fairly reasonable run time
    test = test[:300]

    # load pretrained or train models
    try:
        named_models = utils.pkload(sys.argv[3])
        clock.ring('Load complete')
    except:
        print('Please give path to trained models.')
        exit(0)
    
    try:
        G_pred = utils.pkload(sys.argv[4])
        clock.ring('Load previous computed prediction')
    except:
        G_pred = {}

    # get the masks for each test point
    tmp = [get_mask(item, hidden_var_ratio, n_step, N_missing_slice, missing_at_tail) for item in test]
    mask_all, mask_query = list(zip(*tmp))
    
    # start LL and prediction procedure
    for n,m in named_models:
        # skip this model
        if n in ['DGBN', 'RNN-IndMGx2']:
            continue

        # loglikelihoods
        if n != 'RNN-STD':
            LL = np.array([m.mass(seq, logmode=1) for seq in test])
            if DEBUG:
                TrainLL = np.array([m.mass(seq, logmode=1) for seq in train])

            for d in drop_list:
                fmt_str = f'{n} LL-{100-d}%: ' + LL_fmt_str
                log.write( fmt_str.format( *tinylib.logmass_statistic(-drop_side(-LL, drop=d)) ), echo=1)
                if DEBUG:
                    fmt_str = f'{n} TrainLL-{100-d}%: ' + LL_fmt_str
                    log.write( fmt_str.format( *tinylib.logmass_statistic(-drop_side(-TrainLL, drop=d)) ), echo=1)
        
        if n in G_pred:
            pred = G_pred[n]
        else:
            # regression
            pred = []
            for seq, miss in zip(test, mask_all):
                X = seq.copy()
                X[miss] = np.nan
                pred.append( m.fill_sequence(X, miss) )
        
        show_pred_result(n, pred, test, mask_query)
        if n_step > 0:
            show_pred_result('ALL-' + n, pred, test, mask_all)

        clock.ring(f'{n} LL calculation and prediction')
        G_pred[n] = pred

    utils.pkdump(G_pred,f'{output_dir}/preds.pkl')
    print('Results saved.')

    #  plot model training status 
    if 0:
        for n,m in named_models:
            fig = plt.figure()
            if n == 'DGBN' or 'Bag' in n:
                continue
            elif n == 'RNN-STD':
                nn = m.nn
            else:
                nn = m.dist[0].nn
            
            fig = plt.figure()
            plt.plot(nn.train_loss[3:], label = 'Train')
            plt.plot(nn.valid_loss[3:], label = 'Valid')

            fig_title = n
            plt.title(fig_title)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.savefig(f'{output_dir}/{fig_title}.jpg')
    

