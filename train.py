import optuna, tinylib, sys
import torch.multiprocessing as mp
import numpy as np
# mp.set_sharing_strategy('file_system')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from utmLib import utils, shell, clses
from models.dgbn.base import DynamicGBN
from models.ours.base import DynamicNeuralMG, DynamicRnnMG
from models.rnn.base import DynamicRNNIndMG
from models.rnn.core import RNN_Model


def tune_model(name, model_class, train, kwargs, out_dir):
    D = train[0].shape[1]
    best_score = float('inf')
    best_model = None

    def obj_func(trial):
        nonlocal best_model, best_score

        dr = trial.suggest_float("drop_rate", 0.0, 0.4)
        lr = trial.suggest_float("learn_rate", 1e-4, 5e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1, log=True)
        hs = trial.suggest_int("hidden_size", 2*D, min(8*D, 1000), step = 2*D)
        
        if name != 'NN-MG':
            kwargs['hidden_size'] = hs
        else:
            kwargs['feature_size'] = hs
        
        m = model_class().fit(train, [], **kwargs, maxlr=lr, drop_out=dr, weight_decay=wd)
        
        if name == 'RNN-STD':
            nn = m.nn
        else:
            nn = m.dist[0].nn
        
        # we want to maximize this score
        ret = -nn.score
        if ret < best_score:
            best_score = ret
            best_model = m
        return ret

    # can resume from previous optimization
    study_path = f'{out_dir}/{name}_study.pkl'
    model_path = f'{out_dir}/{name}.pkl'
    try:
        study = utils.pkload(study_path)
        best_score = study.best_value
        best_model = utils.pkload(model_path)
        print('Load from existing study.')
    except:
        study = optuna.create_study()
    
    
    clock = clses.Timer()
    study.optimize(obj_func, n_trials = 50, gc_after_trial = True)
    clock.ring(msg = f"Model:{name} train time for 50 trial")
    best_param = sorted(study.best_params.items())
    best_param = [v for k,v in best_param]
    print(f"{name} Best hyper-param found: {best_param} ")
    
    utils.pkdump(best_model, model_path)
    utils.pkdump(study, study_path)
    return 



# main logic starts here
def _get(**kwargs):
    return kwargs

if __name__ == '__main__':
    data_path = sys.argv[1]
    print(f"Input data is {data_path}")
    out_dir = 'results/train/{}'.format(sys.argv[2])
    shell.makedir(out_dir)

    train, test = utils.pkload(data_path)
    train, test = tinylib.standardize_TS(train, test, method = 'std')
    nn_kwargs = {'max_epoch':120, 'pre_train':5, 'pre_train_epoch':12}
    nn_kwargs['max_parents'] = min( int(np.ceil(np.log2(train[0].shape[1]))), 5)
    
    # more parameters can be add here
    nn_mg = ['NN-MG', DynamicNeuralMG, _get(**nn_kwargs)]
    rnn_mg = ['RNN-MG', DynamicRnnMG, _get(**nn_kwargs)]
    rnn_std = ['RNN-STD', RNN_Model,  _get(**nn_kwargs)]
    rnn_ind_mg3 = ['RNN-IndMGx3', DynamicRNNIndMG, _get(**nn_kwargs, n_comps=3)]

    ctx = mp.get_context('spawn')
    all_tasks = [nn_mg, rnn_std, rnn_ind_mg3, rnn_mg]
    # assign gpu device
    total_gpu = 2
    assignment = ['cuda:{}'.format(i % total_gpu) for i in range(len(all_tasks))]
    for i,item in enumerate(all_tasks):
        item[2]['device'] = assignment[i]
    
    # create job processes 
    job_process = [ctx.Process(target = tune_model, args=(n, m, train, t_arg, out_dir)) for n,m,t_arg in all_tasks]

    for p in job_process:
        p.start()
    
    dgbn  = DynamicGBN().fit(train, [], max_parents = nn_kwargs['max_parents'])
    ret = [['DGBN', dgbn]]

    for p in job_process:
        p.join()

    for item in all_tasks:
        name = item[0]
        model = utils.pkload(f'{out_dir}/{name}.pkl')
        ret.append([name, model])
    
    utils.pkdump(ret, f'{out_dir}/models.pkl')
        
