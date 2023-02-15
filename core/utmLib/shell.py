# shell command wrappers

import os

def cd(directoty):
    os.chdir(directoty)
    return

def bash(command, verbose = True):
    if verbose:
        print(command)
    ret = os.system(command)
    if (ret != 0):
        print('Command [%s] execute failed' % command)
        exit(ret)

def makedir(path):
    if (not os.path.exists(path)):
        print('Making directory %s' % path)
        os.makedirs(path)
    else:
        print('Directory already exist!')


