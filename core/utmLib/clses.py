# coding: utf-8
import time, random, sys, os, datetime
from enum import Enum

class CONST(Enum):
    CONTINUOUS = 0
    DISCRETE = 1

# updated 07-12-22 00:47
class ProgressIndicator:
    def __init__(self, total, msg = 'Running', pid = False, step_pct = 1):
        self.t = time.time()
        self.N = total
        self.msg = msg
        self.pid = pid
        # simple solution for ceil
        self.step_size = int(total * step_pct / 100.0 + 1 - 1e-6)
        self.at(-1)

    # def reset_time(self):
        # self.t = time.time()

    def at(self, j, info = None):
        i = j+1
        msg = self.msg if info is None else info
        
        if self.pid:
            myid = os.getpid()
            msg = 'Process {} - {}'.format(myid, msg)
        
        out = sys.stdout
        if i < self.N:
            end_char = '\r'
        else:
            end_char = '\n'

        # estimate the time left
        cur_time = time.time()
        time_used = cur_time-self.t
        time_left =  time_used / (i + 0.01) * max(self.N - i, 0)

        if i % self.step_size == 0 or i <= self.N:
            percent = 100 * i / float(self.N)
            out.write('[{} {}/{} {:.2f}% {}+{}]  {}'.format(msg,i,self.N,percent, Timer.sec2time(time_used) ,Timer.sec2time(time_left) ,end_char))
            out.flush()

class MyObject:
    @staticmethod
    def show_attr(obj):
        max_name_length = 0
        for name in obj.__dict__.keys():
            max_name_length = max(max_name_length, len(name))

        format_str = '{:>%ds} -> {}' % max_name_length
        for k,v in obj.__dict__.items():
            print(format_str.format(k,v))

    def __getitem__(self,key):
        return self.__dict__[key]

    def __setitem__(self,k,v):
        self.__dict__[k] = v

    def show(self):
        MyObject.show_attr(self)

class Logger:
    def __init__(self, fpath, with_time = True):
        self.file = fpath
        self.with_time = with_time
        self.write('--> LOG START <--', echo = False)

    def write(self, msg, echo = True):
        if echo:
            print(msg)

        if self.with_time:
            cur_time = Timer.get_time() + '\t'
        else:
            cur_time = ''

        with open(self.file, 'a+') as f:
            f.write('{}{}\n'.format(cur_time, msg))

class Timer:
    @staticmethod
    def get_time(fmt = "%Y_%m_%d_%H_%M_%S"):
        return time.strftime(fmt, time.localtime())

    @staticmethod
    def sleep(low, high=None, verbose = False):
        if high == None:
            secs=low
        else:
            secs=random.randint(low,high)
        if verbose:
            print("Sleep for {} seconds....".format(secs))
        time.sleep(secs)
        return

    @staticmethod
    def sec2time(t):
        return datetime.timedelta(seconds = int(t))

    def __init__(self):
        self.reset()

    def ring(self, msg = 'Execution complete'):
        now_time = time.time()
        elapsed_secs = now_time - self.checkpoint
        self.checkpoint = now_time
        print('{}, elapsed time: {}'.format(msg, Timer.sec2time(elapsed_secs)))

    def reset(self):
        self.checkpoint = time.time()


if __name__ == "__main__":
    print(Timer.get_time())
    pass
