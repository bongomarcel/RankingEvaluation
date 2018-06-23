'''
Created on 09 Apr 2016

@author: marcel
'''
from __future__ import division
import time
import copy_reg
import types

def _pickle_method(method):
    # Author: Steven Bethard (author of argparse)
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-
    # instancemethods
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    # Author: Steven Bethard
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-
    # instancemethods
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

# def f_init(q):
    # http://stackoverflow.com/a/3843313/190597 (Olson)
#     global fq
#     fq = q
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class Pipeline(object):
    '''The steps that should be processed sequentially and independently.
    Each data point should be independent of other data points.'''
    
    def __init__(self, steps, nr_threads=None, update_interval=1000, verbose=False):
        self.nr_threads = nr_threads
        self.update_interval = update_interval
        self.verbose = verbose
        self.steps = list()
        for step in steps:
            if isinstance(step, tuple) and len(step) == 2 and isinstance(step[1], tuple):
                self.steps.append(step)
            elif hasattr(step, '__call__'):
                self.steps.append((step, None))
            else:
                print('Warning: wrong parameters for pipeline: ', step)
                print('Warning: has to be of type: [(method) or (method, (tuple of parameters))]')
        if self.verbose:
            print('Pipeline created with the following functions:')
            print([func.__name__ + ':' + str(params) for func, params in self.steps])
    
    def execute(self, data, f_result_handler=None, chunksize=50, pool=None):
        from multiprocessing import Pool, TimeoutError
        closePool = False
        if pool is None:
            closePool = True
            pool = Pool(self.nr_threads)
        if self.verbose:
            print 'Created pool with %d processes.' %pool._processes
        time_begin = time.time()
        count = len(data)
        if self.verbose:
            print 'Items to process: %d' %(count)         
        if isinstance(data, dict):
            all_results = pool.imap_unordered(self.wrapper, data.iteritems(), chunksize=chunksize)
        elif isinstance(data, list):
            all_results = pool.imap_unordered(self.wrapper, data, chunksize=chunksize)
        else:
            print 'Warning data type not list or dictionary, returning'
            if closePool:
                pool.close()
                pool.join()
            return
        results = list()
        for c in range(1, count + 1):
            if c % self.update_interval == 0 and self.verbose:
                print 'Done %d of %d, time=%f' %(c, count, (time.time()-time_begin)/60.0)
            try:
                result = all_results.next()
            except TimeoutError: # TimeoutError can only be thrown if .next() is called with a timeout and imap chunksize is 1.
                print('Timeout')
            if f_result_handler is None:
                results.append(result)
            else:
                try:
                    iter(result)
                    f_result_handler(*result)
                except TypeError, te:
                    f_result_handler(*[result])
                
        if closePool:
            pool.close()
            pool.join()
        if self.verbose:
            print 'Total time=%f' %((time.time() - time_begin)/60.0)
        return results
     
    
    def wrapper(self, data): #this is the pipeline where results are parameters for the next function
        d = data
        for func, params in self.steps:
            try:
                iter(d)
            except TypeError, te:
                d = [d]
            if params is None:
                d = func(*d)
            else:
                d = func(*d + params)
        return d

def result_handler(paper_id, data):
    print('result:', paper_id, data)
    
def f1(paper_id, data, param1, param2):
    return (paper_id, data**(param1+param2))

def f2(paper_id, data):
    return (paper_id, data / 2)


def dummy(k, v):
    return k * v
    
    