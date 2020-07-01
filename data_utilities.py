#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data_utilities


# In[ ]:





# In[1]:


import argparse
import shutil
import pathlib
import os
import sys
import glob


# In[2]:


import logging
import decorator
import time
import inspect

import numbers
import decimal


# In[3]:


from importlib import reload


# In[4]:


sys.path.append('/Applications/anaconda3/lib/python3.7/site-packages/splunk_sdk-1.6.12-py3.7.egg')
import splunklib.client as client
import splunklib.results as results


# In[5]:


from ipywidgets import widgets, interact


# In[231]:


from IPython.display import display
from IPython.display import Javascript as d_js
from IPython.display import HTML
from IPython.display import display_html


# In[8]:


import pickle


# In[17]:


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from pandas.io.json import json_normalize
from pandas import DataFrame as DF
pd.options.display.max_rows= 999


# In[12]:


import ast


# In[13]:


from numbers import Number


# In[14]:


import numpy as np


# In[15]:


import re
import json
import warnings
import csv
import string
import functools
import statistics
import math as m


# In[16]:


from copy import deepcopy
from glob import glob


# In[25]:


from dateutil import parser as dateparser
import datetime as dt


# In[19]:


import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# In[20]:


from itertools import tee
from itertools import zip_longest
from itertools import chain
import itertools as it
from itertools import groupby
from itertools import takewhile


# In[21]:


from operator import *
from operator import itemgetter
from pprint import pprint


# In[22]:


from collections import OrderedDict
from collections import Counter
from functools import partial


# In[26]:


import warnings
warnings.filterwarnings('ignore')


# In[27]:


import random


# In[28]:


from contextlib import contextmanager


# In[29]:


import getpass


# In[ ]:





# In[ ]:


#####


# In[30]:


path_parent = lambda path: str(Path(path).parent)


# In[31]:


def argv_pairs_to_dict(argv_list):
    return rc(mat([ALL],lambda x:x.split('=')), select(lambda x: len(x)==2), dict, key_map(lambda s: s.lstrip('-')))(argv_list)


# In[32]:


def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'), raw=True)


# In[33]:


execfile = lambda file: exec(open(file).read())


# In[34]:


# transpose of DataFrame
DFT = lambda d: DF(d).T


# In[36]:


# https://stackoverflow.com/a/567697/1472770
def retry(howmany, *exception_types, **kwargs):
    timeout = kwargs.get('timeout', 0.0) # seconds
    @decorator.decorator
    def tryIt(func, *fargs, **fkwargs):
        for _ in range(howmany):
            try: return func(*fargs, **fkwargs)
            except exception_types or Exception:
                if timeout is not None: time.sleep(timeout)
    return tryIt


# In[37]:


def insert_at(sub, index):
    return lambda x: x[:index] + sub + x[index:]


# In[38]:


def notebook_name(path=False):
    d_js("iPython.notebook.kernel.execute('nb_name =\"' + iPython.notebook.notebook_name + '\"' ')")
    if path:
        return os.path.join(os.getcwd(), nb_name)
    else:
        return nb_name


# In[39]:


def string_split(c):
    return lambda s: s.split(c)


# In[40]:


def part(i):
    return lambda x: x[i]


# In[41]:


def isa(t):
    return lambda x: isinstance(x,t)


# In[42]:


def isnota(t):
    return lambda x: not isinstance(x,t)


# In[43]:


def isequal(v):
    return lambda x: x==v


# In[44]:


def isunequal(v):
    return lambda x: x!=v


# In[45]:


def greater_than(v):
    return lambda x: x>v


# In[46]:


def greater_equal_than(v):
    return lambda x: x>v


# In[47]:


def less_than(v):
    return lambda x: x<v


# In[48]:


def less_equal_than(v):
    return lambda x: x<=v


# In[49]:


def in_any(y):
    return lambda d: any([x for x in d if y in x])


# In[50]:


def in_all(y):
    return lambda d: all([x for x in d if y in x])


# In[51]:


def is_int_string(s):
    assert(isinstance(s,str))
    return all(map(str.isdigit,s))


# In[53]:


def join_varg(x):
    def wrap_non_iterable(x):
        if isinstance(x,str):
            return [x]
        elif not hasattr(x,'__iter__'):
            return [x]
        else:
            return x
        
    return trap(join)(*mat([ALL],wrap_non_iterable)(x))


# In[54]:


def len_truncate(n):
    return lambda x:x[:n]


# In[55]:


def __is_in(x,lst):
    if isinstance(lst,(list,tuple)):
        return x in lst
    elif isinstance(lst,str):
        if isinstance(x,str):
            return x in lst
        else:
            return False
    else:
        return False
    
def is_in(lst):
    return lambda x: __is_in(x,lst)


# In[58]:


def __replace(x,fg):
    assert isinstance(fg,dict)
    for f,g in fg.items():
        if f(x):
            return g(x)
    return x

def replace(fg):
    return lambda x: __replace(x,fg)


# In[59]:


def through(*f):
    return lambda x: tuple([g(x) for g in f])


# In[60]:


def through_d(d):
    return lambda x: {k(x):f(x) for k,f in d.items()}


# In[61]:


def lay(d):
    data = {'type': type(d)}
    if isinstance(d,(dict,list,tuple)):
        data = join(data,{'len':len(d), '1-level': mat([ALL], through(type,len))(d)})
    return data


# In[62]:


isnumeric = lambda x: x.isnumeric()


# In[63]:


def contains(y):
    return lambda x: y in x


# In[64]:


def containedin(y):
    return lambda x: x in y


# In[65]:


def contains_q(x,y):
    try:
        return x in y
    except:
        return False


# In[68]:


def set_venn(a,b):
    seta = set(a)
    setb = set(b)
    return {'intersection': tuple(seta.intersection(setb)), 'union': tuple(seta.union(setb)), 'a-b': tuple(seta-setb), 'b-a':tuple(setb-seta)}


# In[69]:


def apply(f):
    return lambda x:f(*x)


# In[71]:


minus = lambda x,y: x-y


# In[72]:


def reverse(x):
    if isinstance(x,(list,tuple)):
        return x[::-1]
    elif instance(x,dict):
        return dict(tuple(x.items())[::-1])


# In[ ]:


# DICT


# In[73]:


def __position(f,l):
    if isinstance(l,(list,tuple)):
        return [i for i,j in enumerate(l) if f(j)]
    if isinstance(l,dict):
        return [k for k,v in l.items() if f(v)]

def position(f):
    return lambda l: __position(f,l)


# In[75]:


def venn_pair__(d,ka,kb,union_False,out_format=tuple):
    seta = set(d[ka])
    setb = set(d[kb])
    f = out_format
    if union:
        return {(ka,'intersection',kb): f(seta.intersection(setb)), 
                (ka, 'minus', kb): f(seta-setb),
                (kb, 'minus', ka): f(setb-seta),
                (ka,'union',kb): f(seta.union(setb))}
    else:
        return  {(ka,'intersection',kb): f(seta.intersection(setb)), 
                (ka, 'minus', kb): f(seta-setb),
                (kb, 'minus', ka): f(setb-seta)}

def venn_pair(ka, kb, **kwarg):
    return lambda d: venn_pair__(d,ka,kb, **kwarg)


# In[76]:


#useful for key_value_map
kv_key = lambda k,v: k
kv_value = lambda k,v: v


# In[77]:


def key_value_map(f,g):
    return lambda d: {f(k,v): g(k,v) for k,v in d.items()}


# In[79]:


def prepend(d2):
    return lambda d: join(d2,d)

def append(d2):
    return lambda d: join(d,d2)


# In[80]:


def delete_empty_items(x):
    if isinstance(x,(list,tuple)):
        return type(x)([y for y in x if y not in [{},[],()]])
    elif isinstance(x,dict):
        return {k:v for k,v in x.items() if v not in [{},[],()]}


# In[81]:


# improve by adding optional lambda to filter similar none-type missing values
def delete_none(d):
    if isinstance(d,(list,tuple)):
        return [x for x in d if x!=None]
    if isinstance(d,dict):
        return {k:v for k,v in d.items() if v!=None}


# In[82]:


def __delete_by(d,f):
    if isinstance(d,(list,tuple)):
        return [x for x in d if not f(x)]
    elif isinstance(d,dict):
        return {k:v for k,v in d.items() if not f(v)}
    
def delete_by(f):
    return lambda d: __delete_by(d,f)


# In[83]:


def ismissing(x):
    if x in (None,'Missing'):
        return True
    else:
        return False


# In[84]:


def dict2tuples(d):
    return tuple([(k,v) for k,v in d.items()])

def tuples2dict(t):
    return {x[0]:x[1] for x in t}


# In[85]:


def dict_thread(keys,vlist):
    return [dict(zip(*(keys,v))) for v in vlist]


# In[86]:


def df_dict(df,**kwarg):
    return list(df.to_dict(**kwarg).values())


# In[88]:


keys = lambda x: tuple(x.keys())
values = lambda x: tuple(x.values())
items = lambda x: tuple(x.items())


# In[90]:


def __select(f,x):
    if isinstance(x,(list,tuple)):
        return type(x)(filter(f,x))
    elif isinstance(x,dict):
        return dict(filter(rc(last,f), x.items()))
    
def select(f):
    return lambda x: __select(f,x)

def __key_select(f,x):
    return dict(filter(rcomp(first,f), x.items()))

def key_select(f):
    return lambda x: __key_select(f,x)


# In[92]:


def __nest(f,x,n):
    if n==0:
        return x
    else:
        return __nest(f,f(x),n-1)
    
def nest(f,n):
    return lambda x: __nest(f,x,n)

def __nest_list(f,x,n):
    if n==0:
        return (x)
    else:
        return join_varg(((x), __nest_list(f,f(x),n-1)))
        
def nest_list(f,n):
    return lambda x: __nest_list(f,x,n)

# collatz = lamda x: 3*x+1 if x%2==1 else int(x/2)
# __nest_list(collatz,17,20)


# In[93]:


def any_in(a,b):
    return any([x for x in a if x in b])

def all_in(a,b):
    return any([x for x in a if x in b])


# In[94]:


def to_tuple(*arg):
    return tuple(arg)


# In[95]:


def arg_sum(*arg):
    return sum(tuple(arg))


# In[96]:


def string_join(*arg):
    return ''.join(tuple(arg))


# In[97]:


def string_riffle(sep, varg_input=False):
    if varg_input:
        return lambda *arg: sep(join(tuple(arg)))
    else:
        return lambda lst: sep.join(tuple(lst))


# In[98]:


def __seq_riffle(l,sep):
    return type(l)(chain(*[(i,sep) for i in l]))[0:-1]

def seq_riffle(sep):
    return lambda l: __seq_riffle(l,sep)


# In[99]:


def outer(f,g):
    return lambda d1,d2: {f(k1,k2):g(v1,v2) for k1,v1 in d1.items for k2,v2 in d2.items()}


# In[100]:


#renamed from out_pickle
def __to_pickle(fname,data):
    print('pickle to ' + fname)
    out_file = open(fname,'wb')
    pickle.dump(data,out_file)
    out_file.close()
    
def to_pickle(fname):
    return lambda d: __to_pickle(fname,d)


# In[101]:


# renamed from in_pickle
def from_pickle(fname):
    in_file = open(fname,'rb')
    res = pickle.load(in_file)
    in_file.close()
    return res


# In[102]:


def drop_file_suffix(x):
    x_split = x.split('.')
    if len(x_split)==1:
        return x
    else:
        return '.'.join(x_split[:-1])


# In[103]:


def is_iterable(x):
    try:
        iter(x)
        return(True)
    except TypeError:
        return(False)


# In[104]:


# this should be args not kwarg
def __get(d,k,*kwarg):
    if isinstance(d,dict):
        try:
            return d.get(k,*kwarg)
        except:
            return d
    else:
        return {}.get(k,*kwarg)
    
def ge(k,*kwarg):
    return lambda d: __get(d,k,*kwarg)


# In[105]:


def __key_take(k,keys):
    return {k:v for k,v in d.items() if k in keys}

#def keyTake(keys):
#    return lambda d: __key_take(d,keys)

def key_take(keys):
    return lambda d: __key_take(d,keys)

def __key_drop(d,keys):
    return {k:v for k,v in d.items() if k not in keys}

#def keyDrop(keys):
#    return lambda d: __key_drop(d,keys)

def key_drop(keys):
    return lambda d: __key_drop(d,keys)

def __key(d,k):
    return d.get(k)

def key(k):
    return lambda d: __key(d,k)

# renamed from dict_slice


# In[106]:


def __key_select(f,d):
    return {k:v for k,v in d.items() if f(k)}

def key_select(f):
    return lambda d: __key_select(f,d)


# In[108]:


def true(x):
    return True

def false(x):
    return False


# In[110]:


def keys(d):
    return tuple([k for k,v in d.items()])

def values(d):
    return tuple([v for k,v in d.items()])


# In[111]:


def to_key(k):
    return lambda d: {k:d}

def enumerate_dict_values(lst):
    return {x[1]:x[0] for x in enumerate(lst)}

def enumerate_dict_keys(lst):
    return {x[0]:x[1] for x in enumerate(lst)}

key_enumerate = enumerate_dict_keys


# In[112]:


def total(d):
    return sum([v for k,v in d.items()])


# In[113]:


def join(*l):
    if isinstance(l[0], (list,tuple)):
        return [i for j in l for i in j]
    elif isinstance(l[0],dict):
        return {k:v for j in l for k,v in j.items()}
    
update = join


# In[114]:


def __dict_slice(d,slice):
    if type(slice) is int:
        k = list(d)[slice]
        return d[k]
    else:
        return {k:d[k] for k in list(d)[slice]}
    
def dict_slice(slice):
    return lambda d: __dict_slice(d,slice)


# In[115]:


def __take(x,*slc):
    if isinstance(x,(list,tuple)):
        return x[slice(*slc)]
    
def take(*slc):
    return lambda x: __take(x,*slc)


# In[116]:


def dindex(i):
    return lambda d: d[list(d)[i]]

dix = dindex


# In[117]:


def ix(i):
    return lambda d: d[i]


# In[118]:


def dslice(*args):
    return lambda d: {k:d[k] for k in list(d)[slice(*args)]}


# In[273]:


def map_dict(f,d):
    return {k:f(v) for k,v in d.items()}

def __dict_map(f,d):
    if isinstance(d, (list,tuple)):
        return {x:f(x) for x in d}
    elif isinstance(d,dict):
        return {k:f(v) for k,v in d.items()}

def dict_map(f):
    return lambda d: __dict_map(f,d)


# In[120]:


def __key_map(f,dict):
    return {f(k):v for k,v in dict.items()}

def key_map(f):
    return lambda d: __key_map(f,d)


# In[122]:


def __key_subkey_map(f,d):
    return key_value_map(lambda k,v:k, lambda k,v: key_map(lambda x: f(k,x))(v))(d)

def key_subkey_map(f=lambda k,x: (k,x)):
    return lambda d: __key_subkey_map(f,d)


# In[123]:


def map_dispatch(dispatch, only=False):
    keys=list(dispatch)
    if(only):
        return lambda d: {k:dispatch.get(k, lambda x:x)(v) for k,v in d.items() if k in keys}
    else:
        return lambda d: {k:dispatch.get(k, lambda x:x)(v) for k,v in d.items()}


# In[125]:


def __sget(data,path):
    assert len(path)>0
    head,rest = (path[0], path[1:])
    if len(path)==1:
        return trap(lambda x:x[head])(data)
    else:
        return trap(lambda x: __sget(x[head],rest))(data)
        
def sget(path):
    return lambda data: __sget(data,path)


# In[126]:


get_path = lambda path: rcomp(*[get(x) for x in path])


# In[131]:


class All(object):
    def __init__(self):
        pass
    
ALL = All()

def match_q(x, item):
    if isinstance(item,All):
        return True
    try:
        if isinstance(item,str):
            return x==item
        return x==item or x in item
    except TypeError:
        return False
    
def query_path_item(data, path_item, f):
    if isinstance(data, dict):
        return {k:(f(v) if match_q(k, path_item) else v) for k,v in data.items()}
    if isinstance(data, (list,tuple)):
        return [f(x[1]) if match_q(x[0],path_item) else x[1] for x in enumerate(data)]
    
def get_path_item(data, path_item, f):
    if isinstance(data, dict):
        out = {k:f(v) for k,v in data.items() if match_q(k,path_item)}
        return out
    if isinstance(data, (list,tuple)):
        out = [f(x[1]) for x in enumerate(data) if match_q(x[0],path_item)]
        return out
    
def get_select_path_item(data, path_item, f,g):
    if isinstance(data,dict):
        matches = {k:f(v) for k,v in data.items() if match_q(k,path_item) and g(v)}
        if isinstance(path_item,(list,All)):
            return matches
        else:
            return matches.get(path_item)
    elif isinstance(data, (list,tuple)):
        matches = [f(x[1]) for x in enumerate(data) if match_q(x[0],path_item) and g(x[1])]
        if isinstance(path_item,(list,All)):
            return matches
        else:
            return matches[0]
        
def drop_path_item(data, path_item, f):
    if isinstance(data, dict):
        return {k:f(v) for k,v in data.items() if not match_q(k,path_item)}
    if isinstance(data, (list,tuple)):
        return [f(x[1]) for x in enumerate(data) if not match_q(x[0],path_item)]
    
def query(data, path, **kwarg):
    if path==[]:
        head=ALL
    else:
        head,rest = (path[0],path[1:])
    if len(path)>1:
        if 'MAP' in kwarg or 'SET' in kwarg:
            return query_path_item(data, head, lambda x: query(x, rest, **kwarg))
        else:
            return get_path_item(data,head,lambda x: query(x,rest,**kwarg))
    else:
        if 'GET' in kwarg:
            if 'SELECT' in kwarg:
                return get_select_path_item(data,head,lambda x: kwarg.get('GET')(x),lambda x: kwarg.get('SELECT')(x))
            else:
                return get_path_item(data, head, lambda x: kwarg.get('GET')(x))
        elif 'SET' in kwarg:
            return query_path_item(data, head, lambda x: kwarg.get('SET'))
        elif 'MAP' in kwarg:
            return query_path_item(data, head, lambda x: kwarg.get('MAP')(x))
        elif 'DROP' in kwarg:
            return get_path_item(data, head, lambda x: key_drop(x, kwarg.get('DROP')))
        else:
            # implicit GET
            return get_path_item(data, head, lambda x:x)


# In[133]:


def query2(data, path, **kwarg):
    if path==[]:
        head=All
    else:
        head, rest = (path[0], path[1:])
    if len(path)>1:
        if 'MAP' in kwarg or 'SET' in kwarg:
            return query_path_item(data, head, lambda x: query(x,rest,**kwarg))
        elif 'GET' in kwarg:
            if 'SELECT' in kwarg:
                #print('in select')
                return get_select_path_item(data, head, lambda x: query(x, rest, **kwarg))
            else:
                return get_path_item(data, head, lambda x: kwarg.get('GET')(x))
        else:
            return get_path_item(data, head, lambda x: query(x, rest, **kwarg))
    else:
        if 'GET' in kwarg:
            if 'SELECT' in kwarg:
                return get_select_path_item(data, head, lambda x: kwarg.get('GET')(x), lambda x: kwarg.get('SELECT')(x))
            else:
                return get_path_item(data,head, lambda x: kwarg.get('GET')(x))
        elif 'SET' in kwarg:
            return query_path_item(data, head, lambda x: kwarg.get('SET'))
        elif 'MAP' in kwarg:
            return query_path_item(data, head, lambda x: kwarg.get('MAP')(x))
        elif 'DROP' in kwarg:
            return get_path_item(data, head, lambda x: key_drop(x,kwarg.get('DROP')))
        else:
            # implicit GET
            return get_path_item(data, head, lambda x:x)


# In[135]:


def map_at(path,f):
    if path==[]:
        return lambda d: f(d)
    else:
        return lambda d: query(d,path,MAP=f)
    
def get_at(path,f,g=true):
    return lambda d: query(d,path,GET=f,SELECT=g)

mat = map_at
gat = get_at


# In[136]:


def level_transpose(i,j=()):
    tr_fun = rcomp(transpose,mat([ALL],delete_none))
    if j==():
        j = i+1
    if abs(i-j)==1:
        return mat([ALL]*min(i,j), tr_fun)
    else:
        level_range = join(list(range(i,j)),list(range(i,j))[:-1][::-1])
        return rcomp(*[mat([ALL]*k,tr_fun) for k in level_range])


# In[137]:


def level_cycle(*args):
    if len(args)<2:
        return level_transpose(*args)
    else:
        return rcomp(*[level_transpose(*x) for x in kgram(2)(args)])


# In[138]:


is_not = lambda x: x is not True


# In[139]:


def is_empty(x):
    return x==[] or x=={}


# In[140]:


def is_not_empty(x):
    return not is_empty(x)


# In[141]:


# http://code.activestate.com/recipes/577879-create-a-nested-dictionary-from-oswalk/

def directory_dict(rootdir):
    dir = {}
    rootdir = rootdir.rstrip(os.sep)
    start = rootdir.rfind(os.sep) + 1
    for path, dirs, files in os.walk(rootdir):
        folders = path[start:].split(os.sep)
        subdir = dict(map(lambda x: (x,path+x if path[-1]==os.sep else path+os.sep+x),files))
        parent = functools.reduce(dict.get, folders[:-1], dir)
        parent[folders[-1]] = subdir
    return dir[folders[0]]


# In[142]:


# use in nested dicts to filter out by key or value (eg directory tree walk)

def recursive_value_filter(data0,value):
    data = deepcopy(data0)
    if isinstance(data,dict):
        return {k:recursive_value_filter(v,value) for k,v in data.items() if v!=value}
    return data

def recursive_key_filter(data,**kwarg):
    options = {'filter':(), 'key_map':identity, 'value_map':identity}
    options.update(**kwarg)
    
    f = options.get('key_map')
    g = options.get('value_map')
    key = options.get('filter')
    
    if isinstance(data,dict):
        if isinstance(key,(list,tuple)):
            return {f(k):recursive_key_filter(v,**kwarg) for k,v in data.items() if k not in key}
        else:
            return {f(k):recursive_key_filter(v,**kwarg) for k,v in data.items() if k!=key}
    return g(data)


# In[143]:


# partition a list into overapping consecutive pairs
# deprecated by nwise
def partition_list(k):
    return [[f,s] for f,s in zip(x,x[1:])]


# In[145]:


def nwise(lst,k=2):
    return list(zip(*[lst[i:] for i in range(k)]))


# In[148]:


# operator form of nwise, but pass k explicitly
def kgram(k):
    return lambda lst: nwise(lst,k)


# In[150]:


def unique_consecutive(x):
    return [v for i,v in enumerate(x) if i==0 or v!=x[i-1]]

dedup_consecutive = unique_consecutive


# In[151]:


def __dedup_consecutive_by(f,x):
    return [v for i,v in enumerate(x) if i==0 or f(v)!=f(x[i-1])]

def dedup_consecutive_by(f):
    return lambda x: __dedup_consecutive_by(f,x)


# In[153]:


def __unique_consecutive_d(d,**kwarg):
    options = {'pick':dslice(0,1)}
    options.update()
    return rcomp(splitby_d(**kwrag),mat([ALL],options['pick']),join_varg)(d)

def unique_consecutive_d(**kwarg):
    return lambda d: __unique_consecutive_d(d,**kwarg)


# In[154]:


def is_subseq(x,y):
    it = iter(y)
    return all(c in it for c in x)


# In[155]:


def islice_dict(dict,*pos):
    return dict(islice(dict.items()),*pos)


# In[158]:


def __block_map(f,ls,n,offset=1,delete_tail=False):
    if isinstance(ls,(list,tuple)):
        if delete_tail:
            return type(ls)([f(ls[i:i+n]) for i in range(0,len(ls)-1,offset) if len(ls[i:i+n])==n])
        else:
            return type(ls)([f(ls[i:i+n]) for i in range(0,len(ls)-1,offset)])
    elif isinstance(ls,dict):
        return __block_map(rcomp(dict,map_at([ALL],f)),list(ls.items()),n,offset,delete_tail)

def block_map(f,*arg,**kwarg):
    return lambda ls: __block_map(f,ls,*arg,**kwarg)


# In[159]:


# zip longest 
from itertools import zip_longest
def grouper(iterable, n, fill=None):
    args = [iter(iterable)]*n
    return zip_longest(*args, fillvalue=fill)


# In[160]:


# returns a sliding window of width n over iterable
from itertools import islice
def window(seq,n=2):
    it = iter(seq)
    result = tuple(islice(it,n))
    if len(result)==n:
        yield result
    for elem in it:
        result = result[1:]+(elem,)
        yield result


# In[161]:


# return indices of list elements based on predicate
def find_indices(lst,cond):
    return [i for i,elem in enumerate(lst) if cond(elem)]


# In[162]:


# improved version of splitby_d that works with both lists/tuples and dict - return a nested list of segments
def __splitby(d,f,**kwarg):
    if len(d)<2:
        return [d]
    else:
        if isinstance(d,dict):
            return [dict(v) for k,v in groupby(dict_to_tuple(d), key=rcomp(last,f))]
        elif isinstance(d,(list,tuple)):
            return [list(v) for k,v in grouopby(d,key=f)]
        
def splitby(f,**kwarg):
    return lambda d: __splitby(d,f,**kwarg)


# In[163]:


#a slighly faster solution not involving a callable can be done with 
# https://stackoverflow.com/questions/54315501/pands-stack-groupby-to-dataframe-multiindex-without-aggregating
def group_to_index(df,grouper):
    return pd.concat({k: v for k, v in df.groupby(by=grouper)})


# In[167]:


# fails with None
type_markup = lambda x: (x is not None, '' if isinstance(x,Number) else type(x).__name, x)


# In[170]:


def groupby_dict(d, f, **kwarg):
    vf = kwarg.get('map')
    if isinstance(d,(list,tuple)):
        if vf:
            return {k:vf(list(g)) for k,g in grouopby(sorted(d, key=lambda x: type_markup(f(x))), f)}
        else:
            return {k:list(g) for k,g in groupby(sorted(d, key=lambda x: type_markup(f(x))), f)}
    elif isinstance(d,dict):
        grouped = grouopby_dict(list(d.values()), f)
        if vf:
            return {k0:{k:vf(v) for k,v in d.items() if v in v0} for k0,v0 in grouped.items()}
        else:
            return {k0:{k:v for k,v in d.items() if v in v0} for k0,v0 in grouped.items()}
    else:
        raise TypeError('unsupported operand type', type(d))


# In[171]:


# bug at len(x[1])- possibly due to refactor of sort_dict
def groupby_sort_dict(d,f,**kwarg):
    dg = groupby_dict(d,f,**kwarg)
    return sort_dict(dg,key=lambda x:len(x), reverse=True)


# In[172]:


def groupby_d(f,**kwarg):
    return lambda d: groupby_dict(d,f,**kwarg)


# In[173]:


# this groups by key and uses key itself as a higher key

def groupby_key(k,**kwarg):
    options = {'superkey':False, 'missing_value':None}
    options.update(**kwarg)
    if options['superkey']:
        return lambda x: to_key(k)(groupby_sort_dict(x,lambda x:x.get(k,options['missing_value'])))
    else:
        return lambda x: groupby_sort_dict(x,lambda x:x.get(k,options['missing_value']))
    
def groupby_nokey(k):
    return lambda x: groupby_sort_dict(x,lambda x:x.get(k))


# In[174]:


def dekey(d):
    k = list(d.keys())
    return d[k[0]]


# In[175]:


def keygroupby_dict(d,f):
    if type(d) == dict:
        grouped = groupby_dict(list,d.keys(), f)
        return {k0: {k:v for k,v in d.items() if k in v0} for k0,v0 in grouped.items()}
    else:
        raise TypeError('unsuppored operand type', type(d))
        
def keygroupby_d(f):
    return lambda d: keygroupby_dict(d,f)


# In[176]:


def list_trie(d):
    if isinstance(d,list):
        return grouby_dict(d,first,map=lambda x:list_trie([y for y in query(x,[ALL],MAP=rest) if y!=[]]))
    else:
        raise TypeError("unsupported operand type ", type(d))


# In[180]:


def thread_keys(keys):
    return lambda vals: dict(zip(keys,vals))

def thread_values(vals):
    return lambda keys: dict(zip(keys,vals))


# In[181]:


#first = lambda x: x[0]
#last = lambda x: x[-1]
def nth(n):return lambda x: x[n]


# In[182]:


identity = lambda x:x


# In[183]:


def iloc(i):
    def _iloc(d):
        partial = DF.from_dict(d,orient='index').iloc[i][0]
        try:
            return partial.to_dict()
        except AttributeError:
            return partial
    return _iloc


# In[184]:


def sort_dict(d,**kwarg):
    if kwarg.get('key'):
        kwarg['key'] = rcomp(last,kwarg['key'])
    else:
        kwarg['key'] = lambda x: x[1]
    return dict(sorted(d.items(), **kwarg))

def sort_d(**kwarg):
    return lambda d: sort_dict(d,**kwarg)


# In[185]:


def keysort_dict(d, **kwargs):
    if isinstance(d,dict):
        return {k:d[k] for k in sorted(d.keys(), **kwargs)}
    else:
        raise TypeError("unsupported argument type")


# In[186]:


def keysort_dict__(**kwargs):
    return lambda d: {k:d[k] for k in sorted(d.keys(), **kwargs)}

def keysort_d(**kwawrgs):
    return keysort_dict__(**kwargs)


# In[187]:


# sort pandas dataframe

def __sort_df(f,df0,**kwarg):
    df = df0.copy()
    df = df.assign(sort_value=f).sort_values('sort_value', **key_drop('show_sort_values')(kwarg))
    if(kwarg.get('show_sort_values')):
        return df
    else:
        return df.drop(columns=['sort_value'])
    
def sort_df(f,**kwarg):
    return lambda df0: __sort_df(f,df0,**kwarg)


# In[188]:


# experimental missing-value object - to capture missing(None), missing(TypeError) uniformly
class Missing(object):
    def __init(self, value=None):
        self.value = value
    def __repr__(self):
        return str(self.value)


# In[189]:


def first_keys_to_level(j):
    return lambda d: {k:rc(*([dix(0)]*k),rc(keys,first))(d) for k in range(j)}


# In[191]:


def merge_dict(f,*args,filterNone=True,missing_handler=lambda x:x!=None):
    outkeys = set().union(*[x.keys() for x in args])
    def outvals(x):
        return [y.get(x,None) for y in args]
    if(filterNone):
        return {x:f(tuple(filter(missing_handler,outvals(x)))) for x in outkeys}
    else:
        return {x:f(tuple(outvals(x))) for x in outkeys}
    
def merge_d(f,**kwarg):
    return lambda d: merge_dict(f,*d,**kwarg)


# In[192]:


def flatten_list(l):
    return type(l)(chain.from_iterable(l))


# In[193]:


def flatten_container(container):
    for i in container:
        if isinstance(i,(list,tuple)):
            for j in flatten_container(i):
                yield j
            else:
                yield i


# In[194]:


def flatten(c):
    return type(c)(flatten_container(c))


# In[197]:


def flatten_at(pos):
    if isinstance(pos,int):
        pos = [pos]
    return lambda lst: join(*[v if k in pos else [v] for k,v in enumerate(lst)])


# In[198]:


dict_to_tuples = lambda d: [(k,v) for k,v in d.items()]
tuples_to_dict = lambda t: {x[0]:x[1] for x in t}


# In[200]:


# stackoverflow 19647596/1472770
def __flatten_keys(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
           } if isinstance(dd, dict) else {prefix: dd}

def flatten_keys(**kwarg):
    return lambda d: __flatten_keys(d,**kwarg)


# In[201]:


# stackoverflow 56877666 flatten-nested-dictionaries-with-tuple-keys
def flatten_dict(deep_dict,join_values=False):
    def do_flatten(deep_dict, current_key):
        for key, value in deep_dict.items():
            new_key = current_key + (key,)
            if isinstance(value,dict):
                yield from do_flatten(value, new_key)
            else:
                yield (new_key, value)
                
    def do_value_join(k,v):
        if isinstance(v,(list,tuple)):
            return join(k,v)
        else:
            return join(k,(v,))
        
    if join_values:
        return [do_value_join(k,v) for k,v in dict(do_flatten(deep_dict, ())).items()]
    else:
        return dict(do_flatten(deep_dict, ()))


# In[202]:


# TODO: this is suspect, why split
def flatten_d(d):
    return [k.split('_')+[v] for k,v in flatten_dict(d,separator='_').items()]


# In[204]:


def transpose(d,**kwarg):
    #options = {'fillna':np.nan}
    options = {'fillna':None}
    options.update(**kwarg)
    
    if isinstance(d,(list,tuple)) and all([isinstance(x,(list,tuple)) for x in d]):
        return list(zip(*d))
    if isinstance(d,(list,tuple)) and all([isinstance(x,dict) for x in d]):
        return {k:type(d)(v.values()) for k,v in transpose(key_enumerate(d),**options).items()}
    elif isinstance(d,dict):
        inner_keys = list(set(flatten_list([v.keys() for k,v in d.items()])))
        return {i:{k:v.get(i,options['fillna']) for k,v in d.items()} for i in inner_keys}
    
def transpose__(**kwarg):
    return lambda d: transpose(d,**kwarg)

tr = transpose__


# In[205]:


# keys_to_dict(['a','b'])((10,20))

def __keys_to_dict(data,keys):
    if len(data)==len(keys):
        return dict(transpose((keys,data)))
    else:
        raise ValueError('Length of data and keys must match')
        
def keys_to_dict(keys):
    return lambda d: __keys_to_dict(d,keys)


# In[206]:


# gets the first key at leats level in a nested dict
def first_key_at_levels(i,j):
    return lambda d: {k:rc(*([dix(0)]*k),rc(keys,first))(d) for k in range(i,j)}


# In[207]:


# of form for list
def sort(**kwarg):
    return lambda x:sorted(x,**kwarg)


# In[208]:


def sorted_count(l):
    return {l:w for l,w in 
            sorted([(k,v) for k,v in Counter(l).items()], key=itemgetter(1),reverse=True)
           }


# In[209]:


def catch(errors):
    def catch_f(*args,**kwargs):
        try:
            return lambda f: f(*args,**kwargs)
        except errors as e:
            return e
    return catch_f


# In[210]:


def trap(f,**kwarg2):
    options = {'error': BaseException, 'value':'error'}
    options.update(**kwarg2)
    def __trap__(*args,**kwargs):
        try:
            return f(*args,**kwargs)
        except options['error'] as e:
            if options['value']=='error':
                return type(e)
            elif options['value']=='pass':
                pass
            else:
                return options['value']
    return __trap__


# In[211]:


def type_trap(f):
    def __trap__(*ars,**kwarg):
        try:
            return f(*args,**kwarg)
        except BaseException as e:
            return type(e)
    return __trap__


# In[213]:


def catch(func, handle=lambda e: e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return handle(e)


# In[214]:


def index(pos):
    return trap(lambda x: x[pos],error=IndexError)

first = index(0)
second = index(1)
third = index(2)

last = index(-1)
rest = index(slice(1,None))


# In[216]:


# for pandas Series or List
def value_counts_freq(ser,rounding=3):
    if type(ser) == pd.core.series.Series:
        tmp = se.value_counts(dropna=False)
        tmp2 = tmp.to_frame((ser.name,'#'))
        tmp3 = round(tmp/tmp.sum(),rounding).to_frame((ser.name,'%'))
        return pd.merge(tmp2,tmp3,left_index=True,right_index=True)
    
    elif isinstance(ser,(list,tuple)):
        return value_counts_freq(pd.Series(ser).rename(0),rounding)
    else:
        return ser
    
value_count_freq = value_counts_freq

def count_fre(ser,rounding=3):
    return value_counts_freq(ser,rounding)[0]

def count_freq(ser,rounding=3):
    options= {'rounding':3,'dict':True}
    options.update(**kwarg)
    
    if options['dict']:
        try:
            return value_counts_freq(ser,options['rounding'])[0].T.to_dict()
        except Excption as e:
            return type(e)
    else:
        try:
            return value_counts_freq(ser,options['roudning'])[0]
        except Exception as e:
            return type(e)
        
count_freq_dict = lambda x: count_freq(x,dict=False)
count_freq_df = lambda x: count_freq(x,dict=True)


# In[217]:


count = lambda d: rcomp(count_freq, mat([ALL],get('#')))(d)


# In[219]:


def freq(d):
    tot = sum(values(d))
    return query(d,[ALL],MAP=lambda x: {'#':x, '%': x/tot})


# In[221]:


def __len_freq(d,rounding=3,key='#'):
    lend = map_at([ALL],len)(d)
    tot = sum(lend.values())
    return sort_d(key=lambda x:x[key],reverse=True)(map_at([ALL],lambda x:{'#':x, '%':round(x/tot,rounding)})(lend))

def len_freq(rounding=3):
    return lambda d: __len_freq(d,rounding)


# In[222]:


# dataframe row and column sums
def df_row_col_sum(df):
    df['col_sum'] = df.sum(axis=1)
    df = df.append(df.sum(axis=0).rename('row_sum'),ignore_index=False)
    return df.sort_values(by='col_sum',ascending=False)


# In[223]:


def frange(start,stop,step):
    i = start
    while i < stop:
        yield i
        i += step


# In[224]:


def frange(start=0, stop=1, jump=0.1):
    nsteps = int((stop-start)/jump)
    dy = stop-start
    # f(i) goes from start to stop as i goes from 0 to nsteps
    return [start + float(i)*dy/nsteps for i in range(nsteps)]


# In[225]:


from fractions import Fraction

def rrange(start=0, stop=1, jump=0.1):
    nsteps = int((stop-start)/jump)
    return [Fraction(i, nsteps) for i in range(nsteps)]


# In[226]:


def levenshteinDistance(s1,s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
        
    distances = range(len(s1) + 1)
    for i1, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


# In[227]:


def tokenize(sr):
    return sr.apply(lambda x: str(x).lower()).apply(lambda x: " ".join([w for w in x.split() if w not in stopwords])).apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation + "-" + "0123456789"), ' ', x))


# In[228]:


def ngram_gen(sr,k):
    summary_grams = nwise(" ".join(tokenize(sr).tolist()).split(' '),k)
    return [" ".join(x) for x in summary_grams if "" not in x]


# In[229]:


def ngram_range_gen(sr,rng):
    summary_grams = {k:nwise(" ".join(tokenize(sr).tolist()).split(' '),k) for k in rng}
    return {k:[" ".join(x) for x in v if "" not in x] for k,v in summary_grams.items()}


# In[232]:


# display dataframes side by side

from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


# In[233]:


# stackoverflow 54315536/1472770
# coldspeed answer

def add_level(df,grouper):
    return pd.concat({k:v for k,v in df.groupby(by=grouper)})


# In[234]:


def map_items(f,x):
    return [f(i) for i in x.items()]


# In[235]:


def dict_to_tuple(d):
    if type(d)==dict:
        if len(d)>1:
            return [(k,dict_to_tuple(v)) for k,v in d.items()]
        else:
            return [(k,dict_to_tuple(v)) for k,v in d.items()][0]
    else:
        return d


# In[236]:


# stackoverflow 29920015/1472770
# camel case split

def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',identifier)
    return [m.group(0) for m in matches]


# In[237]:


#stackoverflow 43357135
# dima pasecnnik

def apply_f(a,f):
    try:
        return f(a)
    except:
        return map(lambda t:apply_f(t,f),a)


# In[238]:


res = apply_f([(1,2),[[5]],[7,(8,[9,11])]], lambda t:t**2)


# In[241]:


def __plot_d(d,**kwarg):
    return plt.plot(*(d.keys(),d.values()))

def plot_d(**kwarg):
    return lambda d: __plot_d(d,**kwarg)


# In[242]:


def list_scatterplot(lst):
    return plt.scatter(*zip(*lst))


# In[243]:


def char_range(mini, maxi, step=1):
    return [char(c) for c in range(ord(mini), ord(maxi)+1, step)]


# In[244]:


def any_all(lst):
    return {'any': any(lst), 'all': all(lst)}


# In[245]:


def min_max(lst):
    return [min(lst),max(lst)]


# In[246]:


def replace_by(d):
    return lambda x: d[x] if x in d.keys() else x


# In[247]:


def key_replace(d):
    return lambda d0: {replace_by(d)(k):v for k,v in d0.items()}


# In[248]:


def replace_value(oin,out):
    return lambda x: out if x==oin else x


# In[249]:


def replace_values(oin_list,out):
    return lambda x: out if x in oin_list else x


# In[250]:


def __case(d,f,g):
    if isinstance(d,dict):
        return {k:g(v) for k,v in d.items() if f(v)}
    elif isinstance(d,(list,tuple)):
        return [g(v) for v in d if f(v)]
    
def case(f,g=identity):
    return lambda d: __case(d,f,g)

def __key_case(d,f,g):
    return {g(k):v for k,v in d.items() if f(k)}

def key_case(f,g=identity):
    return lambda d: __key_case(d,f,g)


# In[251]:


# deprecated - uses cases dict form 
#def cases_pairs(*fg_list):
#    return rcomp(mat([ALL],lambda x: get(True)({f[0](x):f[1](x) for f in fg_list[::-1]})),delete_none) 


# In[252]:


def cases(fg):
    assert(isinstance(fg,dict))
    return rcomp(mat([ALL],lambda x: get(True)({f(x):g(x) for f,g in reverse(fg).items()})),delete_none)


# In[ ]:


#stats


# In[269]:


def quantiles(qlist,**kwarg):
    options = {'labeled':True}
    options.update(**kwarg)
    if options['labeled']:
        qlabels = {0:'min',0.25:'lqt', 0.5:'med', 0.75:'hqt', 1:'max'}
        return lambda data: {replace_by(qlabels)(x):np.quantile(data,x) for x in qlist}
    else:
        return lambda data: {x:np.quantile(data,x) for x in qlist}


# In[255]:


def labeled_quantiles(qlist):
    qlabels = {0:'min', 0.25:'lqt', 0.5:'med', 0.75:'hqt', 1:'max'}
    return lambda data: {x:np.quantile(data,x) for x in qlist}


# In[256]:


quartiles = [0, 0.25, 0.5, 0.75, 1.0]


# In[258]:


def quantile_stats(qt, **kwarg):
    return rc(groupby_d(lambda x: isinstance(x,numbers.Number)), mat([True],rc(through(quantiles(qt,**kwarg),rc(len,to_key('count'))),join_varg)), mat([False],rc(len,to_key('missing_count'))),
              lambda x: join_varg([x.get(True,join(dict_map(lambda x:None)(qt), {'count':0})), x.get(False,{'missing_count':0})]))


# In[274]:


#quantile_stats(quartiles)([1,2,4,4,4,4,6,2])


# In[260]:


def percentiles(*args,**kwargs):
    return rcomp(quantiles(*args,**kwargs), key_map(lambda x: str(100*x)+'%'))


# In[ ]:


# stackoverflow 17303428 colors

####
####

#print(color['BLUE'] + 'Data_' + color['RED'] + 'Utilities')


# In[ ]:


# activestate recipes 384122 infix operators



# In[262]:


# stackoverflow 548349333 (mad physicist)

def slice2range(s):
    step = 1 if s.step is None else s.step
    if (s.stop is None and step > 0) or (s.start is None and step <0):
        raise ValueError('Must have valid stop')
    start = 0 if s.start is None else s.start
    stop = -1 if s.stop is None else s.stop
    return range(start, stop, step)


# In[263]:


# generator approach to subset

def numberlist(nums, limit):
    def f(nums, limit):
        sum = 0
        for x in nums:
            sum += x
            yield x
            if sum > limit:
                return
            
    return list(f(nums,limit))


# In[266]:


def select_to_limit(d,func,limit):
    def f(d, limit):
        sum = 0
        for k,v in d.items():
            sum += func(v)
            yield {k:v}
            if sum > limit:
                return
                
    return functools.reduce(lambda x,y: {**x,**y}, f(d,limit))


# In[267]:


def right_compose(*fn):
    return lambda x: functools.reduce(lambda f, g: g(f), list(fn), x)

def compose(*fn):
    return right_compose(*reversed(fn))

r_comp = right_compose
rcomp = right_compose
rc = right_compose


# In[275]:


def value_len():
    return lambda kv: len(kv[1])


# In[276]:


def subsubme_kv(k_name, v_name='data'):
    return lambda d:[{k_name:k, vname:v} for k,v in d.items()]


# In[277]:


enumerate_dict = rcomp(enumerate,dict)


# In[279]:


def to_csv(file_path):
    def writer(data):
        with open(file_path, 'w', encoding='utf8', newline='') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
            return [tsv_writer.writerow(row) for row in data]
    return writer


# In[ ]:


# JSON


# In[282]:


def normalize_json_key_type(x):
    if isinstance(x,(dt.date,dt.datetime)):
        return x.isoformat()
    elif isinstance(x,(str,int, float, bool, type(None))):
        return x
    else:
        return str(x)
    
def normalize_json_value_type(x):
    if isinstance(x,(dt.date, dt.datetime)):
        return x.isoformat()
    elif isinstance(x,type):
        return str(x)
    else:
        return x

def normalize_json_container(d):
    if isinstance(d, dict):
        return {normalize_json_key_type(k): normalize_json_value_type(normalize_json_cointainer(v)) for k,v in d.items()}
    elif isinstance(d,(list,tuple)):
        return [normalize_json_value_type(normalize_json_container(x)) for x in d]
    else:
        return d


# In[283]:


def __to_json(path,data,normalize=True, **kwarg):
    options = {'sort_keys': False, 'default': str}
    options.update(**kwarg)
    with open(path,'w') as outfile:
        if(normalize):
            json.dump(normalize_json_container(data), outfile, indent=4, **options)
        else:
            json.dump(data, outfile, indent=4,**options)
            
def to_json(path,**kwarg):
    return lambda d: __to_json(path,d,**kwarg)


# In[285]:


def __from_json(fp,**kwarg):
    with open(fp,'r') as infile:
        return json.load(infile,**kwarg)
    
def from_json(**kwarg):
    return lambda fp: __from_json(fp,**kwarg)


# In[286]:


def __to_json_string(data,normalize=True,**kwarg):
    options = {'sort_keys':False, 'default':str}
    options.update(**kwarg)
    if(normalize):
        return json.dumps(normalize_json_container(data), indent=4,**options)
    else:
        return json.dumps(data, indent=4,**options)
    
def to_json_string(**kwarg):
    return lambda d: __to_json_string(d,**kwarg)


# In[ ]:


# date/time


# In[287]:


def datetimestring(**kwarg):
    opt = {'format':'compact','sep':''}
    opt.update(**kwarg)
    
    if opt['format']=='compact':
        return opt['sep'] + dt.datetime.now().strftime('%Y%m%d_%H%M') + opt['sep']
    if opt['format']=='iso':
        return opt['sep'] + dt.datetime.now().strftime('%Y-%m-%dT%H%M') + opt['sep']


# In[288]:


def date_range(d1,d2,**kwarg):
    options = {'format': '%Y-%m-%d'}
    options.update(**kwarg)
    
    d1_date = dt.datetime.strptime(d1,'%Y-%m-%d')
    d2_date = dt.datetime.strptime(d2,'%Y-%m-%d')
    days_range = (d2_date-d1_date).days
    
    date_list = [d1_date + dt.timedelta(days=x) for x in range(0,days_range+1)]
    
    if options.get('format')==False:
        return date_list
    else:
        return [x.strftime(options.get('format')) for x in date_list]


# In[289]:


def __date_plus_days(d,n):
    if(isinstance(d,dt.datetime)):
        return d+dt.timedelta(days=n)
    elif(isinstance(d,str)):
        return (dt.datetime.strptime(d,'%Y-%m-%d')+dt.timedelta(days=n)).strftime('%Y-%m-%d')
    
def date_plus_days(n):
    return lambda d: __date_plus_days(d,n)


# In[ ]:


def to_iso_date(s):
    return dt.datetime.fromisoformat(s)

def from_iso_date(d,time=False):
    if time=='iso':
        return d.isoformat()
    elif time=='HMS':
        return d.strftime('%Y-%m-%dT%H%M%S')
    elif time==True:
        return d.strftime('%Y-%m-%dT%H%M%Sf')
    elif time==False:
        return d.strftime('%Y-%m-%d')
    
    dt.datetime.isoformat(x)
    
# deprecated
# iso_date = to_iso_date


# In[290]:


# convert iso date to numerical rep for matplotlib
datestr2num = lambda s: mp.dates.date2num(to_iso_datetime(s))


# In[291]:


def deltatime_msec(t,s):
    try:
        return (t-s).total_seconds()*1000
    except:
        return None


# In[292]:


def millis_interval(start,end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 *60 *1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


# In[293]:


def deltatime_sec(t,s):
    try:
        return (t-s).total_seconds()
    except:
        return None


# In[294]:


def to_weekday(d,names=True):
    assert(isinstance(d,(str,dt.datetime)))
    if isinstance(d,str):
        d = to_iso_date(d)
    if names:
        return d.strftime('%A')
    else:
        return d.weekday()


# In[298]:


def date_hm_delta(date):
    
    def delta_fun(dt_date):
        return lambda h,m: dt.datetime.isoformat(dt_date + dt.timedelta(hours=h) + dt.timedelta(minutes=m))
    
    if isinstance(date,dt.datetime):
        return delta_fun(date)
    elif isinstance(date,str):
        return delta_fun(to_iso_datetime(date))


# In[ ]:


# MISC


# In[300]:


def rotate_left(l,n):
    if isinstance(n,int):
        return l[n:] + l[:n]
    if isinstance(n,(tuple,list,range)):
        return [rotate_left(l,x) for x in n]
    
def rotate_right(l,n):
    rotate_left(l,-n)


# In[301]:


def digits(x):
    return [int(i) for i in str(x)]


# In[302]:


def ensure_dir(file_path,type=dir):
    if type=='dir':
        directory = file_path
    elif type=='file':
        directory = os.path.dirname(file_path)
    else:
        directory = file_path
    if not os.path.exists(directory):
        os.makedirs(directory)


# In[303]:


safe_sum = lambda x: rcomp(values,sum)(x) if isa(dict)(x) else 0

