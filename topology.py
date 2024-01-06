import numpy as np
import scipy.sparse as sp
import random


def make_rand_topology(n_pe, n_ce):
    

    ## Backbone edges, the BB network should be a full mesh graph 
    a_bb = np.ones((n_pe, n_pe), dtype=int)
    for i in range(n_pe):
        a_bb[i][i] = 0  


    ## CE-PE edges
    pes = np.random.randint(0, n_pe, size=n_ce)

    a_tmp = np.zeros((n_pe, n_ce), dtype=int)
    a = np.append(a_bb, a_tmp, axis=1)
    a_tmp = np.zeros((n_ce, n_pe+n_ce), dtype=int)
    a = np.append(a, a_tmp, axis=0)
    del a_tmp

    for i in range(n_ce):
        ce = n_pe + i
        pe = pes[i] 
        a[ce][pe] = 1
        a[pe][ce] = 1 
    
    a = sp.csr_matrix(a)

    return a



    
