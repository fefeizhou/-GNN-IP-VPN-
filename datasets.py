import numpy as np
import scipy.sparse as sp
import json
import random

from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCSConv, GlobalAvgPool
from spektral.layers.pooling import TopKPool
from spektral.transforms.normalize_adj import NormalizeAdj

import topology
import features
import CEPE_features
import PEs_features 


import time


################################################################################
# LOAD DATA
################################################################################
class L3VPN_dataset(Dataset):
    """
    The dataset contain BGP/MPLS L3 VPN configuration graphs, each graph represent a Customer configuration.
    """
    
    def __init__(self, n_samples=10, n_pe=5, n_ce_min=5, n_ce_max=10, VPN_Type=2, cust_routing=2, **kwargs):
        self.n_samples = n_samples
        self.n_pe = n_pe
        self.n_ce_min = n_ce_min
        self.n_ce_max = n_ce_max
        self.VPN_Type = VPN_Type # 0:Full-mesh; 1:Hub&Spoke; 2:Both 
        self.cust_routing = cust_routing
        super().__init__(**kwargs)
    
    def read(self):
        def make_graph(rd, G_Hub_Spoke):
        
            n_ce = random.randint(self.n_ce_min, self.n_ce_max)
            n = self.n_pe + n_ce
            
            # Edges
            a = topology.make_rand_topology(n_pe=self.n_pe, n_ce=n_ce)
            
            # Node features
            # G_Hub_Spoke = random.randint(0,1) * self.Hub_Spoke # 0 for full-mesh; 1 for Hub&Spoke;
            x = features.make_node_features(n_pe=self.n_pe, n_ce=n_ce, a=a, rd=rd, Hub_Spoke=G_Hub_Spoke, cust_routing=self.cust_routing) 
            
            return Graph(x=x, a=a)

        # We must return a list of Graph objects
        if self.VPN_Type == 0:
            Hub_Spoke = np.zeros((self.n_samples,), dtype=int)
        elif self.VPN_Type == 1:
            Hub_Spoke = np.ones((self.n_samples,), dtype=int)
        else:
            Hub_Spoke = np.random.randint(2, size=self.n_samples)
            
        return [make_graph(i+1, Hub_Spoke[i]) for i in range(self.n_samples)]


################################################################################
# CE-PE routing dataset without Edge features
################################################################################
class CEPE_dataset(Dataset):
    """
    A dataset of graphs of CE-PE routing configuration in BGP/MPLS L3 VPN architecture.
    
    The task is to classify each graph with the fault that occurs on it.

    Each graph have two nodes : a CE node  and a PE node.
    
    The output is a vector of n CE-PE routing configuration faults + 1 class if the CE-PE routing configuration is valid.
    """

    def __init__(self, L3VPN_dataset, n_faults=29, **kwargs):
        self.L3VPN_dataset = L3VPN_dataset
        self.n_faults = n_faults
        
        super().__init__(**kwargs)
    

    def read(self):
        def make_graph(ce, xce, pe, xpe, fault):
            n = 2
            
            # Edges
            a = [[0, 1],
                 [1, 0]] 
            a = sp.csr_matrix(a)

            # Node features
            x = CEPE_features.make_node_features(ce, xce, pe, xpe) 

                      
            # generate the fault
            if fault!=0 :
                x = CEPE_features.gen_fault(fault=fault, node_features=x)     
            

            # Labels
            y = np.zeros((self.n_faults+1,))
            y[fault] = 1

            return Graph(x=x, a=a, y=y)


        CEPE_graphs = [] 
        for L3VPN_graph in self.L3VPN_dataset:
            i = 0
            while L3VPN_graph.x[i][0] == 1:
                i=i+1 
            n_pe = i
            n_ce = L3VPN_graph.n_nodes - n_pe
            
            bgp_faults = [4, 7, 8, 10, 11, 12, 13, 16, 26]
            static_faults = [17, 18, 19, 20, 21, 22, 27, 28]
        
            all_faults =  np.arange(self.n_faults+1)

            faults = np.zeros((n_ce,),dtype=int)
            for i in range(n_ce):       
                possible_faults = []             
                if L3VPN_graph.x[n_pe+i][13] == 0:  # BGP AS
                    for f in all_faults:
                        if f not in bgp_faults:
                            possible_faults.append(f)
                    # possible_faults = [ f * (f not in bgp_faults) for f in all_faults] 
                    # possible_faults = np.delete(all_faults, bgp_faults)
                else:
                    for f in all_faults:
                        if f not in static_faults:
                            possible_faults.append(f)
                    # possible_faults = [ f * (f not in static_faults) for f in all_faults]
                    # possible_faults = np.delete(all_faults, static_faults)
                faults[i] = np.random.choice(possible_faults)    

            #faults = np.random.randint(0, self.n_faults+1, n_ce)
            
            #print(faults)
            indices = np.transpose(L3VPN_graph.a.nonzero())
            k=0
            for i, j in indices:
                if L3VPN_graph.x[i][0]==0:
                    CEPE_graphs.append(make_graph(ce=k, xce=L3VPN_graph.x[i], pe=j, xpe=L3VPN_graph.x[j], fault=faults[k]))
                    k=k+1

        # We must return a list of Graph objects
        return CEPE_graphs




################################################################################
# PEs dataset 
################################################################################
class PEs_dataset(Dataset):
    """
    The dataset contain configurations of PE nodes in a BGP/MPLS L3 VPN architecture.

    The task is to predict for each node if it contain a configuration fault or not.
    
    The output for each node is a vector of n configuration faults + 1 class if the node is valid.
    
    """
    def __init__(self, L3VPN_dataset, **kwargs):  #, n_faults=8 
        #self.n_samples = L3VPN_dataset.n_graphs
        self.L3VPN_dataset = L3VPN_dataset
        # self.n_faults = n_faults
        
        
        super().__init__(**kwargs) 

    def read(self):
        def make_graph(L3VPN_graph):

            i = 0
            while L3VPN_graph.x[i][0] == 1:
                i=i+1 
            n = i

            Hub_Spoke = L3VPN_graph.x[0][1]

           
            
            
            # Edges
            a = np.ones((n,n)) 
            for i in range(n):
                a[i][i]=0
            a = sp.csr_matrix(a)

            
            # Node features
            n_features = 8 
            x = np.zeros((n,n_features), dtype=int)
            
            # faults = np.random.randint(0, self.n_faults+1, size=n)
            faults = np.zeros((n,),dtype=int)

            for i in range(n):
                Hub = L3VPN_graph.x[i][2]
                configured = L3VPN_graph.x[i][3]
                
                x[i][0]=L3VPN_graph.x[i][3]                #VRF configured ?
                x[i][1]=L3VPN_graph.x[i][4]                #VRF RD
                x[i][2]=L3VPN_graph.x[i][5]                #VRF RT Import
                x[i][3]=L3VPN_graph.x[i][6]                #VRF RT Export
                
                x[i][4]=L3VPN_graph.x[i][7]                #Import if prefix equal to the Hub CE subnet (Enabled on Spoke PEs) 
                x[i][5]=L3VPN_graph.x[i][8]                #Import if extcommunity RT (configured on Spoke PEs)
                x[i][6]=L3VPN_graph.x[i][9]                #Export if prefix equal to the connected CE subnet (Enabled on Spoke PEs)  
                x[i][7]=L3VPN_graph.x[i][10]               #Export with extcommunity RT (configured on Spoke PEs)
                #x[i][8]=not Hub_Spoke or Hub

                # generate the fault
                if not configured:
                    possible_faults = [0]
                elif not Hub_Spoke or Hub:
                    possible_faults = [0, 1, 2, 3, 4] 
                else:
                    possible_faults = [0, 1, 2, 3, 5, 6, 7, 8] 
                
                faults[i] = np.random.choice(possible_faults)
                #print(i, x[i], possible_faults, faults[i])

                
                if faults[i]!=0 :
                    x[i]  = PEs_features.gen_fault(fault=faults[i], node_features=x[i], Hub_Spoke=Hub_Spoke, Hub=Hub)
                

                             
            # Edge features 
            e = []#  np.zeros((n*n, self.g['n_vrf']))
            indices = np.transpose(a.nonzero())
            for i, j in indices:
                edge_features = [0] # ,0]
                if (L3VPN_graph.x[i][3]==L3VPN_graph.x[j][3]==1)  and ((not Hub_Spoke) or L3VPN_graph.x[i][2] or L3VPN_graph.x[j][2]): 
                    edge_features[0]  = 1
                # if ((not Hub_Spoke) or L3VPN_graph.x[i][2] or L3VPN_graph.x[j][2]): 
                #    edge_features[1]  = 1 
                e.append(edge_features)
            e = np.array(e)
                         

            # Labels
            n_faults = 8 # 4 + 4 * Hub_Spoke
            y = np.zeros((n,n_faults+1))
            for i in range(n):
                y[i][faults[i]] = 1
            
             
            
            return Graph(x=x, a=a, e=e, y=y)

        # We must return a list of Graph objects
        return [make_graph(self.L3VPN_dataset[i]) for i in range(self.L3VPN_dataset.n_graphs)]
