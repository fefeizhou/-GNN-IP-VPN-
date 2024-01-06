import numpy as np
import scipy.sparse as sp
import json


def json_to_array(features, n_nodes):
    
    x=np.zeros((n_nodes,8))
    pes = [[] for i in range(n_nodes)]
    for f in features: 
        id = f["ID"]
        x[id][0]=f["VRF"]
        x[id][1]=f["VRF_RD"]
        x[id][2]=f["VRF_RT_IMPORT"]
        x[id][3]=f["VRF_RT_EXPORT"]

        x[id][4]=f["IMPORT_RPLC_IF_SUBNET"]
        x[id][5]=f["IMPORT_RPLC_IF_EXTCOMMUNITY"]
        x[id][6]=f["EXPORT_RPLC_IF_SUBNET"]
        x[id][7]=f["EXPORT_RPLC_APPLY_EXTCOMMUNITY_RT"]

        pes[id]= f["PEs"] 


    a = np.ones((n_nodes,n_nodes)) 
    for i in range(n_nodes):
        a[i][i]=0
    a = sp.csr_matrix(a)

    
    e = []
    indices = np.transpose(a.nonzero())
    for i, j in indices:
        edge_features = [0]
        if j in pes[i] : 
            edge_features = [1] 
        e.append(edge_features)
    e = np.array(e)

    return x, a, e




######## Faults generation ####### 

#   -   1 : VRF Not configured on PE node,
def vrf_fault(node_features):
    '''
    VRF Not configured on PE node, all features must be equale to 0.

    '''
    for i in range(len(node_features)):
        node_features[i] = 0
    
    return node_features

#   -   2 : VRF RD misonfigured on PE node,
def rd_fault(node_features):
    '''
    VRF RD misconfigured on PE node, RD feature must be equale to 0 or another faulty value.

    '''
    # RD feature 
    #node_features[1]= 0
    correct_rd = node_features[1]
    while  node_features[1] == correct_rd :
        node_features[1] = np.random.randint(0, 999)
    
    return node_features

#   -   3 : VRF RT Import misonfigured on PE node,
def rt_import_fault(node_features):
    '''
    VRF RT Import misconfigured on PE node, RT Import feature must be equale to 0 or another faulty value.

    '''
    # RT Import feature 
    #node_features[2]= 0
    correct_rt_import = node_features[2]
    while  node_features[2] == correct_rt_import :
        node_features[2] = np.random.randint(0, 999)
    
    return node_features

#   -   4 : VRF RT Export misonfigured on PE node,
def rt_export_fault(node_features):
    '''
    VRF RT Export misconfigured on PE node, RT Export feature must be equale to 0 or another faulty value.

    '''
    # RT Export feature 
    #node_features[3]= 0
    correct_rt_export = node_features[3]
    while  node_features[3] == correct_rt_export :
        node_features[3] = np.random.randint(0, 999)
    
    return node_features

#   -   5 :  Import RT Policy if match subnet not configured,
def import_rtp_subnet_fault(node_features):
    '''
     Import RT Policy if match subnet not configured, Import RT Policy if match subnet feature must be equale to 0.

    '''
    # Import RT Policy if match subnet feature 
    node_features[4]= 0
    
    return node_features

#   -   6 :  Import RT Policy if match RT not configured or misconfigured,
def import_rtp_rt_fault(node_features):
    '''
     Import RT Policy if match RT not configured, Import RT Policy if match RT feature must be equale to 0 or another faulty value.

    '''
    # Import RT Policy if match RT feature 
    #node_features[5]= 0
    correct_rt_import = node_features[5]
    while  node_features[5] == correct_rt_import :
        node_features[5] = np.random.randint(0, 999)
    
    return node_features

#   -   7 :  Export RT Policy if match subnet not configured,
def export_rtp_subnet_fault(node_features):
    '''
     Export RT Policy if match subnet not configured, Export RT Policy if match subnet feature must be equale to 0.

    '''
    # Export RT Policy if match subnet feature 
    node_features[6]= 0
    
    return node_features

#   -   8 :  Export RT Policy if apply RT extcommunity not configured or misconfigured,
def export_rtp_rt_fault(node_features):
    '''
    Export RT Policy if apply RT extcommunity not configured, Export RT Policy if apply RT extcommunity feature must be equale to 0.

    '''
    # Export RT Policy if apply RT extcommunity feature 
    #node_features[7]= 0
    correct_rt_export = node_features[7]
    while  node_features[7] == correct_rt_export :
        node_features[7] = np.random.randint(0, 999)
    
    return node_features


def gen_fault(fault, node_features, Hub_Spoke, Hub):

    '''
    # Faults list :

    #   -   0 : No fault,

    #   -   1 : VRF Not configured on PE node, 

    #   -   2 : VRF RD misconfigured on PE node,

    #   -   3 : VRF RT Import misconfigured on PE node,

    #   -   4 : VRF RT Export misconfigured on PE node,,
    
    #   -   5 : Import RT Policy if match subnet not configured,
    
    #   -   6 : Import RT Policy if match RT import not configured,
    
    #   -   7 : Export RT Policy if match subnet not configured,
    
    #   -   8 : Export RT Policy apply RT extcommunity not configured,

    '''

    switcher = {
        1: vrf_fault,
        2: rd_fault,
        3: rt_import_fault,
        4: rt_export_fault,
        5: import_rtp_subnet_fault,
        6: import_rtp_rt_fault,
        7: export_rtp_subnet_fault,
        8: export_rtp_rt_fault,

        } 

    
    fault = switcher.get(fault)
    node_features = fault(node_features) # Hub_Spoke, Hub 


    return node_features

