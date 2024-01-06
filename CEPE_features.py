import numpy as np
import scipy.sparse as sp
import json



def make_node_features(ce, xce, pe, xpe):

    n = 2

    #init node features array 
    features = []   
    
    #PE features
    f=[]
    f.append(xpe[0])                    #Router type
    f.append(xpe[3])                    #VRF configured ?  is it connected to a CE ?  
    f.append(xpe[4])                    #VRF RD

    f.append(xpe[11])                    #Int LAN IP address  
    f.append(xpe[12])                    #Int LAN IP mask

    f.append(xpe[13])                    #BGP as

    f.append(xpe[14+(15*ce)])             #Int IP address
    f.append(xpe[15+(15*ce)])             #Int IP mask   
    f.append(xpe[16+(15*ce)])            #Int assigned to VRF ?  is it connected to a CE ?  
    f.append(xpe[17+(15*ce)])            #Int VRF RD

    f.append(xpe[18+(15*ce)])            #BGP Neighbor IP address
    f.append(xpe[19+(15*ce)])            #BGP Neighbor remote-as 
    f.append(xpe[20+(15*ce)])            #BGP IPv4
    f.append(xpe[21+(15*ce)])            #BGP Neighbor VRF RD
    f.append(xpe[22+(15*ce)])            #BGP Network IP address
    f.append(xpe[23+(15*ce)])            #BGP Network IP mask
    f.append(xpe[24+(15*ce)])            #BGP Redistribute static routes

    f.append(xpe[25+(15*ce)])            #Static route subnet IP address 
    f.append(xpe[26+(15*ce)])            #Static route subnet IP mask 
    f.append(xpe[27+(15*ce)])            #Static route next-hop IP address 
    f.append(xpe[28+(15*ce)])            #Static route VRF RD
    
    features.append(f)
    
    #CE features
    f=[]
    f.append(xce[0])                    #Router type
    f.append(xce[3])                    #VRF configured ?  is it connected to a CE ?  
    f.append(xce[4])                    #VRF RD

    f.append(xce[11])                    #Int LAN IP address  
    f.append(xce[12])                    #Int LAN IP mask

    f.append(xce[13])                    #BGP as

    f.append(xce[14])                    #Int IP address
    f.append(xce[15])                    #Int IP mask   
    f.append(xce[16])                   #Int assigned to VRF ?  is it connected to a CE ?  
    f.append(xce[17])                   #Int VRF RD

    f.append(xce[18])                   #BGP Neighbor IP address
    f.append(xce[19])                   #BGP Neighbor remote-as 
    f.append(xce[20])                   #BGP IPv4
    f.append(xce[21])                   #BGP Neighbor VRF RD 
    f.append(xce[22])                   #BGP Network IP address
    f.append(xce[23])                   #BGP Network IP mask
    f.append(xce[24])                   #BGP Redistribute static routes

    f.append(xce[25])                   #Static route subnet IP address 
    f.append(xce[26])                   #Static route subnet IP mask 
    f.append(xce[27])                   #Static route next-hope IP address 
    f.append(xce[28])                   #Static route VRF RD

    features.append(f)
        
        
    return np.array(features)


def json_to_array(features):
    x=[] 
    for node in ["PE", "CE"]: 
        x_node = [] 
        x_node.append(features[node]["ROUTER_TYPE"])
        x_node.append(features[node]["VRF"])
        x_node.append(features[node]["VRF_RD"])

        x_node.append(features[node]["LAN_INT_IP"])
        x_node.append(features[node]["LAN_INT_MASK"])

        x_node.append(features[node]["BGP_AS"])

        x_node.append(features[node]["INT_IP"])
        x_node.append(features[node]["INT_MASk"])
        x_node.append(features[node]["INT_VRF"])
        x_node.append(features[node]["INT_VRF_RD"])

        x_node.append(features[node]["BGP_NEIGHBOR_IP"])
        x_node.append(features[node]["BGP_NEIGHBOR_MASK"])
        x_node.append(features[node]["BGP_IPV4"])
        x_node.append(features[node]["BGP_NEIGHBOR_VRF_RD"])
        x_node.append(features[node]["BGP_NETWORK_IP"])
        x_node.append(features[node]["BGP_NETWORK_MASK"])
        x_node.append(features[node]["BGP_REDISTRIBUTE_STATIC"])

        x_node.append(features[node]["STATIC_SUBNET_IP"])
        x_node.append(features[node]["STATIC_SUBNET_MASK"])
        x_node.append(features[node]["STATIC_NEXTHOP"])
        x_node.append(features[node]["STATIC_VRF_RD"])

        x.append(x_node)

    return np.array(x)

######## Faults generation ####### 

#   -   1 : VRF Not configured on PE node,
def vrf_pe_node_fault(node_features):
    '''
    VRF Not configured on PE node, The features VRF and VRF_RD must be equale to 0.

    '''
    # VRF feature on PE node
    node_features[0][1] = 0

    # VRF RD feature on PE node
    node_features[0][2] = 0 
    
    return node_features

#   -   2 : VRF RD misconfigured on PE node,
def vrf_rd_pe_node_fault(node_features):
    '''
    VRF RD misconfigured on PE node, The feature VRF_RD must be updated randomly.

    '''
    # VRF RD feature on PE node
    correct_vrf_rd = node_features[0][2]
    while  node_features[0][2] == correct_vrf_rd :
        node_features[0][2] = np.random.randint(1, 999)
    
    return node_features

#   -   3 : BGP not configured or misconfigured on PE node,
def bgp_pe_fault(node_features):
    '''
    BGP not configured or misconfigured on PE node, the feature BGP_AS on PE node must be updated to 0 or to an incorrect value. 
    '''

    # BGP AS feature on PE node
    correct_bgp_as = node_features[0][5]  
    while  node_features[0][5] == correct_bgp_as :
        node_features[0][5] = np.random.randint(0, 99999)
    
    return node_features

#   -   4 : BGP not configured or misconfigured on CE node,
def bgp_ce_fault(node_features):
    '''
    BGP not configured or misconfigured on CE node, the feature BGP_AS on CE node must be updated to 0 or to an incorrect value. 
    '''

    # BGP AS feature on PE node
    correct_bgp_as = node_features[1][5]  
    while  node_features[1][5] == correct_bgp_as :
        node_features[1][5] = np.random.randint(0, 99999)
    
    return node_features

#   -   5 : Interface not assigned to a VRF on PE node,
def vrf_pe_int_fault(node_features):
    '''
    Interface not assigned to a VRF on PE node, The features Int_VRF_assigned and Int_VRF_RD must be equal to 0 on PE node features.

    '''
    # Int_VRF_assigned feature on PE node
    node_features[0][8] = 0

    # Int_VRF_RD feature on PE node
    node_features[0][9] = 0  
    
    return node_features

#   -   6 : VRF RD misconfigured on PE node,
def vrf_rd_pe_int_fault(node_features):
    '''
    Interface VRF RD misconfigured on PE node, The feature Int_VRF_RD must be updated randomly.

    '''

    # Int_VRF_RD feature on PE interface
    correct_vrf_rd = node_features[0][9]
    while  node_features[0][9] == correct_vrf_rd :
        node_features[0][9] = np.random.randint(1, 999)
    
    return node_features

#   -   7 : BGP neighbor remote-as misconfigured on PE node,
def bgp_remote_as_pe_neigh_fault(node_features):
    '''
    BGP neighbor remote-as misconfigured on PE node, the feature BGP_Neighbor_remote-AS on PE node must be updated to an incorrect value. 
    '''

    # BGP neighbor remote-as feature on PE node
    correct_bgp_remote_as = node_features[0][11]  
    while  node_features[0][11] == correct_bgp_remote_as :
        node_features[0][11] = np.random.randint(1, 99999)
    
    return node_features

#   -   8 : BGP neighbor remote-as misconfigured on CE node,
def bgp_remote_as_ce_neigh_fault(node_features):
    '''
    BGP neighbor remote-as misconfigured on CE node, the feature BGP_Neighbor_remote-AS on CE node must be updated to an incorrect value. 
    '''

    # BGP neighbor remote-as feature on CE node
    correct_bgp_remote_as = node_features[1][11]  
    while  node_features[1][11] == correct_bgp_remote_as :
        node_features[1][11] = np.random.randint(1, 99999)
    
    return node_features

#   -   9 : BGP VRF RD not configured or misconfigured on PE node,
def bgp_vrf_rd_pe_fault(node_features):
    '''
    BGP VRF RD misconfigured on PE node, The feature BGP_VRF_RD must be updated randomly.

    '''

    # BGP_VRF_RD feature on PE node
    correct_bgp_vrf_rd = node_features[0][13]
    while  node_features[0][13] == correct_bgp_vrf_rd :
        node_features[0][13] = np.random.randint(1, 999)
    
    return node_features

#   -   10 : IPv4 forwarding not activated for BGP neighbor on PE node,
def bgp_ipv4_pe_neigh_fault(node_features):
    '''
    IPv4 forwarding not activated for BGP neighbor on PE node, The feature BGP_Neighbor_IPv4 must be equal to 0.

    '''

    # BGP_Neighbor_IPv4 feature on PE node
    node_features[0][12] = 0
    
    return node_features

#   -   11 : IPv4 forwarding not activated for BGP neighbor on CE node,
def bgp_ipv4_ce_neigh_fault(node_features):
    '''
    IPv4 forwarding not activated for BGP neighbor on CE node, The feature BGP_Neighbor_IPv4 must be equal to 0.

    '''

    # BGP_Neighbor_IPv4 feature on CE node
    node_features[1][12] = 0
    
    return node_features

#   -   12 : BGP neighbor IP address misconfigured on PE node,
def bgp_ip_addr_pe_neigh_fault(node_features):
    '''
    BGP neighbor IP address misconfigured on PE node,, The feature BGP_Neighbor_IP_address must be updated randomly.

    '''

    # BGP_Neighbor_IP_address feature on PE node
    correct_bgp_neighbor_ip_address = node_features[0][10]
    while  node_features[0][10] == correct_bgp_neighbor_ip_address :
        node_features[0][10] = np.random.randint(1, 99999)
    
    return node_features

#   -   13 : BGP neighbor IP address misconfigured on CE node,
def bgp_ip_addr_ce_neigh_fault(node_features):
    '''
    BGP neighbor IP address misconfigured on CE node,, The feature BGP_Neighbor_IP_address must be updated randomly.

    '''

    # BGP_Neighbor_IP_address feature on CE node
    correct_bgp_neighbor_ip_address = node_features[1][10]
    while  node_features[1][10] == correct_bgp_neighbor_ip_address :
        node_features[1][10] = np.random.randint(1, 99999)
    
    return node_features

#   -   14 : IP address misconfigured on PE interface,
def ip_addr_pe_int_fault(node_features):
    '''
    IP address misconfigured on PE interface, The feature Int_IP_address must be updated randomly.

    '''

    # Int_IP_address feature on PE node
    correct_int_ip_address = node_features[0][6]
    while  node_features[0][6] == correct_int_ip_address :
        node_features[0][6] = np.random.randint(1, 99999)
    
    return node_features

#   -   15 : IP address misconfigured on CE interface,
def ip_addr_ce_int_fault(node_features):
    '''
    IP address misconfigured on CE interface, The feature Int_IP_address must be updated randomly.

    '''

    # Int_IP_address feature on CE node
    correct_int_ip_address = node_features[1][6]
    while  node_features[1][6] == correct_int_ip_address :
        node_features[1][6] = np.random.randint(1, 99999)
    
    return node_features

#   -   16 : BGP Network IP address misconfigured on CE node,
def bgp_net_ip_addr_pe_fault(node_features):
    '''
    BGP network IP address misconfigured on CE node, The feature BGP_Network_IP_address must be updated randomly.

    '''

    # BGP_Network_IP_address feature on CE node
    correct_bgp_network_ip_address = node_features[1][14]
    while  node_features[1][14] == correct_bgp_network_ip_address :
        node_features[1][14] = np.random.randint(0, 99999)
    
    return node_features

#   -   17 : BGP Static routes Redistribution not activated on PE node,
def bgp_static_redist_pe_fault(node_features):
    '''
    BGP static routes Redistribution not activated on PE node, The feature BGP_static_routes_Redistribution must be equal to 0.

    '''

    # BGP_static_routes_Redistribution feature on PE node
    node_features[0][16] = 0
    
    return node_features

#   -   18 : Static route destination IP address misconfigured on PE node,
def static_dest_ip_addr_pe_fault(node_features):
    '''
    Static route destination IP address misconfigured on PE node, The feature Static_dest_IP_address must be updated randomly.

    '''

    # Static_dest_IP_address feature on PE node
    correct_static_dest_ip_address = node_features[0][17]
    while  node_features[0][17] == correct_static_dest_ip_address :
        node_features[0][17] = np.random.randint(0, 99999)
    
    return node_features

#   -   19 : Static route destination IP address misconfigured on CE node,
def static_dest_ip_addr_ce_fault(node_features):
    '''
    Static route destination IP address misconfigured on CE node, The feature Static_dest_IP_address must be updated randomly.

    '''

    # Static_dest_IP_address feature on CE node
    correct_static_dest_ip_address = node_features[1][17]
    while  node_features[1][17] == correct_static_dest_ip_address :
        node_features[1][17] = np.random.randint(0, 99999)
    
    return node_features

#   -   20 : Static route next-hop misconfigured on PE node,
def static_next_hop_pe_fault(node_features):
    '''
    Static route next-hop IP address misconfigured on PE node, The feature Static_next_hop must be updated randomly.

    '''

    # Static_next_hop feature on PE node
    correct_static_next_hop = node_features[0][19]
    while  node_features[0][19] == correct_static_next_hop :
        node_features[0][19] = np.random.randint(0, 99999)
    
    return node_features

#   -   21 : Static route next-hope misconfigured on CE node,
def static_next_hop_ce_fault(node_features):
    '''
    Static route next-hop IP address misconfigured on CE node, The feature Static_next_hop must be updated randomly.

    '''

    # Static_next_hop feature on CE node
    correct_static_next_hop = node_features[1][19]
    while  node_features[1][19] == correct_static_next_hop :
        node_features[1][19] = np.random.randint(0, 99999)
    
    return node_features

#   -   22 : Static route VRF RD misconfigured on PE node,
def static_vrf_rd_pe_fault(node_features):
    '''
    Static route VRF RD misconfigured on PE node, The feature Static_vrf_rd must be updated randomly.

    '''

    # Static_vrf_rd feature on PE node
    correct_static_vrf_rd = node_features[0][20]
    while  node_features[0][20] == correct_static_vrf_rd :
        node_features[0][20] = np.random.randint(1, 999)
    
    return node_features

#   -   23 : Interface LAN IP address misconfigured on CE node,
def int_lan_ip_addr_ce_fault(node_features):
    '''
    Interface LAN IP address misconfigured on CE interface, The feature Int_lan_IP_address must be updated randomly.

    '''

    # Int_lan_IP_address feature on CE node
    correct_int_lan_ip_address = node_features[1][3]
    while  node_features[1][3] == correct_int_lan_ip_address :
        node_features[1][3] = np.random.randint(1, 99999)
    
    return node_features



#   -   24 : IP mask misconfigured on PE interface,
def ip_mask_pe_int_fault(node_features):
    '''
    IP mask misconfigured on PE interface, The feature Int_IP_mask must be updated randomly.

    '''

    # Int_IP_mask feature on PE node
    correct_int_ip_mask = node_features[0][7]
    while  node_features[0][7] == correct_int_ip_mask :
        node_features[0][7] = np.random.randint(0, 99999)
    
    return node_features

#   -   25 : IP mask misconfigured on CE interface,
def ip_mask_ce_int_fault(node_features):
    '''
    IP mask misconfigured on CE interface, The feature Int_IP_mask must be updated randomly.

    '''

    # Int_IP_mask feature on CE node
    correct_int_ip_mask = node_features[1][7]
    while  node_features[1][7] == correct_int_ip_mask :
        node_features[1][7] = np.random.randint(0, 99999)
    
    return node_features

#   -   26 : BGP Network IP mask misconfigured on CE node,
def bgp_net_ip_mask_pe_fault(node_features):
    '''
    BGP network IP mask misconfigured on CE node, The feature BGP_Network_IP_mask must be updated randomly.

    '''

    # BGP_Network_IP_mask feature on CE node
    correct_bgp_network_ip_mask = node_features[1][15]
    while  node_features[1][15] == correct_bgp_network_ip_mask :
        node_features[1][15] = np.random.randint(0, 99999)
    
    return node_features

#   -   27 : Static route destination Mask misconfigured on PE node,
def static_dest_ip_mask_pe_fault(node_features):
    '''
    Static route destination IP mask misconfigured on PE node, The feature Static_dest_IP_mask must be updated randomly.

    '''

    # Static_dest_IP_mask feature on PE node
    correct_static_dest_ip_mask = node_features[0][18]
    while  node_features[0][18] == correct_static_dest_ip_mask :
        node_features[0][18] = np.random.randint(0, 99999)
    
    return node_features

#   -   28 : Static route subnet Mask misconfigured on CE node,
def static_dest_ip_mask_ce_fault(node_features):
    '''
    Static route destination IP mask misconfigured on CE node, The feature Static_dest_IP_mask must be updated randomly.

    '''

    # Static_dest_IP_mask feature on CE node
    correct_static_dest_ip_mask = node_features[1][18]
    while  node_features[1][18] == correct_static_dest_ip_mask :
        node_features[1][18] = np.random.randint(0, 99999)
    
    return node_features

#   -   29 : Interface LAN IP Mask misconfigured on CE node.
def int_lan_ip_mask_ce_fault(node_features):
    '''
    Interface LAN IP mask misconfigured on CE interface, The feature Int_lan_IP_mask must be updated randomly.

    '''

    # Int_lan_IP_mask feature on CE node
    correct_int_lan_ip_mask = node_features[1][4]
    while  node_features[1][4] == correct_int_lan_ip_mask :
        node_features[1][4] = np.random.randint(0, 99999)
    
    return node_features


def gen_fault(fault, node_features):

    '''

    # Faults list :

    #   -   0 : No fault,

    #   -   1 : VRF Not configured on PE node, 

    #   -   2 : VRF RD misconfigured on PE node,

    #   -   3 : BGP not configured or misconfigured on PE node,

    #   -   4 : BGP not configured or misconfigured on CE node,
    
    #   -   5 : Interface not assigned to a VRF on PE node,
    
    #   -   6 : VRF RD misconfigured on PE node,
    
    #   -   7 : BGP neighbor remote-as misconfigured on PE node,
    
    #   -   8 : BGP neighbor remote-as misconfigured on CE node,
    
    #   -   9 : BGP VRF RD not configured or misconfigured on PE node,

    #   -   10 : IPv4 forwarding not activated for BGP neighbor on PE node,

    #   -   11 : IPv4 forwarding not activated for BGP neighbor on CE node,
    
    #   -   12 : BGP neighbor IP address misconfigured on PE node,

    #   -   13 : BGP neighbor IP address misconfigured on CE node,
    
    #   -   14 : IP address misconfigured on PE interface,

    #   -   15 : IP address misconfigured on CE interface,

    #   -   16 : IP mask misconfigured on PE interface,

    #   -   17 : BGP Static routes Redistribution not activated on PE node,

    #   -   18 : Static route destination IP address misconfigured on PE node,

    #   -   19 : Static route destination IP address misconfigured on CE node,

    #   -   20 : Static route next-hope misconfigured on PE node,

    #   -   21 : Static route next-hope misconfigured on CE node,

    #   -   22 : Static route VRF RD misconfigured on PE node,

    #   -   23 : Interface LAN IP address misconfigured on CE node,

    #   -   24 : IP mask misconfigured on CE interface,

    #   -   25 : BGP Network IP address misconfigured on PE node,

    #   -   26 : BGP Network IP mask misconfigured on PE node,

    #   -   27 : Static route destination Mask misconfigured on PE node,

    #   -   28 : Static route destination Mask misconfigured on CE node,

    #   -   29 : Interface LAN IP Mask misconfigured on CE node.






    '''

    switcher = {
        1: vrf_pe_node_fault,
        2: vrf_rd_pe_node_fault,
        3: bgp_pe_fault,
        4: bgp_ce_fault,
        5: vrf_pe_int_fault,
        6: vrf_rd_pe_int_fault,
        7: bgp_remote_as_pe_neigh_fault,
        8: bgp_remote_as_ce_neigh_fault,
        9: bgp_vrf_rd_pe_fault,
        10: bgp_ipv4_pe_neigh_fault,
        11: bgp_ipv4_ce_neigh_fault,
        12: bgp_ip_addr_pe_neigh_fault,
        13: bgp_ip_addr_ce_neigh_fault,
        14: ip_addr_pe_int_fault,
        15: ip_addr_ce_int_fault,
        16: bgp_net_ip_addr_pe_fault,
        17: bgp_static_redist_pe_fault,
        18: static_dest_ip_addr_pe_fault,
        19: static_dest_ip_addr_ce_fault,
        20: static_next_hop_pe_fault,
        21: static_next_hop_ce_fault,
        22: static_vrf_rd_pe_fault,
        23: int_lan_ip_addr_ce_fault,
        24: ip_mask_pe_int_fault,
        25: ip_mask_ce_int_fault,
        26: bgp_net_ip_mask_pe_fault,
        27: static_dest_ip_mask_pe_fault,
        28: static_dest_ip_mask_ce_fault,
        29: int_lan_ip_mask_ce_fault,
        } 

    
    fault = switcher.get(fault)
    node_features = fault(node_features)


    return node_features

