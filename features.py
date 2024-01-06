import numpy as np
import random
import scipy.sparse as sp
import json




def make_node_features(n_pe, n_ce, a, rd, Hub_Spoke=0, cust_routing=2):
    
    n = n_pe + n_ce
    features = []
    n_edges = np.diff(a.indptr) 
    #print(n_edges)
    hub = random.randint(0,n_pe-1)
    while n_edges[hub]<=(n_pe-1):
        hub = random.randint(0,n_pe-1)


    if cust_routing == 0:  # static 
        bgp=np.zeros((n_ce,), dtype=int)
    elif cust_routing == 1: # bgp 
        bgp=np.ones((n_ce,), dtype=int)
    else:     # both 
        bgp = np.random.randint(0, 2, size=n_ce) # 1 for bgp and 0 for static 
    #print(bgp)
    for i in range(n_pe): #PEs features
        #hub =  random.randint(0,1) * i
        f=[]
        f.append(1)                                                     #Router type
        f.append(Hub_Spoke)                                             #VPN type
        f.append(hub==i)                                                #VPN Hub 
        f.append(int(n_edges[i]>(n_pe-1)))                              #VRF configured ?  is it connected to a CE ?  
        f.append(int(n_edges[i]>(n_pe-1))*rd)                           #VRF RD
        f.append(int(n_edges[i]>(n_pe-1))*rd)                           #VRF RT Import
        f.append(int(n_edges[i]>(n_pe-1))*((not Hub_Spoke) or (hub==i))*(rd+1) - 1)  #VRF RT Export

        # Route Policies 
        f.append(int(n_edges[i]>(n_pe-1))*Hub_Spoke*(hub!=i)*(1+1) - 1)           #Import if prefix equal to the Hub CE subnet (Enabled on Spoke PEs)  *(1+1) - 1 
        f.append(int(n_edges[i]>(n_pe-1))*Hub_Spoke*(hub!=i)*(rd+1) - 1)                   #Import if extcommunity RT (configured on Spoke PEs)  *(rd+1) - 1

        f.append(int(n_edges[i]>(n_pe-1))*Hub_Spoke*(hub!=i)*(1+1) - 1)           #Export if prefix equal to the connected CE subnet (Enabled on Spoke PEs)  *(1+1) - 1
        f.append(int(n_edges[i]>(n_pe-1))*Hub_Spoke*(hub!=i)*(rd+1) - 1)        #Export with extcommunity RT (configured on Spoke PEs)

        f.append(0)                                     #Int LAN IP address
        f.append(0)                                     #Int LAN IP mask 

        f.append(100)                                   #BGP as
        
        for j in range(n_pe, n):
            f.append(a[i,j]*(j*10+1))                   #Int IP address
            f.append(a[i,j]*30)                         #Int IP mask   
            f.append(a[i,j])                            #Int assigned to VRF ?  is it connected to a CE ?  
            f.append(a[i,j]*rd)                         #Int VRF RD

            f.append(a[i,j]*bgp[j-n_pe]*(j*10+2))           #BGP Neighbor IP address
            f.append(a[i,j]*bgp[j-n_pe]*j)                  #BGP Neighbor remote-as 
            f.append(a[i,j]*bgp[j-n_pe])                    #BGP IPv4
            f.append(a[i,j]*rd)                             #BGP VRF RD
            f.append(-1)                                    #BGP Network IP address
            f.append(-1)                                    #BGP Network IP mask
            f.append(a[i,j]*(not bgp[j-n_pe]))              #BGP Redistribute static routes

            f.append(a[i,j]*(not bgp[j-n_pe])*(j*100) + bgp[j-n_pe]*(-1))       #Static route subnet IP address 
            f.append(a[i,j]*(not bgp[j-n_pe])*24 + bgp[j-n_pe]*(-1))            #Static route subnet IP mask 
            f.append(a[i,j]*(not bgp[j-n_pe])*(j*10+2))                         #Static route next-hope IP address 
            f.append(a[i,j]*(not bgp[j-n_pe])*rd)                               #Static route VRF RD


        features.append(f)

    for i in range(n_pe, n): #CEs features
        f=[]
        f.append(0)                             #Router type
        f.append(Hub_Spoke)                     #VPN type
        f.append(0)                             #VPN Hub 
        f.append(0)                             #VRF configured ?  is it connected to a CE ?  
        f.append(0)                             #VRF RD 
        f.append(0)                             #VRF RT Import
        f.append(0)                             #VRF RT Export

        # Route Policies 
        f.append(0)                             #Export if prefix equal to the connected CE subnet (Enabled on Spoke PEs)  
        f.append(0)                             #Export with extcommunity RT (configured on Spoke PEs)

        f.append(0)                             #Import if prefix equal to the Hub CE subnet (Enabled on Spoke PEs)  
        f.append(0)                             #Import if extcommunity RT (configured on Spoke PEs)  


        f.append(i*100+1)                       #Int LAN IP address
        f.append(24)                            #Int LAN IP mask 
        
        f.append(bgp[i-n_pe]*i)                 #BGP as
        
        f.append(i*10+2)                                        #Int IP address
        f.append(30)                                            #Int IP mask   
        f.append(0)                                             #VRF assigned to VRF ?  is it connected to a CE ?  
        f.append(0)                                             #Int VRF RD
        f.append(bgp[i-n_pe]*(i*10+1))                          #BGP Neighbor IP address
        f.append(bgp[i-n_pe]*100)                               #BGP Neighbor remote-as 
        f.append(bgp[i-n_pe]*1)                                 #BGP IPv4
        f.append(0)                                             #BGP VRF RD
        f.append(bgp[i-n_pe]*i*100 + (not bgp[i-n_pe])*(-1))    #BGP Network IP address
        f.append(bgp[i-n_pe]*24 + (not bgp[i-n_pe])*(-1))       #BGP Network IP mask
        f.append(0)                                             #BGP Redistribute static routes 
        
        f.append((bgp[i-n_pe])*(-1))            #Static route subnet IP address 
        f.append((bgp[i-n_pe])*(-1))            #Static route subnet IP mask 
        f.append((not bgp[i-n_pe])*(i*10+1))    #Static route next-hope IP address 
        f.append(0)                             #Static route VRF RD

        for j in range(n_pe+1, n):
            for _ in range(15):
                f.append(0) 


        features.append(f)      
    #print(features)
    return np.array(features)

def make_edge_features(n_rr, n_p, n_pe , n_ce, n_vrf, vrfs, a):
    n_bb = n_rr + n_p + n_pe
    n = n_bb + n_ce

    # number of edge features 
    n_edge_features = 11

    # init edge features 
    e = np.array([ [ [None for j in range(n_edge_features)] for i in range(n)] for k in range(n)] ) 
    #e = np.zeros((n, n, n_edge_features))
    
    edges = a.toarray()
    
    # BB edges
    for i in range(n_bb):
        # BB edges
        for j in range(i,n_bb):
            if edges[i][j] == 1 :
                # topology features 
                e[i][j][0] = 'BB' #Edge Type
                e[i][j][1] = str(i) #Edge 1 Router ID
                e[i][j][2] = str(j) #Edge 2 Router ID

                # ospf features 
                e[i][j][3] = True #Edge 1 OSPF
                e[i][j][4] = True #Edge 2 OSPF
                e[i][j][5] = '0' #Edge 2 OSPF Area
                e[i][j][6] = '0' #Edge 2 OSPF Area

                # mpls features 
                e[i][j][7] = True #Edge 1 MPLS
                e[i][j][8] = True #Edge 2 MPLS

                # vrf features 
                e[i][j][9] = 'global' #Edge 1 VRF Name
                e[i][j][10] = 'global' #Edge 2 VRF Name

                e[j][i] = e[i][j]  

        # CE_PE edges
        for j in range(n_bb,n):
            if edges[i][j] == 1 :
                # topology features 
                e[i][j][0] = 'CE_PE' #Edge Type
                e[i][j][1] = str(i) #Edge 1 Router ID
                e[i][j][2] = str(j) #Edge 2 Router ID

                # ospf features 
                e[i][j][3] = False #Edge 1 OSPF
                e[i][j][4] = False #Edge 2 OSPF
                #e[i][j][5] = '0' #Edge 2 OSPF Area
                #e[i][j][6] = '0' #Edge 2 OSPF Area

                # mpls features 
                e[i][j][7] = False #Edge 1 MPLS
                e[i][j][8] = False #Edge 2 MPLS

                # vrf features 
                e[i][j][9] = 'customer_'+str(vrfs[j-n_bb]) #Edge 1 VRF Name
                e[i][j][10] = 'global' #Edge 2 VRF Name

                e[j][i] = e[i][j]

    return e
