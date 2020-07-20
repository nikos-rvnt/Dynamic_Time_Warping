#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import scipy.spatial.distance as scdist
from scipy import interpolate, signal

# L_infinite distance aka Chebyshev distance 
def Linf_Dist( dat1, dat2):

    #linf_dist = scdist.cdist( dat1, dat2, metric='chebyshev').flatten()
    if len(dat1)==4:
        #linf_dist = np.max( linf_dist[::5] )
        linf_dist = max( abs(dat1[0]-dat2[0]), abs(dat1[1]-dat2[1]), abs(dat1[2]-dat2[2]), abs(dat1[3]-dat2[3]) )
    
        # dist1 = np.max( abs(dat1[0,0] - dat2[0,0]) + abs(dat1[0,1] - dat2[0,1]))
        # dist2 = np.max( abs(dat1[1,0] - dat2[1,0]) + abs(dat1[1,1] - dat2[1,1]))
        # dist3 = np.max( abs(dat1[2,0] - dat2[2,0]) + abs(dat1[2,1] - dat2[2,1]))
        # dist4 = np.max( abs(dat1[3,0] - dat2[3,0]) + abs(dat1[3,1] - dat2[3,1]))  
        
        # linf_dist = max( dist1, dist2, dist3, dist4)
    
    elif len(dat1)==2:
        #linf_dist = max( abs(dat1[0]-dat2[0]), abs(dat1[1]-dat2[1]) )
    
        dist1 = max( abs(dat1[0,0] - dat2[0,0]) , abs(dat1[0,1] - dat2[0,1]))
        dist2 = max( abs(dat1[1,0] - dat2[1,0]) , abs(dat1[1,1] - dat2[1,1]))  
        
        linf_dist = max( dist1, dist2)
        
    return linf_dist


# Kim lower bound features, that is first, last, lower and higher points 
'''
paper: An index-basedapproach for similarity search supporting time warpingin large sequence databases
'''
def Kim_bounDemLow_4pnt( trajC, trajQ):    
                      
    # extract higher, lower point (x,y) of query
    lowerQ = np.argmin(trajQ[1:-1,1])
    higherQ = np.argmax(trajQ[1:-1,1]) 
    
    # extract higher, lower point (x,y) of candidate
    lowerC = np.argmin(trajC[1:-1,1])
    higherC = np.argmax(trajC[1:-1,1]) 
    
    # left/right hand trajectory simplified as first, lower, higher and last point
    # dat_query = np.array([ trajQ[0,1], trajQ[-1,1], trajQ[lowerQ,1], trajQ[higherQ,1] ])
    # dat_candidate = np.array([ trajC[0,1], trajC[-1,1], trajC[lowerC,1], trajC[higherC,1] ])

    dat_query = np.array([ trajQ[0,1], trajQ[-1,1], trajQ[lowerQ,1], trajQ[higherQ,1] ])
    dat_candidate = np.array([ trajC[0,1], trajC[-1,1], trajC[lowerC,1], trajC[higherC,1] ])

    loba = Linf_Dist( dat_query, dat_candidate)
    loba = loba / (trajC.shape[0] + trajQ.shape[0])

    return loba
    
# Kim lower bound features, that is first and last
def Kim_bounDemLow_2pnt( trajC, trajQ):
        
    # left/right hand trajectory simplified as first, and last point
    dat_query = np.array([ trajQ[0,:], trajQ[-1,:] ])
    dat_candidate = np.array([ trajC[0,:], trajC[-1,:] ])

    # dat_query = np.array([ trajQ[0,:], trajQ[-1,:] ])
    # dat_candidate = np.array([ trajC[0,:], trajC[-1,:] ])

    loba = Linf_Dist( dat_query, dat_candidate)
    loba = loba / (trajC.shape[0] + trajQ.shape[0])
    
    return loba

'''
paper: Efficient retrieval of similar time sequences under time warping
'''    
def Yi_bounDemLow( trajC, trajQ):
                 
    numFeats = trajC.shape[1]
    
    # extract higher, lower point (x,y) of left/right hand trajectory
    lowerQ_y = np.min(trajQ[:,1])
    higherQ_y = np.max(trajQ[:,1])                
    lowerQ_x = np.min(trajQ[:,0])
    higherQ_x = np.max(trajQ[:,0])
    higherQ = [ higherQ_x, higherQ_y]
    lowerQ = [ lowerQ_x, lowerQ_y]

    lowerC_y = np.min(trajC[:,1])
    higherC_y = np.max(trajC[:,1]) 
    lowerC_x = np.min(trajC[:,0])
    higherC_x = np.max(trajC[:,0])
    higherC = [ higherC_x, higherC_y]
    lowerC = [ lowerC_x, lowerC_y]
    
    sum_ = 0    
    # check whether Rx, Ry are disjoint, overlap each other or the one encloses the other
    
    # Rx, Ry are disjoint
    if lowerC_y > higherQ_y:

        sum_x, sum_y = 0, 0
        for i in range( trajC.shape[0] ):
            sum_x += abs(trajC[i,1] - higherQ[1])
            sum_x += abs(trajC[i,0] - higherQ[0])
            
        for j in range( trajQ.shape[0] ):       
            sum_y += abs(trajQ[j,1] - lowerC[1])
            sum_y += abs(trajQ[j,0] - lowerC[0])
            
        sum_ = max(sum_x,sum_y)
    
    # Rx, Ry overlap 
    elif lowerC_y <= higherQ_y and lowerC_y >= lowerQ_y:
        
        for i in range( trajC.shape[0] ):
            for j in range(numFeats):
                if trajC[i,j] > higherQ[j]:
                    sum_ += abs(trajC[i,j] - higherQ[j])
                
        for i in range( trajQ.shape[0] ):
            for j in range(numFeats):
                if trajQ[i,j] < lowerC[j]:
                    sum_ += abs(lowerC[j] - trajQ[i,j])
                
    # Rx encloses Ry
    elif lowerC_y < lowerQ_y:
    
        for p in range( trajC.shape[0] ):
            for j in range(numFeats):
                if trajC[p,j] > higherQ[j]:
                    sum_ += abs(trajC[p,j] - higherQ[j])
                
                if trajC[p,j] < lowerQ[j]:
                    sum_ += abs(lowerQ[j] - trajC[p,j])                                   
    
            # if trajC[p,0] > higherQ_x:
            #     sum_squared_ += abs(trajC[p,0] - higherQ_x)
            
            # if trajC[p,0] < lowerQ_x:
            #     sum_squared_ += abs(lowerQ_x - trajC[p,0])

    loba = sum_ / (trajC.shape[0] + trajQ.shape[0])

    return loba


import matplotlib.pyplot as plt


def findUpperLowerEnvelope(traj,w):

    lenT = traj.shape[0]
    numFeats = traj.shape[1]
    
    upp = np.zeros( traj.shape, dtype=np.float64 )
    low = np.zeros( traj.shape, dtype=np.float64 ) 
     
    for i in range(w,lenT-w):
        for j in range(traj.shape[1]):
            upp[i,j] = np.max( traj[i-w:i+w,j] )
            low[i,j] = np.min( traj[i-w:i+w,j] )
    
    for p in range(numFeats):
        upp[:w,p] = upp[w,p]
        low[:w,p] = low[w,p]
        upp[lenT-w:,p] = upp[lenT-w-1,p]
        low[lenT-w:,p] = low[lenT-w-1,p]    
    
    return upp, low

def resmpl(trj, lenLong, shapeLong):
    trj_longer = np.zeros( shapeLong )
    trj_longer[1:-1] = signal.resample( trj[1:-1], lenLong-2)
    trj_longer[0], trj_longer[-1] = trj[0], trj[-1]
    
    return trj_longer

def makeLengthEqual( trajC, trajQ):
    
    lenQ = trajQ.shape[0]
    lenC = trajC.shape[0]
    
    if lenQ>lenC:    
        trajC = resmpl( trajC, lenQ, trajQ.shape)
  
    elif lenQ<lenC:    
        trajQ = resmpl( trajQ, lenC, trajC.shape)
        
    return trajC, trajQ

'''
paper: Exact  indexing  of  dynamic  time  warping
'''
def Keogh_bounDemLow_SakoeChiba( trajC, trajQ, w):

    lenQ_in = trajQ.shape[0]
    lenC_in = trajC.shape[0]
    
    # trajC = resmpl( trajC, 80, (80,2))
    # trajQ = resmpl( trajQ, 80, (80,2))

    if lenQ_in!=lenC_in:
        trajC, trajQ = makeLengthEqual( trajC, trajQ)
    
    lenC = trajC.shape[0]
    numFeats = trajC.shape[1]

    UpperQ, LowerQ = findUpperLowerEnvelope( trajQ, w)
    
    #x_spc = np.arange( 0, lenQ, 1)
    # path2save = '/media/nikos/Data/didak/HealthSign/implement/healthSign_ditywa/keogh_lb_envUL/'
    # if 'fig_' + str(gls) + '.png' not in path2save:
    #     plt.rcParams.update({'figure.max_open_warning': 0})
    #     plt.figure()
    #     plt.plot( x_spc, UpperQ, 'r-', x_spc, LowerQ, 'b-',  x_spc, trajQ[:], 'g-')
    #     plt.savefig( path2save + 'fig_' + str(gls) + '.png')                
    
    sum_cand_ULQ = 0    
    for i in range(lenC):
        for j in range(numFeats):
            if trajC[i,j] > UpperQ[i,j]:
                sum_cand_ULQ += (trajC[i,j] - UpperQ[i,j])**2
                
            elif trajC[i,j] < LowerQ[i,j]:
                sum_cand_ULQ += (trajC[i,j] - LowerQ[i,j])**2
            
       
    loba = np.sqrt(sum_cand_ULQ) / (lenQ_in + lenC_in)

    return loba, UpperQ, LowerQ

'''
paper: Faster retrieval with a two-pass dynamic-time-warping lower bound
'''
def Improved_bounDemLow( trajC, trajQ, w):

    lenQ = trajQ.shape[0]
    lenC = trajC.shape[0]
    if lenQ!=lenC:
        trajC, trajQ = makeLengthEqual( trajC, trajQ)
    
    uppQ, lowQ = findUpperLowerEnvelope( trajQ, w)

    lenC = trajC.shape[0]
    numFeats = trajC.shape[1]

    trajC_n = np.zeros( trajC.shape, dtype=np.float64)
    # print('TrajC.shape: ' + str(trajC.shape))
    # print('uppQ.shape: ' + str(uppQ.shape))
    for i in range(lenC):
        for j in range(numFeats):
            if trajC[i,j] >= uppQ[i,j]:
                trajC_n[i,j] = uppQ[i,j]
            elif trajC[i,j] <= lowQ[i,j]:
                trajC_n[i,j] = lowQ[i,j]
            else:
                trajC_n[i,j] = trajC[i,j]
    
    loba, uppC, lowC = Keogh_bounDemLow_SakoeChiba( trajQ, trajC_n, w)
    
    return loba


'''
Tightness  of  a  lower  bounding  measure  can  be  defined  as  the ratio  of  lower  bound  distance  to  the  actual  DTW  distance.
The ratio  is  in  the  range[0,1].The  higher  the  ratio,  the  tighter  is  the bound.Pruning  ratio  is  the  number  of  DTW
 computation  while using   lower  bounding   measure   to  that without  using   lower bounding measure.Smaller the ratio,better the pruning power. 
'''

'''
paper: Elastic bands across the path:A new framework and method to lower bound DTW
'''
'''
	uppQ: upper envelope of query series (trajQ)
	lowQ: lower envelope of query series (trajQ)
	w: warping window
	v: tightness parameter specifying the number of left and right bands
	d: current distance to Nearest Neighbour 
'''
def Enhanced_bounDemLow( trajC, trajQ, w, v, d):

    dist = lambda x, y: np.sqrt( np.dot(x, x) - 2*np.dot(x, y) + np.dot(y, y) ) 
    
    lenQ = trajQ.shape[0]
    lenC = trajC.shape[0]
    if lenQ!=lenC:
        trajC, trajQ = makeLengthEqual( trajC, trajQ)

    uppQ, lowQ = findUpperLowerEnvelope( trajQ, w)

    L = trajC.shape[0] - 2
    res = dist(trajC[0,:],trajQ[0,:]) + dist(trajC[-2,:],trajQ[-2,:])
    numOfBands = min( int(L/2), v)

    for i in range(1,numOfBands):
    	
    	rightEnd = L-i
    	minL = dist(trajC[i,:],trajQ[i,:])
    	minR = dist(trajC[L-i,:],trajQ[L-i,:])
    	for j in range(max(0,i-w),i):

    		rightStart = L-j
    		temp_minL = min( minL, dist(trajC[i,:],trajQ[j,:]))
    		minL = min( temp_minL, dist(trajC[j,:],trajQ[i,:]))

    		temp_minR = min( minR, dist(trajC[rightEnd,:],trajQ[rightStart,:]))
    		minR = min( temp_minR, dist(trajC[rightStart,:],trajQ[rightEnd,:]))


    	res += (minL + minR)

    if (res/(lenQ+lenC))>=d:
     	return np.inf

    temp_res = 0
    for i in range(numOfBands,L-numOfBands+1):
    	for j in range(trajC.shape[1]):

    		if trajC[i,j] > uppQ[i,j]:
	            temp_res += ((trajC[i,j] - uppQ[i,j])**2)
	            	            
    		elif trajC[i,j] < lowQ[i,j]:
	            temp_res += ((trajC[i,j] - lowQ[i,j])**2)
	            
    res = (res + np.sqrt(temp_res)) / (lenC + lenQ)
    
    return res


def Enhanced1_bounDemLow( trajC, trajQ, w, d):

    dist = lambda x, y: np.sqrt( np.dot(x, x) - 2*np.dot(x, y) + np.dot(y, y) ) 

    lenQ = trajQ.shape[0]
    lenC = trajC.shape[0]
    if lenQ!=lenC:
        trajC, trajQ = makeLengthEqual( trajC, trajQ)

    uppQ, lowQ = findUpperLowerEnvelope( trajQ, w)

    L = trajC.shape[0] - 2
    res = dist(trajC[0,:],trajQ[0,:]) + dist(trajC[-2,:],trajQ[-2,:])

    if (res/(lenQ+lenC))>=d:
     	return np.inf
    
    temp_res = 0
    for i in range(1,L):
    	for j in range(trajC.shape[1]):

    		if trajC[i,j] > uppQ[i,j]:
	            temp_res += ((trajC[i,j] - uppQ[i,j])**2)
	            
    		elif trajC[i,j] < lowQ[i,j]:
	            temp_res += ((trajC[i,j] - lowQ[i,j])**2)

    res = (res + np.sqrt(temp_res)) / (lenC + lenQ)
    
    return res

