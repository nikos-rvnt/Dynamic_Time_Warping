

import numpy as np


def evaluateAlignment( Dp, path):

	# constant that helps bigger commonalities 
	# be more important to smaller ones
	c = 0.1

	# find diagonal elements
	matches = 0
	len_p = len(path)
	for i in range(len_p):
		if path[i][0] == path[i][1]:
			matches += 1

	# Fitness score - weighted average evaluation of DTW distance
	Fp = (Dp + c)/(matches + 1)

	return Fp



# first order differential:
# provides shape information
def firstDerivative(v):

	v_1st_deriv = ( (v[1] - v[0]) + ((v[2] - v[0])/2) ) / 2

	return v_1st_deriv

# second order differential:
# provides min value, max value, infection point information   
def secondDerivative(v):

	v_sec_deriv = v[2] + v[0] - 2*v[0]

	return v_sec_deriv
	

#############################################################################
#
# Input Arguments:
# 	- s: query
#   - t: template
#   - version: 0 -> classic distance matrix computation
#			   1 -> distance matrix computation based on 1st,2nd derivatives
#     (paper: Human Action Recognition based on DTW and Movement Trajectory)
#   - symmetry: 0 -> symmetric, 1 -> quasi-symmetric, 2-> assymetric
#   - w: warping window
#
# Output Arguments:
#   - DTW[-1,-1]: DTW normalised distance
#   - cost_matrix: distance matrix
#   - DTW: cumulative distance matrix
#   - path: DTW distance path
#
############################################################################
def dtwDistance( s, t, version = 0, symmetry = 0, w = np.inf):

    
    # n-> rows, m -> columns 
    n = len(s)
    m = len(t)
    
    # warping window to constrain DTW path computation near the diagonal
    warp = max( w, np.abs(n-m))
    
    # distance function
    dist = lambda x, y: np.sqrt( np.dot(x, x) - 2*np.dot(x, y) + np.dot(y, y) ) 
    
    # cost_matrix -> distance matrix
    cost_matrix = np.zeros((n,m))
    if version == 0:
    	for i in range(n):
            for j in range(m):
                cost_matrix[i,j] = dist( s[i], t[j])
    else:
        cost_matrix[0,0] = dist( s[0], t[0])
        cost_matrix[n-1,m-1] = dist( s[n-1], t[m-1])
   		# dtw derivative weights
        w3, w2, w1 = 2.4, 1.2, 0.6
        for i in range( 1, n-1):
		
            s_deriv1 = firstDerivative(s[i-1:i+2])
            s_deriv2 = secondDerivative(s[i-1:i+2])
            for j in range( 1, m-1):
                
                t_deriv1 = firstDerivative(t[j-1:j+2])
                t_deriv2 = secondDerivative(t[j-1:j+2])
                cost_matrix[i,j] = w1*dist(s[i],t[j]) + w2*dist(s_deriv1,t_deriv1) + w3*dist(s_deriv2,t_deriv2)
    

	# cumulative distance matrix
    #DTW = np.zeros((n,m))
    DTW = np.full( (n,m), np.inf)
    # if symmetric DTW
    if symmetry == 0:

        DTW[0,0] = 2*cost_matrix[0,0]
        for i in range( 1, min(n,warp + 1)):
            DTW[i,0] = DTW[i-1,0] + cost_matrix[i,0]
            #DTW[i,0] = np.sum( cost_matrix[:i+1,0] )

        for i in range( 1, min(m,warp + 1)):
            DTW[0,i] = DTW[0,i-1] + cost_matrix[0,i]
            #DTW[0,i] = np.sum( cost_matrix[0,:i+1] )

        for i in range( 1, n):
            for j in range( max( 1, i-warp), min( m, i+warp+1)):  # range(1,m):

                DTW[i][j] = min( DTW[i-1][j] + cost_matrix[i,j],  DTW[i][j-1] + cost_matrix[i,j], DTW[i-1][j-1] + 2*cost_matrix[i,j] )
        # time-normalised distance
        DTW[-1][-1] = DTW[-1][-1] / (n+m)    

    # if quazi-symmetric DTW
    elif symmetry == 1:

        DTW[0,0] = cost_matrix[0,0]
        for i in range(1,min(n,warp + 1)):
            DTW[i,0] = DTW[i-1,0] + cost_matrix[i,0]
            #DTW[i,0] = np.sum( cost_matrix[:i+1,0] )

        for i in range(1,min(m,warp + 1)):
            DTW[0,i] = DTW[0,i-1] + cost_matrix[0,i]
            #DTW[0,i] = np.sum( cost_matrix[0,:i+1] )

        for i in range(1,n):
            for j in range( max( 1, i-warp), min( m, i+warp+1)):  # 
            #for j in range(1,m):

                DTW[i][j] = cost_matrix[i][j] + min( DTW[i-1][j],  DTW[i][j-1], DTW[i-1][j-1])
        # time-normalised distance
        DTW[-1][-1] = (DTW[-1][-1]) / (n+m) 

    # if asymmetric DTW
    elif symmetry == 2:

        DTW[0,0] = cost_matrix[0,0]
        for i in range(1,min(n,warp + 1)):
            DTW[i,0] = DTW[i-1,0] + cost_matrix[i,0]
            #DTW[i,0] = np.sum( cost_matrix[:i+1,0] )

        for i in range(1,min(m,warp + 1)):
            DTW[0,i] = DTW[0,i-1] + cost_matrix[0,i]
            #DTW[0,i] = np.sum( cost_matrix[0,:i+1] )

        for i in range(1,n):
            for j in range( max( 1, i-warp), min( m, i+warp+1)):  # range(1,m):

                DTW[i][j] = min( DTW[i-1][j] + cost_matrix[i][j],  DTW[i][j-1], DTW[i-1][j-1] + cost_matrix[i][j])
        # time-normalised distance
        DTW[-1][-1] = DTW[-1][-1] / (n)

    # backtrace to find path
    i, j = np.array(DTW.shape) - 2
    path = []
    path = [(n-1,m-1)]
    while (i >= 0) and (j >= 0):

        if i==0 and j>0:
            path.append( (i,j) )
            j = j - 1

        elif j==0 and i>0:
            path.append( (i,j) )
            i = i - 1
 
        else:
            min_ind = np.argmin(( DTW[i, j], DTW[i, j + 1], DTW[i + 1, j]))
            if min_ind == 0:
                path.append( (i,j) )
                i -= 1
                j -= 1
            elif min_ind == 1:
                path.append( (i,j+1) )
                i -= 1
            else:  
                path.append( (i+1,j) )
                j -= 1

    path.reverse()

    #return DTW[ -1, -1], cost_matrix, DTW, path
    return DTW[ -1, -1], cost_matrix, DTW
