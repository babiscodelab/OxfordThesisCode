import numpy as np
from numpy import sqrt, log2, reshape

#
# calculation of multi-dimensional Brownian Bridge
#
# function dW = bb(Z,T)
# function dW = bb(Z,T,C)
#
# T       -- time interval
# Z(:,:)  -- input vectors of unit Normal variables
# C(:,:)  -- square PCA matrix for multi-dimensional Brownian motion (optional)
# dW(:,:) -- output vectors of Brownian path increments
#
# first dimension of Z/dW corresponds to Brownian dimension * #timesteps
# second dimension of Z/dW corresponds to #paths
#

def bb(W,T,C=None):

    W = W.copy()

    #
    # one-dimensional Brownian motion
    #

    if C is None:
        K = 1
        N = W.shape[0]

    #
    # for multi-dimensional Brownian motion, reorder input
    # to get contiguous time
    #

    else:

        K = C.shape[0]
        if C.shape[0] != C.shape[1]:
            raise ValueError('matrix C not square')

        nW = W.shape[0]*W.shape[1]
        N = W.shape[0]
        if N%K != 0:
            raise ValueError('first dimension of W not divisible by first dimension of C')

        N = int(N/K)

        p = reshape(reshape(np.arange(N*K), (N,K)).T, (N*K, ))
        W = reshape(W[p,:], (N,-1), order='F')

    #
    # perform Brownian Bridge interpolation
    #

    M = int(log2(N))
    if N != 2**M:
        raise ValueError('number of timesteps not a power of 2')

    for m in range(1,M+1):
        mask = np.arange(2**m).reshape((-1,2)).T.flatten()
        #mask = np.array(range(0,2**m+1,2) + range(1,2**m+1,2))
        W[mask,:] = np.vstack(\
        [ 0.5*W[:2**(m-1),:]+W[2**(m-1):2**m,:]/sqrt(2)**(m+1), \
        0.5*W[:2**(m-1),:]-W[2**(m-1):2**m,:]/sqrt(2)**(m+1) ])

    #
    # for multi-dimensional Brownian motion, reorder output
    # to get contiguous multiple dimensions and apply PCA
    #

    if C is not None:
        W = reshape(W, (N*K, -1), order='F')
        p = reshape(reshape(np.arange(N*K), (N,K)).T, (N*K, ))
        W = reshape(W[p,:],(K,-1), order='F')
        W = reshape(C.dot(W),(N*K,-1), order='F')

    #
    # finally, adjust for non-unit time interval
    #

    W = sqrt(T)*W

    return W
