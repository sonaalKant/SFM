import numpy as np
from utils import *

# def make_A(X, x, K):
# 	X = np.hstack((X, np.ones((len(X), 1))))
# 	x = np.hstack((x, np.ones((len(x), 1))))
# 	for i in range(len(X)):
# 		xi = np.linalg.inv(K)@x[i]
# 		xi = (xi/xi[2])[:2]
# 		x1 = xi[0]
# 		y1 = xi[1]
# 		Xi = X[i].reshape((4,1))
# 		# import pdb; pdb.set_trace()
# 		r1 = np.hstack((Xi.T, np.zeros((1,4)), -x1*Xi.T))
# 		r2 = np.hstack((np.zeros((1,4)), Xi.T, -y1*Xi.T))
# 		stack = np.vstack((r1, r2))
# 		if i == 0:
# 			A = stack
# 		else:
# 			A = np.vstack((A, stack))
# 	return A

# def LinearPnP(X, x, K):
# 	# x_norm = np.linalg.inv(K)@x
# 	# x_norm = (x_norm/x_norm[2])[:2]
# 	A = make_A(X, x, K)
# 	u, s, v = np.linalg.svd(A)
# 	# import pdb; pdb.set_trace()
# 	P = v[-1, :].reshape((3, 4))
# 	U, S, V = np.linalg.svd(P)
# 	C = V[-1, :]
# 	R = np.linalg.inv(K)@P[:, :3]
# 	return R, C

def LinearPnP(K,x,X):
    x = to_homogeneous(x)
    X = to_homogeneous(X)

    K_inv = np.linalg.inv(K)
    x = (K_inv@x.T).T

    zeros = np.zeros((4))
    A = list()
    for i in range(len(x)):
        u,v,_ = x[i]
        a = np.vstack( (np.hstack((zeros, -X[i], v*X[i])), np.hstack((X[i], zeros, -u*X[i])), np.hstack((-v*X[i], u*X[i], zeros)) ))
        A.append(a)
    
    A = np.array(A).reshape(-1,12)
    U,S,V = np.linalg.svd(A)
    P = V[-1].reshape(3,4)
    R = P[:,:3]
    U,S,V = np.linalg.svd(R)
    R = U@V
    t = P[:,3]
    C = -np.linalg.inv(R)@t
    if np.linalg.det(R)<0:
    	R = -R
    	C = -C

    return R,C
	

