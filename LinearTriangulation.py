import numpy as np

K = np.array([[568.996140852, 0, 643.21055941], 
	[0, 568.988362396, 477.982801038], 
	[0, 0, 1]])

def make_A(P1, P2, pt1, pt2):
	y1 = pt1[1]
	y2 = pt2[1]
	x1 = pt1[0]
	x2 = pt2[0]
	p1 = P1[0,:]
	p2 = P1[1,:]
	p3 = P1[2,:]
	p1_ = P2[0,:]
	p2_ = P2[1,:]
	p3_ = P2[2,:]
	# import pdb; pdb.set_trace()
	r1 = y1*p3 - p2
	r2 = p1 - x1*p3
	r3 = y2*p3_ - p2_
	r4 = p1_ - x2*p3_

	A = np.vstack((r1, r2, r3, r4))
	return A

def make_P(R, C):
	RC = R@(C)
	# import pdb; pdb.set_trace()
	M = np.hstack((R, RC.reshape((3,1))))
	P = K@M
	return P



def LinearTriangulation(K, C2, R2, pt1, pt2):
	C1 = np.zeros((3,1))
	R1 = np.eye(3)
	P1 = make_P(R1, C1)
	P2 = make_P(R2, C2)
	X_set = []
	for i in range(len(pt1)):
		A = make_A(P1, P2, pt1[i], pt2[i])
		u, s, v = np.linalg.svd(A)
		X = v[-1, :]
		X = (X/X[3])[:3]
		X_set.append(X)
	return np.array(X_set)
	