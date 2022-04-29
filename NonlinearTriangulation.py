import numpy as np
import scipy.optimize as spo
from LinearTriangulation import make_P

def error_func(X, P, pts1, pts2):
	X = X.reshape((-1, 3))
	X_ = np.hstack((X, np.ones((len(X), 1))))
	error = []
	for i, (p1, p2) in enumerate(zip(pts1, pts2)):
		error_1 = (p1[0] - ((P[0][0,:]@X_[i].T)/(P[0][2,:]@X_[i].T)))**2 + (p1[1] - ((P[0][1,:]@X_[i].T)/(P[0][2,:]@X_[i].T)))**2
		error_2 = (p2[0] - ((P[1][0,:]@X_[i].T)/(P[1][2,:]@X_[i].T)))**2 + (p2[1] - ((P[1][1,:]@X_[i].T)/(P[1][2,:]@X_[i].T)))**2
		error.append(error_1+error_2)
	return error


def NonlinearTriangulation(R1, C1, R2, C2, X, pts1, pts2): # Pass X as 3x1
	# C1 = np.zeros((3,1))
	# R1 = np.eye(3)
	P = [make_P(R1, C1), make_P(R2, C2)]
	params = (P, pts1, pts2)
	# import pdb; pdb.set_trace()
	result = spo.least_squares(error_func, X.ravel(), args = params, verbose=2, diff_step = 0.1)
	new_X = result.x
	return new_X.reshape((-1,3))