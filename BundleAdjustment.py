import numpy as np
from scipy.spatial.transform import Rotation
from utils import *
import scipy.optimize as spo
from LinearTriangulation import *

def error(X, P, p1):
	# import pdb; pdb.set_trace()
	# X = X.reshape((-1, 1))
	X_ = np.vstack((X.reshape((-1,1)), np.eye(1)))
	error = (p1[0] - ((P[0,:].reshape((1, 4))@X_)/(P[2,:].reshape((1, 4))@X_)))**2 + (p1[1] - ((P[1,:].reshape((1, 4))@X_)/(P[2,:].reshape((1, 4))@X_)))**2
	return error[0][0]


def loss_func(mainX, C_set, R_set, X, K, traj_pts, V):
	# C_set, R_set, X = get_CRX(mainX, len(C_set))
	try:
		proj_error = []
		for i in range(len(C_set)):
			P = make_P(R_set[i], C_set[i])
			for j in range(len(X)):
				if j >= V.shape[1]:
					break
				if V[i, j] == 1 and tuple(X[j]) in traj_pts and i in traj_pts[tuple(X[j])]:
					# import pdb; pdb.set_trace()
					proj_error.append(error(X[j], P, traj_pts[tuple(X[j])][i]))
	except:
		import pdb; pdb.set_trace()
	# import pdb; pdb.set_trace()
	return proj_error

def get_CRX(x, lenC):
	idx = 3*lenC
	idx2 = 4*lenC
	# import pdb; pdb.set_trace()
	C = x[:idx].reshape((-1,3)).tolist()
	q = x[idx:(idx+idx2)].reshape((-1,4)).tolist()
	X = x[idx+idx2:].reshape((-1,3))

	R = [Rotation.from_quat(m).as_matrix() for m in q]
	return C, R, X



def BundleAdjustment(C_Set, R_set, X, K, traj_pts, V):
	# convert R to q
	# import pdb; pdb.set_trace()
	lenC = len(C_Set)

	r = [Rotation.from_matrix(i) for i in R_set]
	q_set = [j.as_quat() for j in r]
	params = [C_Set, R_set, X, K, traj_pts, V]
	X_ravel = X.ravel()
	X_ravel = X_ravel.reshape((len(X_ravel),))
	C_ravel = [np.array(c).reshape((3,)) for c in C_Set]
	C_ravel = np.array(C_ravel).ravel()
	q_ravel = np.array(q_set).ravel()
	mainX = np.hstack((C_ravel, q_ravel, X_ravel))
	# try:
	result = spo.least_squares(loss_func, mainX, args=params, verbose=2, diff_step = 1)
	# except:
	# 	import pdb; pdb.set_trace()
	new_C, new_R, X_new = get_CRX(result.x, lenC)

	# Convert q to R
	return new_C, new_R, X_new