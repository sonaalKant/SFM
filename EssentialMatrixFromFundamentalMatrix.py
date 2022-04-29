import numpy as np


def get_E(F, K):
	E = K.T@F@K
	U, S, V = np.linalg.svd(E)
	E = U@np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])@V
	
	# import pdb; pdb.set_trace()

	return E