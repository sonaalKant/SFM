import numpy as np

def BuildVisibilityMatrix(traj):
	V = np.zeros((len(traj), 6))
	for i, j in enumerate(traj.values()):
		V[i][j] = 1
	return V.T