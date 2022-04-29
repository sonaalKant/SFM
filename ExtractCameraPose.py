import numpy as np

def ExtractCameraPose(E):
	W = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
	U, S, V = np.linalg.svd(E)
	C1 = U[:,2]
	C2 = -U[:,2]
	C3 = U[:,2]
	C4 = -U[:,2]
	R1 = U@W@V
	
	R2 = U@W@V
	
	R3 = U@W.T@V
	
	R4 = U@W.T@V
	
	print(np.linalg.det(R1), np.linalg.det(R2), np.linalg.det(R3), np.linalg.det(R4))
	if np.linalg.det(R4) < 0:
		R4 = -R4
		C4 = -C4
	if np.linalg.det(R3) < 0:
		R3 = -R3
		C3 = -C3
	if np.linalg.det(R2) < 0:
		R2 = -R2
		C2 = -C2
	if np.linalg.det(R1) < 0:
		R1 = -R1
		C1 = -C1
	# P = K@R3@np.hstack((np.eye(3),-C3.reshape((3,1))))
	return [R1, R2, R3, R4], [C1, C2, C3, C4]