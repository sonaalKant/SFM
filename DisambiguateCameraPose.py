from LinearTriangulation import *

def cheirality(X, R_set, C_set):
	pos_count = []
	idxs = [[], [], [], []]
	for c, i in enumerate(X):
		positive = 0
		r3 = R_set[c][2,:]
		C = C_set[c]
		for m, j in enumerate(i):
			# import pdb; pdb.set_trace()
			if r3@(j-C).T > 0 and j[2] > 0:
				positive += 1
				idxs[c].append(m)
		pos_count.append(positive)
	pos_count = np.array(pos_count)
	test = np.argsort(pos_count)[::-1][:2]
	if np.std(X[test[0]][:,2]) > np.std(X[test[1]][:,2]):
		idx = test[0]
	else:
		idx = test[1]
	# idx = np.argmax(pos_count)
	idxs = idxs[idx]
	return idx, idxs

def DisambiguateCameraPose(C_set, R_set, X):
	# X = [[], [], [], []]
	# for i, (R, C) in enumerate(zip(R_set, C_set)):
	# 	for p1, p2 in zip(pts1, pts2):
	# 		X[i].append(LinearTriangulation(p1, p2))
	idx, idxs = cheirality(X, R_set, C_set)
	return R_set[idx], C_set[idx], X[idx], idxs, idx