import numpy as np

def do_chirality(C_set, R_set, X_set):
    best = []
    max_pos = -1
    for i, (C,R,X) in enumerate(zip(C_set, R_set, X_set)):
        X = np.array(X)
        Z = R[-1,:].T @ (X - C.T).T
        import pdb;pdb.set_trace()
        idxs = np.where(np.bitwise_and(Z>0, X[:,-1] >0))[0]
        
        print(len(idxs), max_pos)
        if len(idxs) > len(best):
            best = idxs
            max_pos = i

    C = C_set[max_pos]
    R = R_set[max_pos]
    X = np.array(X_set[max_pos])[best]

    import pdb;pdb.set_trace()

    return C,R,X