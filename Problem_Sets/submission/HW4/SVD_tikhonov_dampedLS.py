import numpy as np
import scipy



def SVD_tikhonov_dampedLS(G, d, a, order=0):

    def DOF(mrow, ncol):
        if mrow>ncol:
            return mrow-ncol
        else:
            return mrow
    
    if order == 0:
        # perform SVD
        U, s, VT = scipy.linalg.svd(G)
        S = np.diag(s)
        
        # calculate generalized inverse
        A = (G.T @ G + (a**2.)* np.eye(len(s)))
        Gsharp = np.linalg.inv(A) @ G.T

        # calculate resolution matrices
        VFVT = Gsharp @ G
        UFUT = G @ Gsharp
        Rm = np.copy(VFVT)
        Rd = np.copy(UFUT)

        # calculate model solution
        msharp = Gsharp @ d

        # estimate DOF for discrepancy principle
        dof = DOF(G.shape[0], G.shape[1])
        return msharp, Gsharp, Rm, Rd, dof
    
    elif order == 1:
        mrows, ncols = G.shape
        L = -np.eye(ncols) + np.diag(np.ones(ncols-1), k=1)

        GTG = G.T @ G
        LTL = L.T @ L
        GTd = G.T @ d
        
        # solve first order tikhanov (forget the matrix algebra bruh, just use linalg.solve)
        m_tikh = np.linalg.solve(GTG + (a**2.)*LTL, GTd)
        
        # calculate generalized inverse
        A = (GTG + (a**2.)* LTL)
        Gsharp = np.linalg.inv(A) @ G.T
        
        VFVT = Gsharp @ G
        UFUT = G @ Gsharp
        Rm = np.copy(VFVT)
        Rd = np.copy(UFUT)

        dof = DOF(mrows, ncols)
        return m_tikh, Gsharp, Rm, Rd, L, dof
    

    elif order == 2:
        mrows, ncols = G.shape

        def construct_L2_operator(n):
            # k=0 is the main diagonal
            # k=1 is the super-diagonal
            # k=2 is the second super-diagonal
            L = (np.diag(np.ones(n), k=0) + 
            np.diag(-2 * np.ones(n-1), k=1) + 
            np.diag(np.ones(n-2), k=2))
            return L
        
        L = construct_L2_operator(ncols)

        GTG = G.T @ G
        LTL = L.T @ L
        GTd = G.T @ d
        
        # solve first order tikhanov (forget the matrix algebra bruh, just use linalg.solve)
        m_tikh = np.linalg.solve(GTG + (a**2.)*LTL, GTd)
        
        # calculate generalized inverse
        A = (GTG + (a**2.)* LTL)
        Gsharp = np.linalg.inv(A) @ G.T
        
        VFVT = Gsharp @ G
        UFUT = G @ Gsharp
        Rm = np.copy(VFVT)
        Rd = np.copy(UFUT)

        dof = DOF(mrows, ncols)
        return m_tikh, Gsharp, Rm, Rd, L, dof
