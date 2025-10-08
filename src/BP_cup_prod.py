import numpy as np
from sage.all import *
from BP_codes_sage import regular_rep_matrix, lift_matrix_over_group_algebra, BalancedProductCode, DistanceEst_Gap, ga_transpose_reverse, ga_reverse

def MatrixToSet(M):
    """
    Given a matrix M over a group algebra, decompose it into monomial matrices,
    each with exactly one basis element of the group algebra in one entry.
    """
    monomials = []
    nrows, ncols = M.nrows(), M.ncols()
    R = M.base_ring()
    zero = M.parent().zero()
    
    for i in range(nrows):
        for j in range(ncols):
            entry = M[i, j]
            if entry != 0:
                for g in entry.support():
                    coeff = entry[g]
                    # mono_mat = zero.copy()
                    mono_mat = zero_matrix(R, nrows, ncols)
                    mono_mat[i, j] = coeff * R(g)
                    monomials.append(mono_mat)
    return monomials

def unlift_first_row_to_group_algebra(first_row_lifted, basis, R):
    """
    Reconstruct a 1-row matrix over the group algebra R from the first row
    of its lifted regular representation matrix.

    Arguments:
    - first_row_lifted: 1D list or vector of length n * ncols (field elements)
    - basis: list of group elements (basis of the group algebra)
    - R: group algebra

    Returns:
    - 1-row matrix over R
    """
    n = len(basis)
    G = R.group()
    field = R.base_ring()

    first_row_lifted = vector(field, first_row_lifted)
    ncols = len(first_row_lifted) // n

    # Precompute regular representation of each basis element
    regrep = {g: regular_rep_matrix(R(g), basis, field) for g in basis}

    # Form the matrix over R
    row = []

    for j in range(ncols):
        block = first_row_lifted[j*n:(j+1)*n]

        # Build matrix A with flattened columns of regrep[g]
        A = Matrix(field, [vector(regrep[g][0]) for g in basis]).transpose()
        b = vector(field, block)

        try:
            x = A.solve_right(b)
        except:
            raise ValueError(f"Column {j} not representable as group algebra element")

        elem = sum(x[k] * R(basis[k]) for k in range(n) if x[k] != 0)
        row.append(elem)

    return Matrix(R, [row])

    
class CoChain():
    def __init__(self, level:int, s_vec:Matrix, delta:Matrix, orientation_mats:dict):
        assert level in [0, 1]
        # assert isinstance(s_vec, F2BiPolyMatrix)
#         dim1, dim0 = delta.shape
        self.level = level
        self.s_vec = s_vec
        self.delta = delta
        self.orientation_mats = orientation_mats
        self.l = lift

    
    def boundary(self):
        if self.level == 1:
            return 0
        else:
            nei = self.neighbors()
            return nei[0] + nei[1] + nei[2]
    
    def neighbors(self):
        assert self.level == 0
#         assert len(v.c_set) in [0, 1]
        d_in, d_out, d_free = list(self.orientation_mats.values())
        s_vec = self.s_vec
        e_in_vecs, e_out_vecs, e_free_vecs = MatrixToSet(d_in*s_vec), MatrixToSet(d_out*s_vec), MatrixToSet(d_free*s_vec)
        
        e_ins = [CoChain(level=1, s_vec=x1, delta=self.delta, orientation_mats=self.orientation_mats) for x1 in e_in_vecs]
        e_outs = [CoChain(level=1, s_vec=x1, delta=self.delta, orientation_mats=self.orientation_mats) for x1 in e_out_vecs]
        e_frees = [CoChain(level=1, s_vec=x1, delta=self.delta, orientation_mats=self.orientation_mats) for x1 in e_free_vecs]
    
        return [e_ins, e_outs, e_frees]
        
    def cup_prod(self, v1):
        a1 = self
        a2 = v1
        a1_level, a2_level = a1.level, a2.level
        
        if (a1_level == 0) and (a2_level == 0):
            if a1 == a2:
                return a1
            else: 
                return 0
            
        elif (a1_level == 0) and (a2_level == 1):
            a1_neighbors = a1.neighbors()
            if a2 in a1_neighbors[1]: # a2 in delta_out(a1)
                return a2
            else:
                return 0

        elif (a1_level == 1) and (a2_level == 0):
            a2_neighbors = a2.neighbors()
            if a1 in a2_neighbors[0]: # a1 in delta_ins(a2)
                return a1
            else:
                return 0
                
        else:
            return 0

    def group_action(self, g):
        s_vec = self.s_vec*(g)
        return CoChain(level=self.level, s_vec=s_vec, delta=self.delta, orientation_mats=self.orientation_mats)
    
    def __eq__(self, v1):
        assert isinstance(v1, CoChain) or (v1 == 0)
        if isinstance(v1, CoChain):
#             assert v1.level == self.level
            if v1.level != self.level:
                return False
            else:
#                 return (len(set(self.c_set + v1.c_set)) == len(set(self.c_set))) and (len(set(self.c_set + v1.c_set)) == len(set(v1.c_set)))
                return self.s_vec == v1.s_vec
        else:
#             return len(self.c_set) == v1
            return False


def Integ_Leibniz(a1:CoChain, a2:CoChain, a3:CoChain):
    assert (a1.level + a2.level + a3.level) == 0
    prod_es = []
    for d1 in a1.boundary():
        if d1.cup_prod(a2) != 0:
            if (d1.cup_prod(a2)).cup_prod(a3) != 0: 
                prod_es.append((d1.cup_prod(a2)).cup_prod(a3))
    for d2 in a2.boundary():
        if a1.cup_prod(d2) != 0:
            if (a1.cup_prod(d2)).cup_prod(a3) != 0: 
                prod_es.append((a1.cup_prod(d2)).cup_prod(a3))
    for d3 in a3.boundary():
        if a1.cup_prod(a2) != 0:
            if (a1.cup_prod(a2)).cup_prod(d3) != 0: 
                prod_es.append((a1.cup_prod(a2)).cup_prod(d3))
    return len(prod_es)%2

def GetZeroCochains(delta, orientation_mats):
    cochains_0 = []
    n = delta.ncols()
    R = delta.base_ring()
    G = R.group()
    G_basis = G.list()
    for i in range(n):
        for g in G_basis:
            M = zero_matrix(R, n, 1)
            M[i,0] = g
            cochains_0.append(CoChain(level=0, s_vec=M, delta=delta, orientation_mats=orientation_mats))
    return cochains_0
    
def IfSatInteg_Leibniz(delta:Matrix, orientation_mats:dict):
    cochains_0 = GetZeroCochains(delta, orientation_mats)
    IL_values = []
    for a1 in cochains_0:
        for a2 in cochains_0:
            for a3 in cochains_0:
                IL_values.append(Integ_Leibniz(a1, a2, a3))
    return sum(IL_values) == 0



# 2D, CZ
class ProdCoChain_2D():
    def __init__(self, a1:CoChain, a2:CoChain):
#         assert level in [0, 1]
        self.level = a1.level + a2.level
        
        self.a1 = a1
        self.a2 = a2
    
    def cup_prod(self, q2):
        a1, a2 = self.a1, self.a2
        a1_p, a2_p = q2.a1, q2.a2

        R = a1.s_vec.base_ring()
        G = R.group()
        G_basis = G.list()

        prods = []
        for g in G_basis:
            g_inv = g**(-1)
            a1_c = a1.group_action(g)
            a2_c = a2.group_action(g_inv)
            prod1, prod2 = a1_c.cup_prod(a1_p), a2_c.cup_prod(a2_p)
            if prod1 != 0 and prod2 != 0:
                prods.append(ProdCoChain_2D(prod1, prod2))
        if prods == []:
            return 0
        else:
            return prods
    
    
    def __eq__(self, q2):
        assert isinstance(q2, ProdCoChain_2D) or (q2 == 0)
        if q2 == 0:
            return False
        else:
            return (self.a1 == q2.a1) and (self.a2 == q2.a2)

def inverse_Kron_BP(R_vec, n, m):
    """
    Invert Kron_LP assuming A and B are 1*n and 1*m row vectors,
    each with exactly one nonzero entry.

    Args:
        result (np.ndarray): Result of Kron_LP, shape (1, n*m)
        n (int): Length of A
        m (int): Length of B
        l (int): Modulus used in Kron_LP

    Returns:
        A (np.ndarray): 1 x n row vector
        B (np.ndarray): 1 x m row vector
    """
    # assesrt poly.string_matrix.shape[1] == n*m
    assert R_vec.nrows() == 1
    R = R_vec.base_ring()
    G = R.group()
    
    # A = np.array([['0']*n], dtype='<U10')
    # B = np.array([['0']*m], dtype='<U10')
    A = zero_matrix(R, 1, n)
    B = zero_matrix(R, 1, m)
    zero = R(0)
    one = R(G.identity())

    # Find nonzero index
    indices = []
    for i in range(R_vec.ncols()):
        if R_vec[0, i] != zero:
            indices.append(i)
    if len(indices) != 1:
        raise ValueError("Result must contain exactly one nonzero entry.")
    
    idx = indices[0]
    val = R_vec[0][idx]

    j0 = idx // m
    f0 = idx % m

    # Fix A[0, j0] = 1, then solve for B[0, f0]
    a = one
    b = val

    A[0, j0] = a
    B[0, f0] = b

    return A, B


def XoperToProdCoChains_2D(x_vec, deltas:dict, orientation_mats:dict):
    delta_a, delta_b = list(deltas.values())
    orientation_mats_a, orientation_mats_b = list(orientation_mats.values())
    
    # input: x_vec \in F_2^n, output: {(a_i, b_i)}_i, where a_i \in C^1_a (C^0_a), b_i \in C^0_b (C^1_b)
    r_a, n_a = delta_a.ncols(), delta_a.nrows()
    r_b, n_b = delta_b.ncols(), delta_b.nrows()

    R = delta_a.base_ring()
    G = R.group()
    G_basis = G.list()
    
    x_vecs = MatrixToSet(unlift_first_row_to_group_algebra(x_vec, G_basis, R))
    
    prod_cochains = []
    
    for x_vec in x_vecs:
        x_vec = ga_reverse(x_vec)
        # print(x_vec, x_vec.nrows(), x_vec.ncols(), n_b*r_a)
        x_vec_L = x_vec[:, :n_b*r_a] # the L sector C^1_b\times C^0_a
        x_vec_R = x_vec[:, n_b*r_a:] # the R sector C^0_b\times C^1_a
        
        if x_vec_L:
            vec_a, vec_b = inverse_Kron_BP(x_vec_L, r_a, n_b)
            
            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            
            v_a = CoChain(level=0, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            e_b = CoChain(level=1, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            prod_cochains.append(ProdCoChain_2D(v_a, e_b))
        
        if x_vec_R:
            vec_a, vec_b = inverse_Kron_BP(x_vec_R, n_a, r_b)
            
            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            
            e_a = CoChain(level=1, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            v_b = CoChain(level=0, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            prod_cochains.append(ProdCoChain_2D(e_a, v_b))
            
    return prod_cochains

def CupInteg_2D(x_oper1, x_oper2, deltas, orientation_mats):
    x_oper_cochains1 = XoperToProdCoChains_2D(x_oper1, deltas, orientation_mats)
    x_oper_cochains2 = XoperToProdCoChains_2D(x_oper2, deltas, orientation_mats)
    
    # prod_X_cochains = []
    prod_X_cochains_counter = 0
    for q_cochain_1 in x_oper_cochains1:
        for q_cochain_2 in x_oper_cochains2:
            prod_X_cochain = q_cochain_1.cup_prod(q_cochain_2)
            if prod_X_cochain != 0:
                prod_X_cochains_counter += 1
    return prod_X_cochains_counter%2


# calculate physical CZ matrix
def PhysicalCZMat(deltas, orientation_mats):
    delta_a, delta_b = list(deltas.values())
    r_a, n_a = delta_a.ncols(), delta_a.nrows()
    r_b, n_b = delta_b.ncols(), delta_b.nrows()
    R = delta_a.base_ring()
    G = R.group()
    L = G.order()
    n = (r_a*n_b + n_a*r_b)*L
    
    def Run(ixs):
        i, j = ixs
        x_oper1 = np.zeros(n, dtype=int)
        x_oper1[i] = 1
        x_oper2 = np.zeros(n, dtype=int)
        x_oper2[j] = 1
        phase = CupInteg_2D(x_oper1, x_oper2, deltas, orientation_mats)
        return phase

    ixs = [(i, j) for i in range(n) for j in range(n)]
    # phases = parmap(Run, ixs, nprocs = mp.cpu_count())
    phases = np.array([Run(ix) for ix in ixs])
    P_CZ_mat = np.reshape(phases, [n, n])
    return P_CZ_mat

# Verify cohomology operation
def CalPhase_CZ(x_oper1, x_oper2, P_CZ_mat):
    # return (x_oper1@P_CZ_mat@x_oper2)%2
    return np.einsum('ij,i,j->', P_CZ_mat, x_oper1, x_oper2)%2
    # return (np.sum(P_CZ_mat*x_oper1[:, None] * x_oper2[None, :]))%2


def VerifyCohomology_CZ(eval_code, P_CZ_mat):
    X_check_prods = []
    for hx1 in eval_code.hx.toarray():
        for hx2 in eval_code.hx.toarray():
            phase = CalPhase_CZ(hx1, hx2, P_CZ_mat)
            X_check_prods.append(phase)

    Check_L_prods = []
    for hx1 in eval_code.hx.toarray():
        for lx2 in eval_code.lx.toarray():
            phase = CalPhase_CZ(hx1, lx2, P_CZ_mat)
            Check_L_prods.append(phase)
    
    return sum(X_check_prods) == 0 and sum(Check_L_prods) == 0

def LogicalCZMat(eval_code, P_CZ_mat):
    k = eval_code.lx.shape[0]
    CZ_mat = np.zeros([k, k], dtype=int)
    for i in range(k):
        for j in range(k):
            CZ_mat[i, j] = CalPhase_CZ(eval_code.lx.toarray()[i], eval_code.lx.toarray()[j], P_CZ_mat)

    return CZ_mat    


# 3D, CCZ
def inverse_Kron_BP_3D(R_vec, n1, n2, n3):
    """
    Invert Kron_LP assuming A and B are 1 x n and 1 x m row vectors,
    each with exactly one nonzero entry.

    Args:
        result (np.ndarray): Result of Kron_LP, shape (1, n*m)
        n (int): Length of A
        m (int): Length of B
        l (int): Modulus used in Kron_LP

    Returns:
        A (np.ndarray): 1 x n row vector
        B (np.ndarray): 1 x m row vector
    """
    AB, C = inverse_Kron_BP(R_vec, n1*n2, n3)
    A, B = inverse_Kron_BP(AB, n1, n2)

    return A, B, C   

class ProdCoChain_3D():
    def __init__(self, a1:CoChain, a2:CoChain, a3:CoChain):
        self.level = a1.level + a2.level + a3.level
        
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    
    def cup_prod(self, q2):
#         assert q2 == 
        a1, a2, a3 = self.a1, self.a2, self.a3
        a1_p, a2_p, a3_p = q2.a1, q2.a2, q2.a3

        R = a1.s_vec.base_ring()
        G = R.group()
        G_basis = G.list()
        
        prods = []
        for g in G:
            g_inv = g**(-1)
            for g_p in G:
                g_p_inv = g_p**(-1)
                a1_c = a1.group_action(g)
                a2_c = (a2.group_action(g_inv)).group_action(g_p)
                a3_c = a3.group_action(g_p_inv)

                prod1, prod2, prod3 = a1_c.cup_prod(a1_p), a2_c.cup_prod(a2_p), a3_c.cup_prod(a3_p)
                if prod1 != 0 and prod2 != 0 and prod3 != 0:
                    prods.append(ProdCoChain_3D(prod1, prod2, prod3))
        if prods == []:
            return 0
        else:
            return prods
    
    def __eq__(self, q2):
        assert isinstance(q2, ProdCoChain_2D) or (q2 == 0)
        if q2 == 0:
            return False
        else:
            return (self.a1 == q2.a1) and (self.a2 == q2.a2) and (self.a3 == q2.a3)  


def XoperToProdCoChains_3D(x_vec, deltas:dict, orientation_mats:dict):
    delta_a, delta_b, delta_c = list(deltas.values())
    orientation_mats_a, orientation_mats_b, orientation_mats_c = list(orientation_mats.values())
    # input: x_vec \in F_2^n, output: {(a_i, b_i)}_i, where a_i \in C^1_a (C^0_a), b_i \in C^0_b (C^1_b)
    r_a, n_a = delta_a.ncols(), delta_a.nrows()
    r_b, n_b = delta_b.ncols(), delta_b.nrows()
    r_c, n_c = delta_c.ncols(), delta_c.nrows()
    
    # x_vecs = VecToMonimials(x_vec, l)
    R = delta_a.base_ring()
    G = R.group()
    G_basis = G.list()
    
    x_vecs = MatrixToSet(unlift_first_row_to_group_algebra(x_vec, G_basis, R))

    prod_cochains = []
    
    for x_vec in x_vecs:
        x_vec = ga_reverse(x_vec)
        x_vec_1 = x_vec[:, :r_a*r_b*n_c] # the first sector C^1_c\times C^0_b\times C^0_a
        x_vec_2 = x_vec[:, n_c*r_b*r_a:n_c*r_b*r_a + r_c*n_b*r_a] # the second sector C^0_c \times C^1_b\times C^0_a
        x_vec_3 = x_vec[:, n_c*r_b*r_a + r_c*n_b*r_a:] # the thurd sector C^0_c \times C^0_b \times C^1_a
    
        if x_vec_1:
            vec_a, vec_b, vec_c = inverse_Kron_BP_3D(x_vec_1, r_a, r_b, n_c)
            
            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            vec_c = vec_c.transpose()
            
            v_a = CoChain(level=0, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            v_b = CoChain(level=0, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            e_c = CoChain(level=1, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
            prod_cochains.append(ProdCoChain_3D(v_a, v_b, e_c))
            
        if x_vec_2:
            vec_a, vec_b, vec_c = inverse_Kron_BP_3D(x_vec_2, r_a, n_b, r_c)
            
            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            vec_c = vec_c.transpose()
            
            v_a = CoChain(level=0, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            e_b = CoChain(level=1, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            v_c = CoChain(level=0, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
            prod_cochains.append(ProdCoChain_3D(v_a, e_b, v_c))
            
        if x_vec_3:
            vec_a, vec_b, vec_c = inverse_Kron_BP_3D(x_vec_3, n_a, r_b, r_c)
            
            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            vec_c = vec_c.transpose()
            
            e_a = CoChain(level=1, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            v_b = CoChain(level=0, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            v_c = CoChain(level=0, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
            prod_cochains.append(ProdCoChain_3D(e_a, v_b, v_c))
            
    return prod_cochains 

def CupInteg_3D(x_oper1, x_oper2, x_oper3, deltas, orientation_mats):
    delta_a, delta_b, delta_c = list(deltas.values())
    orientation_mat_a, orientation_mat_b, orientation_mat_c = list(orientation_mats.values())
    
    x_oper_cochains1 = XoperToProdCoChains_3D(x_oper1, deltas, orientation_mats)
    x_oper_cochains2 = XoperToProdCoChains_3D(x_oper2, deltas, orientation_mats)
    x_oper_cochains3 = XoperToProdCoChains_3D(x_oper3, deltas, orientation_mats)
    
    prod_X_cochains_counter = 0
    for q_cochain_1 in x_oper_cochains1:
        for q_cochain_2 in x_oper_cochains2:
            for q_cochain_3 in x_oper_cochains3:
                q_cochain_12s = q_cochain_1.cup_prod(q_cochain_2)
                if q_cochain_12s != 0: 
                    for q_cochain_12 in q_cochain_12s:
                        prod_X_cochain = q_cochain_12.cup_prod(q_cochain_3)
                        if prod_X_cochain != 0:
                            prod_X_cochains_counter += 1
    return prod_X_cochains_counter%2


# Verify cohomology operation
def CalPhase_CCZ(x_oper1, x_oper2, x_oper3, P_CCZ_mat):
    return np.einsum('ijk,i,j,k->', P_CCZ_mat, x_oper1, x_oper2, x_oper3)%2


def VerifyCohomology_CCZ(eval_code, P_CCZ_mat):
    XXXs = []
    for hx1 in eval_code.hx.toarray():
        for hx2 in eval_code.hx.toarray():
            for hx3 in eval_code.hx.toarray():
                phase = CalPhase_CCZ(hx1, hx2, hx3, P_CCZ_mat)
                if phase != 0:
                    print('XXXs fail:', phase)
                XXXs.append(phase)

    XXLs = []
    for hx1 in eval_code.hx.toarray():
        for hx2 in eval_code.hx.toarray():
            for lx3 in eval_code.lx.toarray():
                phase = CalPhase_CCZ(hx1, hx2, lx3, P_CCZ_mat)
                XXLs.append(phase)
                
    XLLs = []
    for hx1 in eval_code.hx.toarray():
        for lx2 in eval_code.lx.toarray():
            for lx3 in eval_code.lx.toarray():
                phase = CalPhase_CCZ(hx1, lx2, lx3, P_CCZ_mat)
                XLLs.append(phase)
                
    return sum(XXXs) == 0 and sum(XXLs) == 0 and sum(XLLs) == 0

def LogicalCCZMat(eval_code, P_CCZ_mat):
    k = eval_code.lx.shape[0]
    CCZ_mat = np.zeros([k, k, k], dtype=int)
    for i in range(k):
        for j in range(k):
            for t in range(k):
                CCZ_mat[i, j, t] = CalPhase_CCZ(eval_code.lx.toarray()[i], eval_code.lx.toarray()[j], eval_code.lx.toarray()[t], P_CCZ_mat)

    return CCZ_mat 