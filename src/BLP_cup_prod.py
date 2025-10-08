import numpy as np
import itertools
import copy
import multiprocessing as mp

# from f2poly_matrix import F2PolyMatrix
from f2poly_matrix import F2BiPolyMatrix

# Modify the multiprocessing functions
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


def vec_basis_BB(n, l):
    vecs = []
    for i in range(n):
        for alpha in range(l):
            for beta in range(l):
                str_mat = np.array([['0']]*n, dtype='<U10')
                if alpha == 0 and beta == 0:
                    str_mat[i] = ['1']
                elif alpha == 0:
                    str_mat[i] = ['y^' + str(beta)]
                elif beta == 0:
                    str_mat[i] = ['x^' + str(alpha)]
                else:
                    str_mat[i] = ['x^' + str(alpha) + '*y^' + str(beta)]
                vecs.append(F2BiPolyMatrix(str_mat, l))
    return vecs

def Group_BB(l):
    vecs = []
    for alpha in range(l):
        for beta in range(l):
            str_mat = np.array([['0']], dtype='<U10')
            if alpha == 0 and beta == 0:
                str_mat[0] = ['1']
            elif alpha == 0:
                str_mat[0] = ['y^' + str(beta)]
            elif beta == 0:
                str_mat[0] = ['x^' + str(alpha)]
            else:
                str_mat[0] = ['x^' + str(alpha) + '*y^' + str(beta)]
            vecs.append(F2BiPolyMatrix(str_mat, l))
    return vecs



class CoChain():
    def __init__(self, level:int, lift:int, s_vec:F2BiPolyMatrix, delta:F2BiPolyMatrix, orientation_mats:dict):
        assert level in [0, 1]
        assert isinstance(s_vec, F2BiPolyMatrix)
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
        e_in_vecs, e_out_vecs, e_free_vecs = (d_in.multiply(s_vec)).vec_to_set(), (d_out.multiply(s_vec)).vec_to_set(), (d_free.multiply(s_vec)).vec_to_set()
        
        e_ins = [CoChain(level=1, lift=self.l, s_vec=x1, delta=self.delta, orientation_mats=self.orientation_mats) for x1 in e_in_vecs]
        e_outs = [CoChain(level=1, lift=self.l, s_vec=x1, delta=self.delta, orientation_mats=self.orientation_mats) for x1 in e_out_vecs]
        e_frees = [CoChain(level=1, lift=self.l, s_vec=x1, delta=self.delta, orientation_mats=self.orientation_mats) for x1 in e_free_vecs]
    
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
#             a1_neighbors =  Orientation(self.delta, self.orientation_mats, a1)
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
        s_vec = self.s_vec.multiply(g)
        return CoChain(level=self.level, lift=self.l, s_vec=s_vec, delta=self.delta, orientation_mats=self.orientation_mats)
#         self.s_vec = g.multiply(self.s_vec)
        
    def __add__(self, v1):
        assert (isinstance(v1, CoChain)) or (v1 == 0)
        if v1 == 0:
            return self
        else:
            assert v1.level == self.level
            v = CoChain(level=self.level, lift=self.l, c_set=VecToIxs((self.vec_rep() + v1.vec_rep())%2), delta=self.delta, orientation_mats=self.orientation_mats)
            return v
        
    def __radd__(self, other):
        # Needed for commutativity when left operand doesn't know how to add
        return self.__add__(other)
    
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


# Verify the leibniz condition
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

def IfSatInteg_Leibniz(cochains_0):
    IL_values = []
    for a1 in cochains_0:
        for a2 in cochains_0:
            for a3 in cochains_0:
                IL_values.append(Integ_Leibniz(a1, a2, a3))
    return sum(IL_values) == 0


def VecToPolyMonimials(lifted_row, L):
    """
    Reconstruct the first row of a F2BiPolyMatrix from the first row of its binary lifted version.
    lifted_row: numpy array of shape (n * L^2,)
    Returns a new F2BiPolyMatrix with a single row.
    """
    import numpy as np

#     S = generate_circulant(L)
    S = np.zeros((L, L), dtype=int)
    for i in range(L):
        S[i, (i + 1) % L] = 1
    I = np.eye(L, dtype=int)
    X = np.kron(S, I)
    Y = np.kron(I, S)
    L2 = L * L
    n = len(lifted_row) // L2

    # Precompute monomial basis vectors for x^i y^j
    monomial_basis = {}
    for i in range(L):
        for j in range(L):
            vec = (np.linalg.matrix_power(X, i) @ np.linalg.matrix_power(Y, j))[0] % 2  # First row only
            monomial_basis[(i, j)] = vec

    string_row = []

    for j in range(n):
        block_row = lifted_row[j*L2:(j+1)*L2] % 2
        terms = []
        for (xi, yj), vec in monomial_basis.items():
            if np.array_equal(block_row & vec, vec):
                terms.append(
                    ("x" if xi > 0 else "") + (f"^{xi}" if xi > 1 else "") +
                    ("y" if yj > 0 else "") + (f"^{yj}" if yj > 1 else "")
                    if xi or yj else "1"
                )
                block_row = (block_row ^ vec) % 2  # Remove matched monomial
        string_row.append('+'.join(terms) if terms else '0')
    poly = F2BiPolyMatrix([string_row], L)

    return poly.vec_to_set()

def inverse_Kron_LP_BB(poly, n, m, l):
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
    assert poly.string_matrix.shape[1] == n*m
    
    A = np.array([['0']*n], dtype='<U10')
    B = np.array([['0']*m], dtype='<U10')

    # Find nonzero index
#     indices = np.nonzero(result)[0]
    indices = np.where(poly.string_matrix[0] != '0')[0]
    if len(indices) != 1:
        raise ValueError("Result must contain exactly one nonzero entry.")
    
    idx = indices
    val = poly.string_matrix[0][idx]

    j0 = idx // m
    f0 = idx % m

    # Fix A[0, j0] = 1, then solve for B[0, f0]
    a = '1'
    b = val

    A[0, j0] = a
    B[0, f0] = b

    return F2BiPolyMatrix(A, l), F2BiPolyMatrix(B, l)



class ProdCoChain_2D():
    def __init__(self, a1:CoChain, a2:CoChain):
#         assert level in [0, 1]
        self.level = a1.level + a2.level
#         self.c_set = list(itertools.product(a1.c_set, a2.c_set))
#         self.dim = a1.dim*a2.dim
        
        self.a1 = a1
        self.a2 = a2
    
    def cup_prod(self, q2):
        a1, a2 = self.a1, self.a2
        a1_p, a2_p = q2.a1, q2.a2

        prods = []
        G = Group_BB(a1.l)
        for g in G:
            g_inv = g.transpose_reverse()
            # a1_c, a2_c = copy.deepcopy(a1), copy.deepcopy(a2)
            # a1_c.group_action(g)
            # a2_c.group_action(g_inv)
            a1_c = a1.group_action(g)
            a2_c = a2.group_action(g_inv)
            if a1_c.cup_prod(a1_p) != 0 and a2_c.cup_prod(a2_p) != 0:
                prods.append(ProdCoChain_2D(a1_c.cup_prod(a1_p), a2_c.cup_prod(a2_p)))
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
    
# def XoperToProdCoChains_2D(x_vec, l, delta_a, orientation_mats_a, delta_b, orientation_mats_b):
#     # input: x_vec \in F_2^n, output: {(a_i, b_i)}_i, where a_i \in C^1_a (C^0_a), b_i \in C^0_b (C^1_b)
#     r_a, n_a = delta_a.transpose_reverse().string_matrix.shape
#     r_b, n_b = delta_b.transpose_reverse().string_matrix.shape
    
# #     x_vecs = VecToMonimials(x_vec, l)
#     x_vecs = VecToPolyMonimials(x_vec, l)
    
#     prod_cochains = []
    
#     for x_vec in x_vecs:
#         x_vec = x_vec.reverse()
#         x_vec_L = F2BiPolyMatrix(x_vec.string_matrix[:, :n_b*r_a], l) # the L sector C^1_b\times C^0_a
#         x_vec_R = F2BiPolyMatrix(x_vec.string_matrix[:, n_b*r_a:], l) # the R sector C^0_b\times C^1_a
        
# #         print(x_vec_L.string_matrix, np.where(x_vec_L.string_matrix[0] != '0')[0])
#         if len(np.where(x_vec_L.string_matrix[0] != '0')[0]) > 0:
#             x_vec_L_prod = inverse_Kron_LP_BB(x_vec_L, n_b, r_a, l)
#             vec_b, vec_a = x_vec_L_prod
            
#             # vec_a_str = np.reshape(vec_a.string_matrix, [vec_a.string_matrix.shape[1], 1])
#             # vec_b_str = np.reshape(vec_b.string_matrix, [vec_b.string_matrix.shape[1], 1])
#             # vec_a = F2BiPolyMatrix(vec_a_str, l)
#             # vec_b = F2BiPolyMatrix(vec_b_str, l)

#             vec_a = vec_a.transpose() # transform to column vectors
#             vec_b = vec_b.transpose()
            
#             v_a = CoChain(level=0, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
#             e_b = CoChain(level=1, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
#             prod_cochains.append(ProdCoChain_2D(v_a, e_b))
        
# #         print(x_vec_R.string_matrix, np.where(x_vec_R.string_matrix[0] != '0')[0])
#         if len(np.where(x_vec_R.string_matrix[0] != '0')[0]) > 0:
# #             print(x_vec_R.string_matrix)
#             x_vec_R_prod = inverse_Kron_LP_BB(x_vec_R, r_b, n_a, l)
# #             print(x_vec_R_prod)
#             vec_b, vec_a = x_vec_R_prod
            
# #             print(vec_a.string_matrix, vec_b.string_matrix)
#             # vec_a_str = np.reshape(vec_a.string_matrix, [vec_a.string_matrix.shape[1], 1])
#             # vec_b_str = np.reshape(vec_b.string_matrix, [vec_b.string_matrix.shape[1], 1])
#             # vec_a = F2BiPolyMatrix(vec_a_str, l)
#             # vec_b = F2BiPolyMatrix(vec_b_str, l)

#             vec_a = vec_a.transpose() # transform to column vectors
#             vec_b = vec_b.transpose()
            
#             e_a = CoChain(level=1, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
#             v_b = CoChain(level=0, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
#             prod_cochains.append(ProdCoChain_2D(e_a, v_b))
            
#     return prod_cochains

def XoperToProdCoChains_2D(x_vec, l, delta_a, orientation_mats_a, delta_b, orientation_mats_b):
    # input: x_vec \in F_2^n, output: {(a_i, b_i)}_i, where a_i \in C^1_a (C^0_a), b_i \in C^0_b (C^1_b)
    r_a, n_a = delta_a.transpose_reverse().string_matrix.shape
    r_b, n_b = delta_b.transpose_reverse().string_matrix.shape
    
#     x_vecs = VecToMonimials(x_vec, l)
    x_vecs = VecToPolyMonimials(x_vec, l)
    
    prod_cochains = []
    
    for x_vec in x_vecs:
        x_vec = x_vec.reverse()
        x_vec_L = F2BiPolyMatrix(x_vec.string_matrix[:, :n_b*r_a], l) # the L sector C^1_b\times C^0_a
        x_vec_R = F2BiPolyMatrix(x_vec.string_matrix[:, n_b*r_a:], l) # the R sector C^0_b\times C^1_a
        
#         print(x_vec_L.string_matrix, np.where(x_vec_L.string_matrix[0] != '0')[0])
        if len(np.where(x_vec_L.string_matrix[0] != '0')[0]) > 0:
            x_vec_L_prod = inverse_Kron_LP_BB(x_vec_L, r_a, n_b, l)
            vec_a, vec_b = x_vec_L_prod
            
            # vec_a_str = np.reshape(vec_a.string_matrix, [vec_a.string_matrix.shape[1], 1])
            # vec_b_str = np.reshape(vec_b.string_matrix, [vec_b.string_matrix.shape[1], 1])
            # vec_a = F2BiPolyMatrix(vec_a_str, l)
            # vec_b = F2BiPolyMatrix(vec_b_str, l)

            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            
            v_a = CoChain(level=0, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            e_b = CoChain(level=1, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            prod_cochains.append(ProdCoChain_2D(v_a, e_b))
        
#         print(x_vec_R.string_matrix, np.where(x_vec_R.string_matrix[0] != '0')[0])
        if len(np.where(x_vec_R.string_matrix[0] != '0')[0]) > 0:
#             print(x_vec_R.string_matrix)
            x_vec_R_prod = inverse_Kron_LP_BB(x_vec_R, n_a, r_b, l)
#             print(x_vec_R_prod)
            vec_a, vec_b = x_vec_R_prod
            
#             print(vec_a.string_matrix, vec_b.string_matrix)
            # vec_a_str = np.reshape(vec_a.string_matrix, [vec_a.string_matrix.shape[1], 1])
            # vec_b_str = np.reshape(vec_b.string_matrix, [vec_b.string_matrix.shape[1], 1])
            # vec_a = F2BiPolyMatrix(vec_a_str, l)
            # vec_b = F2BiPolyMatrix(vec_b_str, l)

            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            
            e_a = CoChain(level=1, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            v_b = CoChain(level=0, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            prod_cochains.append(ProdCoChain_2D(e_a, v_b))
            
    return prod_cochains

# def CupInteg_2D(x_oper1, x_oper2, l, delta_a, orientation_mats_a, delta_b, orientation_mats_b):
def CupInteg_2D(x_oper1, x_oper2, l, deltas, orientation_mats):
    delta_a, delta_b = list(deltas.values())
    orientation_mats_a, orientation_mats_b = list(orientation_mats.values())
    x_oper_cochains1 = XoperToProdCoChains_2D(x_oper1, l, delta_a, orientation_mats_a, delta_b, orientation_mats_b)
    x_oper_cochains2 = XoperToProdCoChains_2D(x_oper2, l, delta_a, orientation_mats_a, delta_b, orientation_mats_b)
    
    # prod_X_cochains = []
    prod_X_cochains_counter = 0
    for q_cochain_1 in x_oper_cochains1:
        for q_cochain_2 in x_oper_cochains2:
            prod_X_cochain = q_cochain_1.cup_prod(q_cochain_2)
            if prod_X_cochain != 0:
                # prod_X_cochains += prod_X_cochain
                prod_X_cochains_counter += 1
    # return len(prod_X_cochains)%2
    return prod_X_cochains_counter%2


# calculate physical CZ matrix
def PhysicalCZMat(l, deltas, orientation_mats):
    delta_a, delta_b = list(deltas.values())
    r_a, n_a = delta_a.string_matrix.shape
    r_b, n_b = delta_b.string_matrix.shape
    n = (r_a*n_b + n_a*r_b)*delta_a.L**2
    
    def Run(ixs):
        i, j = ixs
        x_oper1 = np.zeros(n, dtype=int)
        x_oper1[i] = 1
        x_oper2 = np.zeros(n, dtype=int)
        x_oper2[j] = 1
        phase = CupInteg_2D(x_oper1, x_oper2, l, deltas, orientation_mats)
        return phase

    ixs = [(i, j) for i in range(n) for j in range(n)]
    phases = parmap(Run, ixs, nprocs = mp.cpu_count())
    P_CZ_mat = np.reshape(phases, [n, n])
    return P_CZ_mat

# Verify cohomology operation
def CalPhase_CZ(x_oper1, x_oper2, P_CZ_mat):
    # return (x_oper1@P_CZ_mat@x_oper2)%2
    return np.einsum('ij,i,j->', P_CZ_mat, x_oper1, x_oper2)%2
    # return (np.sum(P_CZ_mat*x_oper1[:, None] * x_oper2[None, :]))%2


def VerifyCohomology_CZ(eval_code, P_CZ_mat):
    X_check_prods = []
    for hx1 in eval_code.hx:
        for hx2 in eval_code.hx:
            phase = CalPhase_CZ(hx1, hx2, P_CZ_mat)
            X_check_prods.append(phase)

    Check_L_prods = []
    for hx1 in eval_code.hx:
        for lx2 in eval_code.lx:
            phase = CalPhase_CZ(hx1, lx2, P_CZ_mat)
            Check_L_prods.append(phase)
    
    return sum(X_check_prods) == 0 and sum(Check_L_prods) == 0

def LogicalCZMat(eval_code, P_CZ_mat):
    k = eval_code.lx.shape[0]
    CZ_mat = np.zeros([k, k], dtype=int)
    for i in range(k):
        for j in range(k):
            CZ_mat[i, j] = CalPhase_CZ(eval_code.lx[i], eval_code.lx[j], P_CZ_mat)

    return CZ_mat    


# # 3D, CCZ
# def inverse_Kron_LP_BB_3D(poly, n1, n2, n3, l):
#     """
#     Invert Kron_LP assuming A and B are 1 x n and 1 x m row vectors,
#     each with exactly one nonzero entry.

#     Args:
#         result (np.ndarray): Result of Kron_LP, shape (1, n*m)
#         n (int): Length of A
#         m (int): Length of B
#         l (int): Modulus used in Kron_LP

#     Returns:
#         A (np.ndarray): 1 x n row vector
#         B (np.ndarray): 1 x m row vector
#     """
#     AB, C = inverse_Kron_LP_BB(poly, n1*n2, n3, l)
#     A, B = inverse_Kron_LP_BB(AB, n1, n2, l)

#     return A, B, C   

# def XoperToProdCoChains_3D(x_vec, l, deltas:dict, orientation_mats:dict):
#     delta_a, delta_b, delta_c = list(deltas.values())
#     orientation_mat_a, orientation_mat_b, orientation_mat_c = list(orientation_mats.values())
#     # input: x_vec \in F_2^n, output: {(a_i, b_i)}_i, where a_i \in C^1_a (C^0_a), b_i \in C^0_b (C^1_b)
#     r_a, n_a = delta_a.transpose_reverse().string_matrix.shape
#     r_b, n_b = delta_b.transpose_reverse().string_matrix.shape
#     r_c, n_c = delta_c.transpose_reverse().string_matrix.shape
    
#     # x_vecs = VecToMonimials(x_vec, l)
#     x_vecs = VecToPolyMonimials(x_vec, l)
#     prod_cochains = []
    
#     for x_vec in x_vecs:
#         x_vec = x_vec.reverse()
#         x_vec_1 = F2BiPolyMatrix(x_vec.string_matrix[:n_c*r_b*r_a], l) # the first sector C^1_c\times C^0_b\times C^0_a
#         x_vec_2 = F2BiPolyMatrix(x_vec.string_matrix[n_c*r_b*r_a:n_c*r_b*r_a + r_c*n_b*r_a], l) # the second sector C^0_c \times C^1_b\times C^0_a
#         x_vec_3 = F2BiPolyMatrix(x_vec.string_matrix[n_c*r_b*r_a + r_c*n_b*r_a:], l) # the thurd sector C^0_c \times C^0_b \times C^1_a
    
#         if len(np.where(x_vec_1.string_matrix[0] != '0')[0]) > 0:
#             x_vec_prod = inverse_Kron_LP_3D(x_vec_1, n_c, r_b, r_a,l)
#             vec_c, vec_b, vec_a = x_vec_prod
#             v_a = CoChain(level=0, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
#             v_b = CoChain(level=0, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
#             e_c = CoChain(level=1, lift=l, s_vec=vec_c, delta=delta_b, orientation_mats=orientation_mats_b)
#             prod_cochains.append(ProdCoChain_3D(v_a, v_b, e_c))
            
#         if len(np.where(x_vec_2.string_matrix[0] != '0')[0]) > 0:
#             x_vec_prod = inverse_Kron_LP_3D(x_vec_2, r_c, n_b, r_a, l)
#             vec_c, vec_b, vec_a = x_vec_prod
#             v_a = CoChain(level=0, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
#             e_b = CoChain(level=1, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
#             v_c = CoChain(level=0, lift=l, s_vec=vec_c, delta=delta_b, orientation_mats=orientation_mats_b)
#             prod_cochains.append(ProdCoChain_3D(v_a, e_b, v_c))
            
#         if len(np.where(x_vec_3.string_matrix[0] != '0')[0]) > 0:
#             x_vec_prod = inverse_Kron_LP_3D(x_vec_3, r_c, r_b, n_a, l)
#             vec_c, vec_b, vec_a = x_vec_prod
#             e_a = CoChain(level=1, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
#             v_b = CoChain(level=0, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
#             v_c = CoChain(level=0, lift=l, s_vec=vec_c, delta=delta_b, orientation_mats=orientation_mats_b)
#             prod_cochains.append(ProdCoChain_3D(e_a, v_b, v_c))
            
#     return prod_cochains 




# 3D, CCZ
def inverse_Kron_LP_BB_3D(poly, n1, n2, n3, l):
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
    AB, C = inverse_Kron_LP_BB(poly, n1*n2, n3, l)
    A, B = inverse_Kron_LP_BB(AB, n1, n2, l)

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
        prods = []
        G = Group_BB(a1.l)
        for g in G:
            g_inv = g.transpose_reverse()
            for g_p in G:
                g_p_inv = g_p.transpose_reverse()
                a1_c = a1.group_action(g)
                a2_c = (a2.group_action(g_inv)).group_action(g_p)
                a3_c = a3.group_action(g_p_inv)
                
                if a1_c.cup_prod(a1_p) != 0 and a2_c.cup_prod(a2_p) != 0 and a3_c.cup_prod(a3_p) != 0:
                    prods.append(ProdCoChain_3D(a1_c.cup_prod(a1_p), a2_c.cup_prod(a2_p), a3_c.cup_prod(a3_p)))
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

        
# def XoperToProdCoChains_3D(x_vec, l, deltas:dict, orientation_mats:dict):
#     delta_a, delta_b, delta_c = list(deltas.values())
#     orientation_mat_a, orientation_mat_b, orientation_mat_c = list(orientation_mats.values())
#     # input: x_vec \in F_2^n, output: {(a_i, b_i)}_i, where a_i \in C^1_a (C^0_a), b_i \in C^0_b (C^1_b)
#     r_a, n_a = delta_a.transpose_reverse().string_matrix.shape
#     r_b, n_b = delta_b.transpose_reverse().string_matrix.shape
#     r_c, n_c = delta_c.transpose_reverse().string_matrix.shape
    
#     # x_vecs = VecToMonimials(x_vec, l)
#     x_vecs = VecToPolyMonimials(x_vec, l)
#     prod_cochains = []
    
#     for x_vec in x_vecs:
#         x_vec = x_vec.reverse()
#         x_vec_1 = F2BiPolyMatrix(x_vec.string_matrix[:, :n_c*r_b*r_a], l) # the first sector C^1_c\times C^0_b\times C^0_a
#         x_vec_2 = F2BiPolyMatrix(x_vec.string_matrix[:, n_c*r_b*r_a:n_c*r_b*r_a + r_c*n_b*r_a], l) # the second sector C^0_c \times C^1_b\times C^0_a
#         x_vec_3 = F2BiPolyMatrix(x_vec.string_matrix[:, n_c*r_b*r_a + r_c*n_b*r_a:], l) # the thurd sector C^0_c \times C^0_b \times C^1_a
    
#         if len(np.where(x_vec_1.string_matrix[0] != '0')[0]) > 0:
#             x_vec_prod = inverse_Kron_LP_BB_3D(x_vec_1, n_c, r_b, r_a,l)
#             vec_c, vec_b, vec_a = x_vec_prod
            
#             vec_a = vec_a.transpose() # transform to column vectors
#             vec_b = vec_b.transpose()
#             vec_c = vec_c.transpose()
            
#             v_a = CoChain(level=0, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
#             v_b = CoChain(level=0, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
#             e_c = CoChain(level=1, lift=l, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
#             prod_cochains.append(ProdCoChain_3D(v_a, v_b, e_c))
            
#         if len(np.where(x_vec_2.string_matrix[0] != '0')[0]) > 0:
#             x_vec_prod = inverse_Kron_LP_BB_3D(x_vec_2, r_c, n_b, r_a, l)
#             vec_c, vec_b, vec_a = x_vec_prod
            
#             vec_a = vec_a.transpose() # transform to column vectors
#             vec_b = vec_b.transpose()
#             vec_c = vec_c.transpose()
            
#             v_a = CoChain(level=0, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
#             e_b = CoChain(level=1, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
#             v_c = CoChain(level=0, lift=l, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
#             prod_cochains.append(ProdCoChain_3D(v_a, e_b, v_c))
            
#         if len(np.where(x_vec_3.string_matrix[0] != '0')[0]) > 0:
#             x_vec_prod = inverse_Kron_LP_BB_3D(x_vec_3, r_c, r_b, n_a, l)
#             vec_c, vec_b, vec_a = x_vec_prod
            
#             vec_a = vec_a.transpose() # transform to column vectors
#             vec_b = vec_b.transpose()
#             vec_c = vec_c.transpose()
            
#             e_a = CoChain(level=1, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
#             v_b = CoChain(level=0, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
#             v_c = CoChain(level=0, lift=l, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
#             prod_cochains.append(ProdCoChain_3D(e_a, v_b, v_c))
            
#     return prod_cochains 

def XoperToProdCoChains_3D(x_vec, l, deltas:dict, orientation_mats:dict):
    delta_a, delta_b, delta_c = list(deltas.values())
    orientation_mats_a, orientation_mats_b, orientation_mats_c = list(orientation_mats.values())
    # input: x_vec \in F_2^n, output: {(a_i, b_i)}_i, where a_i \in C^1_a (C^0_a), b_i \in C^0_b (C^1_b)
    r_a, n_a = delta_a.transpose_reverse().string_matrix.shape
    r_b, n_b = delta_b.transpose_reverse().string_matrix.shape
    r_c, n_c = delta_c.transpose_reverse().string_matrix.shape
    
    # x_vecs = VecToMonimials(x_vec, l)
    x_vecs = VecToPolyMonimials(x_vec, l)
    prod_cochains = []
    
    for x_vec in x_vecs:
        x_vec = x_vec.reverse()
        x_vec_1 = F2BiPolyMatrix(x_vec.string_matrix[:, :r_a*r_b*n_c], l) # the first sector C^1_c\times C^0_b\times C^0_a
        x_vec_2 = F2BiPolyMatrix(x_vec.string_matrix[:, n_c*r_b*r_a:n_c*r_b*r_a + r_c*n_b*r_a], l) # the second sector C^0_c \times C^1_b\times C^0_a
        x_vec_3 = F2BiPolyMatrix(x_vec.string_matrix[:, n_c*r_b*r_a + r_c*n_b*r_a:], l) # the thurd sector C^0_c \times C^0_b \times C^1_a
    
        if len(np.where(x_vec_1.string_matrix[0] != '0')[0]) > 0:
            x_vec_prod = inverse_Kron_LP_BB_3D(x_vec_1, r_a, r_b, n_c, l)
            vec_a, vec_b, vec_c = x_vec_prod
            
            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            vec_c = vec_c.transpose()
            
            v_a = CoChain(level=0, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            v_b = CoChain(level=0, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            e_c = CoChain(level=1, lift=l, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
            prod_cochains.append(ProdCoChain_3D(v_a, v_b, e_c))
            
        if len(np.where(x_vec_2.string_matrix[0] != '0')[0]) > 0:
            x_vec_prod = inverse_Kron_LP_BB_3D(x_vec_2, r_a, n_b, r_c, l)
            vec_a, vec_b, vec_c = x_vec_prod
            
            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            vec_c = vec_c.transpose()
            
            v_a = CoChain(level=0, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            e_b = CoChain(level=1, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            v_c = CoChain(level=0, lift=l, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
            prod_cochains.append(ProdCoChain_3D(v_a, e_b, v_c))
            
        if len(np.where(x_vec_3.string_matrix[0] != '0')[0]) > 0:
            x_vec_prod = inverse_Kron_LP_BB_3D(x_vec_3, n_a, r_b, r_c, l)
            vec_a, vec_b, vec_c = x_vec_prod
            
            vec_a = vec_a.transpose() # transform to column vectors
            vec_b = vec_b.transpose()
            vec_c = vec_c.transpose()
            
            e_a = CoChain(level=1, lift=l, s_vec=vec_a, delta=delta_a, orientation_mats=orientation_mats_a)
            v_b = CoChain(level=0, lift=l, s_vec=vec_b, delta=delta_b, orientation_mats=orientation_mats_b)
            v_c = CoChain(level=0, lift=l, s_vec=vec_c, delta=delta_c, orientation_mats=orientation_mats_c)
            prod_cochains.append(ProdCoChain_3D(e_a, v_b, v_c))
            
    return prod_cochains 

def CupInteg_3D(x_oper1, x_oper2, x_oper3, l, deltas, orientation_mats):
    delta_a, delta_b, delta_c = list(deltas.values())
    orientation_mat_a, orientation_mat_b, orientation_mat_c = list(orientation_mats.values())
    
    x_oper_cochains1 = XoperToProdCoChains_3D(x_oper1, l, deltas, orientation_mats)
    x_oper_cochains2 = XoperToProdCoChains_3D(x_oper2, l, deltas, orientation_mats)
    x_oper_cochains3 = XoperToProdCoChains_3D(x_oper3, l, deltas, orientation_mats)
    
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


# calculate physical CZ matrix
def PhysicalCCZMat(l, deltas, orientation_mats):
    delta_a, delta_b, delta_c = list(deltas.values())
    # r_a, n_a = delta_a.string_matrix.shape
    # r_b, n_b = delta_b.string_matrix.shape
    # r_c, n_c = delta_c.string_matrix.shape
    r_a, n_a = delta_a.transpose_reverse().string_matrix.shape
    r_b, n_b = delta_b.transpose_reverse().string_matrix.shape
    r_c, n_c = delta_c.transpose_reverse().string_matrix.shape
    n = (n_c*r_b*r_a + r_c*n_b*r_a + r_c*r_b*n_a)*delta_a.L**2
    
    def Run(ixs):
        i, j, k = ixs
        x_oper1 = np.zeros(n, dtype=int)
        x_oper1[i] = 1
        x_oper2 = np.zeros(n, dtype=int)
        x_oper2[j] = 1
        x_oper3 = np.zeros(n, dtype=int)
        x_oper3[k] = 1
        phase = CupInteg_3D(x_oper1, x_oper2, x_oper3, l, deltas, orientation_mats)
        return phase

    ixs = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]
    phases = parmap(Run, ixs, nprocs = mp.cpu_count())
    P_CCZ_mat = np.reshape(phases, [n, n, n])
    return P_CCZ_mat



# Verify cohomology operation
def CalPhase_CCZ(x_oper1, x_oper2, x_oper3, P_CCZ_mat):
    return np.einsum('ijk,i,j,k->', P_CCZ_mat, x_oper1, x_oper2, x_oper3)%2


def VerifyCohomology_CCZ(eval_code, P_CCZ_mat):
    XXXs = []
    for hx1 in eval_code.hx:
        for hx2 in eval_code.hx:
            for hx3 in eval_code.hx:
                phase = CalPhase_CCZ(hx1, hx2, hx3, P_CCZ_mat)
                if phase != 0:
                    print('XXXs fail:', phase)
                XXXs.append(phase)

    XXLs = []
    for hx1 in eval_code.hx:
        for hx2 in eval_code.hx:
            for lx3 in eval_code.lx:
                phase = CalPhase_CCZ(hx1, hx2, lx3, P_CCZ_mat)
                XXLs.append(phase)
                
    XLLs = []
    for hx1 in eval_code.hx:
        for lx2 in eval_code.lx:
            for lx3 in eval_code.lx:
                phase = CalPhase_CCZ(hx1, lx2, lx3, P_CCZ_mat)
                XLLs.append(phase)
                
    return sum(XXXs) == 0 and sum(XXLs) == 0 and sum(XLLs) == 0

def LogicalCCZMat(eval_code, P_CCZ_mat):
    k = eval_code.lx.shape[0]
    CCZ_mat = np.zeros([k, k, k], dtype=int)
    for i in range(k):
        for j in range(k):
            for t in range(k):
                CCZ_mat[i, j, t] = CalPhase_CCZ(eval_code.lx[i], eval_code.lx[j], eval_code.lx[t], P_CCZ_mat)

    return CCZ_mat 


def LogicalCCZMat_Parallel(eval_code, l, deltas, orientation_mats):
    k = eval_code.lx.shape[0]
    def Run(ixs):
        i, j, t = ixs
        phase = CupInteg_3D(eval_code.lx[i], eval_code.lx[j], eval_code.lx[t], l, deltas, orientation_mats)
        return phase
        
    ixs = [(i, j, t) for i in range(k) for j in range(k) for t in range(k)]
    phases = parmap(Run, ixs, nprocs = mp.cpu_count())
    L_CCZ_mat = np.reshape(phases, [k, k, k])
    return L_CCZ_mat



