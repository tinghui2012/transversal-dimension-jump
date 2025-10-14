import numpy as np
from bposd.css import css_code
from scipy.linalg import block_diag
from sage.all import *
import sys
sys.path.append('./src/')
gap('LoadPackage("QDistRnd");')
from BP_codes_sage import lift_matrix_over_group_algebra


def get_HXHZ_2D(H_A, H_B):
    r_A, n_A = H_A.shape
    r_B, n_B = H_B.shape

    H_Z_T_top = np.kron(H_A, np.eye(n_B))
    H_Z_T_bot = np.kron(np.eye(n_A), H_B)
    H_Z_T = np.vstack((H_Z_T_top, H_Z_T_bot))
    H_Z = H_Z_T.T

    H_X_left = np.kron(np.eye(r_A), H_B)
    H_X_right = np.kron(H_A, np.eye(r_B))
    H_X = np.hstack((H_X_left, H_X_right))
    
    return H_X, H_Z

def getHXHZ_3D(H_A, H_B, H_C):
    # using convention: Q0 = X checks, Q1 = qubits, Q2 = Z checks, Q3 = meta-Z checks
    r_A, n_A = H_A.shape
    r_B, n_B = H_B.shape
    r_C, n_C = H_C.shape

    H_Z_T_top_left = np.kron(np.kron(np.eye(r_A), H_B), np.eye(n_C))
    H_Z_T_top_mid = np.kron(np.kron(H_A, np.eye(r_B)), np.eye(n_C))
    H_Z_T_top_right = np.zeros((r_A*r_B*n_C, n_A*n_B*r_C)) # need to flip order for np
    H_Z_T_top = np.hstack((H_Z_T_top_left, H_Z_T_top_mid, H_Z_T_top_right))

    H_Z_T_mid_left = np.kron(np.kron(np.eye(r_A), np.eye(n_B)), H_C)
    H_Z_T_mid_mid = np.zeros((r_A*n_B*r_C, n_A*r_B*n_C)) # need to flip order for np
    H_Z_T_mid_right = np.kron(np.kron(H_A, np.eye(n_B)), np.eye(r_C))
    H_Z_T_mid = np.hstack((H_Z_T_mid_left, H_Z_T_mid_mid, H_Z_T_mid_right))

    H_Z_T_bot_left = np.zeros((n_A*r_B*r_C, r_A*n_B*n_C)) # need to flip order for np
    H_Z_T_bot_mid = np.kron(np.kron(np.eye(n_A), np.eye(r_B)), H_C)
    H_Z_T_bot_right = np.kron(np.kron(np.eye(n_A), H_B), np.eye(r_C))
    H_Z_T_bot = np.hstack((H_Z_T_bot_left, H_Z_T_bot_mid, H_Z_T_bot_right))

    H_Z_T = np.vstack((H_Z_T_top, H_Z_T_mid, H_Z_T_bot))
    H_Z = H_Z_T.T

    H_X_left = np.kron(np.kron(np.eye(r_A), np.eye(r_B)), H_C)
    H_X_mid = np.kron(np.kron(np.eye(r_A), H_B), np.eye(r_C))
    H_X_right = np.kron(np.kron(H_A, np.eye(r_B)), np.eye(r_C))
    H_X = np.hstack((H_X_left, H_X_mid, H_X_right))

    return H_X, H_Z

def create_2D_HGP_code(H_A, H_B, compute_distance=False):
    H_X, H_Z = get_HXHZ_2D(H_A, H_B)
    code_2D = css_code(H_X, H_Z)
    if compute_distance:
        code_2D.compute_code_distance()
    return code_2D

def create_3D_HGP_code(H_A, H_B, H_C, compute_distance=False):
    H_X, H_Z = getHXHZ_3D(H_A, H_B, H_C)
    code_3D = css_code(H_X, H_Z)
    if compute_distance:
        code_3D.compute_code_distance()
    return code_3D

def create_homomorphism_from_2D_to_3D_HGP(H_A, H_B, H_C, e_i):
    # input H_A, H_B, H_C of original 3D code
    # assuming we puncture base code A
    # rA_2D, nA_2D = 1, 0
    rA_3D, nA_3D = H_A.shape
    rB, nB = H_B.shape
    rC, nC = H_C.shape
    gamma_A_0 = np.zeros((rA_3D, 1)) # can be arbitrary rx1 but set to [1,0,...,0].T
    gamma_A_0[e_i, 0] = 1
    gamma_A_1 = np.empty((nA_3D, 0))
    gamma_B_0 = np.eye(rB)
    gamma_B_1 = np.eye(nB)
    gamma_C_0 = np.eye(rC) # flip r and n for code C since its check matrix is transposed ?
    gamma_C_1 = np.eye(nC)

    # numpy v 1.21.6 can't handle np.kron with empty arrays, so manually define the empty arrays
    gamma_000 = np.kron(np.kron(gamma_A_0, gamma_B_0), gamma_C_0)

    gamma_001 = np.kron(np.kron(gamma_A_0, gamma_B_0), gamma_C_1)
    gamma_010 = np.kron(np.kron(gamma_A_0, gamma_B_1), gamma_C_0)
    # gamma_100 = np.kron(np.kron(gamma_A_1, gamma_B_0), gamma_C_0)
    gamma_100 = np.empty((nA_3D*rB*rC, 0))

    gamma_011 = np.kron(np.kron(gamma_A_0, gamma_B_1), gamma_C_1)
    # gamma_101 = np.kron(np.kron(gamma_A_1, gamma_B_0), gamma_C_1)
    # gamma_110 = np.kron(np.kron(gamma_A_1, gamma_B_1), gamma_C_0)
    gamma_101 = np.empty((nA_3D*rB*nC, 0))
    gamma_110 = np.empty((nA_3D*nB*rC, 0))

    # gamma_111 = np.kron(np.kron(gamma_A_1, gamma_B_1), gamma_C_1)
    gamma_111 = np.empty((nA_3D*nB*nC, 0))

    gamma_0 = gamma_000
    gamma_1 = block_diag(gamma_001, gamma_010, gamma_100)  # maps x checks
    gamma_2 = block_diag(gamma_011, gamma_101, gamma_110)  # maps qubit space
    gamma_3 = gamma_111  # maps z checks

    return gamma_0, gamma_1, gamma_2, gamma_3

def check_homomorphism_commutes(H3D, H2D, gamma_l, gamma_r):
    left = H3D @ gamma_l % 2
    right = gamma_r @ H2D % 2
    return np.all(left == right)

def zero(R, m, n):
    id_mat = zero_matrix(R, m, n)
    return id_mat

def identity(R, G, n):
    id_mat = zero_matrix(R, n, n)
    for i in range(n):
        id_mat[i, i] = G.identity()
    return id_mat

def block_diag_R(R, *blocks):
    """
    Build a block-diagonal matrix over ring R from the given blocks.

    Args:
        R: Base ring for the output matrix.
        *blocks: Any number of matrices (or coercible arrays) to place on the diagonal.

    Returns:
        A matrix over R whose diagonal blocks are the given inputs, in order.
    """
    # Allow a single iterable of blocks too: block_diag(R, [A,B,C])
    if len(blocks) == 1 and isinstance(blocks[0], (list, tuple)):
        blocks = tuple(blocks[0])
    if not blocks:
        return zero(R, 0, 0)

    # Coerce and precompute sizes
    coerced = []
    total_rows = 0
    total_cols = 0
    for B in blocks:
        try:
            M = matrix(R, B)  # coerce entries into R
        except Exception as e:
            raise TypeError(f"Block {B!r} cannot be coerced to a matrix over R: {e}")
        coerced.append(M)
        total_rows += M.nrows()
        total_cols += M.ncols()

    # Allocate and place blocks
    out = zero(R, total_rows, total_cols)
    r = c = 0
    for M in coerced:
        rr, cc = M.nrows(), M.ncols()
        out[r:r+rr, c:c+cc] = M
        r += rr
        c += cc
    return out

def create_homomorphism_from_2D_to_3D_LP_A(HA, HB, HC, R, G, one, e_i, asbinary=True):
    # input H_A, H_B, H_C of original 3D code
    # assuming we contract base code A
    # rA_2D, nA_2D = 1, 0
    rA_3D, nA_3D = HA.nrows(), HA.ncols()
    rB, nB = HB.nrows(), HB.ncols()
    rC, nC = HC.nrows(), HC.ncols()
    # print(f"rA_3D: {rA_3D}, nA_3D: {nA_3D}, rB: {rB}, nB: {nB}, rC: {rC}, nC: {nC}")
    # rA_3D, nA_3D = rA_3D, nA_3D
    # rB, nB = l*rB, l*nB

    # assume for now that rA_3D = 1, nA_3D = 1
    gamma_A_0 = zero(R, rA_3D, 1) # can be arbitrary rx1 unit vector
    gamma_A_0[e_i, 0] = one
    # zeros = [zero_matrix(R, 1, 1) for _ in range(nA_3D-1)]
    gamma_A_1_mat = np.empty((nA_3D, 0))
    # gamma_A_0 = Matrix(R, gamma_A_0_mat)
    gamma_A_1 = Matrix(R, gamma_A_1_mat)
    gamma_B_0 = identity(R, G, rB)  # identity matrix of size rB
    gamma_B_1 = identity(R, G, nB)
    gamma_C_0 = identity(R, G, rC)
    gamma_C_1 = identity(R, G, nC)

    # numpy v 1.21.6 can't handle np.kron with empty arrays, so manually define the empty arrays
    gamma_000 = gamma_A_0.tensor_product(gamma_B_0).tensor_product(gamma_C_0)
    # gamma_000 = np.kron(np.kron(gamma_A_0, gamma_B_0), gamma_C_0)

    gamma_001 = gamma_A_0.tensor_product(gamma_B_0).tensor_product(gamma_C_1)
    # gamma_001 = np.kron(np.kron(gamma_A_0, gamma_B_0), gamma_C_1)
    gamma_010 = gamma_A_0.tensor_product(gamma_B_1).tensor_product(gamma_C_0)
    # gamma_010 = np.kron(np.kron(gamma_A_0, gamma_B_1), gamma_C_0)
    # gamma_100 = np.kron(np.kron(gamma_A_1, gamma_B_0), gamma_C_0)
    gamma_100_mat = np.empty((nA_3D*rB*rC, 0))
    gamma_100 = Matrix(R, gamma_100_mat)

    gamma_011 = gamma_A_0.tensor_product(gamma_B_1).tensor_product(gamma_C_1)
    # gamma_011 = np.kron(np.kron(gamma_A_0, gamma_B_1), gamma_C_1)
    # gamma_101 = np.kron(np.kron(gamma_A_1, gamma_B_0), gamma_C_1)
    # gamma_110 = np.kron(np.kron(gamma_A_1, gamma_B_1), gamma_C_0)
    gamma_101_mat = np.empty((nA_3D*rB*nC, 0))
    gamma_110_mat = np.empty((nA_3D*nB*rC, 0))
    gamma_101 = Matrix(R, gamma_101_mat)
    gamma_110 = Matrix(R, gamma_110_mat)

    # gamma_111 = np.kron(np.kron(gamma_A_1, gamma_B_1), gamma_C_1)
    gamma_111_mat = np.empty((nA_3D*nB*nC, 0))
    gamma_111 = Matrix(R, gamma_111_mat)

    if asbinary:
        gamma_0 = lift_matrix_over_group_algebra(gamma_000, G.list()) # maps x checks
        gamma_1 = block_diag(
            lift_matrix_over_group_algebra(gamma_001, G.list()), 
            lift_matrix_over_group_algebra(gamma_010, G.list()), 
            lift_matrix_over_group_algebra(gamma_100, G.list()))  # maps qubit space
        gamma_2 = block_diag(
            lift_matrix_over_group_algebra(gamma_011, G.list()), 
            lift_matrix_over_group_algebra(gamma_101, G.list()), 
            lift_matrix_over_group_algebra(gamma_110, G.list()))  # maps z checks
        gamma_3 = lift_matrix_over_group_algebra(gamma_111, G.list())  # maps z checks
    else:
        gamma_0 = gamma_000
        gamma_1 = block_diag_R(R, gamma_001, gamma_010, gamma_100)
        gamma_2 = block_diag_R(R, gamma_011, gamma_101, gamma_110)
        gamma_3 = gamma_111

    return gamma_0, gamma_1, gamma_2, gamma_3

def create_homomorphism_from_2D_to_3D_LP_B(HA, HB, HC, R, G, one, e_i, asbinary=True):
    # input H_A, H_B, H_C of original 3D code
    # assuming we contract base code A
    # rA_2D, nA_2D = 1, 0
    rA, nA = HA.nrows(), HA.ncols()
    rB_3D, nB_3D = HB.nrows(), HB.ncols()
    rC, nC = HC.nrows(), HC.ncols()

    # assume for now that rA_3D = 1, nA_3D = 1
    gamma_B_0 = zero(R, rB_3D, 1) # can be arbitrary rx1 unit vector
    gamma_B_0[e_i, 0] = one
    # zeros = [zero_matrix(R, 1, 1) for _ in range(nA_3D-1)]
    gamma_B_1_mat = np.empty((nB_3D, 0))
    # gamma_A_0 = Matrix(R, gamma_A_0_mat)
    gamma_B_1 = Matrix(R, gamma_B_1_mat)
    gamma_A_0 = identity(R, G, rA)  # identity matrix of size rB
    gamma_A_1 = identity(R, G, nA)
    gamma_C_0 = identity(R, G, rC)
    gamma_C_1 = identity(R, G, nC)

    # numpy v 1.21.6 can't handle np.kron with empty arrays, so manually define the empty arrays
    gamma_000 = gamma_A_0.tensor_product(gamma_B_0).tensor_product(gamma_C_0)
    # gamma_000 = np.kron(np.kron(gamma_A_0, gamma_B_0), gamma_C_0)

    gamma_001 = gamma_A_0.tensor_product(gamma_B_0).tensor_product(gamma_C_1)
    gamma_010 = gamma_A_0.tensor_product(gamma_B_1).tensor_product(gamma_C_0)
    gamma_100 = gamma_A_1.tensor_product(gamma_B_0).tensor_product(gamma_C_0)
    # gamma_010 = np.kron(np.kron(gamma_A_0, gamma_B_1), gamma_C_0)
    # gamma_100 = np.kron(np.kron(gamma_A_1, gamma_B_0), gamma_C_0)
    # gamma_100_mat = np.empty((nA_3D*rB*rC, 0))
    # gamma_100 = Matrix(R, gamma_100_mat)

    gamma_011 = gamma_A_0.tensor_product(gamma_B_1).tensor_product(gamma_C_1)
    gamma_101 = gamma_A_1.tensor_product(gamma_B_0).tensor_product(gamma_C_1)
    gamma_110 = gamma_A_1.tensor_product(gamma_B_1).tensor_product(gamma_C_0)
    # gamma_011 = np.kron(np.kron(gamma_A_0, gamma_B_1), gamma_C_1)
    # gamma_101 = np.kron(np.kron(gamma_A_1, gamma_B_0), gamma_C_1)
    # gamma_110 = np.kron(np.kron(gamma_A_1, gamma_B_1), gamma_C_0)
    # gamma_101_mat = np.empty((nA_3D*rB*nC, 0))
    # gamma_110_mat = np.empty((nA_3D*nB*rC, 0))
    # gamma_101 = Matrix(R, gamma_101_mat)
    # gamma_110 = Matrix(R, gamma_110_mat)

    # gamma_111 = np.kron(np.kron(gamma_A_1, gamma_B_1), gamma_C_1)
    gamma_111_mat = np.empty((nA*nB_3D*nC, 0))
    gamma_111 = Matrix(R, gamma_111_mat)

    if asbinary:
        gamma_0 = lift_matrix_over_group_algebra(gamma_000, G.list()) # maps x checks
        gamma_1 = block_diag(
            lift_matrix_over_group_algebra(gamma_001, G.list()), 
            lift_matrix_over_group_algebra(gamma_010, G.list()), 
            lift_matrix_over_group_algebra(gamma_100, G.list()))  # maps qubit space
        gamma_2 = block_diag(
            lift_matrix_over_group_algebra(gamma_011, G.list()), 
            lift_matrix_over_group_algebra(gamma_101, G.list()), 
            lift_matrix_over_group_algebra(gamma_110, G.list()))  # maps z checks
        gamma_3 = lift_matrix_over_group_algebra(gamma_111, G.list())  # maps z checks
    else:
        gamma_0 = gamma_000
        gamma_1 = block_diag_R(R, gamma_001, gamma_010, gamma_100)
        gamma_2 = block_diag_R(R, gamma_011, gamma_101, gamma_110)
        gamma_3 = gamma_111

    return gamma_0, gamma_1, gamma_2, gamma_3

def create_homomorphism_from_2D_to_3D_LP_C(HA, HB, HC, R, G, one, e_i, asbinary=True):
    # input H_A, H_B, H_C of original 3D code
    # assuming we contract base code A
    # rA_2D, nA_2D = 1, 0
    rA, nA = HA.nrows(), HA.ncols()
    rB, nB = HB.nrows(), HB.ncols()
    rC_3D, nC_3D = HC.nrows(), HC.ncols()

    # assume for now that rA_3D = 1, nA_3D = 1
    gamma_C_0 = zero(R, rC_3D, 1) # can be arbitrary rx1 unit vector
    gamma_C_0[e_i, 0] = one
    # zeros = [zero_matrix(R, 1, 1) for _ in range(nA_3D-1)]
    
    gamma_C_1_mat = np.empty((nC_3D, 0))
    # gamma_A_0 = Matrix(R, gamma_A_0_mat)
    gamma_C_1 = Matrix(R, gamma_C_1_mat)
    gamma_A_0 = identity(R, G, rA)  # identity matrix of size rB
    gamma_A_1 = identity(R, G, nA)
    gamma_B_0 = identity(R, G, rB)
    gamma_B_1 = identity(R, G, nB)

    # numpy v 1.21.6 can't handle np.kron with empty arrays, so manually define the empty arrays
    gamma_000 = gamma_A_0.tensor_product(gamma_B_0).tensor_product(gamma_C_0)
    # gamma_000 = np.kron(np.kron(gamma_A_0, gamma_B_0), gamma_C_0)

    gamma_001 = gamma_A_0.tensor_product(gamma_B_0).tensor_product(gamma_C_1)
    gamma_010 = gamma_A_0.tensor_product(gamma_B_1).tensor_product(gamma_C_0)
    gamma_100 = gamma_A_1.tensor_product(gamma_B_0).tensor_product(gamma_C_0)
    # gamma_010 = np.kron(np.kron(gamma_A_0, gamma_B_1), gamma_C_0)
    # gamma_100 = np.kron(np.kron(gamma_A_1, gamma_B_0), gamma_C_0)
    # gamma_100_mat = np.empty((nA_3D*rB*rC, 0))
    # gamma_100 = Matrix(R, gamma_100_mat)

    gamma_011 = gamma_A_0.tensor_product(gamma_B_1).tensor_product(gamma_C_1)
    gamma_101 = gamma_A_1.tensor_product(gamma_B_0).tensor_product(gamma_C_1)
    gamma_110 = gamma_A_1.tensor_product(gamma_B_1).tensor_product(gamma_C_0)
    # gamma_011 = np.kron(np.kron(gamma_A_0, gamma_B_1), gamma_C_1)
    # gamma_101 = np.kron(np.kron(gamma_A_1, gamma_B_0), gamma_C_1)
    # gamma_110 = np.kron(np.kron(gamma_A_1, gamma_B_1), gamma_C_0)
    # gamma_101_mat = np.empty((nA_3D*rB*nC, 0))
    # gamma_110_mat = np.empty((nA_3D*nB*rC, 0))
    # gamma_101 = Matrix(R, gamma_101_mat)
    # gamma_110 = Matrix(R, gamma_110_mat)

    # gamma_111 = np.kron(np.kron(gamma_A_1, gamma_B_1), gamma_C_1)
    gamma_111_mat = np.empty((nA*nB*nC_3D, 0))
    gamma_111 = Matrix(R, gamma_111_mat)

    if asbinary:
        gamma_0 = lift_matrix_over_group_algebra(gamma_000, G.list()) # maps x checks
        gamma_1 = block_diag(
            lift_matrix_over_group_algebra(gamma_001, G.list()), 
            lift_matrix_over_group_algebra(gamma_010, G.list()), 
            lift_matrix_over_group_algebra(gamma_100, G.list()))  # maps qubit space
        gamma_2 = block_diag(
            lift_matrix_over_group_algebra(gamma_011, G.list()), 
            lift_matrix_over_group_algebra(gamma_101, G.list()), 
            lift_matrix_over_group_algebra(gamma_110, G.list()))  # maps z checks
        gamma_3 = lift_matrix_over_group_algebra(gamma_111, G.list())  # maps z checks
    else:
        gamma_0 = gamma_000
        gamma_1 = block_diag_R(R, gamma_001, gamma_010, gamma_100)
        gamma_2 = block_diag_R(R, gamma_011, gamma_101, gamma_110)
        gamma_3 = gamma_111

    return gamma_0, gamma_1, gamma_2, gamma_3