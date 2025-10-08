import numpy as np
from bposd.css import css_code
from sage.all import *
import sys
sys.path.append('./src/')
from QuantumExanderCodesGene import GetClassicalCodeParams
from DistanceEst import DistanceEst_BPOSD
gap('LoadPackage("QDistRnd");')
from scipy import sparse
import scipy.io as sio
import re
import copy

# def regular_rep_matrix(g, basis):
#     """
#     Return the regular representation matrix of g acting on group elements via left multiplication.
    
#     Arguments:
#     - g: a group element or zero element in the group algebra
#     - basis: list of group elements (the basis of the group algebra)

#     Returns:
#     - Matrix over QQ representing the action of g
#     """
#     n = len(basis)
#     M = matrix(QQ, n)

#     # If g is zero (in the group algebra), return zero matrix
#     if hasattr(g, "is_zero") and g.is_zero():
#         return M

#     # If g is a group algebra element but not zero, raise error
#     if hasattr(g, "support") and len(g.support()) > 1:
#         raise ValueError("g is not a single group element")

#     # Handle group element or 1-term group algebra element
#     if hasattr(g, "support"):
#         # Extract the single group element from the algebra
#         g = list(g.support())[0]

#     for i, h in enumerate(basis):
#         gh = g * h
#         j = basis.index(gh)
#         M[j, i] = 1

#     return M

def regular_rep_matrix(g, basis, field=QQ):
    """
    Return the regular representation matrix of a group algebra element g
    acting on the group algebra via left multiplication.

    Arguments:
    - g: an element of the group algebra
    - basis: list of group elements forming a basis of the group algebra
    - field: field over which to define the matrix (default QQ)

    Returns:
    - Matrix over `field` representing the action of g by left multiplication
    """
    n = len(basis)
    M = matrix(field, n)

    # If g is zero in the group algebra
    if hasattr(g, "is_zero") and g.is_zero():
        return M

    # Build a lookup from group element to index in basis
    basis_index = {h: i for i, h in enumerate(basis)}

    for col, h in enumerate(basis):
        # Compute g * h (result is in group algebra)
        prod = g * h
        for term in prod.support():
            coeff = prod.coefficient(term)
            row = basis_index[term]
            M[row, col] = field(coeff)

    return M


def lift_matrix_over_group_algebra(M, basis):
    nrows, ncols = M.nrows(), M.ncols()
    n = len(basis)
    big_mat = matrix(QQ, n * nrows, n * ncols)
    
    for i in range(nrows):
        for j in range(ncols):
            block = regular_rep_matrix(M[i,j], basis)
            # Insert block at position (i*n, j*n)
            for bi in range(n):
                for bj in range(n):
                    big_mat[i*n + bi, j*n + bj] = block[bi, bj]
    return np.array(big_mat)

def ga_transpose_reverse(M):
    A = M.base_ring()
    G = A.group()
    def involution(x):
        return A(sum(c * A(g**(-1)) for g, c in x))
    return M.apply_map(involution).transpose()

def ga_reverse(M):
    A = M.base_ring()
    G = A.group()
    def involution(x):
        return A(sum(c * A(g**(-1)) for g, c in x))
    return M.apply_map(involution)

def extract_last_two_digits(s):
    # Find all digits in the string
    digits = re.findall(r'\d', s)
    
    # Take the last two digits (or fewer if less than 2 digits)
    last_two = digits[-2:] if len(digits) >= 2 else digits
    
    # Convert to integers
    return list(map(int, last_two))





class BalancedProductCode:
    def __init__(self, *base_matrices):
        if len(base_matrices) not in [2, 3]:
            raise ValueError("LiftedProductCode requires 2 or 3 F2PolyMatrix inputs.")

        # # Check all base rings match
        Rs = [mat.base_ring() for mat in base_matrices]
        if len(set(Rs)) != 1:
            raise ValueError("All input mats must be defined on the same ring R.")
        self.R = Rs[0]
        self.G = self.R.group()
        
        
        if len(base_matrices) == 2:
            self.H2, self.H3 = base_matrices
            # self.code = self._construct_2d()
            self.Hx, self.Hz_T = self._construct_2d()
        else:
            self.H1, self.H2, self.H3 = base_matrices
            # self.code = self._construct_3d()
            self.Hx, self.Hz_T, self.Mz = self._construct_3d()

        Hx_lifted = lift_matrix_over_group_algebra(self.Hx, self.G.list())
        Hz_lifted = lift_matrix_over_group_algebra(self.Hz_T, self.G.list()).T
        self.Mz_lifted = lift_matrix_over_group_algebra(self.Mz, self.G.list())
        self.code = css_code(hx=Hx_lifted, hz=Hz_lifted)

    def _identity(self, n):
        id_mat = zero_matrix(self.R, n, n)
        for i in range(n):
                id_mat[i, i] = self.G.identity()
        return id_mat

    def _zero(self, m, n):
        id_mat = zero_matrix(self.R, m, n)
        return id_mat

    def _construct_2d(self):
        H2, H3 = self.H2, self.H3
        I_r2 = self._identity(H2.nrows())
        I_r3 = self._identity(H3.nrows())
        I_n2 = self._identity(H2.ncols())
        I_n3 = self._identity(H3.ncols())

        # print(I_r2.base_ring().group(), H3.base_ring().group())
        Hx1 = I_r2.tensor_product(H3)
        Hx2 = H2.tensor_product(I_r3)
        # Hz1 = (ga_transpose_reverse(H2)).tensor_product(I_n3)
        # Hz2 = I_n2.tensor_product(ga_transpose_reverse(H3))
        Hz1 = H2.tensor_product(I_n3)
        Hz2 = I_n2.tensor_product(H3)
        
        # Hx = F2BiPolyMatrix(np.hstack([Hx1, Hx2]), self.L)
        # Hz = F2BiPolyMatrix(np.hstack([Hz1, Hz2]), self.L)

        Hx = block_matrix([[Hx1, Hx2]])
        Hz_T = block_matrix([[Hz1], [Hz2]])

        # return css_code(hx=Hx.lift_to_binary(), hz=Hz.lift_to_binary())
        return Hx, Hz_T

    def _construct_3d(self):
        H1, H2, H3 = self.H1, self.H2, self.H3
        r1, n1 = H1.nrows(), H1.ncols()
        r2, n2 = H2.nrows(), H2.ncols()
        r3, n3 = H3.nrows(), H3.ncols()

        I_r1 = self._identity(r1)
        I_r2 = self._identity(r2)
        I_r3 = self._identity(r3)
        I_n1 = self._identity(n1)
        I_n2 = self._identity(n2)
        I_n3 = self._identity(n3)

        term1 = I_r1.tensor_product(I_r2).tensor_product(H3)
        term2 = I_r1.tensor_product(H2).tensor_product(I_r3)
        term3 = H1.tensor_product(I_r2).tensor_product(I_r3)
        Hx_mat = block_matrix([[term1, term2, term3]])

        R1 = block_matrix([[
            I_r1.tensor_product(H2).tensor_product(I_n3),
            H1.tensor_product(I_r2).tensor_product(I_n3),
            self._zero(r1 * r2 * n3, n1 * n2 * r3)]
        ])
        R2 = block_matrix([[
            I_r1.tensor_product(I_n2).tensor_product(H3),
            self._zero(r1 * n2 * r3, n1 * r2 * n3),
            H1.tensor_product(I_n2).tensor_product(I_r3)]
        ])
        R3 = block_matrix([[
            self._zero(n1 * r2 * r3, r1 * n2 * n3),
            I_n1.tensor_product(I_r2).tensor_product(H3),
            I_n1.tensor_product(H2).tensor_product(I_r3)]
        ])
        # Hz_mat = ga_transpose_reverse(block_matrix([[R1], [R2], [R3]]))
        Hz_mat_T = block_matrix([[R1], [R2], [R3]])

        # meta Z check
        Mz_mat = block_matrix([[H1.tensor_product(I_n2).tensor_product(I_n3)], [I_n1.tensor_product(H2).tensor_product(I_n3)], [I_n1.tensor_product(I_n2).tensor_product(H3)]])

        # return css_code(hx=Hx_mat.lift_to_binary(), hz=Hz_mat.lift_to_binary())
        return Hx_mat, Hz_mat_T, Mz_mat


def DistanceEst_Gap(eval_code, trials):
    hx = copy.deepcopy(eval_code.hx.astype('int').toarray())
    hz = copy.deepcopy(eval_code.hz.astype('int').toarray())
    
    F = GF(2)
    Hz = Matrix(F, hz)
    Hx = Matrix(F, hx)
    
    gap_Hz = gap(Hz)
    gap_Hx = gap(Hx)
    gap_trial = gap(trials)

    # dZ = int(gap.eval("DistRandCSS(%s, %s, 50, 2, 2 : field := GF(2));" % (gap_Hz.name(), gap_Hx.name())))
    # dX = int(gap.eval("DistRandCSS(%s, %s, 50, 2, 2 : field := GF(2));" % (gap_Hx.name(), gap_Hz.name())))
    dZ = int(gap.eval("DistRandCSS(%s, %s, %s, 2, 2 : field := GF(2));" % (gap_Hz.name(), gap_Hx.name(), gap_trial.name())))
    dX = int(gap.eval("DistRandCSS(%s, %s, %s, 2, 2 : field := GF(2));" % (gap_Hx.name(), gap_Hz.name(), gap_trial.name())))

    return dX, dZ