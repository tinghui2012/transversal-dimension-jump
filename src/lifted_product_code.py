import numpy as np
from f2poly_matrix import F2PolyMatrix, F2BiPolyMatrix
from bposd.css import css_code
# import re

class LiftedProductCode:
    def __init__(self, *base_matrices):
        if len(base_matrices) not in [2, 3]:
            raise ValueError("LiftedProductCode requires 2 or 3 F2PolyMatrix inputs.")

        # Check all L values match
        L_values = [mat.L for mat in base_matrices]
        if len(set(L_values)) != 1:
            raise ValueError("All F2PolyMatrix inputs must have the same L value.")

        self.L = L_values[0]

        if len(base_matrices) == 2:
            self.H2, self.H3 = base_matrices
            self.code = self._construct_2d()
        else:
            self.H1, self.H2, self.H3 = base_matrices
            self.code = self._construct_3d()

    def _identity(self, n):
        mat = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                mat[i, j] = [1] + [0] * (self.L - 1) if i == j else [0] * self.L
        return F2PolyMatrix.from_coeff_matrix(mat, self.L)

    def _zero(self, m, n):
        mat = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                mat[i, j] = [0] * self.L
        return F2PolyMatrix.from_coeff_matrix(mat, self.L)

    def _construct_2d(self):
        H2, H3 = self.H2, self.H3
        I_r2 = self._identity(H2.rows)
        I_r3 = self._identity(H3.rows)
        I_n2 = self._identity(H2.cols)
        I_n3 = self._identity(H3.cols)

        Hx = I_r2.kron(H3).matrix
        Hx2 = H2.kron(I_r3).matrix
        Hz1 = H2.transpose_reverse().kron(I_n3).matrix
        Hz2 = I_n2.kron(H3.transpose_reverse()).matrix

        Hx_mat = F2PolyMatrix.from_coeff_matrix(np.hstack([Hx, Hx2]), self.L)
        Hz_mat = F2PolyMatrix.from_coeff_matrix(np.hstack([Hz1, Hz2]), self.L)

        return css_code(hx=Hx_mat.lift_to_binary(), hz=Hz_mat.lift_to_binary())

    def _construct_3d(self):
        H1, H2, H3 = self.H1, self.H2, self.H3
        r1, n1 = H1.rows, H1.cols
        r2, n2 = H2.rows, H2.cols
        r3, n3 = H3.rows, H3.cols

        I_r1 = self._identity(r1)
        I_r2 = self._identity(r2)
        I_r3 = self._identity(r3)
        I_n1 = self._identity(n1)
        I_n2 = self._identity(n2)
        I_n3 = self._identity(n3)

        term1 = I_r1.kron(I_r2).kron(H3).matrix
        term2 = I_r1.kron(H2).kron(I_r3).matrix
        term3 = H1.kron(I_r2).kron(I_r3).matrix
        Hx_mat = F2PolyMatrix.from_coeff_matrix(np.hstack([term1, term2, term3]), self.L)

        R1 = np.hstack([
            I_r1.kron(H2).kron(I_n3).matrix,
            H1.kron(I_r2).kron(I_n3).matrix,
            self._zero(r1 * r2 * n3, n1 * n2 * r3).matrix
        ])
        R2 = np.hstack([
            I_r1.kron(I_n2).kron(H3).matrix,
            self._zero(r1 * n2 * r3, n1 * r2 * n3).matrix,
            H1.kron(I_n2).kron(I_r3).matrix
        ])
        R3 = np.hstack([
            self._zero(n1 * r2 * r3, r1 * n2 * n3).matrix,
            I_n1.kron(I_r2).kron(H3).matrix,
            I_n1.kron(H2).kron(I_r3).matrix
        ])
        Hz_mat = F2PolyMatrix.from_coeff_matrix(np.vstack([R1, R2, R3]), self.L).transpose_reverse()

        return css_code(hx=Hx_mat.lift_to_binary(), hz=Hz_mat.lift_to_binary())




class LiftedProductCode_BB:
    def __init__(self, *base_matrices):
        if len(base_matrices) not in [2, 3]:
            raise ValueError("LiftedProductCode requires 2 or 3 F2PolyMatrix inputs.")

        # Check all L values match
        L_values = [mat.L for mat in base_matrices]
        if len(set(L_values)) != 1:
            raise ValueError("All F2PolyMatrix inputs must have the same L value.")

        self.L = L_values[0]

        if len(base_matrices) == 2:
            self.H2, self.H3 = base_matrices
            self.code = self._construct_2d()
        else:
            self.H1, self.H2, self.H3 = base_matrices
            self.code = self._construct_3d()

    def _identity(self, n):
        string_mat = np.empty((n, n), dtype='<U9')
        for i in range(n):
            for j in range(n):
                string_mat[i, j] = '1' if i == j else '0'
        return F2BiPolyMatrix(string_mat, self.L)

    def _zero(self, m, n):
        string_mat = np.empty((m, n), dtype='<U9')
        for i in range(m):
            for j in range(n):
                string_mat[i, j] = '0'
        return F2BiPolyMatrix(string_mat, self.L)

    def _construct_2d(self):
        H2, H3 = self.H2, self.H3
        I_r2 = self._identity(H2.rows)
        I_r3 = self._identity(H3.rows)
        I_n2 = self._identity(H2.cols)
        I_n3 = self._identity(H3.cols)
        
        Hx1 = I_r2.kron(H3).string_matrix
        Hx2 = H2.kron(I_r3).string_matrix
        Hz1 = H2.transpose_reverse().kron(I_n3).string_matrix
        Hz2 = I_n2.kron(H3.transpose_reverse()).string_matrix
        
        Hx = F2BiPolyMatrix(np.hstack([Hx1, Hx2]), self.L)
        Hz = F2BiPolyMatrix(np.hstack([Hz1, Hz2]), self.L)

        return css_code(hx=Hx.lift_to_binary(), hz=Hz.lift_to_binary())

    def _construct_3d(self):
        H1, H2, H3 = self.H1, self.H2, self.H3
        r1, n1 = H1.rows, H1.cols
        r2, n2 = H2.rows, H2.cols
        r3, n3 = H3.rows, H3.cols

        I_r1 = self._identity(r1)
        I_r2 = self._identity(r2)
        I_r3 = self._identity(r3)
        I_n1 = self._identity(n1)
        I_n2 = self._identity(n2)
        I_n3 = self._identity(n3)

        term1 = I_r1.kron(I_r2).kron(H3).string_matrix
        term2 = I_r1.kron(H2).kron(I_r3).string_matrix
        term3 = H1.kron(I_r2).kron(I_r3).string_matrix
        Hx_mat = F2BiPolyMatrix(np.hstack([term1, term2, term3]), self.L)

        R1 = np.hstack([
            I_r1.kron(H2).kron(I_n3).string_matrix,
            H1.kron(I_r2).kron(I_n3).string_matrix,
            self._zero(r1 * r2 * n3, n1 * n2 * r3).string_matrix
        ])
        R2 = np.hstack([
            I_r1.kron(I_n2).kron(H3).string_matrix,
            self._zero(r1 * n2 * r3, n1 * r2 * n3).string_matrix,
            H1.kron(I_n2).kron(I_r3).string_matrix
        ])
        R3 = np.hstack([
            self._zero(n1 * r2 * r3, r1 * n2 * n3).string_matrix,
            I_n1.kron(I_r2).kron(H3).string_matrix,
            I_n1.kron(H2).kron(I_r3).string_matrix
        ])
        Hz_mat = F2BiPolyMatrix(np.vstack([R1, R2, R3]), self.L).transpose_reverse()

        return css_code(hx=Hx_mat.lift_to_binary(), hz=Hz_mat.lift_to_binary())