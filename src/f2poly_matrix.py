import numpy as np
import re

class F2PolyMatrix:
    def __init__(self, string_matrix, L):
        self.L = L
        self.rows = len(string_matrix)
        self.cols = len(string_matrix[0])
        self.matrix = np.empty((self.rows, self.cols), dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i, j] = self.parse_poly_string(string_matrix[i][j])
        self.string_matrix = string_matrix

    def parse_poly_string(self, s):
        coeffs = [0] * self.L
        s = s.replace(' ', '')
        if s == '0' or s == '':
            return coeffs
        for term in s.split('+'):
            if term == '1':
                coeffs[0] = 1
            elif term == 'x':
                coeffs[1] = 1
            elif term.startswith('x^'):
                power = int(term[2:])
                coeffs[power % self.L] = 1
            else:
                raise ValueError(f"Unrecognized term: '{term}'")
        return coeffs

    def poly_list_to_string(self, coeffs):
        terms = []
        for i, coeff in enumerate(coeffs):
            if coeff:
                if i == 0:
                    terms.append("1")
                elif i == 1:
                    terms.append("x")
                else:
                    terms.append(f"x^{i}")
        return "+".join(terms) if terms else "0"

    def __str__(self):
        string_matrix = [[self.poly_list_to_string(self.matrix[i, j]) for j in range(self.cols)]
                         for i in range(self.rows)]
        col_widths = [max(len(string_matrix[i][j]) for i in range(self.rows)) for j in range(self.cols)]
        lines = []
        for i in range(self.rows):
            row = [string_matrix[i][j].ljust(col_widths[j]) for j in range(self.cols)]
            lines.append("  ".join(row))
        return "\n".join(lines)

    def add_poly(self, a, b):
        return [(a[i] ^ b[i]) for i in range(self.L)]

    def poly_mult_mod(self, a, b):
        result = [0] * self.L
        for i in range(len(a)):
            for j in range(len(b)):
                result[(i + j) % self.L] ^= a[i] & b[j]
        return result

    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Incompatible matrix dimensions.")
        result = np.empty((self.rows, other.cols), dtype=object)
        for i in range(self.rows):
            for j in range(other.cols):
                sum_poly = [0] * self.L
                for k in range(self.cols):
                    a = self.matrix[i, k]
                    b = other.matrix[k, j]
                    prod = self.poly_mult_mod(a, b)
                    sum_poly = self.add_poly(sum_poly, prod)
                result[i, j] = sum_poly
        return F2PolyMatrix.from_coeff_matrix(result, self.L)

    @staticmethod
    def from_coeff_matrix(coeff_matrix, L):
        obj = F2PolyMatrix([['0'] * coeff_matrix.shape[1]] * coeff_matrix.shape[0], L)
        obj.matrix = coeff_matrix
        obj.string_matrix = obj.to_string_matrix()
        return obj

    def kron(self, other):
        A, B = self.matrix, other.matrix
        m, n = A.shape
        p, q = B.shape
        result = np.empty((m * p, n * q), dtype=object)
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    for f in range(q):
                        row = i * p + k
                        col = j * q + f
                        a = A[i, j]
                        b = B[k, f]
                        conv = np.convolve(a, b) % 2
                        folded = [0] * self.L
                        for d in range(len(conv)):
                            folded[d % self.L] ^= conv[d]
                        result[row, col] = folded
        return F2PolyMatrix.from_coeff_matrix(result, self.L)

    def transpose_reverse(self):
        A_T = self.matrix.T
        m, n = A_T.shape
        result = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                poly = A_T[i, j]
                reversed_poly = [0] * self.L
                for deg, coeff in enumerate(poly):
                    if coeff:
                        reversed_poly[-deg % self.L] = 1
                result[i, j] = reversed_poly
        return F2PolyMatrix.from_coeff_matrix(result, self.L)

    def circulant_shift_matrix(self):
        S = np.zeros((self.L, self.L), dtype=int)
        for i in range(self.L):
            S[i][(i + 1) % self.L] = 1
        return S

    def lift_to_binary(self):
        m, n = self.matrix.shape
        S = self.circulant_shift_matrix()
        lifted = np.zeros((m * self.L, n * self.L), dtype=int)
        for i in range(m):
            for j in range(n):
                poly = self.matrix[i, j]
                block = np.zeros((self.L, self.L), dtype=int)
                for k, coeff in enumerate(poly):
                    if coeff:
                        block = (block + np.linalg.matrix_power(S, k)) % 2
                lifted[i * self.L:(i + 1) * self.L, j * self.L:(j + 1) * self.L] = block
        return lifted

    def to_string_matrix(self):
        m, n = self.matrix.shape
        return np.array([[self.poly_list_to_string(self.matrix[i, j]) for j in range(n)] for i in range(m)])

    def vec_to_set(self):
        assert self.string_matrix.shape[1] == 1
        str_mat = self.string_matrix
        l = self.L

        s_set = []
        for i in range(len(str_mat)):
            s_i = str_mat[i][0]
            if s_i != '0':
                m_i = self.parse_poly_string(s_i)

                ixs = np.where(np.array(m_i) == 1)[0]
                for ix in ixs:
                    n_list = [0]*len(m_i)
                    n_list[ix] = 1
                    n_str = self.poly_list_to_string(n_list)

                    n_str_mat = np.array([['0']]*len(str_mat), dtype='<U10')
                    n_str_mat[i] = [n_str]

                    n_vec = F2PolyMatrix(n_str_mat, l)
                    s_set.append(n_vec)
        return s_set


    def __eq__(self, v1):
        assert isinstance(v1, F2PolyMatrix)
        return (self.matrix == v1.matrix).all()




class F2BiPolyMatrix:
    def __init__(self, string_matrix, L):
        self.L = L
        self.rows = len(string_matrix)
        self.cols = len(string_matrix[0])
        self.matrix = np.empty((self.rows, self.cols), dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i, j] = self.parse_poly_string(string_matrix[i][j])
        self.string_matrix = string_matrix

    def parse_poly_string(self, s):
        coeffs = np.zeros((self.L, self.L), dtype=int)
        s = s.replace(' ', '')
        if s == '0' or s == '':
            return coeffs

        terms = s.split('+')
        for term in terms:
            if term == '1':
                coeffs[0, 0] = 1
            else:
                x_deg, y_deg = 0, 0
                matches = re.findall(r'(x(\^\d+)?)|(y(\^\d+)?)', term)
                for m in matches:
                    if m[0].startswith('x'):
                        x_deg = int(m[1][1:]) if m[1] else 1
                    elif m[2].startswith('y'):
                        y_deg = int(m[3][1:]) if m[3] else 1
                coeffs[x_deg % self.L, y_deg % self.L] = 1
        return coeffs

    def poly_array_to_string(self, coeffs):
        terms = []
        for i in range(self.L):
            for j in range(self.L):
                if coeffs[i, j]:
                    term = []
                    if i == 0 and j == 0:
                        terms.append('1')
                        continue
                    if i == 1:
                        term.append('x')
                    elif i > 1:
                        term.append(f'x^{i}')
                    if j == 1:
                        term.append('y')
                    elif j > 1:
                        term.append(f'y^{j}')
                    terms.append('*'.join(term))
        return '+'.join(terms) if terms else '0'

    def __str__(self):
        string_matrix = [[self.poly_array_to_string(self.matrix[i, j]) for j in range(self.cols)]
                         for i in range(self.rows)]
        col_widths = [max(len(string_matrix[i][j]) for i in range(self.rows)) for j in range(self.cols)]
        lines = []
        for i in range(self.rows):
            row = [string_matrix[i][j].ljust(col_widths[j]) for j in range(self.cols)]
            lines.append("  ".join(row))
        return "\n".join(lines)

    def add_poly(self, a, b):
        return (a ^ b) % 2

    def poly_mult_mod(self, a, b):
        conv = np.zeros((2 * self.L - 1, 2 * self.L - 1), dtype=int)
        for i in range(self.L):
            for j in range(self.L):
                if a[i, j]:
                    for k in range(self.L):
                        for l in range(self.L):
                            if b[k, l]:
                                conv[i + k, j + l] ^= 1
        # Modular reduction: fold degrees mod L
        result = np.zeros((self.L, self.L), dtype=int)
        for i in range(conv.shape[0]):
            for j in range(conv.shape[1]):
                result[i % self.L, j % self.L] ^= conv[i, j]
        return result

    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Incompatible matrix dimensions.")
        result = np.empty((self.rows, other.cols), dtype=object)
        for i in range(self.rows):
            for j in range(other.cols):
                sum_poly = np.zeros((self.L, self.L), dtype=int)
                for k in range(self.cols):
                    a = self.matrix[i, k]
                    b = other.matrix[k, j]
                    prod = self.poly_mult_mod(a, b)
                    sum_poly = self.add_poly(sum_poly, prod)
                result[i, j] = sum_poly
        return F2BiPolyMatrix.from_coeff_matrix(result, self.L)

    @staticmethod
    def from_coeff_matrix(coeff_matrix, L):
        dummy = [['0'] * coeff_matrix.shape[1]] * coeff_matrix.shape[0]
        obj = F2BiPolyMatrix(dummy, L)
        obj.matrix = coeff_matrix
        obj.string_matrix = obj.to_string_matrix()
        return obj

    def to_string_matrix(self):
        return np.array([[self.poly_array_to_string(self.matrix[i, j]) for j in range(self.cols)]
                         for i in range(self.rows)])
    
    def circulant_shift_matrix(self):
        S = np.zeros((self.L, self.L), dtype=int)
        for i in range(self.L):
            S[i, (i + 1) % self.L] = 1
        return S
        
    def lift_to_binary(self):
        S = self.circulant_shift_matrix()
        I = np.eye(self.L, dtype=int)

        # Substitution: x ↦ X = S ⊗ I, y ↦ Y = I ⊗ S
        X = np.kron(S, I)
        Y = np.kron(I, S)

        m, n = self.rows, self.cols
        lifted = np.zeros((m * self.L**2, n * self.L**2), dtype=int)

        for i in range(m):
            for j in range(n):
                coeffs = self.matrix[i, j]
                poly_block = np.zeros((self.L**2, self.L**2), dtype=int)
                for xi in range(self.L):
                    for yj in range(self.L):
                        if coeffs[xi, yj]:
                            term = np.linalg.matrix_power(X, xi) @ np.linalg.matrix_power(Y, yj)
                            poly_block ^= term  # XOR for addition mod 2
                row_start = i * self.L**2
                col_start = j * self.L**2
                lifted[row_start:row_start + self.L**2, col_start:col_start + self.L**2] = poly_block

        return lifted
    
    def transpose_reverse(self):
        A_T = self.matrix.T
        m, n = A_T.shape
        result = np.empty((m, n), dtype=object)
        for i in range(m):
            for j in range(n):
                poly = A_T[i, j]
                reversed_poly = np.zeros_like(poly)
                for xi in range(self.L):
                    for yj in range(self.L):
                        if poly[xi, yj]:
                            rev_x = (-xi) % self.L
                            rev_y = (-yj) % self.L
                            reversed_poly[rev_x, rev_y] = 1
                result[i, j] = reversed_poly
        return F2BiPolyMatrix.from_coeff_matrix(result, self.L)

    def transpose(self):
        str_mat = self.string_matrix.T
        return F2BiPolyMatrix(str_mat, self.L)

    def reverse(self):
        return (self.transpose_reverse()).transpose()
    
    
    def kron(self, other):
        A, B = self.matrix, other.matrix
        m, n = A.shape
        p, q = B.shape
        result = np.empty((m * p, n * q), dtype=object)
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    for l in range(q):
                        row = i * p + k
                        col = j * q + l
                        a = A[i, j]
                        b = B[k, l]
                        prod = self.poly_mult_mod(a, b)
                        result[row, col] = prod
        return F2BiPolyMatrix.from_coeff_matrix(result, self.L)
    
    
    def vec_to_set(self):
        monomial_set = []
        for i in range(self.rows):
            for j in range(self.cols):
                coeffs = self.matrix[i, j]
                for xi in range(self.L):
                    for yj in range(self.L):
                        if coeffs[xi, yj]:
                            # Create a zero polynomial matrix
                            mono_mat = np.empty((self.rows, self.cols), dtype=object)
                            for r in range(self.rows):
                                for c in range(self.cols):
                                    mono_mat[r, c] = np.zeros((self.L, self.L), dtype=int)
                            # Set the (i,j)-entry to x^xi * y^yj
                            mono_entry = np.zeros((self.L, self.L), dtype=int)
                            mono_entry[xi, yj] = 1
                            mono_mat[i, j] = mono_entry
                            # Wrap into F2BiPolyMatrix and store
                            monomial_set.append(F2BiPolyMatrix.from_coeff_matrix(mono_mat, self.L))
        return monomial_set
    
    def __eq__(self, v1):
        assert isinstance(v1, F2BiPolyMatrix)
        return (self.lift_to_binary() == v1.lift_to_binary()).all()
