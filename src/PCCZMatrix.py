import numpy as np
from BP_codes_sage import (regular_rep_matrix, 
    lift_matrix_over_group_algebra, 
    BalancedProductCode, 
    DistanceEst_Gap, 
    ga_transpose_reverse
)

# get map from (alpha, ix) to qubit_ix
def MonomialToQIx(alpha, b_ix, n0, G_basis):
    mat = regular_rep_matrix(alpha, G_basis)
    m_ix = np.where(np.array(mat[:, 0]) == 1)[0][0]
    return b_ix*n0 + m_ix

def generate_triples(orientation_mats_a, orientation_mats_b, orientation_mats_c):
    triples = []

    alpha_1, b_ix1 = orientation_mats_c['delta_in'][0][0], 0
    alpha_2, b_ix2 = orientation_mats_b['delta_out'][0][0], 1
    alpha_3, b_ix3 = orientation_mats_a['delta_out'][0][0]*orientation_mats_b['delta_out'][0][0]*ga_transpose_reverse(orientation_mats_b['delta_in'])[0][0], 2
    triples.append([(alpha_1, b_ix1), (alpha_2, b_ix2), (alpha_3, b_ix3)])

    alpha_1, b_ix1 = orientation_mats_b['delta_in'][0][0], 1
    alpha_2, b_ix2 = orientation_mats_a['delta_out'][0][0], 2
    alpha_3, b_ix3 = orientation_mats_c['delta_out'][0][0]*orientation_mats_a['delta_out'][0][0]*ga_transpose_reverse(orientation_mats_a['delta_in'])[0][0], 0
    triples.append([(alpha_1, b_ix1), (alpha_2, b_ix2), (alpha_3, b_ix3)])

    alpha_1, b_ix1 = orientation_mats_a['delta_in'][0][0], 2
    alpha_2, b_ix2 = orientation_mats_c['delta_out'][0][0], 0
    alpha_3, b_ix3 = orientation_mats_b['delta_out'][0][0]*orientation_mats_c['delta_out'][0][0]*ga_transpose_reverse(orientation_mats_c['delta_in'])[0][0], 1
    triples.append([(alpha_1, b_ix1), (alpha_2, b_ix2), (alpha_3, b_ix3)])

    alpha_1, b_ix1 = orientation_mats_c['delta_in'][0][0], 0
    alpha_2, b_ix2 = orientation_mats_a['delta_out'][0][0], 2
    alpha_3, b_ix3 = orientation_mats_a['delta_out'][0][0]*ga_transpose_reverse(orientation_mats_a['delta_in'])[0][0]*orientation_mats_b['delta_out'][0][0], 1
    triples.append([(alpha_1, b_ix1), (alpha_2, b_ix2), (alpha_3, b_ix3)])

    alpha_1, b_ix1 = orientation_mats_b['delta_in'][0][0], 1
    alpha_2, b_ix2 = orientation_mats_c['delta_out'][0][0], 0
    alpha_3, b_ix3 = orientation_mats_c['delta_out'][0][0]*ga_transpose_reverse(orientation_mats_c['delta_in'])[0][0]*orientation_mats_a['delta_out'][0][0], 2
    triples.append([(alpha_1, b_ix1), (alpha_2, b_ix2), (alpha_3, b_ix3)])

    alpha_1, b_ix1 = orientation_mats_a['delta_in'][0][0], 2
    alpha_2, b_ix2 = orientation_mats_b['delta_out'][0][0], 1
    alpha_3, b_ix3 = orientation_mats_b['delta_out'][0][0]*ga_transpose_reverse(orientation_mats_b['delta_in'])[0][0]*orientation_mats_c['delta_out'][0][0], 0
    triples.append([(alpha_1, b_ix1), (alpha_2, b_ix2), (alpha_3, b_ix3)])
    
    return triples

def generate_PCCZ_bivariate(n, R, x, y, lx, ly, triples, MonomialToQIx):
    P_CCZ_mat = np.zeros([n, n, n])
    for i in range(lx):
        for j in range(ly):
            shift = R(x**i*y**j)
            for triple in triples:
                (alpha_1, b_ix1), (alpha_2, b_ix2), (alpha_3, b_ix3) = triple
                q1_ix = MonomialToQIx(shift*alpha_1, b_ix1)
                q2_ix = MonomialToQIx(shift*alpha_2, b_ix2)
                q3_ix = MonomialToQIx(shift*alpha_3, b_ix3)
                P_CCZ_mat[q1_ix, q2_ix, q3_ix] = 1
    return P_CCZ_mat

def generate_PCCZ_trivariate(n, R, x, y, z, lx, ly, lz, triples, MonomialToQIx):
    P_CCZ_mat = np.zeros([n, n, n])
    for i in range(lx):
        for j in range(ly):
            for k in range(lz):
                shift = R(x**i*y**j*z**k)
                for triple in triples:
                    (alpha_1, b_ix1), (alpha_2, b_ix2), (alpha_3, b_ix3) = triple
                    q1_ix = MonomialToQIx(shift*alpha_1, b_ix1)
                    q2_ix = MonomialToQIx(shift*alpha_2, b_ix2)
                    q3_ix = MonomialToQIx(shift*alpha_3, b_ix3)
                    P_CCZ_mat[q1_ix, q2_ix, q3_ix] = 1
    return P_CCZ_mat