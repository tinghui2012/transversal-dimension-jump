import numpy as np
import matplotlib.pyplot as plt
import copy
from bposd.hgp import hgp
import multiprocessing as mp
from ldpc.codes import ring_code
import stim
import itertools

import sys
sys.path.append("./src/")
from CircuitScheduling import get_stab_meas_schedule
from ErrorPlugin import *
import random

# from Decoders_SpaceTime import ST_BP_Decoder_Circuit_Class, ST_BPOSD_Decoder_Circuit_Class
# from Simulators_SpaceTime import CodeSimulator_Circuit_SpaceTime
sys.setrecursionlimit(10000)

MEASUREMENT_INDICES = {}  # store absolute indices of measurements in the circuit, starting from 0
CURRENT_MEASUREMENT_INDEX = 0  # total number of measurements in the circuit so far

# single (half) round of one type of stab measurements
def single_Z_measurement_QEC(code, 
                           data_indices_b1,
                           data_indices_b2,
                           data_indices_b3,
                           Z_ancilla_indices_b1,
                           Z_ancilla_indices_b2,
                           Z_ancilla_indices_b3,
                           scheduling_Z,
                           error_params):
    global CURRENT_MEASUREMENT_INDEX
    global MEASUREMENT_INDICES
    hx = code.hx
    hz = code.hz
    lx = code.lx
    lz = code.lz
    n = int(code.N)
    stab_meas_Z_circuit = stim.Circuit()
    stab_meas_Z_circuit.append("DEPOLARIZE1", 
                               Z_ancilla_indices_b1+Z_ancilla_indices_b2+Z_ancilla_indices_b3,
                               (error_params['p_state_p'])) # Add the state preparation error
    stab_meas_Z_circuit.append("DEPOLARIZE1",
                               data_indices_b1+data_indices_b2+data_indices_b3,
                               (error_params['p_i']))
    stab_meas_Z_circuit.append("TICK")

    # Apply CX gates for Z stabilizers
    num_Z_time_steps = len(scheduling_Z)
    for time_step in range(num_Z_time_steps):
        # block 1
        idling_qubits_b1 = data_indices_b1 + Z_ancilla_indices_b1
        idling_data_indices_b1 = list(copy.deepcopy(data_indices_b1))
        # block 2
        idling_qubits_b2 = data_indices_b2 + Z_ancilla_indices_b2
        idling_data_indices_b2 = list(copy.deepcopy(data_indices_b2))
        # block 3
        idling_qubits_b3 = data_indices_b3 + Z_ancilla_indices_b3
        idling_data_indices_b3 = list(copy.deepcopy(data_indices_b3))
        stab_meas_Z_circuit.append("DEPOLARIZE1", 
                                   idling_qubits_b1+idling_qubits_b2+idling_qubits_b3,
                                   (error_params['p_idling_gate']))
        for j in scheduling_Z[time_step]:
            Z_ancilla_index_b1 = Z_ancilla_indices_b1[j]
            Z_ancilla_index_b2 = Z_ancilla_indices_b2[j]
            Z_ancilla_index_b3 = Z_ancilla_indices_b3[j]
            data_index_b1 = data_indices_b1[scheduling_Z[time_step][j]]
            data_index_b2 = data_indices_b2[scheduling_Z[time_step][j]]
            data_index_b3 = data_indices_b3[scheduling_Z[time_step][j]]
            stab_meas_Z_circuit.append("CX",
                                       [data_index_b1, Z_ancilla_index_b1])
            stab_meas_Z_circuit.append("CX",
                                       [data_index_b2, Z_ancilla_index_b2])
            stab_meas_Z_circuit.append("CX",
                                       [data_index_b3, Z_ancilla_index_b3])
            if data_index_b1 in idling_data_indices_b1:
                idling_data_indices_b1.pop(idling_data_indices_b1.index(data_index_b1))
            if data_index_b2 in idling_data_indices_b2:
                idling_data_indices_b2.pop(idling_data_indices_b2.index(data_index_b2))
            if data_index_b3 in idling_data_indices_b3:
                idling_data_indices_b3.pop(idling_data_indices_b3.index(data_index_b3))
        stab_meas_Z_circuit.append("DEPOLARIZE1",
                                   idling_data_indices_b1+idling_data_indices_b2+idling_data_indices_b3,
                                   (error_params['p_i']))  # idling errors for qubits that are not being checked
        stab_meas_Z_circuit.append("TICK")
    
    stab_meas_Z_circuit.append("DEPOLARIZE1", Z_ancilla_indices_b1+Z_ancilla_indices_b2+Z_ancilla_indices_b3, (error_params['p_state_p']))

    # only measure Z ancillas
    num_Z_ancillas = len(Z_ancilla_indices_b1)
    # block 1
    stab_meas_Z_circuit.append("MR", Z_ancilla_indices_b1)
    CURRENT_MEASUREMENT_INDEX += num_Z_ancillas  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES[f'init_Z_ancilla_b1'] = np.arange(CURRENT_MEASUREMENT_INDEX - num_Z_ancillas, CURRENT_MEASUREMENT_INDEX)

    # block 2
    stab_meas_Z_circuit.append("MR", Z_ancilla_indices_b2)
    CURRENT_MEASUREMENT_INDEX += num_Z_ancillas  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES[f'init_Z_ancilla_b1'] = np.arange(CURRENT_MEASUREMENT_INDEX - num_Z_ancillas, CURRENT_MEASUREMENT_INDEX)

    # block 3
    stab_meas_Z_circuit.append("MR", Z_ancilla_indices_b3)
    CURRENT_MEASUREMENT_INDEX += num_Z_ancillas  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES[f'init_Z_ancilla_b1'] = np.arange(CURRENT_MEASUREMENT_INDEX - num_Z_ancillas, CURRENT_MEASUREMENT_INDEX)

    stab_meas_Z_circuit.append("DEPOLARIZE1",
                               data_indices_b1+data_indices_b2+data_indices_b3,
                               (error_params['p_i'])) # idling errors on data qubits
    
    # NO DETECTORS
    stab_meas_Z_circuit.append("TICK")
    
    return stab_meas_Z_circuit

def d_rounds_QEC_XZ(code,
                 d,
                 code_dim,
                 data_indices,
                 Z_ancilla_indices,
                 X_ancilla_indices,
                 scheduling_X, 
                 scheduling_Z, 
                 error_params):

    global CURRENT_MEASUREMENT_INDEX
    global MEASUREMENT_INDICES

    hx = code.hx
    # hz = code.hz
    lx = code.lx
    n = int(code.N)  # number of physical qubits
    d = int(d)  # number of rounds of QEC
    n_Z_ancilla = len(Z_ancilla_indices)

    ## Repeated code cycles
    circuit_stab_meas_rep1 = stim.Circuit()

    # START with X measurements

    # first round of X stabilizer measurements
    # # Initialize the X ancillas to the + state
    circuit_stab_meas_rep1.append("H", X_ancilla_indices)
    circuit_stab_meas_rep1.append("DEPOLARIZE1", X_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
    circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
    circuit_stab_meas_rep1.append("TICK")
    # Apply CX gates for the X stabilizers
    for time_step in range(len(scheduling_X)):
        # add idling errors for all the qubits during the ancilla shuffling
        idling_qubits = data_indices + X_ancilla_indices
        idling_data_indices = list(copy.deepcopy(data_indices))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate'])) 
        for j in scheduling_X[time_step]:
    #                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
            X_ancilla_index = X_ancilla_indices[j]
            data_index = data_indices[scheduling_X[time_step][j]]
            # data_index = supported_data_qubits[i]
            circuit_stab_meas_rep1.append("CX", [X_ancilla_index, data_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep1.append("TICK")
    
    # Measure X ancillas
    circuit_stab_meas_rep1.append("H", X_ancilla_indices)
    circuit_stab_meas_rep1.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
    # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
    circuit_stab_meas_rep1.append("MR", X_ancilla_indices)
    CURRENT_MEASUREMENT_INDEX += len(X_ancilla_indices)  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES[f'round{0}_X_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(X_ancilla_indices), CURRENT_MEASUREMENT_INDEX)

    # measure the Z ancillas
    circuit_stab_meas_rep1.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
    # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
    circuit_stab_meas_rep1.append("TICK")

    # Appy CX gates for the Z stabilziers
    for time_step in range(len(scheduling_Z)):
        idling_qubits = data_indices + Z_ancilla_indices
        idling_data_indices = list(copy.deepcopy(data_indices))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        for j in scheduling_Z[time_step]: # for each ancilla
    #       supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
            Z_ancilla_index = Z_ancilla_indices[j]
            data_index = data_indices[scheduling_Z[time_step][j]]
            # data_index = supported_data_qubits[i]
            circuit_stab_meas_rep1.append("CX", [data_index, Z_ancilla_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep1.append("TICK")

    # ONLY measure Z ancillas
    circuit_stab_meas_rep1.append("MR", Z_ancilla_indices)
    CURRENT_MEASUREMENT_INDEX += len(Z_ancilla_indices)  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES[f'round{0}_Z_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(Z_ancilla_indices), CURRENT_MEASUREMENT_INDEX)

    for i in range(len(X_ancilla_indices)):
        lookback_index = MEASUREMENT_INDICES[f'round{0}_X_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
        circuit_stab_meas_rep1.append("DETECTOR", [stim.target_rec(lookback_index)], (0))
    circuit_stab_meas_rep1.append("TICK")
    
    # no detectors for first round of Z ancilla measurements b/c non deterministic

    circuit_stab_meas_rep = circuit_stab_meas_rep1
    
    for d_i in range(1, d):
        # repeat the circuit d-1 times
        # rep with difference detectors
        circuit_stab_meas_rep2 = stim.Circuit()
        # START with X measurements

        # measurement the X ancillas
        # # Initialize the X ancillas to the + state
        circuit_stab_meas_rep2.append("H", X_ancilla_indices)
        circuit_stab_meas_rep2.append("DEPOLARIZE1", X_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
        circuit_stab_meas_rep2.append("TICK")
        # Apply CX gates for the X stabilizers
        for time_step in range(len(scheduling_X)):
            idling_qubits = data_indices + X_ancilla_indices
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
            idling_data_indices = list(copy.deepcopy(data_indices))
            for j in scheduling_X[time_step]:
        #       supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
                X_ancilla_index = X_ancilla_indices[j]
                data_index = data_indices[scheduling_X[time_step][j]]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas_rep2.append("CX", [X_ancilla_index, data_index])
                if data_index in idling_data_indices:
                    idling_data_indices.pop(idling_data_indices.index(data_index))
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas_rep2.append("TICK")
        
        # Measure X ancillas
        circuit_stab_meas_rep2.append("H", X_ancilla_indices)
        circuit_stab_meas_rep2.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        circuit_stab_meas_rep2.append("MR", X_ancilla_indices)
        CURRENT_MEASUREMENT_INDEX += len(X_ancilla_indices)  # update the total number of measurements in the circuit so far
        MEASUREMENT_INDICES[f'round{d_i}_X_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(X_ancilla_indices), CURRENT_MEASUREMENT_INDEX)

        # pair detectors with previous round of X ancilla measurements
        for i in range(len(X_ancilla_indices)):
            last_round_index = MEASUREMENT_INDICES[f'round{d_i-1}_X_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
            current_round_index = MEASUREMENT_INDICES[f'round{d_i}_X_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
            circuit_stab_meas_rep2.append("DETECTOR", [stim.target_rec(last_round_index),
                                                       stim.target_rec(current_round_index)], (0))
        
        circuit_stab_meas_rep2.append("TICK")

        ## initialize the Z ancillas
        circuit_stab_meas_rep2.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
        circuit_stab_meas_rep2.append("TICK")
        # Appy CX gates for the Z stabilizers
        for time_step in range(len(scheduling_Z)):
            idling_qubits = data_indices + Z_ancilla_indices
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
            idling_data_indices = list(copy.deepcopy(data_indices))
            for j in scheduling_Z[time_step]:
                # supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
                Z_ancilla_index = Z_ancilla_indices[j]
                data_index = data_indices[scheduling_Z[time_step][j]]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas_rep2.append("CX", [data_index, Z_ancilla_index])
                if data_index in idling_data_indices:
                    idling_data_indices.pop(idling_data_indices.index(data_index))
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas_rep2.append("TICK")

        # Measure Z the ancillas
        # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        circuit_stab_meas_rep2.append("MR", Z_ancilla_indices)
        CURRENT_MEASUREMENT_INDEX += len(Z_ancilla_indices)  # update the total number of measurements in the circuit so far
        MEASUREMENT_INDICES[f'round{d_i}_Z_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(Z_ancilla_indices), CURRENT_MEASUREMENT_INDEX)

        # print(CURRENT_MEASUREMENT_INDEX)
        # print(MEASUREMENT_INDICES)
        # pair detectors with previous round of Z ancilla measurements
        for i in range(len(Z_ancilla_indices)):
            last_round_index = MEASUREMENT_INDICES[f'round{d_i-1}_Z_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
            current_round_index = MEASUREMENT_INDICES[f'round{d_i}_Z_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
            # print(f"lookback for {i}th Z ancilla in round {d_i-1}: ", last_round_index)
            # print(f"lookback for {i}th Z ancilla in round {d_i}: ", current_round_index)
            circuit_stab_meas_rep2.append("DETECTOR", [stim.target_rec(last_round_index),
                                                       stim.target_rec(current_round_index)], (0))
        
        circuit_stab_meas_rep2.append("TICK")

        # add this round of Z and X checks to the full measurement circuit
        circuit_stab_meas_rep += circuit_stab_meas_rep2
    
    return circuit_stab_meas_rep

def build_three_block_circuit_with_CCZ(code, circuit_error_params, p, CCZ_repeat, P_CCZ_mat):
    # 1. prep logical |+> on three 3D code blocks
    # 2. perform one round of Z measurement on each block
    # 3. apply CCZ noise
    # 4. measure X on all three blocks
    # 5. final X measurements + readout on all three code blocks

    global CURRENT_MEASUREMENT_INDEX
    global MEASUREMENT_INDICES

    scheduling_Z = get_stab_meas_schedule(code.hz)
    # scheduling_X = get_stab_meas_schedule(code.hx)

    # set noise model
    error_params = {"p_i": circuit_error_params['p_i']*p, 
                    "p_state_p": circuit_error_params['p_state_p']*p, 
                    "p_m": circuit_error_params['p_m']*p, 
                    "p_CX":circuit_error_params['p_CX']*p, 
                    "p_idling_gate": circuit_error_params['p_idling_gate']*p,
                    "p_CCZ":circuit_error_params['p_CCZ']
                    }
    hz = code.hz
    hx = code.hx
    lz = code.lz
    lx = code.lx

    n = code.N
    num_Z_checks = np.shape(hz)[0]
    num_X_checks = np.shape(hx)[0]

    data_indices_b1 = list(np.arange(0, n))
    Z_ancilla_indices_b1 = list(np.arange(data_indices_b1[-1]+1,
                                            data_indices_b1[-1]+num_Z_checks+1))
    X_ancilla_indices_b1 = list(np.arange(Z_ancilla_indices_b1[-1]+1,
                                          Z_ancilla_indices_b1[-1]+num_X_checks+1))
    data_indices_b2 = list(np.arange(X_ancilla_indices_b1[-1]+1, 
                                     X_ancilla_indices_b1[-1]+n+1))
    Z_ancilla_indices_b2 = list(np.arange(data_indices_b2[-1]+1,
                                            data_indices_b2[-1]+num_Z_checks+1))
    X_ancilla_indices_b2 = list(np.arange(Z_ancilla_indices_b2[-1]+1,
                                          Z_ancilla_indices_b2[-1]+num_X_checks+1))
    data_indices_b3 = list(np.arange(X_ancilla_indices_b2[-1]+1, 
                                     X_ancilla_indices_b2[-1]+n+1))
    Z_ancilla_indices_b3 = list(np.arange(data_indices_b3[-1]+1,
                                            data_indices_b3[-1]+num_Z_checks+1))
    X_ancilla_indices_b3 = list(np.arange(Z_ancilla_indices_b3[-1]+1,
                                          Z_ancilla_indices_b3[-1]+num_X_checks+1))
    data_indices_dict = {1: data_indices_b1,
                         2: data_indices_b2,
                         3: data_indices_b3}

    # 1. prep logical |+> on 3 blocks of code
    init_3b_plus_circuit = stim.Circuit()
    init_3b_plus_circuit.append("RX", data_indices_b1+data_indices_b2+data_indices_b3)
    init_3b_plus_circuit.append("R", Z_ancilla_indices_b1
                                +X_ancilla_indices_b1
                                +Z_ancilla_indices_b2
                                +X_ancilla_indices_b2
                                +Z_ancilla_indices_b3
                                +X_ancilla_indices_b3)
    
    # 2. perform 1 round of ONLY S_Z measurement on each block (no correction)
    stab_meas_Z_circuit = single_Z_measurement_QEC(code, 
                                   data_indices_b1,
                                   data_indices_b2,
                                   data_indices_b3,
                                   Z_ancilla_indices_b1,
                                   Z_ancilla_indices_b2,
                                   Z_ancilla_indices_b3,
                                   scheduling_Z,
                                   error_params)

    # 3. apply "CCZ" by weight-5-all-pauli-Z errors on each data qubit index for each code block. this assumes depth-2 CCZ
    ccz_pauli_circuit = stim.Circuit()

    # block 1:
    # for each i in block 1, loop through all CCZs, pick the two with (1, _, _) = 1
    for i in range(n):
        matching_j1j2 = []
        matching_k1k2 = []
        for j in range(n):
            for k in range(n):
                if P_CCZ_mat[i][j][k] == 1:
                    matching_j1j2.append(j)
                    matching_k1k2.append(k)
        if len(matching_j1j2) != 2 or len(matching_k1k2) != 2:
            print(i, "not depth-2 block 1", matching_j1j2, matching_k1k2)
        matching_i_circuit_index = [data_indices_b1[i]] # should be the same as itself since data_indices_b1 starts from 0
        matching_j1j2_circuit_indices = [data_indices_b2[j_prime] for j_prime in matching_j1j2]
        matching_k1k2_circuit_indices = [data_indices_b3[k_prime] for k_prime in matching_k1k2]
        weight_5_error_indices = matching_i_circuit_index + matching_j1j2_circuit_indices + matching_k1k2_circuit_indices
        ccz_pauli_circuit.append("CORRELATED_ERROR", [f"Z{ei}" for ei in weight_5_error_indices], (error_params['p_CCZ']))
    ccz_pauli_circuit.append("TICK")

    # block 2:
    # loop through all CCZs, pick the two with (_, 1, _) = 1
    for j in range(n):
        matching_i1i2 = []
        matching_k1k2 = []
        for i in range(n):
            for k in range(n):
                if P_CCZ_mat[i][j][k] == 1:
                    matching_i1i2.append(i)
                    matching_k1k2.append(k)
        if len(matching_i1i2) != 2 or len(matching_k1k2) != 2:
            print(j, "not depth-2 block 2", matching_i1i2, matching_k1k2)
        matching_j_circuit_index = [data_indices_b2[j]]
        matching_i1i2_circuit_indices = [data_indices_b1[i_prime] for i_prime in matching_i1i2]
        matching_k1k2_circuit_indices = [data_indices_b3[k_prime] for k_prime in matching_k1k2]
        weight_5_error_indices = matching_j_circuit_index + matching_i1i2_circuit_indices + matching_k1k2_circuit_indices
        ccz_pauli_circuit.append("CORRELATED_ERROR", [f"Z{ei}" for ei in weight_5_error_indices], (error_params['p_CCZ']))
    ccz_pauli_circuit.append("TICK")

    # block 3:
    # for each k in block 3, loop through all CCZs, pick the two with (_, _, 1) = 1
    for k in range(n):
        matching_i1i2 = []
        matching_j1j2 = []
        for i in range(n):
            for j in range(n):
                if P_CCZ_mat[i][j][k] == 1:
                    matching_i1i2.append(i)
                    matching_j1j2.append(j)
        if len(matching_i1i2) != 2 or len(matching_j1j2) != 2:
            print(k, "not depth-2 block 3", matching_i1i2, matching_j1j2)
        matching_k_circuit_index = [data_indices_b3[k]]
        matching_i1i2_circuit_indices = [data_indices_b1[i_prime] for i_prime in matching_i1i2]
        matching_j1j2_circuit_indices = [data_indices_b2[j_prime] for j_prime in matching_j1j2]
        weight_5_error_indices = matching_k_circuit_index + matching_i1i2_circuit_indices + matching_j1j2_circuit_indices
        ccz_pauli_circuit.append("CORRELATED_ERROR", [f"Z{ei}" for ei in weight_5_error_indices], (error_params['p_CCZ']))
    ccz_pauli_circuit.append("TICK")
    
    # 4. measure X on each of the three blocks + set up independent DETECTORS
    measure_X_circuit = stim.Circuit()
    num_data_measurements = len(data_indices_b1)
    measure_X_circuit.append("DEPOLARIZE1", data_indices_b1+data_indices_b2+data_indices_b3, (error_params['p_m'])) # add measurement error
    measure_X_circuit.append("MX", data_indices_b1)
    CURRENT_MEASUREMENT_INDEX += num_data_measurements
    MEASUREMENT_INDICES['final_measure_X_b1'] = np.arange(CURRENT_MEASUREMENT_INDEX - num_data_measurements,
                                                                        CURRENT_MEASUREMENT_INDEX)
    measure_X_circuit.append("MX", data_indices_b2)
    CURRENT_MEASUREMENT_INDEX += num_data_measurements
    MEASUREMENT_INDICES['final_measure_X_b2'] = np.arange(CURRENT_MEASUREMENT_INDEX - num_data_measurements,
                                                                        CURRENT_MEASUREMENT_INDEX)
    measure_X_circuit.append("MX", data_indices_b3)
    CURRENT_MEASUREMENT_INDEX += num_data_measurements
    MEASUREMENT_INDICES['final_measure_X_b3'] = np.arange(CURRENT_MEASUREMENT_INDEX - num_data_measurements,
                                                                        CURRENT_MEASUREMENT_INDEX)
    
    lookback_indices_X_b1 = MEASUREMENT_INDICES['final_measure_X_b1'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_X_b2 = MEASUREMENT_INDICES['final_measure_X_b2'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_X_b3 = MEASUREMENT_INDICES['final_measure_X_b3'] - CURRENT_MEASUREMENT_INDEX
    
    # obtain the syndromes
    for i in range(len(X_ancilla_indices_b1)):
        supported_data_indices_b1 = list(np.where(hx[X_ancilla_indices_b1[i]-X_ancilla_indices_b1[0],:]==1)[0])
        supported_data_indices_b2 = list(np.where(hx[X_ancilla_indices_b2[i]-X_ancilla_indices_b2[0],:]==1)[0])
        supported_data_indices_b3 = list(np.where(hx[X_ancilla_indices_b3[i]-X_ancilla_indices_b3[0],:]==1)[0])

        matching_lookback_indices_b1 = []
        for j in range(len(supported_data_indices_b1)):
            matching_lookback_index_b1 = lookback_indices_X_b1[0] + supported_data_indices_b1[j]
            matching_lookback_indices_b1.append(matching_lookback_index_b1)
        
        matching_lookback_indices_b2 = []
        for j in range(len(supported_data_indices_b2)):
            matching_lookback_index_b2 = lookback_indices_X_b2[0] + supported_data_indices_b2[j]
            matching_lookback_indices_b2.append(matching_lookback_index_b2)
        
        matching_lookback_indices_b3 = []
        for j in range(len(supported_data_indices_b3)):
            matching_lookback_index_b3 = lookback_indices_X_b3[0] + supported_data_indices_b3[j]
            matching_lookback_indices_b3.append(matching_lookback_index_b3)
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b1[k]) for k in range(len(matching_lookback_indices_b1))], (0))
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b2[k]) for k in range(len(matching_lookback_indices_b2))], (0))
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b3[k]) for k in range(len(matching_lookback_indices_b3))], (0))

        measure_X_circuit.append("TICK")

    # 5. final readout on all 3 code blocks
    final_readout_circuit = stim.Circuit()
    # block 1 observables:
    for i in range(len(lx)):
        x_support = np.where(lx[i])[0]
        x_targets_b1 = [
            stim.target_rec(lookback_indices_X_b1[0] + q)
            for q in x_support
        ]
        final_readout_circuit.append("OBSERVABLE_INCLUDE", x_targets_b1, i)
    # block 2 observables:
    for i in range(len(lx)):
        x_support = np.where(lx[i])[0]
        x_targets_b2 = [
            stim.target_rec(lookback_indices_X_b2[0] + q)
            for q in x_support
        ]
        final_readout_circuit.append("OBSERVABLE_INCLUDE", x_targets_b2, i+3)
    # block 3 observables:
    for i in range(len(lx)):
        x_support = np.where(lx[i])[0]
        x_targets_b3 = [
            stim.target_rec(lookback_indices_X_b3[0] + q)
            for q in x_support
        ]
        final_readout_circuit.append("OBSERVABLE_INCLUDE", x_targets_b3, i+6)

    full_circuit = init_3b_plus_circuit + \
                    stab_meas_Z_circuit + \
                    ccz_pauli_circuit + \
                    measure_X_circuit + \
                    final_readout_circuit
    full_circuit = AddCXError(full_circuit, 'DEPOLARIZE2(%f)' % error_params["p_CX"])
    return full_circuit