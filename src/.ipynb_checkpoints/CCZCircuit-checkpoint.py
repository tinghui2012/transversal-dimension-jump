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
from CircuitScheduling import ColorationCircuit, ColorProductCircuit
from ErrorPlugin import *
import random

from Decoders_SpaceTime import ST_BP_Decoder_Circuit_Class, ST_BPOSD_Decoder_Circuit_Class
from Simulators_SpaceTime import CodeSimulator_Circuit_SpaceTime
sys.setrecursionlimit(10000)

MEASUREMENT_INDICES = {}  # store absolute indices of measurements in the circuit, starting from 0
CURRENT_MEASUREMENT_INDEX = 0  # total number of measurements in the circuit so far


def pauli_strings(n):
    paulis = ['I', 'X', 'Y', 'Z']
    strings = [''.join(p) for p in itertools.product(paulis, repeat=n)]
    return [s for s in strings if s != 'I' * n]

def get_stab_meas_schedule(H):
    r, n = H.shape
    if r > n:
        sch_flipped = ColorationCircuit(H.T)
        sch = []
        for stage_flipped in sch_flipped:
            stage_dict_unflipped = {}
            for k, v in stage_flipped.items():
                stage_dict_unflipped[v] = k
            sch.append(stage_dict_unflipped)
    else:
        sch = ColorationCircuit(H)
    return sch

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
    
    # NO DETECTORS
    stab_meas_Z_circuit.append("TICK")
    
    return stab_meas_Z_circuit

# set up 3 blocks of 3D codes, initialized to plus states
def build_three_block_circuit_no_CCZ(code, circuit_error_params, p):
    # 1. prep logical |+> on 3 blocks of code
    # 2. perform 1 round of ONLY S_Z measurement on each block (no correction)
    # 3. measure X on all three blocks + set up independent DETECTORS
    # 4. final readout on 3D code from X measurements

    global CURRENT_MEASUREMENT_INDEX
    global MEASUREMENT_INDICES

    scheduling_Z = get_stab_meas_schedule(code.hz)
    # scheduling_X = get_stab_meas_schedule(code.hx)

    # set noise model
    error_params = {"p_i": circuit_error_params['p_i']*p, 
                    "p_state_p": circuit_error_params['p_state_p']*p, 
                    "p_m": circuit_error_params['p_m']*p, 
                    "p_CX":circuit_error_params['p_CX']*p, 
                    "p_idling_gate": circuit_error_params['p_idling_gate']*p
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

    # 3. measure X on each of the three blocks + set up independent DETECTORS
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
        # matching_lookback_indices_b1 = list(map(int, matching_lookback_indices_b1))
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b1[k]) for k in range(len(matching_lookback_indices_b1))], (0))
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b2[k]) for k in range(len(matching_lookback_indices_b2))], (0))
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b3[k]) for k in range(len(matching_lookback_indices_b3))], (0))

        measure_X_circuit.append("TICK")

    # 4. final X measurements + final readout on all 3 code blocks
    final_readout_circuit = stim.Circuit()
    for i in range(len(lx)):
        x_support = np.where(lx[i])[0]
        # block 1
        # block 2
        # block 3
        # actually, lookback_indices_after_X_3D should have indices for all 3 blocks
        x_targets_b1 = [
            stim.target_rec(lookback_indices_X_b1[0] + q)
            for q in x_support
        ]
        x_targets_b2 = [
            stim.target_rec(lookback_indices_X_b2[0] + q)
            for q in x_support
        ]
        x_targets_b3 = [
            stim.target_rec(lookback_indices_X_b3[0] + q)
            for q in x_support
        ]
        final_readout_circuit.append("OBSERVABLE_INCLUDE", x_targets_b1+x_targets_b2+x_targets_b3, i)

    full_circuit = init_3b_plus_circuit + \
                    stab_meas_Z_circuit + \
                    measure_X_circuit + \
                    final_readout_circuit
    full_circuit = AddCXError(full_circuit, 'DEPOLARIZE2(%f)' % error_params["p_CX"])
    return full_circuit

# set up 3 blocks of 3D codes, initialized to plus states, apply 2 sets of 3-qubit depolarizing noise (to emulate CCZ) 
def build_three_block_circuit_with_CCZ(code, circuit_error_params, p, CCZ_repeat, P_CCZ_mat):
    # 1. prep logical |+> on 3 blocks of code
    # 2. perform 1 round of ONLY S_Z measurement on each block (no correction)
    # 3. measure X on all three blocks + set up independent DETECTORS
    # 4. apply pauli depolarizing noise
    # 5. final X measurements + readout on 3 code blocks

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
                    "p_CCZ":circuit_error_params['p_CCZ']*p
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

    # 3. apply "CCZ" by 3-qubit paulis
    ccz_pauli_circuit = stim.Circuit()
    all_non_id_3_qubit_pauli_strings = pauli_strings(3)
    for _ in range(CCZ_repeat):
        for i1 in range(n):
            for i2 in range(n):
                for i3 in range(n):
                    if P_CCZ_mat[i1][i2][i3] == 1:
                        rand_idx = random.randint(0, len(all_non_id_3_qubit_pauli_strings)-1)
                        rand_pauli_str = all_non_id_3_qubit_pauli_strings[rand_idx]
                        # print(f"APPLYING {rand_pauli_str} to {i1}, {i2}, {i3}")
                        gate_b1, gate_b2, gate_b3 = rand_pauli_str[0], rand_pauli_str[1], rand_pauli_str[2]
                        if gate_b1 != 'I':
                            ccz_pauli_circuit.append(f"{gate_b1}_ERROR", data_indices_b1[i1], (error_params['p_CCZ']))
                        if gate_b2 != 'I':
                            ccz_pauli_circuit.append(f"{gate_b2}_ERROR", data_indices_b2[i2], (error_params['p_CCZ']))
                        if gate_b3 != 'I':
                            ccz_pauli_circuit.append(f"{gate_b3}_ERROR", data_indices_b3[i3], (error_params['p_CCZ']))
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
        # matching_lookback_indices_b1 = list(map(int, matching_lookback_indices_b1))
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b1[k]) for k in range(len(matching_lookback_indices_b1))], (0))
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b2[k]) for k in range(len(matching_lookback_indices_b2))], (0))
        measure_X_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_b3[k]) for k in range(len(matching_lookback_indices_b3))], (0))

        measure_X_circuit.append("TICK")

    # 5. final readout on all 3 code blocks
    final_readout_circuit = stim.Circuit()
    for i in range(len(lx)):
        x_support = np.where(lx[i])[0]
        # block 1
        # block 2
        # block 3
        # actually, lookback_indices_after_X_3D should have indices for all 3 blocks
        x_targets_b1 = [
            stim.target_rec(lookback_indices_X_b1[0] + q)
            for q in x_support
        ]
        x_targets_b2 = [
            stim.target_rec(lookback_indices_X_b2[0] + q)
            for q in x_support
        ]
        x_targets_b3 = [
            stim.target_rec(lookback_indices_X_b3[0] + q)
            for q in x_support
        ]
        final_readout_circuit.append("OBSERVABLE_INCLUDE", x_targets_b1+x_targets_b2+x_targets_b3, i)

    full_circuit = init_3b_plus_circuit + \
                    stab_meas_Z_circuit + \
                    ccz_pauli_circuit + \
                    measure_X_circuit + \
                    final_readout_circuit
    full_circuit = AddCXError(full_circuit, 'DEPOLARIZE2(%f)' % error_params["p_CX"])
    return full_circuit