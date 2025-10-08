import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
from bposd.hgp import hgp
import multiprocessing as mp
from ldpc.codes import ring_code
import stim
import re

import sys
sys.path.append("./src/")
from CircuitScheduling import ColorationCircuit, ColorProductCircuit
from ErrorPlugin import *

from Decoders_SpaceTime import ST_BP_Decoder_Circuit_Class, ST_BPOSD_Decoder_Circuit_Class
from Simulators_SpaceTime import CodeSimulator_Circuit_SpaceTime
sys.setrecursionlimit(10000)

MEASUREMENT_INDICES = {}  # store absolute indices of measurements in the circuit, starting from 0
CURRENT_MEASUREMENT_INDEX = 0  # total number of measurements in the circuit so far

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

def d_rounds_QEC_XZ_merged(code, 
                           d, 
                           code_dim, 
                           data_indices, 
                           Z_ancilla_indices, 
                           X_ancilla_indices, 
                           merged_scheduling, 
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

    # measure the Z ancillas
    circuit_stab_meas_rep1.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
    circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
    circuit_stab_meas_rep1.append("TICK")
    
    # Apply CX gates for the X and Z stabilizers
    for time_step in range(len(merged_scheduling)):
        # add idling errors for all the qubits during the ancilla shuffling
        idling_qubits = data_indices + X_ancilla_indices + Z_ancilla_indices
        idling_data_indices = list(copy.deepcopy(data_indices))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate'])) 
        for meas_type, anc_idx in merged_scheduling[time_step]:
            data_index = data_indices[merged_scheduling[time_step][(meas_type, anc_idx)]]
            if meas_type == 'X':
                X_ancilla_index = X_ancilla_indices[anc_idx]
                circuit_stab_meas_rep1.append("CX", [X_ancilla_index, data_index])
            else:
                Z_ancilla_index = Z_ancilla_indices[anc_idx]
                circuit_stab_meas_rep1.append("CX", [data_index, Z_ancilla_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        if len(idling_data_indices) != 0:
            print("unchecked data qubits", idling_data_indices)
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep1.append("TICK")
    
    # Measure X ancillas
    circuit_stab_meas_rep1.append("H", X_ancilla_indices)
    circuit_stab_meas_rep1.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
    # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
    circuit_stab_meas_rep1.append("MR", X_ancilla_indices)
    CURRENT_MEASUREMENT_INDEX += len(X_ancilla_indices)  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES[f'round{0}_X_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(X_ancilla_indices), CURRENT_MEASUREMENT_INDEX)
    
    # Measure Z ancillas
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
        # Apply CX gates for the X and Z stabilizers
        for time_step in range(len(merged_scheduling)):
            idling_qubits = data_indices + X_ancilla_indices + Z_ancilla_indices
            circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
            idling_data_indices = list(copy.deepcopy(data_indices))
            for meas_type, anc_idx in merged_scheduling[time_step]:
                data_index = data_indices[merged_scheduling[time_step][(meas_type, anc_idx)]]
                if meas_type == 'X':
                    X_ancilla_index = X_ancilla_indices[anc_idx]
                    circuit_stab_meas_rep2.append("CX", [X_ancilla_index, data_index])
                else:
                    Z_ancilla_index = Z_ancilla_indices[anc_idx]
                    circuit_stab_meas_rep2.append("CX", [data_index, Z_ancilla_index])
                if data_index in idling_data_indices:
                    idling_data_indices.pop(idling_data_indices.index(data_index))
            if len(idling_data_indices) != 0:
                print("unchecked data qubits", idling_data_indices)
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



# with merged scheduling for X and Z stabilizers
def build_logical_teleportation_circuit_2D_to_3D(code_2D, code_3D, circuit_error_params, p, gamma_2, gamma_1, logical_psi = "+", merged_scheduling_2D=None, merged_scheduling_3D=None):
    if merged_scheduling_2D is None and merged_scheduling_3D is None:
        return build_logical_teleportation_circuit_2D_to_3D_OLD(code_2D, code_3D, circuit_error_params, p, gamma_2, gamma_1, logical_psi)
    
    # gamma_2: homomorphis defining mapping of Z stabilizer checks
    # 1. prep logical |psi> on 2D code
    # 2. perform d rounds of QEC on 2D code
    # 3. prep logical |+> on 3D code
    # 4. perform 1 round of measurement on 3D code (no correction)
    # 5. perform logical CNOT from 2D to 3D code (by homomorphism)
    # 6. measure Z on 2D code + set up DETECTORS
    # 7. measure X on 3D code + set up DETECTORS
    # 8. final readout on 3D code from X measurements
    global CURRENT_MEASUREMENT_INDEX
    global MEASUREMENT_INDICES

    # set noise model
    error_params = {"p_i": circuit_error_params['p_i']*p, 
                    "p_state_p": circuit_error_params['p_state_p']*p, 
                    "p_m": circuit_error_params['p_m']*p, 
                    "p_CX":circuit_error_params['p_CX']*p, 
                    "p_idling_gate": circuit_error_params['p_idling_gate']*p
                    }

    hx_2D = code_2D.hx
    hz_2D = code_2D.hz
    hx_3D = code_3D.hx
    hz_3D = code_3D.hz
    lx_2D = code_2D.lx
    lz_2D = code_2D.lz
    lx_3D = code_3D.lx
    lz_3D = code_3D.lz

    n_2D = code_2D.N  # number of physical qubits in 2D code
    n_3D = code_3D.N  # number of physical qubits in 3D code
    num_Z_checks_2D = np.shape(hz_2D)[0]  # number of Z checks in 2D code
    num_X_checks_2D = np.shape(hx_2D)[0]
    num_Z_checks_3D = np.shape(hz_3D)[0]  # number of Z checks in 3D code
    num_X_checks_3D = np.shape(hx_3D)[0]

    data_indices_3D = list(np.arange(0, n_3D))
    Z_ancilla_indices_3D = list(np.arange(data_indices_3D[-1]+1, data_indices_3D[-1]+num_Z_checks_3D+1))
    X_ancilla_indices_3D = list(np.arange(Z_ancilla_indices_3D[-1]+1, Z_ancilla_indices_3D[-1]+num_X_checks_3D+1))
    data_indices_2D = list(np.arange(X_ancilla_indices_3D[-1]+1, X_ancilla_indices_3D[-1]+n_2D+1))
    Z_ancilla_indices_2D = list(np.arange(data_indices_2D[-1]+1, data_indices_2D[-1]+num_Z_checks_2D+1))
    X_ancilla_indices_2D = list(np.arange(Z_ancilla_indices_2D[-1]+1, Z_ancilla_indices_2D[-1]+num_X_checks_2D+1))

    # 1. prep logical |psi> on 2D code
    init_2D_circuit = stim.Circuit()
    if logical_psi == "0":
        # prepare logical |0> on 2D code
        init_2D_circuit.append("R", data_indices_2D)
    elif logical_psi == "+":
        # prepare logical |+> on 2D code
        init_2D_circuit.append("RX", data_indices_2D)
    elif logical_psi == "1":
        # prepare logical |1> on 2D code
        init_2D_circuit.append("R", data_indices_2D)
        init_2D_circuit.append("X", data_indices_2D)
    elif logical_psi == "-":
        # prepare logical |-> on 2D code
        init_2D_circuit.append("RX", data_indices_2D)
        init_2D_circuit.append("Z", data_indices_2D)
    else:
        raise ValueError("Logical state not supported: {}".format(logical_psi))
    
    init_2D_circuit.append("R", Z_ancilla_indices_2D + X_ancilla_indices_2D)
    # print("Initial 2D circuit:")
    # print(init_2D_circuit)

    # 2. perform d rounds of QEC on 2D code
    if merged_scheduling_2D is None:
        print("getting scheduling for 2D HX", hx_2D.shape)
        scheduling_2D_X = get_stab_meas_schedule(hx_2D)
        print("getting scheduling for 2D HZ", hz_2D.shape)
        scheduling_2D_Z = get_stab_meas_schedule(hz_2D)
        stab_meas_2D = d_rounds_QEC_XZ(code_2D,
                                    code_2D.D,
                                    2,  # code dimension (2D vs 3D)
                                    data_indices_2D, 
                                    Z_ancilla_indices_2D, 
                                    X_ancilla_indices_2D, 
                                    scheduling_2D_X, 
                                    scheduling_2D_Z,
                                    error_params)
    else:
        stab_meas_2D = d_rounds_QEC_XZ_merged(code_2D,
                                    code_2D.D,
                                    2,  # code dimension (2D vs 3D)
                                    data_indices_2D, 
                                    Z_ancilla_indices_2D, 
                                    X_ancilla_indices_2D, 
                                    merged_scheduling_2D,
                                    error_params)
    # print("Stabilizer measurement circuit for 2D code:")
    # print(stab_meas_2D)

    # 3. prep logical |+> on 3D code
    init_3D_circuit = stim.Circuit()
    init_3D_circuit.append("RX", data_indices_3D)
    init_3D_circuit.append("R", Z_ancilla_indices_3D + X_ancilla_indices_3D)
    init_3D_circuit.append("TICK")
    # print("Initial 3D circuit:")
    # print(init_3D_circuit)

    # 4. perform 1 round of measurement on 3D code (no correction)
    stab_meas_3D = d_rounds_QEC_XZ_merged(code_3D,
                                1,  # only one round of measurement
                                3,  # code dimension (2D vs 3D)
                                data_indices_3D, 
                                Z_ancilla_indices_3D, 
                                X_ancilla_indices_3D, 
                                merged_scheduling_3D, 
                                error_params)
    # print("Stabilizer measurement circuit for 3D code:")
    # print(stab_meas_3D)
    
    # 5. perform logical CNOT from 3D to 2D code (by homomorphism)
    # perform a physical CX from 3D data qubit i to 2D data qubit j if gamma_1[i, j] == 1
    logical_CX_circuit = stim.Circuit()
    CX_indices = []
    for i in range(n_3D):
        for j in range(n_2D):
            if gamma_1[i, j] == 1:
                CX_indices.extend([data_indices_3D[i], data_indices_2D[j]])
    logical_CX_circuit.append("CX", CX_indices)
    logical_CX_circuit.append("TICK")
    # print("Logical CX circuit:")
    # print(logical_CX_circuit)

    # 6. measure Z on data qubits of 2D code
    measure_Z_2D_circuit = stim.Circuit()
    # measure_X_2D_circuit.append("H", data_indices_2D)
    measure_Z_2D_circuit.append("MR", data_indices_2D)
    measure_Z_2D_circuit.append("TICK")
    CURRENT_MEASUREMENT_INDEX += len(data_indices_2D)  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES['final_measure_Z_2D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(data_indices_2D), CURRENT_MEASUREMENT_INDEX)
    
    # set up 3-way detectors for Z checks before/after the logical CX
    lookback_indices_before_Z_3D = MEASUREMENT_INDICES['round0_Z_ancilla_3D'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_before_Z_2D = MEASUREMENT_INDICES[f'round{int(code_2D.D)-1}_Z_ancilla_2D'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_after_Z_2D = MEASUREMENT_INDICES['final_measure_Z_2D'] - CURRENT_MEASUREMENT_INDEX
    for i in range(len(Z_ancilla_indices_2D)):
        # subset of data indices supported for this Z check
        supported_data_indices = list(np.where(hz_2D[Z_ancilla_indices_2D[i] - Z_ancilla_indices_2D[0],:] == 1)[0])
        # print("supported data indices for X check {}: {}".format(i, supported_data_indices))
        matching_lookback_indices_after_Z_2D = []
        for j in range(len(supported_data_indices)):
            # find the lookback index for each supported data index
            matching_lookback_index = lookback_indices_after_Z_2D[0] + supported_data_indices[j]
            # print("X matching_lookback_index after X check {}: {}".format(i, matching_lookback_index))
            matching_lookback_indices_after_Z_2D.append(matching_lookback_index)
        # print("X matching_lookback_indices_after_X_2D: ", matching_lookback_indices_after_X_2D)
        matching_lookback_indices_before_Z_2D = lookback_indices_before_Z_2D[i]
        matching_lookback_indices_before_Z_3D = 0
        for j in range(len(gamma_2)):
            if gamma_2[j, i] == 1:
                # print("FOUND Z check {} in 2D is mapped to Z check {} in 3D".format(i, j))
                matching_lookback_indices_before_Z_3D = lookback_indices_before_Z_3D[j]
                break
        # print("X matching_lookback_indices_before_X_2D index: ", matching_lookback_indices_before_X_2D)
        # print("X matching_lookback_indices_before_X_3D index: ", matching_lookback_indices_before_X_3D)
        # print(MEASUREMENT_INDICES)
        # print("current measurement index: ", CURRENT_MEASUREMENT_INDEX)
        if matching_lookback_indices_before_Z_3D == 0:
            print("DID NOT FIND A MATCH FOR Z CHECK!!!")
        matching_lookback_indices_after_Z_2D.append(matching_lookback_indices_before_Z_2D)
        matching_lookback_indices_after_Z_2D.append(matching_lookback_indices_before_Z_3D)
        matching_lookback_indices_after_Z_2D = list(map(int, matching_lookback_indices_after_Z_2D))
        # print("all lookback indices for detector {}: {}".format(i, matching_lookback_indices_after_X_2D))
        measure_Z_2D_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_after_Z_2D[k]) for k in range(len(matching_lookback_indices_after_Z_2D))], (0))
        measure_Z_2D_circuit.append("TICK")
    
    # measure_X_2D_circuit.append("DETECTOR", [stim.target_rec(combined_lookback_indices[i]) for i in range(len(combined_lookback_indices))], (0))
    # measure_X_2D_circuit.append("TICK")
    # print("Measurement circuit for X on 2D code:")
    # print(measure_X_2D_circuit)
    
    # 7. measure X on data qubits of 3D code
    measure_X_3D_circuit = stim.Circuit()
    measure_X_3D_circuit.append("MRX", data_indices_3D)
    CURRENT_MEASUREMENT_INDEX += len(data_indices_3D)  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES['final_measure_X_3D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(data_indices_3D), CURRENT_MEASUREMENT_INDEX)

    # set up 3-way detectors for X checks before/after the logical CX
    lookback_indices_before_X_2D = MEASUREMENT_INDICES[f'round{int(code_2D.D)-1}_X_ancilla_2D'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_before_X_3D = MEASUREMENT_INDICES['round0_X_ancilla_3D'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_after_X_3D = MEASUREMENT_INDICES['final_measure_X_3D'] - CURRENT_MEASUREMENT_INDEX

    for i in range(len(X_ancilla_indices_3D)):
        # subset of data indices supported for this X check
        supported_data_indices = list(np.where(hx_3D[X_ancilla_indices_3D[i] - X_ancilla_indices_3D[0],:] == 1)[0])
        # print("supported data indices for X check {}: {}".format(i, supported_data_indices))
        matching_lookback_indices_after_X_3D = []
        for j in range(len(supported_data_indices)):
            # find the lookback index for each supported data index
            matching_lookback_index = lookback_indices_after_X_3D[0] + supported_data_indices[j]
            # print("Z matching_lookback_index after Z check {}: {}".format(i, matching_lookback_index))
            matching_lookback_indices_after_X_3D.append(matching_lookback_index)
        # print("Z matching_lookback_indices_after_Z_3D: ", matching_lookback_indices_after_Z_3D)
        matching_lookback_indices_before_X_3D = lookback_indices_before_X_3D[i]
        # print("Z matching_lookback_indices_before_Z_3D index: ", matching_lookback_indices_before_Z_3D)
        matching_lookback_indices_after_X_3D.append(matching_lookback_indices_before_X_3D)
        matching_lookback_indices_after_X_3D = list(map(int, matching_lookback_indices_after_X_3D))
        measure_X_3D_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_after_X_3D[k]) for k in range(len(matching_lookback_indices_after_X_3D))], (0))
        measure_X_3D_circuit.append("TICK")
    # 7. perform CX corrections on 3D code based on 2D measurements
    final_readout_circuit = stim.Circuit()
    # final_readout_circuit.append("DEPOLARIZE1", data_indices_3D, error_params['p_m'])
    # final_readout_circuit.append("M", data_indices_3D)
    
    for i in range(len(lx_3D)):
        x_support = np.where(lx_3D[i])[0]
        # print("support indices for Z-stabilizer {}: {}".format(i, z_support))

    #     # rec indices from most recent MRX (2D) — happens before M (3D)
    #     # x_targets = [
    #     #     stim.target_rec(-len(data_indices_3D) - len(data_indices_2D) + q)
    #     #     for q in x_support
    #     # ]
    #     # rec indices from M (3D)
        x_targets = [
            stim.target_rec(lookback_indices_after_X_3D[0] + q)
            for q in x_support
        ]
        # print("targets (lookback indices) for Z-stabilizer {}: {}".format(i, z_targets))
        final_readout_circuit.append("OBSERVABLE_INCLUDE", x_targets, i)

    # print("Final readout circuit:")
    # print(final_readout_circuit)
    full_circuit = init_2D_circuit + \
                    stab_meas_2D + \
                    init_3D_circuit + \
                    stab_meas_3D + \
                    logical_CX_circuit + \
                    measure_Z_2D_circuit + \
                    measure_X_3D_circuit + \
                    final_readout_circuit
    full_circuit = AddCXError(full_circuit, 'DEPOLARIZE2(%f)' % error_params["p_CX"])
    return full_circuit

# CNOT from 3D to 2D, separated scheduling for X and Z stabilizer measurements
def build_logical_teleportation_circuit_2D_to_3D_OLD(code_2D, code_3D, circuit_error_params, p, gamma_2, gamma_1, logical_psi = "+"):
    # gamma_2: homomorphis defining mapping of Z stabilizer checks
    # 1. prep logical |psi> on 2D code
    # 2. perform d rounds of QEC on 2D code
    # 3. prep logical |+> on 3D code
    # 4. perform 1 round of measurement on 3D code (no correction)
    # 5. perform logical CNOT from 2D to 3D code (by homomorphism)
    # 6. measure Z on 2D code + set up DETECTORS
    # 7. measure X on 3D code + set up DETECTORS
    # 8. final readout on 3D code from X measurements
    global CURRENT_MEASUREMENT_INDEX
    global MEASUREMENT_INDICES

    scheduling_2D_X = get_stab_meas_schedule(code_2D.hx)
    scheduling_2D_Z = get_stab_meas_schedule(code_2D.hz)
    scheduling_3D_X = get_stab_meas_schedule(code_3D.hx)
    scheduling_3D_Z = get_stab_meas_schedule(code_3D.hz)
    # print("got all scheduling for 2D and 3D OLD coloration circuit")

    # set noise model
    error_params = {"p_i": circuit_error_params['p_i']*p, 
                    "p_state_p": circuit_error_params['p_state_p']*p, 
                    "p_m": circuit_error_params['p_m']*p, 
                    "p_CX":circuit_error_params['p_CX']*p, 
                    "p_idling_gate": circuit_error_params['p_idling_gate']*p
                    }
    
    hx_2D = code_2D.hx
    hz_2D = code_2D.hz
    hx_3D = code_3D.hx
    hz_3D = code_3D.hz
    lx_2D = code_2D.lx
    lz_2D = code_2D.lz
    lx_3D = code_3D.lx
    lz_3D = code_3D.lz

    n_2D = code_2D.N  # number of physical qubits in 2D code
    n_3D = code_3D.N  # number of physical qubits in 3D code
    num_Z_checks_2D = np.shape(hz_2D)[0]  # number of Z checks in 2D code
    num_X_checks_2D = np.shape(hx_2D)[0]
    num_Z_checks_3D = np.shape(hz_3D)[0]  # number of Z checks in 3D code
    num_X_checks_3D = np.shape(hx_3D)[0]

    data_indices_3D = list(np.arange(0, n_3D))
    Z_ancilla_indices_3D = list(np.arange(data_indices_3D[-1]+1, data_indices_3D[-1]+num_Z_checks_3D+1))
    X_ancilla_indices_3D = list(np.arange(Z_ancilla_indices_3D[-1]+1, Z_ancilla_indices_3D[-1]+num_X_checks_3D+1))
    data_indices_2D = list(np.arange(X_ancilla_indices_3D[-1]+1, X_ancilla_indices_3D[-1]+n_2D+1))
    Z_ancilla_indices_2D = list(np.arange(data_indices_2D[-1]+1, data_indices_2D[-1]+num_Z_checks_2D+1))
    X_ancilla_indices_2D = list(np.arange(Z_ancilla_indices_2D[-1]+1, Z_ancilla_indices_2D[-1]+num_X_checks_2D+1))

    # 1. prep logical |psi> on 2D code
    init_2D_circuit = stim.Circuit()
    if logical_psi == "0":
        # prepare logical |0> on 2D code
        init_2D_circuit.append("R", data_indices_2D)
    elif logical_psi == "+":
        # prepare logical |+> on 2D code
        init_2D_circuit.append("RX", data_indices_2D)
    elif logical_psi == "1":
        # prepare logical |1> on 2D code
        init_2D_circuit.append("R", data_indices_2D)
        init_2D_circuit.append("X", data_indices_2D)
    elif logical_psi == "-":
        # prepare logical |-> on 2D code
        init_2D_circuit.append("RX", data_indices_2D)
        init_2D_circuit.append("Z", data_indices_2D)
    else:
        raise ValueError("Logical state not supported: {}".format(logical_psi))
    
    init_2D_circuit.append("R", Z_ancilla_indices_2D + X_ancilla_indices_2D)
    # print("Initial 2D circuit:")
    # print(init_2D_circuit)

    # 2. perform d rounds of QEC on 2D code
    stab_meas_2D = d_rounds_QEC_XZ(code_2D,
                                code_2D.D,
                                2,  # code dimension (2D vs 3D)
                                data_indices_2D, 
                                Z_ancilla_indices_2D, 
                                X_ancilla_indices_2D, 
                                scheduling_2D_X, 
                                scheduling_2D_Z, 
                                error_params)
    # print("Stabilizer measurement circuit for 2D code:")
    # print(stab_meas_2D)

    # 3. prep logical |+> on 3D code
    init_3D_circuit = stim.Circuit()
    init_3D_circuit.append("RX", data_indices_3D)
    init_3D_circuit.append("R", Z_ancilla_indices_3D + X_ancilla_indices_3D)
    init_3D_circuit.append("TICK")
    # print("Initial 3D circuit:")
    # print(init_3D_circuit)

    # 4. perform 1 round of measurement on 3D code (no correction)
    stab_meas_3D = d_rounds_QEC_XZ(code_3D,
                                1,  # only one round of measurement
                                3,  # code dimension (2D vs 3D)
                                data_indices_3D, 
                                Z_ancilla_indices_3D, 
                                X_ancilla_indices_3D, 
                                scheduling_3D_X, 
                                scheduling_3D_Z, 
                                error_params)
    # print("Stabilizer measurement circuit for 3D code:")
    # print(stab_meas_3D)
    
    # 5. perform logical CNOT from 3D to 2D code (by homomorphism)
    # perform a physical CX from 3D data qubit i to 2D data qubit j if gamma_1[i, j] == 1
    logical_CX_circuit = stim.Circuit()
    CX_indices = []
    for i in range(n_3D):
        for j in range(n_2D):
            if gamma_1[i, j] == 1:
                CX_indices.extend([data_indices_3D[i], data_indices_2D[j]])
    logical_CX_circuit.append("CX", CX_indices)
    logical_CX_circuit.append("TICK")
    # print("Logical CX circuit:")
    # print(logical_CX_circuit)

    # 6. measure Z on data qubits of 2D code
    measure_Z_2D_circuit = stim.Circuit()
    # measure_X_2D_circuit.append("H", data_indices_2D)
    measure_Z_2D_circuit.append("MR", data_indices_2D)
    measure_Z_2D_circuit.append("TICK")
    CURRENT_MEASUREMENT_INDEX += len(data_indices_2D)  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES['final_measure_Z_2D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(data_indices_2D), CURRENT_MEASUREMENT_INDEX)
    
    # set up 3-way detectors for Z checks before/after the logical CX
    lookback_indices_before_Z_3D = MEASUREMENT_INDICES['round0_Z_ancilla_3D'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_before_Z_2D = MEASUREMENT_INDICES[f'round{int(code_2D.D)-1}_Z_ancilla_2D'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_after_Z_2D = MEASUREMENT_INDICES['final_measure_Z_2D'] - CURRENT_MEASUREMENT_INDEX
    for i in range(len(Z_ancilla_indices_2D)):
        # subset of data indices supported for this Z check
        supported_data_indices = list(np.where(hz_2D[Z_ancilla_indices_2D[i] - Z_ancilla_indices_2D[0],:] == 1)[0])
        # print("supported data indices for X check {}: {}".format(i, supported_data_indices))
        matching_lookback_indices_after_Z_2D = []
        for j in range(len(supported_data_indices)):
            # find the lookback index for each supported data index
            matching_lookback_index = lookback_indices_after_Z_2D[0] + supported_data_indices[j]
            # print("X matching_lookback_index after X check {}: {}".format(i, matching_lookback_index))
            matching_lookback_indices_after_Z_2D.append(matching_lookback_index)
        # print("X matching_lookback_indices_after_X_2D: ", matching_lookback_indices_after_X_2D)
        matching_lookback_indices_before_Z_2D = lookback_indices_before_Z_2D[i]
        matching_lookback_indices_before_Z_3D = 0
        for j in range(len(gamma_2)):
            if gamma_2[j, i] == 1:
                # print("FOUND Z check {} in 2D is mapped to Z check {} in 3D".format(i, j))
                matching_lookback_indices_before_Z_3D = lookback_indices_before_Z_3D[j]
                break
        # print("X matching_lookback_indices_before_X_2D index: ", matching_lookback_indices_before_X_2D)
        # print("X matching_lookback_indices_before_X_3D index: ", matching_lookback_indices_before_X_3D)
        # print(MEASUREMENT_INDICES)
        # print("current measurement index: ", CURRENT_MEASUREMENT_INDEX)
        if matching_lookback_indices_before_Z_3D == 0:
            print("DID NOT FIND A MATCH FOR Z CHECK!!!")
        matching_lookback_indices_after_Z_2D.append(matching_lookback_indices_before_Z_2D)
        matching_lookback_indices_after_Z_2D.append(matching_lookback_indices_before_Z_3D)
        matching_lookback_indices_after_Z_2D = list(map(int, matching_lookback_indices_after_Z_2D))
        # print("all lookback indices for detector {}: {}".format(i, matching_lookback_indices_after_X_2D))
        measure_Z_2D_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_after_Z_2D[k]) for k in range(len(matching_lookback_indices_after_Z_2D))], (0))
        measure_Z_2D_circuit.append("TICK")
    
    # measure_X_2D_circuit.append("DETECTOR", [stim.target_rec(combined_lookback_indices[i]) for i in range(len(combined_lookback_indices))], (0))
    # measure_X_2D_circuit.append("TICK")
    # print("Measurement circuit for X on 2D code:")
    # print(measure_X_2D_circuit)
    
    # 7. measure X on data qubits of 3D code
    measure_X_3D_circuit = stim.Circuit()
    measure_X_3D_circuit.append("MRX", data_indices_3D)
    CURRENT_MEASUREMENT_INDEX += len(data_indices_3D)  # update the total number of measurements in the circuit so far
    MEASUREMENT_INDICES['final_measure_X_3D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(data_indices_3D), CURRENT_MEASUREMENT_INDEX)

    # set up 3-way detectors for X checks before/after the logical CX
    lookback_indices_before_X_2D = MEASUREMENT_INDICES[f'round{int(code_2D.D)-1}_X_ancilla_2D'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_before_X_3D = MEASUREMENT_INDICES['round0_X_ancilla_3D'] - CURRENT_MEASUREMENT_INDEX
    lookback_indices_after_X_3D = MEASUREMENT_INDICES['final_measure_X_3D'] - CURRENT_MEASUREMENT_INDEX

    for i in range(len(X_ancilla_indices_3D)):
        # subset of data indices supported for this X check
        supported_data_indices = list(np.where(hx_3D[X_ancilla_indices_3D[i] - X_ancilla_indices_3D[0],:] == 1)[0])
        # print("supported data indices for X check {}: {}".format(i, supported_data_indices))
        matching_lookback_indices_after_X_3D = []
        for j in range(len(supported_data_indices)):
            # find the lookback index for each supported data index
            matching_lookback_index = lookback_indices_after_X_3D[0] + supported_data_indices[j]
            # print("Z matching_lookback_index after Z check {}: {}".format(i, matching_lookback_index))
            matching_lookback_indices_after_X_3D.append(matching_lookback_index)
        # print("Z matching_lookback_indices_after_Z_3D: ", matching_lookback_indices_after_Z_3D)
        matching_lookback_indices_before_X_3D = lookback_indices_before_X_3D[i]
        # print("Z matching_lookback_indices_before_Z_3D index: ", matching_lookback_indices_before_Z_3D)
        matching_lookback_indices_after_X_3D.append(matching_lookback_indices_before_X_3D)
        matching_lookback_indices_after_X_3D = list(map(int, matching_lookback_indices_after_X_3D))
        measure_X_3D_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_after_X_3D[k]) for k in range(len(matching_lookback_indices_after_X_3D))], (0))
        measure_X_3D_circuit.append("TICK")
    # 7. perform CX corrections on 3D code based on 2D measurements
    final_readout_circuit = stim.Circuit()
    # final_readout_circuit.append("DEPOLARIZE1", data_indices_3D, error_params['p_m'])
    # final_readout_circuit.append("M", data_indices_3D)
    
    for i in range(len(lx_3D)):
        x_support = np.where(lx_3D[i])[0]
        # print("support indices for Z-stabilizer {}: {}".format(i, z_support))

    #     # rec indices from most recent MRX (2D) — happens before M (3D)
    #     # x_targets = [
    #     #     stim.target_rec(-len(data_indices_3D) - len(data_indices_2D) + q)
    #     #     for q in x_support
    #     # ]
    #     # rec indices from M (3D)
        x_targets = [
            stim.target_rec(lookback_indices_after_X_3D[0] + q)
            for q in x_support
        ]
        # print("targets (lookback indices) for Z-stabilizer {}: {}".format(i, z_targets))
        final_readout_circuit.append("OBSERVABLE_INCLUDE", x_targets, i)

    # print("Final readout circuit:")
    # print(final_readout_circuit)
    full_circuit = init_2D_circuit + \
                    stab_meas_2D + \
                    init_3D_circuit + \
                    stab_meas_3D + \
                    logical_CX_circuit + \
                    measure_Z_2D_circuit + \
                    measure_X_3D_circuit + \
                    final_readout_circuit
    full_circuit = AddCXError(full_circuit, 'DEPOLARIZE2(%f)' % error_params["p_CX"])
    return full_circuit

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

    # Measure the ancillas
    # circuit_stab_meas_rep1.append("H", X_ancilla_indices)
    # circuit_stab_meas_rep1.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
    # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
    # circuit_stab_meas_rep1.append("MR", Z_ancilla_indices + X_ancilla_indices)
    
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

        # # repeat the circuit d-1 times
        # print("Number of rounds: ", d)
        # print("repeated times: ", d - 1)
        # print("d type: ", type(d))
        # print("Circuit rep1:")
        # print(type(circuit_stab_meas_rep1))
        # print("Circuit rep2:")
        # print(type(circuit_stab_meas_rep2))
        # circuit_stab_meas_rep = circuit_stab_meas_rep1 + (d - 1) * circuit_stab_meas_rep2
        
    # # final transversal readout
    # circuit_final_meas = stim.Circuit()
    # # circuit_final_meas_f.append("DEPOLARIZE1", data_indices, (1*pz)) # for debug
    # circuit_final_meas.append("DEPOLARIZE1",  data_indices, (error_params['p_m'])) # Add the measurement error
    # circuit_final_meas.append("MX", data_indices)
    # circuit_final_meas.append("SHIFT_COORDS", [], (1))
    # # Obtain the syndromes
    # for i in range(len(X_ancilla_indices)):
    #     supported_data_indices = list(np.where(hx[X_ancilla_indices[i] - X_ancilla_indices[0],:] == 1)[0])
    #     rec_indices = []
    #     for data_index in supported_data_indices:
    #         rec_indices.append(- len(data_indices) + data_index)
    #     rec_indices.append(- len(X_ancilla_indices) + i - len(data_indices))
    #     circuit_final_meas.append("DETECTOR", [stim.target_rec(rec_index) for rec_index in rec_indices], (0))
    # # Obtain the logical measurements result
    # for i in range(len(lx)):
    #     logical_X_qubit_indices = list(np.where(lx[i,:] == 1)[0])
    #     circuit_final_meas.append("OBSERVABLE_INCLUDE", 
    #                        [stim.target_rec(- len(data_indices) + data_index) for data_index in logical_X_qubit_indices],
    #                        (i))

    # OLD CNOT from 2D to 3D. construct a logical teleportation circuit from code A to code B
# def build_logical_teleportation_circuit_2D_to_3D_OLD(code_2D, code_3D, circuit_error_params, p, gamma_0, gamma_1, logical_psi = "0"):
#     # gamma_0: homomorphis defining mapping of X stabilizer checks
#     # 1. prep logical |psi> on 2D code
#     # 2. perform d rounds of QEC on 2D code
#     # 3. prep logical |0> on 3D code
#     # 4. perform 1 round of measurement on 3D code (no correction)
#     # 5. perform logical CNOT from 2D to 3D code (by homomorphism)
#     # 6. measure X on 2D code + set up DETECTORS
#     # 7. measure Z on 3D code + set up DETECTORS
#     # 8. final readout on 3D code from Z measurements
#     global CURRENT_MEASUREMENT_INDEX
#     global MEASUREMENT_INDICES

#     scheduling_2D_X = ColorationCircuit(code_2D.hx)
#     scheduling_2D_Z = ColorationCircuit(code_2D.hz)
#     scheduling_3D_X = ColorationCircuit(code_3D.hx)
#     scheduling_3D_Z = ColorationCircuit(code_3D.hz)

#     # set noise model
#     error_params = {"p_i": circuit_error_params['p_i']*p, 
#                     "p_state_p": circuit_error_params['p_state_p']*p, 
#                     "p_m": circuit_error_params['p_m']*p, 
#                     "p_CX":circuit_error_params['p_CX']*p, 
#                     "p_idling_gate": circuit_error_params['p_idling_gate']*p
#                     }
    
#     hx_2D = code_2D.hx
#     hz_2D = code_2D.hz
#     hx_3D = code_3D.hx
#     hz_3D = code_3D.hz
#     lx_2D = code_2D.lx
#     lz_2D = code_2D.lz
#     lx_3D = code_3D.lx
#     lz_3D = code_3D.lz

#     n_2D = code_2D.N  # number of physical qubits in 2D code
#     n_3D = code_3D.N  # number of physical qubits in 3D code
#     num_Z_checks_2D = np.shape(hz_2D)[0]  # number of Z checks in 2D code
#     num_X_checks_2D = np.shape(hx_2D)[0]
#     num_Z_checks_3D = np.shape(hz_3D)[0]  # number of Z checks in 3D code
#     num_X_checks_3D = np.shape(hx_3D)[0]

#     data_indices_3D = list(np.arange(0, n_3D))
#     Z_ancilla_indices_3D = list(np.arange(data_indices_3D[-1]+1, data_indices_3D[-1]+num_Z_checks_3D+1))
#     X_ancilla_indices_3D = list(np.arange(Z_ancilla_indices_3D[-1]+1, Z_ancilla_indices_3D[-1]+num_X_checks_3D+1))
#     data_indices_2D = list(np.arange(X_ancilla_indices_3D[-1]+1, X_ancilla_indices_3D[-1]+n_2D+1))
#     Z_ancilla_indices_2D = list(np.arange(data_indices_2D[-1]+1, data_indices_2D[-1]+num_Z_checks_2D+1))
#     X_ancilla_indices_2D = list(np.arange(Z_ancilla_indices_2D[-1]+1, Z_ancilla_indices_2D[-1]+num_X_checks_2D+1))
#     # print("Data indices 2D: ", data_indices_2D)
#     # print("Z ancilla indices 2D: ", Z_ancilla_indices_2D)
#     # print("X ancilla indices 2D: ", X_ancilla_indices_2D)
#     # print("Data indices 3D: ", data_indices_3D)
#     # print("Z ancilla indices 3D: ", Z_ancilla_indices_3D)
#     # print("X ancilla indices 3D: ", X_ancilla_indices_3D)

#     # 1. prep logical |psi> on 2D code
    # init_2D_circuit = stim.Circuit()
    # if logical_psi == "0":
    #     # prepare logical |0> on 2D code
    #     init_2D_circuit.append("R", data_indices_2D)
    # elif logical_psi == "+":
    #     # prepare logical |+> on 2D code
    #     init_2D_circuit.append("RX", data_indices_2D)
    # elif logical_psi == "1":
    #     # prepare logical |1> on 2D code
    #     init_2D_circuit.append("R", data_indices_2D)
    #     init_2D_circuit.append("X", data_indices_2D)
    # elif logical_psi == "-":
    #     # prepare logical |-> on 2D code
    #     init_2D_circuit.append("RX", data_indices_2D)
    #     init_2D_circuit.append("Z", data_indices_2D)
    # else:
    #     raise ValueError("Logical state not supported: {}".format(logical_psi))
    
    # init_2D_circuit.append("R", Z_ancilla_indices_2D + X_ancilla_indices_2D)
    # # print("Initial 2D circuit:")
    # # print(init_2D_circuit)

    # # 2. perform d rounds of QEC on 2D code
    # stab_meas_2D = d_rounds_QEC_ZX(code_2D,
    #                             code_2D.D,
    #                             2,  # code dimension (2D vs 3D)
    #                             data_indices_2D, 
    #                             Z_ancilla_indices_2D, 
    #                             X_ancilla_indices_2D, 
    #                             scheduling_2D_X, 
    #                             scheduling_2D_Z, 
    #                             error_params)
    # # print("Stabilizer measurement circuit for 2D code:")
    # # print(stab_meas_2D)

    # # 3. prep logical |0> on 3D code
    # init_3D_circuit = stim.Circuit()
    # init_3D_circuit.append("R", data_indices_3D)
    # init_3D_circuit.append("R", Z_ancilla_indices_3D + X_ancilla_indices_3D)
    # init_3D_circuit.append("TICK")
    # # print("Initial 3D circuit:")
    # # print(init_3D_circuit)

    # # 4. perform 1 round of measurement on 3D code (no correction)
    # stab_meas_3D = d_rounds_QEC_ZX(code_3D,
    #                             1,  # only one round of measurement
    #                             3,  # code dimension (2D vs 3D)
    #                             data_indices_3D, 
    #                             Z_ancilla_indices_3D, 
    #                             X_ancilla_indices_3D, 
    #                             scheduling_3D_X, 
    #                             scheduling_3D_Z, 
    #                             error_params)
    # # print("Stabilizer measurement circuit for 3D code:")
    # # print(stab_meas_3D)
    
    # # 5. perform logical CNOT from 2D to 3D code (by homomorphism)
    # # perform a physical CX from 2D data qubit i to 3D data qubit j if gamma_1[i, j] == 1
    # logical_CX_circuit = stim.Circuit()
    # CX_indices = []
    # for i in range(n_2D):
    #     for j in range(n_3D):
    #         if gamma_1[i, j] == 1:
    #             CX_indices.extend([data_indices_2D[i], data_indices_3D[j]])
    # logical_CX_circuit.append("CX", CX_indices)
    # logical_CX_circuit.append("TICK")
    # # print("Logical CX circuit:")
    # # print(logical_CX_circuit)

    # # 6. measure X on data qubits of 2D code
    # measure_X_2D_circuit = stim.Circuit()
    # # measure_X_2D_circuit.append("H", data_indices_2D)
    # measure_X_2D_circuit.append("MRX", data_indices_2D)
    # measure_X_2D_circuit.append("TICK")
    # CURRENT_MEASUREMENT_INDEX += len(data_indices_2D)  # update the total number of measurements in the circuit so far
    # MEASUREMENT_INDICES['final_measure_X_2D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(data_indices_2D), CURRENT_MEASUREMENT_INDEX)
    
    # # set up 3-way detectors for X checks before/after the logical CX
    # lookback_indices_before_X_3D = MEASUREMENT_INDICES['round0_X_ancilla_3D'] - CURRENT_MEASUREMENT_INDEX
    # lookback_indices_before_X_2D = MEASUREMENT_INDICES[f'round{int(code_2D.D)-1}_X_ancilla_2D'] - CURRENT_MEASUREMENT_INDEX
    # lookback_indices_after_X_2D = MEASUREMENT_INDICES['final_measure_X_2D'] - CURRENT_MEASUREMENT_INDEX

    # for i in range(len(X_ancilla_indices_2D)):
    #     # subset of data indices supported for this Z check
    #     supported_data_indices = list(np.where(hx_2D[X_ancilla_indices_2D[i] - X_ancilla_indices_2D[0],:] == 1)[0])
    #     # print("supported data indices for X check {}: {}".format(i, supported_data_indices))
    #     matching_lookback_indices_after_X_2D = []
    #     for j in range(len(supported_data_indices)):
    #         # find the lookback index for each supported data index
    #         matching_lookback_index = lookback_indices_after_X_2D[0] + supported_data_indices[j]
    #         # print("X matching_lookback_index after X check {}: {}".format(i, matching_lookback_index))
    #         matching_lookback_indices_after_X_2D.append(matching_lookback_index)
    #     # print("X matching_lookback_indices_after_X_2D: ", matching_lookback_indices_after_X_2D)
    #     matching_lookback_indices_before_X_2D = lookback_indices_before_X_2D[i]
    #     matching_lookback_indices_before_X_3D = 0
    #     for j in range(len(gamma_0[0])):
    #         if gamma_0[i, j] == 1:
    #             # print("FOUND X check {} in 2D is mapped to X check {} in 3D".format(i, j))
    #             matching_lookback_indices_before_X_3D = lookback_indices_before_X_3D[j]
    #             break
    #     # print("X matching_lookback_indices_before_X_2D index: ", matching_lookback_indices_before_X_2D)
    #     # print("X matching_lookback_indices_before_X_3D index: ", matching_lookback_indices_before_X_3D)
    #     # print(MEASUREMENT_INDICES)
    #     # print("current measurement index: ", CURRENT_MEASUREMENT_INDEX)
    #     matching_lookback_indices_after_X_2D.append(matching_lookback_indices_before_X_2D)
    #     matching_lookback_indices_after_X_2D.append(matching_lookback_indices_before_X_3D)
    #     matching_lookback_indices_after_X_2D = list(map(int, matching_lookback_indices_after_X_2D))
    #     # print("all lookback indices for detector {}: {}".format(i, matching_lookback_indices_after_X_2D))
    #     measure_X_2D_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_after_X_2D[i]) for i in range(len(matching_lookback_indices_after_X_2D))], (0))
    #     measure_X_2D_circuit.append("TICK")
    
    # # measure_X_2D_circuit.append("DETECTOR", [stim.target_rec(combined_lookback_indices[i]) for i in range(len(combined_lookback_indices))], (0))
    # # measure_X_2D_circuit.append("TICK")
    # # print("Measurement circuit for X on 2D code:")
    # # print(measure_X_2D_circuit)
    
    # # 7. measure Z on data qubits of 3D code
    # measure_Z_3D_circuit = stim.Circuit()
    # measure_Z_3D_circuit.append("MRZ", data_indices_3D)
    # CURRENT_MEASUREMENT_INDEX += len(data_indices_3D)  # update the total number of measurements in the circuit so far
    # MEASUREMENT_INDICES['final_measure_Z_3D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(data_indices_3D), CURRENT_MEASUREMENT_INDEX)

    # # set up 3-way detectors for Z checks before/after the logical CX
    # lookback_indices_before_Z_2D = MEASUREMENT_INDICES[f'round{int(code_2D.D)-1}_Z_ancilla_2D'] - CURRENT_MEASUREMENT_INDEX
    # lookback_indices_before_Z_3D = MEASUREMENT_INDICES['round0_Z_ancilla_3D'] - CURRENT_MEASUREMENT_INDEX
    # lookback_indices_after_Z_3D = MEASUREMENT_INDICES['final_measure_Z_3D'] - CURRENT_MEASUREMENT_INDEX

    # for i in range(len(Z_ancilla_indices_3D)):
    #     # subset of data indices supported for this Z check
    #     supported_data_indices = list(np.where(hz_3D[Z_ancilla_indices_3D[i] - Z_ancilla_indices_3D[0],:] == 1)[0])
    #     # print("supported data indices for Z check {}: {}".format(i, supported_data_indices))
    #     matching_lookback_indices_after_Z_3D = []
    #     for j in range(len(supported_data_indices)):
    #         # find the lookback index for each supported data index
    #         matching_lookback_index = lookback_indices_after_Z_3D[0] + supported_data_indices[j]
    #         # print("Z matching_lookback_index after Z check {}: {}".format(i, matching_lookback_index))
    #         matching_lookback_indices_after_Z_3D.append(matching_lookback_index)
    #     # print("Z matching_lookback_indices_after_Z_3D: ", matching_lookback_indices_after_Z_3D)
    #     matching_lookback_indices_before_Z_3D = lookback_indices_before_Z_3D[i]
    #     # print("Z matching_lookback_indices_before_Z_3D index: ", matching_lookback_indices_before_Z_3D)
    #     matching_lookback_indices_after_Z_3D.append(matching_lookback_indices_before_Z_3D)
    #     matching_lookback_indices_after_Z_3D = list(map(int, matching_lookback_indices_after_Z_3D))
    #     measure_Z_3D_circuit.append("DETECTOR", [stim.target_rec(matching_lookback_indices_after_Z_3D[i]) for i in range(len(matching_lookback_indices_after_Z_3D))], (0))
    #     measure_Z_3D_circuit.append("TICK")

    # # 7. perform CZ corrections on 3D code based on 2D measurements
    # final_readout_circuit = stim.Circuit()
    # # final_readout_circuit.append("DEPOLARIZE1", data_indices_3D, error_params['p_m'])
    # # final_readout_circuit.append("M", data_indices_3D)
    
    # for i in range(len(lz_3D)):
    #     z_support = np.where(lz_3D[i])[0]
    #     # print("support indices for Z-stabilizer {}: {}".format(i, z_support))

    # #     # rec indices from most recent MRX (2D) — happens before M (3D)
    # #     # x_targets = [
    # #     #     stim.target_rec(-len(data_indices_3D) - len(data_indices_2D) + q)
    # #     #     for q in x_support
    # #     # ]
    # #     # rec indices from M (3D)
    #     z_targets = [
    #         stim.target_rec(lookback_indices_after_Z_3D[0] + q)
    #         for q in z_support
    #     ]
    #     # print("targets (lookback indices) for Z-stabilizer {}: {}".format(i, z_targets))
    #     final_readout_circuit.append("OBSERVABLE_INCLUDE", z_targets, i)

    # # print("Final readout circuit:")
    # # print(final_readout_circuit)
    # full_circuit = init_2D_circuit + \
    #                 stab_meas_2D + \
    #                 init_3D_circuit + \
    #                 stab_meas_3D + \
    #                 logical_CX_circuit + \
    #                 measure_X_2D_circuit + \
    #                 measure_Z_3D_circuit + \
    #                 final_readout_circuit
    # full_circuit = AddCXError(full_circuit, 'DEPOLARIZE2(%f)' % error_params["p_CX"])
    # return full_circuit

    # def d_rounds_QEC_ZX(code,
        #             d,
        #             code_dim,
        #             data_indices,
        #             Z_ancilla_indices,
        #             X_ancilla_indices,
        #             scheduling_X, 
        #             scheduling_Z, 
        #             error_params):

        # global CURRENT_MEASUREMENT_INDEX
        # global MEASUREMENT_INDICES

        # hx = code.hx
        # # hz = code.hz
        # lx = code.lx
        # n = int(code.N)  # number of physical qubits
        # d = int(d)  # number of rounds of QEC
        # n_Z_ancilla = len(Z_ancilla_indices)

        # ## Repeated code cycles
        # circuit_stab_meas_rep1 = stim.Circuit()

        # # START with Z measurements
        # # measure the Z ancillas
        # circuit_stab_meas_rep1.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
        # # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
        # circuit_stab_meas_rep1.append("TICK")

        # # Appy CX gates for the Z stabilziers
        # for time_step in range(len(scheduling_Z)):
        #     idling_qubits = data_indices + Z_ancilla_indices
        #     idling_data_indices = list(copy.deepcopy(data_indices))
        #     circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        #     for j in scheduling_Z[time_step]:
        # #       supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
        #         Z_ancilla_index = Z_ancilla_indices[j]
        #         data_index = data_indices[scheduling_Z[time_step][j]]
        #         # data_index = supported_data_qubits[i]
        #         circuit_stab_meas_rep1.append("CX", [data_index, Z_ancilla_index])
        #         if data_index in idling_data_indices:
        #             idling_data_indices.pop(idling_data_indices.index(data_index))
        #     circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        #     circuit_stab_meas_rep1.append("TICK")

        # # Measure the ancillas
        # # circuit_stab_meas_rep1.append("H", X_ancilla_indices)
        # # circuit_stab_meas_rep1.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
        # # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        # # circuit_stab_meas_rep1.append("MR", Z_ancilla_indices + X_ancilla_indices)
        
        # # ONLY measure Z ancillas
        # circuit_stab_meas_rep1.append("MR", Z_ancilla_indices)
        # CURRENT_MEASUREMENT_INDEX += len(Z_ancilla_indices)  # update the total number of measurements in the circuit so far
        # MEASUREMENT_INDICES[f'round{0}_Z_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(Z_ancilla_indices), CURRENT_MEASUREMENT_INDEX)

        # for i in range(len(Z_ancilla_indices)):
        #     lookback_index = MEASUREMENT_INDICES[f'round{0}_Z_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
        #     circuit_stab_meas_rep1.append("DETECTOR", [stim.target_rec(lookback_index)], (0))
        # circuit_stab_meas_rep1.append("TICK")

        # # first round of X stabilizer measurements
        # # # Initialize the X ancillas to the + state
        # circuit_stab_meas_rep1.append("H", X_ancilla_indices)
        # circuit_stab_meas_rep1.append("DEPOLARIZE1", X_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
        # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
        # circuit_stab_meas_rep1.append("TICK")
        # # Apply CX gates for the X stabilizers
        # for time_step in range(len(scheduling_X)):
        #     # add idling errors for all the qubits during the ancilla shuffling
        #     idling_qubits = data_indices + X_ancilla_indices
        #     idling_data_indices = list(copy.deepcopy(data_indices))
        #     circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate'])) 
        #     for j in scheduling_X[time_step]:
        # #                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
        #         X_ancilla_index = X_ancilla_indices[j]
        #         data_index = data_indices[scheduling_X[time_step][j]]
        #         # data_index = supported_data_qubits[i]
        #         circuit_stab_meas_rep1.append("CX", [X_ancilla_index, data_index])
        #         if data_index in idling_data_indices:
        #             idling_data_indices.pop(idling_data_indices.index(data_index))
        #     circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        #     circuit_stab_meas_rep1.append("TICK")
        
        # # Measure X ancillas
        # circuit_stab_meas_rep1.append("H", X_ancilla_indices)
        # circuit_stab_meas_rep1.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
        # # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        # circuit_stab_meas_rep1.append("MR", X_ancilla_indices)
        # CURRENT_MEASUREMENT_INDEX += len(X_ancilla_indices)  # update the total number of measurements in the circuit so far
        # MEASUREMENT_INDICES[f'round{0}_X_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(X_ancilla_indices), CURRENT_MEASUREMENT_INDEX)
        
        # # no detectors for first round of X ancilla measurements b/c non deterministic

        # circuit_stab_meas_rep = circuit_stab_meas_rep1
        
        # for d_i in range(1, d):
        #     # repeat the circuit d-1 times
        #     # rep with difference detectors
        #     circuit_stab_meas_rep2 = stim.Circuit()
        #     # START with Z measurements

        #     ## initialize the Z ancillas
        #     circuit_stab_meas_rep2.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
        #     # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
        #     circuit_stab_meas_rep2.append("TICK")
        #     # Appy CX gates for the Z stabilizers
        #     for time_step in range(len(scheduling_Z)):
        #         idling_qubits = data_indices + Z_ancilla_indices
        #         circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        #         idling_data_indices = list(copy.deepcopy(data_indices))
        #         for j in scheduling_Z[time_step]:
        #             # supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
        #             Z_ancilla_index = Z_ancilla_indices[j]
        #             data_index = data_indices[scheduling_Z[time_step][j]]
        #             # data_index = supported_data_qubits[i]
        #             circuit_stab_meas_rep2.append("CX", [data_index, Z_ancilla_index])
        #             if data_index in idling_data_indices:
        #                 idling_data_indices.pop(idling_data_indices.index(data_index))
        #         circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        #         circuit_stab_meas_rep2.append("TICK")

        #     # Measure Z the ancillas
        #     # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        #     circuit_stab_meas_rep2.append("MR", Z_ancilla_indices)
        #     CURRENT_MEASUREMENT_INDEX += len(Z_ancilla_indices)  # update the total number of measurements in the circuit so far
        #     MEASUREMENT_INDICES[f'round{d_i}_Z_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(Z_ancilla_indices), CURRENT_MEASUREMENT_INDEX)

        #     # print(CURRENT_MEASUREMENT_INDEX)
        #     # print(MEASUREMENT_INDICES)
        #     # pair detectors with previous round of Z ancilla measurements
        #     for i in range(len(Z_ancilla_indices)):
        #         last_round_index = MEASUREMENT_INDICES[f'round{d_i-1}_Z_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
        #         current_round_index = MEASUREMENT_INDICES[f'round{d_i}_Z_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
        #         # print(f"lookback for {i}th Z ancilla in round {d_i-1}: ", last_round_index)
        #         # print(f"lookback for {i}th Z ancilla in round {d_i}: ", current_round_index)
        #         circuit_stab_meas_rep2.append("DETECTOR", [stim.target_rec(last_round_index),
        #                                                 stim.target_rec(current_round_index)], (0))
            
        #     circuit_stab_meas_rep2.append("TICK")

        #     # measurement the X ancillas
        #     # # Initialize the X ancillas to the + state
        #     circuit_stab_meas_rep2.append("H", X_ancilla_indices)
        #     circuit_stab_meas_rep2.append("DEPOLARIZE1", X_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
        #     # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
        #     circuit_stab_meas_rep2.append("TICK")
        #     # Apply CX gates for the X stabilizers
        #     for time_step in range(len(scheduling_X)):
        #         idling_qubits = data_indices + X_ancilla_indices
        #         circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        #         idling_data_indices = list(copy.deepcopy(data_indices))
        #         for j in scheduling_X[time_step]:
        #     #       supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
        #             X_ancilla_index = X_ancilla_indices[j]
        #             data_index = data_indices[scheduling_X[time_step][j]]
        #             # data_index = supported_data_qubits[i]
        #             circuit_stab_meas_rep2.append("CX", [X_ancilla_index, data_index])
        #             if data_index in idling_data_indices:
        #                 idling_data_indices.pop(idling_data_indices.index(data_index))
        #         circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        #         circuit_stab_meas_rep2.append("TICK")
            
        #     # Measure X ancillas
        #     circuit_stab_meas_rep2.append("H", X_ancilla_indices)
        #     circuit_stab_meas_rep2.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
        #     # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        #     circuit_stab_meas_rep2.append("MR", X_ancilla_indices)
        #     CURRENT_MEASUREMENT_INDEX += len(X_ancilla_indices)  # update the total number of measurements in the circuit so far
        #     MEASUREMENT_INDICES[f'round{d_i}_X_ancilla_{code_dim}D'] = np.arange(CURRENT_MEASUREMENT_INDEX - len(X_ancilla_indices), CURRENT_MEASUREMENT_INDEX)

        #     # pair detectors with previous round of Z ancilla measurements
        #     for i in range(len(X_ancilla_indices)):
        #         last_round_index = MEASUREMENT_INDICES[f'round{d_i-1}_X_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
        #         current_round_index = MEASUREMENT_INDICES[f'round{d_i}_X_ancilla_{code_dim}D'][i] - CURRENT_MEASUREMENT_INDEX
        #         circuit_stab_meas_rep2.append("DETECTOR", [stim.target_rec(last_round_index),
        #                                                 stim.target_rec(current_round_index)], (0))
            
        #     circuit_stab_meas_rep2.append("TICK")

        #     # add this round of Z and X checks to the full measurement circuit
        #     circuit_stab_meas_rep += circuit_stab_meas_rep2
        
        # return circuit_stab_meas_rep
