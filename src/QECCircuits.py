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


# QEC Circuit with one-stage decoding
def QECCircuit_OneStage(eval_code, num_rep, circuit_error_params, p):
    # Get the circuit scheduling
    scheduling_X = get_stab_meas_schedule(eval_code.hx)
    scheduling_Z = get_stab_meas_schedule(eval_code.hz)
    
    # Set the noise model
    error_params = {"p_i": circuit_error_params['p_i']*p, "p_state_p": circuit_error_params['p_state_p']*p, 
    "p_m": circuit_error_params['p_m']*p, "p_CX":circuit_error_params['p_CX']*p, "p_idling_gate": circuit_error_params['p_idling_gate']*p}
    
    hx = eval_code.hx
    hz = eval_code.hz
    lx = eval_code.lx
    data_indices = list(np.arange(0, np.shape(hx)[1]))
    n = len(data_indices)
    n_Z_ancilla, n_X_ancilla = np.shape(hz)[0], np.shape(hx)[0]
    Z_ancilla_indices = list(np.arange(n, n + n_Z_ancilla))
    X_ancilla_indices = list(np.arange(n + n_Z_ancilla, n + n_Z_ancilla + n_X_ancilla))

    z_stab_weight = int(np.sum(hz[0,:]))
    x_stab_weight = int(np.sum(hx[0,:]))

    ## Initialization layer
    circuit_init = stim.Circuit()
    circuit_init.append("RX", data_indices)
    circuit_init.append("R", X_ancilla_indices + Z_ancilla_indices)

    ## Repeated code cycles
    circuit_stab_meas_rep1 = stim.Circuit()
    # # Initialize the X ancillas to the + state
    circuit_stab_meas_rep1.append("H", X_ancilla_indices)
    circuit_stab_meas_rep1.append("DEPOLARIZE1", X_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
    # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
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
            data_index = scheduling_X[time_step][j]
            # data_index = supported_data_qubits[i]
            circuit_stab_meas_rep1.append("CX", [X_ancilla_index, data_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep1.append("TICK")

    # meausure the Z ancillas
    circuit_stab_meas_rep1.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
    # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
    circuit_stab_meas_rep1.append("TICK")

    # Appy CX gates for the Z stabilziers
    for time_step in range(len(scheduling_Z)):
        idling_qubits = data_indices + Z_ancilla_indices
        idling_data_indices = list(copy.deepcopy(data_indices))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        for j in scheduling_Z[time_step]:
    #                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
            Z_ancilla_index = Z_ancilla_indices[j]
            data_index = scheduling_Z[time_step][j]
            # data_index = supported_data_qubits[i]
            circuit_stab_meas_rep1.append("CX", [data_index, Z_ancilla_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep1.append("TICK")

    # Measure the ancillas
    circuit_stab_meas_rep1.append("H", X_ancilla_indices)
    circuit_stab_meas_rep1.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
    # circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
    circuit_stab_meas_rep1.append("MR", Z_ancilla_indices + X_ancilla_indices)

    circuit_stab_meas_rep1.append("SHIFT_COORDS", [], (1))
    for i in range(len(X_ancilla_indices)):
        circuit_stab_meas_rep1.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i)], (0))
    circuit_stab_meas_rep1.append("TICK")

    # rep with difference detectors
    circuit_stab_meas_rep2 = stim.Circuit()
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
    #                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
            X_ancilla_index = X_ancilla_indices[j]
            data_index = scheduling_X[time_step][j]
            # data_index = supported_data_qubits[i]
            circuit_stab_meas_rep2.append("CX", [X_ancilla_index, data_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep2.append("TICK")

    # meausure the Z ancillas
    ## initialize the Z ancillas
    circuit_stab_meas_rep2.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
    # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
    circuit_stab_meas_rep2.append("TICK")
    # Appy CX gates for the Z stabilziers
    for time_step in range(len(scheduling_Z)):
        idling_qubits = data_indices + Z_ancilla_indices
        circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        idling_data_indices = list(copy.deepcopy(data_indices))
        for j in scheduling_Z[time_step]:
    #                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
            Z_ancilla_index = Z_ancilla_indices[j]
            data_index = scheduling_Z[time_step][j]
            # data_index = supported_data_qubits[i]
            circuit_stab_meas_rep2.append("CX", [data_index, Z_ancilla_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep2.append("TICK")

    # Measure the ancillas
    circuit_stab_meas_rep2.append("H", X_ancilla_indices)
    circuit_stab_meas_rep2.append("DEPOLARIZE1",  X_ancilla_indices, (3/2*error_params['p_m'])) # Add the measurement error
    # circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
    circuit_stab_meas_rep2.append("MR", Z_ancilla_indices + X_ancilla_indices)

    for i in range(len(X_ancilla_indices)):
        circuit_stab_meas_rep2.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i), 
                                        stim.target_rec(- len(X_ancilla_indices) + i - len(Z_ancilla_indices) - len(X_ancilla_indices))], (0))
    circuit_stab_meas_rep2.append("TICK")


    circuit_stab_meas_rep = circuit_stab_meas_rep1 + (num_rep - 1)*circuit_stab_meas_rep2


    # final transversal readout
    circuit_final_meas = stim.Circuit()
    #         circuit_final_meas_f.append("DEPOLARIZE1", data_indices, (1*pz)) # for debug
    circuit_final_meas.append("DEPOLARIZE1",  data_indices, (error_params['p_m'])) # Add the measurement error
    circuit_final_meas.append("MX", data_indices)
    circuit_final_meas.append("SHIFT_COORDS", [], (1))
    # Obtain the syndromes
    for i in range(len(X_ancilla_indices)):
        supported_data_indices = list(np.where(hx[X_ancilla_indices[i] - n - n_Z_ancilla,:] == 1)[0])
        rec_indices = []
        for data_index in supported_data_indices:
            rec_indices.append(- len(data_indices) + data_index)
        rec_indices.append(- len(X_ancilla_indices) + i - len(data_indices))
        circuit_final_meas.append("Detector", [stim.target_rec(rec_index) for rec_index in rec_indices], (0))
    # Obtain the logical measurements result
    for i in range(len(lx)):
        logical_X_qubit_indices = list(np.where(lx[i,:] == 1)[0])
        circuit_final_meas.append("OBSERVABLE_INCLUDE", 
                           [stim.target_rec(- len(data_indices) + data_index) for data_index in logical_X_qubit_indices],
                           (i))

    noisy_circuit = circuit_init + circuit_stab_meas_rep + circuit_final_meas
    noisy_circuit = AddCXError(noisy_circuit, 'DEPOLARIZE2(%f)' % error_params["p_CX"])
    
    return noisy_circuit