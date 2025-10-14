import numpy as np
import copy
import stim
import sys
sys.path.append("./src/")
from CircuitScheduling import get_stab_meas_schedule
from ErrorPlugin import *
sys.setrecursionlimit(10000)

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

    # STEP 0: Initialization layer
    circuit_init = stim.Circuit()
    circuit_init.append("RX", data_indices)
    circuit_init.append("R", X_ancilla_indices + Z_ancilla_indices)
    circuit_init.append("TICK")

    # STEP 1: Round 1 of SE with full stabilizer measurements
    circuit_stab_meas_rep1 = stim.Circuit()

    # 1.1 Initialize the X ancillas to the + state
    circuit_stab_meas_rep1.append("H", X_ancilla_indices)
    circuit_stab_meas_rep1.append("DEPOLARIZE1", X_ancilla_indices, (error_params['p_state_p'])) # State preparation error
    circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Idle data qubits during the preparation for X ancillas
    circuit_stab_meas_rep1.append("TICK")
    
    # 1.2 Apply CX gates for the X stabilizers
    for time_step in range(len(scheduling_X)):
        # add idling errors for all the qubits (data and X) during the ancilla shuffling
        idling_qubits = data_indices + X_ancilla_indices
        idling_data_indices = list(copy.deepcopy(data_indices))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate'])) 
        for j in scheduling_X[time_step]:
            X_ancilla_index = X_ancilla_indices[j]
            data_index = scheduling_X[time_step][j]
            circuit_stab_meas_rep1.append("CX", [X_ancilla_index, data_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        # if len(idling_data_indices) != 0:
        #     print(time_step, "unchecked data qubits", idling_data_indices)
        # else:
        #     print(time_step, "no unchecked data qubits")
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep1.append("TICK")

    # 1.3 Initialize the Z ancillas to the 0 state (do nothing, just add prep and idle errors)
    circuit_stab_meas_rep1.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # State preparation error
    circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Idle data qubits during the preparation for Z ancillas
    circuit_stab_meas_rep1.append("TICK")

    # 1.4 Apply CX gates for the Z stabilizers
    for time_step in range(len(scheduling_Z)):
        # add idling errors for all the qubits (data and Z) during the ancilla shuffling
        idling_qubits = data_indices + Z_ancilla_indices
        idling_data_indices = list(copy.deepcopy(data_indices))
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        for j in scheduling_Z[time_step]:
            Z_ancilla_index = Z_ancilla_indices[j]
            data_index = scheduling_Z[time_step][j]
            circuit_stab_meas_rep1.append("CX", [data_index, Z_ancilla_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        # if len(idling_data_indices) != 0:
        #     print(time_step, "unchecked data qubits", idling_data_indices)
        # else:
        #     print(time_step, "no unchecked data qubits")
        circuit_stab_meas_rep1.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep1.append("TICK")

    # 1.5 Measure both X and Z ancillas
    circuit_stab_meas_rep1.append("H", X_ancilla_indices) # Change X ancilla basis for measurement in Z basis
    circuit_stab_meas_rep1.append("DEPOLARIZE1",  X_ancilla_indices, (error_params['p_m'])) # Add the measurement error
    circuit_stab_meas_rep1.append("DEPOLARIZE1",  Z_ancilla_indices, (error_params['p_m'])) # Add the measurement error
    circuit_stab_meas_rep1.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
    circuit_stab_meas_rep1.append("MR", Z_ancilla_indices + X_ancilla_indices)

    # 1.6 Set up detectors on X ancilla measurements (deterministic since data qubits start in + state)
    circuit_stab_meas_rep1.append("SHIFT_COORDS", [], (1))
    for i in range(len(X_ancilla_indices)):
        circuit_stab_meas_rep1.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i)], (0))
    circuit_stab_meas_rep1.append("TICK")

    # STEP 2: Subsequent SE rounds with difference detectors
    circuit_stab_meas_rep2 = stim.Circuit()

    # 2.1 Initialize the X ancillas to the + state
    circuit_stab_meas_rep2.append("H", X_ancilla_indices)
    circuit_stab_meas_rep2.append("DEPOLARIZE1", X_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
    circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
    circuit_stab_meas_rep2.append("TICK")
    
    # 2.2 Apply CX gates for the X stabilizers
    for time_step in range(len(scheduling_X)):
        # add idling errors for all the qubits (data and X) during the ancilla shuffling
        idling_qubits = data_indices + X_ancilla_indices
        circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        idling_data_indices = list(copy.deepcopy(data_indices))
        for j in scheduling_X[time_step]:
            X_ancilla_index = X_ancilla_indices[j]
            data_index = scheduling_X[time_step][j]
            circuit_stab_meas_rep2.append("CX", [X_ancilla_index, data_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep2.append("TICK")

    # 2.3 Initialize the Z ancillas to the 0 state (do nothing, just add prep and idle errors)
    circuit_stab_meas_rep2.append("DEPOLARIZE1", Z_ancilla_indices, (error_params['p_state_p'])) # Add the state preparation error
    circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
    circuit_stab_meas_rep2.append("TICK")

    # 2.4 Appy CX gates for the Z stabilziers
    for time_step in range(len(scheduling_Z)):
        # add idling errors for all the qubits (data and Z) during the ancilla shuffling
        idling_qubits = data_indices + Z_ancilla_indices
        circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_qubits, (error_params['p_idling_gate']))
        idling_data_indices = list(copy.deepcopy(data_indices))
        for j in scheduling_Z[time_step]:
            Z_ancilla_index = Z_ancilla_indices[j]
            data_index = scheduling_Z[time_step][j]
            circuit_stab_meas_rep2.append("CX", [data_index, Z_ancilla_index])
            if data_index in idling_data_indices:
                idling_data_indices.pop(idling_data_indices.index(data_index))
        circuit_stab_meas_rep2.append("DEPOLARIZE1", idling_data_indices, (error_params['p_i'])) # idling errors for qubits that are not being checked
        circuit_stab_meas_rep2.append("TICK")

    # 2.5 Measure both X and Z ancillas
    circuit_stab_meas_rep2.append("H", X_ancilla_indices) # Change X ancilla basis for measurement in Z basis
    circuit_stab_meas_rep2.append("DEPOLARIZE1",  X_ancilla_indices, (error_params['p_m'])) # Add X measurement error
    circuit_stab_meas_rep2.append("DEPOLARIZE1",  Z_ancilla_indices, (error_params['p_m'])) # Add Z measurement error
    circuit_stab_meas_rep2.append("DEPOLARIZE1", data_indices, (error_params['p_i'])) # Idle data qubits during the measurement of X ancillas
    circuit_stab_meas_rep2.append("MR", Z_ancilla_indices + X_ancilla_indices)

    # 2.6 Set up difference detectors on X ancilla measurements
    for i in range(len(X_ancilla_indices)):
        circuit_stab_meas_rep2.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i), 
                                        stim.target_rec(- len(X_ancilla_indices) + i - len(Z_ancilla_indices) - len(X_ancilla_indices))], (0))
    circuit_stab_meas_rep2.append("TICK")

    # STEP 3: Repeat STEP 2 (num_rep - 1) times
    circuit_stab_meas_rep = circuit_stab_meas_rep1 + (num_rep - 1)*circuit_stab_meas_rep2

    # STEP 4: Final transversal readout
    circuit_final_meas = stim.Circuit()
    circuit_final_meas.append("DEPOLARIZE1",  data_indices, (error_params['p_m'])) # Measurement error on data qubits
    circuit_final_meas.append("MX", data_indices) # Measure data qubits in X basis
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

    # STEP 5: Combine all layers and add CX errors
    noisy_circuit = circuit_init + circuit_stab_meas_rep + circuit_final_meas
    noisy_circuit = AddCXError(noisy_circuit, 'DEPOLARIZE2(%f)' % error_params["p_CX"])
    
    return noisy_circuit